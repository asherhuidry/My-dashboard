"""Temporal Heterogeneous Graph Attention Network (T-HGAT).

This is the core of FinBrain's intelligence engine. Rather than treating each
asset in isolation, it models the ENTIRE financial universe as a graph:

  Nodes: Assets (265), MacroIndicators (65), Sectors (11), Events (dynamic)
  Edges: SupplyChain, Correlation, MacroImpact, SectorMembership, Causality

Information propagates: TSMC's chip shortage → NVDA's production risk →
GPU infrastructure cost → cloud capex → MSFT/GOOGL/META margins.

The graph attention mechanism learns WHICH relationships matter most for
predicting each asset — and discovers patterns no human analyst would hard-code.

Architecture:
  1. Node feature init: each node gets its modality embedding (from encoders.py)
  2. Edge type embeddings: different relationship types get different projections
  3. Graph Attention (GAT) layer: nodes aggregate from neighbors, weighted by
     learned attention scores that depend on BOTH node states AND edge type
  4. 2-3 hops: after k layers, each node has seen k-hop neighborhood info
  5. Output head: per-node MLP for return prediction

Implementation: pure PyTorch — no torch_geometric dependency.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Edge type registry ─────────────────────────────────────────────────────────

EDGE_TYPES = {
    "supplies_to":      0,   # Company → Company (supply chain)
    "correlates_with":  1,   # Asset ↔ Asset/MacroIndicator (statistical)
    "impacts":          2,   # MacroIndicator → Sector ETF (causal)
    "belongs_to":       3,   # Asset → Sector (classification)
    "competes_with":    4,   # Company ↔ Company (same sector)
    "leads":            5,   # SeriesA → SeriesB (Granger/lag relationship)
    "unknown":          6,
}
N_EDGE_TYPES = len(EDGE_TYPES)


# ── Multi-head Graph Attention Layer ──────────────────────────────────────────

class HeteroGATLayer(nn.Module):
    """One layer of heterogeneous graph attention.

    For each node v:
      1. Compute attention weights alpha_{vu} for all neighbors u
         α_{vu} = softmax_u( LeakyReLU( a^T [W_q h_v || W_k h_u || W_e e_{vu}] ) )
      2. Aggregate: h_v' = ELU( sum_u α_{vu} * W_v h_u )
      3. Multi-head: run H independent heads, concat + project

    The edge type embedding W_e e_{vu} is the key heterogeneous element:
    a supply chain edge is treated differently from a correlation edge.

    Args:
        in_dim:      Input node feature dimension.
        out_dim:     Output node feature dimension per head.
        n_heads:     Number of attention heads.
        n_edge_types: Number of distinct edge types.
        dropout:     Dropout on attention weights.
        residual:    Whether to add residual connection.
    """

    def __init__(
        self,
        in_dim:       int,
        out_dim:      int,
        n_heads:      int   = 4,
        n_edge_types: int   = N_EDGE_TYPES,
        dropout:      float = 0.1,
        residual:     bool  = True,
    ) -> None:
        super().__init__()
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.n_heads   = n_heads
        self.head_dim  = out_dim // n_heads
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        # Query/Key/Value projections per head
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Edge type embeddings (heterogeneous: different edge types = different biases)
        self.edge_embeddings = nn.Embedding(n_edge_types, out_dim)

        # Attention scoring vector per head
        self.attn_vec = nn.Parameter(torch.zeros(n_heads, self.head_dim * 3))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)

        # Residual: project input if dims differ
        self.residual = residual
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = None

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index:    torch.Tensor,
        edge_types:    Optional[torch.Tensor] = None,
        edge_weights:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Graph attention forward pass.

        Args:
            node_features: (N, in_dim)  — N = total nodes in graph.
            edge_index:    (2, E)       — [src_nodes, dst_nodes] for E edges.
            edge_types:    (E,)         — edge type index for each edge.
            edge_weights:  (E,)         — optional scalar edge weight (e.g. correlation r).

        Returns:
            (N, out_dim) — updated node features.
        """
        N = node_features.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # ── Project to Q/K/V ──────────────────────────────────────────────────
        # (N, n_heads, head_dim)
        Q = self.W_q(node_features).view(N, self.n_heads, self.head_dim)
        K = self.W_k(node_features).view(N, self.n_heads, self.head_dim)
        V = self.W_v(node_features).view(N, self.n_heads, self.head_dim)

        # ── Edge type embedding ───────────────────────────────────────────────
        if edge_types is not None:
            E_emb = self.edge_embeddings(edge_types)   # (E, out_dim)
            E_emb = E_emb.view(-1, self.n_heads, self.head_dim)  # (E, H, D)
        else:
            E_emb = torch.zeros(edge_index.size(1), self.n_heads, self.head_dim,
                                device=node_features.device)

        # ── Attention scores ──────────────────────────────────────────────────
        # For each edge (u→v): score = a^T * LeakyReLU([Q_v || K_u || E_emb])
        Q_dst = Q[dst_idx]   # (E, H, D)
        K_src = K[src_idx]   # (E, H, D)

        # Concatenate along head_dim → (E, H, 3D)
        cat = torch.cat([Q_dst, K_src, E_emb], dim=-1)   # (E, H, 3D)
        scores = F.leaky_relu(
            (cat * self.attn_vec.unsqueeze(0)).sum(dim=-1),  # (E, H)
            negative_slope=0.2,
        )

        # Incorporate optional edge weights (e.g. |pearson_r|)
        if edge_weights is not None:
            scores = scores + edge_weights.unsqueeze(-1).log1p()

        # Softmax over incoming edges per node per head
        # Use scatter_softmax pattern via subtraction trick
        # First: find max score per (dst_node, head) for numerical stability
        attn = torch.zeros(N, self.n_heads, device=node_features.device)
        score_exp = torch.exp(scores - scores.max())  # (E, H) — approximate stability

        # Scatter: accumulate exp(scores) at destination nodes
        attn_sum = torch.zeros(N, self.n_heads, device=node_features.device)
        attn_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(score_exp), score_exp)

        # Normalize (avoid div by zero for isolated nodes)
        norm_scores = score_exp / (attn_sum[dst_idx] + 1e-9)  # (E, H)
        norm_scores = self.dropout(norm_scores)

        # ── Weighted value aggregation ────────────────────────────────────────
        # h_v' = sum_{u in N(v)} alpha_{vu} * V_u
        V_src = V[src_idx]                   # (E, H, D)
        weighted_V = norm_scores.unsqueeze(-1) * V_src  # (E, H, D)

        # Scatter sum at destination nodes
        agg = torch.zeros(N, self.n_heads, self.head_dim, device=node_features.device)
        idx = dst_idx.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)
        agg.scatter_add_(0, idx, weighted_V)

        # Reshape: (N, n_heads, head_dim) → (N, out_dim)
        agg = agg.reshape(N, self.out_dim)
        out = F.elu(self.out_proj(agg))

        # ── Residual ──────────────────────────────────────────────────────────
        if self.residual:
            res = self.res_proj(node_features) if self.res_proj else node_features
            out = out + res

        return self.norm(out)


# ── Full graph network (stacked GAT layers) ────────────────────────────────────

class FinBrainGNN(nn.Module):
    """Stacked heterogeneous GAT layers forming the knowledge propagation network.

    Information flow across 3 hops:
      Layer 1: each node sees its direct neighbors (supply chain partners,
               correlated assets, macro sector impacts)
      Layer 2: each node sees 2-hop neighbors (e.g. NVDA learns from TSMC's
               macro context via TSMC's embedding)
      Layer 3: 3-hop — systemic effects propagate (credit stress → banks →
               broader market)

    After all layers, each node embedding encodes:
      - Its own temporal state
      - The state of everything it's connected to
      - The state of things connected to its connections

    Args:
        in_dim:     Input node feature dimension (from modality encoders).
        hidden_dim: Hidden layer dimension.
        out_dim:    Output embedding dimension per node.
        n_layers:   Number of GAT layers (hops).
        n_heads:    Attention heads per layer.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        in_dim:     int   = 128,
        hidden_dim: int   = 256,
        out_dim:    int   = 128,
        n_layers:   int   = 3,
        n_heads:    int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            HeteroGATLayer(
                in_dim  = dims[i],
                out_dim = dims[i + 1],
                n_heads = n_heads,
                dropout = dropout,
            )
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index:    torch.Tensor,
        edge_types:    Optional[torch.Tensor] = None,
        edge_weights:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run GNN forward pass.

        Args:
            node_features: (N, in_dim)
            edge_index:    (2, E)
            edge_types:    (E,)
            edge_weights:  (E,)

        Returns:
            (N, out_dim) — enriched node embeddings after graph propagation.
        """
        h = node_features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(h, edge_index, edge_types, edge_weights)
        return h


# ── Graph builder: convert knowledge base to tensor form ──────────────────────

class FinancialGraph:
    """Converts FinBrain's knowledge base into a graph tensor representation.

    Maintains the node index map, edge list, and edge attributes.
    Updated dynamically as new correlations are discovered.

    Usage:
        graph = FinancialGraph()
        graph.add_assets(EQUITIES + ETFS + CRYPTO_YF)
        graph.add_macro_indicators([s[0] for s in MACRO_SERIES])
        graph.add_supply_chain_edges(SUPPLY_CHAIN_MAP)
        graph.add_correlation_edges(findings)

        # Get tensors for GNN:
        node_ids, edge_index, edge_types, edge_weights = graph.to_tensors()
    """

    def __init__(self) -> None:
        self.node_to_idx: dict[str, int] = {}
        self.node_types:  dict[int, str]  = {}   # idx → 'asset'|'macro'|'sector'
        self._edges: list[tuple[int, int, int, float]] = []  # (src, dst, type, weight)

    @property
    def n_nodes(self) -> int:
        return len(self.node_to_idx)

    def _get_or_add(self, node_id: str, node_type: str = "asset") -> int:
        if node_id not in self.node_to_idx:
            idx = len(self.node_to_idx)
            self.node_to_idx[node_id] = idx
            self.node_types[idx] = node_type
        return self.node_to_idx[node_id]

    def add_assets(self, symbols: list[str]) -> None:
        for s in symbols:
            self._get_or_add(s, "asset")

    def add_macro_indicators(self, series_ids: list[str]) -> None:
        for s in series_ids:
            self._get_or_add(s, "macro")

    def add_sectors(self, sectors: list[str]) -> None:
        for s in sectors:
            self._get_or_add(f"SECTOR:{s}", "sector")

    def add_supply_chain_edges(
        self,
        supply_map: list[tuple[str, str, str, float | None]],
    ) -> None:
        """Add supply chain edges from SUPPLY_CHAIN_MAP."""
        etype = EDGE_TYPES["supplies_to"]
        for supplier, customer, _product, rev_pct in supply_map:
            s = self._get_or_add(supplier, "asset")
            c = self._get_or_add(customer, "asset")
            weight = float(rev_pct) if rev_pct else 0.1
            self._edges.append((s, c, etype, weight))
            # Bidirectional (customer also knows about supplier)
            self._edges.append((c, s, etype, weight * 0.5))

    def add_macro_sector_edges(
        self,
        macro_impacts: list[tuple[str, str, str, str]],
    ) -> None:
        """Add macro → sector impact edges from MACRO_SECTOR_IMPACTS."""
        etype = EDGE_TYPES["impacts"]
        for macro, etf, direction, _mechanism in macro_impacts:
            m = self._get_or_add(macro, "macro")
            a = self._get_or_add(etf,   "asset")
            weight = 1.0 if direction == "positive" else -1.0
            self._edges.append((m, a, etype, abs(weight)))

    def add_sector_membership_edges(self, sector_map: dict[str, str]) -> None:
        """Add Asset → Sector edges from SECTOR_MAP."""
        etype = EDGE_TYPES["belongs_to"]
        for symbol, sector in sector_map.items():
            a = self._get_or_add(symbol, "asset")
            s = self._get_or_add(f"SECTOR:{sector}", "sector")
            self._edges.append((a, s, etype, 1.0))
            self._edges.append((s, a, etype, 1.0))  # sector state flows back to members

    def add_correlation_edges(self, findings: list) -> None:
        """Add discovered correlation edges from CorrelationFinding list."""
        for f in findings:
            etype = EDGE_TYPES.get(f.relationship_type, EDGE_TYPES["correlates_with"])
            if etype == EDGE_TYPES["leads"]:
                # Directed: series_a → series_b (series_a leads)
                a = self._get_or_add(f.series_a, "asset")
                b = self._get_or_add(f.series_b, "asset")
                self._edges.append((a, b, etype, abs(f.pearson_r)))
            else:
                # Undirected: add both directions
                a = self._get_or_add(f.series_a, "asset")
                b = self._get_or_add(f.series_b, "asset")
                w = abs(f.pearson_r)
                self._edges.append((a, b, etype, w))
                self._edges.append((b, a, etype, w))

    def add_causal_edges(self, relationships: list[tuple[str, str, int, str]]) -> None:
        """Add Granger-causal edges from KNOWN_RELATIONSHIPS."""
        for cause, effect, _lag, _category in relationships:
            etype = EDGE_TYPES["leads"]
            c = self._get_or_add(cause,  "macro")
            e = self._get_or_add(effect, "asset")
            self._edges.append((c, e, etype, 1.0))

    def to_tensors(self, device: str = "cpu") -> tuple[
        list[str],          # node_ids in order
        torch.Tensor,       # edge_index (2, E)
        torch.Tensor,       # edge_types (E,)
        torch.Tensor,       # edge_weights (E,)
    ]:
        """Convert graph to tensor form for GNN input."""
        node_ids = [None] * self.n_nodes
        for nid, idx in self.node_to_idx.items():
            node_ids[idx] = nid

        if not self._edges:
            # Empty graph — return minimal self-loops so model can run
            N = self.n_nodes
            edge_index   = torch.zeros(2, N, dtype=torch.long, device=device)
            edge_types   = torch.zeros(N,    dtype=torch.long, device=device)
            edge_weights = torch.ones(N,     dtype=torch.float, device=device)
            return node_ids, edge_index, edge_types, edge_weights

        srcs, dsts, etypes, eweights = zip(*self._edges)
        edge_index   = torch.tensor([list(srcs), list(dsts)], dtype=torch.long,  device=device)
        edge_types   = torch.tensor(list(etypes),             dtype=torch.long,  device=device)
        edge_weights = torch.tensor(list(eweights),           dtype=torch.float, device=device)
        return node_ids, edge_index, edge_types, edge_weights

    def get_neighbors(self, node_id: str, max_hops: int = 2) -> dict[str, list[str]]:
        """Get neighborhood of a node for explanation/visualization.

        Args:
            node_id:  Node identifier.
            max_hops: Number of hops to traverse.

        Returns:
            {hop_1: [neighbor_ids], hop_2: [...], ...}
        """
        if node_id not in self.node_to_idx:
            return {}
        start = self.node_to_idx[node_id]
        # Build adjacency list
        adj: dict[int, list[int]] = {}
        for src, dst, _etype, _w in self._edges:
            adj.setdefault(src, []).append(dst)

        result: dict[str, list[str]] = {}
        frontier = {start}
        visited  = {start}
        idx_to_id = {v: k for k, v in self.node_to_idx.items()}

        for hop in range(1, max_hops + 1):
            next_frontier: set[int] = set()
            for node in frontier:
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            result[f"hop_{hop}"] = [idx_to_id[n] for n in next_frontier]
            frontier = next_frontier
            if not frontier:
                break

        return result

    @classmethod
    def build_from_knowledge_base(cls, device: str = "cpu") -> "FinancialGraph":
        """Factory: build the full graph from FinBrain's knowledge base.

        Pulls from:
          - universe.py (EQUITIES, ETFS, MACRO_SERIES, SECTOR_MAP, KNOWN_RELATIONSHIPS)
          - knowledge_builder.py (SUPPLY_CHAIN_MAP, MACRO_SECTOR_IMPACTS)

        Returns:
            Fully populated FinancialGraph ready for GNN.
        """
        from data.ingest.universe import (
            EQUITIES, ETFS, CRYPTO_YF, MACRO_SERIES, SECTOR_MAP, KNOWN_RELATIONSHIPS,
        )
        from data.agents.knowledge_builder import SUPPLY_CHAIN_MAP, MACRO_SECTOR_IMPACTS

        g = cls()

        # Add all nodes
        g.add_assets(EQUITIES + ETFS + CRYPTO_YF)
        g.add_macro_indicators([s[0] for s in MACRO_SERIES])
        sectors = list(set(SECTOR_MAP.values()))
        g.add_sectors(sectors)

        # Add all edges
        g.add_supply_chain_edges(SUPPLY_CHAIN_MAP)
        g.add_macro_sector_edges(MACRO_SECTOR_IMPACTS)
        g.add_sector_membership_edges(SECTOR_MAP)
        g.add_causal_edges(KNOWN_RELATIONSHIPS)

        return g
