"""FinBrainNet — unified multi-modal graph neural network.

This is the single model that sees EVERYTHING simultaneously:
  - Price/volume time series for every asset in the universe
  - 65 macro FRED series
  - Supply chain and correlation relationships in graph form
  - Fundamental ratios (where available)
  - Social sentiment signals

And produces:
  - Per-asset direction probability (5-day forward return > 0)
  - Per-asset return magnitude estimate
  - Macro regime embedding (for unsupervised regime detection)
  - Node embeddings (for pattern discovery and visualization)

The power comes from the graph: NVDA's prediction is informed by:
  - Its own price/technical state
  - TSMC's production state (supplier edge)
  - T10Y2Y yield curve (macro impact edge)
  - MSFT/GOOGL/META demand state (customer edges)
  - SOX semiconductor sector peers state
  - VIX risk-off signal (macro impact edge)

No human analyst could track all these simultaneously. The GNN does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.neural.encoders import (
    TemporalEncoder, MacroEncoder, FundamentalEncoder,
    SentimentEncoder, ModalFusion,
)
from ml.neural.graph_net import FinBrainGNN, FinancialGraph, EDGE_TYPES

log = logging.getLogger(__name__)


# ── Input / output data structures ────────────────────────────────────────────

@dataclass
class NodeInputs:
    """All inputs for a single node in the graph.

    All fields are optional — the model gracefully handles missing modalities
    by treating missing encoder inputs as zero embeddings.

    Attributes:
        node_id:       Symbol or series identifier (e.g. 'AAPL', 'T10Y2Y').
        node_type:     'asset' | 'macro' | 'sector'.
        price_seq:     (seq_len, n_features) float tensor — technical features.
        macro_seq:     (seq_len, 4) float tensor — [value, z, chg, sign].
        fundamentals:  (n_fund_features,) float tensor.
        sentiment:     (n_sent_features,) float tensor.
    """
    node_id:      str
    node_type:    str              = "asset"
    price_seq:    Optional[torch.Tensor] = None   # (T, F_price)
    macro_seq:    Optional[torch.Tensor] = None   # (T, 4)
    fundamentals: Optional[torch.Tensor] = None   # (F_fund,)
    sentiment:    Optional[torch.Tensor] = None   # (F_sent,)


@dataclass
class GraphPrediction:
    """Output of FinBrainNet for the target node(s).

    Attributes:
        node_ids:         List of node identifiers.
        direction_probs:  (N,) probability that 5-day return > 0.
        return_estimates: (N,) estimated 5-day return (%).
        confidence:       (N,) model confidence (entropy-based).
        node_embeddings:  (N, embed_dim) for visualization / pattern mining.
        attention_weights: Dict of attention scores on top neighbors.
    """
    node_ids:         list[str]
    direction_probs:  torch.Tensor
    return_estimates: torch.Tensor
    confidence:       torch.Tensor
    node_embeddings:  torch.Tensor
    attention_weights: dict[str, float] | None = None


# ── Prediction head ────────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """Per-node prediction head: embedding → direction prob + return magnitude.

    Args:
        embed_dim: Input embedding dimension.
        dropout:   Dropout probability.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Direction: binary classification
        self.direction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Return magnitude: regression (signed % return)
        self.return_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),          # bounded: ±1 = ±100% (scaled at inference)
        )
        # Uncertainty head: aleatoric uncertainty estimate
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),       # positive uncertainty
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict direction, return, and uncertainty.

        Args:
            x: (batch, embed_dim) node embeddings.

        Returns:
            (direction_prob, return_estimate, uncertainty) each of shape (batch, 1).
        """
        h  = self.shared(x)
        return (
            self.direction_head(h),
            self.return_head(h) * 20,  # scale to ±20% range
            self.uncertainty_head(h),
        )


# ── Main unified model ─────────────────────────────────────────────────────────

class FinBrainNet(nn.Module):
    """The unified financial intelligence model.

    Combines all modality encoders + graph attention + prediction head
    into a single differentiable model that can be trained end-to-end.

    Architecture:
      1. Encode each node's available data into a fixed-dim embedding
         (TemporalEncoder for assets, MacroEncoder for FRED series)
      2. Fuse modalities with cross-attention (ModalFusion)
      3. Run graph attention across 3 hops (FinBrainGNN)
         — information propagates across supply chains, correlations, macro
      4. Predict from enriched graph embeddings (PredictionHead)

    Args:
        price_input_size:  Number of technical features per timestep.
        fundamental_size:  Number of fundamental ratio features.
        sentiment_size:    Number of sentiment signal features.
        embed_dim:         Embedding dimension throughout the model.
        gnn_hidden_dim:    GNN hidden layer width.
        gnn_layers:        Number of graph propagation hops.
        n_heads:           Attention heads.
        seq_len:           Time series sequence length.
        dropout:           Dropout probability.
    """

    def __init__(
        self,
        price_input_size:  int   = 80,
        fundamental_size:  int   = 20,
        sentiment_size:    int   = 8,
        embed_dim:         int   = 128,
        gnn_hidden_dim:    int   = 256,
        gnn_layers:        int   = 3,
        n_heads:           int   = 4,
        seq_len:           int   = 63,
        macro_seq_len:     int   = 63,
        dropout:           float = 0.15,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # ── Modality encoders ──────────────────────────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            input_size  = price_input_size,
            embed_dim   = embed_dim,
            hidden_size = 256,
            num_layers  = 2,
            dropout     = dropout,
        )
        self.macro_encoder = MacroEncoder(
            seq_len   = macro_seq_len,
            embed_dim = embed_dim,
            d_model   = 64,
            n_heads   = 4,
            n_layers  = 2,
            dropout   = dropout,
        )
        self.fundamental_encoder = FundamentalEncoder(
            input_size = fundamental_size,
            embed_dim  = embed_dim,
            hidden_dim = 256,
            dropout    = dropout,
        )
        self.sentiment_encoder = SentimentEncoder(
            input_size = sentiment_size,
            embed_dim  = embed_dim,
            dropout    = dropout,
        )

        # ── Modal fusion ───────────────────────────────────────────────────────
        self.modal_fusion = ModalFusion(
            embed_dim    = embed_dim,
            n_modalities = 4,  # price, macro, fundamental, sentiment
            n_heads      = n_heads,
            dropout      = dropout,
        )

        # ── Graph attention network ────────────────────────────────────────────
        self.gnn = FinBrainGNN(
            in_dim     = embed_dim,
            hidden_dim = gnn_hidden_dim,
            out_dim    = embed_dim,
            n_layers   = gnn_layers,
            n_heads    = n_heads,
            dropout    = dropout,
        )

        # ── Prediction head ────────────────────────────────────────────────────
        self.pred_head = PredictionHead(embed_dim=embed_dim, dropout=dropout)

        # ── Node type embeddings ───────────────────────────────────────────────
        # Learnable biases per node type (asset/macro/sector behave differently)
        self.node_type_emb = nn.Embedding(3, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_node(self, inputs: NodeInputs) -> torch.Tensor:
        """Encode a single node's data to an embedding.

        This is called per-node before graph propagation.
        Returns a (1, embed_dim) embedding.

        Args:
            inputs: NodeInputs dataclass with available modality data.

        Returns:
            (1, embed_dim) initial node embedding.
        """
        modalities = []

        # Price/volume technical features
        if inputs.price_seq is not None:
            x = inputs.price_seq.unsqueeze(0)  # (1, T, F)
            modalities.append(self.temporal_encoder(x))
        else:
            modalities.append(torch.zeros(1, self.embed_dim))

        # Macro series (for macro-type nodes)
        if inputs.macro_seq is not None:
            x = inputs.macro_seq.unsqueeze(0)  # (1, T, 4)
            modalities.append(self.macro_encoder(x))
        else:
            modalities.append(torch.zeros(1, self.embed_dim))

        # Fundamental ratios
        if inputs.fundamentals is not None:
            x = inputs.fundamentals.unsqueeze(0)  # (1, F_fund)
            modalities.append(self.fundamental_encoder(x))
        else:
            modalities.append(torch.zeros(1, self.embed_dim))

        # Sentiment signals
        if inputs.sentiment is not None:
            x = inputs.sentiment.unsqueeze(0)  # (1, F_sent)
            modalities.append(self.sentiment_encoder(x))
        else:
            modalities.append(torch.zeros(1, self.embed_dim))

        # Fuse modalities
        return self.modal_fusion(modalities)  # (1, embed_dim)

    def forward(
        self,
        node_inputs:   list[NodeInputs],
        graph:         FinancialGraph,
        target_ids:    list[str] | None = None,
        device:        str = "cpu",
    ) -> GraphPrediction:
        """Full forward pass: encode all nodes, propagate through graph, predict.

        Args:
            node_inputs: List of NodeInputs for each node with data available.
                         Nodes in the graph without inputs get zero embeddings.
            graph:       FinancialGraph defining the structure.
            target_ids:  Which nodes to return predictions for (None = all).
            device:      Torch device.

        Returns:
            GraphPrediction with direction probs, returns, confidence, embeddings.
        """
        N = graph.n_nodes
        node_ids, edge_index, edge_types, edge_weights = graph.to_tensors(device)

        # ── Build node feature matrix ──────────────────────────────────────────
        # Start with zeros; fill in for nodes that have data
        node_features = torch.zeros(N, self.embed_dim, device=device)

        # Add node-type bias embeddings
        type_to_idx = {"asset": 0, "macro": 1, "sector": 2}
        for idx, nid in enumerate(node_ids):
            if nid is not None:
                ntype = graph.node_types.get(idx, "asset")
                type_idx = type_to_idx.get(ntype, 0)
                node_features[idx] += self.node_type_emb(
                    torch.tensor(type_idx, device=device)
                )

        # Encode each node with available data
        input_map = {ni.node_id: ni for ni in node_inputs}
        for idx, nid in enumerate(node_ids):
            if nid is not None and nid in input_map:
                emb = self.encode_node(input_map[nid]).squeeze(0).to(device)
                node_features[idx] = emb

        # ── Graph propagation ──────────────────────────────────────────────────
        enriched = self.gnn(node_features, edge_index, edge_types, edge_weights)
        # (N, embed_dim) — each node now contains information from its k-hop neighborhood

        # ── Predict for target nodes ───────────────────────────────────────────
        if target_ids is None:
            target_ids = [nid for nid in node_ids if nid is not None]

        target_indices = []
        valid_ids = []
        for tid in target_ids:
            if tid in graph.node_to_idx:
                target_indices.append(graph.node_to_idx[tid])
                valid_ids.append(tid)

        if not target_indices:
            return GraphPrediction(
                node_ids         = [],
                direction_probs  = torch.tensor([]),
                return_estimates = torch.tensor([]),
                confidence       = torch.tensor([]),
                node_embeddings  = torch.tensor([]),
            )

        target_embs = enriched[target_indices]  # (n_targets, embed_dim)
        dir_probs, ret_est, uncertainty = self.pred_head(target_embs)

        # Confidence = 1 / (1 + uncertainty)
        confidence = 1.0 / (1.0 + uncertainty)

        return GraphPrediction(
            node_ids         = valid_ids,
            direction_probs  = dir_probs.squeeze(-1),
            return_estimates = ret_est.squeeze(-1),
            confidence       = confidence.squeeze(-1),
            node_embeddings  = target_embs,
        )

    def get_attention_explanation(
        self,
        target_id: str,
        graph:     FinancialGraph,
        top_k:     int = 10,
    ) -> list[dict[str, Any]]:
        """Explain which neighbors most influenced a node's prediction.

        Returns the top-k most attended neighbors with their attention scores
        and relationship types, enabling human-readable explanation:
          "NVDA's prediction was 42% influenced by T10Y2Y (macro impact edge)"

        Args:
            target_id: The node whose prediction we're explaining.
            graph:     The FinancialGraph.
            top_k:     Number of top-influencing neighbors to return.

        Returns:
            List of {neighbor_id, edge_type, attention_score, description}.
        """
        if target_id not in graph.node_to_idx:
            return []

        target_idx = graph.node_to_idx[target_id]
        idx_to_id  = {v: k for k, v in graph.node_to_idx.items()}
        type_to_str = {v: k for k, v in EDGE_TYPES.items()}

        # Collect all incoming edges for target node
        influences = []
        for src, dst, etype, weight in graph._edges:
            if dst == target_idx:
                neighbor_id = idx_to_id.get(src, f"node_{src}")
                edge_type   = type_to_str.get(etype, "unknown")
                influences.append({
                    "neighbor_id":     neighbor_id,
                    "edge_type":       edge_type,
                    "edge_weight":     round(weight, 3),
                    "neighbor_type":   graph.node_types.get(src, "unknown"),
                })

        # Sort by edge weight as proxy for importance
        influences.sort(key=lambda x: x["edge_weight"], reverse=True)
        return influences[:top_k]


# ── Model registry helpers ─────────────────────────────────────────────────────

MODEL_CONFIG_DEFAULT = {
    "price_input_size": 80,
    "fundamental_size": 20,
    "sentiment_size":   8,
    "embed_dim":        128,
    "gnn_hidden_dim":   256,
    "gnn_layers":       3,
    "n_heads":          4,
    "seq_len":          63,
    "macro_seq_len":    63,
    "dropout":          0.15,
}


def build_model(config: dict | None = None) -> FinBrainNet:
    """Build FinBrainNet from config dict.

    Args:
        config: Model hyperparameters. Defaults to MODEL_CONFIG_DEFAULT.

    Returns:
        Initialized FinBrainNet.
    """
    cfg = {**MODEL_CONFIG_DEFAULT, **(config or {})}
    return FinBrainNet(**cfg)


def save_model(
    model:    FinBrainNet,
    path:     str,
    config:   dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Save FinBrainNet checkpoint.

    Args:
        model:    Trained model.
        path:     Save path (.pt).
        config:   Model config for reconstruction.
        metadata: Extra metadata (training metrics, graph stats, etc.).
    """
    torch.save({
        "model_state":  model.state_dict(),
        "config":       config or MODEL_CONFIG_DEFAULT,
        "metadata":     metadata or {},
        "version":      "finbrainnet_v1",
    }, path)
    log.info("Saved FinBrainNet to %s", path)


def load_model(path: str, device: str = "cpu") -> tuple[FinBrainNet, dict]:
    """Load FinBrainNet from checkpoint.

    Args:
        path:   Checkpoint path.
        device: Target device.

    Returns:
        (model, checkpoint_dict) with model in eval mode.
    """
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    model  = build_model(ckpt.get("config"))
    model.load_state_dict(ckpt["model_state"])
    model  = model.to(device)
    model.eval()
    return model, ckpt
