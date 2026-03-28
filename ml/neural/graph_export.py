"""Graph export — Neo4j to tensor pipeline for GNN training.

Reads the live knowledge graph from Neo4j (nodes + edges with confidence,
evidence_count, and statistical properties) and converts it into the
tensor format that FinBrainGNN expects.

This bridges the materialised graph (Neo4j) and the neural representation
layer (graph_net.py) so that the GNN trains on the latest graph state
including edge confidence as attention weight priors.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from skills.logger import get_logger

log = get_logger(__name__)

# Maps Neo4j rel types to the edge type indices used by graph_net.py (EDGE_TYPES)
# See ml/neural/graph_net.py for canonical definitions:
#   0=supplies_to, 1=correlates_with, 2=impacts, 3=belongs_to,
#   4=competes_with, 5=leads, 6=unknown
_REL_TYPE_MAP: dict[str, int] = {
    "CORRELATED_WITH": 1,   # correlates_with
    "CAUSES":          5,   # leads (Granger-causal)
    "SENSITIVE_TO":    2,   # impacts (factor exposure)
    "BELONGS_TO":      3,   # belongs_to
    "TRIGGERED_BY":    5,   # leads (event → indicator, temporal)
    "IMPACTS":         2,   # impacts
    "GENERATES":       6,   # no direct match → unknown
    "TRAINED_ON":      6,   # no direct match → unknown
}


def export_graph_tensors(
    min_confidence: float = 0.0,
    use_confidence_weights: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export the Neo4j knowledge graph as tensors for GNN consumption.

    Queries all nodes and edges from Neo4j, assigns each node an integer
    index, and builds the edge_index / edge_type / edge_weight tensors
    that FinBrainGNN.forward() expects.

    When ``use_confidence_weights=True``, edge confidence scores are used
    as edge weights (attention priors), giving the GNN stronger signal
    from well-supported relationships.

    Args:
        min_confidence: Only include edges with confidence >= this value.
        use_confidence_weights: Use edge confidence as edge weight.
        device: Torch device string.

    Returns:
        Dict with:
        - node_ids: list[str] in index order
        - node_types: dict[int, str] mapping index → type
        - edge_index: Tensor (2, E)
        - edge_types: Tensor (E,)
        - edge_weights: Tensor (E,)
        - n_nodes: int
        - n_edges: int
        - stats: summary dict
    """
    from db.neo4j.client import get_driver

    driver = get_driver()

    # ── Fetch nodes ─────────────────────────────────────────────────────
    node_to_idx: dict[str, int] = {}
    node_types: dict[int, str] = {}

    with driver.session() as sess:
        # Assets
        result = sess.run("MATCH (a:Asset) RETURN a.ticker AS id")
        for record in result:
            nid = record["id"]
            if nid and nid not in node_to_idx:
                idx = len(node_to_idx)
                node_to_idx[nid] = idx
                node_types[idx] = "asset"

        # MacroIndicators
        result = sess.run("MATCH (m:MacroIndicator) RETURN m.series_id AS id")
        for record in result:
            nid = record["id"]
            if nid and nid not in node_to_idx:
                idx = len(node_to_idx)
                node_to_idx[nid] = idx
                node_types[idx] = "macro"

        # Sectors
        result = sess.run("MATCH (s:Sector) RETURN s.name AS id")
        for record in result:
            nid = record["id"]
            if nid and nid not in node_to_idx:
                idx = len(node_to_idx)
                node_to_idx[f"SECTOR:{nid}"] = idx
                node_types[idx] = "sector"

        # Events
        result = sess.run("MATCH (e:Event) RETURN e.event_id AS id")
        for record in result:
            nid = record["id"]
            if nid and nid not in node_to_idx:
                idx = len(node_to_idx)
                node_to_idx[nid] = idx
                node_types[idx] = "event"

    n_nodes = len(node_to_idx)
    log.info("Exported %d nodes from Neo4j", n_nodes)

    # ── Fetch edges ─────────────────────────────────────────────────────
    edge_query = """
        MATCH (a)-[r]->(b)
        RETURN
            coalesce(a.ticker, a.series_id, a.name, a.event_id) AS src,
            coalesce(b.ticker, b.series_id, b.name, b.event_id) AS tgt,
            type(r)                  AS rel_type,
            r.confidence             AS confidence,
            r.pearson_r              AS pearson_r,
            coalesce(r.evidence_count, 1) AS evidence_count,
            duration.between(
              coalesce(r.last_confirmed_at, r.first_seen_at, datetime()),
              datetime()
            ).days                   AS age_days
    """
    srcs: list[int] = []
    dsts: list[int] = []
    etypes: list[int] = []
    eweights: list[float] = []
    skipped = 0
    unmapped_types: set[str] = set()

    with driver.session() as sess:
        result = sess.run(edge_query)
        for record in result:
            src_id = record["src"]
            tgt_id = record["tgt"]

            # Handle sector node ID format
            src_key = f"SECTOR:{src_id}" if src_id not in node_to_idx and f"SECTOR:{src_id}" in node_to_idx else src_id
            tgt_key = f"SECTOR:{tgt_id}" if tgt_id not in node_to_idx and f"SECTOR:{tgt_id}" in node_to_idx else tgt_id

            if src_key not in node_to_idx or tgt_key not in node_to_idx:
                skipped += 1
                continue

            src_idx = node_to_idx[src_key]
            tgt_idx = node_to_idx[tgt_key]
            rel = record["rel_type"]
            if rel not in _REL_TYPE_MAP:
                unmapped_types.add(rel)
            etype = _REL_TYPE_MAP.get(rel, 6)  # 6 = unknown

            # Compute effective confidence with temporal decay
            if use_confidence_weights and record["confidence"] is not None:
                from data.agents.edge_confidence import decay_confidence
                raw_conf = float(record["confidence"])
                age = float(record["age_days"]) if record["age_days"] is not None else 0.0
                ev = int(record["evidence_count"])
                weight = decay_confidence(raw_conf, age, evidence_count=ev, rel_type=rel)
            elif record["pearson_r"] is not None:
                weight = abs(float(record["pearson_r"]))
            else:
                weight = 0.5

            # Apply min_confidence filter against effective weight
            if min_confidence > 0 and weight < min_confidence:
                skipped += 1
                continue

            srcs.append(src_idx)
            dsts.append(tgt_idx)
            etypes.append(etype)
            eweights.append(weight)

    if unmapped_types:
        log.warning("Unmapped edge types in export: %s", unmapped_types)

    n_edges = len(srcs)
    log.info("Exported %d edges from Neo4j (skipped %d)", n_edges, skipped)

    # ── Build tensors ───────────────────────────────────────────────────
    if n_edges > 0:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
        edge_types_t = torch.tensor(etypes, dtype=torch.long, device=device)
        edge_weights_t = torch.tensor(eweights, dtype=torch.float, device=device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_types_t = torch.zeros(0, dtype=torch.long, device=device)
        edge_weights_t = torch.zeros(0, dtype=torch.float, device=device)

    # Node IDs in index order
    node_ids = [""] * n_nodes
    for nid, idx in node_to_idx.items():
        node_ids[idx] = nid

    # Type distribution stats
    type_counts: dict[str, int] = {}
    for t in node_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: dict[str, int] = {}
    for et in etypes:
        # reverse lookup
        for name, idx in _REL_TYPE_MAP.items():
            if idx == et:
                rel_counts[name] = rel_counts.get(name, 0) + 1
                break

    return {
        "node_ids": node_ids,
        "node_types": node_types,
        "node_to_idx": node_to_idx,
        "edge_index": edge_index,
        "edge_types": edge_types_t,
        "edge_weights": edge_weights_t,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "stats": {
            "node_type_counts": type_counts,
            "rel_type_counts": rel_counts,
            "skipped_edges": skipped,
            "min_confidence_filter": min_confidence,
            "confidence_weighted": use_confidence_weights,
        },
    }


def export_to_file(
    path: str = "ml/outputs/graph_snapshot.pt",
    min_confidence: float = 0.0,
) -> str:
    """Export graph tensors and save to a .pt file.

    Args:
        path: Output file path.
        min_confidence: Minimum edge confidence to include.

    Returns:
        Path to saved file.
    """
    data = export_graph_tensors(min_confidence=min_confidence)

    save_data = {
        "node_ids": data["node_ids"],
        "node_types": data["node_types"],
        "edge_index": data["edge_index"],
        "edge_types": data["edge_types"],
        "edge_weights": data["edge_weights"],
        "stats": data["stats"],
    }

    torch.save(save_data, path)
    log.info("Graph snapshot saved to %s (%d nodes, %d edges)",
             path, data["n_nodes"], data["n_edges"])
    return path
