"""Graph enrichment layer for the FinBrain research loop.

Converts evidence-backed claims into local graph nodes and edges.
Nothing here auto-promotes, calls external services, or modifies model weights.

Public API::

    from ml.graph import (
        NodeType, GraphEdgeStatus,
        GraphNode, GraphEdge,
        GraphStore,
        nodes_from_claim, edge_from_claim,
        enrich_graph_from_claims, EnrichmentReport,
    )
"""
from ml.graph.schema import (
    NodeType,
    GraphEdgeStatus,
    GraphNode,
    GraphEdge,
)
from ml.graph.store import GraphStore
from ml.graph.converter import nodes_from_claim, edge_from_claim
from ml.graph.enrichment import enrich_graph_from_claims, EnrichmentReport

__all__ = [
    # schema
    "NodeType",
    "GraphEdgeStatus",
    "GraphNode",
    "GraphEdge",
    # store
    "GraphStore",
    # converter
    "nodes_from_claim",
    "edge_from_claim",
    # enrichment
    "enrich_graph_from_claims",
    "EnrichmentReport",
]
