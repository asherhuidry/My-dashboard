"""Local-first graph store with JSON persistence.

Stores GraphNodes and GraphEdges in a single JSON file, with write-through
semantics (every mutation saves immediately).  Mirrors the design of
ClaimStore so the two layers stay consistent in style.

Default storage path: ``data/graph/graph.json``
Override with env var: ``FINBRAIN_GRAPH_PATH``

Usage::

    from ml.graph.store import GraphStore
    from ml.graph.schema import GraphNode, GraphEdge, GraphEdgeStatus, NodeType

    store = GraphStore()

    node = GraphNode.from_entity_id("feature:rsi_14")
    store.upsert_node(node)

    edge = GraphEdge.new(
        source_node_id = "feature:rsi_14",
        target_node_id = "model:mlp_dataset:abc123",
        relation       = "improves_accuracy_on",
        status         = GraphEdgeStatus.ACTIVE,
        confidence     = 0.65,
        claim_id       = "abc...",
        evidence_ids   = ["ev1", "ev2"],
    )
    store.add_edge(edge)

    # query
    neighbors = store.neighbors("feature:rsi_14")
    edges = store.edges_by_claim("abc...")
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.graph.schema import GraphEdge, GraphEdgeStatus, GraphNode, NodeType

log = logging.getLogger(__name__)

_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "graph" / "graph.json"
)
_FILE_VERSION = "1"


class GraphStore:
    """Local graph store backed by a JSON file.

    All mutations are write-through.  Nodes are keyed by ``node_id``; edges
    are keyed by ``edge_id``.

    Args:
        path: Path to the JSON file.  Defaults to
              ``data/graph/graph.json`` (or ``FINBRAIN_GRAPH_PATH`` env var).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = os.environ.get("FINBRAIN_GRAPH_PATH", str(_DEFAULT_PATH))
        self._path = Path(path)
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load state from disk if the file exists."""
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
            for nd in raw.get("nodes", []):
                node = GraphNode.from_dict(nd)
                self._nodes[node.node_id] = node
            for ed in raw.get("edges", []):
                edge = GraphEdge.from_dict(ed)
                self._edges[edge.edge_id] = edge
        except Exception:
            log.exception("Failed to load graph from %s; starting empty.", self._path)

    def _save(self) -> None:
        """Write current state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version":    _FILE_VERSION,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "nodes":      [n.to_dict() for n in self._nodes.values()],
            "edges":      [e.to_dict() for e in self._edges.values()],
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ── Node operations ───────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the store.

        Raises:
            ValueError: If a node with the same ``node_id`` already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists.")
        self._nodes[node.node_id] = node
        self._save()

    def upsert_node(self, node: GraphNode) -> bool:
        """Insert or update a node.  Returns True if a new node was created.

        If the node already exists, its ``properties``, ``label``, and
        ``updated_at`` are updated in place.

        Args:
            node: The GraphNode to upsert.

        Returns:
            True if created, False if updated.
        """
        is_new = node.node_id not in self._nodes
        if not is_new:
            existing = self._nodes[node.node_id]
            existing.label      = node.label
            existing.properties.update(node.properties)
            existing.updated_at = datetime.now(tz=timezone.utc).isoformat()
        else:
            self._nodes[node.node_id] = node
        self._save()
        return is_new

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the node with the given ID, or None if not found."""
        return self._nodes.get(node_id)

    def list_nodes(self, node_type: NodeType | str | None = None) -> list[GraphNode]:
        """Return all nodes, optionally filtered by node_type.

        Args:
            node_type: If given, only nodes of this type are returned.

        Returns:
            List of matching GraphNodes.
        """
        nodes = list(self._nodes.values())
        if node_type is not None:
            nt = NodeType(node_type) if isinstance(node_type, str) else node_type
            nodes = [n for n in nodes if n.node_type == nt]
        return nodes

    # ── Edge operations ───────────────────────────────────────────────────

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the store.

        Raises:
            ValueError: If an edge with the same ``edge_id`` already exists.
        """
        if edge.edge_id in self._edges:
            raise ValueError(f"Edge '{edge.edge_id}' already exists.")
        self._edges[edge.edge_id] = edge
        self._save()

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Return the edge with the given ID, or None if not found."""
        return self._edges.get(edge_id)

    def list_edges(
        self,
        relation: str | None = None,
        status:   GraphEdgeStatus | str | None = None,
    ) -> list[GraphEdge]:
        """Return all edges, optionally filtered.

        Args:
            relation: If given, only edges with this relation string.
            status:   If given, only edges with this status.

        Returns:
            List of matching GraphEdges.
        """
        edges = list(self._edges.values())
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        if status is not None:
            st = GraphEdgeStatus(status) if isinstance(status, str) else status
            edges = [e for e in edges if e.status == st]
        return edges

    def edges_by_claim(self, claim_id: str) -> list[GraphEdge]:
        """Return all edges derived from a specific claim.

        Args:
            claim_id: The claim ID to look up.

        Returns:
            List of GraphEdges whose ``claim_id`` matches.
        """
        return [e for e in self._edges.values() if e.claim_id == claim_id]

    def edges_from(self, node_id: str) -> list[GraphEdge]:
        """Return all outgoing edges from a node.

        Args:
            node_id: The source node ID.

        Returns:
            List of GraphEdges where ``source_node_id == node_id``.
        """
        return [e for e in self._edges.values() if e.source_node_id == node_id]

    def edges_to(self, node_id: str) -> list[GraphEdge]:
        """Return all incoming edges to a node.

        Args:
            node_id: The target node ID.

        Returns:
            List of GraphEdges where ``target_node_id == node_id``.
        """
        return [e for e in self._edges.values() if e.target_node_id == node_id]

    def neighbors(self, node_id: str) -> list[GraphNode]:
        """Return all directly connected nodes (outgoing and incoming edges).

        Args:
            node_id: The node to find neighbors for.

        Returns:
            Deduplicated list of neighbor GraphNodes that exist in the store.
        """
        connected_ids: set[str] = set()
        for e in self._edges.values():
            if e.source_node_id == node_id:
                connected_ids.add(e.target_node_id)
            elif e.target_node_id == node_id:
                connected_ids.add(e.source_node_id)
        return [
            self._nodes[nid]
            for nid in connected_ids
            if nid in self._nodes
        ]

    # ── Summary ───────────────────────────────────────────────────────────

    def summary_stats(self) -> dict[str, Any]:
        """Return a summary of the graph store contents.

        Returns:
            Dict with ``n_nodes``, ``n_edges``, ``by_node_type``, and
            ``by_edge_status``.
        """
        by_type: dict[str, int] = {}
        for n in self._nodes.values():
            by_type[n.node_type.value] = by_type.get(n.node_type.value, 0) + 1

        by_status: dict[str, int] = {}
        for e in self._edges.values():
            by_status[e.status.value] = by_status.get(e.status.value, 0) + 1

        return {
            "n_nodes":       len(self._nodes),
            "n_edges":       len(self._edges),
            "by_node_type":  by_type,
            "by_edge_status": by_status,
        }
