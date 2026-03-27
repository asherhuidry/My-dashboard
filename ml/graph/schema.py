"""Core graph node and edge schemas for the FinBrain graph layer.

Everything is serializable plain data.  No inference, no external database,
no auto-promotion.  Nodes represent named entities (features, assets, models,
data sources, datasets).  Edges represent relationships that are backed by at
least one evidence-verified claim.

GraphNode
    A named entity in the research domain.

GraphEdge
    A directed relationship between two nodes, backed by a claim and its
    evidence.

GraphEdgeStatus
    Lifecycle of an edge — mirrors claim status so the graph stays in sync.

Design principles
-----------------
- Only supported or explicitly allowed proposed claims produce edges.
- Unsupported or rejected claims are never silently treated as valid edges.
- confidence on an edge comes from the source claim, not from any inference.
- Everything round-trips through JSON losslessly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enumerations ────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """High-level category of a graph node."""
    SOURCE   = "source"   # data source, e.g. FRED_GDP
    FEATURE  = "feature"  # engineered feature, e.g. rsi_14
    MODEL    = "model"    # trained model instance
    ASSET    = "asset"    # tradeable asset / ticker
    DATASET  = "dataset"  # versioned dataset snapshot
    DOMAIN   = "domain"   # abstract domain, e.g. macro_regime
    UNKNOWN  = "unknown"  # fallback for unrecognised namespaces


class GraphEdgeStatus(str, Enum):
    """Lifecycle of a graph edge."""
    ACTIVE   = "active"    # edge is live and based on a supported claim
    PROPOSED = "proposed"  # edge derived from a PROPOSED claim (not yet supported)
    WEAK     = "weak"      # edge derived from a WEAK claim
    REJECTED = "rejected"  # the source claim was rejected; edge kept for audit
    ARCHIVED = "archived"  # edge is no longer relevant


# ── GraphNode ────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A named entity in the research domain.

    Attributes:
        node_id:    Stable identifier, e.g. ``"feature:rsi_14"``.
                    Must be unique within a GraphStore.
        node_type:  High-level category (NodeType enum value).
        label:      Human-readable display name.
        properties: Arbitrary extra metadata (ticker info, description, etc.).
        created_at: ISO-8601 UTC timestamp of when the node was first recorded.
        updated_at: ISO-8601 UTC timestamp of last modification.
    """
    node_id:    str
    node_type:  NodeType
    label:      str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "node_id":    self.node_id,
            "node_type":  self.node_type.value,
            "label":      self.label,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphNode":
        """Reconstruct from a dictionary produced by ``to_dict()``."""
        d = dict(d)
        d["node_type"] = NodeType(d["node_type"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_entity_id(
        cls,
        entity_id:  str,
        properties: dict[str, Any] | None = None,
    ) -> "GraphNode":
        """Create a GraphNode by parsing a namespaced entity ID.

        Supports the standard evidence naming conventions::

            source:<id>       → NodeType.SOURCE,  label = "<id>"
            feature:<name>    → NodeType.FEATURE, label = "<name>"
            model:<desc>      → NodeType.MODEL,   label = "<desc>"
            asset:<ticker>    → NodeType.ASSET,   label = "<ticker>"
            dataset:<version> → NodeType.DATASET, label = "<version>"
            domain:<desc>     → NodeType.DOMAIN,  label = "<desc>"

        Unknown or missing namespace prefixes fall back to NodeType.UNKNOWN.

        Args:
            entity_id:  The namespaced entity string (must be non-empty).
            properties: Optional extra metadata.

        Returns:
            A new GraphNode.
        """
        _NS_MAP = {
            "source":  NodeType.SOURCE,
            "feature": NodeType.FEATURE,
            "model":   NodeType.MODEL,
            "asset":   NodeType.ASSET,
            "dataset": NodeType.DATASET,
            "domain":  NodeType.DOMAIN,
        }
        if ":" in entity_id:
            ns, _, rest = entity_id.partition(":")
            node_type = _NS_MAP.get(ns, NodeType.UNKNOWN)
            label = rest
        else:
            node_type = NodeType.UNKNOWN
            label = entity_id

        return cls(
            node_id    = entity_id,
            node_type  = node_type,
            label      = label,
            properties = properties or {},
        )


# ── GraphEdge ────────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    """A directed relationship between two graph nodes.

    Edges carry the source claim's confidence and evidence IDs so the
    connection is fully auditable.  An edge should only be created for
    claims that have been reviewed (or explicitly allowed as proposed).

    Attributes:
        edge_id:        Unique identifier (UUID hex).
        source_node_id: The subject/from node.
        target_node_id: The object/to node.
        relation:       The predicate from the source claim.
        status:         Lifecycle status.
        confidence:     Copied from the source claim; in [0.0, 1.0].
        claim_id:       ID of the claim this edge was derived from.
        evidence_ids:   IDs of EvidenceItems supporting the claim.
        properties:     Extra metadata — uncertainty notes, counterpoints, etc.
        created_at:     ISO-8601 UTC creation timestamp.
        updated_at:     ISO-8601 UTC last-modification timestamp.
    """
    edge_id:        str
    source_node_id: str
    target_node_id: str
    relation:       str
    status:         GraphEdgeStatus
    confidence:     float
    claim_id:       str
    evidence_ids:   list[str] = field(default_factory=list)
    properties:     dict[str, Any] = field(default_factory=dict)
    created_at:     str = ""
    updated_at:     str = ""

    def __post_init__(self) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "edge_id":        self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation":       self.relation,
            "status":         self.status.value,
            "confidence":     self.confidence,
            "claim_id":       self.claim_id,
            "evidence_ids":   list(self.evidence_ids),
            "properties":     self.properties,
            "created_at":     self.created_at,
            "updated_at":     self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraphEdge":
        """Reconstruct from a dictionary produced by ``to_dict()``."""
        d = dict(d)
        d["status"] = GraphEdgeStatus(d["status"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        source_node_id: str,
        target_node_id: str,
        relation:       str,
        status:         GraphEdgeStatus,
        confidence:     float,
        claim_id:       str,
        evidence_ids:   list[str] | None = None,
        properties:     dict[str, Any] | None = None,
    ) -> "GraphEdge":
        """Create a new GraphEdge with a fresh UUID.

        Args:
            source_node_id: The subject/from node ID.
            target_node_id: The object/to node ID.
            relation:       Predicate string.
            status:         Initial GraphEdgeStatus.
            confidence:     Float in [0.0, 1.0].
            claim_id:       ID of the originating claim.
            evidence_ids:   List of evidence item IDs.
            properties:     Optional extra metadata dict.

        Returns:
            A new GraphEdge.
        """
        return cls(
            edge_id        = uuid.uuid4().hex,
            source_node_id = source_node_id,
            target_node_id = target_node_id,
            relation       = relation,
            status         = status,
            confidence     = confidence,
            claim_id       = claim_id,
            evidence_ids   = list(evidence_ids or []),
            properties     = dict(properties or {}),
        )
