# Graph Enrichment Layer

`ml/graph/` provides a local, deterministic graph representation built on top
of the evidence and claim layer.  It converts evidence-backed claims into
structured graph nodes and edges without requiring an external graph database.

---

## What it is

A local-first system for translating reviewed claims into a queryable graph
of entities and relationships.  Nodes represent named research entities
(features, assets, models, data sources, datasets, domains).  Edges represent
relationships that are directly backed by a specific claim and its evidence.

---

## What it is NOT

- **Not a full graph platform.**  No graph crawling, no path-finding, no
  inference, no LLM integration.
- **Not auto-enrichment.**  No claim becomes a graph edge without explicit
  eligibility (supported status by default, proposed only behind a flag).
- **Not a Neo4j replacement.**  The local store is a JSON file.  It is
  designed to be a staging layer that can later feed a real graph DB.
- **Not ground truth.**  Nodes and edges are derived from claims.  Claims may
  be wrong.

---

## Core concepts

### GraphNode

A named entity, keyed by a stable namespaced ID.

| Namespace | Node type | Example |
|---|---|---|
| `source:<id>` | SOURCE | `source:FRED_GDP` |
| `feature:<name>` | FEATURE | `feature:rsi_14` |
| `model:<desc>` | MODEL | `model:mlp_on_AAPL` |
| `asset:<ticker>` | ASSET | `asset:AAPL` |
| `dataset:<version>` | DATASET | `dataset:abc123` |
| `domain:<desc>` | DOMAIN | `domain:macro_regime` |

Nodes are upserted — adding the same entity ID twice updates its properties
rather than raising.

### GraphEdge

A directed relationship between two nodes, derived from a single claim.

```
source_node_id  ──[relation]──>  target_node_id
                  confidence
                  claim_id
                  evidence_ids
```

Edge status mirrors claim status:

| Claim status | Edge status |
|---|---|
| SUPPORTED | ACTIVE |
| PROPOSED | PROPOSED |
| WEAK | WEAK |
| REJECTED | REJECTED |
| ARCHIVED | ARCHIVED |

### GraphEdgeStatus

`ACTIVE`, `PROPOSED`, `WEAK`, `REJECTED`, `ARCHIVED`.

---

## Using the graph store

```python
from ml.graph import GraphNode, GraphEdge, GraphEdgeStatus, GraphStore

store = GraphStore()   # persists to data/graph/graph.json

# Add / upsert a node
node = GraphNode.from_entity_id("feature:rsi_14", {"description": "RSI 14-bar"})
store.upsert_node(node)

# Add an edge
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

# Query
nbrs = store.neighbors("feature:rsi_14")
by_claim = store.edges_by_claim("abc...")
active = store.list_edges(status=GraphEdgeStatus.ACTIVE)
print(store.summary_stats())
# {'n_nodes': 2, 'n_edges': 1, 'by_node_type': {...}, 'by_edge_status': {...}}
```

---

## What qualifies a claim for enrichment

| Claim status | Enriched by default | Enriched with `include_proposed=True` |
|---|---|---|
| SUPPORTED | YES | YES |
| PROPOSED | NO | YES |
| WEAK | NO | NO |
| REJECTED | NO | NO |
| ARCHIVED | NO | NO |

Claim types supported:
- `RELATIONSHIP`
- `SOURCE_USEFULNESS`
- `FEATURE_USEFULNESS`
- `PERFORMANCE`

`GENERAL` claims are skipped — they lack the structured subject/object
semantics needed for a meaningful edge.

---

## Enriching the graph from claims

```python
from ml.evidence.store import ClaimStore
from ml.graph.store import GraphStore
from ml.graph.enrichment import enrich_graph_from_claims

claim_store = ClaimStore()
graph_store = GraphStore()

# Only SUPPORTED claims (default)
report = enrich_graph_from_claims(claim_store, graph_store)

# Also include PROPOSED claims
report = enrich_graph_from_claims(
    claim_store, graph_store, include_proposed=True
)

# Only specific claims
report = enrich_graph_from_claims(
    claim_store, graph_store, claim_ids=["abc...", "def..."]
)

print(report.summary())
# EnrichmentReport
#   nodes created : 6
#   nodes updated : 2
#   edges created : 4
#   claims skipped: 1
#   skip details:
#     abc123def456... — status=proposed — pass include_proposed=True to include
```

The report's `to_dict()` method produces a JSON-serializable summary suitable
for logging to the evolution log.

---

## Converting claims manually

```python
from ml.graph.converter import nodes_from_claim, edge_from_claim
from ml.evidence.schema import Claim, ClaimType, ClaimStatus

# Derive the two endpoint nodes from a claim
subject_node, object_node = nodes_from_claim(claim)
store.upsert_node(subject_node)
store.upsert_node(object_node)

# Derive the directed edge
edge = edge_from_claim(claim)
store.add_edge(edge)
```

---

## Persistence

State is stored in `data/graph/graph.json` (override with
`FINBRAIN_GRAPH_PATH` environment variable).  The format is a plain JSON file
with `version`, `updated_at`, `nodes`, and `edges` arrays.

The file is **not** ignored by git.  Commit it alongside
`data/evidence/claims.json` to track the graph state alongside the experiment
registry.

---

## How this supports future graph DB integration safely

The local graph layer is designed as a staging area:

1. **Claims are reviewed before enrichment.**  Nothing reaches the graph from
   a weak or rejected claim.
2. **Every edge carries its `claim_id` and `evidence_ids`.**  Any future
   migration to Neo4j or another graph DB can preserve the full audit trail.
3. **Node IDs follow the evidence naming conventions.**  They are stable and
   can be used as external IDs in a graph DB without remapping.
4. **The `GraphStore.summary_stats()` method** makes it easy to report on the
   graph contents programmatically for system-health monitoring.

The local store does not call external services, mutate model weights, or
trigger downstream actions.  It is a structured representation, not an agent.
