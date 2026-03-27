# Source Registry

**Module:** `data/registry/`
**Persistence:** `data/registry/sources.json` (local-first, upgradeable to DB)

---

## Purpose

Every piece of data FinBrain ingests must come from a registered, evaluated source.
The source registry provides:

1. A **single inventory** of all known data sources.
2. A **lifecycle** tracking which sources are trustworthy.
3. A **filter API** so ingest connectors can query for approved sources.

---

## Lifecycle states

```
discovered → sampled → validated → approved
                               ↘ rejected
             ↘ quarantined (was approved, now failing)
```

| Status | Meaning |
|--------|---------|
| `discovered` | Known but not yet tested |
| `sampled` | A small sample has been fetched and stored |
| `validated` | Sample passed all quality checks |
| `approved` | Actively used by the ingest pipeline |
| `rejected` | Failed validation or no longer useful |
| `quarantined` | Was approved, now producing bad data |

---

## Quick start

```python
from data.registry import SourceRegistry, SourceRecord, SourceStatus

# Load (or create) registry
reg = SourceRegistry()

# Add a source
reg.add(SourceRecord(
    source_id          = "my_api",
    name               = "My Financial API",
    category           = "price",
    url                = "https://example.com/api",
    auth_required      = True,
    free_tier          = False,
    rate_limit_notes   = "100 req/min on paid tier",
    update_frequency   = "realtime",
    asset_classes      = ["equity"],
    data_types         = ["ohlcv"],
))

# Advance through the lifecycle
reg.update_status("my_api", SourceStatus.SAMPLED)
reg.update_status("my_api", SourceStatus.VALIDATED)
reg.update_status("my_api", SourceStatus.APPROVED)

# Query
approved_macro = reg.filter(category="macro", status=SourceStatus.APPROVED)
```

---

## Seeding the registry

The module ships with 13 high-value public sources pre-defined:

```bash
python -m data.registry.seed_sources
```

Or programmatically:

```python
from data.registry.seed_sources import seed
registry = seed()
```

---

## SourceRecord fields

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Unique snake_case identifier |
| `name` | `str` | Human-readable name |
| `category` | `str` | `price`, `macro`, `fundamental`, `sentiment`, `news`, `alternative`, `options`, `volatility` |
| `url` | `str` | Base URL or documentation link |
| `acquisition_method` | `str` | `api`, `file_download`, `scrape`, `sdk` |
| `auth_required` | `bool` | Whether an API key is required |
| `free_tier` | `bool` | Whether a free tier exists |
| `rate_limit_notes` | `str` | Human-readable rate limit description |
| `update_frequency` | `str` | `realtime`, `daily`, `weekly`, `monthly`, `quarterly`, `annual` |
| `asset_classes` | `list[str]` | Asset classes covered |
| `data_types` | `list[str]` | Data types provided |
| `reliability_score` | `float` | 0–1; updated from validation history |
| `status` | `SourceStatus` | Current lifecycle status |
| `notes` | `str` | Free-text notes |
| `discovered_at` | `str` | ISO-8601 timestamp |
| `last_checked_at` | `str \| None` | ISO-8601 timestamp of last check |

---

## Environment variable

Override the default storage path:

```bash
export FINBRAIN_REGISTRY_PATH=/path/to/custom/sources.json
```

---

## Upgrading to database persistence

The registry is local-first by design. To sync to Supabase:

```python
from db.supabase.client import get_client
from data.registry import SourceRegistry

reg = SourceRegistry()
sb  = get_client()
for src in reg.all():
    sb.table("source_registry").upsert(src.to_dict()).execute()
```

A `source_registry` table migration will be added in Phase 2.
