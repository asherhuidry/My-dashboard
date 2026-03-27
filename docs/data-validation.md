# Data Validation and Quarantine Layer

**Module:** `data/validation/`
**Persistence:** `data/quarantine/` (local-first, configurable via env var)

---

## Purpose

Every DataFrame that enters the ingest pipeline passes through the validation layer before any write to a live store. Datasets that fail ERROR-severity checks are routed to the quarantine store for inspection and reprocessing rather than being silently dropped or stored corrupt.

---

## Quick start

```python
from data.validation import validate_ohlcv, QuarantineStore

qs = QuarantineStore()

df = fetch_prices("AAPL")
report = validate_ohlcv(df, source_id="yfinance", symbol="AAPL")

if not report.passed:
    qs.save(report, df)
    return          # do not write to live store

db.bulk_insert_prices(df)
```

For generic time-series data (non-OHLCV):

```python
from data.validation import validate_timeseries

report = validate_timeseries(
    df,
    source_id    = "fred_api",
    dataset_key  = "DGS10",
    required_cols= ["date", "value"],
    time_col     = "date",
    min_rows     = 5,
    max_null_pct = 0.02,
)
```

---

## Check suite

### Generic time-series checks (`validate_timeseries`)

| Check | Severity | Description |
|-------|----------|-------------|
| `required_columns` | ERROR | All specified columns are present |
| `row_count` | ERROR | At least `min_rows` rows (default 10) |
| `null_rate` | ERROR | No key column exceeds `max_null_pct` null rate (default 5%) |
| `duplicate_rows` | WARNING | No fully duplicate rows |
| `duplicate_timestamps` | ERROR | No duplicate values in `time_col` |
| `stale_data` | WARNING | Latest row within `max_stale_days` calendar days (default 7) |
| `monotonic_timestamps` | WARNING | `time_col` is monotonically increasing |
| `schema_drift` | WARNING | Column set matches `expected_cols` if provided |

### OHLCV-specific checks (`validate_ohlcv`)

Runs all generic checks (stricter defaults: `max_null_pct=0.01`, `max_stale_days=5`, `min_rows=20`) plus:

| Check | Severity | Description |
|-------|----------|-------------|
| `price_sanity` | ERROR | `high >= low`, `close` within `[low, high]`, `volume >= 0` |
| `numeric_range_open` | ERROR | All open prices > 0 |
| `numeric_range_high` | ERROR | All high prices > 0 |
| `numeric_range_low` | ERROR | All low prices > 0 |
| `numeric_range_close` | ERROR | All close prices > 0 |
| `numeric_range_volume` | ERROR | Volume >= 0 |

---

## Severity levels

| Severity | Effect |
|----------|--------|
| `ERROR` | Failure sets `report.passed = False`; dataset must be quarantined |
| `WARNING` | Logged but does not block ingestion |
| `INFO` | Informational only; emitted for skipped checks (missing columns) |

---

## ValidationReport

```python
report.passed        # bool — False if any ERROR check failed
report.errors        # list[CheckResult] — ERROR failures
report.warnings      # list[CheckResult] — WARNING failures
report.row_count     # int
report.schema_hash   # str — hash of sorted column list
report.summary_line()# str — one-line human-readable summary
report.to_dict()     # dict — JSON-serializable
```

---

## Quarantine store

### Saving

```python
qs = QuarantineStore()                         # default: data/quarantine/
qs = QuarantineStore(directory="/custom/path") # or custom

entry = qs.save(report, df)       # writes report.json + data.csv
entry.entry_id                    # short UUID (8 chars)
entry.quarantined_at              # ISO-8601 timestamp
```

### Listing

```python
qs.list()                          # all entries, most recent first
qs.list(source_id="yfinance")      # filter by source
qs.list(resolved=False)            # unresolved only
qs.list(dataset_key="AAPL")        # filter by dataset key
```

### Inspecting

```python
report_dict = qs.load_report(entry_id)   # full validation report dict
df          = qs.load_data(entry_id)     # quarantined DataFrame (or None)
```

### Resolving

```python
qs.resolve(entry_id, notes="Fixed upstream gap in yfinance data")
```

### Summary

```python
qs.summary()
# {
#   "total": 12,
#   "unresolved": 3,
#   "resolved": 9,
#   "by_source": {"yfinance": 8, "fred_api": 4}
# }
```

---

## Directory structure

```
data/quarantine/
  index.json          ← master index of all entries
  <entry_id>/
    report.json       ← serialized ValidationReport
    data.csv          ← DataFrame snapshot (if provided)
```

---

## Environment variable

Override the default quarantine directory:

```bash
export FINBRAIN_QUARANTINE_PATH=/path/to/custom/quarantine
```

---

## Integration pattern

```python
# In any ingest connector:
from data.validation import validate_ohlcv, QuarantineStore

_qs = QuarantineStore()  # module-level singleton

def ingest_symbol(symbol: str, source_id: str) -> bool:
    df = fetch(symbol)
    report = validate_ohlcv(df, source_id=source_id, symbol=symbol)

    if not report.passed:
        _qs.save(report, df)
        log.error("Quarantined %s/%s — %d errors", source_id, symbol, len(report.errors))
        return False

    db.write(df)
    return True
```
