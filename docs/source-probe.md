# Source Probe

`data/scout/probe.py` and `data/scout/probe_registry.py` provide a
lightweight HTTP reachability check for source candidates, integrated with
the source registry lifecycle.

---

## What it does

Performs a single conservative HTTP request against a candidate URL and
records a structured `ProbeResult`:

- HTTP status code
- Latency in milliseconds
- Content-Type header
- Final URL after redirects
- Whether the server is reachable
- Any error message

A reachable probe on a `DISCOVERED` source advances its status to `SAMPLED`
in the source registry.

---

## What it does NOT do

- **No content parsing.** At most 1 KB of body is read on GET fallback; the
  content is discarded.
- **No schema inference.** Probe results do not describe the API's data format.
- **No auto-approval.** A successful probe only moves a source from
  `DISCOVERED` → `SAMPLED`.  All further status advances require human review.
- **No broad crawling.** One request per candidate.
- **No authentication.** The probe does not attempt login flows; 401/403
  responses are counted as "reachable" (server is live) but flag auth
  requirements.

---

## Probe strategy

```
1. Send HEAD request
   ├── 2xx / 3xx / 401 / 403 / 404 / 429  → ok=True
   ├── 500+                                 → ok=False
   ├── 405 / 501 (HEAD not supported)       → retry with GET (step 2)
   └── Network error / timeout              → ok=False

2. Send minimal GET (fallback)
   ├── Read up to 1 KB then close
   └── Same status mapping as HEAD
```

SDK sources (`acquisition_method = "sdk"`) skip the HTTP probe entirely and
return `ok=True` with `method_used="NONE"`.  They must be verified by
attempting to import the SDK package.

---

## How probe status maps to registry lifecycle

| Probe outcome | Current registry status | Registry action |
|---|---|---|
| ok=True | DISCOVERED | → SAMPLED |
| ok=False | DISCOVERED | Notes appended; status unchanged |
| ok=True or False | SAMPLED / VALIDATED / APPROVED | Notes appended; status unchanged |
| ok=True or False | REJECTED | Notes appended; status unchanged |

---

## Usage

### Single probe

```python
from data.scout.schema import normalize_source_candidate
from data.scout.probe import probe_source_url

candidate = normalize_source_candidate({
    "name": "FRED API",
    "url":  "https://api.stlouisfed.org/fred",
})

result = probe_source_url(candidate, timeout=10)
print(result.ok)           # True
print(result.http_status)  # 200
print(result.latency_ms)   # e.g. 142.3
print(result.content_type) # e.g. "application/json"
print(result.to_dict())    # JSON-serializable summary
```

### Probe and register in one call

```python
from data.registry.source_registry import SourceRegistry
from data.scout.probe_registry import probe_and_register

registry  = SourceRegistry()
result    = probe_and_register(candidate, registry, score=0.82)

print(result.action)         # "sampled" | "updated" | "created"
print(result.record.status)  # SourceStatus.SAMPLED
print(result.probe.ok)       # True
```

### Apply a probe result to an existing registry entry

```python
from data.scout.probe_registry import apply_probe_to_registry

# If you have a ProbeResult from a previous run:
result = apply_probe_to_registry(
    source_id          = "fred_api",
    probe              = probe_result,
    registry           = registry,
    advance_to_sampled = True,   # default
)
```

### Record probe as evidence

```python
from data.scout.probe_evidence import evidence_from_probe
from ml.evidence.store import ClaimStore

ev    = evidence_from_probe(candidate, probe_result)
store = ClaimStore()
store.add_evidence(ev)
# ev.structured_data contains: ok, http_status, latency_ms, content_type,
# final_url, redirected, probed_at
```

---

## ProbeResult fields

| Field | Type | Description |
|---|---|---|
| `url` | str | URL that was probed |
| `ok` | bool | True if server gave a meaningful response |
| `http_status` | int \| None | HTTP status code, or None on connection failure |
| `method_used` | str | "HEAD", "GET", or "NONE" (SDK) |
| `latency_ms` | float | Round-trip time in milliseconds |
| `content_type` | str | Content-Type header value (may be empty) |
| `final_url` | str | URL after redirects |
| `error` | str | Error message if probe failed |
| `probed_at` | str | ISO-8601 UTC timestamp |
| `notes` | str | Human-readable interpretation |

Convenience properties: `redirected` (bool), `auth_required` (bool).

---

## Timeouts and safety

- Default timeout: **10 seconds**.
- Maximum redirects followed: **5**.
- Maximum body read on GET: **1 024 bytes**.
- User-Agent is explicitly identified as a reachability probe.

DNS failures, TLS errors, connection refusals, and timeouts all produce
`ok=False` with an informative `error` string.
