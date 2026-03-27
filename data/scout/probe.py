"""Lightweight HTTP probe for source candidate URLs.

Performs a single conservative reachability check against a candidate URL.
The probe is intentionally minimal: it does not parse page content, does not
follow authentication flows, and does not download data.  Its sole purpose is
to answer: "Is this URL alive, and what does the server say about itself?"

Probe strategy
--------------
1. Try HEAD first (no body transferred).
2. If the server returns 405 (Method Not Allowed) or 501 (Not Implemented),
   retry with a minimal GET (max 1 KB read, connection closed immediately).
3. Any other error or timeout → unreachable result.

Redirects are followed automatically up to a fixed limit (5).  The final URL
after redirects is recorded.

Nothing beyond response headers and a tiny body peek is ever read.

Usage::

    from data.scout.schema import normalize_source_candidate
    from data.scout.probe import probe_source_url

    candidate = normalize_source_candidate({
        "name": "FRED API", "url": "https://api.stlouisfed.org/fred",
    })
    result = probe_source_url(candidate)
    print(result.ok)          # True / False
    print(result.http_status) # e.g. 200
    print(result.latency_ms)  # e.g. 142
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
import urllib.request
import urllib.error

if TYPE_CHECKING:
    from data.scout.schema import SourceCandidate

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT   = 10          # seconds
_MAX_REDIRECTS     = 5
_PEEK_BYTES        = 1024        # max bytes read on GET fallback
_USER_AGENT        = (
    "FinBrain-SourceProbe/1.0 "
    "(reachability check; single request; no data collection)"
)

# HTTP status codes that suggest HEAD is not supported → retry with GET
_HEAD_UNSUPPORTED  = {405, 501}

# Status ranges considered "reachable" (server responded meaningfully)
_OK_STATUSES       = set(range(200, 400))   # 2xx + 3xx
_AUTH_STATUSES     = {401, 403}             # reachable but needs auth
_REACHABLE_STATUSES = _OK_STATUSES | _AUTH_STATUSES | {404, 429}


# ── ProbeResult ───────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Outcome of a single lightweight HTTP probe.

    Attributes:
        url:          The URL that was probed (before redirects).
        ok:           True if the server gave a meaningful response.
        http_status:  HTTP status code, or None on connection failure.
        method_used:  "HEAD" or "GET".
        latency_ms:   Round-trip time in milliseconds.
        content_type: Value of the Content-Type header, if present.
        final_url:    URL after following redirects (may equal url).
        error:        Error message if the probe failed, else "".
        probed_at:    ISO-8601 UTC timestamp of when the probe ran.
        notes:        Human-readable interpretation of the result.
    """
    url:          str
    ok:           bool
    http_status:  int | None
    method_used:  str
    latency_ms:   float
    content_type: str        = ""
    final_url:    str        = ""
    error:        str        = ""
    probed_at:    str        = ""
    notes:        str        = ""

    def __post_init__(self) -> None:
        if not self.probed_at:
            self.probed_at = datetime.now(tz=timezone.utc).isoformat()
        if not self.final_url:
            self.final_url = self.url

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "url":          self.url,
            "ok":           self.ok,
            "http_status":  self.http_status,
            "method_used":  self.method_used,
            "latency_ms":   round(self.latency_ms, 1),
            "content_type": self.content_type,
            "final_url":    self.final_url,
            "error":        self.error,
            "probed_at":    self.probed_at,
            "notes":        self.notes,
        }

    @property
    def redirected(self) -> bool:
        """True if the final URL differs from the probed URL."""
        return self.final_url.rstrip("/") != self.url.rstrip("/")

    @property
    def auth_required(self) -> bool:
        """True if the server indicated authentication is needed."""
        return self.http_status in _AUTH_STATUSES


# ── probe_source_url ──────────────────────────────────────────────────────────

def probe_source_url(
    candidate: "SourceCandidate",
    timeout:   int = _DEFAULT_TIMEOUT,
) -> ProbeResult:
    """Perform a lightweight reachability probe on a SourceCandidate's URL.

    Uses HEAD first; falls back to a minimal GET if HEAD is not supported.
    The candidate's ``acquisition_method`` influences probe behaviour:

    - ``sdk``           → returns a synthetic ok=None result (cannot HTTP-probe SDKs)
    - ``file_download`` → probes with HEAD/GET as normal (direct URL)
    - ``api``           → probes with HEAD/GET as normal
    - ``feed``          → probes with HEAD/GET as normal
    - ``scrape``        → probes with HEAD/GET as normal

    Args:
        candidate: A normalized SourceCandidate.
        timeout:   Request timeout in seconds (default: 10).

    Returns:
        A ProbeResult describing the outcome.
    """
    if candidate.acquisition_method == "sdk":
        return ProbeResult(
            url         = candidate.url,
            ok          = True,
            http_status = None,
            method_used = "NONE",
            latency_ms  = 0.0,
            notes       = (
                "Acquisition method is 'sdk' — HTTP probe not applicable. "
                "Marked ok=True; verify by importing the SDK package."
            ),
        )

    return _do_probe(candidate.url, timeout)


def probe_url(url: str, timeout: int = _DEFAULT_TIMEOUT) -> ProbeResult:
    """Probe a raw URL directly (without a SourceCandidate wrapper).

    Useful for quick ad-hoc checks or testing.

    Args:
        url:     The URL to probe.
        timeout: Request timeout in seconds.

    Returns:
        A ProbeResult.
    """
    return _do_probe(url, timeout)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _do_probe(url: str, timeout: int) -> ProbeResult:
    """Execute the HEAD → GET probe sequence."""
    result = _attempt(url, "HEAD", timeout)
    if result.http_status in _HEAD_UNSUPPORTED:
        result = _attempt(url, "GET", timeout)
    return result


def _attempt(url: str, method: str, timeout: int) -> ProbeResult:
    """Perform a single HTTP request and return a ProbeResult."""
    start = time.monotonic()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT},
            method=method,
        )
        # urllib follows redirects automatically for GET; for HEAD we must
        # handle manually because Python's default opener doesn't follow HEAD
        # redirects.  We use a custom opener that caps redirects.
        opener = _build_opener(_MAX_REDIRECTS)
        with opener.open(req, timeout=timeout) as resp:
            latency_ms  = (time.monotonic() - start) * 1000
            status      = resp.status
            content_type = resp.headers.get("Content-Type", "")
            final_url   = resp.url if hasattr(resp, "url") else url

            if method == "GET":
                # Read a tiny peek to satisfy servers that require a body read
                try:
                    resp.read(_PEEK_BYTES)
                except Exception:
                    pass

        ok    = status in _REACHABLE_STATUSES
        notes = _interpret_status(status, method)
        return ProbeResult(
            url          = url,
            ok           = ok,
            http_status  = status,
            method_used  = method,
            latency_ms   = latency_ms,
            content_type = content_type,
            final_url    = final_url,
            notes        = notes,
        )

    except urllib.error.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        status     = exc.code
        ok         = status in _REACHABLE_STATUSES
        notes      = _interpret_status(status, method)
        return ProbeResult(
            url         = url,
            ok          = ok,
            http_status = status,
            method_used = method,
            latency_ms  = latency_ms,
            error       = f"HTTPError {status}: {exc.reason}",
            notes       = notes,
        )

    except urllib.error.URLError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        reason     = str(exc.reason)
        return ProbeResult(
            url         = url,
            ok          = False,
            http_status = None,
            method_used = method,
            latency_ms  = latency_ms,
            error       = f"URLError: {reason}",
            notes       = _interpret_url_error(reason),
        )

    except TimeoutError:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url         = url,
            ok          = False,
            http_status = None,
            method_used = method,
            latency_ms  = latency_ms,
            error       = f"Timeout after {timeout}s",
            notes       = "Request timed out. Source may be slow, rate-limiting, or unreachable.",
        )

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url         = url,
            ok          = False,
            http_status = None,
            method_used = method,
            latency_ms  = latency_ms,
            error       = f"{type(exc).__name__}: {exc}",
            notes       = "Unexpected error during probe. Check URL format.",
        )


def _build_opener(max_redirects: int) -> urllib.request.OpenerDirector:
    """Build a URL opener with a capped redirect count."""
    class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
        max_repeats = max_redirects
        max_redirections = max_redirects

    return urllib.request.build_opener(_LimitedRedirectHandler)


def _interpret_status(status: int, method: str) -> str:
    """Return a human-readable interpretation of an HTTP status."""
    if 200 <= status < 300:
        return f"Server responded {status} via {method}. URL is reachable."
    if 300 <= status < 400:
        return f"Redirect ({status}) — final URL may differ. Counted as reachable."
    if status == 401:
        return "401 Unauthorized — URL is live but requires authentication."
    if status == 403:
        return "403 Forbidden — URL is live but access is restricted (may still be usable with auth)."
    if status == 404:
        return "404 Not Found — URL path does not exist; verify documentation URL."
    if status == 405 and method == "HEAD":
        return "405 Method Not Allowed for HEAD — will retry with GET."
    if status == 429:
        return "429 Too Many Requests — server is reachable but rate-limiting this probe."
    if status >= 500:
        return f"Server error {status} — source may be temporarily unavailable."
    return f"HTTP {status} via {method}."


def _interpret_url_error(reason: str) -> str:
    """Return a human-readable note for a URLError reason."""
    r = reason.lower()
    if "name or service not known" in r or "nodename nor servname" in r:
        return "DNS resolution failed. URL host may be incorrect or offline."
    if "connection refused" in r:
        return "Connection refused. Server is not accepting connections on this port."
    if "timed out" in r or "timeout" in r:
        return "Connection timed out. Source may be slow or unreachable."
    if "ssl" in r or "certificate" in r:
        return "SSL/TLS error. Certificate may be invalid or expired."
    return f"Network error: {reason}"
