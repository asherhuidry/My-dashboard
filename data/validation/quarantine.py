"""Quarantine store for datasets that fail validation.

Failed ingests are written here with their ValidationReport so they can
be inspected, fixed, and re-processed rather than silently dropped.

Storage is local-first: each quarantine entry is a JSON file in
``data/quarantine/`` (configurable via FINBRAIN_QUARANTINE_PATH).
The index is a single ``index.json`` file for fast listing.

Usage::

    from data.validation.quarantine import QuarantineStore
    from data.validation.validator import validate_ohlcv

    qs = QuarantineStore()
    report = validate_ohlcv(df, "yfinance", "AAPL")
    if not report.passed:
        qs.save(report, df)          # writes report + data to quarantine dir
        # later...
        entries = qs.list(source_id="yfinance")
        qs.resolve(entry.entry_id)   # mark as resolved after fix
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.validation.validator import ValidationReport

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent.parent / "data" / "quarantine"
QUARANTINE_DIR = Path(os.getenv("FINBRAIN_QUARANTINE_PATH", str(_DEFAULT_DIR)))
_INDEX_FILE    = QUARANTINE_DIR / "index.json"


@dataclass
class QuarantineEntry:
    """Metadata for one quarantined dataset.

    Attributes:
        entry_id:       Unique UUID for this quarantine entry.
        source_id:      Registry ID of the data source.
        dataset_key:    Identifier for the dataset (e.g. symbol + date).
        reason:         Short human-readable reason for quarantine.
        error_count:    Number of ERROR-severity check failures.
        warning_count:  Number of WARNING-severity check failures.
        row_count:      Number of rows in the quarantined dataset.
        quarantined_at: ISO-8601 timestamp.
        resolved_at:    ISO-8601 timestamp if resolved, else None.
        report_path:    Path to the JSON validation report.
        data_path:      Path to the CSV/Parquet snapshot of the bad data.
    """
    entry_id:       str
    source_id:      str
    dataset_key:    str
    reason:         str
    error_count:    int
    warning_count:  int
    row_count:      int
    quarantined_at: str
    resolved_at:    str | None        = None
    report_path:    str               = ""
    data_path:      str               = ""

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuarantineEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class QuarantineStore:
    """Local-first quarantine store backed by a directory of JSON + CSV files.

    Args:
        directory: Root directory for quarantine files.
                   Defaults to ``QUARANTINE_DIR``.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir   = Path(directory) if directory else QUARANTINE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: list[QuarantineEntry] = self._load_index()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_index(self) -> list[QuarantineEntry]:
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                return [QuarantineEntry.from_dict(e) for e in raw.get("entries", [])]
            except Exception as exc:
                log.warning("Could not load quarantine index: %s", exc)
        return []

    def _save_index(self) -> None:
        payload = {
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "entries": [e.to_dict() for e in self._index],
        }
        self._index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def save(
        self,
        report: ValidationReport,
        df:     pd.DataFrame | None = None,
        reason: str = "",
    ) -> QuarantineEntry:
        """Quarantine a failed dataset.

        Writes the validation report as JSON and optionally the DataFrame
        as a CSV snapshot, then updates the index.

        Args:
            report: ValidationReport from the validation layer.
            df:     The problematic DataFrame (optional but recommended).
            reason: Short human-readable reason override.  If empty,
                    auto-generated from the report's error list.

        Returns:
            The QuarantineEntry written to the index.
        """
        entry_id = str(uuid.uuid4())[:8]
        now      = datetime.now(tz=timezone.utc).isoformat()

        if not reason:
            errs   = [c.message for c in report.errors]
            reason = "; ".join(errs[:3]) if errs else "validation failed"

        # Write validation report JSON
        report_dir  = self._dir / entry_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Write data snapshot
        data_path = ""
        if df is not None and not df.empty:
            data_path = str(report_dir / "data.csv")
            df.to_csv(data_path, index=True)

        entry = QuarantineEntry(
            entry_id       = entry_id,
            source_id      = report.source_id,
            dataset_key    = report.dataset_key,
            reason         = reason,
            error_count    = len(report.errors),
            warning_count  = len(report.warnings),
            row_count      = report.row_count,
            quarantined_at = now,
            report_path    = str(report_path),
            data_path      = data_path,
        )
        self._index.append(entry)
        self._save_index()

        log.warning(
            "Quarantined %s/%s (errors=%d, warnings=%d): %s",
            report.source_id, report.dataset_key,
            entry.error_count, entry.warning_count, reason,
        )
        return entry

    def resolve(self, entry_id: str, notes: str = "") -> QuarantineEntry:
        """Mark a quarantine entry as resolved.

        Args:
            entry_id: The entry's UUID prefix.
            notes:    Optional resolution notes.

        Returns:
            Updated QuarantineEntry.

        Raises:
            KeyError: If entry_id is not found.
        """
        entry = self._get(entry_id)
        entry.resolved_at = datetime.now(tz=timezone.utc).isoformat()
        if notes:
            # Append notes to the report file if it exists
            report_path = Path(entry.report_path)
            if report_path.exists():
                try:
                    d = json.loads(report_path.read_text())
                    d["resolution_notes"] = notes
                    report_path.write_text(json.dumps(d, indent=2))
                except Exception:
                    pass
        self._save_index()
        log.info("Quarantine entry %s resolved", entry_id)
        return entry

    def load_report(self, entry_id: str) -> dict[str, Any]:
        """Load the full validation report for a quarantine entry.

        Args:
            entry_id: Entry UUID prefix.

        Returns:
            Parsed report dict.
        """
        entry       = self._get(entry_id)
        report_path = Path(entry.report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        return json.loads(report_path.read_text())

    def load_data(self, entry_id: str) -> pd.DataFrame | None:
        """Load the quarantined DataFrame for a quarantine entry.

        Args:
            entry_id: Entry UUID prefix.

        Returns:
            DataFrame or None if no data was saved.
        """
        entry = self._get(entry_id)
        if not entry.data_path:
            return None
        data_path = Path(entry.data_path)
        if not data_path.exists():
            return None
        return pd.read_csv(data_path, index_col=0)

    # ── Query ─────────────────────────────────────────────────────────────────

    def list(
        self,
        source_id:     str | None  = None,
        resolved:      bool | None = None,
        dataset_key:   str | None  = None,
    ) -> list[QuarantineEntry]:
        """Filter quarantine entries.

        Args:
            source_id:   Filter by source registry ID.
            resolved:    True = resolved only; False = unresolved only; None = all.
            dataset_key: Filter by dataset key.

        Returns:
            Matching entries, most recent first.
        """
        results = list(self._index)
        if source_id is not None:
            results = [e for e in results if e.source_id == source_id]
        if resolved is not None:
            results = [e for e in results if e.is_resolved == resolved]
        if dataset_key is not None:
            results = [e for e in results if e.dataset_key == dataset_key]
        return sorted(results, key=lambda e: e.quarantined_at, reverse=True)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the quarantine store."""
        total      = len(self._index)
        unresolved = sum(1 for e in self._index if not e.is_resolved)
        by_source: dict[str, int] = {}
        for e in self._index:
            by_source[e.source_id] = by_source.get(e.source_id, 0) + 1
        return {
            "total":      total,
            "unresolved": unresolved,
            "resolved":   total - unresolved,
            "by_source":  by_source,
        }

    def _get(self, entry_id: str) -> QuarantineEntry:
        for e in self._index:
            if e.entry_id == entry_id:
                return e
        raise KeyError(f"Quarantine entry not found: {entry_id!r}")

    def __len__(self) -> int:
        return len(self._index)
