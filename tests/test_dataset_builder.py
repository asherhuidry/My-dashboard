"""Tests for ml.data.dataset_builder — dataset assembly and versioning."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.dataset_builder import (
    DatasetMeta,
    _compute_version,
    assemble_dataset,
    assemble_sequence_dataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame large enough for all split sizes."""
    np.random.seed(42)
    end   = datetime.now(tz=timezone.utc)
    idx   = pd.bdate_range(end=end, periods=n, tz="UTC")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high":   close * (1 + np.random.uniform(0.001, 0.015, n)),
            "low":    close * (1 - np.random.uniform(0.001, 0.015, n)),
            "close":  close,
            "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv(n=300)


# ── DatasetMeta serialisation ─────────────────────────────────────────────────

class TestDatasetMeta:
    def _make_meta(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_version   = "abc123def456",
            symbol            = "AAPL",
            feature_cols      = ["feat_a", "feat_b"],
            target_definition = "binary_direction_1bar",
            n_rows            = 250,
            n_train           = 175,
            n_val             = 37,
            n_test            = 38,
            time_range_start  = "2022-01-03T00:00:00+00:00",
            time_range_end    = "2024-12-31T00:00:00+00:00",
            train_frac        = 0.70,
            val_frac          = 0.15,
            target_horizon    = 1,
            seq_len           = None,
            generated_at      = "2025-01-01T00:00:00+00:00",
            notes             = "test",
        )

    def test_to_dict_has_all_fields(self):
        d = self._make_meta().to_dict()
        for field in (
            "dataset_version", "symbol", "feature_cols", "target_definition",
            "n_rows", "n_train", "n_val", "n_test", "time_range_start",
            "time_range_end", "train_frac", "val_frac", "target_horizon",
            "seq_len", "generated_at", "notes",
        ):
            assert field in d

    def test_round_trip(self):
        meta = self._make_meta()
        assert DatasetMeta.from_dict(meta.to_dict()) == meta

    def test_from_dict_ignores_unknown_keys(self):
        d = self._make_meta().to_dict()
        d["unexpected_field"] = "value"
        # Should not raise
        meta = DatasetMeta.from_dict(d)
        assert meta.symbol == "AAPL"

    def test_to_dataset_info_keys(self):
        info = self._make_meta().to_dataset_info()
        for key in ("dataset_version", "symbol", "n_rows", "n_train", "n_val", "n_test",
                    "time_range_start", "time_range_end", "target_horizon",
                    "target_definition"):
            assert key in info

    def test_to_dataset_info_feature_cols_truncated(self):
        meta = self._make_meta()
        meta.feature_cols = [f"f{i}" for i in range(30)]
        info = meta.to_dataset_info()
        assert len(info["feature_cols"]) <= 10


# ── Version hashing ───────────────────────────────────────────────────────────

class TestComputeVersion:
    def _version(self, **kwargs) -> str:
        defaults = dict(
            symbol="AAPL", feature_cols=["a", "b"], target_horizon=1,
            n_rows=200, time_range_start="2022-01-01T00:00:00",
            time_range_end="2024-12-31T00:00:00",
            train_frac=0.70, val_frac=0.15, seq_len=None,
        )
        defaults.update(kwargs)
        return _compute_version(**defaults)

    def test_returns_12_char_hex(self):
        v = self._version()
        assert len(v) == 12
        int(v, 16)  # must be valid hex

    def test_same_params_same_version(self):
        assert self._version() == self._version()

    def test_different_symbol_different_version(self):
        assert self._version(symbol="AAPL") != self._version(symbol="MSFT")

    def test_case_insensitive_symbol(self):
        assert self._version(symbol="aapl") == self._version(symbol="AAPL")

    def test_different_horizon_different_version(self):
        assert self._version(target_horizon=1) != self._version(target_horizon=5)

    def test_feature_order_does_not_matter(self):
        assert self._version(feature_cols=["a", "b"]) == self._version(feature_cols=["b", "a"])

    def test_seq_len_affects_version(self):
        assert self._version(seq_len=None) != self._version(seq_len=20)

    def test_different_n_rows_different_version(self):
        assert self._version(n_rows=200) != self._version(n_rows=201)


# ── assemble_dataset ──────────────────────────────────────────────────────────

class TestAssembleDataset:
    def test_returns_tuple_of_data_and_meta(self, ohlcv):
        result = assemble_dataset(ohlcv, symbol="AAPL")
        assert isinstance(result, tuple) and len(result) == 2
        data, meta = result
        assert isinstance(data, dict)
        assert isinstance(meta, DatasetMeta)

    def test_data_has_required_keys(self, ohlcv):
        data, _ = assemble_dataset(ohlcv, symbol="AAPL")
        for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                    "close_test", "signal_index", "feature_cols", "n_train",
                    "n_val", "n_test"):
            assert key in data

    def test_split_sizes_consistent(self, ohlcv):
        data, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert len(data["X_train"]) == data["n_train"] == meta.n_train
        assert len(data["X_val"])   == data["n_val"]   == meta.n_val
        assert len(data["X_test"])  == data["n_test"]  == meta.n_test

    def test_splits_sum_to_total(self, ohlcv):
        data, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert data["n_train"] + data["n_val"] + data["n_test"] == meta.n_rows

    def test_x_y_shapes_match(self, ohlcv):
        data, _ = assemble_dataset(ohlcv, symbol="AAPL")
        for split in ("train", "val", "test"):
            assert len(data[f"X_{split}"]) == len(data[f"y_{split}"])

    def test_y_is_binary(self, ohlcv):
        data, _ = assemble_dataset(ohlcv, symbol="AAPL")
        y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
        assert set(y_all).issubset({0.0, 1.0})

    def test_no_inf_in_features(self, ohlcv):
        data, _ = assemble_dataset(ohlcv, symbol="AAPL")
        for split in ("train", "val", "test"):
            X = data[f"X_{split}"]
            assert np.all(np.isfinite(X)), f"Inf/NaN in X_{split}"

    def test_close_test_aligned_to_x_test(self, ohlcv):
        data, _ = assemble_dataset(ohlcv, symbol="AAPL")
        assert len(data["close_test"]) == len(data["X_test"])

    def test_meta_version_is_12_chars(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert len(meta.dataset_version) == 12

    def test_meta_symbol_uppercased(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="aapl")
        assert meta.symbol == "AAPL"

    def test_meta_target_definition_contains_horizon(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL", target_horizon=5)
        assert "5bar" in meta.target_definition

    def test_meta_seq_len_is_none_for_flat(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert meta.seq_len is None

    def test_meta_time_range_populated(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert meta.time_range_start != ""
        assert meta.time_range_end   != ""

    def test_same_data_same_version(self, ohlcv):
        _, meta1 = assemble_dataset(ohlcv, symbol="AAPL")
        _, meta2 = assemble_dataset(ohlcv, symbol="AAPL")
        assert meta1.dataset_version == meta2.dataset_version

    def test_different_horizon_different_version(self, ohlcv):
        _, meta1 = assemble_dataset(ohlcv, symbol="AAPL", target_horizon=1)
        _, meta2 = assemble_dataset(ohlcv, symbol="AAPL", target_horizon=3)
        assert meta1.dataset_version != meta2.dataset_version

    def test_notes_stored_in_meta(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL", notes="test run")
        assert meta.notes == "test run"

    def test_to_dataset_info_has_version(self, ohlcv):
        _, meta = assemble_dataset(ohlcv, symbol="AAPL")
        info = meta.to_dataset_info()
        assert info["dataset_version"] == meta.dataset_version

    def test_feature_cols_non_empty(self, ohlcv):
        data, meta = assemble_dataset(ohlcv, symbol="AAPL")
        assert len(data["feature_cols"]) > 0
        assert len(meta.feature_cols) > 0


# ── assemble_sequence_dataset ─────────────────────────────────────────────────

class TestAssembleSequenceDataset:
    def test_returns_five_tuple(self, ohlcv):
        result = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=5)
        assert isinstance(result, tuple) and len(result) == 5

    def test_datasets_have_len(self, ohlcv):
        train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(
            ohlcv, symbol="AAPL", seq_len=5
        )
        assert len(train_ds) > 0
        assert len(val_ds)   > 0
        assert len(test_ds)  > 0

    def test_all_splits_non_overlapping_sizes(self, ohlcv):
        train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(
            ohlcv, symbol="AAPL", seq_len=5
        )
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert meta.n_train == len(train_ds)
        assert meta.n_val   == len(val_ds)
        assert meta.n_test  == len(test_ds)

    def test_close_test_aligned_to_test_ds(self, ohlcv):
        train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(
            ohlcv, symbol="AAPL", seq_len=5
        )
        assert len(close_test) == len(test_ds)

    def test_meta_seq_len_set(self, ohlcv):
        _, _, _, _, meta = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=10)
        assert meta.seq_len == 10

    def test_meta_target_definition_contains_seq(self, ohlcv):
        _, _, _, _, meta = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=5)
        assert "seq5" in meta.target_definition

    def test_sequence_version_differs_from_flat_version(self, ohlcv):
        _, flat_meta  = assemble_dataset(ohlcv, symbol="AAPL")
        _, _, _, _, seq_meta = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=5)
        assert flat_meta.dataset_version != seq_meta.dataset_version

    def test_train_ds_n_features(self, ohlcv):
        train_ds, _, _, _, _ = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=5)
        assert train_ds.n_features > 0

    def test_close_test_is_series(self, ohlcv):
        _, _, _, close_test, _ = assemble_sequence_dataset(ohlcv, symbol="AAPL", seq_len=5)
        import pandas as pd
        assert isinstance(close_test, pd.Series)
