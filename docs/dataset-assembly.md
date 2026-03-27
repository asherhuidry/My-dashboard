# Dataset Assembly and Versioning

`ml/data/dataset_builder.py` provides a reproducible way to assemble supervised
learning datasets from OHLCV price history and attach a stable version hash to
every assembly run.

## Why versioning matters

Two experiments are only directly comparable if they used the same features,
the same target definition, and the same chronological split.  Without
versioning this is impossible to verify after the fact.  The `DatasetMeta`
object gives every dataset assembly a 12-character SHA-256 hash computed from
the parameters that define it.  Two assemblies sharing that hash used identical
inputs.

## Flat dataset (MLP / logistic baseline)

```python
from ml.data.dataset_builder import assemble_dataset

data, meta = assemble_dataset(df, symbol="AAPL", target_horizon=1)
```

`data` is a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `X_train` | `np.ndarray float32` | Training feature matrix |
| `y_train` | `np.ndarray float32` | Binary training targets (0/1) |
| `X_val` | `np.ndarray float32` | Validation features |
| `y_val` | `np.ndarray float32` | Validation targets |
| `X_test` | `np.ndarray float32` | Test features (held-out) |
| `y_test` | `np.ndarray float32` | Test targets |
| `close_test` | `pd.Series` | Close prices aligned to test rows (for backtest) |
| `signal_index` | `DatetimeIndex` | Dates for test rows |
| `feature_cols` | `list[str]` | Ordered feature column names |
| `n_train` / `n_val` / `n_test` | `int` | Split sizes |

**Default split:** 70 / 15 / 15 (chronological, no shuffling).

## Sequence dataset (LSTM)

```python
from ml.data.dataset_builder import assemble_sequence_dataset

train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(
    df, symbol="AAPL", seq_len=20
)
```

Returns three `PriceSequenceDataset` subsets (compatible with PyTorch
`DataLoader`) plus a `pd.Series` of close prices aligned to the test sequences
for backtesting.

## DatasetMeta

```python
meta.dataset_version   # "3f9b1a2c7e4d"  — stable 12-char hash
meta.symbol            # "AAPL"
meta.n_rows            # 248
meta.feature_cols      # ["rsi_14", "macd", ...]
meta.target_definition # "binary_direction_1bar"
meta.time_range_start  # ISO-8601
meta.time_range_end    # ISO-8601
meta.seq_len           # None for flat, 20 for sequence
meta.generated_at      # ISO-8601

meta.to_dict()         # JSON-serialisable
meta.to_dataset_info() # subset for ExperimentRegistry.dataset_info
```

## Target definition

`y = 1` if `close[t + horizon] > close[t]`, else `0`.

Default horizon = 1 bar (next trading day).  The split is always applied
after feature NaN removal — the first ~30–60 rows are discarded as
rolling-window warmup.

## Feature engineering

Features come from `ml.patterns.features.build_features()` which produces
~80 columns covering:

- Price returns (1d / 5d / 10d / 21d / 63d)
- Volatility (realized, Garman-Klass, ATR)
- Momentum (RSI, MACD, Stochastic, Williams %R)
- Trend (Bollinger Bands, ADX, EMA ribbon, Ichimoku)
- Volume (OBV, VWAP, Chaikin Money Flow)
- Regime (rolling Sharpe, max drawdown)
- Calendar (day-of-week, month, FOMC window)

All infinite values are replaced with 0.0 before returning.  Callers are
expected to apply Z-score normalisation (via `FeatureScaler`) before feeding
features to any model.

## Version hash computation

The version is a 12-char SHA-256 prefix of a canonical JSON string containing:

```json
{
  "symbol": "AAPL",
  "feature_cols": ["...sorted..."],
  "target_horizon": 1,
  "n_rows": 248,
  "time_range_start": "2022-01-03T...",
  "time_range_end": "2024-12-31T...",
  "train_frac": 0.7,
  "val_frac": 0.15,
  "seq_len": null
}
```

Changing any of these parameters produces a different version hash.  Symbol
case is normalised to uppercase.

## Storing version in experiment registry

```python
exp = registry.create(
    ...
    dataset_info = meta.to_dataset_info(),
)
```

This embeds `dataset_version` in the experiment record so comparison utilities
can group runs by dataset.
