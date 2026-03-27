# Backtest Engine

**Module:** `ml/backtest/engine.py`

---

## Purpose

A clean, vectorised backtest engine for measuring how well model signals would have performed on historical price data. Designed to be:

- **Simple** — binary long/flat positions, no leverage, no portfolio construction.
- **Transparent** — every intermediate series (equity curve, daily returns, trade log) is returned for inspection.
- **Honest** — signals are shifted by 1 bar before use, so there is no look-ahead bias from using the same bar's close for both signal and entry.
- **Composable** — outputs integrate directly with `ExperimentRegistry.attach_backtest()`.

---

## Quick start

```python
import pandas as pd
from ml.backtest.engine import run_backtest, BacktestConfig

# Prices: daily closes
prices  = pd.Series([100, 101, 102, 101, 103, 105, 104, 106],
                    index=pd.date_range("2024-01-01", periods=8, freq="B"))

# Signals: model output in [0, 1]; ≥ 0.55 → go long
signals = pd.Series([0.4, 0.6, 0.7, 0.3, 0.8, 0.6, 0.4, 0.7],
                    index=prices.index)

report = run_backtest(prices, signals, BacktestConfig(threshold=0.55))
print(report.summary())
```

---

## BacktestConfig

| Field | Default | Description |
|-------|---------|-------------|
| `threshold` | 0.5 | Signal score >= threshold → long position |
| `transaction_cost` | 0.001 | Round-trip cost as fraction of position (10 bps) |
| `allow_short` | False | If True, go short when signal < threshold |
| `initial_capital` | 10 000 | Starting portfolio value for equity curve |

---

## BacktestReport fields

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_return` | float | Total strategy return (e.g. `0.18` = 18%) |
| `annualised_return` | float | Annualised equivalent |
| `benchmark_return` | float | Buy-and-hold return over same period |
| `hit_rate` | float | Fraction of trades with positive PnL (0–1) |
| `max_drawdown` | float | Max peak-to-trough equity loss (negative) |
| `sharpe` | float | Annualised return / annualised volatility |
| `trade_count` | int | Number of completed round-trip trades |
| `n_bars` | int | Total number of price bars |
| `period_start` | str | ISO-8601 start timestamp |
| `period_end` | str | ISO-8601 end timestamp |
| `equity_curve` | `pd.Series` | Portfolio value over time |
| `daily_returns` | `pd.Series` | Day-by-day strategy returns |
| `positions` | `pd.Series` | Position at each bar (1=long, 0=flat, -1=short) |
| `trades` | `pd.DataFrame` | Per-trade log with entry/exit/pnl/correct |
| `per_trade_pnl` | `list[float]` | Net PnL for each trade |

---

## Integration with Experiment Registry

```python
from ml.registry import ExperimentRegistry

reg = ExperimentRegistry()
exp = reg.create("mlp_v1", "mlp", hyperparams=cfg.to_dict())

# ... train model, record metrics ...
reg.finish(exp.experiment_id, metrics=eval_metrics, checkpoint_path=ckpt)

# Run backtest
report = run_backtest(prices, signals)
reg.attach_backtest(exp.experiment_id, report.to_backtest_result())

# Promote if results are good
if report.hit_rate > 0.55 and report.cumulative_return > report.benchmark_return:
    reg.promote(exp.experiment_id, notes=f"Hit rate {report.hit_rate:.0%}")
```

---

## No-look-ahead guarantee

Signals are shifted forward by 1 bar before computing returns:

```
bar[t]: signal generated after close  →  position entered at bar[t+1] open
```

This is approximated as bar[t+1] close, which is slightly optimistic for the entry price but avoids using the same bar for both signal and return.

---

## Trade log columns

| Column | Type | Description |
|--------|------|-------------|
| `entry_date` | timestamp | Bar when position was entered |
| `exit_date` | timestamp | Bar when position was closed |
| `direction` | int | 1 = long, -1 = short |
| `entry_price` | float | Close price at entry |
| `exit_price` | float | Close price at exit |
| `pnl` | float | Net return including transaction cost |
| `correct` | bool | True if pnl > 0 |
