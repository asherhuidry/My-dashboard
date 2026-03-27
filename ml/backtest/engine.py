"""Clean backtest engine for evaluating model signals on historical data.

The engine is deliberately simple and transparent:
- One signal per bar (day/week): ``score`` in [0, 1].
- Position sizing is binary: long when ``score >= threshold``, flat (or short)
  otherwise.
- Transaction costs are applied as a fixed fraction of position value on each
  trade entry/exit.
- A buy-and-hold benchmark is computed over the same period for comparison.

The engine operates on pandas DataFrames and returns a ``BacktestReport``
dataclass that is directly attachable to an ``ExperimentRecord``.

Usage::

    from ml.backtest.engine import run_backtest, BacktestConfig

    report = run_backtest(
        prices  = df["close"],          # pd.Series, DatetimeIndex
        signals = pd.Series([...]),     # float in [0, 1], same index as prices
        config  = BacktestConfig(threshold=0.55, transaction_cost=0.001),
    )
    print(report.summary())
    bt = report.to_backtest_result()   # → BacktestResult for ExperimentRegistry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Parameters for the backtest engine.

    Attributes:
        threshold:        Signal score >= threshold → go long.
        transaction_cost: Round-trip cost as fraction of position value
                          (e.g. 0.001 = 10 bps).  Applied once per trade.
        allow_short:      If True, go short (−1) instead of flat when below
                          threshold.
        initial_capital:  Starting portfolio value (for equity curve).
        benchmark_col:    Not used by the engine; placeholder for future use.
    """
    threshold:        float = 0.5
    transaction_cost: float = 0.001
    allow_short:      bool  = False
    initial_capital:  float = 10_000.0


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestReport:
    """Full output from a backtest run.

    Attributes:
        period_start:       First bar timestamp (ISO-8601 string).
        period_end:         Last bar timestamp (ISO-8601 string).
        n_bars:             Total number of price bars.
        cumulative_return:  Strategy total return as a decimal (e.g. 0.18 = 18%).
        annualised_return:  Annualised equivalent of cumulative_return.
        benchmark_return:   Buy-and-hold total return over same period.
        hit_rate:           Fraction of trades with correct direction.
        max_drawdown:       Maximum peak-to-trough equity loss (negative decimal).
        sharpe:             Sharpe-like ratio (annualised return / annualised vol).
        trade_count:        Number of round-trip trades.
        equity_curve:       pd.Series — portfolio value over time.
        daily_returns:      pd.Series — daily strategy returns.
        positions:          pd.Series — position at each bar (1 = long, 0 = flat,
                            -1 = short).
        trades:             pd.DataFrame — one row per trade with entry/exit info.
        per_trade_pnl:      list[float] — PnL for each completed trade.
    """
    period_start:      str
    period_end:        str
    n_bars:            int
    cumulative_return: float
    annualised_return: float
    benchmark_return:  float
    hit_rate:          float
    max_drawdown:      float
    sharpe:            float
    trade_count:       int
    equity_curve:      pd.Series
    daily_returns:     pd.Series
    positions:         pd.Series
    trades:            pd.DataFrame
    per_trade_pnl:     list[float] = field(default_factory=list)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary."""
        return (
            f"Period: {self.period_start[:10]} → {self.period_end[:10]}  "
            f"({self.n_bars} bars, {self.trade_count} trades)\n"
            f"Return:      {self.cumulative_return*100:+.1f}%  "
            f"(benchmark {self.benchmark_return*100:+.1f}%)\n"
            f"Annualised:  {self.annualised_return*100:+.1f}%\n"
            f"Hit rate:    {self.hit_rate*100:.1f}%\n"
            f"Max DD:      {self.max_drawdown*100:.1f}%\n"
            f"Sharpe:      {self.sharpe:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dict (no DataFrames)."""
        return {
            "period_start":      self.period_start,
            "period_end":        self.period_end,
            "n_bars":            self.n_bars,
            "cumulative_return": round(self.cumulative_return, 6),
            "annualised_return": round(self.annualised_return, 6),
            "benchmark_return":  round(self.benchmark_return, 6),
            "hit_rate":          round(self.hit_rate, 4),
            "max_drawdown":      round(self.max_drawdown, 6),
            "sharpe":            round(self.sharpe, 4),
            "trade_count":       self.trade_count,
        }

    def to_backtest_result(self):
        """Convert to a BacktestResult attachable to an ExperimentRecord.

        Returns:
            ``ml.registry.experiment_registry.BacktestResult`` instance.
        """
        from ml.registry.experiment_registry import BacktestResult
        return BacktestResult(
            cumulative_return = self.cumulative_return,
            annualised_return = self.annualised_return,
            hit_rate          = self.hit_rate,
            max_drawdown      = self.max_drawdown,
            sharpe            = self.sharpe,
            trade_count       = self.trade_count,
            benchmark_return  = self.benchmark_return,
            period_start      = self.period_start,
            period_end        = self.period_end,
            extra             = {"n_bars": self.n_bars},
        )


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest(
    prices:  pd.Series,
    signals: pd.Series,
    config:  BacktestConfig | None = None,
) -> BacktestReport:
    """Run a vectorised backtest of model signals against historical prices.

    The engine aligns ``signals`` to ``prices`` on their index.  Signals are
    assumed to be generated *before* the corresponding price bar closes (i.e.
    the position is entered at the *next* bar's open, approximated here as the
    same bar's close + 1 period).

    Args:
        prices:  Close prices as a ``pd.Series`` with a ``DatetimeIndex``.
                 Must have at least 2 data points.
        signals: Model output scores in [0, 1], indexed like ``prices`` (or a
                 subset).  Values < config.threshold → flat (or short);
                 values >= threshold → long.
        config:  BacktestConfig.  Defaults to ``BacktestConfig()``.

    Returns:
        BacktestReport with all metrics and intermediate series.

    Raises:
        ValueError: If ``prices`` or ``signals`` have fewer than 2 data points,
                    or if the aligned overlap is empty.
    """
    cfg = config or BacktestConfig()

    # ── Align and validate ────────────────────────────────────────────────────
    prices  = prices.sort_index()
    signals = signals.sort_index()
    common  = prices.index.intersection(signals.index)

    if len(common) < 2:
        raise ValueError(
            f"Backtest requires at least 2 aligned bars; got {len(common)}."
        )

    prices  = prices.loc[common]
    signals = signals.loc[common]

    if prices.isnull().any():
        log.warning("prices contain %d NaNs; forward-filling.", prices.isnull().sum())
        prices = prices.ffill().bfill()

    if signals.isnull().any():
        log.warning("signals contain %d NaNs; filling with 0.", signals.isnull().sum())
        signals = signals.fillna(0.0)

    # ── Positions ─────────────────────────────────────────────────────────────
    # Shift signals by 1 bar so we enter *after* the signal is generated
    raw_pos = (signals >= cfg.threshold).astype(int)
    if cfg.allow_short:
        raw_pos = raw_pos * 2 - 1   # maps {0,1} → {-1, 1}

    positions = raw_pos.shift(1).fillna(0).astype(int)

    # ── Returns ───────────────────────────────────────────────────────────────
    price_returns = prices.pct_change().fillna(0.0)
    strat_returns = positions.shift(1).fillna(0) * price_returns

    # Apply transaction costs on trades (position changes)
    trades_mask = positions.diff().fillna(0) != 0
    strat_returns[trades_mask] -= cfg.transaction_cost

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity = (1 + strat_returns).cumprod() * cfg.initial_capital
    equity.name = "equity"

    # ── Metrics ───────────────────────────────────────────────────────────────
    cum_return   = float(equity.iloc[-1] / cfg.initial_capital - 1.0)
    bm_return    = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    n_bars       = len(prices)
    years        = n_bars / 252.0
    ann_return   = float((1 + cum_return) ** (1.0 / max(years, 1e-6)) - 1.0)
    daily_vol    = float(strat_returns.std())
    ann_vol      = daily_vol * np.sqrt(252)
    sharpe       = ann_return / ann_vol if ann_vol > 1e-9 else 0.0
    max_dd       = _max_drawdown(equity)

    # ── Trade log ─────────────────────────────────────────────────────────────
    trade_log, per_trade_pnl = _build_trade_log(prices, positions, cfg.transaction_cost)
    trade_count = len(trade_log)
    hit_rate    = _hit_rate(trade_log) if trade_count > 0 else 0.0

    period_start = str(prices.index[0].isoformat()) if hasattr(prices.index[0], "isoformat") \
        else str(prices.index[0])
    period_end   = str(prices.index[-1].isoformat()) if hasattr(prices.index[-1], "isoformat") \
        else str(prices.index[-1])

    log.info(
        "Backtest: %s → %s  return=%.2f%%  benchmark=%.2f%%  "
        "hit_rate=%.1f%%  max_dd=%.1f%%  sharpe=%.2f  trades=%d",
        period_start[:10], period_end[:10],
        cum_return * 100, bm_return * 100,
        hit_rate * 100, max_dd * 100, sharpe, trade_count,
    )

    return BacktestReport(
        period_start      = period_start,
        period_end        = period_end,
        n_bars            = n_bars,
        cumulative_return = cum_return,
        annualised_return = ann_return,
        benchmark_return  = bm_return,
        hit_rate          = hit_rate,
        max_drawdown      = max_dd,
        sharpe            = sharpe,
        trade_count       = trade_count,
        equity_curve      = equity,
        daily_returns     = strat_returns,
        positions         = positions,
        trades            = trade_log,
        per_trade_pnl     = per_trade_pnl,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_drawdown(equity: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown.

    Args:
        equity: Equity curve Series.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.12 = -12%).
    """
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def _build_trade_log(
    prices:           pd.Series,
    positions:        pd.Series,
    transaction_cost: float,
) -> tuple[pd.DataFrame, list[float]]:
    """Build a per-trade log from positions.

    Args:
        prices:           Close price series.
        positions:        Position series (1, 0, or -1).
        transaction_cost: Cost per trade (fractional).

    Returns:
        Tuple of (trade_log DataFrame, per_trade_pnl list).

    The trade_log DataFrame has columns:
        entry_date, exit_date, direction, entry_price, exit_price, pnl, correct
    """
    rows: list[dict[str, Any]] = []
    per_trade_pnl: list[float] = []

    pos_diff = positions.diff().fillna(0)
    changes  = pos_diff[pos_diff != 0]

    entry_date: Any  = None
    entry_price: float = 0.0
    direction:   int   = 0

    for date, delta in changes.items():
        # Close previous trade if we were in a position
        if direction != 0 and entry_date is not None:
            exit_price = float(prices.loc[date])
            raw_pnl = (exit_price / entry_price - 1.0) * direction
            pnl     = raw_pnl - transaction_cost
            correct = pnl > 0
            rows.append({
                "entry_date":  entry_date,
                "exit_date":   date,
                "direction":   direction,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "pnl":         pnl,
                "correct":     correct,
            })
            per_trade_pnl.append(pnl)

        # Open new trade if we're entering a position
        new_pos = int(positions.loc[date])
        if new_pos != 0:
            entry_date  = date
            entry_price = float(prices.loc[date])
            direction   = new_pos
        else:
            entry_date = None
            direction  = 0

    trade_log = pd.DataFrame(rows)
    if trade_log.empty:
        trade_log = pd.DataFrame(columns=[
            "entry_date", "exit_date", "direction",
            "entry_price", "exit_price", "pnl", "correct",
        ])
    return trade_log, per_trade_pnl


def _hit_rate(trade_log: pd.DataFrame) -> float:
    """Compute the fraction of trades that were correct.

    Args:
        trade_log: DataFrame from ``_build_trade_log``.

    Returns:
        Hit rate in [0, 1], or 0.0 if trade_log is empty.
    """
    if trade_log.empty or "correct" not in trade_log.columns:
        return 0.0
    return float(trade_log["correct"].mean())
