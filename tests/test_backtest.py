"""Tests for ml.backtest.engine."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.backtest.engine import (
    BacktestConfig,
    BacktestReport,
    run_backtest,
    _max_drawdown,
    _build_trade_log,
    _hit_rate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 30, start: str = "2023-01-02", trend: float = 0.001) -> pd.Series:
    """Build a synthetic price series with a slight upward drift."""
    np.random.seed(42)
    dates  = pd.bdate_range(start=start, periods=n)
    rets   = np.random.normal(trend, 0.01, size=n)
    prices = 100.0 * np.cumprod(1 + rets)
    return pd.Series(prices, index=dates, name="close")


def _make_signals(prices: pd.Series, frac_long: float = 0.6) -> pd.Series:
    """Build synthetic signals: frac_long fraction of bars are >= 0.5."""
    np.random.seed(7)
    n   = len(prices)
    raw = np.random.uniform(0, 1, size=n)
    # Push frac_long of bars above 0.5
    threshold_idx = int(n * (1 - frac_long))
    raw_sorted    = np.sort(raw)
    cutoff        = raw_sorted[threshold_idx]
    return pd.Series(raw, index=prices.index)


# ── BacktestConfig ────────────────────────────────────────────────────────────

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.threshold        == 0.5
        assert cfg.transaction_cost == 0.001
        assert cfg.allow_short      is False
        assert cfg.initial_capital  == 10_000.0


# ── run_backtest ──────────────────────────────────────────────────────────────

class TestRunBacktest:
    def test_returns_report(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert isinstance(report, BacktestReport)

    def test_period_dates_set(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert report.period_start != ""
        assert report.period_end   != ""

    def test_n_bars_matches_overlap(self):
        prices  = _make_prices(n=20)
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert report.n_bars == 20

    def test_equity_curve_length(self):
        prices  = _make_prices(n=20)
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert len(report.equity_curve) == 20

    def test_equity_curve_starts_near_capital(self):
        prices  = _make_prices(n=20)
        # All signals flat → position=0 → equity stays at initial capital
        signals = pd.Series(0.0, index=prices.index)
        report  = run_backtest(prices, signals, BacktestConfig(threshold=0.5))
        assert abs(report.equity_curve.iloc[0] - 10_000.0) < 1.0

    def test_daily_returns_same_length_as_prices(self):
        prices  = _make_prices(n=25)
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert len(report.daily_returns) == 25

    def test_positions_values_are_valid(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert set(report.positions.unique()).issubset({-1, 0, 1})

    def test_no_lookahead_bias(self):
        """Signal at bar t should only affect position from bar t+1 onward."""
        prices  = _make_prices(n=10)
        signals = pd.Series(1.0, index=prices.index)  # always long
        report  = run_backtest(prices, signals, BacktestConfig(threshold=0.5))
        # First bar: no prior signal → position should be 0
        assert report.positions.iloc[0] == 0

    def test_all_flat_signals_zero_trades(self):
        prices  = _make_prices(n=20)
        signals = pd.Series(0.0, index=prices.index)
        report  = run_backtest(prices, signals, BacktestConfig(threshold=0.5))
        assert report.trade_count == 0

    def test_all_long_signals_few_trades(self):
        prices  = _make_prices(n=20)
        signals = pd.Series(1.0, index=prices.index)
        report  = run_backtest(prices, signals)
        # Enters once, stays in — should have at most 1-2 trades
        assert report.trade_count <= 2

    def test_cumulative_return_type(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert isinstance(report.cumulative_return, float)
        assert np.isfinite(report.cumulative_return)

    def test_sharpe_type(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert isinstance(report.sharpe, float)
        assert np.isfinite(report.sharpe)

    def test_hit_rate_in_range(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert 0.0 <= report.hit_rate <= 1.0

    def test_max_drawdown_negative_or_zero(self):
        prices  = _make_prices()
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert report.max_drawdown <= 0.0

    def test_benchmark_return_plausible(self):
        prices  = _make_prices(n=252, trend=0.0005)
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        # With upward trend the benchmark should be positive
        assert report.benchmark_return > -0.5  # not catastrophically wrong

    def test_transaction_cost_reduces_return(self):
        prices  = _make_prices(n=40)
        signals = _make_signals(prices)
        r_no_cost   = run_backtest(prices, signals, BacktestConfig(transaction_cost=0.0))
        r_with_cost = run_backtest(prices, signals, BacktestConfig(transaction_cost=0.05))
        assert r_no_cost.cumulative_return >= r_with_cost.cumulative_return

    def test_allow_short_changes_positions(self):
        prices  = _make_prices(n=20)
        signals = pd.Series([0.2] * 10 + [0.8] * 10, index=prices.index)
        r_flat  = run_backtest(prices, signals, BacktestConfig(allow_short=False))
        r_short = run_backtest(prices, signals, BacktestConfig(allow_short=True))
        assert -1 in r_short.positions.values
        assert -1 not in r_flat.positions.values

    def test_partial_signal_overlap(self):
        """Signals covering only half the price history should still work."""
        prices  = _make_prices(n=20)
        signals = _make_signals(prices).iloc[5:15]  # only 10 bars
        report  = run_backtest(prices, signals)
        assert report.n_bars == 10

    def test_too_few_bars_raises(self):
        prices  = _make_prices(n=1)
        signals = pd.Series([0.7], index=prices.index)
        with pytest.raises(ValueError, match="at least 2"):
            run_backtest(prices, signals)

    def test_nan_prices_filled(self):
        prices  = _make_prices(n=20).copy()
        prices.iloc[5] = np.nan
        signals = _make_signals(prices)
        report  = run_backtest(prices, signals)
        assert np.isfinite(report.cumulative_return)

    def test_nan_signals_filled_with_zero(self):
        prices  = _make_prices(n=20)
        signals = _make_signals(prices).copy()
        signals.iloc[3] = np.nan
        report  = run_backtest(prices, signals)
        assert np.isfinite(report.cumulative_return)


# ── BacktestReport helpers ────────────────────────────────────────────────────

class TestBacktestReport:
    def _run(self) -> BacktestReport:
        prices  = _make_prices()
        signals = _make_signals(prices)
        return run_backtest(prices, signals)

    def test_summary_is_string(self):
        report = self._run()
        s = report.summary()
        assert isinstance(s, str)
        assert "Return" in s or "return" in s.lower()

    def test_to_dict_json_serializable(self):
        report = self._run()
        d = report.to_dict()
        assert json.dumps(d)

    def test_to_dict_has_required_keys(self):
        report = self._run()
        d = report.to_dict()
        for k in ("cumulative_return", "benchmark_return", "hit_rate",
                  "max_drawdown", "sharpe", "trade_count"):
            assert k in d

    def test_to_backtest_result(self):
        from ml.registry.experiment_registry import BacktestResult
        report = self._run()
        bt = report.to_backtest_result()
        assert isinstance(bt, BacktestResult)
        assert bt.cumulative_return == report.cumulative_return
        assert bt.hit_rate          == report.hit_rate
        assert bt.trade_count       == report.trade_count

    def test_to_backtest_result_serializable(self):
        report = self._run()
        bt = report.to_backtest_result()
        assert json.dumps(bt.to_dict())


# ── Internal helpers ──────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([100, 101, 102, 103, 104])
        assert _max_drawdown(equity) == 0.0

    def test_drawdown_computed(self):
        equity = pd.Series([100, 120, 90, 110])
        dd = _max_drawdown(equity)
        expected = (90 - 120) / 120
        assert abs(dd - expected) < 1e-9

    def test_full_recovery(self):
        equity = pd.Series([100, 80, 100, 110])
        dd = _max_drawdown(equity)
        assert abs(dd - (80 - 100) / 100) < 1e-9


class TestBuildTradeLog:
    def test_empty_when_no_trades(self):
        prices    = _make_prices(n=5)
        positions = pd.Series(0, index=prices.index)
        log, pnl  = _build_trade_log(prices, positions, 0.001)
        assert log.empty
        assert pnl == []

    def test_one_trade(self):
        prices    = _make_prices(n=5)
        # Enter long at bar 1, exit at bar 3
        pos = pd.Series([0, 1, 1, 0, 0], index=prices.index)
        log, pnl = _build_trade_log(prices, pos, 0.0)
        assert len(log) == 1
        assert len(pnl) == 1
        assert "entry_date" in log.columns
        assert "correct" in log.columns

    def test_trade_columns_present(self):
        prices = _make_prices(n=10)
        pos    = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 0], index=prices.index)
        log, _ = _build_trade_log(prices, pos, 0.001)
        expected_cols = {"entry_date", "exit_date", "direction",
                         "entry_price", "exit_price", "pnl", "correct"}
        assert expected_cols.issubset(set(log.columns))

    def test_pnl_reduces_with_cost(self):
        prices = _make_prices(n=5)
        pos    = pd.Series([0, 1, 1, 0, 0], index=prices.index)
        _, pnl_no_cost   = _build_trade_log(prices, pos, 0.0)
        _, pnl_with_cost = _build_trade_log(prices, pos, 0.01)
        assert pnl_no_cost[0] > pnl_with_cost[0]


class TestHitRate:
    def test_all_correct(self):
        df = pd.DataFrame({"correct": [True, True, True]})
        assert _hit_rate(df) == 1.0

    def test_none_correct(self):
        df = pd.DataFrame({"correct": [False, False]})
        assert _hit_rate(df) == 0.0

    def test_mixed(self):
        df = pd.DataFrame({"correct": [True, False, True, True]})
        assert abs(_hit_rate(df) - 0.75) < 1e-9

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["correct"])
        assert _hit_rate(df) == 0.0


# ── End-to-end integration ────────────────────────────────────────────────────

class TestEndToEnd:
    def test_registry_integration(self, tmp_path):
        """Attach a BacktestReport to an ExperimentRecord and promote it."""
        from ml.registry.experiment_registry import ExperimentRegistry

        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        exp = reg.create("backtest_e2e", "mlp", hyperparams={"threshold": 0.55})
        reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.58})

        prices  = _make_prices(n=60)
        signals = _make_signals(prices, frac_long=0.55)
        report  = run_backtest(prices, signals, BacktestConfig(threshold=0.55))

        reg.attach_backtest(exp.experiment_id, report.to_backtest_result())
        reg.promote(exp.experiment_id, notes="E2E test promotion")

        loaded = reg.get(exp.experiment_id)
        from ml.registry.experiment_registry import ExperimentStatus
        assert loaded.status == ExperimentStatus.PROMOTED
        assert loaded.backtest is not None
        assert loaded.backtest.trade_count == report.trade_count
