"""FinBrain ingestion orchestrator.

Entry point for all data ingestion. Runs all four connectors in sequence:
  1. yfinance  — equities, forex, commodities
  2. CoinGecko — top 20 crypto
  3. FRED      — 10 macro indicators
  4. Alpha Vantage — supplemental equity data (top 10 only, free tier limit)

Usage:
    python data/ingest/run.py [--dry-run]

Flags:
    --dry-run   Print what would be fetched without hitting any APIs.
"""

from __future__ import annotations

import argparse
import sys

from data.ingest import (
    alpha_vantage_connector,
    coingecko_connector,
    fred_connector,
    yfinance_connector,
)
from data.ingest.universe import (
    CRYPTO,
    MACRO_SERIES,
    get_yfinance_universe,
)
from skills.logger import get_logger

logger = get_logger(__name__)

# Alpha Vantage free tier: 25 req/day — use the top 10 equities only
AV_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "JPM",  "TSLA", "UNH",  "V",
]


def _dry_run() -> None:
    """Print what would be fetched without making any API calls."""
    universe = get_yfinance_universe()
    print(f"\n[DRY RUN] yfinance: {len(universe)} tickers")
    for t, ac in list(universe.items())[:5]:
        print(f"  {t} ({ac})")
    print(f"  ... and {len(universe) - 5} more")

    print(f"\n[DRY RUN] CoinGecko: {len(CRYPTO)} coins")
    for coin_id, ticker in CRYPTO[:3]:
        print(f"  {ticker} ({coin_id})")

    print(f"\n[DRY RUN] FRED: {len(MACRO_SERIES)} series")
    for sid, name, _ in MACRO_SERIES:
        print(f"  {sid}: {name}")

    print(f"\n[DRY RUN] Alpha Vantage: {len(AV_TICKERS)} tickers")
    for t in AV_TICKERS:
        print(f"  {t}")


def run_all(period: str = "2y") -> dict[str, int]:
    """Run all four ingestion connectors and return a summary.

    Args:
        period: yfinance period string applied to equity/forex/commodity fetch.

    Returns:
        Dict mapping connector name to total rows written.
    """
    summary: dict[str, int] = {}

    # 1. yfinance
    logger.info("ingestion_start", connector="yfinance")
    yf_results = yfinance_connector.run(get_yfinance_universe(), period=period)
    summary["yfinance"] = sum(r.rows_written for r in yf_results if r.error is None)
    logger.info("ingestion_done", connector="yfinance", rows=summary["yfinance"])

    # 2. CoinGecko
    logger.info("ingestion_start", connector="coingecko")
    cg_results = coingecko_connector.run(CRYPTO)
    summary["coingecko"] = sum(r.rows_written for r in cg_results if r.error is None)
    logger.info("ingestion_done", connector="coingecko", rows=summary["coingecko"])

    # 3. FRED
    logger.info("ingestion_start", connector="fred")
    fred_results = fred_connector.run(MACRO_SERIES)
    summary["fred"] = sum(r.rows_written for r in fred_results if r.error is None)
    logger.info("ingestion_done", connector="fred", rows=summary["fred"])

    # 4. Alpha Vantage (supplemental)
    logger.info("ingestion_start", connector="alpha_vantage")
    av_results = alpha_vantage_connector.run(AV_TICKERS)
    summary["alpha_vantage"] = sum(r.rows_written for r in av_results if r.error is None)
    logger.info("ingestion_done", connector="alpha_vantage", rows=summary["alpha_vantage"])

    total = sum(summary.values())
    logger.info("ingestion_complete", total_rows=total, breakdown=summary)
    return summary


def main() -> None:
    """CLI entry point for the ingestion orchestrator."""
    parser = argparse.ArgumentParser(description="FinBrain data ingestion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without calling any APIs")
    parser.add_argument("--period", default="2y",
                        help="yfinance history period (default: 2y)")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        sys.exit(0)

    summary = run_all(period=args.period)
    print("\nIngestion complete:")
    for connector, rows in summary.items():
        print(f"  {connector}: {rows:,} rows written")
    print(f"  TOTAL: {sum(summary.values()):,} rows")


if __name__ == "__main__":
    main()
