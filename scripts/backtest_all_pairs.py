"""Batch backtest all 28 G10 currency pairs with realistic per-pair spreads.

Downloads M5 + H1 history for each pair, runs mean_reversion backtest,
filters by PF >= threshold, and outputs a ranked summary table.

Usage:
    python scripts/backtest_all_pairs.py -f 2025-01-01 -t 2025-12-31
    python scripts/backtest_all_pairs.py -f 2025-01-01 --download-only
    python scripts/backtest_all_pairs.py -f 2025-01-01 --save-json results/g10.json
    python scripts/backtest_all_pairs.py -f 2025-01-01 --pairs EUR_USD GBP_JPY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from backtest.metrics import PerformanceReport
from backtest.simulator import BacktestSimulator
from src.api.client import OANDAClient
from src.data.history import HistoryFetcher
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# ── All 28 G10 currency pairs ──────────────────────────────────────────────
G10_PAIRS = [
    # Majors (7)
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "NZD_USD", "USD_CAD",
    # Crosses (21)
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_NZD", "EUR_CAD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_NZD", "GBP_CAD",
    "AUD_JPY", "AUD_CHF", "AUD_NZD", "AUD_CAD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CHF_JPY", "CAD_JPY", "CAD_CHF",
]

# ── Realistic OANDA practice spreads (pips) ────────────────────────────────
SPREAD_TABLE: dict[str, float] = {
    # Tight (1.0 pip)
    "EUR_USD": 1.0, "USD_JPY": 1.0,
    # Medium (1.5 pips)
    "GBP_USD": 1.5, "USD_CHF": 1.5, "AUD_USD": 1.5, "USD_CAD": 1.5,
    "EUR_GBP": 1.5, "EUR_JPY": 1.5,
    # Wide (2.0-2.5 pips)
    "NZD_USD": 2.0, "GBP_JPY": 2.5, "EUR_CHF": 2.0, "EUR_AUD": 2.0,
    "EUR_CAD": 2.0, "AUD_JPY": 2.0,
    # Wider (3.0-4.0 pips)
    "EUR_NZD": 3.5, "GBP_CHF": 3.5, "GBP_AUD": 3.5,
    "GBP_NZD": 4.0, "GBP_CAD": 3.5,
    "AUD_CHF": 3.0, "AUD_NZD": 3.0, "AUD_CAD": 3.0,
    "NZD_JPY": 3.0, "CHF_JPY": 3.0, "CAD_JPY": 3.0,
    # Widest (4.0-5.0 pips)
    "NZD_CHF": 4.5, "NZD_CAD": 4.0, "CAD_CHF": 4.5,
}


async def download_all_history(
    instruments: list[str],
    from_date: str,
    to_date: str | None,
) -> None:
    """Download M5 and H1 data for all instruments."""
    config = load_config()
    client = OANDAClient(config.oanda)
    fetcher = HistoryFetcher(
        client, cache_dir=config.data.get("parquet_dir", "data/parquet")
    )

    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = (
        datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if to_date
        else datetime.now(timezone.utc)
    )

    total = len(instruments) * 2  # M5 + H1 per pair
    done = 0
    for instrument in instruments:
        for granularity in ("M5", "H1"):
            done += 1
            print(f"  [{done}/{total}] Downloading {instrument} {granularity}...")
            try:
                df = await fetcher.fetch_candles(
                    instrument=instrument,
                    granularity=granularity,
                    from_time=from_dt,
                    to_time=to_dt,
                    use_cache=True,
                )
                print(f"    {len(df)} candles")
            except Exception as e:
                print(f"    FAILED: {e}")


async def backtest_single_pair(
    instrument: str,
    from_date: str | None,
    to_date: str | None,
    initial_equity: float,
    spread_pips: float,
    slippage_pips: float,
    pair_params: dict | None = None,
) -> PerformanceReport | None:
    """Run mean_reversion backtest for a single pair."""
    config = load_config()
    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    # Load M5
    try:
        m5_df = loader.load_candles(instrument, "M5", from_date, to_date)
    except FileNotFoundError:
        print(f"  {instrument}: No M5 data -- skipping")
        return None
    if m5_df.empty:
        print(f"  {instrument}: Empty M5 data -- skipping")
        return None
    m5_events = loader.df_to_candle_events(m5_df, instrument, "M5")

    # Load H1
    h1_events = None
    try:
        h1_df = loader.load_candles(instrument, "H1", from_date, to_date)
        if not h1_df.empty:
            h1_events = loader.df_to_candle_events(h1_df, instrument, "H1")
    except FileNotFoundError:
        pass

    # Strategy — inject per-pair params if provided
    strategy_config = dict(config.strategy.get("mean_reversion", config.strategy))
    if pair_params:
        strategy_config["pair_params"] = pair_params
    strategy = MeanReversionStrategy(config=strategy_config)

    # Simulator with per-pair spread
    simulator = BacktestSimulator(
        initial_equity=initial_equity,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
    )

    # MTF filter
    mtf = MultiTimeframeFilter() if h1_events else None

    # Load conversion rate time series for cross-currency sizing (no look-ahead)
    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_cache = ConversionRateCache(load_conversion_series(data_dir))

    # Engine
    engine = BacktestEngine(
        strategy=strategy,
        simulator=simulator,
        config=config.raw,
        multi_tf_filter=mtf,
        account_currency=config.raw.get("account_currency", "AUD"),
        conversion_cache=conversion_cache,
    )

    report = await engine.run(m5_events, h1_events)
    return report


async def run_batch_backtest(
    instruments: list[str],
    from_date: str | None,
    to_date: str | None,
    initial_equity: float,
    slippage_pips: float,
    save_json: str | None,
    min_pf: float,
    pair_params: dict | None = None,
) -> None:
    """Run backtests for all instruments and print summary."""
    results: list[dict] = []
    total = len(instruments)
    using_pp = pair_params is not None

    for idx, instrument in enumerate(instruments, 1):
        spread = SPREAD_TABLE.get(instrument, 2.0)
        pp_marker = " [optimized]" if using_pp and instrument in (pair_params or {}) else ""
        print(f"\n  [{idx}/{total}] {instrument} (spread={spread} pips){pp_marker}...")

        try:
            report = await backtest_single_pair(
                instrument=instrument,
                from_date=from_date,
                to_date=to_date,
                initial_equity=initial_equity,
                spread_pips=spread,
                slippage_pips=slippage_pips,
                pair_params=pair_params,
            )
        except Exception as e:
            print(f"  {instrument}: ERROR - {e}")
            continue

        if report is None or report.total_trades == 0:
            print(f"  {instrument}: No trades generated")
            continue

        results.append({
            "instrument": instrument,
            "trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "sharpe": report.sharpe_ratio,
            "sortino": report.sortino_ratio,
            "max_dd_pct": report.max_drawdown_pct,
            "total_pnl": report.total_pnl,
            "trades_per_day": report.trades_per_day,
            "spread_pips": spread,
        })

        pf_marker = " OK" if report.profit_factor >= min_pf else ""
        print(
            f"    {report.total_trades} trades, "
            f"WR={report.win_rate:.1%}, PF={report.profit_factor:.3f}, "
            f"Sharpe={report.sharpe_ratio:.3f}{pf_marker}"
        )

    # ── Summary table ──
    params_label = " (per-pair optimized)" if using_pp else " (uniform params)"
    print("\n" + "=" * 100)
    print(f"  BATCH BACKTEST SUMMARY — Mean Reversion Strategy{params_label}")
    print("=" * 100)

    # Sort by profit factor descending
    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    header = (
        f"{'Pair':<10} {'Trades':>6} {'Tr/Day':>6} {'WinRate':>8} "
        f"{'PF':>7} {'Sharpe':>7} {'MaxDD%':>8} {'PnL':>12} {'Spread':>7}"
    )
    print(header)
    print("-" * 100)

    qualified = []
    for r in results:
        marker = " *" if r["profit_factor"] >= min_pf else "  "
        print(
            f"{r['instrument']:<10} {r['trades']:>6} "
            f"{r['trades_per_day']:>6.1f} "
            f"{r['win_rate']:>7.1%} {r['profit_factor']:>7.3f} "
            f"{r['sharpe']:>7.3f} {r['max_dd_pct']:>7.2%} "
            f"{r['total_pnl']:>12,.2f} {r['spread_pips']:>6.1f}{marker}"
        )
        if r["profit_factor"] >= min_pf:
            qualified.append(r["instrument"])

    print("-" * 100)
    print(f"  Total pairs tested: {len(results)}")
    print(f"  Qualified (PF >= {min_pf}): {len(qualified)}")
    if qualified:
        print(f"  Qualified pairs: {', '.join(qualified)}")
    else:
        print("  Qualified pairs: NONE")
    print("  (* = qualified)")
    print("=" * 100)

    # ── Estimated daily volume ──
    if qualified:
        qualified_results = [r for r in results if r["instrument"] in qualified]
        total_trades_day = sum(r["trades_per_day"] for r in qualified_results)
        print(f"\n  Estimated trades/day across qualified pairs: {total_trades_day:.1f}")

    # ── Save JSON ──
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "qualified_pairs": qualified,
            "min_profit_factor": min_pf,
            "from_date": from_date,
            "to_date": to_date,
            "results": results,
        }
        with open(save_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {save_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch backtest all 28 G10 pairs with realistic spreads"
    )
    parser.add_argument("-f", "--from-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-t", "--to-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("--min-pf", type=float, default=1.05, help="Min profit factor to qualify")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use cached data")
    parser.add_argument("--save-json", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Override pair list (default: all 28 G10)",
    )
    parser.add_argument(
        "--use-pair-params", default=None, nargs="?", const="config/pair_params.yaml",
        help="Use per-pair optimized params from YAML (default: config/pair_params.yaml)",
    )

    args = parser.parse_args()
    setup_logging(level="WARNING", log_format="console")

    instruments = args.pairs or G10_PAIRS

    # Load per-pair params if requested
    pair_params = None
    if args.use_pair_params:
        import yaml
        pp_path = Path(args.use_pair_params)
        if pp_path.exists():
            with open(pp_path) as f:
                pair_params = yaml.safe_load(f) or {}
            print(f"\nLoaded per-pair params from {pp_path} ({len(pair_params)} pairs)")
        else:
            print(f"\nWARNING: {pp_path} not found — using uniform params")

    print(f"\nBatch Backtest: {len(instruments)} G10 pairs")
    print(f"Period: {args.from_date} to {args.to_date or 'now'}")
    print(f"Equity: {args.equity:,.0f}, Slippage: {args.slippage} pips")
    print(f"Min PF: {args.min_pf}")
    if pair_params:
        print(f"Params: per-pair optimized ({len(pair_params)} pairs, rest use defaults)")

    if not args.skip_download:
        print("\n--- Phase 1: Downloading history ---")
        asyncio.run(download_all_history(instruments, args.from_date, args.to_date))

    if args.download_only:
        print("\nDownload complete.")
        return

    print("\n--- Phase 2: Running backtests ---")
    asyncio.run(run_batch_backtest(
        instruments=instruments,
        from_date=args.from_date,
        to_date=args.to_date,
        initial_equity=args.equity,
        slippage_pips=args.slippage,
        save_json=args.save_json,
        min_pf=args.min_pf,
        pair_params=pair_params,
    ))


if __name__ == "__main__":
    main()
