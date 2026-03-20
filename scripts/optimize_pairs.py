"""Per-pair parameter optimizer — grid search with train/test split.

For each pair, tests combinations of (bb_std, rsi_thresholds, sl_multiplier)
on training data (first 70%), validates on test data (last 30%), and saves
optimal parameters to config/pair_params.yaml.

Usage:
    python scripts/optimize_pairs.py -f 2025-01-01
    python scripts/optimize_pairs.py -f 2025-01-01 --skip-download --pairs EUR_USD GBP_JPY
    python scripts/optimize_pairs.py -f 2025-01-01 -o config/pair_params.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

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
from src.utils.logger import setup_logging

# ── Reuse pair/spread definitions from backtest_all_pairs ──
from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE

# ── Parameter grid ─────────────────────────────────────────────────────────
PARAM_GRID = {
    "bb_std": [1.5, 1.75, 2.0, 2.25, 2.5],
    "rsi_thresholds": [(25, 75), (30, 70), (35, 65), (40, 60)],
    "sl_multiplier": [1.5, 2.0, 2.5, 3.0, 3.5],
}

TRAIN_RATIO = 0.70  # 70% train, 30% test
MIN_TRAIN_TRADES = 10  # Minimum trades on train set to be considered


def _build_strategy_config(
    base_config: dict,
    bb_std: float,
    rsi_oversold: int,
    rsi_overbought: int,
    sl_multiplier: float,
) -> dict:
    """Create a strategy config dict with overridden parameters."""
    cfg = dict(base_config)
    mr = dict(cfg.get("mean_reversion", cfg))
    mr["bb_std"] = bb_std
    mr["rsi_oversold"] = rsi_oversold
    mr["rsi_overbought"] = rsi_overbought
    mr["sl_multiplier"] = sl_multiplier
    cfg["mean_reversion"] = mr
    return cfg


async def _run_single_backtest(
    m5_events, h1_events, strategy_config, full_config, spread_pips, slippage_pips, equity,
    conversion_cache: ConversionRateCache | None = None,
) -> PerformanceReport:
    """Run one backtest with given parameters."""
    strategy = MeanReversionStrategy(config=strategy_config)
    simulator = BacktestSimulator(
        initial_equity=equity,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
    )
    mtf = MultiTimeframeFilter() if h1_events else None
    engine = BacktestEngine(
        strategy=strategy,
        simulator=simulator,
        config=full_config,
        multi_tf_filter=mtf,
        account_currency=full_config.get("account_currency", "AUD"),
        conversion_cache=conversion_cache,
    )
    return await engine.run(m5_events, h1_events)


def _split_events(events, ratio):
    """Split candle events into train/test by index (preserving time order)."""
    split_idx = int(len(events) * ratio)
    return events[:split_idx], events[split_idx:]


async def optimize_single_pair(
    instrument: str,
    from_date: str | None,
    to_date: str | None,
    spread_pips: float,
    slippage_pips: float,
    equity: float,
    base_strategy_config: dict,
    full_config: dict,
) -> dict | None:
    """Optimize parameters for a single pair via grid search."""
    config = load_config()
    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    # Load data
    try:
        m5_df = loader.load_candles(instrument, "M5", from_date, to_date)
    except FileNotFoundError:
        print(f"    No M5 data — skipping")
        return None
    if m5_df.empty:
        print(f"    Empty M5 data — skipping")
        return None
    m5_all = loader.df_to_candle_events(m5_df, instrument, "M5")

    h1_all = None
    try:
        h1_df = loader.load_candles(instrument, "H1", from_date, to_date)
        if not h1_df.empty:
            h1_all = loader.df_to_candle_events(h1_df, instrument, "H1")
    except FileNotFoundError:
        pass

    # Split into train/test
    m5_train, m5_test = _split_events(m5_all, TRAIN_RATIO)
    h1_train, h1_test = None, None
    if h1_all:
        h1_train, h1_test = _split_events(h1_all, TRAIN_RATIO)

    if len(m5_train) < 100 or len(m5_test) < 50:
        print(f"    Insufficient data (train={len(m5_train)}, test={len(m5_test)})")
        return None

    # Load conversion rate time series for cross-currency sizing (no look-ahead)
    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_series = load_conversion_series(data_dir)

    # Generate parameter combinations
    combos = list(itertools.product(
        PARAM_GRID["bb_std"],
        PARAM_GRID["rsi_thresholds"],
        PARAM_GRID["sl_multiplier"],
    ))

    best_pf = -float("inf")
    best_params = None
    best_report = None

    for i, (bb_std, (rsi_os, rsi_ob), sl_mult) in enumerate(combos):
        strat_cfg = _build_strategy_config(
            base_strategy_config, bb_std, rsi_os, rsi_ob, sl_mult,
        )

        report = await _run_single_backtest(
            m5_train, h1_train, strat_cfg, full_config,
            spread_pips, slippage_pips, equity,
            conversion_cache=ConversionRateCache(conversion_series),
        )

        if report.total_trades >= MIN_TRAIN_TRADES and report.profit_factor > best_pf:
            best_pf = report.profit_factor
            best_params = {
                "bb_std": bb_std,
                "rsi_oversold": rsi_os,
                "rsi_overbought": rsi_ob,
                "sl_multiplier": sl_mult,
            }
            best_report = report

    if best_params is None:
        print(f"    No valid parameter set found (all < {MIN_TRAIN_TRADES} trades)")
        return None

    # Validate on test set
    strat_cfg = _build_strategy_config(
        base_strategy_config,
        best_params["bb_std"],
        best_params["rsi_oversold"],
        best_params["rsi_overbought"],
        best_params["sl_multiplier"],
    )
    test_report = await _run_single_backtest(
        m5_test, h1_test, strat_cfg, full_config,
        spread_pips, slippage_pips, equity,
        conversion_cache=ConversionRateCache(conversion_series),
    )

    result = {
        **best_params,
        "train_pf": round(best_report.profit_factor, 3),
        "train_trades": best_report.total_trades,
        "train_win_rate": round(best_report.win_rate, 3),
        "test_pf": round(test_report.profit_factor, 3),
        "test_trades": test_report.total_trades,
        "test_win_rate": round(test_report.win_rate, 3),
        "overfit": test_report.profit_factor < 1.0,
    }

    status = "OVERFIT" if result["overfit"] else "OK"
    print(
        f"    Best: bb_std={best_params['bb_std']}, "
        f"rsi={best_params['rsi_oversold']}/{best_params['rsi_overbought']}, "
        f"sl={best_params['sl_multiplier']} | "
        f"Train PF={result['train_pf']:.3f} ({result['train_trades']}t) | "
        f"Test PF={result['test_pf']:.3f} ({result['test_trades']}t) [{status}]"
    )

    return result


async def run_optimizer(
    instruments: list[str],
    from_date: str,
    to_date: str | None,
    equity: float,
    slippage_pips: float,
    output_file: str,
) -> None:
    """Run optimization for all instruments."""
    config = load_config()
    base_strategy_config = config.strategy
    full_config = config.raw

    results: dict[str, dict] = {}
    total = len(instruments)
    start_time = time.monotonic()

    for idx, instrument in enumerate(instruments, 1):
        spread = SPREAD_TABLE.get(instrument, 2.0)
        elapsed = time.monotonic() - start_time
        if idx > 1:
            per_pair = elapsed / (idx - 1)
            remaining = per_pair * (total - idx + 1)
            eta = f" (ETA: {remaining / 60:.0f}min)"
        else:
            eta = ""

        print(f"\n  [{idx}/{total}] {instrument} (spread={spread}){eta}")

        result = await optimize_single_pair(
            instrument=instrument,
            from_date=from_date,
            to_date=to_date,
            spread_pips=spread,
            slippage_pips=slippage_pips,
            equity=equity,
            base_strategy_config=base_strategy_config,
            full_config=full_config,
        )

        if result is not None:
            results[instrument] = result

    # ── Summary ──
    print("\n" + "=" * 110)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 110)

    header = (
        f"{'Pair':<10} {'bb_std':>6} {'RSI':>7} {'SL':>5} "
        f"{'TrainPF':>8} {'TrainTr':>8} {'TestPF':>7} {'TestTr':>7} {'Status':>8}"
    )
    print(header)
    print("-" * 110)

    valid_count = 0
    overfit_count = 0
    for instrument in instruments:
        if instrument not in results:
            print(f"{instrument:<10} {'--':>6} {'--':>7} {'--':>5} {'--':>8} {'--':>8} {'--':>7} {'--':>7} {'NO DATA':>8}")
            continue

        r = results[instrument]
        status = "OVERFIT" if r["overfit"] else "OK"
        if r["overfit"]:
            overfit_count += 1
        else:
            valid_count += 1

        print(
            f"{instrument:<10} {r['bb_std']:>6.2f} "
            f"{r['rsi_oversold']:>2}/{r['rsi_overbought']:<2} "
            f"{r['sl_multiplier']:>5.1f} "
            f"{r['train_pf']:>8.3f} {r['train_trades']:>8} "
            f"{r['test_pf']:>7.3f} {r['test_trades']:>7} "
            f"{status:>8}"
        )

    print("-" * 110)
    print(f"  Valid: {valid_count} | Overfit (test PF < 1.0): {overfit_count} | No data: {total - len(results)}")
    print("=" * 110)

    # ── Save YAML ──
    # Only save non-overfit results; overfit pairs will use global defaults
    output = {}
    for instrument, r in results.items():
        if not r["overfit"]:
            output[instrument] = {
                "bb_std": r["bb_std"],
                "rsi_oversold": r["rsi_oversold"],
                "rsi_overbought": r["rsi_overbought"],
                "sl_multiplier": r["sl_multiplier"],
                "train_pf": r["train_pf"],
                "test_pf": r["test_pf"],
                "train_trades": r["train_trades"],
                "test_trades": r["test_trades"],
            }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("# Per-pair optimized parameters for mean_reversion strategy\n")
        f.write(f"# Generated by optimize_pairs.py | {datetime.now().isoformat()}\n")
        f.write(f"# Train/test split: {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}\n")
        f.write(f"# Pairs with test PF < 1.0 excluded (use global defaults)\n\n")
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Saved {len(output)} pair configs to {output_file}")

    elapsed_total = time.monotonic() - start_time
    print(f"  Total time: {elapsed_total / 60:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize mean_reversion parameters per pair (grid search + train/test)"
    )
    parser.add_argument("-f", "--from-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-t", "--to-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "-o", "--output", default="config/pair_params.yaml",
        help="Output file for optimized params",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Override pair list (default: all 28 G10)",
    )

    args = parser.parse_args()
    setup_logging(level="WARNING", log_format="console")

    instruments = args.pairs or G10_PAIRS
    combos = (
        len(PARAM_GRID["bb_std"])
        * len(PARAM_GRID["rsi_thresholds"])
        * len(PARAM_GRID["sl_multiplier"])
    )

    print(f"\nPer-Pair Parameter Optimizer")
    print(f"  Pairs: {len(instruments)}")
    print(f"  Combinations per pair: {combos}")
    print(f"  Total backtests: {len(instruments) * combos}")
    print(f"  Train/test split: {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}")
    print(f"  Period: {args.from_date} to {args.to_date or 'now'}")

    if not args.skip_download:
        from scripts.backtest_all_pairs import download_all_history
        print("\n--- Phase 1: Downloading history ---")
        asyncio.run(download_all_history(instruments, args.from_date, args.to_date))

    print("\n--- Phase 2: Optimizing parameters ---")
    asyncio.run(run_optimizer(
        instruments=instruments,
        from_date=args.from_date,
        to_date=args.to_date,
        equity=args.equity,
        slippage_pips=args.slippage,
        output_file=args.output,
    ))


if __name__ == "__main__":
    main()
