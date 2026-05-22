"""Walk-forward parameter optimizer with full trial recording.

Replaces the single 70/30 split in `optimize_pairs.py` with a rolling-window
walk-forward analysis that:

  1. Slides a (train, test) window forward in time, refitting parameters on
     each train window and evaluating UNTOUCHED on the next test window.
  2. Records every grid trial's train-set Sharpe at every fold — this is the
     input that the Deflated Sharpe Ratio analysis needs.
  3. Aggregates per-pair OOS (out-of-sample) trade PnLs across all folds to
     produce realistic OOS metrics — these are the numbers to trust, not the
     train-set PF.
  4. Tracks parameter stability across folds — if the optimiser picks wildly
     different params each fold, that is a regime-fragility signal.

Output structure (per pair) saved as JSON to data/walk_forward/<pair>.json:

    {
      "instrument": "EUR_USD",
      "spread_pips": 1.0,
      "config": { window_days, test_days, step_days, grid, ... },
      "folds": [
        {
          "fold_idx": 0,
          "train_start": "2023-01-01", "train_end": "2023-12-31",
          "test_start":  "2024-01-01", "test_end":  "2024-03-31",
          "trials": [   # one entry per grid combination
            { "params": {...}, "train_sharpe": 0.42, "train_pf": 1.23,
              "train_trades": 87 },
            ...
          ],
          "selected_params": {...},
          "test_pf": 1.05, "test_sharpe": 0.18, "test_trades": 24,
          "test_pnls": [ ... per-trade pnl list ... ]
        },
        ...
      ],
      "oos_aggregate": {
         "total_trades": N,
         "pnls": [ ... all OOS pnls concatenated ... ],
         "sharpe_per_trade": ...,   # raw (un-annualised), needed for DSR
         "sharpe_annualised": ...,
         "profit_factor": ...,
         "win_rate": ...,
         "skew": ...,                # of per-trade returns
         "kurtosis": ...,            # excess kurtosis
         "max_drawdown_pct": ...
      },
      "param_stability": {
         "bb_std":         {"mean": 2.1, "std": 0.18, "unique": 3},
         "rsi_oversold":   {...},
         ...
      }
    }

Usage:
    # Coarse smoke test (1 pair, ~3 folds, ~30s per fold)
    python scripts/walk_forward_optimize.py -f 2023-01-01 \
        --pairs EUR_USD --train-days 365 --test-days 90 --step-days 90

    # Full run on all qualified pairs (overnight job)
    python scripts/walk_forward_optimize.py -f 2022-01-01 \
        --train-days 365 --test-days 90 --step-days 90
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from backtest.simulator import BacktestSimulator
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.logger import setup_logging

from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE

# ── Cost-model injection seams ───────────────────────────────────────────────
# Both default to None → the file produces byte-for-byte identical output to
# the OANDA baseline run. Overrides are set by scripts/run_ecn_experiment.py.
#
# SIMULATOR_FACTORY(initial_equity, spread_pips, slippage_pips, conversion_cache)
#   → returns a BacktestSimulator (or subclass). If None, build BacktestSimulator
#     with the current arguments.
# SPREAD_OVERRIDE(instrument, oanda_spread_pips) → ecn_spread_pips.
#   If None, use SPREAD_TABLE[instrument] unchanged.
SIMULATOR_FACTORY = None
SPREAD_OVERRIDE = None


# ── Parameter grid (identical to optimize_pairs.py for comparability) ────────
PARAM_GRID = {
    "bb_std": [1.5, 1.75, 2.0, 2.25, 2.5],
    "rsi_thresholds": [(25, 75), (30, 70), (35, 65), (40, 60)],
    "sl_multiplier": [1.5, 2.0, 2.5, 3.0, 3.5],
}

# Minimum trades on a training window for a combo to be eligible for selection.
# Anything below this is statistical noise.
MIN_TRAIN_TRADES = 20

# Minimum total OOS trades across all folds for a pair to be considered viable.
MIN_OOS_TRADES = 100


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_strategy_config(
    base_config: dict,
    bb_std: float,
    rsi_oversold: int,
    rsi_overbought: int,
    sl_multiplier: float,
) -> dict:
    cfg = dict(base_config)
    mr = dict(cfg.get("mean_reversion", cfg))
    mr["bb_std"] = bb_std
    mr["rsi_oversold"] = rsi_oversold
    mr["rsi_overbought"] = rsi_overbought
    mr["sl_multiplier"] = sl_multiplier
    cfg["mean_reversion"] = mr
    return cfg


async def _run_backtest(
    m5_events, h1_events, strategy_config, full_config,
    spread_pips, slippage_pips, equity, conversion_cache,
):
    """Run a single backtest and return (PerformanceReport, list[float] pnls)."""
    strategy = MeanReversionStrategy(config=strategy_config)
    if SIMULATOR_FACTORY is not None:
        simulator = SIMULATOR_FACTORY(
            initial_equity=equity,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            conversion_cache=conversion_cache,
        )
    else:
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
    report = await engine.run(m5_events, h1_events)
    pnls = [float(c.pnl) for c in simulator.closes]
    return report, pnls


def _split_by_time(events, start: datetime, end: datetime):
    """Slice candle events by their .timestamp into [start, end)."""
    return [e for e in events if start <= e.timestamp < end]


def _per_trade_sharpe(pnls: list[float]) -> float:
    """Un-annualised Sharpe of per-trade PnLs. This is what DSR consumes."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd)


def _annualised_sharpe(pnls: list[float], total_days: float) -> float:
    """Sharpe scaled to annual frequency assuming the trades span total_days."""
    if len(pnls) < 2 or total_days <= 0:
        return 0.0
    sr_per_trade = _per_trade_sharpe(pnls)
    trades_per_year = len(pnls) * (365.0 / total_days)
    return sr_per_trade * float(np.sqrt(trades_per_year))


def _profit_factor(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    arr = np.array(pnls)
    gp = float(arr[arr > 0].sum())
    gl = float(-arr[arr < 0].sum())
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _max_drawdown_pct(pnls: list[float], initial_equity: float) -> float:
    if not pnls:
        return 0.0
    equity = initial_equity + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[initial_equity], equity]))
    dd = (peak[1:] - equity) / peak[1:]
    return float(dd.max())


def _skew_kurt(pnls: list[float]) -> tuple[float, float]:
    """Sample skewness and excess kurtosis (Fisher) — needed for DSR."""
    if len(pnls) < 4:
        return 0.0, 0.0
    arr = np.array(pnls, dtype=float)
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0
    z = (arr - mu) / sd
    n = len(arr)
    # Bias-corrected sample skewness
    g1 = (n / ((n - 1) * (n - 2))) * (z ** 3).sum()
    # Bias-corrected excess kurtosis
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (z ** 4).sum() \
         - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(g1), float(g2)


# ── Fold and pair processing ─────────────────────────────────────────────────


@dataclass
class FoldResult:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    trials: list[dict] = field(default_factory=list)
    selected_params: dict | None = None
    test_pf: float = 0.0
    test_sharpe_annualised: float = 0.0
    test_trades: int = 0
    test_pnls: list[float] = field(default_factory=list)


async def _process_pair(
    instrument: str,
    spread_pips: float,
    slippage_pips: float,
    equity: float,
    base_strategy_config: dict,
    full_config: dict,
    from_date: datetime,
    to_date: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    conversion_series,
) -> dict | None:

    config = load_config()
    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    # Load entire history once
    try:
        m5_df = loader.load_candles(
            instrument, "M5",
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
        )
    except FileNotFoundError:
        print(f"    No M5 data — skipping")
        return None
    if m5_df.empty:
        print(f"    Empty M5 data — skipping")
        return None
    m5_all = loader.df_to_candle_events(m5_df, instrument, "M5")

    h1_all = None
    try:
        h1_df = loader.load_candles(
            instrument, "H1",
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
        )
        if not h1_df.empty:
            h1_all = loader.df_to_candle_events(h1_df, instrument, "H1")
    except FileNotFoundError:
        pass

    combos = list(itertools.product(
        PARAM_GRID["bb_std"],
        PARAM_GRID["rsi_thresholds"],
        PARAM_GRID["sl_multiplier"],
    ))

    # Build folds
    folds: list[FoldResult] = []
    fold_idx = 0
    train_start = from_date
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > to_date:
            break

        m5_train = _split_by_time(m5_all, train_start, train_end)
        m5_test = _split_by_time(m5_all, test_start, test_end)
        h1_train = _split_by_time(h1_all, train_start, train_end) if h1_all else None
        h1_test = _split_by_time(h1_all, test_start, test_end) if h1_all else None

        if len(m5_train) < 500 or len(m5_test) < 100:
            train_start += timedelta(days=step_days)
            fold_idx += 1
            continue

        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        )

        # ── Grid search on train ─────────────────────────────────────────────
        best_pf = -float("inf")
        best_combo = None
        t0 = time.monotonic()

        for (bb_std, (rsi_os, rsi_ob), sl_mult) in combos:
            strat_cfg = _build_strategy_config(
                base_strategy_config, bb_std, rsi_os, rsi_ob, sl_mult,
            )
            report, pnls = await _run_backtest(
                m5_train, h1_train, strat_cfg, full_config,
                spread_pips, slippage_pips, equity,
                ConversionRateCache(conversion_series),
            )

            train_sr = _per_trade_sharpe(pnls)
            trial = {
                "params": {
                    "bb_std": bb_std,
                    "rsi_oversold": rsi_os,
                    "rsi_overbought": rsi_ob,
                    "sl_multiplier": sl_mult,
                },
                "train_sharpe_per_trade": train_sr,
                "train_pf": float(report.profit_factor),
                "train_trades": int(report.total_trades),
            }
            fold.trials.append(trial)

            if (report.total_trades >= MIN_TRAIN_TRADES
                    and report.profit_factor > best_pf):
                best_pf = report.profit_factor
                best_combo = (bb_std, rsi_os, rsi_ob, sl_mult)

        if best_combo is None:
            print(f"    Fold {fold_idx}: no combo met MIN_TRAIN_TRADES — skipping fold")
            folds.append(fold)
            train_start += timedelta(days=step_days)
            fold_idx += 1
            continue

        # ── Apply best to test (untouched) ───────────────────────────────────
        bb_std, rsi_os, rsi_ob, sl_mult = best_combo
        strat_cfg = _build_strategy_config(
            base_strategy_config, bb_std, rsi_os, rsi_ob, sl_mult,
        )
        report, pnls = await _run_backtest(
            m5_test, h1_test, strat_cfg, full_config,
            spread_pips, slippage_pips, equity,
            ConversionRateCache(conversion_series),
        )

        fold.selected_params = {
            "bb_std": bb_std, "rsi_oversold": rsi_os,
            "rsi_overbought": rsi_ob, "sl_multiplier": sl_mult,
        }
        fold.test_pf = float(report.profit_factor)
        fold.test_sharpe_annualised = float(report.sharpe_ratio)
        fold.test_trades = int(report.total_trades)
        fold.test_pnls = pnls

        elapsed = time.monotonic() - t0
        print(
            f"    Fold {fold_idx} [{fold.train_start}→{fold.test_end}] "
            f"params=({bb_std},{rsi_os}/{rsi_ob},{sl_mult}) | "
            f"trainPF={best_pf:.2f} testPF={fold.test_pf:.2f} "
            f"testTr={fold.test_trades} | {elapsed:.0f}s"
        )

        folds.append(fold)
        train_start += timedelta(days=step_days)
        fold_idx += 1

    if not folds:
        return None

    # ── Aggregate OOS metrics ────────────────────────────────────────────────
    all_oos_pnls: list[float] = []
    total_test_days = 0.0
    for f in folds:
        if f.selected_params is None:
            continue
        all_oos_pnls.extend(f.test_pnls)
        d0 = datetime.fromisoformat(f.test_start)
        d1 = datetime.fromisoformat(f.test_end)
        total_test_days += (d1 - d0).total_seconds() / 86400.0

    skew, kurt = _skew_kurt(all_oos_pnls)
    oos_aggregate = {
        "total_trades": len(all_oos_pnls),
        "total_test_days": round(total_test_days, 1),
        "pnls": all_oos_pnls,
        "sharpe_per_trade": _per_trade_sharpe(all_oos_pnls),
        "sharpe_annualised": _annualised_sharpe(all_oos_pnls, total_test_days),
        "profit_factor": _profit_factor(all_oos_pnls),
        "win_rate": (
            float(np.mean([p > 0 for p in all_oos_pnls])) if all_oos_pnls else 0.0
        ),
        "skew": skew,
        "kurtosis": kurt,
        "max_drawdown_pct": _max_drawdown_pct(all_oos_pnls, equity),
        "total_pnl": float(sum(all_oos_pnls)),
    }

    # ── Parameter stability across folds ─────────────────────────────────────
    selected = [f.selected_params for f in folds if f.selected_params]
    stability: dict[str, dict] = {}
    if selected:
        for key in ("bb_std", "rsi_oversold", "rsi_overbought", "sl_multiplier"):
            vals = [s[key] for s in selected]
            stability[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "unique": int(len(set(vals))),
                "values": vals,
            }

    return {
        "instrument": instrument,
        "spread_pips": spread_pips,
        "config": {
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "n_grid_combos": len(combos),
            "n_folds": len(folds),
            "min_train_trades": MIN_TRAIN_TRADES,
            "slippage_pips": slippage_pips,
            "initial_equity": equity,
        },
        "folds": [asdict(f) for f in folds],
        "oos_aggregate": oos_aggregate,
        "param_stability": stability,
    }


# ── Orchestration ────────────────────────────────────────────────────────────


async def run_walk_forward(
    instruments: list[str],
    from_date: str,
    to_date: str | None,
    train_days: int,
    test_days: int,
    step_days: int,
    equity: float,
    slippage_pips: float,
    output_dir: str,
) -> None:
    config = load_config()
    base_strategy_config = config.strategy
    full_config = config.raw

    # Load conversion series once
    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_series = load_conversion_series(data_dir)

    # Parquet candle timestamps are tz-aware UTC, so the fold boundaries must
    # be UTC-aware too — otherwise the [start, end) comparison in _split_by_time
    # raises TypeError on naive↔aware comparison.
    from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)
    to_dt = (
        datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc)
        if to_date else datetime.now(timezone.utc)
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    start_time = time.monotonic()
    total = len(instruments)

    for idx, instrument in enumerate(instruments, 1):
        oanda_spread = SPREAD_TABLE.get(instrument, 2.0)
        spread = (
            SPREAD_OVERRIDE(instrument, oanda_spread)
            if SPREAD_OVERRIDE is not None else oanda_spread
        )
        elapsed = time.monotonic() - start_time
        if idx > 1:
            per_pair = elapsed / (idx - 1)
            remaining = per_pair * (total - idx + 1)
            eta = f" (ETA: {remaining / 60:.0f}min)"
        else:
            eta = ""

        print(f"\n  [{idx}/{total}] {instrument} (spread={spread} pips){eta}")

        result = await _process_pair(
            instrument=instrument,
            spread_pips=spread,
            slippage_pips=slippage_pips,
            equity=equity,
            base_strategy_config=base_strategy_config,
            full_config=full_config,
            from_date=from_dt,
            to_date=to_dt,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            conversion_series=conversion_series,
        )

        if result is None:
            continue

        # Persist full per-pair JSON
        pair_file = out_path / f"{instrument}.json"
        with open(pair_file, "w") as f:
            json.dump(result, f, default=str, indent=2)
        print(f"    Saved: {pair_file}")

        oos = result["oos_aggregate"]
        summary.append({
            "instrument": instrument,
            "n_folds": result["config"]["n_folds"],
            "oos_trades": oos["total_trades"],
            "oos_pf": round(oos["profit_factor"], 3),
            "oos_sharpe_ann": round(oos["sharpe_annualised"], 3),
            "oos_total_pnl": round(oos["total_pnl"], 2),
            "oos_max_dd_pct": round(oos["max_drawdown_pct"], 3),
            "oos_skew": round(oos["skew"], 3),
            "oos_kurt": round(oos["kurtosis"], 3),
            "viable": (
                oos["total_trades"] >= MIN_OOS_TRADES
                and oos["profit_factor"] >= 1.10
            ),
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("  WALK-FORWARD SUMMARY (out-of-sample aggregated across all folds)")
    print("=" * 120)
    header = (
        f"{'Pair':<10} {'Folds':>5} {'OOS Tr':>7} {'OOS PF':>7} "
        f"{'OOS SR':>7} {'OOS PnL':>10} {'OOS MaxDD':>10} {'Skew':>6} {'Kurt':>6} {'Viable':>8}"
    )
    print(header)
    print("-" * 120)
    summary.sort(key=lambda r: r["oos_pf"], reverse=True)
    for r in summary:
        print(
            f"{r['instrument']:<10} {r['n_folds']:>5} {r['oos_trades']:>7} "
            f"{r['oos_pf']:>7.3f} {r['oos_sharpe_ann']:>7.3f} "
            f"{r['oos_total_pnl']:>10.2f} {r['oos_max_dd_pct']:>9.1%} "
            f"{r['oos_skew']:>6.2f} {r['oos_kurt']:>6.2f} "
            f"{'YES' if r['viable'] else 'NO':>8}"
        )
    print("-" * 120)
    viable_count = sum(1 for r in summary if r["viable"])
    print(f"  Viable (OOS trades ≥ {MIN_OOS_TRADES} and OOS PF ≥ 1.10): "
          f"{viable_count} / {len(summary)}")
    print("=" * 120)

    # Persist summary as YAML for easy diffing vs old pair_params.yaml
    summary_file = out_path / "_summary.yaml"
    with open(summary_file, "w") as f:
        f.write(f"# Walk-forward summary | Generated {datetime.now().isoformat()}\n")
        f.write(f"# Period: {from_date} to {to_date or 'now'}\n")
        f.write(f"# Train/Test/Step (days): {train_days}/{test_days}/{step_days}\n\n")
        yaml.dump({"pairs": summary}, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved summary: {summary_file}")

    elapsed_total = time.monotonic() - start_time
    print(f"  Total time: {elapsed_total / 60:.1f} minutes")
    print(f"\n  Next: python scripts/deflated_sharpe_analysis.py "
          f"--input-dir {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward optimizer with full trial recording for DSR.",
    )
    parser.add_argument("-f", "--from-date", required=True,
                        help="Start date YYYY-MM-DD (longer history → more folds)")
    parser.add_argument("-t", "--to-date", default=None,
                        help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--train-days", type=int, default=365,
                        help="Training window length in days (default: 365)")
    parser.add_argument("--test-days", type=int, default=90,
                        help="Test window length in days (default: 90)")
    parser.add_argument("--step-days", type=int, default=90,
                        help="Step forward between folds in days "
                             "(default: 90 = non-overlapping OOS)")
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument(
        "-o", "--output-dir", default="data/walk_forward",
        help="Directory to write per-pair JSON + _summary.yaml",
    )
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Override pair list (default: all G10 pairs)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip history download phase")

    args = parser.parse_args()
    setup_logging(level="WARNING", log_format="console")

    instruments = args.pairs or G10_PAIRS
    n_combos = (
        len(PARAM_GRID["bb_std"])
        * len(PARAM_GRID["rsi_thresholds"])
        * len(PARAM_GRID["sl_multiplier"])
    )

    print("\nWalk-Forward Optimizer")
    print(f"  Pairs:                {len(instruments)}")
    print(f"  Grid combos per fold: {n_combos}")
    print(f"  Train / Test / Step:  {args.train_days}d / {args.test_days}d / "
          f"{args.step_days}d")
    print(f"  Period:               {args.from_date} → {args.to_date or 'now'}")
    print(f"  Output dir:           {args.output_dir}")

    if not args.skip_download:
        from scripts.backtest_all_pairs import download_all_history
        print("\n--- Phase 1: Downloading history ---")
        asyncio.run(download_all_history(
            instruments, args.from_date, args.to_date,
        ))

    print("\n--- Phase 2: Walk-forward grid search ---")
    asyncio.run(run_walk_forward(
        instruments=instruments,
        from_date=args.from_date,
        to_date=args.to_date,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        equity=args.equity,
        slippage_pips=args.slippage,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
