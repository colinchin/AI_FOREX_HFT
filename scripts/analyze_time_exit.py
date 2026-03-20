"""Analyze optimal time-exit duration for mean reversion strategy.

Runs backtests across all qualified pairs with different time-exit cutoffs
(profitable trades only) to empirically determine the best max_trade_duration_minutes.

Usage:
    python scripts/analyze_time_exit.py -f 2025-01-01 -t 2025-12-31
    python scripts/analyze_time_exit.py -f 2025-01-01 --cutoffs 30 60 90 120 180 360
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from backtest.simulator import BacktestSimulator, SimulatedPosition
from src.core.events import CandleEvent, Direction, TradeCloseEvent
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.helpers import pip_value
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# 11 qualified pairs from settings.yaml
QUALIFIED_PAIRS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD",
    "NZD_USD", "EUR_GBP", "EUR_JPY", "EUR_CHF", "GBP_AUD", "NZD_CAD",
]

# Realistic OANDA practice spreads (same as backtest_all_pairs.py)
SPREAD_TABLE: dict[str, float] = {
    "EUR_USD": 1.0, "USD_JPY": 1.0,
    "GBP_USD": 1.5, "USD_CHF": 1.5, "AUD_USD": 1.5,
    "EUR_GBP": 1.5, "EUR_JPY": 1.5,
    "NZD_USD": 2.0, "EUR_CHF": 2.0,
    "GBP_AUD": 3.5, "NZD_CAD": 4.0,
}

DEFAULT_CUTOFFS = [0, 30, 60, 90, 120, 150, 180, 240, 360]


class TimeExitSimulator(BacktestSimulator):
    """BacktestSimulator extended with time-based exit for profitable trades."""

    def __init__(
        self,
        max_trade_duration_minutes: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_duration_sec = (
            max_trade_duration_minutes * 60 if max_trade_duration_minutes else None
        )

    def process_candle(self, candle: CandleEvent) -> list[TradeCloseEvent]:
        """Check time-exit BEFORE normal SL/TP checks."""
        time_closed = []
        to_remove = []

        if self._max_duration_sec is not None:
            for trade_id, pos in self._positions.items():
                if pos.instrument != candle.instrument:
                    continue

                age_sec = (candle.timestamp - pos.entry_time).total_seconds()
                if age_sec <= self._max_duration_sec:
                    continue

                # Check if trade is in profit at current candle close
                if pos.direction is Direction.LONG:
                    pnl = (candle.close - pos.entry_price) * pos.units
                else:
                    pnl = (pos.entry_price - candle.close) * pos.units

                if pnl > 0:
                    event = self._close_position(pos, candle.close, candle, "time_exit")
                    time_closed.append(event)
                    to_remove.append(trade_id)

            for tid in to_remove:
                del self._positions[tid]

        # Normal SL/TP processing on remaining positions
        normal_closed = super().process_candle(candle)
        return time_closed + normal_closed


async def run_single(
    instrument: str,
    from_date: str | None,
    to_date: str | None,
    initial_equity: float,
    slippage_pips: float,
    cutoff_minutes: int | None,
    pair_params: dict | None = None,
) -> dict | None:
    """Run one backtest with a specific time-exit cutoff."""
    config = load_config()
    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    # Load M5
    try:
        m5_df = loader.load_candles(instrument, "M5", from_date, to_date)
    except FileNotFoundError:
        return None
    if m5_df.empty:
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

    # Strategy with per-pair params
    strategy_config = dict(config.strategy.get("mean_reversion", config.strategy))
    if pair_params:
        strategy_config["pair_params"] = pair_params
    strategy = MeanReversionStrategy(config=strategy_config)

    # Simulator with time exit
    spread = SPREAD_TABLE.get(instrument, 2.0)
    simulator = TimeExitSimulator(
        max_trade_duration_minutes=cutoff_minutes,
        initial_equity=initial_equity,
        spread_pips=spread,
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

    if report.total_trades == 0:
        return None

    # Compute avg trade duration from fills + closes
    fills_by_id = {f.trade_id: f for f in simulator.fills}
    durations = []
    for c in simulator.closes:
        fill = fills_by_id.get(c.trade_id)
        if fill:
            dur_min = (c.timestamp - fill.timestamp).total_seconds() / 60
            durations.append(dur_min)

    # Close reason breakdown
    reasons = {}
    for c in simulator.closes:
        r = c.reason or "unknown"
        reasons.setdefault(r, {"count": 0, "pnl": 0.0})
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += c.pnl

    return {
        "instrument": instrument,
        "cutoff": cutoff_minutes or 0,
        "trades": report.total_trades,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "total_pnl": report.total_pnl,
        "sharpe": report.sharpe_ratio,
        "max_dd_pct": report.max_drawdown_pct,
        "avg_duration_min": float(np.mean(durations)) if durations else 0,
        "median_duration_min": float(np.median(durations)) if durations else 0,
        "reasons": reasons,
    }


async def main(
    from_date: str,
    to_date: str | None,
    cutoffs: list[int],
    initial_equity: float,
    slippage_pips: float,
    pairs: list[str] | None,
) -> None:
    instruments = pairs or QUALIFIED_PAIRS

    # Load per-pair params
    pair_params = None
    pp_path = Path("config/pair_params.yaml")
    if pp_path.exists():
        import yaml
        with open(pp_path) as f:
            pair_params = yaml.safe_load(f) or {}
        print(f"Loaded per-pair params ({len(pair_params)} pairs)")

    print(f"\nTime-Exit Duration Analysis")
    print(f"Pairs: {len(instruments)}")
    print(f"Period: {from_date} to {to_date or 'now'}")
    print(f"Cutoffs: {cutoffs} minutes (0 = no time exit / baseline)")
    print()

    # Run all combinations
    # cutoff_results[cutoff] = list of per-pair results
    cutoff_results: dict[int, list[dict]] = {c: [] for c in cutoffs}

    total_runs = len(instruments) * len(cutoffs)
    done = 0

    for instrument in instruments:
        for cutoff in cutoffs:
            done += 1
            cutoff_label = f"{cutoff}min" if cutoff > 0 else "baseline"
            print(f"  [{done}/{total_runs}] {instrument} @ {cutoff_label}...", end="", flush=True)

            result = await run_single(
                instrument=instrument,
                from_date=from_date,
                to_date=to_date,
                initial_equity=initial_equity,
                slippage_pips=slippage_pips,
                cutoff_minutes=cutoff if cutoff > 0 else None,
                pair_params=pair_params,
            )

            if result:
                cutoff_results[cutoff].append(result)
                print(f" {result['trades']} trades, PF={result['profit_factor']:.3f}")
            else:
                print(" no data")

    # ── Aggregate results per cutoff ──
    print("\n" + "=" * 110)
    print("  AGGREGATE RESULTS BY TIME-EXIT CUTOFF (all pairs combined)")
    print("=" * 110)
    header = (
        f"{'Cutoff':>8} {'Trades':>7} {'WinRate':>8} {'PF':>8} "
        f"{'TotalPnL':>12} {'AvgPnL':>10} {'AvgDur':>8} {'MedDur':>8} "
        f"{'TimeExits':>10} {'TE%':>6}"
    )
    print(header)
    print("-" * 110)

    best_pf = 0
    best_cutoff = 0

    for cutoff in cutoffs:
        results = cutoff_results[cutoff]
        if not results:
            continue

        total_trades = sum(r["trades"] for r in results)
        total_pnl = sum(r["total_pnl"] for r in results)
        gross_profit = sum(
            sum(p for p in [c.pnl for c in []] if p > 0)
            for r in results
        )

        # Weighted metrics
        weighted_wr = sum(r["win_rate"] * r["trades"] for r in results) / total_trades if total_trades else 0
        weighted_dur = sum(r["avg_duration_min"] * r["trades"] for r in results) / total_trades if total_trades else 0
        weighted_med_dur = sum(r["median_duration_min"] * r["trades"] for r in results) / total_trades if total_trades else 0

        # Aggregate PF from per-pair gross profit/loss
        agg_gross_profit = 0.0
        agg_gross_loss = 0.0
        time_exit_count = 0
        for r in results:
            for reason, data in r.get("reasons", {}).items():
                if data["pnl"] > 0:
                    agg_gross_profit += data["pnl"]
                else:
                    agg_gross_loss += data["pnl"]
                if reason == "time_exit":
                    time_exit_count += data["count"]

        agg_pf = agg_gross_profit / abs(agg_gross_loss) if agg_gross_loss != 0 else float("inf")
        avg_pnl = total_pnl / total_trades if total_trades else 0
        te_pct = time_exit_count / total_trades * 100 if total_trades else 0

        cutoff_label = f"{cutoff}min" if cutoff > 0 else "none"
        print(
            f"{cutoff_label:>8} {total_trades:>7} {weighted_wr:>7.1%} {agg_pf:>8.3f} "
            f"{total_pnl:>12,.2f} {avg_pnl:>10.4f} {weighted_dur:>7.1f}m {weighted_med_dur:>7.1f}m "
            f"{time_exit_count:>10} {te_pct:>5.1f}%"
        )

        if agg_pf > best_pf:
            best_pf = agg_pf
            best_cutoff = cutoff

    print("-" * 110)
    best_label = f"{best_cutoff}min" if best_cutoff > 0 else "none (no time exit)"
    print(f"  Best aggregate PF: {best_pf:.3f} at cutoff = {best_label}")
    print()

    # ── Per-pair breakdown for best cutoff ──
    if best_cutoff > 0:
        print(f"\n  PER-PAIR BREAKDOWN at {best_cutoff}min cutoff vs baseline")
        print("-" * 90)
        print(f"{'Pair':<10} {'Base PF':>8} {'Exit PF':>8} {'Diff':>8} {'Base Tr':>8} {'Exit Tr':>8} {'TE':>5}")
        print("-" * 90)

        baseline_by_pair = {r["instrument"]: r for r in cutoff_results.get(0, [])}
        exit_by_pair = {r["instrument"]: r for r in cutoff_results.get(best_cutoff, [])}

        for inst in instruments:
            base = baseline_by_pair.get(inst)
            exit_r = exit_by_pair.get(inst)
            if not base or not exit_r:
                continue

            te_count = exit_r.get("reasons", {}).get("time_exit", {}).get("count", 0)
            diff = exit_r["profit_factor"] - base["profit_factor"]
            marker = " +" if diff > 0 else ""
            print(
                f"{inst:<10} {base['profit_factor']:>8.3f} {exit_r['profit_factor']:>8.3f} "
                f"{marker}{diff:>7.3f} {base['trades']:>8} {exit_r['trades']:>8} {te_count:>5}"
            )

        print("-" * 90)

    # ── Close reason breakdown for baseline vs best ──
    print(f"\n  CLOSE REASON BREAKDOWN")
    for label, cutoff in [("Baseline (no time exit)", 0), (f"Best ({best_cutoff}min)", best_cutoff)]:
        if cutoff not in cutoff_results:
            continue
        results = cutoff_results[cutoff]
        agg_reasons: dict[str, dict] = {}
        for r in results:
            for reason, data in r.get("reasons", {}).items():
                agg_reasons.setdefault(reason, {"count": 0, "pnl": 0.0})
                agg_reasons[reason]["count"] += data["count"]
                agg_reasons[reason]["pnl"] += data["pnl"]

        total = sum(d["count"] for d in agg_reasons.values())
        print(f"\n  {label}:")
        for reason, data in sorted(agg_reasons.items(), key=lambda x: -x[1]["count"]):
            pct = data["count"] / total * 100 if total else 0
            avg = data["pnl"] / data["count"] if data["count"] else 0
            print(f"    {reason:20s}  {data['count']:5d} ({pct:5.1f}%)  total: {data['pnl']:+10.2f}  avg: {avg:+.4f}")

    print("\n" + "=" * 110)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Analyze optimal time-exit duration")
    parser.add_argument("-f", "--from-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-t", "--to-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument(
        "--cutoffs", nargs="+", type=int, default=DEFAULT_CUTOFFS,
        help="Time-exit cutoffs in minutes to test (0 = baseline)",
    )
    parser.add_argument("--pairs", nargs="+", default=None, help="Override pair list")

    args = parser.parse_args()
    setup_logging(level="WARNING", log_format="console")

    # Ensure 0 (baseline) is always included
    cutoffs = sorted(set([0] + args.cutoffs))

    asyncio.run(main(
        from_date=args.from_date,
        to_date=args.to_date,
        cutoffs=cutoffs,
        initial_equity=args.equity,
        slippage_pips=args.slippage,
        pairs=args.pairs,
    ))


if __name__ == "__main__":
    cli()
