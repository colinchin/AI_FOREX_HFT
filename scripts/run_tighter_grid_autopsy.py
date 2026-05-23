"""Tighter-grid autopsy — final check on whether ANY grid combination produces
a signal that survives realistic costs.

The original 100-combo walk-forward + DSR run was deflated against 1300 trials
per pair, which is brutal. The original brief noted that if everything REJECTS
under that bar, a tighter 24-combo grid may produce a more tractable DSR test.
After the corrected-spread experiment showed all 28 pairs REJECT under the
locked settings.yaml defaults, this script runs the brief's tighter grid as
an autopsy: for each pair, try all 24 combos under corrected spreads + ECN
costs, pick the best, and ask "did ANYTHING in the tighter grid produce a
signal?".

Two reads are reported per pair:
  * PSR(0) on the best combo (no correction) — the most generous statistical
    read; useful for "is there even a hint of signal".
  * DSR with N_trials=24 (selection-bias-corrected) — the honest test.

Output per pair (data/autopsy/<PAIR>.json):
  {
    "instrument": "EUR_USD",
    "grid": [ {params, metrics}, ... ]  # all 24 combos
    "best_by_pf": { ... },
    "best_by_sharpe": { ... },
  }

This is read-only on the broker side (uses cached parquet); no orders.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.cost_model import (
    COMMISSION_BPS_ROUND_TRIP,
    CostAwareSimulator,
    ecn_spread_for,
)
from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE
from scripts.run_locked_params_experiment import (
    _annualised_sharpe, _max_drawdown_pct, _per_trade_sharpe,
    _profit_factor, _skew_kurt, _load_spread_overrides,
)
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.logger import setup_logging


# Brief's tighter grid (24 combos).
GRID = {
    "bb_std": [2.0, 2.25, 2.5],                # 3
    "rsi_thresholds": [(25, 75), (30, 70)],    # 2
    "sl_multiplier": [2.0, 2.5, 3.0, 3.5],     # 4
}


async def _run_one_combo(
    instrument: str, m5_events, h1_events,
    base_strategy_config: dict, full_config: dict,
    spread_pips: float, slippage_pips: float, equity: float,
    cost_model: str, conversion_series,
    bb_std: float, rsi_os: int, rsi_ob: int, sl_mult: float,
) -> dict:
    strat_cfg = dict(base_strategy_config)
    mr = dict(strat_cfg.get("mean_reversion", strat_cfg))
    mr.update({
        "bb_std": bb_std,
        "rsi_oversold": rsi_os,
        "rsi_overbought": rsi_ob,
        "sl_multiplier": sl_mult,
    })
    strat_cfg["mean_reversion"] = mr

    sim_spread = ecn_spread_for(spread_pips) if cost_model == "ecn" else spread_pips
    commission_bps = COMMISSION_BPS_ROUND_TRIP if cost_model == "ecn" else 0.0

    conversion_cache = ConversionRateCache(conversion_series)
    strategy = MeanReversionStrategy(config=strat_cfg)
    simulator = CostAwareSimulator(
        initial_equity=equity,
        spread_pips=sim_spread,
        slippage_pips=slippage_pips,
        commission_bps_round_trip=commission_bps,
        conversion_cache=conversion_cache,
    )
    mtf = MultiTimeframeFilter() if h1_events else None
    engine = BacktestEngine(
        strategy=strategy, simulator=simulator, config=full_config,
        multi_tf_filter=mtf,
        account_currency=full_config.get("account_currency", "AUD"),
        conversion_cache=conversion_cache,
    )
    await engine.run(m5_events, h1_events)
    pnls = [float(c.pnl) for c in simulator.closes]
    skew, kurt = _skew_kurt(pnls)
    return {
        "params": {
            "bb_std": bb_std, "rsi_oversold": rsi_os,
            "rsi_overbought": rsi_ob, "sl_multiplier": sl_mult,
        },
        "metrics": {
            "total_trades": len(pnls),
            "total_pnl": float(sum(pnls)),
            "sharpe_per_trade": _per_trade_sharpe(pnls),
            "profit_factor": _profit_factor(pnls),
            "win_rate": float(np.mean([p > 0 for p in pnls])) if pnls else 0.0,
            "max_drawdown_pct": _max_drawdown_pct(pnls, equity),
            "skew": skew, "kurtosis": kurt,
        },
    }


async def run_pair_autopsy(
    instrument: str, cost_model: str,
    from_dt: datetime, to_dt: datetime,
    equity: float, slippage_pips: float,
    spread_overrides: dict[str, float],
    output_dir: str,
) -> None:
    config = load_config()
    base_strategy_config = config.strategy
    full_config = config.raw
    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_series = load_conversion_series(data_dir)
    loader = BacktestDataLoader(data_dir)

    oanda_spread = spread_overrides.get(instrument, SPREAD_TABLE.get(instrument, 2.0))

    try:
        m5_df = loader.load_candles(instrument, "M5",
            from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        print(f"  {instrument}: no M5 data, skip"); return
    if m5_df.empty:
        print(f"  {instrument}: empty M5 data, skip"); return
    m5_events = loader.df_to_candle_events(m5_df, instrument, "M5")

    h1_events = None
    try:
        h1_df = loader.load_candles(instrument, "H1",
            from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d"))
        if not h1_df.empty:
            h1_events = loader.df_to_candle_events(h1_df, instrument, "H1")
    except FileNotFoundError:
        pass

    combos = list(itertools.product(
        GRID["bb_std"], GRID["rsi_thresholds"], GRID["sl_multiplier"],
    ))
    print(f"  {instrument}: {len(combos)} combos, oanda_spread={oanda_spread} pips, cost_model={cost_model}")

    results = []
    t0 = time.monotonic()
    for bb_std, (rsi_os, rsi_ob), sl_mult in combos:
        r = await _run_one_combo(
            instrument, m5_events, h1_events,
            base_strategy_config, full_config,
            oanda_spread, slippage_pips, equity, cost_model, conversion_series,
            bb_std, rsi_os, rsi_ob, sl_mult,
        )
        results.append(r)

    # Best picks
    best_pf = max(results, key=lambda r: r["metrics"]["profit_factor"])
    best_sr = max(results, key=lambda r: r["metrics"]["sharpe_per_trade"])

    out = {
        "instrument": instrument,
        "cost_model": cost_model,
        "grid_size": len(combos),
        "config": {
            "from_date": from_dt.strftime("%Y-%m-%d"),
            "to_date": to_dt.strftime("%Y-%m-%d"),
            "oanda_spread_pips": oanda_spread,
            "commission_bps_round_trip": COMMISSION_BPS_ROUND_TRIP if cost_model == "ecn" else 0.0,
        },
        "grid": results,
        "best_by_pf": best_pf,
        "best_by_sharpe": best_sr,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{instrument}.json", "w") as f:
        json.dump(out, f, default=str, indent=2)

    elapsed = time.monotonic() - t0
    print(f"  {instrument}: done in {elapsed:.0f}s | "
          f"bestPF={best_pf['metrics']['profit_factor']:.3f}@{best_pf['params']} | "
          f"bestSR={best_sr['metrics']['sharpe_per_trade']:+.4f}@{best_sr['params']}")


async def run_all(
    instruments: list[str], cost_model: str,
    from_date: str, to_date: str | None,
    equity: float, slippage_pips: float,
    spread_overrides: dict[str, float], output_dir: str,
) -> None:
    from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)
    to_dt = (datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc)
             if to_date else datetime.now(timezone.utc))
    print(f"\nTighter-grid autopsy — {cost_model.upper()}")
    print(f"  Pairs:   {len(instruments)}  Grid: {len(GRID['bb_std']) * len(GRID['rsi_thresholds']) * len(GRID['sl_multiplier'])} combos")
    print(f"  Period:  {from_date} -> {to_date or 'now'} ({(to_dt-from_dt).days} days)")
    print(f"  Output:  {output_dir}\n")
    for instrument in instruments:
        await run_pair_autopsy(
            instrument, cost_model, from_dt, to_dt,
            equity, slippage_pips, spread_overrides, output_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tighter-grid (24-combo) autopsy.")
    parser.add_argument("--cost-model", choices=("oanda", "ecn"), required=True)
    parser.add_argument("-f", "--from-date", required=True)
    parser.add_argument("-t", "--to-date", default=None)
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--spread-table-yaml", default=None)
    args = parser.parse_args()

    setup_logging(level="WARNING", log_format="console")
    instruments = args.pairs or G10_PAIRS
    output_dir = args.output_dir or f"data/autopsy_{args.cost_model}"
    overrides = _load_spread_overrides(args.spread_table_yaml)

    asyncio.run(run_all(
        instruments=instruments,
        cost_model=args.cost_model,
        from_date=args.from_date,
        to_date=args.to_date,
        equity=args.equity,
        slippage_pips=args.slippage,
        spread_overrides=overrides,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
