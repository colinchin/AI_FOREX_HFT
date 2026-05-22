"""Locked-parameters broker-cost experiment.

A simpler, purer controlled experiment than the walk-forward version:

  * **Single, locked strategy config** — uses ``config/settings.yaml``'s
    documented defaults (bb_std=2.0, rsi 35/65, sl_multiplier=2.5). No grid
    search, no per-fold parameter selection.
  * **One continuous backtest per pair** over the full 2022-01-01 → now
    period. No folds, no walk-forward refitting.
  * **Only the cost model varies** — OANDA-v2 (full round-trip spread, no
    commission) vs ECN (ECN spreads + 0.8 bps commission). Both go through
    the same ``CostAwareSimulator``.

This removes the grid-search variance that contaminated the walk-forward
smoke (where different cost models picked different best-on-train params,
flipping aggregate OOS PFs even when same-params per-fold PFs consistently
favoured ECN). The locked-params comparison is direct: same signal, same
data, same windows — only broker cost differs.

Output format
-------------
Per-pair JSON to ``data/locked_params_<oanda|ecn>/<PAIR>.json``::

    {
      "instrument": "EUR_USD",
      "cost_model": "ecn",
      "config": { strategy params + costs + period },
      "metrics": {
         "total_trades", "total_pnl",
         "sharpe_per_trade", "sharpe_annualised",
         "profit_factor", "win_rate", "max_drawdown_pct",
         "skew", "kurtosis"
      },
      "pnls": [ ... ]  # full per-trade PnL series in quote currency
    }

Usage
-----
    # Both cost models concurrently:
    python scripts/run_locked_params_experiment.py --cost-model oanda \\
        -f 2022-01-01 -o data/locked_params_oanda
    python scripts/run_locked_params_experiment.py --cost-model ecn \\
        -f 2022-01-01 -o data/locked_params_ecn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.cost_model import (
    COMMISSION_BPS_ROUND_TRIP,
    CostAwareSimulator,
    ecn_spread_for,
)
from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.logger import setup_logging


# Documented strategy defaults from config/settings.yaml lines 59-64.
LOCKED_PARAMS = {
    "bb_std": 2.0,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "sl_multiplier": 2.5,
}


# ── Metrics helpers ──────────────────────────────────────────────────────────


def _per_trade_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls, dtype=float)
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd > 0 else 0.0


def _annualised_sharpe(pnls: list[float], total_days: float) -> float:
    if len(pnls) < 2 or total_days <= 0:
        return 0.0
    sr_pt = _per_trade_sharpe(pnls)
    trades_per_year = len(pnls) * (365.0 / total_days)
    return sr_pt * float(np.sqrt(trades_per_year))


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
    if len(pnls) < 4:
        return 0.0, 0.0
    arr = np.array(pnls, dtype=float)
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0
    z = (arr - mu) / sd
    n = len(arr)
    g1 = (n / ((n - 1) * (n - 2))) * (z ** 3).sum()
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (z ** 4).sum() \
         - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(g1), float(g2)


# ── One-pair runner ──────────────────────────────────────────────────────────


async def _run_pair(
    instrument: str,
    spread_pips: float,
    cost_model: str,
    slippage_pips: float,
    equity: float,
    strategy_config_base: dict,
    full_config: dict,
    from_dt: datetime,
    to_dt: datetime,
    conversion_series,
) -> dict | None:
    config = load_config()
    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    try:
        m5_df = loader.load_candles(
            instrument, "M5",
            from_dt.strftime("%Y-%m-%d"),
            to_dt.strftime("%Y-%m-%d"),
        )
    except FileNotFoundError:
        return None
    if m5_df.empty:
        return None
    m5_events = loader.df_to_candle_events(m5_df, instrument, "M5")

    h1_events = None
    try:
        h1_df = loader.load_candles(
            instrument, "H1",
            from_dt.strftime("%Y-%m-%d"),
            to_dt.strftime("%Y-%m-%d"),
        )
        if not h1_df.empty:
            h1_events = loader.df_to_candle_events(h1_df, instrument, "H1")
    except FileNotFoundError:
        pass

    # Lock strategy config to the documented defaults
    strat_cfg = dict(strategy_config_base)
    mr = dict(strat_cfg.get("mean_reversion", strat_cfg))
    mr.update(LOCKED_PARAMS)
    strat_cfg["mean_reversion"] = mr

    # Build the cost-aware simulator
    if cost_model == "ecn":
        sim_spread = ecn_spread_for(spread_pips)
        commission_bps = COMMISSION_BPS_ROUND_TRIP
    else:
        sim_spread = spread_pips
        commission_bps = 0.0

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
        strategy=strategy,
        simulator=simulator,
        config=full_config,
        multi_tf_filter=mtf,
        account_currency=full_config.get("account_currency", "AUD"),
        conversion_cache=conversion_cache,
    )

    await engine.run(m5_events, h1_events)
    pnls = [float(c.pnl) for c in simulator.closes]

    total_days = (to_dt - from_dt).total_seconds() / 86400.0
    skew, kurt = _skew_kurt(pnls)

    return {
        "instrument": instrument,
        "cost_model": cost_model,
        "config": {
            "from_date": from_dt.strftime("%Y-%m-%d"),
            "to_date": to_dt.strftime("%Y-%m-%d"),
            "total_days": round(total_days, 1),
            "locked_params": LOCKED_PARAMS,
            "spread_pips": sim_spread,
            "oanda_spread_pips": spread_pips,
            "commission_bps_round_trip": commission_bps,
            "slippage_pips": slippage_pips,
            "initial_equity": equity,
        },
        "metrics": {
            "total_trades": len(pnls),
            "total_pnl": float(sum(pnls)),
            "sharpe_per_trade": _per_trade_sharpe(pnls),
            "sharpe_annualised": _annualised_sharpe(pnls, total_days),
            "profit_factor": _profit_factor(pnls),
            "win_rate": (
                float(np.mean([p > 0 for p in pnls])) if pnls else 0.0
            ),
            "max_drawdown_pct": _max_drawdown_pct(pnls, equity),
            "skew": skew,
            "kurtosis": kurt,
        },
        "pnls": pnls,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_all(
    instruments: list[str], cost_model: str,
    from_date: str, to_date: str | None,
    equity: float, slippage_pips: float, output_dir: str,
) -> None:
    config = load_config()
    base_strategy_config = config.strategy
    full_config = config.raw

    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_series = load_conversion_series(data_dir)

    from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)
    to_dt = (
        datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc)
        if to_date else datetime.now(timezone.utc)
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLocked-params {cost_model.upper()} run")
    print(f"  Pairs:        {len(instruments)}")
    print(f"  Period:       {from_date} -> {to_date or 'now'} ({(to_dt-from_dt).days} days)")
    print(f"  Locked params: {LOCKED_PARAMS}")
    print(f"  Cost model:   {cost_model}")
    if cost_model == "ecn":
        print(f"                ECN raw spreads (max(0.2, OANDA*0.30) pips) + {COMMISSION_BPS_ROUND_TRIP} bps commission")
    else:
        print(f"                OANDA spreads (corrected full round-trip)")
    print(f"  Output dir:   {output_dir}")
    print()

    start_time = time.monotonic()
    for idx, instrument in enumerate(instruments, 1):
        oanda_spread = SPREAD_TABLE.get(instrument, 2.0)
        t0 = time.monotonic()
        result = await _run_pair(
            instrument=instrument,
            spread_pips=oanda_spread,
            cost_model=cost_model,
            slippage_pips=slippage_pips,
            equity=equity,
            strategy_config_base=base_strategy_config,
            full_config=full_config,
            from_dt=from_dt,
            to_dt=to_dt,
            conversion_series=conversion_series,
        )
        elapsed = time.monotonic() - t0
        if result is None:
            print(f"  [{idx}/{len(instruments)}] {instrument}: no data, skipped ({elapsed:.0f}s)")
            continue
        m = result["metrics"]
        print(
            f"  [{idx}/{len(instruments)}] {instrument}: "
            f"trades={m['total_trades']:5d}  PF={m['profit_factor']:>6.3f}  "
            f"SR_pt={m['sharpe_per_trade']:+.4f}  PnL={m['total_pnl']:>+10.2f}  "
            f"({elapsed:.0f}s)"
        )
        # Persist (strip pnls list to keep file size sane for 13k-trade pairs)
        out_file = out_path / f"{instrument}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, default=str, indent=2)

    total_min = (time.monotonic() - start_time) / 60.0
    print(f"\nTotal time: {total_min:.1f} minutes")
    print(f"Output: {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked-parameters broker-cost experiment (no grid search).",
    )
    parser.add_argument("--cost-model", choices=("oanda", "ecn"), required=True)
    parser.add_argument("-f", "--from-date", required=True)
    parser.add_argument("-t", "--to-date", default=None)
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--pairs", nargs="+", default=None)
    args = parser.parse_args()

    setup_logging(level="WARNING", log_format="console")

    instruments = args.pairs or G10_PAIRS
    output_dir = args.output_dir or (
        "data/locked_params_ecn" if args.cost_model == "ecn"
        else "data/locked_params_oanda"
    )

    asyncio.run(run_all(
        instruments=instruments,
        cost_model=args.cost_model,
        from_date=args.from_date,
        to_date=args.to_date,
        equity=args.equity,
        slippage_pips=args.slippage,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
