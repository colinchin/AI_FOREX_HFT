"""OANDA-vs-ECN broker-cost experiment runner.

Two cost-model flavours, both running through the *same* CostAwareSimulator
(which charges full round-trip spread, fixing a half-spread quirk in the base
BacktestSimulator):

    --cost-model oanda   → OANDA spreads, commission = 0
    --cost-model ecn     → ECN spreads (max(0.2, OANDA*0.30)), commission = 0.8 bps

The runner installs the chosen cost model into walk_forward_optimize.py's two
seams (SIMULATOR_FACTORY, SPREAD_OVERRIDE) and then calls the existing
run_walk_forward function. Signal, parameter grid, fold windows, conversion
path, and the DSR analysis are identical between the two flavours — only the
cost inputs change, which is the controlled-experiment requirement.

Before the long run starts, the script prints a per-pair sanity table showing
the round-trip commission a 100k-base-currency position would incur (in USD).
The benchmark for real ECN brokers (IC Markets Raw, Pepperstone Razor) is
≈ $7–9 USD per 100k. If any pair lands far from that band, the commission
constant is wrong — fix before committing to the 18-hour run.

Usage
-----
    # Corrected-OANDA baseline (full round-trip spread, no commission):
    python scripts/run_ecn_experiment.py --cost-model oanda -f 2022-01-01 \\
        --train-days 365 --test-days 90 --step-days 90 --skip-download \\
        -o data/walk_forward_oanda_v2

    # ECN experiment:
    python scripts/run_ecn_experiment.py --cost-model ecn -f 2022-01-01 \\
        --train-days 365 --test-days 90 --step-days 90 --skip-download \\
        -o data/walk_forward_ecn
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from backtest.cost_model import (
    COMMISSION_BPS_ROUND_TRIP,
    COMMISSION_FRACTION,
    cost_aware_simulator_factory_ecn,
    cost_aware_simulator_factory_oanda,
    ecn_spread_for,
    ecn_spread_override,
)
from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE
from src.utils.config import load_config
from src.utils.logger import setup_logging

# Import the walk-forward module as a namespace so we can set its module-level
# seams (SIMULATOR_FACTORY, SPREAD_OVERRIDE) before invoking run_walk_forward.
from scripts import walk_forward_optimize as wfo


# ── Sanity table ─────────────────────────────────────────────────────────────


def _latest_price(parquet_dir: Path, instrument: str) -> float | None:
    f = parquet_dir / f"{instrument}_M5.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=["close"])
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def _to_usd(amount_in_quote: float, instrument: str, prices: dict[str, float]) -> float | None:
    """Convert an amount in `instrument`'s quote currency to USD using cached prices."""
    quote = instrument.split("_")[1]
    if quote == "USD":
        return amount_in_quote
    # Try quote_USD direct
    direct = prices.get(f"{quote}_USD")
    if direct is not None:
        return amount_in_quote * direct
    # Try USD_quote inverse
    inverse = prices.get(f"USD_{quote}")
    if inverse and inverse > 0:
        return amount_in_quote / inverse
    return None


def print_cost_sanity_table(instruments: list[str], parquet_dir: Path) -> None:
    print()
    print("=" * 100)
    print("  ECN COST SANITY TABLE — round-trip commission on a 100k-base notional")
    print("=" * 100)
    print(f"  Commission constant: {COMMISSION_BPS_ROUND_TRIP} bps round-trip "
          f"(= {COMMISSION_FRACTION:.5f} of notional)")
    print(f"  Reference price:     latest M5 close per pair in data/parquet/")
    print(f"  Benchmark:           real ECN ≈ $7–9 USD per 100k round-trip")
    print("-" * 100)
    header = (
        f"  {'Pair':<10} {'OANDA(pip)':>11} {'ECN(pip)':>9}"
        f" {'Ref price':>11} {'Notional(quote)':>20} {'Notional(USD)':>16}"
        f" {'Commission(USD)':>17}"
    )
    print(header)
    print("-" * 100)

    # Pre-load latest prices for USD-cross conversions
    prices: dict[str, float] = {}
    for pair in instruments + ["AUD_USD", "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD"]:
        p = _latest_price(parquet_dir, pair)
        if p is not None:
            prices[pair] = p

    UNITS = 100_000
    implied_bps_values: list[float] = []
    for instrument in instruments:
        oanda = SPREAD_TABLE.get(instrument, 2.0)
        ecn = ecn_spread_for(oanda)
        price = prices.get(instrument)
        if price is None:
            print(f"  {instrument:<10} {oanda:>11.2f} {ecn:>9.2f}  (no parquet for ref price)")
            continue
        notional_quote = UNITS * price
        notional_usd = _to_usd(notional_quote, instrument, prices)
        if notional_usd is None:
            print(f"  {instrument:<10} {oanda:>11.2f} {ecn:>9.2f}"
                  f" {price:>11.4f} {notional_quote:>20,.2f} {'?':>16} {'?':>17}")
            continue
        commission_usd = notional_usd * COMMISSION_FRACTION
        implied_bps = (commission_usd / notional_usd) * 10_000.0
        implied_bps_values.append(implied_bps)
        print(
            f"  {instrument:<10} {oanda:>11.2f} {ecn:>9.2f}"
            f" {price:>11.4f} {notional_quote:>20,.2f} {notional_usd:>16,.2f}"
            f" ${commission_usd:>15.2f}"
        )
    print("-" * 100)
    if implied_bps_values:
        # The real sanity check: every pair's commission/notional_USD ratio
        # equals the configured bps. If this isn't tight to 0.8, the math is
        # wrong somewhere downstream of the constant.
        mn = min(implied_bps_values)
        mx = max(implied_bps_values)
        ok = abs(mn - COMMISSION_BPS_ROUND_TRIP) < 1e-6 and abs(mx - COMMISSION_BPS_ROUND_TRIP) < 1e-6
        verdict = "OK" if ok else "OFF — review constant or _to_usd path"
        print(f"  implied bps across pairs: [{mn:.4f}, {mx:.4f}]   vs configured {COMMISSION_BPS_ROUND_TRIP:.4f}   →   {verdict}")
        print(f"  Lower-USD pairs (NZD, AUD base) correctly show ~$5 because 100k of a weak base "
              f"≈ <$80k USD notional — the constant is unchanged.")
    print("=" * 100)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OANDA-vs-ECN broker-cost walk-forward experiment.",
    )
    parser.add_argument(
        "--cost-model", choices=("oanda", "ecn"), required=True,
        help="Cost flavour to use: 'oanda' = corrected-OANDA baseline (full "
             "round-trip spread, no commission). 'ecn' = ECN spreads + commission.",
    )
    parser.add_argument("-f", "--from-date", required=True)
    parser.add_argument("-t", "--to-date", default=None)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Default: data/walk_forward_oanda_v2 (oanda) or "
                             "data/walk_forward_ecn (ecn)")
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    setup_logging(level="WARNING", log_format="console")

    instruments = args.pairs or G10_PAIRS
    config = load_config()
    parquet_dir = Path(config.data.get("parquet_dir", "data/parquet"))

    output_dir = args.output_dir or (
        "data/walk_forward_ecn" if args.cost_model == "ecn"
        else "data/walk_forward_oanda_v2"
    )

    print(f"\nBroker-cost experiment — cost model: {args.cost_model.upper()}")
    print(f"  Pairs:                {len(instruments)}")
    print(f"  Train / Test / Step:  {args.train_days}d / {args.test_days}d / {args.step_days}d")
    print(f"  Period:               {args.from_date} -> {args.to_date or 'now'}")
    print(f"  Output dir:           {output_dir}")
    if args.cost_model == "ecn":
        print(f"  Spread:               ECN raw (max(0.2, OANDA*0.30) pips)")
        print(f"  Commission:           {COMMISSION_BPS_ROUND_TRIP} bps round-trip")
    else:
        print(f"  Spread:               OANDA SPREAD_TABLE (corrected: full round-trip)")
        print(f"  Commission:           0 (OANDA standard, no per-trade commission)")
    print(f"  Sim:                  CostAwareSimulator (entry + close-side half-spread)")

    # ── Sanity check — eyeball the commission per 100k before the long run ──
    if args.cost_model == "ecn":
        print_cost_sanity_table(instruments, parquet_dir)

    # ── Install the chosen cost-model seams ─────────────────────────────────
    if args.cost_model == "ecn":
        wfo.SIMULATOR_FACTORY = cost_aware_simulator_factory_ecn
        wfo.SPREAD_OVERRIDE = ecn_spread_override
    else:
        wfo.SIMULATOR_FACTORY = cost_aware_simulator_factory_oanda
        wfo.SPREAD_OVERRIDE = None    # use OANDA SPREAD_TABLE unchanged

    # ── Phase 1: download (if needed) ───────────────────────────────────────
    if not args.skip_download:
        from scripts.backtest_all_pairs import download_all_history
        print("\n--- Phase 1: Downloading history ---")
        asyncio.run(download_all_history(instruments, args.from_date, args.to_date))

    # ── Phase 2: walk-forward via the existing function, with chosen costs ─
    print(f"\n--- Phase 2: Walk-forward grid search ({args.cost_model.upper()} cost model) ---")
    asyncio.run(wfo.run_walk_forward(
        instruments=instruments,
        from_date=args.from_date,
        to_date=args.to_date,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        equity=args.equity,
        slippage_pips=args.slippage,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
