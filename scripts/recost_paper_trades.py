"""Re-cost paper-trade OANDA trades to ECN equivalents and recompute PSR(0).

What this does
--------------
The system trades on OANDA Practice with OANDA's bid/ask spreads embedded in
fill prices (full round-trip spread cost). The locked-params backtest under
ECN (in ``data/locked_params_ecn/<PAIR>.json``) used the corrected ECN cost
model (full round-trip ECN spread + 0.8 bps commission). To compare the
paper-trade OOS data against the backtest cleanly, we re-cost each paper-
traded trade to an ECN equivalent:

    spread_savings_quote = (OANDA_pips - ECN_pips) * pip_value * units
    commission_quote     = units * entry_price * 0.00008
    recosted_pnl         = recorded_pnl + spread_savings_quote - commission_quote

Then we combine the re-costed paper-trade PnLs with the locked-params backtest
OOS PnLs, recompute PSR(0), and report the verdict.

Verdict thresholds (mirroring the locked-params experiment):
    PSR(0) ≥ 0.95 → STRONG    — live-money pilot at minimum size
    PSR(0) ≥ 0.85 → MARGINAL  — extend paper-trade another 90 days
    PSR(0) <  0.85 → DECAYED  — escalate to the regime diagnostic

Usage
-----
    python scripts/recost_paper_trades.py \\
        --instrument GBP_NZD \\
        --since 2026-05-23 \\
        --combine-with data/locked_params_ecn/GBP_NZD.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.cost_model import COMMISSION_FRACTION, ecn_spread_for
from scripts.backtest_all_pairs import SPREAD_TABLE
from src.utils.helpers import pip_value


def psr(sr_hat: float, sr_star: float, n: int, skew: float, kurt_non_excess: float) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)."""
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + ((kurt_non_excess - 1.0) / 4.0) * sr_hat ** 2
    if denom_sq <= 0:
        return 0.5
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def _sample_moments(pnls: np.ndarray) -> tuple[float, float, float]:
    """Return per-trade Sharpe, sample skewness, excess kurtosis."""
    if len(pnls) < 2:
        return 0.0, 0.0, 0.0
    mu = pnls.mean()
    sd = pnls.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0, 0.0
    sr = float(mu / sd)
    if len(pnls) < 4:
        return sr, 0.0, 0.0
    z = (pnls - mu) / sd
    n = len(pnls)
    g1 = (n / ((n - 1) * (n - 2))) * (z ** 3).sum()
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (z ** 4).sum() \
         - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return sr, float(g1), float(g2)


def fetch_paper_trades(db_path: str, instrument: str, since: str) -> list[dict]:
    """Pull all closed trades for the instrument since the given ISO date."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT trade_id, instrument, direction, units, entry_price, exit_price,
                  entry_time, exit_time, pnl, exit_reason
           FROM trades
           WHERE instrument = ? AND entry_time >= ?
           ORDER BY entry_time""",
        (instrument, since),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def recost_to_ecn(trade: dict, oanda_spread_pips: float, ecn_spread_pips: float) -> float:
    """Convert an OANDA-traded PnL into the ECN-equivalent PnL (quote currency).

    Assumes the recorded PnL has OANDA's full round-trip bid/ask spread cost
    already embedded (which is how OANDA actually charges on a live order).

    spread_savings = (OANDA_pips - ECN_pips) * pip_value * units
    commission     = units * entry_price * COMMISSION_FRACTION  (round-trip)
    """
    units = int(trade["units"])
    entry_price = float(trade["entry_price"])
    recorded_pnl = float(trade["pnl"])
    pv = pip_value(trade["instrument"])
    spread_savings = (oanda_spread_pips - ecn_spread_pips) * pv * units
    commission = units * entry_price * COMMISSION_FRACTION
    return recorded_pnl + spread_savings - commission


def verdict(psr0: float) -> str:
    if psr0 >= 0.95:
        return "STRONG"
    if psr0 >= 0.85:
        return "MARGINAL"
    return "DECAYED"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-cost paper-trade OANDA trades to ECN, recompute PSR(0)."
    )
    parser.add_argument("--instrument", required=True, help="e.g. GBP_NZD")
    parser.add_argument(
        "--since", required=True,
        help="ISO date — only trades with entry_time >= this are included",
    )
    parser.add_argument(
        "--combine-with", default=None,
        help="Path to backtest JSON (data/locked_params_ecn/<PAIR>.json). "
             "If provided, the backtest pnls are combined with the re-costed "
             "paper-trade pnls before PSR(0) is computed.",
    )
    parser.add_argument("--db", default="data/trades.db")
    args = parser.parse_args()

    oanda_spread = SPREAD_TABLE.get(args.instrument)
    if oanda_spread is None:
        print(f"ERROR: {args.instrument} not in SPREAD_TABLE.")
        sys.exit(1)
    ecn_spread = ecn_spread_for(oanda_spread)

    print("\n" + "=" * 78)
    print(f"  RE-COST PAPER-TRADE → ECN — {args.instrument}")
    print("=" * 78)
    print(f"  Source DB:     {args.db}")
    print(f"  Trades since:  {args.since}")
    print(f"  OANDA spread:  {oanda_spread:.2f} pips")
    print(f"  ECN spread:    {ecn_spread:.2f} pips  (savings {oanda_spread - ecn_spread:+.2f} pips round-trip)")
    print(f"  Commission:    {COMMISSION_FRACTION*10_000:.2f} bps round-trip")
    print()

    paper_trades = fetch_paper_trades(args.db, args.instrument, args.since)
    if not paper_trades:
        print(f"  No paper-trade closes found for {args.instrument} since {args.since}.")
        if args.combine_with:
            print("  Cannot compute combined PSR(0) without paper-trade data.")
        sys.exit(0)

    print(f"  Paper-trade closes: {len(paper_trades)}")

    # Re-cost each paper trade
    recosted_pnls = []
    for t in paper_trades:
        new_pnl = recost_to_ecn(t, oanda_spread, ecn_spread)
        recosted_pnls.append(new_pnl)
    recosted = np.array(recosted_pnls, dtype=float)

    # Forward-only stats — the cleanest OOS signal
    sr_paper, skew_paper, kurt_paper = _sample_moments(recosted)
    psr_paper = psr(sr_paper, 0.0, len(recosted), skew_paper, kurt_paper + 3.0)
    v_paper = verdict(psr_paper)

    # Backtest + combined stats (only if --combine-with provided)
    bt_pnls = None
    sr_bt = skew_bt = kurt_bt = psr_bt = None
    sr_c = skew_c = kurt_c = psr_c = None
    v_combined = None
    backtest_meta = None
    if args.combine_with:
        backtest_path = Path(args.combine_with)
        if not backtest_path.exists():
            print(f"\n  ERROR: --combine-with file not found: {backtest_path}")
            sys.exit(1)
        backtest = json.load(open(backtest_path))
        backtest_meta = backtest
        bt_pnls = np.array(backtest["pnls"], dtype=float)
        sr_bt, skew_bt, kurt_bt = _sample_moments(bt_pnls)
        psr_bt = psr(sr_bt, 0.0, len(bt_pnls), skew_bt, kurt_bt + 3.0)
        combined = np.concatenate([bt_pnls, recosted])
        sr_c, skew_c, kurt_c = _sample_moments(combined)
        psr_c = psr(sr_c, 0.0, len(combined), skew_c, kurt_c + 3.0)
        v_combined = verdict(psr_c)

    # ── Side-by-side report ─────────────────────────────────────────────────
    print()
    print("  " + "-" * 76)
    if args.combine_with:
        print(f"  {'Metric':<22} {'Backtest OOS':>14} {'Forward (paper)':>17} {'Combined':>14}")
        print("  " + "-" * 76)
        print(f"  {'Trades':<22} {len(bt_pnls):>14} {len(recosted):>17} {len(combined):>14}")
        print(f"  {'Total PnL (quote)':<22} {bt_pnls.sum():>+14.2f} {recosted.sum():>+17.2f} {combined.sum():>+14.2f}")
        print(f"  {'SR/trade':<22} {sr_bt:>+14.4f} {sr_paper:>+17.4f} {sr_c:>+14.4f}")
        print(f"  {'Skew':<22} {skew_bt:>+14.2f} {skew_paper:>+17.2f} {skew_c:>+14.2f}")
        print(f"  {'Excess kurtosis':<22} {kurt_bt:>+14.2f} {kurt_paper:>+17.2f} {kurt_c:>+14.2f}")
        print(f"  {'PSR(0)':<22} {psr_bt:>14.4f} {psr_paper:>17.4f} {psr_c:>14.4f}")
        print(f"  {'Verdict':<22} {verdict(psr_bt):>14} {v_paper:>17} {v_combined:>14}")
    else:
        print(f"  {'Metric':<22} {'Forward (paper)':>17}")
        print("  " + "-" * 76)
        print(f"  {'Trades':<22} {len(recosted):>17}")
        print(f"  {'Total PnL (quote)':<22} {recosted.sum():>+17.2f}")
        print(f"  {'SR/trade':<22} {sr_paper:>+17.4f}")
        print(f"  {'Skew':<22} {skew_paper:>+17.2f}")
        print(f"  {'Excess kurtosis':<22} {kurt_paper:>+17.2f}")
        print(f"  {'PSR(0)':<22} {psr_paper:>17.4f}")
        print(f"  {'Verdict':<22} {v_paper:>17}")
    print("  " + "-" * 76)

    # ── Interpretation ──────────────────────────────────────────────────────
    print()
    print("  Forward-only PSR(0) is the cleanest read on whether the backtest edge")
    print("  held out-of-sample. Combined PSR(0) is dominated by the much larger")
    print("  backtest sample (≈20x more trades) — useful for total-evidence sizing")
    print("  but masks short-run decay.")
    print()
    print(f"  PRIMARY VERDICT (forward-only): {v_paper}")
    if args.combine_with:
        print(f"  Combined verdict:               {v_combined}")
    print()
    if v_paper == "STRONG":
        print("  → Forward-only is STRONG. Eligible for a live-money pilot at an ECN")
        print("    broker (minimum size — 1 unit, not 1% risk).")
    elif v_paper == "MARGINAL":
        print("  → Forward-only is MARGINAL. Extend paper-trade another 90 days and")
        print("    re-evaluate. Do not size up.")
    else:
        print("  → Forward-only DECAYED — the 4-year backtest edge did not hold OOS.")
        print("    Escalate to the regime diagnostic per the brief's escalation path.")

    # Disagreement check
    if args.combine_with and v_paper != v_combined:
        print()
        print(f"  NOTE: forward-only ({v_paper}) and combined ({v_combined}) verdicts")
        print(f"  disagree. Trust forward-only — it's the genuine OOS test. The")
        print(f"  combined number is inflated by the backtest mass.")
    print("=" * 78)


if __name__ == "__main__":
    main()
