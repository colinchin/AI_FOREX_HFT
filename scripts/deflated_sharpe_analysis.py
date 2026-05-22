"""Deflated Sharpe Ratio (DSR) analysis for the walk-forward output.

Implements López de Prado's framework for assessing whether an observed Sharpe
ratio survives multiple-testing correction. This is the statistical filter that
should sit between "I have a backtest with PF=1.5" and "I am putting live money
on this."

References:
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting, and Non-Normality."
  Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier."
  López de Prado (2018), "Advances in Financial Machine Learning", Ch. 14.

What this script computes per pair:

  1. PSR(0)            — Probabilistic Sharpe Ratio at the zero threshold.
                          P(true Sharpe > 0 | observed sample).
                          Standard significance test, ignores selection bias.

  2. SR_star_dsr       — E[max SR over N independent trials under the null].
                          The "deflation" baseline — what you'd expect the best
                          of N random strategies to look like.

  3. DSR               — PSR(SR_star_dsr).
                          P(true Sharpe > the selection-bias baseline).
                          THIS is the number that matters.

  4. MinTRL            — Minimum Track Record Length at α=0.05.
                          How many trades you need before PSR(0) > 0.95
                          given the current sample's skew/kurt.

Verdict thresholds:
  DSR ≥ 0.95   → STRONG    — pair has statistically significant edge
                              after correcting for the grid search trials.
  DSR ≥ 0.80   → MARGINAL  — paper-trade, do not size up.
  DSR <  0.80  → REJECT    — almost certainly overfit. Do not trade.

Usage:
    python scripts/deflated_sharpe_analysis.py \
        --input-dir data/walk_forward \
        --output    data/walk_forward/_dsr_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

# Euler–Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329


# ── Core statistical functions ───────────────────────────────────────────────


def psr(
    sr_hat: float,
    sr_star: float,
    n: int,
    skew: float,
    kurt_non_excess: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    Args:
        sr_hat: Observed un-annualised per-trade Sharpe ratio.
        sr_star: Threshold Sharpe to test against (0 for "is it > 0",
                 SR_star_dsr for the deflated test).
        n: Number of trade observations.
        skew: Sample skewness of per-trade returns.
        kurt_non_excess: NON-excess kurtosis (normal = 3). If you have Fisher
                         excess kurtosis, pass excess + 3.

    Returns:
        Probability that the true Sharpe exceeds sr_star, in [0, 1].
    """
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + ((kurt_non_excess - 1.0) / 4.0) * sr_hat ** 2
    # Guard against pathological negative variance from extreme skew/kurt
    if denom_sq <= 0:
        return 0.5
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def expected_max_sharpe(
    n_trials: int,
    variance_of_sharpes: float,
) -> float:
    """E[max SR_N] under the null — the selection-bias baseline.

    Bailey & López de Prado (2014), equation derived from the asymptotic
    distribution of the maximum of N independent normal samples.

    Args:
        n_trials: Number of independent strategies/configurations tried.
        variance_of_sharpes: Variance of the Sharpe ratios across the trials.

    Returns:
        Expected maximum un-annualised Sharpe a random search would produce.
    """
    if n_trials <= 1 or variance_of_sharpes <= 0:
        return 0.0
    sd = math.sqrt(variance_of_sharpes)
    term1 = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd * (term1 + term2))


def min_track_record_length(
    sr_hat: float,
    skew: float,
    kurt_non_excess: float,
    alpha: float = 0.05,
) -> float:
    """Minimum number of trades required for PSR(0) > 1 - alpha.

    Tells you how much more sample you need before the observed Sharpe is
    statistically distinguishable from zero (ignoring selection bias).
    """
    if sr_hat <= 0:
        return float("inf")
    z = norm.ppf(1.0 - alpha)
    denom = 1.0 - skew * sr_hat + ((kurt_non_excess - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        return float("inf")
    return 1.0 + denom * (z / sr_hat) ** 2


def verdict(dsr_value: float) -> str:
    if dsr_value >= 0.95:
        return "STRONG"
    if dsr_value >= 0.80:
        return "MARGINAL"
    return "REJECT"


# ── Pair-level analysis ──────────────────────────────────────────────────────


def analyse_pair(pair_data: dict) -> dict:
    instrument = pair_data["instrument"]
    cfg = pair_data["config"]
    oos = pair_data["oos_aggregate"]

    sr_hat = float(oos["sharpe_per_trade"])
    n = int(oos["total_trades"])
    skew = float(oos["skew"])
    kurt_excess = float(oos["kurtosis"])
    kurt_non_excess = kurt_excess + 3.0

    # Gather Sharpes from every trial across every fold — this is the
    # population we selected the winner FROM, so its variance is the right
    # input to the deflation formula.
    trial_sharpes: list[float] = []
    for fold in pair_data["folds"]:
        for trial in fold.get("trials", []):
            sr = trial.get("train_sharpe_per_trade")
            # Filter out degenerate runs (too few trades to produce a Sharpe)
            if sr is not None and trial.get("train_trades", 0) >= 5:
                trial_sharpes.append(float(sr))

    n_trials_effective = len(trial_sharpes)
    var_sr = float(np.var(trial_sharpes, ddof=1)) if n_trials_effective > 1 else 0.0
    mean_sr = float(np.mean(trial_sharpes)) if trial_sharpes else 0.0

    sr_star_dsr = expected_max_sharpe(n_trials_effective, var_sr)

    psr_zero = psr(sr_hat, 0.0, n, skew, kurt_non_excess) if n >= 2 else 0.0
    dsr = psr(sr_hat, sr_star_dsr, n, skew, kurt_non_excess) if n >= 2 else 0.0
    min_trl = min_track_record_length(sr_hat, skew, kurt_non_excess, alpha=0.05)

    return {
        "instrument": instrument,
        "oos_trades": n,
        "oos_sharpe_per_trade": round(sr_hat, 4),
        "oos_sharpe_annualised": round(float(oos["sharpe_annualised"]), 3),
        "oos_profit_factor": round(float(oos["profit_factor"]), 3),
        "oos_total_pnl": round(float(oos["total_pnl"]), 2),
        "oos_max_dd_pct": round(float(oos["max_drawdown_pct"]), 4),
        "skew": round(skew, 3),
        "kurtosis_excess": round(kurt_excess, 3),
        "n_trials_effective": n_trials_effective,
        "var_sharpe_trials": round(var_sr, 6),
        "mean_sharpe_trials": round(mean_sr, 4),
        "sr_star_dsr": round(sr_star_dsr, 4),
        "psr_zero": round(psr_zero, 4),
        "dsr": round(dsr, 4),
        "min_trl_trades": (
            round(min_trl, 0) if math.isfinite(min_trl) else None
        ),
        "more_trades_needed": (
            max(0, int(round(min_trl)) - n)
            if math.isfinite(min_trl) else None
        ),
        "verdict": verdict(dsr),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deflated Sharpe Ratio analysis on walk-forward output."
    )
    parser.add_argument(
        "-i", "--input-dir", default="data/walk_forward",
        help="Directory containing per-pair JSON files from walk_forward_optimize.py",
    )
    parser.add_argument(
        "-o", "--output", default="data/walk_forward/_dsr_report.json",
        help="Where to write the DSR report JSON",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        sys.exit(1)

    pair_files = sorted(p for p in in_dir.glob("*.json") if not p.name.startswith("_"))
    if not pair_files:
        print(f"No pair JSON files in {in_dir}. Run walk_forward_optimize.py first.")
        sys.exit(1)

    print(f"\nDeflated Sharpe Analysis")
    print(f"  Input dir:  {in_dir}")
    print(f"  Pair files: {len(pair_files)}")

    results: list[dict] = []
    for pf in pair_files:
        with open(pf) as f:
            data = json.load(f)
        results.append(analyse_pair(data))

    # Sort by DSR descending — best evidence first
    results.sort(key=lambda r: r["dsr"], reverse=True)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 140)
    print("  DEFLATED SHARPE RATIO — OOS evidence after selection-bias correction")
    print("=" * 140)
    header = (
        f"{'Pair':<10} {'Tr':>5} {'SR/tr':>7} {'SR ann':>7} {'PF':>6} "
        f"{'Skew':>6} {'Kurt':>6} {'NTri':>5} {'SR*':>7} "
        f"{'PSR(0)':>7} {'DSR':>7} {'MinTRL':>8} {'Need':>6} {'Verdict':>9}"
    )
    print(header)
    print("-" * 140)
    for r in results:
        min_trl_str = (
            f"{r['min_trl_trades']:>8.0f}"
            if r["min_trl_trades"] is not None else f"{'∞':>8}"
        )
        need_str = (
            f"{r['more_trades_needed']:>6}"
            if r["more_trades_needed"] is not None else f"{'-':>6}"
        )
        print(
            f"{r['instrument']:<10} {r['oos_trades']:>5} "
            f"{r['oos_sharpe_per_trade']:>7.3f} "
            f"{r['oos_sharpe_annualised']:>7.3f} "
            f"{r['oos_profit_factor']:>6.2f} "
            f"{r['skew']:>6.2f} {r['kurtosis_excess']:>6.2f} "
            f"{r['n_trials_effective']:>5} "
            f"{r['sr_star_dsr']:>7.3f} "
            f"{r['psr_zero']:>7.3f} {r['dsr']:>7.3f} "
            f"{min_trl_str} {need_str} {r['verdict']:>9}"
        )
    print("-" * 140)

    n_strong = sum(1 for r in results if r["verdict"] == "STRONG")
    n_marg = sum(1 for r in results if r["verdict"] == "MARGINAL")
    n_rej = sum(1 for r in results if r["verdict"] == "REJECT")
    print(f"  STRONG: {n_strong}   MARGINAL: {n_marg}   REJECT: {n_rej}")
    print("=" * 140)

    # ── Interpretation ───────────────────────────────────────────────────────
    print("\nInterpretation:")
    print("  SR/tr     : Un-annualised per-trade Sharpe (the raw input to DSR).")
    print("  SR ann    : Sharpe scaled to per-year frequency.")
    print("  PSR(0)    : P(true Sharpe > 0). Standard significance test.")
    print("  SR*       : E[max SR] expected from N random trials — the bar to beat.")
    print("  DSR       : P(true Sharpe > SR*). The selection-bias-corrected verdict.")
    print("  MinTRL    : Trades needed for PSR(0) > 0.95 at current skew/kurt.")
    print("  Need      : MinTRL minus current OOS trades. Trades still required.")
    print("\nDecision rule:")
    print("  STRONG    (DSR ≥ 0.95) → eligible for live-money pilot at minimum size.")
    print("  MARGINAL  (DSR ≥ 0.80) → paper-trade only. Collect more data.")
    print("  REJECT    (DSR <  0.80)→ drop the pair. The OOS Sharpe is inside the")
    print("                            noise band of what 100×N_folds random tries")
    print("                            of this grid would produce.")

    # ── Persist report ───────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_pairs": len(results),
            "summary": {
                "strong": n_strong,
                "marginal": n_marg,
                "reject": n_rej,
            },
            "pairs": results,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
