# Claude Code Brief — Walk-Forward + Deflated Sharpe pipeline

**Repo:** AI_FOREX_HFT (revived)
**Host:** goldie-oci (personal, not Rassure AWS)
**Owner:** Colin Chin
**Date:** 2026-05-21

## Why we are doing this

The existing `scripts/optimize_pairs.py` uses a single 70/30 train/test split
across ~100 parameter combinations per pair. That methodology produced 11
"qualified" pairs out of 28 G10 — but with that many trials and one split, a
PF ≥ 1.0 gate is almost certainly not statistically significant. This is the
same alpha-collapse failure mode that killed the LGBM model in
`tennis_quant` once it was compared against the Pinnacle closing line.

Before any live capital goes back on this system, we need:

1. **Honest OOS evidence.** Walk-forward refit, non-overlapping test windows,
   aggregate the OOS PnLs.
2. **Selection-bias correction.** Deflated Sharpe Ratio (López de Prado) on
   the OOS Sharpe, deflated by the full grid × fold trial count.
3. **A small list of survivors.** Expectation: 0–3 pairs pass STRONG. That is
   the honest answer, not a failure.

## New files (drop into the repo, no other changes required)

- `scripts/walk_forward_optimize.py`
- `scripts/deflated_sharpe_analysis.py`

Both are self-contained and reuse the existing `BacktestEngine`,
`BacktestSimulator`, `MeanReversionStrategy`, `MultiTimeframeFilter`,
`ConversionRateCache`, `BacktestDataLoader`, `SPREAD_TABLE`, and `G10_PAIRS`
imports. No edits to existing files.

## Dependencies to check before running

```bash
python -c "import scipy, numpy, yaml, pandas; print('ok')"
```

`scipy` is the only new dependency. If absent:

```bash
pip install scipy
```

## Run plan

### Step 1 — Smoke test (≤ 5 minutes)

Validates wiring on a single pair with a coarse window before committing to
the full overnight run.

```bash
python scripts/walk_forward_optimize.py \
    -f 2024-01-01 \
    --pairs EUR_USD \
    --train-days 180 \
    --test-days 60 \
    --step-days 60 \
    --skip-download    # only if parquet cache already populated
```

**Pass criteria:**
- Script completes without exceptions.
- `data/walk_forward/EUR_USD.json` exists and contains ≥ 2 folds.
- `oos_aggregate.total_trades` > 0.
- Summary table prints at the end.

If anything throws, fix imports / data paths before continuing. Do NOT
silently catch — these scripts are meant to fail loudly on bad inputs.

### Step 2 — Full walk-forward run (overnight)

```bash
nohup python scripts/walk_forward_optimize.py \
    -f 2022-01-01 \
    --train-days 365 \
    --test-days 90 \
    --step-days 90 \
    > logs/walk_forward.log 2>&1 &
```

- 28 pairs × ~12 folds × 100 grid combos = ~33,600 backtests.
- Expect 4–10 hours on goldie-oci depending on CPU.
- `tail -f logs/walk_forward.log` to monitor.
- Output: `data/walk_forward/<pair>.json` (per pair) + `_summary.yaml`.

### Step 3 — Deflated Sharpe analysis (seconds)

```bash
python scripts/deflated_sharpe_analysis.py \
    --input-dir data/walk_forward \
    --output    data/walk_forward/_dsr_report.json
```

Prints the verdict table and writes the JSON report.

## What success looks like

| Outcome | Meaning | Action |
|---|---|---|
| **STRONG** on ≥ 1 pair (DSR ≥ 0.95) | OOS Sharpe is significantly above the selection-bias baseline. Real evidence of edge. | Eligible for **live-money pilot at minimum size** (1 unit, not 1% risk) for ≥ 200 trades on a tiny live account. |
| **MARGINAL** on ≥ 1 pair (0.80 ≤ DSR < 0.95) | Possible edge, not statistically confirmed. | Paper-trade on OANDA Practice with that exact config for 90 days. Re-run DSR after. Do **not** size up. |
| All **REJECT** (DSR < 0.80) | OOS Sharpe is inside the noise band of random search. No evidence of edge. | Do **not** trade live. Either re-think the signal (regime filter, different timeframe, broker change) or shelve the project. |

## Reading the DSR table honestly

DSR is deliberately very harsh. Empirically:

- A grid of **100 combos × 10 folds = 1000 trials** requires a per-trade
  Sharpe ≈ 0.25 (annualised Sharpe ~6–8) to pass DSR ≥ 0.95. That bar is
  almost impossible for retail FX scalping at OANDA Practice spreads, let
  alone live.
- A pair with `PSR(0) ≥ 0.95` (significantly positive Sharpe) but
  `DSR < 0.80` is **not necessarily worthless** — it means the OOS Sharpe
  cannot be statistically distinguished from "best of N random configurations
  of this grid". Combine with `OOS PF`, `oos_total_pnl`, and parameter
  stability for the final judgement. A pair with PSR(0)=0.92, DSR=0.4,
  OOS PF=1.35, total OOS PnL clearly positive, and stable selected params
  across folds is a paper-trade candidate even though DSR rejects it.
- "Everything REJECTS" is the most likely outcome and is not a script bug.

### If everything REJECTS — re-run with a tighter grid

The grid in `walk_forward_optimize.py` defaults to the same 100 combos as
the original optimiser to keep the comparison fair. If the first pass
produces no STRONG verdicts, edit `PARAM_GRID` at the top of
`walk_forward_optimize.py` to a tighter search before deciding the strategy
is dead. Suggested tighter grid (24 combos):

```python
PARAM_GRID = {
    "bb_std":          [2.0, 2.25, 2.5],            # 3
    "rsi_thresholds":  [(25, 75), (30, 70)],        # 2
    "sl_multiplier":   [2.0, 2.5, 3.0, 3.5],        # 4
}
```

Re-run Steps 2 and 3. With ~24 combos × 10 folds = 240 trials, DSR becomes
achievable at per-trade Sharpe ≈ 0.15–0.18 (annualised ~4–5), which is
ambitious but no longer impossible. **Document the change** — you no
longer searched the full original grid, so the result only generalises to
the smaller hypothesis space.

## What this pipeline does NOT validate

Be explicit with Colin about these — they are real and remain unresolved
regardless of how good the DSR result looks:

1. **OANDA Practice ≠ Live.** Practice has perfect fills, no spread widening,
   no requoting. The DSR result is on backtest data, which sits even further
   from live than Practice does. A STRONG verdict is necessary but not
   sufficient for live capital.
2. **No regime detection.** Mean-reversion dies in trending and high-vol
   regimes. The walk-forward catches some of this implicitly (different
   regimes appear in different folds), but does not protect against a future
   regime shift.
3. **No live latency model.** Sydney → OANDA NY round-trip is 200–400ms. Not
   modelled in backtest.
4. **No broker counter-action.** OANDA may widen spreads or degrade
   execution if the account becomes consistently profitable on scalping
   strategies. Not modelled.

## Hard rules for this work (Claude Code Pyramid of Success)

- **Verification.** After Step 1, manually `cat data/walk_forward/EUR_USD.json
  | python -m json.tool | head -100` and confirm folds + trials structure
  matches the docstring. Do not proceed to Step 2 until you have seen this.
- **No production placeholders.** Both scripts have zero `TODO` / `FIXME`. If
  Claude Code finds it needs to stub something, stop and ask Colin.
- **Scope discipline.** Do not "improve" the existing `optimize_pairs.py`,
  `mean_reversion.py`, or any other file. These two new scripts are
  additive only.
- **No live trading commands.** Under no circumstances run `src/main.py`,
  modify `config/.env`, or call any OANDA endpoint other than the historical
  data download already in `download_all_history`. The output of this
  pipeline is **evidence for a decision**, not a decision.
- **Context hygiene.** When Step 2 is running, do not interleave other work
  in the same session — it is a long job and you want the log readable.

## After the run — report back to Colin with

1. The full DSR table (pasted from `_dsr_report.json` or the printed table).
2. Top 3 pairs by DSR with their `param_stability` block from the per-pair
   JSON — if the selected params drift wildly fold-to-fold, that is a red
   flag even at STRONG.
3. Any pair where `oos_kurtosis > 5` or `|oos_skew| > 2` — the DSR formula
   becomes less reliable at extreme moments and warrants a closer look at
   the OOS PnL distribution.
4. The total OOS PnL across all surviving (STRONG/MARGINAL) pairs. If the
   answer is "$200 over 18 months of OOS", that is not a project worth live
   capital regardless of statistical significance.
