# Changes — Walk-Forward + DSR + ECN Cost Experiment

Tracks every modification made under the two briefs in `CLAUDE_CODE_BRIEF.md`
(walk-forward + DSR) and `ECN_COST_EXPERIMENT_BRIEF.md`. The repository's
existing files are touched only where briefs explicitly permit; everything
else is additive.

## Files added

| File | Brief | Purpose |
|---|---|---|
| `scripts/walk_forward_optimize.py` | Walk-forward | Walk-forward parameter optimizer with full trial recording (DSR-ready) |
| `scripts/deflated_sharpe_analysis.py` | Walk-forward | DSR analysis on walk-forward output |
| `scripts/run_step2.sh` | Walk-forward (ops) | 8-way parallel launcher for the 28-pair Step 2 run on Windows |
| `backtest/cost_model.py` | ECN | `CostAwareSimulator` (subclasses `BacktestSimulator`) — charges full round-trip spread (entry half retained from parent + close-side half added here) and an optional round-trip commission (bps of notional in quote ccy). Two factories: `..._oanda` (commission=0) and `..._ecn` (commission=0.8 bps). Plus the ECN spread function. |
| `scripts/run_ecn_experiment.py` | ECN | Runner that installs the chosen cost model into `walk_forward_optimize`'s seams (`--cost-model {oanda,ecn}`) and calls the existing `run_walk_forward(...)` |
| `scripts/run_step2_broker_experiment.sh` | ECN (ops) | Walk-forward broker launcher — UNUSED in the final cut (grid search was dropped, see below). Kept for reference. |
| `scripts/run_locked_params_experiment.py` | ECN (final design) | The runner that produced the broker-cost verdict. Locks strategy params to `settings.yaml` defaults (bb_std=2.0, rsi 35/65, sl_mult=2.5) — no grid search — and runs one continuous backtest per pair through `CostAwareSimulator`. The clean one-variable controlled experiment. |
| `scripts/recost_paper_trades.py` | Paper-trade follow-up | Re-costs paper-trade OANDA trades from `trades.db` into ECN equivalents. Reports **forward-only stats alongside the combined backtest+paper-trade series side-by-side**, both with their own verdicts. Primary verdict is forward-only — the combined number is dominated by the ~20x larger backtest sample and can mask short-run decay; the script flags the case where the two verdicts disagree. |
| `docs/PAPER_TRADE_GBP_NZD.md` | Paper-trade follow-up | Instructions for the conditional 90-day GBP_NZD paper-trade — config diffs, start command, weekly monitoring queries, and the decision rule for what to do at the end of the window. |
| `data/walk_forward_smoke_archive/` | Walk-forward (ops) | Archived smoke-test outputs so Step 2's `data/walk_forward/` directory stays clean |

## Existing files modified

### `scripts/walk_forward_optimize.py`

Three edits, all small and documented in-file:

1. **Timezone fix (line ~478).**
   `from_date` / `to_date` were parsed via `datetime.fromisoformat(...)` which
   returns tz-naive datetimes, but the parquet candle events are tz-aware
   UTC. The naive vs. aware comparison in `_split_by_time` raised
   `TypeError`. Patched to `.replace(tzinfo=timezone.utc)` on both, defaulting
   `to_date` to `datetime.now(timezone.utc)`.

2. **`SIMULATOR_FACTORY` seam (top of file + `_run_backtest`).**
   Module-level `SIMULATOR_FACTORY = None`. When `None` (default), behaviour
   is bit-identical: `BacktestSimulator(...)` is constructed as before. When
   set, the factory is called with `(initial_equity, spread_pips,
   slippage_pips, conversion_cache)` to produce the simulator. Used by
   `run_ecn_experiment.py` to inject `CostAwareSimulator`.

3. **`SPREAD_OVERRIDE` seam (top of file + spread resolution in `run_walk_forward`).**
   Module-level `SPREAD_OVERRIDE = None`. When `None` (default), spread is
   `SPREAD_TABLE.get(instrument, 2.0)` exactly as before. When set, it's a
   callable `(instrument, oanda_spread_pips) -> ecn_spread_pips`. Used by
   `run_ecn_experiment.py` to inject `ecn_spread_override`.

   Both seams default to `None` so running `walk_forward_optimize.py`
   directly produces output identical to the OANDA baseline. Verified by the
   ECN brief's Step 0 reproduction gate (GBP_NZD OOS PF / total_trades match
   the original `data/walk_forward/GBP_NZD.json`).

No other existing files are modified. `backtest/simulator.py`,
`backtest/engine.py`, `src/strategy/mean_reversion.py`, and
`scripts/optimize_pairs.py` are unchanged.

## Run outputs

| Directory | Source |
|---|---|
| `data/walk_forward/` | Original OANDA baseline (`BacktestSimulator`, half-spread quirk). DSR: all 28 REJECT. Kept for reference, but not the apples-to-apples comparison target for the ECN run. |
| `data/locked_params_oanda/` | **Locked-params OANDA-v2 run** — corrected-OANDA accounting, no grid search. Compares directly with `data/locked_params_ecn/`. |
| `data/locked_params_ecn/` | **Locked-params ECN run** — final broker-cost experiment output. **GBP_NZD is the only MARGINAL pair** (PSR(0)=0.94, PF=1.087). 27/28 reject. |
| `data/walk_forward_repro/` | Step 0 reproduction through the defaulted seams — verifies the seams don't change behaviour at their defaults. Truncated (killed when we switched to locked-params design). |
| `data/walk_forward_smoke_archive/` | Initial smoke-test artefacts from the walk-forward wiring validation. |
| `data/smoke_oanda_v2/`, `data/smoke_ecn/` | Final 4-fold smokes that surfaced the grid-search-variance problem. |

## Half-spread sim quirk — discovered during ECN smoke

The base `BacktestSimulator` charges only **half** the configured spread —
applied as `fill_price = close ± spread/2` at entry — and closes at the exact
SL/TP price with no spread re-applied. That under-counts the round-trip cost
by half.

For the ECN experiment that asymmetry was structurally biased toward OANDA:
the ECN cost model correctly charges full round-trip commission, but the
OANDA leg under-counts its round-trip spread by half. On tight-spread pairs
the commission can exceed the half-spread "savings" even when full round-trip
ECN cost is strictly lower than full round-trip OANDA cost.

Fix: `CostAwareSimulator._close_position` deducts the close-side half-spread
in addition to commission. Both OANDA-v2 and ECN runs go through the same
class — only `commission_bps_round_trip` differs. The base `BacktestSimulator`
is **unchanged**; the original `data/walk_forward/` outputs are preserved as
historical reference. The corrected accounting lives entirely in
`cost_model.py`.

## Grid search dropped from the broker-cost experiment

A 4-fold smoke under the corrected sim showed that on identical params ECN
beat OANDA-v2 by 5-7% PF per fold (matching the math sanity check), but the
aggregate flipped — different cost models picked different best-on-train
params, and one fold's selection difference flipped the aggregate OOS PF.

That made the grid search a **second variable**. The brief required exactly
one variable (cost). Decision (Colin, 2026-05-22): drop the grid, lock params
to `config/settings.yaml` documented defaults (bb_std=2.0, rsi 35/65,
sl_multiplier=2.5), and run **one continuous backtest per pair per cost
model** over the full 2022-01-02 → 2026-05-21 period. `scripts/run_locked_params_experiment.py`
implements this. ~10 min wall clock (vs ~25h for the abandoned grid design).

## Broker-cost verdict

- **0 STRONG, 1 MARGINAL, 27 REJECT** under ECN.
- The MARGINAL pair is **GBP_NZD**: PF 1.087, PSR(0) 0.94, 1862 trades, annualised Sharpe 0.73.
- ECN materially helped 24/28 pairs (PF deltas +0.01 to +0.36, scaling with
  OANDA spread width). On the 4 tightest-spread USD pairs ECN was marginally
  worse (0.8 bps commission > half-spread savings).
- Two pairs (CHF_JPY, EUR_NZD) crossed PF≥1.0 but are not statistically
  significant (PSR(0) ≈ 0.59).
- Cost was a real contributor, but **not the binding constraint** for most
  pairs. The signal is the problem.

## Paper-trade follow-up — CANCELLED after the spread audit

Initial intent (committed in 95973e7, reverted in d022b58): 90-day
OANDA-Practice paper-trade on GBP_NZD as out-of-sample data collection
against the MARGINAL verdict. See `docs/PAPER_TRADE_GBP_NZD.md` for the
operational steps that *would* have been used. `scripts/recost_paper_trades.py`
remains in the repo because the same machinery would re-cost any future
paper-trade against the ECN model.

## Spread audit — the MARGINAL verdict was a stale-spread artifact

Pre-flight on OCI showed GBP_NZD's live OANDA spread at 7.8 pips vs the
backtest's assumed 4.0 pips (~2× heavier). Built `scripts/audit_live_spreads.py`
to poll OANDA pricing 6× over 3 min for all 28 G10 pairs and reported
realistic = max-of-samples (deliberately pessimistic).

Five pairs were MATERIAL (≥1 pip AND ≥1.5× the table):

| Pair    | Table | Live max | Δ        |
|---------|------:|---------:|---------:|
| GBP_NZD | 4.0   | **8.3**  | **+4.30** |
| EUR_NZD | 3.5   | 6.3      | +2.80    |
| GBP_AUD | 3.5   | 5.3      | +1.80    |
| EUR_AUD | 2.0   | 3.5      | +1.50    |
| EUR_JPY | 1.5   | 2.6      | +1.10    |

Added a `--spread-table-yaml` override to `run_locked_params_experiment.py`
so corrected spreads can be applied at runtime without mutating the canonical
`SPREAD_TABLE`. Re-ran both cost models for all 28 pairs against
`data/spread_audit/corrected_spread_table.yaml` (the audit's pessimistic
output). Outputs landed in `data/locked_params_oanda_v3/` and
`data/locked_params_ecn_v3/`.

**Corrected verdict: 0 STRONG, 0 MARGINAL, 28 REJECT.**

GBP_NZD under corrected spreads:
- PF dropped 1.087 → 0.986 (Δ −0.10)
- PSR(0) collapsed 0.94 (MARGINAL) → 0.40 (REJECT)

The "MARGINAL" verdict was an optimistic-spread artifact. The signal does
not survive realistic OANDA costs on any pair. Per the brief's escalation
path, the binding constraint is the **signal**, not the broker. Next vector
is the **regime diagnostic** (spec to follow from Colin).

## Run outputs — final

| Directory | Purpose |
|---|---|
| `data/locked_params_oanda_v3/` | OANDA-v2 corrected accounting + corrected spreads (final apples-to-apples baseline). |
| `data/locked_params_ecn_v3/` | ECN spreads + 0.8 bps commission + corrected spread base (final ECN result). |
| `data/spread_audit/` | Live-spread audit JSON + corrected SPREAD_TABLE YAML. |
| `data/autopsy_ecn/` | 24-combo tighter-grid autopsy under ECN + corrected spreads. Per-pair JSONs + `_autopsy_verdict.json`. |

`data/locked_params_oanda/` and `data/locked_params_ecn/` (without `_v3`) are
the pre-audit runs with the stale OANDA spreads — kept for reference but
*not* the verdict.

## Tighter-grid autopsy — final closure (2026-05-22)

`scripts/run_tighter_grid_autopsy.py` runs the brief's fallback 24-combo grid
(`bb_std × rsi_thresholds × sl_multiplier = 3×2×4 = 24`) under ECN + corrected
spreads, picks the best combo per pair by Sharpe, and reports DSR with N=24.

This is the most permissive test the strategy could fairly receive — small
grid (weakest selection-bias deflation), best-of-N picked in-sample (most
favourable read), best cost model (ECN), realistic spreads.

**Result: 0 STRONG, 0 MARGINAL, 28 REJECT.** Best DSR was AUD_NZD at 0.788
(needs ≥0.85 for MARGINAL). Even with PSR(0) = 0.98 on AUD_JPY (highly
significant raw Sharpe), selection-bias deflation against N=24 collapses it
to DSR 0.70.

The full strategy postmortem is at `docs/POSTMORTEM_M5_BB_RSI.md`.

## Final state — strategy shelved

The M5 BB+RSI mean-reversion signal on G10 spot is **dead** at every grid
width, cost model, and parameter-selection methodology tested across 4.4
years of data. Live OANDA Practice account is idle (no process running),
config is back at the 11-pair production list, and the only follow-up
queued is a separate planning conversation for the regime-diagnostic
"next project" (Option 3 from the verdict triage). No more measurement
work on this signal is planned.

## Operational notes

- Smoke test ran with `-f 2025-02-01` (not the brief's `-f 2024-01-01`)
  because the cached conversion-cache instruments (e.g. `AUD_USD`) only
  covered 2025-01-01+ at the time. Step 2 was unaffected because Phase 1
  fetched all 28 pairs back to 2022-01-02.
- Step 2 (OANDA baseline) was run on Windows 8-way parallel because the
  goldie-oci shape (4 ARM cores) benchmarked ~2× slower per backtest than
  Windows. Wall clock: ~18 h. Total backtests: 36,400.
- Encoding: scripts emit `→` / `≥` Unicode characters. On Windows console
  set `PYTHONUTF8=1` (the run_step2.sh launcher does this).
