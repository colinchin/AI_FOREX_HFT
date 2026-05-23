# Post-mortem — M5 BB+RSI mean-reversion on G10 forex

**Date:** 2026-05-22
**Verdict:** The M5 Bollinger-Bands + RSI mean-reversion signal does **not**
produce a statistically significant edge on any G10 currency pair under
realistic broker costs, on 4.4 years of data (2022-01-02 → 2026-05-21), across
multiple grid widths, multiple cost models, and multiple parameter-selection
methodologies. **Strategy is shelved.**

---

## Evidence trail — five experiments, same conclusion

| # | Experiment | Methodology | Result |
|---|---|---|---|
| 1 | Original walk-forward + DSR (`data/walk_forward/`) | 100-combo grid, 13 walk-forward folds, DSR with N_trials=1300 | **0 STRONG, 0 MARGINAL, 28 REJECT** |
| 2 | Corrected-sim locked-params, original `SPREAD_TABLE` (`data/locked_params_{oanda,ecn}/`) | Locked defaults, full round-trip spread + ECN commission via `CostAwareSimulator` | **0 STRONG, 1 MARGINAL (GBP_NZD), 27 REJECT** |
| 3 | Corrected-sim locked-params, **corrected** `SPREAD_TABLE` (`data/locked_params_*_v3/`) | Same as #2 but with realistic OANDA spreads from live audit | **0 STRONG, 0 MARGINAL, 28 REJECT** |
| 4 | Tighter-grid autopsy under ECN (`data/autopsy_ecn/`) | 24-combo grid, best-by-Sharpe per pair, DSR with N=24, corrected spreads | **0 STRONG, 0 MARGINAL, 28 REJECT** (final read) |

Each experiment was deliberately more permissive than the last (smaller grid →
weaker selection-bias deflation → easier to pass DSR). None produced a
survivor.

## The two "ghost candidates" and why they were ghosts

### GBP_NZD (briefly MARGINAL in experiment #2)

Locked-defaults backtest produced PF=1.087 and PSR(0)=0.94 — just shy of
STRONG. The paper-trade was queued.

**Audit found the cause was a stale `SPREAD_TABLE`.** Live OANDA spread on
GBP_NZD was **8.3 pips** (peak London), vs the table's **4.0 pip** assumption
— ~2× heavier. With realistic spreads:
- PF dropped 1.087 → 0.986 (loss-making)
- PSR(0) collapsed 0.94 → 0.40 (REJECT)

GBP_NZD was a stale-data artefact, not edge.

### AUD_NZD / AUD_JPY (best of the autopsy)

In the 24-combo autopsy, AUD_NZD reached **PSR(0)=0.92, DSR=0.79** — the
closest any pair came. AUD_JPY had the highest raw PSR(0)=0.98 but DSR=0.70.
Both fail the MARGINAL bar (DSR ≥ 0.85).

These are pairs where the grid found ONE combo that looks promising in-sample;
the DSR correctly deflates it as best-of-24 random selection.

## Five MATERIAL stale spreads found in audit

The original `SPREAD_TABLE` (`scripts/backtest_all_pairs.py:51-67`) had five
pairs materially optimistic compared to live OANDA Practice:

| Pair    | Table | Live max | Δ        |
|---------|------:|---------:|---------:|
| GBP_NZD | 4.0   | 8.3      | +4.3     |
| EUR_NZD | 3.5   | 6.3      | +2.8     |
| GBP_AUD | 3.5   | 5.3      | +1.8     |
| EUR_AUD | 2.0   | 3.5      | +1.5     |
| EUR_JPY | 1.5   | 2.6      | +1.1     |

These are exactly the pairs the table comment described as "wide cross" — the
wide spreads have widened further in live, and the table is now optimistic.
The corrected YAML is at `data/spread_audit/corrected_spread_table.yaml`.

All five were the pairs that *looked* most edge-y in the original walk-forward
verdict (e.g. GBP_NZD ranked #1 by DSR). The correlation isn't coincidence —
optimistic spread assumptions inflate the apparent edge on wide-cross pairs
more than tight-cross pairs.

## Things that materially changed during the investigation

1. **Half-spread sim quirk** in `backtest/simulator.py:106` — the base sim
   charged only half the configured spread (entry only); close was at exact
   SL/TP price. Round-trip cost under-counted by half. The ECN comparison
   would have looked structurally biased toward OANDA on tight-spread pairs.
   Fixed in `backtest/cost_model.py:CostAwareSimulator` for both runs of the
   broker-cost experiment.

2. **Stale `SPREAD_TABLE`** — see above. Original values were "realistic" at
   the time of writing but had drifted by 2026.

3. **Grid-search variance is itself a variable** — in the broker-cost smoke,
   different cost models picked different best-on-train params per fold, and
   one fold flipped the aggregate even when same-params per-fold comparisons
   consistently favoured ECN. Dropping the grid (locked-params) gave a cleaner
   one-variable controlled experiment.

## What would have to be true for the strategy to work — and isn't

The brief framed this in advance. Each prediction held:

| Brief said | Actual |
|---|---|
| "A grid of 100 combos × 10 folds = 1000 trials requires a per-trade Sharpe ≈ 0.25 (annualised ~6–8) to pass DSR ≥ 0.95. That bar is almost impossible for retail FX scalping..." | Best observed per-trade Sharpe in any experiment ≈ 0.11 (AUD_JPY in the autopsy). The strategy is at ~half the bar with no headroom. |
| "Everything REJECTS is the most likely outcome and is not a script bug." | Confirmed. |
| "Cost was not the binding constraint → signal is the problem." | Confirmed: cost-model improvements (ECN spread cut by 70%, commission added) shift PFs by 0.03-0.18 depending on pair, but never enough to lift a pair across the significance bar. |

## What this does NOT say

1. **Not** that mean-reversion is dead on forex globally — only that **this
   specific signal (BB period 20, std 2.0, RSI 14, period-defined oversold/
   overbought)** on **M5** across **G10 spot pairs** does not have an edge
   robust to costs.
2. **Not** that walk-forward + DSR is too harsh — the locked-params experiment
   skipped DSR entirely (no selection) and still rejected everything. Cost +
   signal weakness is what failed, not the statistical methodology.
3. **Not** that a different broker would save it. ECN spreads + commission
   under realistic cost assumptions improve a handful of pairs but don't
   produce a survivor.

## What might work — Option 3 ("next project" conversation)

Conversation to have, **not** task to start without scoping:

1. **Regime conditioning.** The mean-reversion signal probably has REAL edge
   in narrow-range/low-vol regimes and bleeds in trending regimes. A regime
   classifier (vol-of-vol, ATR percentile, ADX, trend-strength filter) gating
   when the signal fires could materially shift the PF distribution. Spec
   needed: which regime indicator(s), which gating threshold, which fold
   structure (regime might be itself a hyperparameter that grid-search would
   over-fit).
2. **Different timeframe.** M15 / M30 / H1 with same signal. Larger bars =
   fewer trades, less spread-cost drag, potentially cleaner mean-reversion
   signal. The cost analysis here suggests the spread/commission per-trade
   eats too large a fraction of M5 trade PnL on G10; longer holds amortise
   cost.
3. **Different signal entirely.** If both #1 and #2 also empty out, the
   M5-on-G10 *frequency band* is the issue, not the signal selection. Move
   to: order-flow imbalance, funding-rate carry-style holds, options-implied
   risk reversal, etc.

The brief explicitly cautioned against "another filter" if regime work is
also empty — at that point the signal is dead and a different alpha source
is the move, not more refinement of this one.

## Files of record

| Path | What |
|---|---|
| `CHANGES.md` | Full change log across the walk-forward, ECN, audit, autopsy, postmortem arc |
| `data/walk_forward/_dsr_report.json` | Original 100-combo DSR verdict (all 28 REJECT) |
| `data/locked_params_oanda_v3/`, `data/locked_params_ecn_v3/` | Corrected-spread + corrected-sim final locked-params runs |
| `data/autopsy_ecn/` | 24-combo tighter-grid autopsy (best per pair) |
| `data/spread_audit/spread_audit.json` | Live-pricing audit results |
| `data/spread_audit/corrected_spread_table.yaml` | Realistic per-pair spreads for any future re-use |
| `backtest/cost_model.py` | `CostAwareSimulator` — the corrected cost-accounting harness; reusable for any future cost experiment |
| `scripts/run_locked_params_experiment.py` | Locked-params runner with `--spread-table-yaml` override |
| `scripts/run_tighter_grid_autopsy.py` | The autopsy runner |
| `scripts/audit_live_spreads.py` | Live OANDA pricing audit |
| `scripts/recost_paper_trades.py` | Paper-trade re-cost machinery — unused on this strategy but kept for the regime-diagnostic work to follow |

## Final operational state

- Live OANDA Practice account `101-011-37145145-001` is **idle**. `src/main.py`
  was never started during this investigation. The 11-pair production
  `instruments` list in `config/settings.yaml` is intact; the GBP_NZD-only
  paper-trade config was reverted before any process was launched.
- Repo `colinchin/AI_FOREX_HFT@master` reflects the final state of code + docs.
- OCI (`goldie-oci` at `/data/projects/AI_FOREX_HFT`) is in sync with master,
  no processes running.

## Closing

The system did what it was designed to do: ship a measurable verdict under
multiple cost regimes and parameter widths. The verdict is that the M5 BB+RSI
mean-reversion signal on G10 spot doesn't have edge. Time to put it down and
have the next-project conversation about what to investigate instead.
