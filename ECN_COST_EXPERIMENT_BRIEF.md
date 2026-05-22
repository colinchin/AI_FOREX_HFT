# Claude Code Brief — ECN Cost Experiment (AI_FOREX_HFT)

**Repo:** AI_FOREX_HFT
**Owner:** Colin Chin
**Date:** 2026-05-22
**Builds on:** the completed walk-forward + DSR run (all 28 pairs REJECT on OANDA-Practice costs)

---

## The one thing that matters

This is a **controlled experiment**. Exactly **one** variable changes versus the
run that just rejected all 28 pairs: **the cost model** (OANDA standard spread →
ECN raw spread + commission). Everything else — signal, parameter grid,
walk-forward windows, DSR analysis — stays **byte-for-byte identical**.

If you find yourself wanting to also tweak the grid, the timeframe, the windows,
or the signal: **stop.** Adding any second variable destroys the experiment. The
whole point is a clean read on "was OANDA's spread the binding constraint?"

Do not grid-search the cost parameters either. The ECN spreads and commission are
**fixed inputs**, not things to optimise. Searching them would add trial count and
raise the DSR bar for nothing.

---

## Deliverables (additive — do not rewrite existing files)

1. **`backtest/cost_model.py`** — new module. The ECN cost definitions + a
   commission-aware simulator.
2. **`scripts/run_ecn_experiment.py`** — new thin runner. Reuses
   `walk_forward_optimize.py`'s functions; only swaps the cost model.
3. **A minimal injection seam in `walk_forward_optimize.py`** — see "Injection"
   below. This is the *only* permitted edit to an existing file, it must default
   to identical current behaviour, and it must be recorded in `CHANGES.md`.
4. **`deflated_sharpe_analysis.py` runs completely unchanged** — it just reads a
   different `--input-dir`.

---

## ECN cost model (fixed, conservative — must NOT flatter ECN)

Use **deliberately conservative** (i.e. *worse than real*) ECN figures so the
experiment cannot fool itself. If pairs improve even under pessimistic ECN costs,
that is robust evidence.

### Raw spread
Derive per-pair ECN raw spread from the **existing** `SPREAD_TABLE` (the OANDA
spreads already used in the rejected run):

```
ecn_raw_spread_pips = max(0.2, oanda_spread_pips * 0.30)
```

Rationale: real ECN raw spread on majors is often 10–20% of OANDA standard.
Using **30% with a 0.2-pip floor** over-charges ECN on purpose. Document this
formula in `cost_model.py`.

### Commission
ECN brokers (IC Markets Raw, Pepperstone Razor) charge ~USD $3.50 per side per
100k notional ≈ 0.7 bps round-trip. Use a conservative round-trip:

```
COMMISSION_BPS_ROUND_TRIP = 0.8     # 0.4 bps per side; > real, on purpose
```

**Commission must be modelled correctly:**
- Charge it as **bps of notional**, *not* a flat per-trade fee. Flat fees
  mis-scale across position sizes.
- Compute commission in the **same account currency (AUD)** and on the **same
  conversion path** the existing `PositionSizer` / simulator uses to book PnL,
  so commission and PnL are in identical units. Reuse the existing
  `ConversionRateCache` — do not invent a second conversion path.
- **Do NOT fold commission into `spread_pips`.** Pip value varies by pair and by
  quote currency; the spread-equivalent approximation breaks across the book.
  Deduct commission from realised PnL at close.

Read `backtest/simulator.py` first and implement the override against the
**actual** close/PnL-booking code. Suggested shape (adapt to real internals):

```python
# backtest/cost_model.py
class CostAwareSimulator(BacktestSimulator):
    """BacktestSimulator + per-trade ECN commission, deducted at close in AUD."""
    def __init__(self, *args, commission_bps_round_trip: float, conversion_cache, **kw):
        super().__init__(*args, **kw)
        self._commission_bps = commission_bps_round_trip
        self._conv = conversion_cache
    # override whatever method books the close PnL; after the existing PnL is
    # computed, subtract: notional_in_aud * (commission_bps_round_trip / 10_000)
```

---

## Injection (the only edit to an existing file)

`walk_forward_optimize.py` currently constructs `BacktestSimulator(...)` inline
inside `_run_backtest`, and reads `SPREAD_TABLE` from
`scripts.backtest_all_pairs`. Add two module-level seams that **default to the
existing behaviour** so the file's default output is unchanged:

```python
# near the top of walk_forward_optimize.py
SIMULATOR_FACTORY = None      # None => use BacktestSimulator (current behaviour)
SPREAD_OVERRIDE   = None      # None => use SPREAD_TABLE (current behaviour)
```

- In `_run_backtest`: if `SIMULATOR_FACTORY` is not None, build the simulator via
  it; else build `BacktestSimulator` exactly as now.
- Where spread is resolved: if `SPREAD_OVERRIDE` is not None, use it; else
  `SPREAD_TABLE`.

`run_ecn_experiment.py` sets these two module attributes to the ECN cost model,
then calls the **existing** `run_walk_forward(...)` with output dir
`data/walk_forward_ecn/`. No other walk-forward logic is touched.

---

## Run plan

### Step 0 — Reproduce the OANDA baseline through the new seam (CRITICAL)

Before any ECN run, run the pipeline with seams at their **defaults** (OANDA
costs) on 1–2 pairs and confirm the JSON output **matches the original rejected
run for those pairs**. This proves the seam introduced no behavioural change.

```bash
python scripts/walk_forward_optimize.py -f 2022-01-01 \
    --pairs GBP_NZD --train-days 365 --test-days 90 --step-days 90 --skip-download \
    -o data/walk_forward_repro
# Compare: oos_aggregate.profit_factor / total_trades vs data/walk_forward/GBP_NZD.json
```

**Gate:** if GBP_NZD's OOS PF / trade count do not match the original within
rounding, the seam is buggy. Fix before proceeding. Do not continue past a
failed reproduction.

### Step 1 — ECN smoke test (≤ 10 min)

```bash
python scripts/run_ecn_experiment.py -f 2022-01-01 \
    --pairs GBP_NZD GBP_AUD --train-days 365 --test-days 90 --step-days 90 \
    --skip-download -o data/walk_forward_ecn
```

**Pass criteria:**
- Completes without exception; `data/walk_forward_ecn/GBP_NZD.json` exists.
- Sanity log line confirming ECN cost < OANDA cost for these pairs (e.g. print
  effective spread + commission per pair at startup).
- OOS PF for GBP_NZD/GBP_AUD is **higher** than the OANDA run (lower costs
  *must* raise PF; if it doesn't, the commission sign or conversion is wrong).

### Step 2 — Full ECN walk-forward (~18 h, 8-way parallel)

Reuse the same 8-way parallel launcher pattern as `run_step2.sh`, pointed at
`run_ecn_experiment.py` and output dir `data/walk_forward_ecn/`. Same 28 pairs,
same windows, `--skip-download` (history already cached from the OANDA run).

### Step 3 — DSR on ECN output (seconds, script unchanged)

```bash
python scripts/deflated_sharpe_analysis.py \
    --input-dir data/walk_forward_ecn \
    --output    data/walk_forward_ecn/_dsr_report.json
```

---

## What to report back to Colin

1. **Side-by-side DSR table: OANDA vs ECN**, per pair — DSR, PSR(0), OOS PF,
   OOS trades, OOS PnL. The delta columns are the result.
2. **Any pair that moved** REJECT→MARGINAL or REJECT→STRONG, or whose DSR
   improved materially (say Δ DSR > 0.2).
3. **Cost delta per pair**: average round-trip cost as % of gross profit under
   OANDA vs ECN. This quantifies how much headroom the broker switch actually
   bought.
4. **Honest verdict** against the decision rule:
   - Any STRONG → that pair earns a 90-day OANDA-Practice paper-trade as the
     next gate (note: you'd ultimately trade it on the ECN, but Practice is the
     cheap forward-test first).
   - Only MARGINAL → paper-trade candidates, do not size up; the broker helped
     but didn't confirm edge.
   - Still all REJECT → cost was **not** the binding constraint. The signal
     itself is the problem. Next vector is the **regime diagnostic** (spec to
     follow), and if that's also empty, the M5 BB+RSI signal is dead and the
     next move is a different alpha source, not another filter.

Expectation, stated honestly: ECN costs alone are **unlikely** to lift GBP_NZD
(implied per-trade Sharpe ~0.09) to the DSR-STRONG bar (~0.18). The realistic
best case is one or two MARGINAL pairs. That is still a useful, clean result —
it tells us whether to invest in the regime work next.

---

## Hard rules (Claude Code Pyramid of Success)

- **No live trading.** Do not run `src/main.py`, do not touch `config/.env`, do
  not hit any OANDA endpoint except the already-cached history (`--skip-download`
  everywhere). This pipeline produces **evidence for a decision**, not a trade.
- **No production placeholders.** Zero TODO/FIXME in `cost_model.py` or the
  runner. If the real simulator's PnL booking is unclear, stop and ask Colin
  rather than guessing the commission deduction point.
- **Scope discipline.** Do not "improve" the signal, the grid, the windows, or
  `mean_reversion.py`. One variable: cost.
- **Verification gate.** Do not proceed past Step 0 until the OANDA reproduction
  matches. Do not proceed past Step 1 until ECN PF > OANDA PF on the smoke pairs.
- **Document the seam.** Record the two-line `walk_forward_optimize.py` change
  and the new files in `CHANGES.md`, alongside the tz-fix and `run_step2.sh`
  notes from last run.
- **Context hygiene.** Kick off Step 2 and stand down; report at completion, not
  mid-run.
