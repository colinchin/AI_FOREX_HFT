# GBP_NZD 90-Day Paper-Trade — Out-of-Sample Data Collection

**Purpose:** Out-of-sample data collection (NOT a profit test). The locked-params
broker-cost experiment produced PSR(0)=0.94 for GBP_NZD under ECN — just shy of
the 0.95 STRONG bar. Another ~90 days of OOS trades, re-costed under ECN at
the end, will tell us whether the marginal edge holds or decays.

If GBP_NZD's PSR(0) **stays at 0.94+ or improves** after the paper-trade: it's
a STRONG candidate for a small live-money pilot.

If it **decays toward 0.5**: the edge was a 4-year backtest artefact. Next move
is the **regime diagnostic** (specced separately by Colin) on the M5 BB+RSI
signal across all 28 pairs.

---

## Pre-flight check (Colin, please review)

1. **Environment must be `practice`, NOT `live`.**
   `config/settings.yaml:2` is currently `environment: practice` ✓.
   Verify `config/.env` points at the practice account (it currently does:
   `OANDA_ENVIRONMENT=practice`, account `101-011-37145145-001`).

2. **No live capital. Practice account only.**
   This is OANDA Practice — fictitious AUD ~89,310 balance. Even if the strategy
   crashes the account, no real money is at stake.

---

## Configuration changes to apply

Two edits, both in `config/`. **Back up the originals first** so you can revert
to your current 11-pair production config after the 90-day window.

### Edit 1 — `config/settings.yaml` (instrument list)

Replace lines 10–25 (the `instruments:` block) with:

```yaml
instruments:
  # 90-day OOS paper-trade for the broker-cost experiment.
  # Only GBP_NZD enabled; locked to settings.yaml defaults (bb_std=2.0,
  # rsi 35/65, sl_multiplier=2.5) via pair_params.yaml override below.
  # Revert to the full 11-pair list after the 90-day window closes.
  - GBP_NZD
```

Leave the rest of `settings.yaml` untouched — `mean_reversion` defaults
(`bb_std=2.0, rsi_oversold=35, rsi_overbought=65, sl_multiplier=2.5`) at
lines 59-64 are already the locked-experiment params.

### Edit 2 — `config/pair_params.yaml`

The current `GBP_NZD` block (lines 105-113) overrides the strategy defaults to
the auto-optimised `bb_std=2.5, rsi 25/75, sl_multiplier=2.5` from the
original optimiser run. For this paper-trade we want the LOCKED experiment
params, so either:

**(a) Remove** the `GBP_NZD:` block entirely (lines 105-113). Without an
override, the strategy uses `settings.yaml`'s defaults — which are the locked
params.

**(b) Or replace** the block with the explicit override:
```yaml
GBP_NZD:
  bb_std: 2.0
  rsi_oversold: 35
  rsi_overbought: 65
  sl_multiplier: 2.5
  # source: locked-params broker-cost experiment (2026-05-22)
  # paper-trade window: 90 days OOS
```

Either approach produces the same runtime config.

---

## Start command

After applying both edits:

```bash
# verify env first
python scripts/diagnose.py
# expect: connected=true, account balance ~AUD 89,310, environment=practice

# start the system
python -m src.main
```

Leave it running for 90 days. The OS-level supervisor (systemd / Task
Scheduler / nohup, whatever you've used previously) keeps it alive across
restarts.

---

## Monitoring during the window

- **Daily**: tail `logs/trading.log` — check for stream-disconnect storms,
  circuit-breaker fires, or pair-specific rejections.
- **Weekly**: query `data/trades.db` for the running GBP_NZD trade count.
  Target: 30+ trades per month (the locked-params backtest averaged ~38/month
  on GBP_NZD over 4 years).
- **Monthly**: run the re-cost script in DRY-RUN mode (below) to see how the
  PSR(0) is trending.

```bash
# Sample weekly query
sqlite3 data/trades.db "
  SELECT COUNT(*), ROUND(SUM(pnl), 2)
  FROM trades
  WHERE instrument = 'GBP_NZD'
    AND entry_time >= '2026-05-23';
"
```

---

## End-of-window analysis

After ~90 days, run the re-cost analysis:

```bash
python scripts/recost_paper_trades.py \
    --instrument GBP_NZD \
    --since 2026-05-23 \
    --combine-with data/locked_params_ecn/GBP_NZD.json
```

This:
1. Pulls all GBP_NZD trades from `data/trades.db` since the start date
2. Re-costs each one to ECN equivalents (add back OANDA round-trip spread
   savings, subtract ECN spread + 0.8 bps commission)
3. Combines the re-costed paper-trade PnLs with the locked-params backtest
   OOS PnLs
4. Recomputes PSR(0) on the combined series and prints the verdict

---

## Decision rule (conditional next step)

| Combined PSR(0) | Action |
|---|---|
| ≥ 0.95 (STRONG) | GBP_NZD earns a small live-money pilot at minimum size (1 unit, not 1% risk) at an ECN broker. Spec the pilot separately. |
| 0.85 ≤ PSR(0) < 0.95 (MARGINAL) | Continue paper-trading for another 90 days. Re-evaluate. |
| < 0.85 (DECAYED) | The 4-year backtest edge was an artefact. Move to the **regime diagnostic** as the next vector. |

Per the brief's escalation path: the regime diagnostic is queued **iff** the
paper-trade shows decay. If GBP_NZD holds up, the regime work is irrelevant —
we have a tradeable edge.

---

## Revert after the window

Once analysis is complete:

```bash
git checkout config/settings.yaml config/pair_params.yaml
```

(Assumes you committed the originals before editing. If not, restore from
your backup.)
