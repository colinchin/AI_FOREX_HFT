# AI Forex HFT — Automated Forex Scalping System

An event-driven, asyncio-based forex scalping system built for the OANDA v20 API. Trades G10 currency pairs on M5 timeframes using per-pair optimised mean-reversion signals with multi-layered risk management.

**Account currency:** AUD | **Environment:** OANDA Practice (configurable for Live) | **Python 3.11+**

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Strategies](#strategies)
- [Risk Management](#risk-management)
- [Backtesting](#backtesting)
- [Scripts & Tools](#scripts--tools)
- [Testing](#testing)
- [Live Trading Results](#live-trading-results)
- [Key Design Decisions](#key-design-decisions)
- [Known Constraints](#known-constraints)

---

## Architecture

The system follows an event-driven pipeline where each component communicates through an asyncio queue-based event bus. No component directly calls another — they publish and subscribe to typed events.

```
OANDA Stream(s)
     │
     ▼
StreamManager (×N chunks of ≤20 instruments)
     │
     ▼
TickEvent ──► TickBuffer (latest tick per instrument)
     │
     ▼
CandleBuilder (M1 → M5 → H1 cascade)
     │
     ▼
CandleEvent ──► Strategy (MeanReversion / MomentumScalp)
     │              │
     │              ▼
     │         MultiTimeframeFilter (H1 trend alignment)
     │              │
     │              ▼
     │         SignalEvent
     │              │
     │              ▼
     │         RiskManager ──► Filter Chain:
     │              │          1. SpreadFilter
     │              │          2. SpreadCostFilter
     │              │          3. SessionFilter
     │              │          4. VolatilityFilter
     │              │          5. NewsFilter (ForexFactory calendar)
     │              │
     │              ▼
     │         OrderEvent
     │              │
     │              ▼
     │         ExecutionHandler ──► OANDA REST API
     │              │
     │              ▼
     │         FillEvent ──► PortfolioTracker
     │                       RiskManager (position count)
     │                       TransactionPoller (register trade ID)
     │
     ▼
TransactionPoller (every 5s)
     │
     ▼
TradeCloseEvent / TradeReducedEvent ──► PortfolioTracker + RiskManager
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **EventBus** | `src/core/bus.py` | asyncio.Queue pub/sub — decouples all components |
| **StreamManager** | `src/api/stream.py` | Persistent OANDA price stream with auto-reconnect |
| **CandleBuilder** | `src/data/candle_builder.py` | Aggregates ticks into M1→M5→H1 candle cascade |
| **TickBuffer** | `src/data/tick_buffer.py` | Ring buffer of latest ticks per instrument |
| **MeanReversionStrategy** | `src/strategy/mean_reversion.py` | Primary strategy — BB + RSI on M5 |
| **MomentumScalpStrategy** | `src/strategy/momentum_scalp.py` | Secondary strategy (disabled — negative edge) |
| **MultiTimeframeFilter** | `src/strategy/multi_tf.py` | H1 EMA trend alignment gate |
| **RiskManager** | `src/risk/manager.py` | Central gatekeeper — filters, limits, sizing |
| **PositionSizer** | `src/risk/position_sizer.py` | ATR-based sizing with cross-currency conversion |
| **ExecutionHandler** | `src/api/execution.py` | OANDA order placement with atomic SL/TP |
| **TransactionPoller** | `src/api/transaction_poller.py` | Detects broker-side SL/TP/trailing stop closures |
| **PortfolioTracker** | `src/portfolio/tracker.py` | Open positions, PnL, deferred trailing stops |
| **TradeStore** | `src/data/store.py` | SQLite trade journal |

---

## Project Structure

```
AI_FOREX_HFT/
├── src/
│   ├── main.py                    # Entry point — system orchestrator
│   ├── api/
│   │   ├── client.py              # OANDA v20 REST client wrapper
│   │   ├── account.py             # Account state management
│   │   ├── execution.py           # Order execution + trailing stops
│   │   ├── stream.py              # Price streaming with reconnect
│   │   └── transaction_poller.py  # Broker-side closure detection
│   ├── core/
│   │   ├── bus.py                 # Event bus (asyncio.Queue pub/sub)
│   │   ├── events.py              # Event dataclasses
│   │   └── models.py              # Domain models (Position, Candle, DailyStats)
│   ├── data/
│   │   ├── tick_buffer.py         # Latest-tick ring buffer
│   │   ├── candle_builder.py      # Tick → candle aggregation
│   │   ├── history.py             # Historical data fetcher + parquet cache
│   │   └── store.py               # SQLite trade persistence
│   ├── strategy/
│   │   ├── base.py                # Abstract Strategy interface
│   │   ├── mean_reversion.py      # BB + RSI mean reversion (primary)
│   │   ├── momentum_scalp.py      # EMA + MACD momentum (disabled)
│   │   ├── indicators.py          # Incremental technical indicators
│   │   └── multi_tf.py            # H1 trend filter
│   ├── risk/
│   │   ├── manager.py             # Central risk manager
│   │   ├── filters.py             # Spread, session, volatility, news filters
│   │   └── position_sizer.py      # ATR-based position sizing
│   ├── portfolio/
│   │   └── tracker.py             # Position tracking + deferred trailing stops
│   └── utils/
│       ├── config.py              # YAML config loader
│       ├── helpers.py             # Pip conversion, currency conversion, time utils
│       └── logger.py              # structlog JSON logging
├── backtest/
│   ├── engine.py                  # Backtest engine + ConversionRateCache
│   ├── simulator.py               # Order simulation with spread/slippage
│   ├── data_loader.py             # Historical data loading
│   └── metrics.py                 # PF, Sharpe, MaxDD, win rate
├── scripts/
│   ├── run_backtest.py            # Single-pair backtest runner
│   ├── download_history.py        # Bulk historical data downloader
│   ├── backtest_all_pairs.py      # Batch backtest all G10 pairs
│   ├── optimize_pairs.py          # Per-pair grid search optimiser
│   ├── analyze_time_exit.py       # Time-exit analysis
│   └── diagnose.py                # OANDA connectivity diagnostics
├── tests/                         # 92 tests, all passing
├── config/
│   ├── settings.yaml              # System config (instruments, strategies, sessions)
│   ├── risk.yaml                  # Risk parameters
│   ├── pair_params.yaml           # Per-pair optimised parameters (auto-generated)
│   └── .env                       # OANDA credentials (not committed)
├── data/
│   ├── parquet/                   # Cached M5 + H1 candle history
│   └── trades.db                  # SQLite trade journal
├── logs/
│   └── trading.log                # JSON-formatted event log
└── hft-forex-design.md            # Full design document
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- OANDA Practice or Live account
- OANDA v20 API access token

### Installation

```bash
git clone <repo-url>
cd AI_FOREX_HFT
pip install -e ".[dev]"
```

### Environment Setup

Create `config/.env` with your OANDA credentials:

```env
OANDA_ACCOUNT_ID=your-account-id
OANDA_ACCESS_TOKEN=your-api-token
```

These are loaded automatically at startup. Never commit this file.

### Verify Connectivity

```bash
python scripts/diagnose.py
```

This checks API credentials, account balance, and stream connectivity.

### Download Historical Data

```bash
python scripts/download_history.py
```

Downloads M5 and H1 candle history for all configured instruments. Data is cached in `data/parquet/`.

### Run Backtests

```bash
# Single pair
python scripts/run_backtest.py --instrument EUR_USD

# All configured pairs
python scripts/backtest_all_pairs.py

# Per-pair parameter optimisation (generates pair_params.yaml)
python scripts/optimize_pairs.py
```

### Start Live Trading

```bash
python src/main.py
```

The system will:
1. Connect to OANDA and verify the account
2. Reconcile any open positions from a prior session
3. Fetch the news calendar from ForexFactory
4. Warm up indicators with 7 days of M5 and 30 days of H1 history
5. Start price streams and begin trading

Stop with `Ctrl+C` for graceful shutdown.

---

## Configuration

### `config/settings.yaml`

Controls which instruments to trade, which sessions are active, and strategy parameters.

```yaml
environment: practice  # practice | live

instruments:
  # 11 qualified G10 pairs (optimizer-filtered, test PF >= 1.0)
  - EUR_USD
  - USD_JPY
  - GBP_USD
  - USD_CHF
  - AUD_USD
  - NZD_USD
  - EUR_GBP
  - EUR_JPY
  - EUR_CHF
  - GBP_AUD
  - NZD_CAD

strategy:
  primary: mean_reversion
  mean_reversion:
    bb_period: 20
    bb_std: 2.0         # Global default — overridden per-pair in pair_params.yaml
    rsi_period: 14
    rsi_oversold: 35
    rsi_overbought: 65
    atr_period: 14
    sl_multiplier: 2.5
    pair_params_file: "config/pair_params.yaml"

sessions:
  active_sessions: [london, new_york, overlap]
```

### `config/risk.yaml`

All risk parameters in one place. See [Risk Management](#risk-management) for details.

### `config/pair_params.yaml`

Auto-generated by `scripts/optimize_pairs.py`. Per-pair overrides for `bb_std`, `rsi_oversold`, `rsi_overbought`, and `sl_multiplier`. Only pairs with test PF >= 1.0 are included.

---

## Strategies

### Mean Reversion (Primary — Active)

Fades extreme M5 moves back toward the Bollinger Band midline.

**Entry conditions (LONG):**
- Price closes at or below the lower Bollinger Band
- RSI ≤ oversold threshold (per-pair, typically 25-35)
- Not a session-range breakout (range position > 5%)
- H1 multi-timeframe trend aligned (via MTF filter)

**Exit:**
- **TP:** Bollinger middle band (SMA20)
- **SL:** ATR × sl_multiplier (per-pair, typically 1.5-3.5)

**Per-pair optimisation:** 13 pairs have custom parameters via grid search with 70/30 train/test split. Most pairs converge on `bb_std=2.25-2.50`, `RSI 25/75` (tighter entries than global defaults).

### Momentum Scalp (Disabled)

Trend-following M5 scalper using EMA crossovers + MACD + RSI momentum. **Disabled in production** — backtest PF 0.88 (negative edge at realistic spreads). Generated 33 trades/day of unprofitable churn in live testing.

If re-enabled, this strategy has:
- Deferred trailing-stop activation (arms after profit exceeds configurable ATR multiple)
- Per-strategy time-exit (45 minutes default)
- Mandatory H1 EMA(50) trend alignment (blocks signals until warmed)

---

## Risk Management

The RiskManager is the central gatekeeper between strategy signals and order execution. Every signal passes through a sequential filter chain, position limits, equity checks, and position sizing before an order is emitted.

### Filter Chain

| Filter | Purpose | Config |
|--------|---------|--------|
| **SpreadFilter** | Reject when current spread > threshold | `max_spread_pips: 5.0` |
| **SpreadCostFilter** | Reject when spread > 25% of TP distance | `max_spread_cost_pct: 0.25` |
| **SessionFilter** | Only trade during London, New York, Overlap | `active_sessions` in settings.yaml |
| **VolatilityFilter** | Reject when ATR > 2σ above 50-candle mean | `max_atr_std_devs: 2.0` |
| **NewsFilter** | Block trading around high-impact economic events | ForexFactory calendar, ±15 min window |

### News Filter

Fetches the ForexFactory weekly economic calendar via the Fair Economy mirror API. Blocks new trades when a high-impact event affects either currency in the traded pair.

- **Source:** `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
- **Caching:** Refreshed every hour (configurable)
- **Fail-closed:** Blocks all signals if the calendar has never been fetched or is stale (>2× cache TTL)
- **Currency matching:** `EUR_USD` is blocked by both EUR and USD events
- **Wall-clock time:** Checks current time, not signal creation time

### Position Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Max open positions | 10 | Across all instruments |
| Max trades/day | 150 | Counted on entry (fill), not close |
| Same-instrument block | — | No opposing position on same pair |
| Daily loss halt | 3% | Halts + flattens all positions |
| Min equity | $100 | Below this = halt |
| Drawdown circuit breaker | 5% | From session start equity |
| Consecutive loss cooldown | 5 losses → 15 min | Resets on any win (including partial closes) |

### Position Sizing

ATR-based fixed-fractional sizing with proper cross-currency conversion:

1. **Risk amount** = equity × `max_risk_per_trade` (1%)
2. **Effective SL** = SL distance + current spread
3. **SL in AUD** = converted via `get_conversion_rate()` with triangulation (up to 3 legs through G10 currencies)
4. **Units** = risk amount / SL in AUD
5. **Notional cap** = equity × `max_position_pct` (4%)
6. **Floor** = reject if units < `min_units` (100)

If the conversion rate is unavailable (missing tick data for conversion pairs), the trade is **rejected** rather than sized on an incorrect approximation.

### Emergency Mechanisms

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| **Circuit breaker** | All streams dead > 60s | Flatten all + halt |
| **Daily loss halt** | Net PnL < -3% of session equity | Flatten all + halt |
| **Equity halt** | Equity < $100 | Flatten all + halt |
| **Weekend close** | Friday 4:00 PM New York (60 min before close) | Flatten all + halt until Sunday 5 PM |
| **API error breaker** | 10 consecutive REST failures | Halt |

All flatten paths use `_emergency_flatten()` with retry escalation: 3 immediate attempts, then background retries every 30s up to 10 times.

### Partial Trade Close Reconciliation

If a trade is partially closed (broker-side or manual intervention), the system:
- Emits a `TradeReducedEvent` with the reduced units and realised PnL
- Updates `Position.units` in PortfolioTracker
- Books partial PnL through `DailyStats.record_trade()` (updates win/loss counts, profit factor, consecutive-loss tracking)
- Does NOT remove the position or decrement the open count — the remainder is still live

Full reductions (`remaining_units == 0`) emit only a `TradeCloseEvent` to avoid double-counting.

### Restart Resilience

On startup, the system reconciles state from the broker:

- **Open positions** seeded into RiskManager (count, instruments) and PortfolioTracker (full Position objects)
- **Strategy name** restored from OANDA `clientExtensions.tag` — preserved for per-strategy time-exit logic
- **Trailing stop state** restored: if broker has an active trailing stop → `trailing_armed=True`; if not yet triggered → `trailing_activate_distance` and `trailing_stop_distance` restored from tag for deferred activation
- **Trade IDs** registered with TransactionPoller

---

## Backtesting

The backtesting engine uses the same strategy and risk code as live trading, with a simulated execution layer.

```bash
python scripts/run_backtest.py --instrument EUR_USD --from 2024-09-01 --to 2025-03-01
```

### Features

- **Same code path:** Strategies, indicators, and PositionSizer are shared between live and backtest
- **Realistic spread:** Per-pair spread modelling from historical data
- **AUD conversion:** `ConversionRateCache` with point-in-time lookups (no look-ahead bias)
- **Slippage:** Configurable slippage model
- **Metrics:** Profit factor, Sharpe ratio, max drawdown, win rate, expectancy

### Optimisation

```bash
python scripts/optimize_pairs.py
```

Runs a grid search over `bb_std`, `rsi_oversold`/`rsi_overbought`, and `sl_multiplier` for each pair with a 70/30 train/test split. Results are written to `config/pair_params.yaml` — only pairs meeting the test PF threshold are included.

---

## Scripts & Tools

| Script | Purpose |
|--------|---------|
| `scripts/run_backtest.py` | Backtest a single pair with custom parameters |
| `scripts/backtest_all_pairs.py` | Batch backtest all 28 G10 pairs, ranked by PF |
| `scripts/optimize_pairs.py` | Per-pair grid search optimiser → `pair_params.yaml` |
| `scripts/download_history.py` | Download M5 + H1 history to parquet cache |
| `scripts/analyze_time_exit.py` | Test time-exit cutoffs (conclusion: no exit is optimal) |
| `scripts/diagnose.py` | Verify OANDA connectivity and account setup |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_news_filter.py -v
```

**92 tests** across 7 test files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_candle_builder.py` | 5 | M1→M5→H1 cascade, caching, multi-instrument |
| `test_execution.py` | 7 | Order execution, SL/TP, short positions, equity tracking |
| `test_indicators.py` | 13 | BB, RSI, ATR, MACD, EMA (batch + incremental) |
| `test_news_filter.py` | 27 | Calendar parsing, blackout windows, boundaries, fail-closed, staleness, refresh |
| `test_risk_manager.py` | 9 | Position sizing, signal filtering, loss halt, cooldown |
| `test_strategy.py` | 9 | Warmup, signal generation, MTF alignment |
| `test_trailing_and_reduction.py` | 14 | Deferred trailing activation, API retry, partial reduction, reconciliation |

Test framework: pytest with `asyncio_mode=auto`.

---

## Live Trading Results

### Day 1 (March 18, 2026)
- 37 round-trip trades, net PL **+$47.03** (broker-confirmed)
- Natural profit factor (TP/SL only): **1.75**
- USD_CHF: 66% of profits (+$31.16)
- Spread cost: $22.79 = 33% of net PL

### Day 2-3 (March 18-19, 2026)
- 20 additional trades, net PL **+$43.28**, PF **1.91**
- Spread cost improved to 20.7% of gross profit (down from 33%)
- Account balance: AUD $89,283

---

## Key Design Decisions

**Why event-driven?** Decouples components for testability and allows the same strategy/risk code to run in both live and backtest modes without modification.

**Why M5 for signals?** M1 is too noisy and spread-sensitive. M5 gives enough signal quality while still being fast enough for scalping. H1 is used only for trend alignment (MTF filter).

**Why fail-closed on news?** A missing news calendar is more dangerous than blocking trades. If the system can't verify there's no upcoming NFP, it should not trade. Blocking resumes automatically once the calendar is fetched.

**Why reject trades on missing conversion rates?** An incorrect position size on a cross pair (e.g., EUR_GBP with AUD account) can easily be 2-5x wrong. Rejecting one trade is better than risking the wrong amount.

**Why store strategy metadata in OANDA clientExtensions?** The system needs to survive restarts without losing per-trade context (strategy name, deferred trailing-stop thresholds). OANDA's `clientExtensions.tag` field (128 chars) round-trips through the broker, making it the only reliable persistence for trade-level metadata.

**Why deferred trailing-stop activation?** The momentum strategy's trailing stop should only arm after the trade reaches a profit threshold (e.g., 1× ATR). Setting it immediately at fill would cut trades far too early. The activation check runs on every tick in PortfolioTracker, with retry on API failure.

---

## Known Constraints

- **OANDA stream drops** every ~5 minutes and reconnects in 1-2 seconds. The circuit breaker uses a 60-second heartbeat timeout — transient disconnects are normal and must not trigger emergency flatten.
- **EUR_NZD / GBP_NZD** have live spreads 2-3x wider than backtested (6-10 vs 3-4 pips). These are excluded from the active instrument list.
- **Momentum scalp** has a negative edge at realistic spreads (PF 0.88). It remains in the codebase but is disabled in config. If spreads tighten (e.g., Core pricing account), it may become viable.
- **clientExtensions.tag** is limited to 128 characters. The current JSON payload (`{"s":"strategy_name","td":0.0005,"ta":0.001}`) fits comfortably, but adding more fields should be done carefully.
- **Partial close reconciliation** handles broker-side reductions but does not track which portion of the original position was closed. The remaining position retains the original entry price and strategy assignment.

---

## Dependencies

```
oandapyV20>=0.7.2       # OANDA v20 API wrapper
pandas>=2.0              # Data manipulation
numpy>=1.24              # Numerical computing
pandas-ta>=0.3.14b       # Technical indicators (batch)
python-dotenv>=1.0       # Environment variable loading
pyyaml>=6.0              # YAML config parsing
structlog>=24.0          # Structured JSON logging
aiohttp>=3.9             # Async HTTP (news calendar, etc.)
aiosqlite>=0.19          # Async SQLite trade journal
```

Dev dependencies: `pytest`, `pytest-asyncio`, `pytest-cov`, `matplotlib`

---

## License

Private project. Not for redistribution.
