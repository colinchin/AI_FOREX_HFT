# High-Frequency Forex Trading System — Design & Coding Plan

## OANDA Practice Account POC | Python | Claude Code Build Guide

**Author:** Colin (teckfuh@gmail.com)
**Date:** March 2026
**Status:** Implementation Complete — Paper Trading Phase

---

## Part 1: Research Findings

### 1.1 What "High-Frequency" Actually Means for Retail Forex

True institutional HFT operates in microseconds with co-located servers and direct market access. That world requires millions in infrastructure. What's realistic and profitable for a retail trader using OANDA's REST API is **high-frequency scalping** — a hybrid approach that borrows HFT principles (speed, automation, microstructure awareness, statistical edge) while operating at a timescale of seconds-to-minutes rather than microseconds.

The research is clear on this distinction. OANDA's average execution speed is approximately 12 milliseconds, and the REST API adds network round-trip latency on top. The platform supports up to 1,000 open trades/orders per account. This makes OANDA excellent for automated scalping but unsuitable for true sub-millisecond HFT.

### 1.2 Proven Strategy Categories That Work at Retail Scale

Based on current research across QuantStart, academic papers, and verified live trading systems, these are the viable strategy families:

**Momentum Scalping** — Trade in the direction of short-term momentum on M1–M5 timeframes during high-volume sessions (London–NY overlap). Use breakout detection and ride brief directional moves. Exit once momentum fades. This is the most accessible strategy for automation.

**Mean-Reversion Scalping** — Fade extreme micro-moves back to the mean. Uses Bollinger Bands, RSI extremes (below 30 / above 70), and order flow imbalances. Works best on M5 and M15 with confirmation filters. Statistically robust when spreads are tight.

**Multi-Timeframe Alignment** — Short timeframe scalps (M1/M5) are only taken when aligned with H1 or H4 trend bias. This reduces countertrend noise significantly and is one of the most consistently profitable approaches documented.

**Statistical Arbitrage (Cross-Pair)** — Exploit temporary price divergences between correlated pairs (e.g., EUR/USD and GBP/USD, or EUR/JPY and USD/JPY). When correlation breaks down temporarily, trade the convergence. Requires monitoring multiple streams simultaneously.

**Session-Based Microstructure Plays** — Exploit predictable liquidity patterns at session opens (Tokyo, London, NY). For example, USD/JPY shows predictable liquidity gaps at the Tokyo open followed by aggressive order flow. Enter on price momentum and exit within seconds.

### 1.3 Real Systems and Open-Source References

| Project | Description | URL |
|---------|-------------|-----|
| QuantStart Event-Driven Engine | Multithreaded event-driven forex engine with OANDA API. Streaming prices, portfolio management, execution handler. | quantstart.com |
| oandapyV20-examples | Official community examples: concurrent streaming, gevent greenlets, order management. | github.com/hootnot/oandapyV20-examples |
| FX-Trading-with-Python-and-Oanda | Full course codebase by Anthony Ng (Nanyang Polytechnic). Covers instrument listing through automated trade execution. | github.com/anthonyng2/ |
| forex-ai-trader (GlitchSwitch) | 2025 series building an AI-powered forex bot with OANDA. Covers data pagination, deduplication, rate limiting, exponential backoff. | github.com/Viajante80/ |
| QSTrader | Open-source backtesting and live trading engine in Python. Supports both retail and institutional strategies. | quantstart.com/qstrader |
| dpf205/python-trading-algo | Python FX trading via OANDA V20 REST API with configparser-based credential management. | github.com/dpf205/ |

### 1.4 OANDA v20 API Architecture

The system will use two Python libraries:

- **`oandapyV20`** (community, pip install oandapyV20) — Mature wrapper with streaming support, request objects for every endpoint, and contrib helpers for order construction. Last updated v0.7.2.
- **`v20`** (official OANDA, pip install v20) — OANDA's own bindings. Simpler but less feature-rich.

**Recommended: `oandapyV20`** for its streaming support, error handling, and extensive examples.

Key API endpoints:

- **Streaming Prices:** `pricing.PricingStream` — Long-lived HTTP connection that pushes tick-by-tick bid/ask updates. This is the primary data feed.
- **Historical Candles:** `/v3/instruments/{instrument}/candles` — Up to 5,000 candles per request. Historical data back to 2005. Requires pagination for large datasets.
- **Order Placement:** `/v3/accounts/{accountId}/orders` — Market, limit, stop, trailing stop orders.
- **Trade Management:** `/v3/accounts/{accountId}/trades/{tradeID}` — Modify TP/SL, close trades.
- **Account State:** Poll account updates via TransactionID to maintain a consistent account snapshot without re-fetching everything.

**Practice environment:** `api-fxpractice.oanda.com` port `443`.

---

## Part 2: Critical Considerations

### 2.1 Spread Is the #1 Enemy

OANDA is a market maker. The average EUR/USD spread on a Standard account was 1.69 pips in August 2025. At Core Pricing, it's tighter but adds a commission. If you're targeting 5-pip gains per trade, a 1.7 pip spread means the market must move 6.7 pips in your favour just to hit target. This fundamentally shapes strategy design:

- **Trade qualified G10 pairs:** 14 pairs qualified after per-pair optimization with realistic spreads (1.0–4.0 pips). High-spread crosses (CHF crosses at 4.5 pips) are excluded — spread consumes the entire edge.
- **Only trade during peak sessions:** London open (08:00–17:00 GMT), NY open (13:00–22:00 GMT), and especially the London–NY overlap (13:00–17:00 GMT).
- **Factor spread into every backtest.** A strategy that looks profitable at 0 spread will often be a net loser at 1.5 pips. Per-pair spread modeling is essential.
- **Consider Core Pricing account** if available — lower spreads + fixed commission is usually cheaper for high-frequency.

### 2.2 Slippage and Execution Latency

OANDA's 12ms average execution is good for retail, but network latency from Sydney to OANDA's servers (likely US/UK) adds 150–300ms round-trip. This means:

- Your effective latency is 160–310ms per order.
- Aggressive scalping on M1 with 2-pip targets will suffer from slippage during volatile moments.
- Use **limit orders** where possible instead of market orders to control fill price.
- OANDA v20 supports a **price bound** on market orders to prevent adverse fills.
- Consider a VPS in the US or UK to reduce latency if moving to live trading.

### 2.3 Rate Limits and Connection Management

- OANDA limits simultaneous connections and request rates.
- The streaming connection should be kept alive persistently — don't reconnect for every tick.
- Implement exponential backoff on API errors.
- The `PricingStream` endpoint is the correct way to receive prices — don't poll the pricing endpoint repeatedly.
- Only one streaming connection per account is recommended. Use it to stream all instruments you trade.

### 2.4 Risk Management — Non-Negotiable

This is where most retail automated systems fail. Research consistently shows that without strict risk controls, high-frequency systems amplify losses faster than they generate profits.

**Per-Trade Controls:**
- Maximum risk per trade: 1–2% of account equity.
- Hard stop-loss on every trade — no exceptions.
- Take-profit targets should be at minimum 1:1 risk:reward, ideally 1.5:1 or better.
- Use OANDA's `stopLossOnFill` and `takeProfitOnFill` parameters so stops are atomic with order placement.

**Session Controls:**
- Maximum daily loss limit (e.g., 3% of equity) — system halts trading for the day.
- Maximum consecutive losses before pause (e.g., 5 losses → 15 min cooldown).
- Maximum open positions at any time (e.g., 3).
- Maximum trades per day cap.

**System Controls:**
- Circuit breaker on connectivity loss — close all positions if stream drops for >30 seconds.
- Heartbeat monitoring on the price stream (OANDA sends heartbeats in the stream).
- Automatic graceful shutdown with position cleanup.

### 2.5 Backtesting Honesty

The gap between backtest and live performance is the graveyard of retail algo traders. Critical rules:

- **Use tick-level or M1 data** — not H1 bars for a scalping strategy.
- **Include realistic spread** — use average spread for the session, not the tightest possible.
- **Model slippage** — add 0.5–1 pip adverse slippage to every market order fill.
- **Walk-forward analysis** — optimise on training window, validate on out-of-sample window, roll forward.
- **Monte Carlo simulation** — randomise trade order to estimate range of drawdown outcomes.
- **Beware overfitting** — if your strategy has >5 tuneable parameters, you're probably curve-fitting.

### 2.6 Realistic Profitability Expectations

Research suggests that well-designed retail scalping systems can target 3–6% monthly returns on a good month with adequate capital ($5,000+). Only about 12% of micro-spread trading opportunities are profitable after all costs. The win rate needed is high — a system targeting 5 pips with 2 pip spread needs roughly 60%+ win rate to break even.

This is a POC on a practice account. The goal is to validate the system architecture and strategy edge before risking real capital. Treat the practice account seriously — trade as if it's real money.

---

## Part 3: System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      OANDA REST v20 API                          │
│              api-fxpractice.oanda.com:443                        │
└──────┬──────────────────┬───────────────────┬──────────┬────────┘
       │ Price Streams     │ Order Execution    │ Account  │ Transactions
       ▼                  ▼                    ▼          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ StreamManager  │  │ Execution    │  │ AccountManager     │   │
│  │ ×N (chunked,   │  │ Handler      │  │ (poll updates,     │   │
│  │  20 pairs max  │  │ (rate-aware, │  │  track P&L,        │   │
│  │  per stream)   │  │  retry,      │  │  margin)           │   │
│  │                │  │  price bound)│  │                    │   │
│  ├────────────────┤  └──────┬───────┘  └──────┬─────────────┘   │
│  │ Transaction    │         │                 │                  │
│  │ Poller         │         │                 │                  │
│  │ (SL/TP detect) │         │                 │                  │
│  └──────┬─────────┘         │                 │                  │
└─────────┼───────────────────┼─────────────────┼──────────────────┘
          │                   │                 │
          ▼                   ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Event Bus (asyncio.Queue)                     │
│  TickEvent | CandleEvent | SignalEvent | OrderEvent | FillEvent  │
│  TradeCloseEvent | ErrorEvent                                    │
└──────────┬───────────────────────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────────┐
     ▼                                    ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│  Strategy Engine    │  │       Risk Manager                │
│  ┌───────────────┐  │  │  - Per-trade position sizing      │
│  │ Candle Builder│  │  │  - Daily loss limit (3%)          │
│  │ M1→M5→H1     │  │  │  - Max 10 open positions          │
│  ├───────────────┤  │  │  - Consecutive loss cooldown       │
│  │ Mean Reversion│  │  │  - Circuit breaker (all streams)  │
│  │ (per-pair     │  │  │  - Spread / session / vol filters │
│  │  optimized)   │  │  └───────────────────────────────────┘
│  ├───────────────┤  │
│  │ Multi-TF      │  │  ┌───────────────────────────────────┐
│  │ H1 Alignment  │  │  │       Portfolio Tracker            │
│  └───────────────┘  │  │  - Open positions + unrealised P&L│
└─────────────────────┘  │  - Trade history (SQLite)          │
                         │  - Running win rate, PF, Sharpe    │
     ┌───────────────────┤  - Per-instrument breakdown        │
     ▼                   └───────────────────────────────────┘
┌─────────────────────┐  ┌───────────────────────────────────┐
│  Data Store         │  │       Logger / Monitor             │
│  - Tick buffer (10K)│  │  - Structured JSON logging         │
│  - Candle cache     │  │  - 60s status reports              │
│  - Trade log (SQL)  │  │  - Stream health monitoring        │
│  - Parquet history  │  │  - Per-pair performance tracking   │
└─────────────────────┘  └───────────────────────────────────┘
```

### 3.2 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.12+ | Your existing expertise, OANDA ecosystem support |
| Async Runtime | `asyncio` + `aiohttp` | Non-blocking I/O for concurrent stream + execution |
| OANDA Wrapper | `oandapyV20` | Mature, streaming support, community examples |
| Data Analysis | `pandas`, `numpy` | Indicator calculation, candle aggregation |
| Technical Indicators | `ta-lib` or `pandas-ta` | RSI, MACD, Bollinger Bands, EMA |
| Database | SQLite (dev) → PostgreSQL (prod) | Trade logging, performance tracking |
| Configuration | YAML + environment variables | Secrets in env, strategy params in YAML |
| Logging | Python `logging` + `structlog` | Structured JSON logs for analysis |
| Testing | `pytest` + `pytest-asyncio` | Unit tests, strategy backtesting |
| Backtesting | Custom engine (same signal logic) | Ensures parity between backtest and live |

### 3.3 Module Structure

```
AI_FOREX_HFT/
├── config/
│   ├── settings.yaml          # Strategy parameters, 14 instruments, sessions
│   ├── risk.yaml              # Risk management rules
│   ├── pair_params.yaml       # Per-pair optimized strategy params (auto-generated)
│   └── .env                   # OANDA_ACCOUNT_ID, OANDA_ACCESS_TOKEN
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point, multi-stream orchestration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── client.py          # OANDA API client wrapper (auth, rate limits)
│   │   ├── stream.py          # Price streaming with heartbeat monitoring
│   │   ├── execution.py       # Order placement, modification, closure
│   │   ├── account.py         # Account state polling and tracking
│   │   └── transaction_poller.py  # Poll /transactions for broker-side SL/TP
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py          # Event classes (Tick, Candle, Signal, Order, Fill, TradeClose, Error)
│   │   ├── bus.py             # Event bus (asyncio.Queue based)
│   │   └── models.py          # Data models (Candle, Position, Trade, Order)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tick_buffer.py     # Ring buffer for recent ticks (10K per instrument)
│   │   ├── candle_builder.py  # Aggregate ticks → M1/M5/H1 candles in real-time
│   │   ├── history.py         # Historical data fetcher with pagination + parquet cache
│   │   └── store.py           # SQLite trade log (aiosqlite)
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract strategy interface
│   │   ├── indicators.py      # Batch + incremental: EMA, RSI, MACD, BB, ATR
│   │   ├── momentum_scalp.py  # Momentum scalping (secondary, PF ~0.88)
│   │   ├── mean_reversion.py  # Mean-reversion (primary, per-pair optimized)
│   │   └── multi_tf.py        # Multi-timeframe H1 alignment filter
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── manager.py         # Risk manager (sizing, limits, circuit breaker)
│   │   ├── position_sizer.py  # 4% notional cap, ATR-based sizing
│   │   └── filters.py         # Spread filter, session filter, volatility filter
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── tracker.py         # Open positions, P&L, trade history
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Structured JSON logging (structlog)
│       ├── config.py          # YAML + env config loader
│       └── helpers.py         # Pip conversion, time helpers, etc.
├── backtest/
│   ├── __init__.py
│   ├── engine.py              # Backtesting engine (reuses strategy modules)
│   ├── data_loader.py         # Load parquet data → candle event streams
│   ├── simulator.py           # Simulated execution with spread/slippage
│   └── metrics.py             # Sharpe, Sortino, max drawdown, profit factor
├── scripts/
│   ├── download_history.py    # Bulk historical data download
│   ├── diagnose.py            # Connectivity + account validation
│   ├── run_backtest.py        # Single-pair backtest runner CLI
│   ├── backtest_all_pairs.py  # Batch backtest 28 G10 pairs with per-pair spreads
│   └── optimize_pairs.py     # Per-pair grid search optimizer (train/test split)
├── data/
│   └── parquet/               # Cached historical candles (M5, H1 per instrument)
├── results/                   # Backtest result JSON files
├── tests/
│   ├── test_indicators.py
│   ├── test_risk_manager.py
│   ├── test_candle_builder.py
│   ├── test_strategy.py
│   └── test_execution.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Part 4: Coding Plan for Claude Code

This is an ordered, phased plan. Each phase produces a working, testable increment. Provide this to Claude Code as the build roadmap.

### Phase 1: Foundation (Sessions 1–2) ✅

**Goal:** Establish project structure, configuration, OANDA connectivity, and validate the practice account.

```
TASKS:
1. Scaffold the project directory structure as shown above.
2. Create pyproject.toml with dependencies:
   - oandapyV20>=0.7.2
   - pandas>=2.0
   - numpy>=1.24
   - pandas-ta>=0.3.14b
   - python-dotenv>=1.0
   - pyyaml>=6.0
   - structlog>=24.0
   - aiohttp>=3.9
   - aiosqlite>=0.19
   - pytest>=8.0
   - pytest-asyncio>=0.23
3. Create config/settings.yaml with:
   - instruments: [EUR_USD, USD_JPY, GBP_USD]
   - practice environment URL
   - default granularities: [M1, M5, H1]
4. Create config/risk.yaml with:
   - max_risk_per_trade: 0.01 (1%)
   - max_daily_loss: 0.03 (3%)
   - max_open_positions: 3
   - max_consecutive_losses: 5
   - cooldown_seconds: 900
   - max_trades_per_day: 50
5. Create src/utils/config.py — YAML loader + dotenv integration.
6. Create src/utils/logger.py — structlog JSON logging.
7. Create src/api/client.py — OANDA API client:
   - Auth via env variables
   - Practice/live environment switching
   - Request wrapper with rate limit handling + exponential backoff
   - Connection health check
8. Create scripts/diagnose.py:
   - Verify token validity
   - Fetch account summary
   - List available instruments
   - Test pricing endpoint
   - Print connection latency
9. Write tests for config loading and API client.
```

**Acceptance:** `python scripts/diagnose.py` connects to OANDA practice, prints account details and current EUR/USD price.

### Phase 2: Data Pipeline (Sessions 3–4) ✅

**Goal:** Real-time tick streaming, candle aggregation, and historical data access.

```
TASKS:
1. Create src/core/events.py — Event dataclasses:
   - TickEvent(instrument, bid, ask, timestamp, spread)
   - CandleEvent(instrument, timeframe, open, high, low, close, volume, timestamp)
   - SignalEvent(instrument, direction, strength, strategy_name, timestamp)
   - OrderEvent(instrument, direction, units, order_type, sl, tp, timestamp)
   - FillEvent(instrument, direction, units, fill_price, timestamp)
   - ErrorEvent(source, message, timestamp)
2. Create src/core/bus.py — AsyncIO event bus:
   - asyncio.Queue based
   - publish(event) and subscribe(event_type, handler) pattern
   - Support multiple subscribers per event type
3. Create src/api/stream.py — Price stream manager:
   - Persistent connection via PricingStream
   - Heartbeat monitoring (OANDA sends heartbeats)
   - Auto-reconnect with exponential backoff on disconnect
   - Parse ticks → TickEvent and publish to event bus
   - Track stream health metrics (ticks/second, last heartbeat)
4. Create src/data/tick_buffer.py — Ring buffer:
   - Fixed-size deque per instrument
   - Store last N ticks (default 10,000)
   - Methods: latest(), window(n), spread_avg(n)
5. Create src/data/candle_builder.py — Real-time candle aggregation:
   - Subscribe to TickEvents
   - Build M1 candles from ticks in real-time
   - Build M5 candles from M1 candles
   - Emit CandleEvent when candle closes
   - Handle partial candles (current open candle)
6. Create src/data/history.py — Historical data fetcher:
   - Paginated candle download (5,000 per request limit)
   - Deduplication on timestamps
   - Rate limit aware
   - Save to pandas DataFrame
   - Cache to local parquet files
7. Create scripts/download_history.py:
   - CLI to download historical data for specified instruments/timeframes/date ranges
   - Save as parquet files for backtesting
8. Write tests for tick buffer, candle builder (feed synthetic ticks, verify candles).
```

**Acceptance:** System connects to OANDA, streams EUR/USD ticks, prints real-time M1 candles as they close, and can download 1 year of M5 historical data.

### Phase 3: Strategy Engine (Sessions 5–7) ✅

**Goal:** Implement indicator calculations, strategies, and multi-timeframe filter.

**Status:** Complete. Mean reversion is the primary (profitable) strategy. Momentum scalp was implemented but is marginally negative (PF ~0.88) — the edge is consumed by spread.

```
IMPLEMENTED:
1. src/strategy/indicators.py:
   - Dual implementation: batch (pandas Series) + incremental (O(1) per update)
   - IncrementalEMA, IncrementalRSI (Wilder's smoothing), IncrementalMACD
   - IncrementalBollingerBands (running sum/sum-of-squares for O(1) updates)
   - IncrementalATR
2. src/strategy/base.py — Abstract strategy interface:
   - on_candle(CandleEvent) → Optional[SignalEvent]
   - on_tick(TickEvent) — optional override for intra-candle logic
   - warmup(candles) — seed indicators with historical data
   - required_history property
3. src/strategy/mean_reversion.py — PRIMARY strategy (profitable):
   ENTRY LONG:
   - M5 candle close <= lower Bollinger Band (period=20, std=per-pair)
   - RSI(14) <= oversold threshold (per-pair, typically 25)
   - Not at session extreme (breakout filter: range_position 5-95%)
   ENTRY SHORT:
   - Mirror conditions with upper BB + RSI >= overbought threshold
   EXIT:
   - TP: Bollinger middle band (SMA)
   - SL: ATR(14) × sl_multiplier (per-pair, typically 1.5–3.0)
   PER-PAIR OPTIMIZATION:
   - bb_std, rsi_oversold/overbought, sl_multiplier loaded from pair_params.yaml
   - Instruments without overrides fall back to global defaults
   - Session range resets on date change
4. src/strategy/momentum_scalp.py — SECONDARY (marginally negative):
   - EMA(9/20) crossover + RSI + MACD confirmation
   - TP: 3x ATR, SL: 2x ATR, trailing stop
   - PF ~0.88 across all pairs — spread consumes edge
   - Kept in codebase but not recommended for live trading
5. src/strategy/multi_tf.py — H1 trend alignment filter:
   - EMA(50) on H1 determines bullish/bearish/neutral bias
   - Signals rejected if M5 direction opposes H1 bias
6. 50 tests across 5 test files, all passing
```

**Key findings:**
- Time-based exits (30-45 min) cut profitable trades — removed
- Trailing stop activation must use separate threshold from trail distance
- Mean reversion SL was buggy (negative sl_distance at BB touch) — fixed with ATR-based SL
- Most pairs converge on bb_std=2.25–2.50, RSI 25/75 (tighter entries than original 35/65)

**Acceptance:** Feed historical M5 data through the strategy, verify it produces sensible signals at expected price levels. Indicator values match reference calculations.

### Phase 4: Risk Management & Execution (Sessions 8–10) ✅

**Goal:** Position sizing, risk controls, and live order execution via OANDA.

**Status:** Complete. All risk controls implemented and tested.

```
TASKS:
1. Create src/risk/position_sizer.py:
   - Fixed fractional: risk X% of equity per trade
   - ATR-based: use ATR to determine SL distance, then size position
     so dollar risk = equity * risk_pct
   - Account for spread in position sizing
   - Minimum/maximum position size bounds
   - Round to OANDA's unit requirements
2. Create src/risk/filters.py:
   - SpreadFilter: reject signals when current spread > threshold
   - SessionFilter: only allow signals during configured trading hours
   - VolatilityFilter: reject when ATR is abnormally high (>2 std devs)
   - NewsFilter: reject during configured blackout windows
3. Create src/risk/manager.py — Central risk manager:
   - Receives SignalEvents from strategy
   - Applies all filters
   - Checks daily loss limit (halt if breached)
   - Checks max open positions
   - Checks consecutive loss counter → cooldown
   - If all checks pass: compute position size → emit OrderEvent
   - Circuit breaker: if stream drops > 30s, close all positions
   - Track all risk metrics in real-time
4. Create src/api/execution.py — Order execution handler:
   - Subscribe to OrderEvents
   - Place market orders with:
     - stopLossOnFill (atomic SL)
     - takeProfitOnFill (atomic TP)
     - priceBound (prevent adverse slippage)
   - Place trailing stop modifications
   - Handle partial fills
   - Emit FillEvent on success, ErrorEvent on failure
   - Retry logic with backoff for transient failures
   - Close position by trade ID
   - Close all positions (emergency)
5. Create src/api/account.py — Account state manager:
   - Initial account snapshot on startup
   - Poll account updates via TransactionID
   - Track: balance, unrealised P&L, margin used, open trade count
   - Provide current equity for position sizing
6. Create src/portfolio/tracker.py — Portfolio tracker:
   - Subscribe to FillEvents
   - Maintain list of open positions with entry price, SL, TP, P&L
   - Track closed trades with full details
   - Calculate running statistics: win rate, avg win, avg loss, profit factor
   - Persist to SQLite via store.py
7. Create src/data/store.py — Trade log database:
   - SQLite with aiosqlite
   - Tables: trades, daily_summary, risk_events
   - Insert trade on close
   - Query methods for performance analysis
8. Write tests:
   - Position sizer: verify sizing at various equity/risk levels
   - Risk manager: test daily loss halt, consecutive loss cooldown
   - Execution: mock OANDA API, verify order construction
   - Portfolio: verify P&L calculations
```

**Acceptance:** End-to-end flow works: stream tick → strategy signal → risk check → order placement → fill confirmation → portfolio update → database log. All on OANDA practice account.

### Phase 5: Backtesting Engine (Sessions 11–12) ✅

**Goal:** Rigorous backtesting using the same strategy and risk modules.

**Status:** Complete. Backtesting uses the same strategy/risk code as live trading with simulated execution.

```
IMPLEMENTED:
1. backtest/data_loader.py:
   - Load parquet historical data files
   - Convert to candle event streams (CandleEvent)
   - Per-instrument, per-timeframe loading
2. backtest/simulator.py — Simulated execution:
   - Configurable per-pair spread (realistic OANDA spreads: 1.0–4.5 pips)
   - Configurable slippage (default 0.5 pips)
   - Candle-direction-aware SL/TP checking (fixed SL-first bias)
   - Tracks equity curve, drawdown
3. backtest/engine.py — Backtest orchestrator:
   - Replays M5 + H1 candle events through strategy
   - Multi-TF filter integration
   - Shared PositionSizer with spread adjustment
4. backtest/metrics.py — Performance analysis:
   - Sharpe ratio, Sortino ratio, max drawdown
   - Profit factor, win rate, expectancy
   - Trade count, avg win/loss, max consecutive losses
5. scripts/run_backtest.py — Single-pair CLI runner
6. scripts/backtest_all_pairs.py — Batch backtest for 28 G10 pairs:
   - Per-pair realistic OANDA spreads from SPREAD_TABLE
   - Ranked summary table output
   - JSON export for comparison
   - --use-pair-params flag for optimized params
7. scripts/optimize_pairs.py — Per-pair parameter optimizer:
   - Grid search: 100 combos (5 bb_std × 4 RSI × 5 SL multiplier)
   - 70/30 train/test split for overfitting protection
   - Rejects pairs with test PF < 1.0
   - Auto-generates config/pair_params.yaml
```

**Key findings:**
- Simulator had SL-first bias (always checked SL before TP) — fixed with candle-direction logic
- SL/TP calculated from candle.close is correct (not fill price) — matches real trading
- Per-pair optimization doubled qualified pairs: 7 → 14 (uniform → optimized)

### Phase 6: Integration & Main Loop (Sessions 13–14) ✅

**Goal:** Wire everything together into the production event loop.

**Status:** Complete. System runs autonomously with multi-stream support for 14 pairs.

```
IMPLEMENTED:
1. src/main.py — TradingSystem orchestrator:
   - Load configuration + per-pair optimized params from YAML
   - Multi-stream architecture: instruments chunked into groups of 20
     per OANDA stream connection (configurable stream_chunk_size)
   - Indicator warmup: 7 days M5 + 30 days H1 history per instrument
   - Event bus wiring: Tick→CandleBuilder→Strategy→RiskManager→Execution
   - Transaction poller: polls OANDA /transactions for broker-side SL/TP
   - Graceful shutdown: SIGINT/SIGTERM (Unix) / KeyboardInterrupt (Windows)
   - Circuit breaker: only triggers when ALL streams unhealthy
2. Monitoring:
   - 60s status reports: equity, balance, open positions, daily PnL,
     win rate, stream health, bus throughput
   - Structured JSON logging to logs/trading.log
   - Per-stream health tracking (ticks received, reconnect count)
3. Strategy wiring:
   - Generic strategy map lookup from config (not hardcoded)
   - Multi-TF filter checks H1 alignment before publishing signals
   - Per-pair params loaded from config/pair_params.yaml
```

### Phase 7: Optimisation & Scaling (Sessions 15–16) ✅

**Goal:** Scale to multiple pairs, per-pair parameter optimization, operational robustness.

**Status:** Complete. Scaled from 3 to 14 qualified pairs with per-pair optimization.

```
IMPLEMENTED:
1. Multi-pair scaling:
   - 28 G10 pairs tested with realistic per-pair OANDA spreads
   - 14 pairs qualified (PF >= 1.05) with optimized params
   - Multi-stream: instruments chunked across StreamManager instances
2. Per-pair parameter optimization:
   - Grid search: 100 combinations per pair (bb_std × RSI × SL multiplier)
   - 70/30 train/test split — rejects overfitting (test PF < 1.0)
   - 13 pairs with custom params, 1 pair qualifies on global defaults
   - Uniform → optimized: 7 → 14 qualified pairs, +38% aggregate PnL
3. Risk scaling for multi-pair:
   - 4% notional cap per position (10 concurrent max = 40% total exposure)
   - Max 150 trades/day, max spread 5.0 pips
4. Code review and hardening:
   - 16 fixes across 2 review passes (see code_review_fixes.md)
   - TransactionPoller for broker-side closure detection
   - Consecutive-loss cooldown uses current streak (not max ever)
   - Momentum warmup no longer double-updates MACD

FUTURE:
- Performance dashboard (Flask/FastAPI)
- Walk-forward re-optimization (quarterly)
- M15 timeframe for high-spread pairs
- Volatility regime filter for trending pairs
```

---

## Part 5: Claude Code Prompting Guide

When working with Claude Code, provide each phase as a focused task. Here's how to structure your prompts:

### Session Pattern

```
Session N — Phase X: [Phase Name]

Context: We're building a high-frequency forex scalping system 
using OANDA's practice account. The project lives at ~/hft-forex/.
[Reference this design doc for architecture details.]

Previous work: [Summarise what's been built so far]

This session's tasks:
1. [Specific task from the phase]
2. [Specific task from the phase]
3. [Specific task from the phase]

Requirements:
- Python 3.12+, asyncio-based
- Type hints on all functions
- Docstrings on all public methods
- Write pytest tests alongside the code
- Follow the module structure in the design doc
- Use structlog for all logging
- Config from YAML + env vars (never hardcode secrets)
```

### Key Instructions for Claude Code

Include these standing instructions when working on this project:

1. **Never hardcode credentials.** All OANDA tokens and account IDs come from environment variables.
2. **Type everything.** Use dataclasses or Pydantic models for all data structures.
3. **Async first.** The main loop is asyncio-based. Use `async/await` throughout.
4. **Test alongside code.** Every module gets a corresponding test file.
5. **Log everything.** Every trade decision, every risk check, every API call outcome.
6. **Fail safe.** On any unexpected error, the system should close positions and halt — never silently continue.
7. **Config-driven.** Strategy parameters, risk limits, instrument lists — all in YAML, never in code.
8. **Same code for backtest and live.** Strategy and risk modules are environment-agnostic. Only the data source and execution differ.

---

## Appendix A: OANDA API Quick Reference

```
Practice URL:     api-fxpractice.oanda.com
Live URL:         api-fxtrade.oanda.com
Port:             443
Auth Header:      Authorization: Bearer {token}

Streaming Prices: GET /v3/accounts/{id}/pricing/stream?instruments=EUR_USD
Candles:          GET /v3/instruments/{instrument}/candles?granularity=M5&count=500
Place Order:      POST /v3/accounts/{id}/orders
Get Trades:       GET /v3/accounts/{id}/trades
Close Trade:      PUT /v3/accounts/{id}/trades/{tradeId}/close
Account Summary:  GET /v3/accounts/{id}/summary
Account Changes:  GET /v3/accounts/{id}/changes?sinceTransactionID={id}
```

## Appendix B: Key Python Libraries

```
oandapyV20        — OANDA REST v20 API wrapper
pandas            — DataFrames for candle/indicator data
numpy             — Numerical operations
pandas-ta         — Technical indicators (RSI, MACD, BB, EMA, ATR)
aiohttp           — Async HTTP (if building custom stream handler)
aiosqlite         — Async SQLite for trade logging
python-dotenv     — Load .env files
pyyaml            — YAML configuration
structlog         — Structured logging
pytest            — Testing framework
pytest-asyncio    — Async test support
matplotlib        — Backtest performance charts
```

## Appendix C: Risk Cheat Sheet

| Parameter | Value | Notes |
|-----------|-------|-------|
| Risk per trade | 1% of equity | Never exceed 2% |
| Daily loss limit | 3% of equity | System halts trading for the day |
| Max open positions | 10 | 4% notional cap each = 40% max exposure |
| Max consecutive losses | 5 | 15-min cooldown after |
| Max trades per day | 150 | Scaled for 14 pairs (~6.3 trades/day expected) |
| Minimum spread threshold | 5.0 pips | Per-pair spreads range 1.0–4.0 pips |
| SL distance | ATR(14) × sl_multiplier | Per-pair: 1.5–3.5 (see pair_params.yaml) |
| TP distance | BB middle band (SMA) | Mean reversion target |
| Max trade duration | None | Removed — time-based exits cut profitable trades |
| Stream heartbeat timeout | 30 seconds | Close all only when ALL streams unhealthy |
| Position size cap | 4% notional | Per position, prevents concentration risk |
| Active instruments | 14 G10 pairs | Qualified via backtest PF >= 1.05 |

## Part 6: Implementation Results

### 6.1 Backtest Performance (Jan 2025 – Mar 2026)

**Per-pair optimized mean reversion strategy, realistic OANDA spreads, 0.5 pip slippage.**

| Pair | Trades | Tr/Day | WinRate | PF | Sharpe | Spread |
|------|--------|--------|---------|-----|--------|--------|
| GBP_NZD | 70 | 0.2 | 52.9% | 1.994 | 2.404 | 4.0 |
| EUR_CHF | 93 | 0.2 | 48.4% | 1.810 | 2.218 | 2.0 |
| EUR_GBP | 77 | 0.2 | 50.6% | 1.590 | 1.713 | 1.5 |
| GBP_AUD | 95 | 0.2 | 51.6% | 1.513 | 1.596 | 3.5 |
| EUR_NZD | 307 | 0.7 | 59.3% | 1.452 | 2.729 | 3.5 |
| NZD_CAD | 78 | 0.2 | 56.4% | 1.419 | 1.350 | 4.0 |
| EUR_JPY | 175 | 0.4 | 44.0% | 1.375 | 1.724 | 1.5 |
| EUR_USD | 166 | 0.4 | 31.9% | 1.370 | 1.502 | 1.0 |
| GBP_USD | 96 | 0.2 | 45.8% | 1.263 | 0.984 | 1.5 |
| USD_JPY | 125 | 0.3 | 40.0% | 1.202 | 0.833 | 1.0 |
| EUR_AUD | 532 | 1.2 | 55.6% | 1.148 | 1.298 | 2.0 |
| CHF_JPY | 542 | 1.2 | 53.7% | 1.103 | 0.988 | 3.0 |
| USD_CHF | 179 | 0.4 | 36.3% | 1.089 | 0.455 | 1.5 |
| NZD_USD | 223 | 0.5 | 52.9% | 1.064 | 0.402 | 2.0 |

**Aggregate:** 14 qualified pairs, ~6.3 trades/day, $1,791 total PnL on $100K equity.

### 6.2 Optimization Impact

| Metric | Uniform Params | Per-Pair Optimized | Change |
|--------|---------------|-------------------|--------|
| Qualified pairs (PF >= 1.05) | 7 | 14 | +100% |
| Aggregate PnL | $1,295 | $1,791 | +38% |
| Trades/day | 8.6 | 6.3 | -27% |

Per-pair optimization favours fewer but higher-quality entries (wider BB bands = bb_std 2.25–2.50, tighter RSI thresholds = 25/75).

### 6.3 Per-Pair Optimized Parameters

| Pair | bb_std | RSI OS/OB | SL Mult | Train PF | Test PF |
|------|--------|-----------|---------|----------|---------|
| EUR_USD | 2.25 | 25/75 | 1.5 | 1.307 | 1.859 |
| USD_JPY | 2.50 | 25/75 | 2.5 | 1.136 | 1.453 |
| GBP_USD | 2.50 | 25/75 | 3.0 | 1.267 | 1.207 |
| USD_CHF | 1.75 | 25/75 | 1.5 | 1.066 | 1.188 |
| NZD_USD | 2.25 | 30/70 | 3.0 | 1.016 | 1.274 |
| EUR_GBP | 2.50 | 25/75 | 2.5 | 1.640 | 1.611 |
| EUR_JPY | 2.50 | 30/70 | 2.0 | 1.492 | 1.012 |
| EUR_CHF | 2.50 | 25/75 | 2.0 | 1.723 | 2.285 |
| EUR_NZD | 2.50 | 35/65 | 2.5 | 1.650 | 1.073 |
| GBP_AUD | 2.50 | 25/75 | 3.0 | 1.704 | 1.227 |
| GBP_NZD | 2.50 | 25/75 | 2.5 | 1.949 | 2.150 |
| NZD_CAD | 2.50 | 25/75 | 3.0 | 1.259 | 2.475 |
| AUD_USD | 1.50 | 25/75 | 3.5 | 0.877 | 1.614 |

EUR_AUD and CHF_JPY use global defaults (bb_std=2.0, RSI 35/65, sl_mult=2.5) and still qualify.

### 6.4 Excluded Pairs (14 of 28)

Pairs excluded due to test PF < 1.0 (overfit) or PF < 1.05 (insufficient edge):

- **High-spread casualties:** GBP_CHF (3.5), NZD_CHF (4.5), CAD_CHF (4.5) — spread exceeds TP distance
- **Trending pairs:** AUD_JPY, AUD_CHF, AUD_NZD, AUD_CAD, CAD_JPY — commodity/risk-sentiment driven, don't mean-revert well at M5
- **Marginal:** USD_CAD, GBP_JPY, GBP_CAD, EUR_CAD, NZD_JPY — PF 0.85–1.00 range

### 6.5 Running the System

```bash
# Paper trade on OANDA practice account (14 pairs, per-pair optimized)
python -m src.main

# Re-optimize parameters (quarterly recommended)
python scripts/optimize_pairs.py -f 2025-01-01 --skip-download

# Batch backtest with optimized params
python scripts/backtest_all_pairs.py -f 2025-01-01 --skip-download --use-pair-params

# Single-pair backtest
python scripts/run_backtest.py -i EUR_USD -f 2025-01-01
```

---

*Disclaimer: This is an engineering design for an experimental trading system on a practice account. Forex trading involves significant risk. Past performance does not indicate future results. This document does not constitute financial advice. Always validate strategies thoroughly before deploying with real capital.*
