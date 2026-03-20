"""Main entry point — wires all components and runs the trading loop.

Lifecycle:
  1. Load configuration
  2. Initialize all components
  3. Connect to OANDA, fetch account state
  4. Pre-load historical candles for indicator warmup
  5. Start price stream
  6. Run event loop: Stream → Candles → Strategy → Risk → Execution
  7. Graceful shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from src.api.account import AccountManager
from src.api.client import OANDAClient
from src.api.execution import ExecutionHandler
from src.api.stream import StreamManager
from src.api.transaction_poller import TransactionPoller
from src.core.bus import EventBus
from src.core.events import (
    CandleEvent,
    ErrorEvent,
    FillEvent,
    OrderEvent,
    SignalEvent,
    TickEvent,
    TradeCloseEvent,
    TradeReducedEvent,
)
from src.data.candle_builder import CandleBuilder
from src.data.history import HistoryFetcher
from src.data.store import TradeStore
from src.data.tick_buffer import TickBuffer
from src.portfolio.tracker import PortfolioTracker
from src.risk.manager import RiskManager
from src.strategy.momentum_scalp import MomentumScalpStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


class TradingSystem:
    """Orchestrates all trading components."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Core
        self._bus = EventBus()
        self._client = OANDAClient(config.oanda)
        self._account = AccountManager(self._client)
        self._tick_buffer = TickBuffer(config.data.get("tick_buffer_size", 10_000))
        self._candle_builder = CandleBuilder(
            self._bus,
            timeframes=["M1", "M5", "H1"],
            cache_size=500,
        )
        self._trade_store = TradeStore()
        self._execution = ExecutionHandler(self._client, self._bus)
        self._portfolio = PortfolioTracker(self._bus, self._trade_store, self._execution)

        # Transaction poller — detects broker-side SL/TP closures
        self._tx_poller = TransactionPoller(
            client=self._client,
            event_bus=self._bus,
            poll_interval=config.monitoring.get("transaction_poll_seconds", 5.0),
        )

        # Streams — split instruments across connections (OANDA limits per stream)
        chunk_size = config.monitoring.get("stream_chunk_size", 20)
        self._streams: list[StreamManager] = []
        for i in range(0, len(config.instruments), chunk_size):
            chunk = config.instruments[i : i + chunk_size]
            stream = StreamManager(
                stream_url=config.oanda.stream_url,
                account_id=config.oanda.account_id,
                access_token=config.oanda.access_token,
                instruments=chunk,
                event_bus=self._bus,
                heartbeat_timeout=config.risk.circuit_breaker.get("stream_timeout_seconds", 30),
            )
            self._streams.append(stream)

        # Multi-TF filter
        self._mtf = MultiTimeframeFilter()

        # Strategies — instantiate from config names
        _STRATEGY_MAP = {
            "momentum_scalp": MomentumScalpStrategy,
            "mean_reversion": MeanReversionStrategy,
        }
        self._strategies: list = []
        strategy_cfg = config.strategy

        # Load per-pair optimized params if configured
        mr_cfg = strategy_cfg.get("mean_reversion", {})
        pp_file = mr_cfg.get("pair_params_file")
        if pp_file:
            pp_path = Path(pp_file)
            if pp_path.exists():
                import yaml
                with open(pp_path) as f:
                    pair_params = yaml.safe_load(f) or {}
                mr_cfg["pair_params"] = pair_params
                strategy_cfg["mean_reversion"] = mr_cfg
                log.info("pair_params_loaded", file=pp_file, pairs=len(pair_params))

        for role in ("primary", "secondary"):
            name = strategy_cfg.get(role)
            if name and name in _STRATEGY_MAP:
                self._strategies.append(
                    _STRATEGY_MAP[name](strategy_cfg, self._bus)
                )

        # Risk manager (account_currency set after account init in start())
        self._risk = RiskManager(
            config=config.risk,
            event_bus=self._bus,
            tick_buffer=self._tick_buffer,
            session_config=config.sessions,
            equity_provider=self._account.get_equity,
            account_currency="AUD",  # updated from broker in start()
            flatten_callback=self._emergency_flatten,
        )

        # History fetcher
        self._history = HistoryFetcher(
            self._client,
            cache_dir=config.data.get("parquet_dir", "data/parquet"),
        )

        # Circuit breaker state — avoid redundant close-all calls
        self._last_close_all: float = 0.0
        _CLOSE_ALL_COOLDOWN = 120.0  # seconds between close-all calls
        self._close_all_cooldown = _CLOSE_ALL_COOLDOWN

    async def start(self) -> None:
        """Initialize and start the trading system."""
        log.info("system_starting")

        # Initialize database
        await self._trade_store.initialize()

        # Connect and verify
        health = await self._client.health_check()
        log.info("oanda_connected", **health)

        # Initialize account
        await self._account.initialize()

        # Set actual account currency from broker (fixes cross-currency sizing)
        acct_ccy = self._account.currency
        self._risk._account_currency = acct_ccy
        self._risk._sizer._account_currency = acct_ccy
        log.info("account_currency_set", currency=acct_ccy)

        # Reconcile broker open positions into in-memory state (survives restarts)
        await self._reconcile_broker_positions()

        # Wire up event bus subscriptions
        self._subscribe_events()

        # Start event bus
        self._bus.start()

        # Fetch news calendar (non-blocking — logs warning on failure)
        await self._risk.news_filter.refresh()

        # Warmup indicators with historical data
        await self._warmup_indicators()

        # Start price streams
        for stream in self._streams:
            await stream.start()

        # Start transaction poller (detects broker-side SL/TP closures)
        await self._tx_poller.start()

        self._running = True

        # Run main loop
        try:
            await self._main_loop()
        finally:
            await self.stop()

    def _subscribe_events(self) -> None:
        """Wire all event handlers to the bus."""
        # Tick handlers
        self._bus.subscribe(TickEvent, self._tick_buffer.on_tick)
        self._bus.subscribe(TickEvent, self._candle_builder.on_tick)
        self._bus.subscribe(TickEvent, self._portfolio.on_tick)

        # Candle handlers
        self._bus.subscribe(CandleEvent, self._on_candle)
        self._bus.subscribe(CandleEvent, self._mtf.on_candle)

        # Signal → Risk Manager
        self._bus.subscribe(SignalEvent, self._risk.on_signal)

        # Order → Execution
        self._bus.subscribe(OrderEvent, self._execution.on_order)

        # Fill → Portfolio + Risk + Transaction poller (track trade IDs)
        self._bus.subscribe(FillEvent, self._portfolio.on_fill)
        self._bus.subscribe(FillEvent, self._risk.on_fill)
        self._bus.subscribe(FillEvent, self._on_fill_track)

        # Trade close → Portfolio + Risk
        self._bus.subscribe(TradeCloseEvent, self._portfolio.on_trade_close)
        self._bus.subscribe(TradeCloseEvent, self._risk.on_trade_close)

        # Partial trade close → Portfolio + Risk
        self._bus.subscribe(TradeReducedEvent, self._portfolio.on_trade_reduced)
        self._bus.subscribe(TradeReducedEvent, self._risk.on_trade_reduced)

        # Errors
        self._bus.subscribe(ErrorEvent, self._on_error)

    async def _emergency_flatten(self, reason: str, respect_cooldown: bool = False) -> None:
        """Close all open positions with retry escalation.

        Shared by all kill-switch paths: loss/drawdown halt, critical error,
        stream timeout, and weekend close. If initial flatten fails, spawns
        a background retry loop.
        """
        import time

        if respect_cooldown:
            now = time.monotonic()
            if now - self._last_close_all <= self._close_all_cooldown:
                log.debug("flatten_cooldown_active", reason=reason)
                return
            self._last_close_all = now
        else:
            self._last_close_all = time.monotonic()

        success = await self._execution.close_all_positions(reason=reason)

        if not success:
            log.error("flatten_incomplete_scheduling_retry", reason=reason)
            asyncio.create_task(self._flatten_retry_loop(reason))

    async def _flatten_retry_loop(self, reason: str) -> None:
        """Background task: keep trying to flatten positions until confirmed clear."""
        max_attempts = 10
        retry_interval = 30.0

        for attempt in range(1, max_attempts + 1):
            await asyncio.sleep(retry_interval)

            if not self._running:
                return

            log.warning("flatten_retry_attempt", attempt=attempt, reason=reason)
            success = await self._execution.close_all_positions(reason=f"{reason}_retry")

            if success:
                log.info("flatten_retry_succeeded", attempt=attempt)
                return

        log.error("flatten_retry_exhausted", max_attempts=max_attempts, reason=reason)

    async def _reconcile_broker_positions(self) -> None:
        """Fetch open trades from broker and seed RiskManager + PortfolioTracker.

        Prevents duplicate positions and bypassed limits after a restart.
        """
        try:
            open_trades = await self._client.get_open_trades()
            if not open_trades:
                log.info("reconcile_no_open_positions")
                return

            # Seed PortfolioTracker with full trade objects
            self._portfolio.reconcile_open_trades(open_trades)

            # Seed RiskManager with count and instrument set
            instruments = {t.get("instrument", "") for t in open_trades if t.get("instrument")}
            self._risk.reconcile_open_positions(instruments, len(open_trades))

            # Seed TransactionPoller (already done in its start(), but ensure consistency)
            for trade in open_trades:
                tid = trade.get("id", "")
                if tid:
                    self._tx_poller.register_trade(tid)

            log.info(
                "broker_positions_reconciled",
                count=len(open_trades),
                instruments=sorted(instruments),
            )
        except Exception as e:
            log.error("reconcile_failed", error=str(e))

    async def _on_fill_track(self, fill: FillEvent) -> None:
        """Register trade IDs with the transaction poller."""
        self._tx_poller.register_trade(fill.trade_id)

    async def _on_candle(self, candle: CandleEvent) -> None:
        """Route candle events to strategies."""
        for strategy in self._strategies:
            if strategy.enabled:
                signal = await strategy.on_candle(candle)
                if signal is not None:
                    # Check multi-TF alignment
                    if self._mtf.is_aligned(signal.instrument, signal.direction):
                        await self._bus.publish(signal)
                    else:
                        log.debug(
                            "signal_mtf_rejected",
                            instrument=signal.instrument,
                            direction=signal.direction.name,
                            bias=self._mtf.get_bias(signal.instrument).name,
                        )

    async def _on_error(self, error: ErrorEvent) -> None:
        """Handle error events."""
        if error.critical:
            log.error("critical_error", source=error.source, message=error.message)
            # Close all positions and halt (with retry escalation)
            await self._emergency_flatten("critical_error", respect_cooldown=False)
            await self._risk._halt(f"critical_error:{error.source}", flatten=False)  # already flattened above
        else:
            log.warning("error_event", source=error.source, message=error.message)
            await self._risk.record_api_error()

    async def _warmup_indicators(self) -> None:
        """Pre-load historical candles to warm up indicators."""
        log.info("warming_up_indicators")

        for instrument in self._config.instruments:
            # Fetch M5 history
            try:
                from_time = datetime.now(timezone.utc) - timedelta(days=7)
                m5_df = await self._history.fetch_candles(
                    instrument, "M5", from_time, use_cache=True,
                )
                if not m5_df.empty:
                    # Drop incomplete candles to avoid double-counting when
                    # the live stream finishes the same candle.
                    if "complete" in m5_df.columns:
                        m5_df = m5_df[m5_df["complete"] == True]  # noqa: E712
                    candles = self._history.df_to_candle_models(m5_df, instrument, "M5")
                    for strategy in self._strategies:
                        strategy.warmup(candles)
                    # Inject into candle builder cache
                    self._candle_builder.inject_candles(instrument, "M5", candles)
                    log.info("m5_warmup_complete", instrument=instrument, candles=len(candles))
            except Exception as e:
                log.warning("m5_warmup_failed", instrument=instrument, error=str(e))

            # Fetch H1 history for MTF
            try:
                from_time = datetime.now(timezone.utc) - timedelta(days=30)
                h1_df = await self._history.fetch_candles(
                    instrument, "H1", from_time, use_cache=True,
                )
                if not h1_df.empty:
                    # Drop incomplete candles (same reason as M5 above)
                    if "complete" in h1_df.columns:
                        h1_df = h1_df[h1_df["complete"] == True]  # noqa: E712
                    closes = h1_df["close"].tolist()
                    self._mtf.warmup(instrument, closes)
                    candles = self._history.df_to_candle_models(h1_df, instrument, "H1")
                    for strategy in self._strategies:
                        strategy.warmup(candles)
                    self._candle_builder.inject_candles(instrument, "H1", candles)
                    log.info("h1_warmup_complete", instrument=instrument, candles=len(candles))
            except Exception as e:
                log.warning("h1_warmup_failed", instrument=instrument, error=str(e))

    async def _main_loop(self) -> None:
        """Main monitoring loop — health checks, status reporting, circuit breaker."""
        status_interval = self._config.monitoring.get("status_interval_seconds", 60)
        health_interval = self._config.monitoring.get("health_check_interval_seconds", 30)

        last_status = 0.0
        last_health = 0.0
        last_trade_age = 0.0
        trade_age_interval = 30.0
        last_weekend_check = 0.0
        weekend_check_interval = 60.0  # Check once per minute (no rush)
        last_news_refresh = 0.0
        news_refresh_interval = 300.0  # Try refresh every 5 min (actual fetch gated by cache TTL)

        log.info("main_loop_started", instruments=self._config.instruments)

        while self._running and not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                pass

            import time
            now = time.monotonic()

            # Periodic health check
            if now - last_health >= health_interval:
                last_health = now

                # Stream health — only circuit-break when ALL streams are unhealthy
                unhealthy = [s for s in self._streams if not s.is_healthy]
                if unhealthy and self._running:
                    for s in unhealthy:
                        log.error("stream_unhealthy", stats=s.stats)
                    if len(unhealthy) == len(self._streams):
                        await self._emergency_flatten("all_streams_timeout", respect_cooldown=True)

                # Account refresh — only clear API errors on success
                if await self._account.refresh():
                    self._risk.clear_api_errors()
                else:
                    await self._risk.record_api_error()

                # Check loss/drawdown limits even when no signals arrive
                await self._risk.check_loss_limits()

            # Periodic trade age check — close profitable trades past max duration
            if now - last_trade_age >= trade_age_interval:
                last_trade_age = now
                await self._check_trade_age()

            # Periodic news calendar refresh (actual fetch gated by cache TTL)
            if now - last_news_refresh >= news_refresh_interval:
                last_news_refresh = now
                await self._risk.news_filter.refresh()

            # Periodic weekend close check
            if now - last_weekend_check >= weekend_check_interval:
                last_weekend_check = now
                await self._check_weekend_close()

            # Periodic status report
            if now - last_status >= status_interval:
                last_status = now
                self._print_status()

    async def _check_trade_age(self) -> None:
        """Close profitable trades that exceed max duration.

        Reads max_trade_duration_minutes from the strategy config that
        originated each position, so momentum_scalp and mean_reversion
        can have independent time-exit settings.
        """
        for trade_id, pos in list(self._portfolio.open_positions.items()):
            strat_cfg = self._config.strategy.get(pos.strategy_name, {})
            max_minutes = strat_cfg.get("max_trade_duration_minutes")
            if max_minutes is None:
                continue

            max_seconds = max_minutes * 60
            if pos.duration_seconds > max_seconds and pos.unrealised_pnl > 0:
                log.info(
                    "time_exit_closing",
                    trade_id=trade_id,
                    instrument=pos.instrument,
                    strategy=pos.strategy_name,
                    duration_min=round(pos.duration_seconds / 60, 1),
                    unrealised_pnl=round(pos.unrealised_pnl, 2),
                )
                try:
                    await self._execution.close_trade(trade_id, reason="time_exit")
                except Exception as e:
                    log.warning("time_exit_failed", trade_id=trade_id, error=str(e))

    async def _check_weekend_close(self) -> None:
        """Close all positions before forex market weekend close.

        Forex closes Friday 5:00 PM New York time. This method closes all
        open positions N minutes before that cutoff to avoid carrying
        positions over the weekend.
        """
        wk_cfg = self._config.risk.weekend_close
        if not wk_cfg.get("enabled", False):
            return

        ny_tz = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny_tz)

        # Auto-resume on Sunday evening when market reopens (5pm NY Sunday)
        # Must be checked BEFORE open_count gate — positions are 0 after Friday close
        if self._risk._halt_reason == "weekend_close":
            if now_ny.weekday() == 6 and now_ny.hour >= 17:
                self._risk.resume()
                log.info("weekend_halt_cleared", day=now_ny.strftime("%A"), time=now_ny.strftime("%H:%M"))
            elif now_ny.weekday() < 5:  # Monday-Thursday
                self._risk.resume()
                log.info("weekend_halt_cleared", day=now_ny.strftime("%A"), time=now_ny.strftime("%H:%M"))
            return

        # Only relevant on Fridays
        if now_ny.weekday() != 4:  # 0=Mon, 4=Fri
            return

        # No open positions → nothing to close
        if self._portfolio.open_count == 0:
            return

        minutes_before = wk_cfg.get("minutes_before_close", 60)

        # Market close = Friday 17:00 New York
        market_close = now_ny.replace(hour=17, minute=0, second=0, microsecond=0)
        cutoff = market_close - timedelta(minutes=minutes_before)

        if now_ny >= cutoff:
            log.warning(
                "weekend_close_triggered",
                now_ny=now_ny.isoformat(),
                cutoff=cutoff.isoformat(),
                open_positions=self._portfolio.open_count,
                minutes_before=minutes_before,
            )
            await self._emergency_flatten("weekend_close", respect_cooldown=False)
            await self._risk._halt("weekend_close", flatten=False)  # already flattened above

    def _print_status(self) -> None:
        """Log periodic status report."""
        account = self._account.summary
        portfolio = self._portfolio.stats
        stream = {
            "ticks_received": sum(s.stats["ticks_received"] for s in self._streams),
            "reconnect_count": sum(s.stats["reconnect_count"] for s in self._streams),
            "healthy": all(s.is_healthy for s in self._streams),
            "stream_count": len(self._streams),
            "healthy_count": sum(1 for s in self._streams if s.is_healthy),
        }
        bus = self._bus.stats
        risk = self._risk.daily_stats

        log.info(
            "status_report",
            equity=account.get("equity"),
            balance=account.get("balance"),
            unrealised_pnl=account.get("unrealised_pnl"),
            open_positions=portfolio.get("open_positions"),
            total_trades_today=risk.trades_taken,
            daily_pnl=risk.net_pnl,
            win_rate=f"{risk.win_rate:.1%}",
            stream_ticks=stream.get("ticks_received"),
            stream_healthy=stream.get("healthy"),
            bus_processed=bus.get("processed"),
            halted=self._risk.is_halted,
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        log.info("system_stopping")
        self._running = False
        self._shutdown_event.set()

        # Stop transaction poller
        await self._tx_poller.stop()

        # Stop streams
        for stream in self._streams:
            await stream.stop()

        # Close open positions (optional — configurable)
        # await self._execution.close_all_positions(reason="shutdown")

        # Stop event bus
        await self._bus.stop()

        # Close database
        await self._trade_store.close()

        log.info("system_stopped", portfolio=self._portfolio.stats)

    def request_shutdown(self) -> None:
        """Signal the system to shut down (called from signal handlers)."""
        log.info("shutdown_requested")
        self._shutdown_event.set()


async def main() -> None:
    """Application entry point."""
    config = load_config()

    setup_logging(
        level=config.logging_cfg.get("level", "INFO"),
        log_format=config.logging_cfg.get("format", "json"),
        log_file=config.logging_cfg.get("file"),
    )

    system = TradingSystem(config)

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, system.request_shutdown)
    else:
        # Windows: handle KeyboardInterrupt in the except block
        pass

    try:
        await system.start()
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        if system._running:
            await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
