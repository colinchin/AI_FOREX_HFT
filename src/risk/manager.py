"""Central risk manager — the gatekeeper between signals and execution.

Applies all filters, enforces limits, computes position size, emits OrderEvents.
This is the most critical module for capital preservation.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.core.events import (
    Direction,
    ErrorEvent,
    FillEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    TradeCloseEvent,
    TradeReducedEvent,
)
from src.core.models import DailyStats
from src.data.tick_buffer import TickBuffer
from src.risk.filters import NewsFilter, SessionFilter, SpreadCostFilter, SpreadFilter, VolatilityFilter
from src.risk.position_sizer import PositionSizer
from src.utils.config import RiskConfig
from src.utils.helpers import price_precision, utc_now
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.core.bus import EventBus

log = get_logger(__name__)


class RiskManager:
    """Central risk management — enforces all trading limits.

    Receives SignalEvents, applies filters and checks, then emits
    OrderEvents if everything passes. Tracks daily P&L and consecutive
    losses for automatic halting.
    """

    def __init__(
        self,
        config: RiskConfig,
        event_bus: EventBus,
        tick_buffer: TickBuffer,
        session_config: dict,
        equity_provider: callable,  # async () -> float
        account_currency: str = "AUD",
        flatten_callback: callable | None = None,  # async (reason) -> None
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._tick_buffer = tick_buffer
        self._equity_provider = equity_provider
        self._account_currency = account_currency
        self._flatten_callback = flatten_callback  # closes all positions on critical halt

        # Position sizing
        ps_cfg = config.position_sizing
        self._sizer = PositionSizer(
            method=ps_cfg.get("method", "atr_based"),
            min_units=ps_cfg.get("min_units", 1),
            max_units=ps_cfg.get("max_units", 100_000),
            max_position_pct=ps_cfg.get("max_position_pct", 0.10),
            account_currency=account_currency,
        )

        # Filters
        active_sessions = session_config.get("active_sessions", ["london", "new_york"])
        self._spread_filter = SpreadFilter(config.max_spread_pips, tick_buffer)
        self._spread_cost_filter = SpreadCostFilter(config.max_spread_cost_pct, tick_buffer)
        self._session_filter = SessionFilter(session_config, active_sessions)
        self._volatility_filter = VolatilityFilter(
            max_std_devs=config.volatility_filter.get("max_atr_std_devs", 2.0),
            lookback=config.volatility_filter.get("atr_lookback", 50),
        )
        self._news_filter = NewsFilter(config.news_filter)

        # State tracking
        self._daily_stats = DailyStats(date=utc_now().strftime("%Y-%m-%d"))
        self._open_position_count = 0
        self._open_instruments: set[str] = set()
        self._closed_trade_ids: set[str] = set()  # dedup close events
        self._halted = False
        self._halt_reason = ""
        self._cooldown_until: datetime | None = None
        self._session_start_equity: float | None = None

        # Circuit breaker
        self._cb_config = config.circuit_breaker
        self._consecutive_api_errors = 0

    def _rate_lookup(self, instrument: str) -> float | None:
        """Look up mid price from tick buffer for currency conversion."""
        tick = self._tick_buffer.latest(instrument)
        return tick.mid if tick else None

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def daily_stats(self) -> DailyStats:
        return self._daily_stats

    @property
    def news_filter(self) -> NewsFilter:
        return self._news_filter

    async def on_signal(self, signal: SignalEvent) -> None:
        """Process a signal through all risk checks."""
        # ── Pre-checks ──
        if self._halted:
            log.info("signal_rejected_halted", reason=self._halt_reason, instrument=signal.instrument)
            return

        # Check cooldown
        if self._cooldown_until and utc_now() < self._cooldown_until:
            log.info("signal_rejected_cooldown", until=self._cooldown_until.isoformat())
            return

        # Reset cooldown if expired
        if self._cooldown_until and utc_now() >= self._cooldown_until:
            self._cooldown_until = None

        # ── Check daily date rollover ──
        today = utc_now().strftime("%Y-%m-%d")
        if today != self._daily_stats.date:
            log.info("daily_reset", prev_date=self._daily_stats.date, new_date=today)
            self._daily_stats = DailyStats(date=today)
            self._session_start_equity = None

        # ── Apply filters ──
        for name, filt in [
            ("spread", self._spread_filter),
            ("spread_cost", self._spread_cost_filter),
            ("session", self._session_filter),
            ("volatility", self._volatility_filter),
            ("news", self._news_filter),
        ]:
            allowed, reason = filt.check(signal)
            if not allowed:
                log.info("signal_filtered", filter=name, reason=reason, instrument=signal.instrument)
                return

        # ── Limit checks ──

        # Max open positions
        if self._open_position_count >= self._config.max_open_positions:
            log.info("signal_rejected_max_positions", count=self._open_position_count)
            return

        # Don't open opposing positions on same instrument
        if signal.instrument in self._open_instruments:
            log.info("signal_rejected_already_open", instrument=signal.instrument)
            return

        # Max trades per day (counted on entry, not close)
        if self._daily_stats.entries_today >= self._config.max_trades_per_day:
            log.info("signal_rejected_max_trades", entries=self._daily_stats.entries_today)
            return

        # Daily loss limit
        equity = await self._equity_provider()
        if self._session_start_equity is None:
            self._session_start_equity = equity

        daily_loss_pct = abs(self._daily_stats.net_pnl) / self._session_start_equity if self._session_start_equity > 0 else 0
        if self._daily_stats.net_pnl < 0 and daily_loss_pct >= self._config.max_daily_loss:
            await self._halt("daily_loss_limit_breached")
            return

        # Minimum equity
        if equity < self._config.min_equity:
            await self._halt(f"equity_below_minimum:{equity:.2f}")
            return

        # Circuit breaker: equity drawdown from session start
        cb_dd = self._cb_config.get("equity_drawdown_pct", 0.05)
        if self._session_start_equity and equity < self._session_start_equity * (1 - cb_dd):
            await self._halt(f"circuit_breaker_drawdown:{cb_dd*100:.0f}%")
            return

        # ── Position sizing ──
        sl_distance = signal.metadata.get("sl_distance", 0)
        if sl_distance <= 0:
            log.warning("signal_no_sl_distance", instrument=signal.instrument)
            return

        tick = self._tick_buffer.latest(signal.instrument)
        if tick is None:
            log.warning("signal_no_price_data", instrument=signal.instrument)
            return

        current_price = tick.mid
        spread = tick.spread

        units = self._sizer.calculate(
            equity=equity,
            risk_pct=self._config.max_risk_per_trade,
            sl_distance=sl_distance,
            instrument=signal.instrument,
            current_price=current_price,
            spread=spread,
            rate_lookup=self._rate_lookup,
        )

        if units <= 0:
            log.warning("signal_zero_units", instrument=signal.instrument)
            return

        # Record ATR for volatility filter
        atr_val = signal.metadata.get("atr")
        if atr_val:
            self._volatility_filter.record_atr(signal.instrument, atr_val)

        # ── Construct order ──
        tp_distance = signal.metadata.get("tp_distance", sl_distance * 1.5)

        # Anchor SL/TP on expected fill price, not mid.
        # LONG fills at ask (mid + half_spread), SHORT at bid (mid - half_spread).
        # Without this, TP is effectively closer by half_spread and SL further,
        # degrading R:R vs what the strategy intended.
        half_spread = spread / 2

        if signal.direction is Direction.LONG:
            entry_est = current_price + half_spread  # expected fill at ask
            sl_price = entry_est - sl_distance
            tp_price = entry_est + tp_distance
            price_bound = current_price + spread  # Max slippage = 1 spread
        else:
            entry_est = current_price - half_spread  # expected fill at bid
            sl_price = entry_est + sl_distance
            tp_price = entry_est - tp_distance
            price_bound = current_price - spread

        trailing_dist = signal.metadata.get("trailing_distance")
        trailing_activate = signal.metadata.get("trailing_activate")

        prec = price_precision(signal.instrument)
        order = OrderEvent(
            instrument=signal.instrument,
            direction=signal.direction,
            units=units,
            order_type=OrderType.MARKET,
            stop_loss=round(sl_price, prec),
            take_profit=round(tp_price, prec),
            price_bound=round(price_bound, prec),
            trailing_stop_distance=trailing_dist,
            trailing_activate_distance=trailing_activate,
            strategy_name=signal.strategy_name,
        )

        log.info(
            "order_emitted",
            instrument=order.instrument,
            direction=order.direction.name,
            units=order.units,
            sl=order.stop_loss,
            tp=order.take_profit,
            strategy=order.strategy_name,
        )

        await self._bus.publish(order)

    async def on_fill(self, fill: FillEvent) -> None:
        """Track open positions on fills and count entries for daily cap."""
        self._open_position_count += 1
        self._open_instruments.add(fill.instrument)
        self._daily_stats.record_entry()
        log.info(
            "position_opened",
            instrument=fill.instrument,
            direction=fill.direction.name,
            units=fill.units,
            price=fill.fill_price,
            open_count=self._open_position_count,
            entries_today=self._daily_stats.entries_today,
        )

    async def on_trade_close(self, event: TradeCloseEvent) -> None:
        """Update stats when a trade closes. Idempotent — ignores duplicate close events."""
        if event.trade_id in self._closed_trade_ids:
            log.debug("risk_close_duplicate_ignored", trade_id=event.trade_id)
            return
        self._closed_trade_ids.add(event.trade_id)

        self._open_position_count = max(0, self._open_position_count - 1)
        self._open_instruments.discard(event.instrument)

        self._daily_stats.record_trade(event.pnl)

        log.info(
            "trade_closed_risk_update",
            instrument=event.instrument,
            pnl=event.pnl,
            reason=event.reason,
            daily_pnl=self._daily_stats.net_pnl,
            consecutive_losses=self._daily_stats.consecutive_losses,
            win_rate=f"{self._daily_stats.win_rate:.1%}",
        )

        # Check consecutive losses → cooldown (use current streak, not max)
        if self._daily_stats.current_loss_streak >= self._config.max_consecutive_losses:
            from datetime import timedelta
            self._cooldown_until = utc_now() + timedelta(seconds=self._config.cooldown_seconds)
            log.warning(
                "consecutive_loss_cooldown",
                current_streak=self._daily_stats.current_loss_streak,
                cooldown_until=self._cooldown_until.isoformat(),
            )

    async def on_trade_reduced(self, event: TradeReducedEvent) -> None:
        """Update risk state for a partial trade close.

        Books the partial PnL via record_trade() so win/loss counts,
        profit factor, and consecutive-loss cooldown stay consistent.
        Does NOT decrement position count or remove the instrument —
        the remaining position is still open.
        """
        self._daily_stats.record_trade(event.pnl)

        log.info(
            "trade_reduced_risk_update",
            trade_id=event.trade_id,
            instrument=event.instrument,
            remaining_units=event.remaining_units,
            partial_pnl=event.pnl,
            daily_pnl=self._daily_stats.net_pnl,
            consecutive_losses=self._daily_stats.consecutive_losses,
        )

        # Check consecutive losses → cooldown
        if self._daily_stats.current_loss_streak >= self._config.max_consecutive_losses:
            from datetime import timedelta
            self._cooldown_until = utc_now() + timedelta(seconds=self._config.cooldown_seconds)
            log.warning(
                "consecutive_loss_cooldown",
                current_streak=self._daily_stats.current_loss_streak,
                cooldown_until=self._cooldown_until.isoformat(),
            )

    async def _halt(self, reason: str, flatten: bool = True) -> None:
        """Halt all trading and optionally close all open positions.

        Args:
            reason: Why we're halting.
            flatten: If True and a flatten_callback is set, close all open
                     positions immediately. Defaults to True so that loss-limit
                     and drawdown halts actually de-risk the book.
        """
        self._halted = True
        self._halt_reason = reason
        log.error("trading_halted", reason=reason, flatten=flatten)

        if flatten and self._flatten_callback and self._open_position_count > 0:
            try:
                await self._flatten_callback(reason)
            except Exception as e:
                log.error("flatten_on_halt_failed", reason=reason, error=str(e))

    def resume(self) -> None:
        """Resume trading after halt (manual intervention)."""
        self._halted = False
        self._halt_reason = ""
        log.info("trading_resumed")

    async def check_loss_limits(self) -> None:
        """Check daily loss and drawdown limits — call periodically from main loop.

        Halts trading if limits are breached, even when no new signals arrive.
        """
        if self._halted:
            return

        equity = await self._equity_provider()
        if self._session_start_equity is None:
            self._session_start_equity = equity

        # Daily loss limit
        if self._session_start_equity > 0 and self._daily_stats.net_pnl < 0:
            daily_loss_pct = abs(self._daily_stats.net_pnl) / self._session_start_equity
            if daily_loss_pct >= self._config.max_daily_loss:
                await self._halt("daily_loss_limit_breached")
                return

        # Minimum equity
        if equity < self._config.min_equity:
            await self._halt(f"equity_below_minimum:{equity:.2f}")
            return

        # Circuit breaker: equity drawdown from session start
        cb_dd = self._cb_config.get("equity_drawdown_pct", 0.05)
        if self._session_start_equity and equity < self._session_start_equity * (1 - cb_dd):
            await self._halt(f"circuit_breaker_drawdown:{cb_dd*100:.0f}%")
            return

    def reconcile_open_positions(self, instruments: set[str], count: int) -> None:
        """Seed open position state from broker on startup.

        Prevents duplicate positions and bypassed limits after a restart.
        """
        self._open_position_count = count
        self._open_instruments = set(instruments)
        log.info(
            "risk_positions_reconciled",
            count=count,
            instruments=sorted(instruments),
        )

    async def record_api_error(self) -> None:
        """Track consecutive API errors for circuit breaker."""
        self._consecutive_api_errors += 1
        max_errors = self._cb_config.get("max_api_errors", 10)
        if self._consecutive_api_errors >= max_errors:
            await self._halt(f"api_errors:{self._consecutive_api_errors}")

    def clear_api_errors(self) -> None:
        self._consecutive_api_errors = 0
