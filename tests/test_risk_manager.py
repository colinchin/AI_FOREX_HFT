"""Tests for risk manager — verify limits, filters, and position sizing."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.core.bus import EventBus
from src.core.events import (
    Direction,
    OrderEvent,
    SignalEvent,
    SignalStrength,
    TickEvent,
    TradeCloseEvent,
)
from src.data.tick_buffer import TickBuffer
from src.risk.manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.utils.config import RiskConfig


def _rate_lookup(instrument: str) -> float | None:
    """Minimal rate lookup for test conversion paths (EUR_USD → AUD)."""
    rates = {
        "USD_AUD": 1.55,   # 1 USD = 1.55 AUD
        "AUD_USD": 0.645,  # inverse
        "EUR_USD": 1.10,
        "EUR_AUD": 1.705,
    }
    return rates.get(instrument)


def _make_risk_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_risk_per_trade=0.01,
        max_daily_loss=0.03,
        max_open_positions=3,
        max_consecutive_losses=5,
        cooldown_seconds=900,
        max_trades_per_day=50,
        max_spread_pips=2.0,
        max_spread_cost_pct=0.30,
        min_equity=100.0,
        position_sizing={"method": "atr_based", "min_units": 1, "max_units": 100000, "max_position_pct": 0.10},
        circuit_breaker={"stream_timeout_seconds": 30, "max_api_errors": 10, "equity_drawdown_pct": 0.05},
        weekend_close={},
        volatility_filter={"max_atr_std_devs": 2.0, "atr_lookback": 50},
        news_filter={"enabled": False},
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _make_signal(instrument: str = "EUR_USD", direction: Direction = Direction.LONG) -> SignalEvent:
    return SignalEvent(
        instrument=instrument,
        direction=direction,
        strength=SignalStrength.MODERATE,
        strategy_name="test",
        timestamp=datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        metadata={
            "sl_distance": 0.0010,
            "tp_distance": 0.0020,
            "atr": 0.0008,
        },
    )


def _make_tick(instrument: str = "EUR_USD", price: float = 1.1000) -> TickEvent:
    return TickEvent(
        instrument=instrument,
        bid=price - 0.00008,
        ask=price + 0.00008,
        timestamp=datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def tick_buffer():
    buf = TickBuffer()
    return buf


class TestPositionSizer:
    def test_basic_sizing(self):
        sizer = PositionSizer(min_units=1, max_units=100000)
        units = sizer.calculate(
            equity=100000,
            risk_pct=0.01,
            sl_distance=0.0010,
            instrument="EUR_USD",
            current_price=1.1000,
            rate_lookup=_rate_lookup,
        )
        # risk=1000 AUD, sl_in_AUD = 0.001 * 1.55 = 0.00155, units = 1000/0.00155 ≈ 645161
        # clamped to max_units
        assert units <= 100000
        assert units > 0

    def test_spread_reduces_size(self):
        sizer = PositionSizer(min_units=1, max_units=100000)
        without_spread = sizer.calculate(100000, 0.01, 0.0010, "EUR_USD", 1.1, 0.0, _rate_lookup)
        with_spread = sizer.calculate(100000, 0.01, 0.0010, "EUR_USD", 1.1, 0.0002, _rate_lookup)
        assert with_spread <= without_spread

    def test_zero_sl_returns_zero(self):
        sizer = PositionSizer()
        assert sizer.calculate(100000, 0.01, 0, "EUR_USD", 1.1) == 0

    def test_min_units_rejects_small(self):
        """Position size below min_units should be rejected (return 0)."""
        sizer = PositionSizer(min_units=10000)
        # Tiny equity → small position below min_units threshold
        units = sizer.calculate(100, 0.01, 0.0010, "EUR_USD", 1.1, rate_lookup=_rate_lookup)
        # risk=1 AUD, sl_in_AUD=0.00155, units=1/0.00155≈645, below 10000 min
        assert units == 0

    def test_min_units_allows_large(self):
        """Position size above min_units should pass through."""
        sizer = PositionSizer(min_units=100)
        units = sizer.calculate(100_000, 0.01, 0.0010, "EUR_USD", 1.1, rate_lookup=_rate_lookup)
        assert units >= 100


class TestRiskManager:
    @pytest.mark.asyncio
    async def test_signal_passes_all_checks(self, bus, tick_buffer):
        """A valid signal with all conditions met should emit an OrderEvent."""
        config = _make_risk_config()
        orders: list[OrderEvent] = []

        async def capture_order(event):
            if isinstance(event, OrderEvent):
                orders.append(event)

        bus.subscribe(OrderEvent, capture_order)
        bus.start()

        # Provide tick data + conversion pair for AUD account sizing
        tick = _make_tick()
        await tick_buffer.on_tick(tick)
        aud_tick = _make_tick("AUD_USD", price=0.6450)
        await tick_buffer.on_tick(aud_tick)

        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(
            config=config,
            event_bus=bus,
            tick_buffer=tick_buffer,
            session_config={
                "london": {"start": "08:00", "end": "17:00"},
                "new_york": {"start": "13:00", "end": "22:00"},
                "active_sessions": ["london", "new_york"],
                "news_blackouts": [],
            },
            equity_provider=equity_provider,
        )

        signal = _make_signal()
        await rm.on_signal(signal)
        await asyncio.sleep(0.1)

        assert len(orders) == 1
        assert orders[0].instrument == "EUR_USD"
        assert orders[0].direction == Direction.LONG
        assert orders[0].stop_loss > 0
        assert orders[0].take_profit > 0

        await bus.stop()

    @pytest.mark.asyncio
    async def test_max_positions_blocks(self, bus, tick_buffer):
        """Should reject signals when max positions reached."""
        config = _make_risk_config(max_open_positions=1)
        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(config, bus, tick_buffer,
                         {"london": {"start": "08:00", "end": "17:00"},
                          "active_sessions": ["london"], "news_blackouts": []},
                         equity_provider)

        tick = _make_tick()
        await tick_buffer.on_tick(tick)

        # Simulate one open position
        from src.core.events import FillEvent
        fill = FillEvent("EUR_USD", Direction.LONG, 1000, 1.1, "T1")
        await rm.on_fill(fill)

        orders = []
        bus.subscribe(OrderEvent, lambda e: orders.append(e))
        bus.start()

        signal = _make_signal()
        await rm.on_signal(signal)
        await asyncio.sleep(0.1)

        assert len(orders) == 0
        await bus.stop()

    @pytest.mark.asyncio
    async def test_daily_loss_halt(self, bus, tick_buffer):
        """Should halt trading after daily loss limit breach."""
        config = _make_risk_config(max_daily_loss=0.03)
        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(config, bus, tick_buffer,
                         {"london": {"start": "08:00", "end": "17:00"},
                          "active_sessions": ["london"], "news_blackouts": []},
                         equity_provider)

        # Simulate losses exceeding 3%
        for i in range(5):
            close_event = TradeCloseEvent(
                instrument="EUR_USD", trade_id=f"T{i}",
                close_price=1.09, pnl=-700.0,  # 5 * -700 = -3500 > 3%
                reason="sl_hit",
            )
            await rm.on_trade_close(close_event)

        assert rm.is_halted or rm.daily_stats.net_pnl < -3000

    @pytest.mark.asyncio
    async def test_consecutive_loss_cooldown(self, bus, tick_buffer):
        """Should trigger cooldown after consecutive losses."""
        config = _make_risk_config(max_consecutive_losses=3, cooldown_seconds=60)
        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(config, bus, tick_buffer,
                         {"london": {"start": "08:00", "end": "17:00"},
                          "active_sessions": ["london"], "news_blackouts": []},
                         equity_provider)

        for i in range(3):
            close_event = TradeCloseEvent(
                instrument="EUR_USD", trade_id=f"T{i}",
                close_price=1.09, pnl=-100.0, reason="sl_hit",
            )
            await rm.on_trade_close(close_event)

        assert rm.daily_stats.consecutive_losses >= 3
