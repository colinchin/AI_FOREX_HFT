"""Tests for candle builder — verify tick aggregation into M1/M5/H1 candles."""

import asyncio
from datetime import datetime, timezone

import pytest

from src.core.bus import EventBus
from src.core.events import CandleEvent, TickEvent
from src.data.candle_builder import CandleBuilder


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def builder(bus):
    return CandleBuilder(bus, timeframes=["M1", "M5", "H1"], cache_size=100)


def _tick(instrument: str, price: float, minute: int, second: int = 0) -> TickEvent:
    """Create a tick at a specific minute:second."""
    ts = datetime(2025, 1, 15, 10, minute, second, tzinfo=timezone.utc)
    spread = 0.00016
    return TickEvent(
        instrument=instrument,
        bid=price - spread / 2,
        ask=price + spread / 2,
        timestamp=ts,
    )


class TestCandleBuilder:
    @pytest.mark.asyncio
    async def test_m1_candle_creation(self, builder, bus):
        """Ticks in minute 0 should form an M1 candle when minute 1 starts."""
        collected: list[CandleEvent] = []

        async def collector(event: CandleEvent):
            collected.append(event)

        bus.subscribe(CandleEvent, collector)
        bus.start()

        # Feed ticks in minute 0
        await builder.on_tick(_tick("EUR_USD", 1.1000, 0, 0))
        await builder.on_tick(_tick("EUR_USD", 1.1005, 0, 15))
        await builder.on_tick(_tick("EUR_USD", 1.0998, 0, 30))
        await builder.on_tick(_tick("EUR_USD", 1.1002, 0, 45))

        # First tick of minute 1 closes the minute 0 candle
        await builder.on_tick(_tick("EUR_USD", 1.1003, 1, 0))

        # Wait for event processing
        await asyncio.sleep(0.1)

        m1_candles = [c for c in collected if c.timeframe == "M1"]
        assert len(m1_candles) >= 1

        candle = m1_candles[0]
        assert candle.instrument == "EUR_USD"
        assert candle.complete is True
        assert candle.volume == 4
        # High should be the highest mid price
        assert candle.high >= candle.open
        assert candle.high >= candle.close
        assert candle.low <= candle.open
        assert candle.low <= candle.close

        await bus.stop()

    @pytest.mark.asyncio
    async def test_m5_cascade(self, builder, bus):
        """M1 candles should cascade into M5 candles."""
        collected: list[CandleEvent] = []

        async def collector(event: CandleEvent):
            collected.append(event)

        bus.subscribe(CandleEvent, collector)
        bus.start()

        # Feed ticks across minutes 0-9 to span two M5 periods (0-4, 5-9)
        # The M5 candle for period 0-4 closes when minute 5 starts
        for minute in range(11):
            await builder.on_tick(_tick("EUR_USD", 1.1000 + minute * 0.0001, minute, 0))
            await builder.on_tick(_tick("EUR_USD", 1.1001 + minute * 0.0001, minute, 30))

        await asyncio.sleep(0.3)

        m1_candles = [c for c in collected if c.timeframe == "M1"]
        m5_candles = [c for c in collected if c.timeframe == "M5"]

        assert len(m1_candles) >= 10
        # M5 candle should be created after first 5-minute block completes
        assert len(m5_candles) >= 1

        await bus.stop()

    @pytest.mark.asyncio
    async def test_candle_cache(self, builder, bus):
        """Completed candles should be stored in the cache."""
        bus.start()

        # Generate 3 M1 candles
        for minute in range(4):
            await builder.on_tick(_tick("EUR_USD", 1.1000 + minute * 0.0001, minute, 0))

        await asyncio.sleep(0.1)

        candles = builder.get_candles("EUR_USD", "M1")
        assert len(candles) >= 2  # At least 2 completed candles

        await bus.stop()

    @pytest.mark.asyncio
    async def test_current_candle(self, builder, bus):
        """Should be able to get the current (incomplete) candle."""
        bus.start()

        await builder.on_tick(_tick("EUR_USD", 1.1000, 0, 0))
        await builder.on_tick(_tick("EUR_USD", 1.1005, 0, 30))

        current = builder.get_current_candle("EUR_USD", "M1")
        assert current is not None
        assert current.complete is False
        assert current.volume == 2

        await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_instruments(self, builder, bus):
        """Should handle multiple instruments independently."""
        bus.start()

        await builder.on_tick(_tick("EUR_USD", 1.1000, 0, 0))
        await builder.on_tick(_tick("USD_JPY", 150.50, 0, 0))
        await builder.on_tick(_tick("EUR_USD", 1.1005, 1, 0))
        await builder.on_tick(_tick("USD_JPY", 150.55, 1, 0))

        await asyncio.sleep(0.1)

        eur = builder.get_candles("EUR_USD", "M1")
        jpy = builder.get_candles("USD_JPY", "M1")

        # Each should have at least 1 completed candle
        assert len(eur) >= 1
        assert len(jpy) >= 1

        await bus.stop()
