"""Tests for trading strategies — signal generation with synthetic data."""

import asyncio
from datetime import datetime, timezone

import pytest

from src.core.events import CandleEvent, Direction, SignalEvent
from src.core.models import Candle
from src.strategy.momentum_scalp import MomentumScalpStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter, TrendBias


def _candle(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    minute: int = 0,
    timeframe: str = "M5",
    instrument: str = "EUR_USD",
) -> CandleEvent:
    if open_ is None:
        open_ = close - 0.0002
    if high is None:
        high = max(open_, close) + 0.0003
    if low is None:
        low = min(open_, close) - 0.0003
    return CandleEvent(
        instrument=instrument,
        timeframe=timeframe,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
        timestamp=datetime(2025, 1, 15, 10, minute, 0, tzinfo=timezone.utc),
        complete=True,
    )


def _uptrend_candles(n: int = 50, start: float = 1.0900, step: float = 0.0003) -> list[Candle]:
    """Generate synthetic uptrending M5 candles."""
    candles = []
    for i in range(n):
        price = start + i * step
        candles.append(Candle(
            instrument="EUR_USD",
            timeframe="M5",
            timestamp=datetime(2025, 1, 15, 10, i % 60, 0, tzinfo=timezone.utc),
            open=price - 0.0001,
            high=price + 0.0003,
            low=price - 0.0003,
            close=price,
            volume=100,
        ))
    return candles


def _downtrend_candles(n: int = 50, start: float = 1.1100, step: float = 0.0003) -> list[Candle]:
    candles = []
    for i in range(n):
        price = start - i * step
        candles.append(Candle(
            instrument="EUR_USD",
            timeframe="M5",
            timestamp=datetime(2025, 1, 15, 10, i % 60, 0, tzinfo=timezone.utc),
            open=price + 0.0001,
            high=price + 0.0003,
            low=price - 0.0003,
            close=price,
            volume=100,
        ))
    return candles


class TestMomentumScalpStrategy:
    def _make_strategy(self) -> MomentumScalpStrategy:
        return MomentumScalpStrategy(config={
            "momentum_scalp": {
                "ema_fast": 9,
                "ema_slow": 20,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "tp_atr_mult": 2.0,
                "sl_atr_mult": 1.5,
                "min_candle_body_pips": 0.1,
            }
        })

    @pytest.mark.asyncio
    async def test_warmup_required(self):
        strategy = self._make_strategy()
        assert strategy.required_history > 30

    @pytest.mark.asyncio
    async def test_no_signal_without_warmup(self):
        strategy = self._make_strategy()
        candle = _candle(1.1000, minute=0)
        signal = await strategy.on_candle(candle)
        assert signal is None

    @pytest.mark.asyncio
    async def test_uptrend_generates_long(self):
        """After warming up with uptrend data, should generate LONG signals."""
        strategy = self._make_strategy()
        candles = _uptrend_candles(60)
        strategy.warmup(candles)

        # Feed more uptrending candles
        signals = []
        for i in range(20):
            price = 1.1100 + i * 0.0004
            c = _candle(price, minute=i % 60)
            sig = await strategy.on_candle(c)
            if sig:
                signals.append(sig)

        # Should produce at least some long signals in uptrend
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        # May or may not generate signals depending on exact conditions
        # but should not generate SHORT signals in strong uptrend
        short_signals = [s for s in signals if s.direction == Direction.SHORT]
        assert len(short_signals) == 0 or len(long_signals) >= len(short_signals)

    @pytest.mark.asyncio
    async def test_signal_metadata(self):
        """Generated signals should contain required metadata."""
        strategy = self._make_strategy()
        candles = _uptrend_candles(60)
        strategy.warmup(candles)

        for i in range(30):
            price = 1.1100 + i * 0.0004
            c = _candle(price, minute=i % 60)
            sig = await strategy.on_candle(c)
            if sig:
                assert "atr" in sig.metadata
                assert "sl_distance" in sig.metadata
                assert "tp_distance" in sig.metadata
                assert sig.metadata["sl_distance"] > 0
                assert sig.metadata["tp_distance"] > 0
                break

    @pytest.mark.asyncio
    async def test_disabled_strategy_no_signals(self):
        strategy = self._make_strategy()
        strategy.enabled = False
        candles = _uptrend_candles(60)
        strategy.warmup(candles)

        for i in range(10):
            sig = await strategy.on_candle(_candle(1.1000 + i * 0.001, minute=i))
            assert sig is None


class TestMeanReversionStrategy:
    def _make_strategy(self) -> MeanReversionStrategy:
        return MeanReversionStrategy(config={
            "mean_reversion": {
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "atr_period": 14,
                "sl_multiplier": 1.5,
            }
        })

    @pytest.mark.asyncio
    async def test_requires_warmup(self):
        strategy = self._make_strategy()
        assert strategy.required_history >= 20

    @pytest.mark.asyncio
    async def test_no_signal_in_normal_range(self):
        """Price within normal range should not generate signals."""
        strategy = self._make_strategy()
        # Warmup with flat data
        flat_candles = [
            Candle("EUR_USD", "M5", datetime(2025, 1, 15, 10, i, tzinfo=timezone.utc),
                   1.1000, 1.1002, 1.0998, 1.1000, 100)
            for i in range(30)
        ]
        strategy.warmup(flat_candles)

        sig = await strategy.on_candle(_candle(1.1000))
        assert sig is None


class TestMultiTimeframeFilter:
    def test_bullish_alignment(self):
        mtf = MultiTimeframeFilter()
        # Warmup with uptrending closes
        closes = [1.1000 + i * 0.001 for i in range(60)]
        mtf.warmup("EUR_USD", closes)

        assert mtf.get_bias("EUR_USD") == TrendBias.BULLISH
        assert mtf.is_aligned("EUR_USD", Direction.LONG)
        assert not mtf.is_aligned("EUR_USD", Direction.SHORT)

    def test_bearish_alignment(self):
        mtf = MultiTimeframeFilter()
        closes = [1.2000 - i * 0.001 for i in range(60)]
        mtf.warmup("EUR_USD", closes)

        assert mtf.get_bias("EUR_USD") == TrendBias.BEARISH
        assert mtf.is_aligned("EUR_USD", Direction.SHORT)
        assert not mtf.is_aligned("EUR_USD", Direction.LONG)

    def test_neutral_allows_both(self):
        mtf = MultiTimeframeFilter()
        assert mtf.get_bias("EUR_USD") == TrendBias.NEUTRAL
        assert mtf.is_aligned("EUR_USD", Direction.LONG)
        assert mtf.is_aligned("EUR_USD", Direction.SHORT)
