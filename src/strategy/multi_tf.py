"""Multi-timeframe alignment filter.

Maintains H1/H4 trend state and provides alignment checks
for lower-timeframe strategies.
"""

from __future__ import annotations

from enum import Enum, auto

from src.core.events import CandleEvent, Direction
from src.strategy.indicators import IncrementalEMA
from src.utils.logger import get_logger

log = get_logger(__name__)


class TrendBias(Enum):
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()

    def is_aligned(self, direction: Direction) -> bool:
        if self is TrendBias.NEUTRAL:
            return True  # Allow trades in ranging markets
        if self is TrendBias.BULLISH and direction is Direction.LONG:
            return True
        if self is TrendBias.BEARISH and direction is Direction.SHORT:
            return True
        return False


class MultiTimeframeFilter:
    """Determines higher-timeframe trend bias for alignment filtering.

    Uses EMA(20) and EMA(50) on H1 to determine trend:
    - BULLISH: close > EMA(20) > EMA(50) — clear uptrend
    - BEARISH: close < EMA(20) < EMA(50) — clear downtrend
    - NEUTRAL: mixed/choppy — EMAs interleaved
    """

    def __init__(self) -> None:
        # H1 EMAs per instrument
        self._ema_20: dict[str, IncrementalEMA] = {}
        self._ema_50: dict[str, IncrementalEMA] = {}
        self._last_close: dict[str, float] = {}
        self._bias: dict[str, TrendBias] = {}

    def _ensure(self, instrument: str) -> None:
        if instrument not in self._ema_20:
            self._ema_20[instrument] = IncrementalEMA(20)
            self._ema_50[instrument] = IncrementalEMA(50)
            self._bias[instrument] = TrendBias.NEUTRAL

    async def on_candle(self, candle: CandleEvent) -> None:
        """Update trend state from H1 candles."""
        if candle.timeframe != "H1" or not candle.complete:
            return

        instrument = candle.instrument
        self._ensure(instrument)

        self._ema_20[instrument].update(candle.close)
        self._ema_50[instrument].update(candle.close)
        self._last_close[instrument] = candle.close

        self._update_bias(instrument)

    def warmup(self, instrument: str, closes: list[float]) -> None:
        """Seed with historical H1 closes."""
        self._ensure(instrument)
        for c in closes:
            self._ema_20[instrument].update(c)
            self._ema_50[instrument].update(c)
            self._last_close[instrument] = c
        self._update_bias(instrument)

    def _update_bias(self, instrument: str) -> None:
        ema20 = self._ema_20[instrument].value
        ema50 = self._ema_50[instrument].value
        close = self._last_close.get(instrument)

        if ema20 is None or ema50 is None or close is None:
            self._bias[instrument] = TrendBias.NEUTRAL
            return

        if close > ema20 > ema50:
            self._bias[instrument] = TrendBias.BULLISH
        elif close < ema20 < ema50:
            self._bias[instrument] = TrendBias.BEARISH
        else:
            self._bias[instrument] = TrendBias.NEUTRAL

    def get_bias(self, instrument: str) -> TrendBias:
        return self._bias.get(instrument, TrendBias.NEUTRAL)

    def is_aligned(self, instrument: str, direction: Direction) -> bool:
        """Check if a trade direction aligns with the H1 trend."""
        return self.get_bias(instrument).is_aligned(direction)
