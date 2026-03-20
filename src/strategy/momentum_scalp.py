"""Momentum scalping strategy — primary strategy.

Entry logic:
  LONG:  M5 close > EMA(20), RSI crosses above 50, MACD histogram positive & rising,
         spread < threshold, H1 trend bullish (close > EMA(50))
  SHORT: Mirror conditions

Exit logic:
  - TP: 2x ATR(14) on M5
  - SL: 1.5x ATR(14) on M5
  - Trailing stop: activate at 1x ATR profit, trail at 0.5x ATR
  - Time exit: close if open > 30 minutes

Filters:
  - Active sessions only (London, NY, overlap)
  - Min candle body size (avoid doji)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.events import CandleEvent, Direction, SignalEvent, SignalStrength
from src.core.models import Candle
from src.strategy.base import Strategy
from src.strategy.indicators import (
    IncrementalATR,
    IncrementalEMA,
    IncrementalMACD,
    IncrementalRSI,
)
from src.utils.helpers import pip_value, utc_now

if TYPE_CHECKING:
    from src.core.bus import EventBus


class MomentumScalpStrategy(Strategy):
    """Trend-following momentum scalper on M5 with H1 alignment."""

    def __init__(self, config: dict, event_bus: EventBus | None = None) -> None:
        super().__init__("momentum_scalp", config, event_bus)

        cfg = config.get("momentum_scalp", config)

        # Indicator parameters
        self._ema_fast_period = cfg.get("ema_fast", 9)
        self._ema_slow_period = cfg.get("ema_slow", 20)
        self._rsi_period = cfg.get("rsi_period", 14)
        self._macd_fast = cfg.get("macd_fast", 12)
        self._macd_slow = cfg.get("macd_slow", 26)
        self._macd_signal = cfg.get("macd_signal", 9)
        self._atr_period = cfg.get("atr_period", 14)

        # Exit parameters
        self._tp_atr_mult = cfg.get("tp_atr_mult", 2.0)
        self._sl_atr_mult = cfg.get("sl_atr_mult", 1.5)
        self._trailing_activate = cfg.get("trailing_activate_atr", 1.0)
        self._trailing_distance = cfg.get("trailing_distance_atr", 0.5)
        self._min_body_pips = cfg.get("min_candle_body_pips", 1.5)

        # Minimum EMA separation in pips to avoid chop
        self._min_ema_separation_pips = cfg.get("min_ema_separation_pips", 0.8)

        # Cooldown: minimum M5 candles between signals on the same instrument
        self._signal_cooldown_candles = cfg.get("signal_cooldown_candles", 3)  # 15 min

        # M5 indicators (per instrument)
        self._m5_indicators: dict[str, _InstrumentIndicators] = {}

        # H1 trend EMA (per instrument)
        self._h1_ema: dict[str, IncrementalEMA] = {}
        self._h1_last_close: dict[str, float] = {}

        # Previous state for crossover detection
        self._prev_rsi: dict[str, float | None] = {}
        self._prev_histogram: dict[str, float | None] = {}

        # Cooldown tracking: instrument → candles since last signal
        self._candles_since_signal: dict[str, int] = {}

        # Previous candle tracking for breakout confirmation
        self._prev_candle: dict[str, CandleEvent | None] = {}

    def _get_m5(self, instrument: str) -> _InstrumentIndicators:
        if instrument not in self._m5_indicators:
            self._m5_indicators[instrument] = _InstrumentIndicators(
                ema_fast_period=self._ema_fast_period,
                ema_slow_period=self._ema_slow_period,
                rsi_period=self._rsi_period,
                macd_fast=self._macd_fast,
                macd_slow=self._macd_slow,
                macd_signal=self._macd_signal,
                atr_period=self._atr_period,
            )
        return self._m5_indicators[instrument]

    def _get_h1_ema(self, instrument: str) -> IncrementalEMA:
        if instrument not in self._h1_ema:
            self._h1_ema[instrument] = IncrementalEMA(50)
        return self._h1_ema[instrument]

    @property
    def required_history(self) -> int:
        # Need enough M5 candles for the slowest indicator to warm up
        return max(self._macd_slow + self._macd_signal, 50) + 5

    def warmup(self, candles: list[Candle]) -> None:
        """Seed indicators from historical candle data."""
        if not candles:
            return

        instrument = candles[0].instrument
        tf = candles[0].timeframe

        if tf == "M5":
            ind = self._get_m5(instrument)
            for c in candles:
                ind.update(c.high, c.low, c.close)
                # Track RSI and histogram history for crossover detection
                self._prev_rsi[instrument] = ind.rsi.value
                self._prev_histogram[instrument] = ind.macd_histogram

        elif tf == "H1":
            h1_ema = self._get_h1_ema(instrument)
            for c in candles:
                h1_ema.update(c.close)
                self._h1_last_close[instrument] = c.close

    async def on_candle(self, candle: CandleEvent) -> SignalEvent | None:
        """Process an M5 or H1 candle."""
        if not self._enabled:
            return None

        if candle.timeframe == "H1":
            self._update_h1(candle)
            return None

        if candle.timeframe != "M5":
            return None

        if not candle.complete:
            return None

        instrument = candle.instrument
        ind = self._get_m5(instrument)

        # Update indicators
        ind.update(candle.high, candle.low, candle.close)

        # Track cooldown
        self._candles_since_signal[instrument] = self._candles_since_signal.get(instrument, 999) + 1

        if not ind.ready:
            return None

        # Get current indicator values
        ema_fast = ind.ema_fast.value
        ema_slow = ind.ema_slow.value
        rsi_val = ind.rsi.value
        _, _, histogram = ind.macd_line, ind.macd_signal_line, ind.macd_histogram
        atr_val = ind.atr.value

        if any(v is None for v in [ema_fast, ema_slow, rsi_val, histogram, atr_val]):
            return None

        # Get previous values for crossover detection
        prev_rsi = self._prev_rsi.get(instrument)
        prev_hist = self._prev_histogram.get(instrument)

        # Store current values for next iteration
        self._prev_rsi[instrument] = rsi_val
        self._prev_histogram[instrument] = histogram

        if prev_rsi is None or prev_hist is None:
            self._prev_candle[instrument] = candle
            return None

        # ── Cooldown: don't signal too frequently ──
        if self._candles_since_signal[instrument] < self._signal_cooldown_candles:
            self._prev_candle[instrument] = candle
            return None

        # ── Minimum candle body check (avoid doji / indecision) ──
        pv = pip_value(instrument)
        body_pips = candle.body / pv
        if body_pips < self._min_body_pips:
            self._prev_candle[instrument] = candle
            return None

        # ── EMA separation check (avoid chop) ──
        ema_sep_pips = abs(ema_fast - ema_slow) / pv
        if ema_sep_pips < self._min_ema_separation_pips:
            self._prev_candle[instrument] = candle
            return None

        # ── H1 trend alignment ──
        h1_ema = self._h1_ema.get(instrument)
        h1_close = self._h1_last_close.get(instrument)

        # Get previous candle for breakout confirmation
        prev_candle = self._prev_candle.get(instrument)

        # ── Check LONG conditions ──
        long_signal = self._check_long(
            candle=candle,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_val=rsi_val,
            prev_rsi=prev_rsi,
            histogram=histogram,
            prev_hist=prev_hist,
            h1_ema=h1_ema,
            h1_close=h1_close,
            prev_candle=prev_candle,
        )
        if long_signal:
            self._candles_since_signal[instrument] = 0
            self._prev_candle[instrument] = candle
            return self._create_signal(instrument, Direction.LONG, rsi_val, histogram, atr_val)

        # ── Check SHORT conditions ──
        short_signal = self._check_short(
            candle=candle,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_val=rsi_val,
            prev_rsi=prev_rsi,
            histogram=histogram,
            prev_hist=prev_hist,
            h1_ema=h1_ema,
            h1_close=h1_close,
            prev_candle=prev_candle,
        )
        if short_signal:
            self._candles_since_signal[instrument] = 0
            self._prev_candle[instrument] = candle
            return self._create_signal(instrument, Direction.SHORT, rsi_val, histogram, atr_val)

        self._prev_candle[instrument] = candle

        return None

    def _check_long(
        self,
        candle: CandleEvent,
        ema_fast: float,
        ema_slow: float,
        rsi_val: float,
        prev_rsi: float,
        histogram: float,
        prev_hist: float,
        h1_ema: IncrementalEMA | None,
        h1_close: float | None,
        prev_candle: CandleEvent | None = None,
    ) -> bool:
        """Evaluate long entry — trend-following with momentum confirmation.

        Requires ALL of:
        1. EMA(9) > EMA(20) — uptrend established
        2. Price > EMA(9) — momentum intact
        3. RSI 50-70 — momentum present, not overbought
        4. MACD histogram positive
        5. Bullish candle closing above previous candle high (breakout)
        6. H1 trend bullish — mandatory when available
        """
        close = candle.close

        # 1. Uptrend: fast EMA above slow EMA
        if ema_fast <= ema_slow:
            return False

        # 2. Price above fast EMA (strong trend position)
        if close <= ema_fast:
            return False

        # 3. RSI 50-70: momentum present but not overbought
        if rsi_val < 50 or rsi_val > 70:
            return False

        # 4. MACD histogram positive
        if histogram <= 0:
            return False

        # 5. Bullish candle with breakout above previous high
        if not candle.is_bullish:
            return False
        if prev_candle is not None and close <= prev_candle.high:
            return False

        # 6. H1 trend must be bullish — block if H1 data not yet available
        if h1_ema is None or not h1_ema.ready or h1_close is None:
            return False
        if h1_close <= h1_ema.value:
            return False

        return True

    def _check_short(
        self,
        candle: CandleEvent,
        ema_fast: float,
        ema_slow: float,
        rsi_val: float,
        prev_rsi: float,
        histogram: float,
        prev_hist: float,
        h1_ema: IncrementalEMA | None,
        h1_close: float | None,
        prev_candle: CandleEvent | None = None,
    ) -> bool:
        """Evaluate short entry — mirror of long."""
        close = candle.close

        # Downtrend: fast EMA below slow EMA
        if ema_fast >= ema_slow:
            return False

        # Price below fast EMA
        if close >= ema_fast:
            return False

        # RSI 30-50: momentum present, not oversold
        if rsi_val > 50 or rsi_val < 30:
            return False

        # MACD histogram negative
        if histogram >= 0:
            return False

        # Bearish candle with breakout below previous low
        if candle.is_bullish:
            return False
        if prev_candle is not None and close >= prev_candle.low:
            return False

        # H1 trend must be bearish — block if H1 data not yet available
        if h1_ema is None or not h1_ema.ready or h1_close is None:
            return False
        if h1_close >= h1_ema.value:
            return False

        return True

    def _create_signal(
        self,
        instrument: str,
        direction: Direction,
        rsi_val: float,
        histogram: float,
        atr_val: float,
    ) -> SignalEvent:
        """Create a signal event with metadata."""
        # Determine signal strength based on indicator confluence
        strength = SignalStrength.MODERATE
        if direction is Direction.LONG:
            if rsi_val > 55 and histogram > 0:
                strength = SignalStrength.STRONG
        else:
            if rsi_val < 45 and histogram < 0:
                strength = SignalStrength.STRONG

        return SignalEvent(
            instrument=instrument,
            direction=direction,
            strength=strength,
            strategy_name=self._name,
            timestamp=utc_now(),
            metadata={
                "rsi": round(rsi_val, 2),
                "macd_histogram": round(histogram, 6),
                "atr": round(atr_val, 6),
                "tp_distance": round(atr_val * self._tp_atr_mult, 6),
                "sl_distance": round(atr_val * self._sl_atr_mult, 6),
                "trailing_activate": round(atr_val * self._trailing_activate, 6),
                "trailing_distance": round(atr_val * self._trailing_distance, 6),
            },
        )

    def _update_h1(self, candle: CandleEvent) -> None:
        """Update H1 trend indicators."""
        h1_ema = self._get_h1_ema(candle.instrument)
        h1_ema.update(candle.close)
        self._h1_last_close[candle.instrument] = candle.close


class _InstrumentIndicators:
    """Container for all M5 indicators for a single instrument."""

    __slots__ = (
        "ema_fast", "ema_slow", "rsi", "macd",
        "atr", "macd_line", "macd_signal_line", "macd_histogram",
    )

    def __init__(
        self,
        ema_fast_period: int,
        ema_slow_period: int,
        rsi_period: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        atr_period: int,
    ) -> None:
        self.ema_fast = IncrementalEMA(ema_fast_period)
        self.ema_slow = IncrementalEMA(ema_slow_period)
        self.rsi = IncrementalRSI(rsi_period)
        self.macd = IncrementalMACD(macd_fast, macd_slow, macd_signal)
        self.atr = IncrementalATR(atr_period)
        self.macd_line: float | None = None
        self.macd_signal_line: float | None = None
        self.macd_histogram: float | None = None

    def update(self, high: float, low: float, close: float) -> None:
        self.ema_fast.update(close)
        self.ema_slow.update(close)
        self.rsi.update(close)
        ml, sl, hist = self.macd.update(close)
        self.macd_line = ml
        self.macd_signal_line = sl
        self.macd_histogram = hist
        self.atr.update(high, low, close)

    @property
    def ready(self) -> bool:
        return (
            self.ema_fast.ready
            and self.ema_slow.ready
            and self.rsi.ready
            and self.macd.ready
            and self.atr.ready
        )
