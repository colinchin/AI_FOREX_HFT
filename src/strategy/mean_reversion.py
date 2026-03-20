"""Mean-reversion scalping strategy — secondary strategy.

Entry logic:
  LONG:  Price touches lower Bollinger Band on M5, RSI < 30, within session range
  SHORT: Price touches upper Bollinger Band on M5, RSI > 70, within session range

Exit logic:
  - TP: Bollinger middle band (SMA)
  - SL: 1.5x distance from entry to band edge
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.events import CandleEvent, Direction, SignalEvent, SignalStrength
from src.core.models import Candle
from src.strategy.base import Strategy
from src.strategy.indicators import (
    IncrementalATR,
    IncrementalBollingerBands,
    IncrementalRSI,
)
from src.utils.helpers import utc_now

if TYPE_CHECKING:
    from src.core.bus import EventBus


class MeanReversionStrategy(Strategy):
    """Fade extreme moves back to the mean using BB + RSI."""

    def __init__(self, config: dict, event_bus: EventBus | None = None) -> None:
        super().__init__("mean_reversion", config, event_bus)

        cfg = config.get("mean_reversion", config)
        self._bb_period = cfg.get("bb_period", 20)
        self._bb_std = cfg.get("bb_std", 2.0)
        self._rsi_period = cfg.get("rsi_period", 14)
        self._rsi_oversold = cfg.get("rsi_oversold", 30)
        self._rsi_overbought = cfg.get("rsi_overbought", 70)
        self._atr_period = cfg.get("atr_period", 14)
        self._sl_multiplier = cfg.get("sl_multiplier", 1.5)

        # Per-instrument parameter overrides (from pair_params_file)
        self._pair_params: dict[str, dict] = cfg.get("pair_params", {})

        # Indicators per instrument
        self._bb: dict[str, IncrementalBollingerBands] = {}
        self._rsi: dict[str, IncrementalRSI] = {}
        self._atr: dict[str, IncrementalATR] = {}

        # Resolved per-instrument config (populated in _ensure_indicators)
        self._icfg: dict[str, dict] = {}

        # Session tracking for range detection
        self._session_high: dict[str, float] = {}
        self._session_low: dict[str, float] = {}
        self._candle_count: dict[str, int] = {}
        self._last_date: dict[str, str] = {}  # instrument → YYYY-MM-DD

    def _ensure_indicators(self, instrument: str) -> None:
        if instrument not in self._bb:
            # Resolve per-instrument params (override globals with pair_params)
            pp = self._pair_params.get(instrument, {})
            icfg = {
                "bb_std": pp.get("bb_std", self._bb_std),
                "rsi_oversold": pp.get("rsi_oversold", self._rsi_oversold),
                "rsi_overbought": pp.get("rsi_overbought", self._rsi_overbought),
                "sl_multiplier": pp.get("sl_multiplier", self._sl_multiplier),
            }
            self._icfg[instrument] = icfg

            self._bb[instrument] = IncrementalBollingerBands(self._bb_period, icfg["bb_std"])
            self._rsi[instrument] = IncrementalRSI(self._rsi_period)
            self._atr[instrument] = IncrementalATR(self._atr_period)
            self._session_high[instrument] = -float("inf")
            self._session_low[instrument] = float("inf")
            self._candle_count[instrument] = 0

    @property
    def required_history(self) -> int:
        return max(self._bb_period, self._rsi_period, self._atr_period) + 5

    def warmup(self, candles: list[Candle]) -> None:
        if not candles:
            return
        # Only accept M5 candles — H1 data would corrupt BB/RSI/ATR state
        if candles[0].timeframe != "M5":
            return
        instrument = candles[0].instrument
        self._ensure_indicators(instrument)
        for c in candles:
            self._bb[instrument].update(c.close)
            self._rsi[instrument].update(c.close)
            self._atr[instrument].update(c.high, c.low, c.close)
            self._session_high[instrument] = max(self._session_high[instrument], c.high)
            self._session_low[instrument] = min(self._session_low[instrument], c.low)
            self._candle_count[instrument] = self._candle_count.get(instrument, 0) + 1

    async def on_candle(self, candle: CandleEvent) -> SignalEvent | None:
        if not self._enabled or candle.timeframe != "M5" or not candle.complete:
            return None

        instrument = candle.instrument
        self._ensure_indicators(instrument)

        # Reset session range on date change
        date_str = candle.timestamp.strftime("%Y-%m-%d")
        if self._last_date.get(instrument) != date_str:
            self._last_date[instrument] = date_str
            self.reset_session(instrument)

        # Update indicators
        upper, middle, lower = self._bb[instrument].update(candle.close)
        rsi_val = self._rsi[instrument].update(candle.close)
        atr_val = self._atr[instrument].update(candle.high, candle.low, candle.close)

        # Update session range
        self._session_high[instrument] = max(self._session_high[instrument], candle.high)
        self._session_low[instrument] = min(self._session_low[instrument], candle.low)
        self._candle_count[instrument] = self._candle_count.get(instrument, 0) + 1

        if not all(v is not None for v in [upper, middle, lower, rsi_val, atr_val]):
            return None

        # Check if price is within session range (not breakout)
        session_range = self._session_high[instrument] - self._session_low[instrument]
        if session_range > 0 and self._candle_count.get(instrument, 0) > 12:
            # Price should not be at new session extremes (breakout filter)
            range_position = (candle.close - self._session_low[instrument]) / session_range
            is_breakout = range_position > 0.95 or range_position < 0.05
        else:
            is_breakout = False

        # Per-instrument thresholds
        icfg = self._icfg[instrument]
        rsi_oversold = icfg["rsi_oversold"]
        rsi_overbought = icfg["rsi_overbought"]
        sl_multiplier = icfg["sl_multiplier"]

        # ── LONG: price at/below lower BB + RSI oversold ──
        if (
            candle.close <= lower
            and rsi_val <= rsi_oversold
            and not is_breakout
        ):
            # TP = middle band, SL = ATR-based below entry
            tp_distance = middle - candle.close
            sl_distance = atr_val * sl_multiplier

            if tp_distance > 0 and sl_distance > 0:
                return SignalEvent(
                    instrument=instrument,
                    direction=Direction.LONG,
                    strength=SignalStrength.MODERATE if rsi_val < 25 else SignalStrength.WEAK,
                    strategy_name=self._name,
                    timestamp=utc_now(),
                    metadata={
                        "rsi": round(rsi_val, 2),
                        "bb_upper": round(upper, 6),
                        "bb_middle": round(middle, 6),
                        "bb_lower": round(lower, 6),
                        "atr": round(atr_val, 6),
                        "tp_distance": round(tp_distance, 6),
                        "sl_distance": round(sl_distance, 6),
                    },
                )

        # ── SHORT: price at/above upper BB + RSI overbought ──
        if (
            candle.close >= upper
            and rsi_val >= rsi_overbought
            and not is_breakout
        ):
            tp_distance = candle.close - middle
            sl_distance = atr_val * sl_multiplier

            if tp_distance > 0 and sl_distance > 0:
                return SignalEvent(
                    instrument=instrument,
                    direction=Direction.SHORT,
                    strength=SignalStrength.MODERATE if rsi_val > 75 else SignalStrength.WEAK,
                    strategy_name=self._name,
                    timestamp=utc_now(),
                    metadata={
                        "rsi": round(rsi_val, 2),
                        "bb_upper": round(upper, 6),
                        "bb_middle": round(middle, 6),
                        "bb_lower": round(lower, 6),
                        "atr": round(atr_val, 6),
                        "tp_distance": round(tp_distance, 6),
                        "sl_distance": round(sl_distance, 6),
                    },
                )

        return None

    def reset_session(self, instrument: str) -> None:
        """Reset session range tracking (call at session boundaries)."""
        self._session_high[instrument] = -float("inf")
        self._session_low[instrument] = float("inf")
        self._candle_count[instrument] = 0
