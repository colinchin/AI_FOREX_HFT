"""Technical indicators — optimized for both batch (backtest) and incremental (live) use.

All batch functions operate on pandas Series.
Incremental classes maintain rolling state for live tick-by-tick updates.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  Batch (pandas Series) — for backtesting
# ═══════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (EMA-based)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Use Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Handle edge cases: zero loss → RSI=100, zero gain → RSI=0
    result = pd.Series(np.nan, index=series.index)
    valid = avg_gain.notna() & avg_loss.notna()
    zero_loss = valid & (avg_loss == 0)
    zero_gain = valid & (avg_gain == 0)
    normal = valid & ~zero_loss & ~zero_gain

    result[zero_loss] = 100.0
    result[zero_gain] = 0.0
    rs = avg_gain[normal] / avg_loss[normal]
    result[normal] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, middle (SMA), lower."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════
#  Incremental — for live trading (O(1) update per tick/candle)
# ═══════════════════════════════════════════════════════════════

class IncrementalEMA:
    """EMA with O(1) incremental updates."""

    __slots__ = ("_period", "_multiplier", "_value", "_count")

    def __init__(self, period: int) -> None:
        self._period = period
        self._multiplier = 2.0 / (period + 1)
        self._value: float | None = None
        self._count = 0

    def update(self, price: float) -> float:
        self._count += 1
        if self._value is None:
            self._value = price
        else:
            self._value = (price - self._value) * self._multiplier + self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def seed(self, prices: list[float]) -> None:
        """Warm up with historical prices."""
        for p in prices:
            self.update(p)


class IncrementalRSI:
    """RSI with O(1) incremental updates using Wilder's smoothing."""

    __slots__ = ("_period", "_avg_gain", "_avg_loss", "_prev_price", "_count", "_value")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._prev_price: float | None = None
        self._count = 0
        self._value: float | None = None

    def update(self, price: float) -> float | None:
        if self._prev_price is None:
            self._prev_price = price
            return None

        delta = price - self._prev_price
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._prev_price = price
        self._count += 1

        alpha = 1.0 / self._period
        if self._count <= self._period:
            # Accumulate initial SMA
            self._avg_gain += gain / self._period
            self._avg_loss += loss / self._period
            if self._count == self._period:
                if self._avg_loss == 0:
                    self._value = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    self._value = 100.0 - 100.0 / (1.0 + rs)
        else:
            # Wilder's smoothing
            self._avg_gain = self._avg_gain * (1 - alpha) + gain * alpha
            self._avg_loss = self._avg_loss * (1 - alpha) + loss * alpha
            if self._avg_loss == 0:
                self._value = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                self._value = 100.0 - 100.0 / (1.0 + rs)

        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def seed(self, prices: list[float]) -> None:
        for p in prices:
            self.update(p)


class IncrementalMACD:
    """MACD with O(1) incremental updates."""

    __slots__ = ("_ema_fast", "_ema_slow", "_ema_signal", "_macd_line", "_signal_line", "_histogram")

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self._ema_fast = IncrementalEMA(fast)
        self._ema_slow = IncrementalEMA(slow)
        self._ema_signal = IncrementalEMA(signal)
        self._macd_line: float | None = None
        self._signal_line: float | None = None
        self._histogram: float | None = None

    def update(self, price: float) -> tuple[float | None, float | None, float | None]:
        fast_val = self._ema_fast.update(price)
        slow_val = self._ema_slow.update(price)

        if fast_val is not None and slow_val is not None:
            self._macd_line = fast_val - slow_val
            sig = self._ema_signal.update(self._macd_line)
            self._signal_line = sig
            if sig is not None:
                self._histogram = self._macd_line - sig

        return self._macd_line, self._signal_line, self._histogram

    @property
    def ready(self) -> bool:
        return self._ema_slow.ready and self._ema_signal.ready

    def seed(self, prices: list[float]) -> None:
        for p in prices:
            self.update(p)


class IncrementalBollingerBands:
    """Bollinger Bands with efficient rolling window."""

    __slots__ = ("_period", "_num_std", "_window", "_sum", "_sum_sq")

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std
        self._window: deque[float] = deque(maxlen=period)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    def update(self, price: float) -> tuple[float | None, float | None, float | None]:
        # Remove oldest if full
        if len(self._window) == self._period:
            old = self._window[0]
            self._sum -= old
            self._sum_sq -= old * old

        self._window.append(price)
        self._sum += price
        self._sum_sq += price * price

        if len(self._window) < self._period:
            return None, None, None

        mean = self._sum / self._period
        variance = (self._sum_sq / self._period) - (mean * mean)
        std = max(variance, 0.0) ** 0.5
        upper = mean + self._num_std * std
        lower = mean - self._num_std * std
        return upper, mean, lower

    @property
    def ready(self) -> bool:
        return len(self._window) >= self._period

    def seed(self, prices: list[float]) -> None:
        for p in prices:
            self.update(p)


class IncrementalATR:
    """Average True Range with O(1) incremental updates."""

    __slots__ = ("_period", "_prev_close", "_value", "_count")

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._prev_close: float | None = None
        self._value: float | None = None
        self._count = 0

    def update(self, high: float, low: float, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None

        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )
        self._prev_close = close
        self._count += 1

        alpha = 1.0 / self._period
        if self._value is None:
            self._value = tr
        else:
            self._value = self._value * (1 - alpha) + tr * alpha

        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def seed(self, highs: list[float], lows: list[float], closes: list[float]) -> None:
        for h, l, c in zip(highs, lows, closes):
            self.update(h, l, c)
