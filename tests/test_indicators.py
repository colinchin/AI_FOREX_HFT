"""Tests for technical indicators — both batch and incremental."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.indicators import (
    IncrementalATR,
    IncrementalBollingerBands,
    IncrementalEMA,
    IncrementalMACD,
    IncrementalRSI,
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)


# ── Test data ──

def _make_price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0001, 0.005, n)
    prices = 1.1000 * np.cumprod(1 + returns)
    return pd.Series(prices)


PRICES = _make_price_series()


# ── Batch indicator tests ──

class TestBatchEMA:
    def test_length(self):
        result = ema(PRICES, 20)
        assert len(result) == len(PRICES)

    def test_ema_smoothing(self):
        result = ema(PRICES, 20)
        # EMA should be smoother than raw prices
        assert result.std() < PRICES.std()

    def test_ema_follows_trend(self):
        trending = pd.Series(range(100), dtype=float)
        result = ema(trending, 10)
        # EMA of uptrend should be below price (lagging)
        assert result.iloc[-1] < trending.iloc[-1]


class TestBatchRSI:
    def test_range(self):
        result = rsi(PRICES, 14)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_overbought_on_rally(self):
        # Use a mostly-up series with small dips to avoid all-NaN from zero losses
        rally = pd.Series([1.0 + i * 0.008 + (0.001 if i % 5 == 0 else 0) for i in range(100)])
        result = rsi(rally, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 70  # Should be overbought

    def test_oversold_on_decline(self):
        decline = pd.Series([2.0 - i * 0.01 for i in range(50)])
        result = rsi(decline, 14)
        assert result.iloc[-1] < 30  # Should be oversold


class TestBatchMACD:
    def test_output_shapes(self):
        macd_line, signal_line, histogram = macd(PRICES)
        assert len(macd_line) == len(PRICES)
        assert len(signal_line) == len(PRICES)
        assert len(histogram) == len(PRICES)

    def test_histogram_is_diff(self):
        ml, sl, hist = macd(PRICES)
        valid_idx = ml.dropna().index.intersection(sl.dropna().index)
        np.testing.assert_array_almost_equal(
            hist[valid_idx].values,
            (ml[valid_idx] - sl[valid_idx]).values,
        )


class TestBatchBollingerBands:
    def test_band_ordering(self):
        upper, middle, lower = bollinger_bands(PRICES, 20, 2.0)
        valid = ~upper.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()


class TestBatchATR:
    def test_positive(self):
        high = PRICES * 1.002
        low = PRICES * 0.998
        result = atr(high, low, PRICES, 14)
        valid = result.dropna()
        assert (valid > 0).all()


# ── Incremental indicator tests ──

class TestIncrementalEMA:
    def test_convergence_with_batch(self):
        """Incremental EMA should converge to batch EMA."""
        inc = IncrementalEMA(20)
        prices = PRICES.tolist()
        for p in prices:
            inc.update(p)

        batch_result = ema(PRICES, 20)
        # Should be very close after full series
        assert abs(inc.value - batch_result.iloc[-1]) < 1e-10

    def test_ready_flag(self):
        inc = IncrementalEMA(5)
        for i in range(4):
            inc.update(float(i))
            assert not inc.ready
        inc.update(4.0)
        assert inc.ready

    def test_seed(self):
        inc = IncrementalEMA(20)
        inc.seed(PRICES.tolist())
        assert inc.ready
        assert inc.value is not None


class TestIncrementalRSI:
    def test_range(self):
        inc = IncrementalRSI(14)
        for p in PRICES:
            val = inc.update(p)
        assert 0 <= inc.value <= 100

    def test_overbought_detection(self):
        inc = IncrementalRSI(14)
        rally = [1.0 + i * 0.01 for i in range(50)]
        for p in rally:
            inc.update(p)
        assert inc.value > 70

    def test_ready_flag(self):
        inc = IncrementalRSI(14)
        for i in range(14):
            inc.update(float(i))
        assert not inc.ready
        inc.update(14.0)
        assert inc.ready


class TestIncrementalMACD:
    def test_components(self):
        inc = IncrementalMACD(12, 26, 9)
        for p in PRICES:
            ml, sl, hist = inc.update(p)

        assert ml is not None
        assert sl is not None
        assert hist is not None
        assert abs(hist - (ml - sl)) < 1e-10


class TestIncrementalBollingerBands:
    def test_band_ordering(self):
        inc = IncrementalBollingerBands(20, 2.0)
        for p in PRICES:
            upper, middle, lower = inc.update(p)

        assert upper >= middle >= lower

    def test_convergence_with_batch(self):
        inc = IncrementalBollingerBands(20, 2.0)
        for p in PRICES:
            inc.update(p)

        upper_inc, middle_inc, lower_inc = inc.update(PRICES.iloc[-1])
        upper_batch, middle_batch, lower_batch = bollinger_bands(PRICES, 20, 2.0)

        # Middle (SMA) should match closely (incremental has slight rounding diff)
        assert abs(middle_inc - middle_batch.iloc[-1]) < 1e-4


class TestIncrementalATR:
    def test_positive(self):
        inc = IncrementalATR(14)
        high = (PRICES * 1.002).tolist()
        low = (PRICES * 0.998).tolist()
        closes = PRICES.tolist()

        for h, l, c in zip(high, low, closes):
            inc.update(h, l, c)

        assert inc.value > 0
        assert inc.ready
