"""Backtest data loader — load parquet/CSV historical data into event streams."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd

from src.core.events import CandleEvent, TickEvent
from src.core.models import Candle


class BacktestDataLoader:
    """Load historical data for backtesting."""

    def __init__(self, data_dir: str = "data/parquet") -> None:
        self._data_dir = Path(data_dir)

    def load_candles(
        self,
        instrument: str,
        granularity: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """Load candle data from parquet file."""
        path = self._data_dir / f"{instrument}_{granularity}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data file: {path}")

        df = pd.read_parquet(path)

        if "timestamp" in df.columns and df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        if from_date:
            df = df[df["timestamp"] >= pd.Timestamp(from_date, tz="UTC")]
        if to_date:
            df = df[df["timestamp"] <= pd.Timestamp(to_date, tz="UTC")]

        return df.sort_values("timestamp").reset_index(drop=True)

    def df_to_candle_events(
        self, df: pd.DataFrame, instrument: str, timeframe: str
    ) -> list[CandleEvent]:
        """Convert DataFrame rows to CandleEvent objects."""
        events = []
        for _, row in df.iterrows():
            events.append(CandleEvent(
                instrument=instrument,
                timeframe=timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                timestamp=row["timestamp"],
                complete=True,
            ))
        return events

    def df_to_candle_models(
        self, df: pd.DataFrame, instrument: str, timeframe: str
    ) -> list[Candle]:
        """Convert DataFrame rows to Candle model objects."""
        candles = []
        for _, row in df.iterrows():
            candles.append(Candle(
                instrument=instrument,
                timeframe=timeframe,
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                complete=True,
            ))
        return candles

    def generate_synthetic_ticks(
        self, candles: list[CandleEvent], ticks_per_candle: int = 10
    ) -> list[TickEvent]:
        """Generate synthetic ticks from candles for tick-level backtesting.

        Uses OHLC to create a realistic price path within each candle.
        """
        ticks = []
        for candle in candles:
            # Simulate price path: Open → High/Low → Close
            # Determine direction
            if candle.is_bullish:
                # O → L → H → C
                path = [candle.open, candle.low, candle.high, candle.close]
            else:
                # O → H → L → C
                path = [candle.open, candle.high, candle.low, candle.close]

            # Interpolate to desired tick count
            import numpy as np
            prices = np.interp(
                np.linspace(0, len(path) - 1, ticks_per_candle),
                range(len(path)),
                path,
            )

            for i, price in enumerate(prices):
                # Simulate a small spread (0.00015 for majors)
                half_spread = 0.00008
                ticks.append(TickEvent(
                    instrument=candle.instrument,
                    bid=price - half_spread,
                    ask=price + half_spread,
                    timestamp=candle.timestamp,
                ))

        return ticks
