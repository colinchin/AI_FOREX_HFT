"""Backtest engine — replays historical data through strategy and risk modules.

Uses the SAME strategy and risk logic as live trading for consistency.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backtest.data_loader import BacktestDataLoader
from backtest.metrics import PerformanceReport, calculate_metrics
from backtest.simulator import BacktestSimulator
from src.core.events import (
    CandleEvent,
    Direction,
    OrderEvent,
    OrderType,
    SignalEvent,
)
from src.risk.position_sizer import PositionSizer
from src.strategy.base import Strategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.helpers import is_within_session, pip_value, spread_pips
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_conversion_series(data_dir: str = "data/parquet") -> dict[str, "pd.DataFrame"]:
    """Load time-indexed close prices for all available M5 instruments.

    Returns a dict of instrument → DataFrame(timestamp, close) sorted by time.
    Used by ConversionRateCache for point-in-time rate lookups with no
    look-ahead bias.
    """
    from pathlib import Path
    import pandas as pd

    series: dict[str, pd.DataFrame] = {}
    data_path = Path(data_dir)
    for f in data_path.glob("*_M5.parquet"):
        instrument = f.stem.replace("_M5", "")
        try:
            df = pd.read_parquet(f, columns=["timestamp", "close"])
            if not df.empty:
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                df = df.sort_values("timestamp").reset_index(drop=True)
                series[instrument] = df
        except Exception:
            pass
    return series


# Keep backwards-compatible name for callers that don't need time-series
def load_conversion_rates(data_dir: str = "data/parquet") -> dict[str, float]:
    """Load latest close prices — DEPRECATED, use load_conversion_series instead.

    Kept for backwards compatibility but introduces look-ahead bias in backtests.
    """
    series = load_conversion_series(data_dir)
    return {inst: float(df["close"].iloc[-1]) for inst, df in series.items() if not df.empty}


class ConversionRateCache:
    """Point-in-time conversion rate lookup for backtesting.

    Uses bisect to find the most recent close price at or before a given
    timestamp for each instrument. No look-ahead bias.
    """

    def __init__(self, series: dict[str, "pd.DataFrame"]) -> None:
        import numpy as np
        # Pre-extract numpy arrays for fast bisect lookups
        self._timestamps: dict[str, "np.ndarray"] = {}
        self._closes: dict[str, "np.ndarray"] = {}
        for instrument, df in series.items():
            self._timestamps[instrument] = df["timestamp"].values  # numpy datetime64
            self._closes[instrument] = df["close"].values

        # Cache: instrument → current rate (updated as backtest advances)
        self._current_rates: dict[str, float] = {}

    def advance_time(self, timestamp) -> None:
        """Update all conversion rates to the most recent price at or before timestamp.

        Call this once per candle in the replay loop.
        """
        import numpy as np

        ts = np.datetime64(timestamp)
        for instrument, ts_arr in self._timestamps.items():
            idx = np.searchsorted(ts_arr, ts, side="right") - 1
            if idx >= 0:
                self._current_rates[instrument] = float(self._closes[instrument][idx])

    def set_rate(self, instrument: str, price: float) -> None:
        """Manually set a rate (e.g. from the primary instrument's candle)."""
        self._current_rates[instrument] = price

    def lookup(self, instrument: str) -> float | None:
        """Look up the current rate for an instrument."""
        return self._current_rates.get(instrument)


class BacktestEngine:
    """Replay historical candles through strategy → risk → simulated execution.

    The engine feeds candles sequentially, collects signals, applies basic
    risk checks, and executes through the simulator.
    """

    def __init__(
        self,
        strategy: Strategy,
        simulator: BacktestSimulator,
        config: dict[str, Any],
        multi_tf_filter: MultiTimeframeFilter | None = None,
        account_currency: str = "AUD",
        conversion_rates: dict[str, float] | None = None,
        conversion_cache: ConversionRateCache | None = None,
    ) -> None:
        self._strategy = strategy
        self._simulator = simulator
        self._config = config
        self._mtf = multi_tf_filter
        self._account_currency = account_currency

        # Point-in-time conversion rate cache (preferred — no look-ahead bias).
        # Falls back to static dict if no cache provided (backwards compat).
        self._conversion_cache = conversion_cache
        self._conversion_rates: dict[str, float] = dict(conversion_rates or {})

        # Risk parameters
        risk = config.get("risk", config)
        if "risk" in risk:
            risk = risk["risk"]
        self._max_risk_per_trade = risk.get("max_risk_per_trade", 0.01)
        self._max_open_positions = risk.get("max_open_positions", 3)
        self._max_spread_pips = risk.get("max_spread_pips", 2.0)
        self._max_daily_loss = risk.get("max_daily_loss", 0.03)
        self._max_trades_per_day = risk.get("max_trades_per_day", 50)
        self._max_consecutive_losses = risk.get("max_consecutive_losses", 5)

        # Position sizer — same as live to ensure consistency
        ps_cfg = risk.get("position_sizing", {})
        self._sizer = PositionSizer(
            method=ps_cfg.get("method", "atr_based"),
            min_units=ps_cfg.get("min_units", 1),
            max_units=ps_cfg.get("max_units", 100_000),
            max_position_pct=ps_cfg.get("max_position_pct", 0.10),
            account_currency=account_currency,
        )

        # Session config
        sessions = config.get("sessions", {})
        self._active_sessions = sessions.get("active_sessions", ["london", "new_york"])
        self._session_times = sessions

        # Stats
        self._signals_generated = 0
        self._signals_filtered = 0
        self._orders_placed = 0
        self._current_date = ""
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._halted_today = False

    async def run(
        self,
        candles: list[CandleEvent],
        h1_candles: list[CandleEvent] | None = None,
        warmup_count: int | None = None,
    ) -> PerformanceReport:
        """Run the backtest on historical candle data.

        Args:
            candles: M5 candle events (primary signal timeframe)
            h1_candles: H1 candle events for multi-TF alignment
            warmup_count: Number of candles to use for indicator warmup
        """
        if warmup_count is None:
            warmup_count = self._strategy.required_history

        # Warmup H1 if available
        if h1_candles and self._mtf:
            h1_warmup = h1_candles[:warmup_count]
            from src.core.models import Candle
            instrument = h1_warmup[0].instrument if h1_warmup else ""
            closes = [c.close for c in h1_warmup]
            self._mtf.warmup(instrument, closes)

        # Warmup M5
        warmup_candles = candles[:warmup_count]
        from src.core.models import Candle
        warmup_models = [
            Candle(
                instrument=c.instrument, timeframe=c.timeframe,
                timestamp=c.timestamp, open=c.open, high=c.high,
                low=c.low, close=c.close, volume=c.volume,
            )
            for c in warmup_candles
        ]
        self._strategy.warmup(warmup_models)

        # Build H1 candle index for alignment during replay
        h1_by_time = {}
        if h1_candles:
            for hc in h1_candles:
                h1_by_time[hc.timestamp] = hc

        total = len(candles)
        log.info("backtest_started", total_candles=total, warmup=warmup_count)

        # Replay
        h1_idx = 0
        for i, candle in enumerate(candles[warmup_count:], start=warmup_count):
            # Feed any H1 candles that occurred before this M5 candle
            if h1_candles and self._mtf:
                while h1_idx < len(h1_candles) and h1_candles[h1_idx].timestamp <= candle.timestamp:
                    await self._mtf.on_candle(h1_candles[h1_idx])
                    h1_idx += 1

            # Advance conversion rate cache to current timestamp (no look-ahead)
            if self._conversion_cache is not None:
                self._conversion_cache.advance_time(candle.timestamp)
                self._conversion_cache.set_rate(candle.instrument, candle.close)
            else:
                self._conversion_rates[candle.instrument] = candle.close

            # Check open positions for SL/TP on this candle
            close_events = self._simulator.process_candle(candle)
            for ce in close_events:
                self._on_close(ce)

            # Daily reset
            date_str = candle.timestamp.strftime("%Y-%m-%d")
            if date_str != self._current_date:
                self._current_date = date_str
                self._daily_trades = 0
                self._daily_pnl = 0.0
                self._halted_today = False
                self._consecutive_losses = 0

            if self._halted_today:
                continue

            # Session filter — no new signals outside active sessions
            # SL/TP continues to protect existing positions on all candles
            if not self._is_session_active(candle):
                continue

            # Generate signal
            signal = await self._strategy.on_candle(candle)
            if signal is None:
                continue

            self._signals_generated += 1

            # Apply risk checks
            if not self._pass_risk_checks(signal, candle):
                self._signals_filtered += 1
                continue

            # Multi-TF alignment
            if self._mtf and not self._mtf.is_aligned(signal.instrument, signal.direction):
                self._signals_filtered += 1
                continue

            # Position sizing — uses same PositionSizer as live
            equity = self._simulator.equity
            sl_distance = signal.metadata.get("sl_distance", 0)
            if sl_distance <= 0:
                continue

            # Account for simulated spread in sizing (matches live behaviour)
            pv = pip_value(signal.instrument)
            spread = self._simulator._spread_pips * pv

            units = self._sizer.calculate(
                equity=equity,
                risk_pct=self._max_risk_per_trade,
                sl_distance=sl_distance,
                instrument=signal.instrument,
                current_price=candle.close,
                spread=spread,
                rate_lookup=self._rate_lookup,
            )
            if units <= 0:
                continue

            tp_distance = signal.metadata.get("tp_distance", sl_distance * 1.5)
            trailing_dist = signal.metadata.get("trailing_distance")
            trailing_activate = signal.metadata.get("trailing_activate")

            if signal.direction is Direction.LONG:
                sl = candle.close - sl_distance
                tp = candle.close + tp_distance
            else:
                sl = candle.close + sl_distance
                tp = candle.close - tp_distance

            order = OrderEvent(
                instrument=signal.instrument,
                direction=signal.direction,
                units=units,
                order_type=OrderType.MARKET,
                stop_loss=sl,
                take_profit=tp,
                trailing_stop_distance=trailing_dist,
                trailing_activate_distance=trailing_activate,
                strategy_name=signal.strategy_name,
            )

            fill = self._simulator.execute_order(order, candle)
            if fill:
                self._orders_placed += 1
                self._daily_trades += 1

            # Progress logging every 10%
            if i % max(total // 10, 1) == 0:
                pct = i / total * 100
                log.info(
                    "backtest_progress",
                    pct=f"{pct:.0f}%",
                    signals=self._signals_generated,
                    trades=self._orders_placed,
                    equity=f"{self._simulator.equity:,.2f}",
                )

        # Force close remaining positions at last candle
        if candles:
            last = candles[-1]
            remaining = self._simulator.force_close_all(last, reason="backtest_end")
            for ce in remaining:
                self._on_close(ce)

        report = calculate_metrics(
            self._simulator.closes,
            self._simulator.equity_curve,
            self._simulator._initial_equity,
        )

        log.info(
            "backtest_complete",
            trades=report.total_trades,
            pnl=f"{report.total_pnl:,.2f}",
            win_rate=f"{report.win_rate:.1%}",
            sharpe=f"{report.sharpe_ratio:.3f}",
            max_dd=f"{report.max_drawdown_pct:.2%}",
        )

        return report

    def _rate_lookup(self, instrument: str) -> float | None:
        """Look up price from conversion rate cache for currency conversion."""
        if self._conversion_cache is not None:
            return self._conversion_cache.lookup(instrument)
        return self._conversion_rates.get(instrument)

    def _on_close(self, event) -> None:
        self._daily_pnl += event.pnl
        if event.pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _pass_risk_checks(self, signal: SignalEvent, candle: CandleEvent) -> bool:
        """Apply risk management checks."""
        # Max open positions
        if len(self._simulator.open_positions) >= self._max_open_positions:
            return False

        # Already have position on this instrument
        for pos in self._simulator.open_positions.values():
            if pos.instrument == signal.instrument:
                return False

        # Max trades per day
        if self._daily_trades >= self._max_trades_per_day:
            return False

        # Daily loss limit
        equity = self._simulator.equity
        initial = self._simulator._initial_equity
        if self._daily_pnl < 0 and abs(self._daily_pnl) / initial >= self._max_daily_loss:
            self._halted_today = True
            return False

        # Consecutive losses cooldown
        if self._consecutive_losses >= self._max_consecutive_losses:
            return False

        return True

    def _is_session_active(self, candle: CandleEvent) -> bool:
        """Check if candle falls within active trading sessions."""
        for session_name in self._active_sessions:
            session = self._session_times.get(session_name, {})
            start = session.get("start")
            end = session.get("end")
            if start and end and is_within_session(candle.timestamp, start, end):
                return True
        return False
