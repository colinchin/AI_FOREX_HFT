"""Tests for the backtest simulator execution logic."""

from datetime import datetime, timezone

import pytest

from backtest.simulator import BacktestSimulator
from src.core.events import CandleEvent, Direction, OrderEvent, OrderType


def _candle(close: float, high: float | None = None, low: float | None = None) -> CandleEvent:
    return CandleEvent(
        instrument="EUR_USD",
        timeframe="M5",
        open=close - 0.0002,
        high=high or close + 0.0003,
        low=low or close - 0.0003,
        close=close,
        volume=100,
        timestamp=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )


def _order(direction: Direction = Direction.LONG, units: int = 1000) -> OrderEvent:
    if direction == Direction.LONG:
        return OrderEvent(
            instrument="EUR_USD",
            direction=Direction.LONG,
            units=units,
            order_type=OrderType.MARKET,
            stop_loss=1.0990,
            take_profit=1.1020,
            strategy_name="test",
        )
    return OrderEvent(
        instrument="EUR_USD",
        direction=Direction.SHORT,
        units=units,
        order_type=OrderType.MARKET,
        stop_loss=1.1010,
        take_profit=1.0980,
        strategy_name="test",
    )


class TestBacktestSimulator:
    def test_order_execution(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=1.5, slippage_pips=0.5)
        candle = _candle(1.1000)
        order = _order()

        fill = sim.execute_order(order, candle)
        assert fill is not None
        assert fill.direction == Direction.LONG
        assert fill.units == 1000
        assert fill.fill_price > 1.1000  # Should include spread + slippage for LONG

    def test_stop_loss_hit(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=1.5, slippage_pips=0.5)

        # Open a long position
        entry_candle = _candle(1.1000)
        order = _order(Direction.LONG)
        fill = sim.execute_order(order, entry_candle)
        assert len(sim.open_positions) == 1

        # Candle that hits stop loss (low goes below SL)
        sl_candle = _candle(1.0985, low=1.0985)
        closes = sim.process_candle(sl_candle)
        assert len(closes) == 1
        assert closes[0].reason == "sl_hit"
        assert closes[0].pnl < 0  # Should be a loss

    def test_take_profit_hit(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=1.5, slippage_pips=0.5)

        entry_candle = _candle(1.1000)
        order = _order(Direction.LONG)
        sim.execute_order(order, entry_candle)

        # Candle that hits take profit
        tp_candle = _candle(1.1025, high=1.1025)
        closes = sim.process_candle(tp_candle)
        assert len(closes) == 1
        assert closes[0].reason == "tp_hit"
        assert closes[0].pnl > 0  # Should be a profit

    def test_short_stop_loss(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=1.5, slippage_pips=0.5)

        entry_candle = _candle(1.1000)
        order = _order(Direction.SHORT)
        sim.execute_order(order, entry_candle)

        # Candle that hits short SL (high goes above SL)
        sl_candle = _candle(1.1015, high=1.1015)
        closes = sim.process_candle(sl_candle)
        assert len(closes) == 1
        assert closes[0].reason == "sl_hit"

    def test_equity_tracking(self):
        sim = BacktestSimulator(initial_equity=100000)
        assert sim.equity == 100000
        assert sim.balance == 100000

    def test_force_close_all(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=0, slippage_pips=0)

        entry = _candle(1.1000)
        sim.execute_order(_order(Direction.LONG), entry)
        assert len(sim.open_positions) == 1

        close_candle = _candle(1.1005)
        closed = sim.force_close_all(close_candle, reason="test_close")
        assert len(closed) == 1
        assert len(sim.open_positions) == 0

    def test_multiple_positions(self):
        sim = BacktestSimulator(initial_equity=100000, spread_pips=0, slippage_pips=0)

        c1 = CandleEvent("EUR_USD", "M5", 1.1, 1.101, 1.099, 1.1, 100,
                          datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc))
        c2 = CandleEvent("USD_JPY", "M5", 150.0, 150.1, 149.9, 150.0, 100,
                          datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc))

        o1 = OrderEvent("EUR_USD", Direction.LONG, 1000, OrderType.MARKET, 1.099, 1.102, strategy_name="test")
        o2 = OrderEvent("USD_JPY", Direction.SHORT, 500, OrderType.MARKET, 150.2, 149.8, strategy_name="test")

        sim.execute_order(o1, c1)
        sim.execute_order(o2, c2)

        assert len(sim.open_positions) == 2
