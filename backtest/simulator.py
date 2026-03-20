"""Simulated execution for backtesting — models spread, slippage, and order management."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.events import (
    CandleEvent,
    Direction,
    FillEvent,
    OrderEvent,
    TradeCloseEvent,
)
from src.utils.helpers import pip_value, utc_now
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SimulatedPosition:
    """A position in the simulated broker."""
    trade_id: str
    instrument: str
    direction: Direction
    units: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: any  # datetime
    strategy_name: str
    trailing_stop_distance: float | None = None
    trailing_activate_distance: float | None = None
    trailing_stop_price: float | None = None
    highest_favorable: float = 0.0


class BacktestSimulator:
    """Simulated broker for backtesting.

    Models:
    - Configurable spread (fixed or per-instrument)
    - Configurable slippage (fixed + random component)
    - Stop loss / take profit execution on each candle
    - Trailing stop management
    - Position tracking and P&L calculation
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        commission_per_unit: float = 0.0,
    ) -> None:
        self._initial_equity = initial_equity
        self._equity = initial_equity
        self._balance = initial_equity
        self._spread_pips = spread_pips
        self._slippage_pips = slippage_pips
        self._commission = commission_per_unit

        self._positions: dict[str, SimulatedPosition] = {}
        self._trade_counter = 0

        # Results
        self._fills: list[FillEvent] = []
        self._closes: list[TradeCloseEvent] = []
        self._equity_curve: list[tuple] = []  # (timestamp, equity)

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def open_positions(self) -> dict[str, SimulatedPosition]:
        return self._positions

    @property
    def fills(self) -> list[FillEvent]:
        return self._fills

    @property
    def closes(self) -> list[TradeCloseEvent]:
        return self._closes

    @property
    def equity_curve(self) -> list[tuple]:
        return self._equity_curve

    def execute_order(self, order: OrderEvent, current_candle: CandleEvent) -> FillEvent | None:
        """Simulate order execution with spread and slippage."""
        self._trade_counter += 1
        trade_id = f"SIM-{self._trade_counter:06d}"

        pv = pip_value(order.instrument)
        spread_price = self._spread_pips * pv
        slippage_price = self._slippage_pips * pv

        # Fill price with adverse slippage
        if order.direction is Direction.LONG:
            fill_price = current_candle.close + spread_price / 2 + slippage_price
        else:
            fill_price = current_candle.close - spread_price / 2 - slippage_price

        # Commission
        commission = self._commission * order.units

        # Create position
        position = SimulatedPosition(
            trade_id=trade_id,
            instrument=order.instrument,
            direction=order.direction,
            units=order.units,
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            entry_time=current_candle.timestamp,
            strategy_name=order.strategy_name,
            trailing_stop_distance=order.trailing_stop_distance,
            trailing_activate_distance=order.trailing_activate_distance,
        )
        position.highest_favorable = fill_price

        self._positions[trade_id] = position
        self._balance -= commission

        fill = FillEvent(
            instrument=order.instrument,
            direction=order.direction,
            units=order.units,
            fill_price=fill_price,
            trade_id=trade_id,
            timestamp=current_candle.timestamp,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            strategy_name=order.strategy_name,
        )
        self._fills.append(fill)
        return fill

    def process_candle(self, candle: CandleEvent) -> list[TradeCloseEvent]:
        """Check all open positions against this candle for SL/TP hits.

        Returns list of closed positions.
        """
        closed = []
        to_remove = []

        for trade_id, pos in self._positions.items():
            if pos.instrument != candle.instrument:
                continue

            close_event = self._check_exit(pos, candle)
            if close_event:
                closed.append(close_event)
                to_remove.append(trade_id)
            else:
                # Update trailing stop
                self._update_trailing_stop(pos, candle)

        for tid in to_remove:
            del self._positions[tid]

        # Update equity curve
        unrealised = self._calculate_unrealised(candle)
        self._equity = self._balance + unrealised
        self._equity_curve.append((candle.timestamp, self._equity))

        return closed

    def _check_exit(self, pos: SimulatedPosition, candle: CandleEvent) -> TradeCloseEvent | None:
        """Check if a position should be closed on this candle.

        Uses candle direction to infer intra-bar price path and determine
        whether SL or TP was hit first when both are within the candle range:
          - Bullish candle (close > open): assume O → Low → High → C
          - Bearish candle (close < open): assume O → High → Low → C
        This eliminates the SL-first bias that destroys backtest accuracy.
        """
        pv = pip_value(pos.instrument)
        slippage = self._slippage_pips * pv

        if pos.direction is Direction.LONG:
            sl_hit = pos.stop_loss > 0 and candle.low <= pos.stop_loss
            tp_hit = pos.take_profit > 0 and candle.high >= pos.take_profit
            trail_hit = pos.trailing_stop_price is not None and candle.low <= pos.trailing_stop_price

            if sl_hit and tp_hit:
                # Both touched — use candle direction to decide order
                if candle.is_bullish:
                    # O→L→H→C: SL (low) hit first
                    close_price = pos.stop_loss - slippage
                    return self._close_position(pos, close_price, candle, "sl_hit")
                else:
                    # O→H→L→C: TP (high) hit first
                    close_price = pos.take_profit
                    return self._close_position(pos, close_price, candle, "tp_hit")

            if trail_hit and tp_hit:
                if candle.is_bullish:
                    close_price = pos.trailing_stop_price - slippage
                    return self._close_position(pos, close_price, candle, "trailing_sl_hit")
                else:
                    close_price = pos.take_profit
                    return self._close_position(pos, close_price, candle, "tp_hit")

            if sl_hit:
                close_price = pos.stop_loss - slippage
                return self._close_position(pos, close_price, candle, "sl_hit")
            if trail_hit:
                close_price = pos.trailing_stop_price - slippage
                return self._close_position(pos, close_price, candle, "trailing_sl_hit")
            if tp_hit:
                close_price = pos.take_profit
                return self._close_position(pos, close_price, candle, "tp_hit")

        else:  # SHORT
            sl_hit = pos.stop_loss > 0 and candle.high >= pos.stop_loss
            tp_hit = pos.take_profit > 0 and candle.low <= pos.take_profit
            trail_hit = pos.trailing_stop_price is not None and candle.high >= pos.trailing_stop_price

            if sl_hit and tp_hit:
                if not candle.is_bullish:
                    # O→H→L→C: SL (high) hit first
                    close_price = pos.stop_loss + slippage
                    return self._close_position(pos, close_price, candle, "sl_hit")
                else:
                    # O→L→H→C: TP (low) hit first
                    close_price = pos.take_profit
                    return self._close_position(pos, close_price, candle, "tp_hit")

            if trail_hit and tp_hit:
                if not candle.is_bullish:
                    close_price = pos.trailing_stop_price + slippage
                    return self._close_position(pos, close_price, candle, "trailing_sl_hit")
                else:
                    close_price = pos.take_profit
                    return self._close_position(pos, close_price, candle, "tp_hit")

            if sl_hit:
                close_price = pos.stop_loss + slippage
                return self._close_position(pos, close_price, candle, "sl_hit")
            if trail_hit:
                close_price = pos.trailing_stop_price + slippage
                return self._close_position(pos, close_price, candle, "trailing_sl_hit")
            if tp_hit:
                close_price = pos.take_profit
                return self._close_position(pos, close_price, candle, "tp_hit")

        return None

    def _update_trailing_stop(self, pos: SimulatedPosition, candle: CandleEvent) -> None:
        """Update trailing stop based on favorable price movement."""
        if not pos.trailing_stop_distance:
            return

        # Use separate activation threshold if provided, else fall back to distance
        activate_dist = pos.trailing_activate_distance or pos.trailing_stop_distance

        if pos.direction is Direction.LONG:
            if candle.high > pos.highest_favorable:
                pos.highest_favorable = candle.high
                new_trail = pos.highest_favorable - pos.trailing_stop_distance
                if pos.trailing_stop_price is None or new_trail > pos.trailing_stop_price:
                    profit = pos.highest_favorable - pos.entry_price
                    if profit >= activate_dist:
                        pos.trailing_stop_price = new_trail
        else:
            if candle.low < pos.highest_favorable or pos.highest_favorable == pos.entry_price:
                if pos.highest_favorable == pos.entry_price:
                    pos.highest_favorable = candle.low
                pos.highest_favorable = min(pos.highest_favorable, candle.low)
                new_trail = pos.highest_favorable + pos.trailing_stop_distance
                if pos.trailing_stop_price is None or new_trail < pos.trailing_stop_price:
                    profit = pos.entry_price - pos.highest_favorable
                    if profit >= activate_dist:
                        pos.trailing_stop_price = new_trail

    def _close_position(
        self, pos: SimulatedPosition, close_price: float,
        candle: CandleEvent, reason: str,
    ) -> TradeCloseEvent:
        """Close a position and update balance."""
        if pos.direction is Direction.LONG:
            pnl = (close_price - pos.entry_price) * pos.units
        else:
            pnl = (pos.entry_price - close_price) * pos.units

        self._balance += pnl

        event = TradeCloseEvent(
            instrument=pos.instrument,
            trade_id=pos.trade_id,
            close_price=close_price,
            pnl=pnl,
            timestamp=candle.timestamp,
            reason=reason,
        )
        self._closes.append(event)
        return event

    def _calculate_unrealised(self, candle: CandleEvent) -> float:
        """Calculate total unrealised P&L."""
        total = 0.0
        for pos in self._positions.values():
            if pos.instrument == candle.instrument:
                if pos.direction is Direction.LONG:
                    total += (candle.close - pos.entry_price) * pos.units
                else:
                    total += (pos.entry_price - candle.close) * pos.units
        return total

    def force_close_all(self, candle: CandleEvent, reason: str = "time_exit") -> list[TradeCloseEvent]:
        """Force close all positions at current candle close."""
        closed = []
        for trade_id in list(self._positions.keys()):
            pos = self._positions.pop(trade_id)
            if pos.instrument == candle.instrument:
                event = self._close_position(pos, candle.close, candle, reason)
                closed.append(event)
            else:
                self._positions[trade_id] = pos
        return closed
