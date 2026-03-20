"""Portfolio tracker — manages open positions, calculates P&L, tracks performance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.events import Direction, FillEvent, TickEvent, TradeCloseEvent, TradeReducedEvent
from src.core.models import ClosedTrade, Position
from src.data.store import TradeStore
from src.utils.helpers import utc_now
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.api.execution import ExecutionHandler
    from src.core.bus import EventBus

log = get_logger(__name__)


class PortfolioTracker:
    """Tracks open positions, manages trailing stops, and records trade history."""

    def __init__(
        self,
        event_bus: EventBus,
        trade_store: TradeStore,
        execution: ExecutionHandler | None = None,
    ) -> None:
        self._bus = event_bus
        self._store = trade_store
        self._execution = execution  # needed for deferred trailing-stop activation
        self._open_positions: dict[str, Position] = {}  # trade_id → Position
        self._closed_trades: list[ClosedTrade] = []

        # Running stats
        self._total_trades: int = 0
        self._total_wins: int = 0
        self._total_pnl: float = 0.0
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0
        self._max_drawdown: float = 0.0
        self._peak_pnl: float = 0.0

    @property
    def open_positions(self) -> dict[str, Position]:
        return self._open_positions

    @property
    def open_count(self) -> int:
        return len(self._open_positions)

    def reconcile_open_trades(self, broker_trades: list[dict]) -> None:
        """Seed open positions from broker state on startup.

        Creates Position objects for each open trade so that
        portfolio tracking, P&L updates, and time-exit checks work
        correctly after a restart.

        Restores strategy name and deferred trailing-stop config from
        the tradeClientExtensions.tag set at order time.
        """
        import json
        from src.utils.helpers import parse_oanda_time

        for trade in broker_trades:
            trade_id = trade.get("id", "")
            if not trade_id or trade_id in self._open_positions:
                continue

            instrument = trade.get("instrument", "")
            units = int(trade.get("currentUnits", trade.get("initialUnits", 0)))
            direction = Direction.LONG if units > 0 else Direction.SHORT
            entry_price = float(trade.get("price", 0))
            sl = float(trade.get("stopLossOrder", {}).get("price", 0)) if trade.get("stopLossOrder") else 0.0
            tp = float(trade.get("takeProfitOrder", {}).get("price", 0)) if trade.get("takeProfitOrder") else 0.0
            open_time = parse_oanda_time(trade.get("openTime", "2000-01-01T00:00:00Z"))

            # Parse strategy metadata from client extensions tag
            strategy_name = "reconciled"
            trailing_activate = None
            tag_trailing_distance = None
            try:
                tag_str = trade.get("clientExtensions", {}).get("tag", "")
                if tag_str:
                    tag = json.loads(tag_str)
                    strategy_name = tag.get("s", "reconciled")
                    tag_trailing_distance = tag.get("td")
                    trailing_activate = tag.get("ta")
            except (json.JSONDecodeError, AttributeError):
                pass

            # Restore trailing-stop state from broker
            tsl_order = trade.get("trailingStopLossOrder")
            broker_trailing_distance = float(tsl_order.get("distance", 0)) if tsl_order else None
            trailing_armed = broker_trailing_distance is not None and broker_trailing_distance > 0

            # Use broker distance if already armed, otherwise restore from tag
            trailing_distance = (
                broker_trailing_distance if trailing_armed
                else tag_trailing_distance
            )

            position = Position(
                trade_id=trade_id,
                instrument=instrument,
                direction=direction,
                units=abs(units),
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                entry_time=open_time,
                strategy_name=strategy_name,
                trailing_stop_distance=trailing_distance,
                trailing_activate_distance=trailing_activate,
                trailing_armed=trailing_armed,
            )
            self._open_positions[trade_id] = position

        if broker_trades:
            log.info(
                "portfolio_reconciled",
                open_positions=len(self._open_positions),
                trade_ids=sorted(self._open_positions.keys()),
            )

    async def on_fill(self, fill: FillEvent) -> None:
        """Register a new position from a fill."""
        position = Position(
            trade_id=fill.trade_id,
            instrument=fill.instrument,
            direction=fill.direction,
            units=fill.units,
            entry_price=fill.fill_price,
            stop_loss=fill.stop_loss,
            take_profit=fill.take_profit,
            entry_time=fill.timestamp,
            strategy_name=fill.strategy_name,
            trailing_stop_distance=fill.trailing_stop_distance,
            trailing_activate_distance=fill.trailing_activate_distance,
        )
        self._open_positions[fill.trade_id] = position

        log.info(
            "portfolio_position_opened",
            trade_id=fill.trade_id,
            instrument=fill.instrument,
            direction=fill.direction.name,
            units=fill.units,
            entry=fill.fill_price,
        )

    async def on_tick(self, tick: TickEvent) -> None:
        """Update unrealised P&L and check deferred trailing-stop activation."""
        price = tick.mid
        for pos in self._open_positions.values():
            if pos.instrument != tick.instrument:
                continue
            pos.update_pnl(price)

            # Deferred trailing-stop: arm once profit exceeds activation threshold
            if (
                pos.trailing_stop_distance
                and pos.trailing_activate_distance
                and not pos.trailing_armed
                and self._execution is not None
            ):
                # Compute current price distance from entry in the profitable direction
                if pos.direction is Direction.LONG:
                    profit_distance = price - pos.entry_price
                else:
                    profit_distance = pos.entry_price - price

                if profit_distance >= pos.trailing_activate_distance:
                    pos.trailing_armed = True
                    try:
                        await self._execution._set_trailing_stop(
                            trade_id=pos.trade_id,
                            distance=pos.trailing_stop_distance,
                            instrument=pos.instrument,
                        )
                        log.info(
                            "trailing_stop_activated",
                            trade_id=pos.trade_id,
                            instrument=pos.instrument,
                            profit_distance=round(profit_distance, 6),
                            activate_threshold=pos.trailing_activate_distance,
                        )
                    except Exception as e:
                        pos.trailing_armed = False  # retry on next tick
                        log.warning("trailing_stop_activate_failed",
                                    trade_id=pos.trade_id, error=str(e))

    async def on_trade_close(self, event: TradeCloseEvent) -> None:
        """Record a closed trade and update statistics.

        Idempotent: if trade_id is not in open positions (already closed
        by another path e.g. execution vs poller), skip to avoid double-counting.
        """
        position = self._open_positions.pop(event.trade_id, None)
        if position is None:
            log.debug("trade_close_duplicate_ignored", trade_id=event.trade_id)
            return

        closed = ClosedTrade(
            trade_id=event.trade_id,
            instrument=event.instrument,
            direction=position.direction,
            units=position.units,
            entry_price=position.entry_price,
            exit_price=event.close_price,
            entry_time=position.entry_time,
            exit_time=event.timestamp,
            pnl=event.pnl,
            strategy_name=position.strategy_name,
            exit_reason=event.reason,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
        )

        self._closed_trades.append(closed)
        self._update_stats(closed)

        # Persist to database
        try:
            await self._store.insert_trade(closed)
        except Exception as e:
            log.error("trade_persist_failed", trade_id=event.trade_id, error=str(e))

        log.info(
            "portfolio_trade_closed",
            trade_id=event.trade_id,
            instrument=event.instrument,
            pnl=event.pnl,
            reason=event.reason,
            total_trades=self._total_trades,
            win_rate=f"{self.win_rate:.1%}",
        )

    async def on_trade_reduced(self, event: TradeReducedEvent) -> None:
        """Update position when a trade is partially closed."""
        position = self._open_positions.get(event.trade_id)
        if position is None:
            log.debug("trade_reduced_unknown", trade_id=event.trade_id)
            return

        old_units = position.units
        position.units = event.remaining_units

        # Book partial PnL through the same stats path as full closes
        self._total_trades += 1
        self._total_pnl += event.pnl
        if event.pnl > 0:
            self._total_wins += 1
            self._gross_profit += event.pnl
        else:
            self._gross_loss += event.pnl
        self._peak_pnl = max(self._peak_pnl, self._total_pnl)
        self._max_drawdown = max(self._max_drawdown, self._peak_pnl - self._total_pnl)

        log.info(
            "portfolio_trade_reduced",
            trade_id=event.trade_id,
            instrument=event.instrument,
            old_units=old_units,
            remaining_units=event.remaining_units,
            pnl=event.pnl,
        )

    def _update_stats(self, trade: ClosedTrade) -> None:
        self._total_trades += 1
        self._total_pnl += trade.pnl

        if trade.is_winner:
            self._total_wins += 1
            self._gross_profit += trade.pnl
        else:
            self._gross_loss += trade.pnl

        # Drawdown tracking
        self._peak_pnl = max(self._peak_pnl, self._total_pnl)
        drawdown = self._peak_pnl - self._total_pnl
        self._max_drawdown = max(self._max_drawdown, drawdown)

    @property
    def win_rate(self) -> float:
        return self._total_wins / self._total_trades if self._total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return self._gross_profit / abs(self._gross_loss) if self._gross_loss != 0 else float("inf")

    @property
    def expectancy(self) -> float:
        return self._total_pnl / self._total_trades if self._total_trades > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "total_trades": self._total_trades,
            "wins": self._total_wins,
            "losses": self._total_trades - self._total_wins,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self._total_pnl, 2),
            "gross_profit": round(self._gross_profit, 2),
            "gross_loss": round(self._gross_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 4),
            "max_drawdown": round(self._max_drawdown, 2),
            "open_positions": self.open_count,
        }

    def get_positions_by_instrument(self, instrument: str) -> list[Position]:
        return [p for p in self._open_positions.values() if p.instrument == instrument]

    def has_position(self, instrument: str) -> bool:
        return any(p.instrument == instrument for p in self._open_positions.values())
