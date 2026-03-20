"""Order execution handler — places, modifies, and closes trades via OANDA.

Uses atomic stopLossOnFill and takeProfitOnFill for guaranteed risk management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.events import (
    Direction,
    ErrorEvent,
    FillEvent,
    OrderEvent,
    TradeCloseEvent,
)
from src.utils.helpers import format_price, utc_now
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.api.client import OANDAClient
    from src.core.bus import EventBus

log = get_logger(__name__)


class ExecutionHandler:
    """Handles order placement and trade management via OANDA API."""

    def __init__(self, client: OANDAClient, event_bus: EventBus) -> None:
        self._client = client
        self._bus = event_bus

    async def on_order(self, order: OrderEvent) -> None:
        """Execute an order against OANDA."""
        try:
            order_data = self._build_order_data(order)
            result = await self._client.place_order(order_data)
            await self._process_order_result(order, result)
        except Exception as e:
            log.error("order_execution_failed", instrument=order.instrument, error=str(e))
            await self._bus.publish(ErrorEvent(
                source="execution",
                message=f"Order failed: {e}",
                critical=False,
            ))

    def _build_order_data(self, order: OrderEvent) -> dict:
        """Construct OANDA order request body."""
        units = order.units * order.direction.sign
        fmt = lambda p: format_price(p, order.instrument)

        order_body: dict = {
            "type": "MARKET",
            "instrument": order.instrument,
            "units": str(units),
            "timeInForce": "FOK",  # Fill or Kill
            "positionFill": "DEFAULT",
        }

        # Atomic stop loss (guaranteed with order)
        if order.stop_loss:
            order_body["stopLossOnFill"] = {
                "price": fmt(order.stop_loss),
                "timeInForce": "GTC",
            }

        # Atomic take profit
        if order.take_profit:
            order_body["takeProfitOnFill"] = {
                "price": fmt(order.take_profit),
                "timeInForce": "GTC",
            }

        # Price bound to prevent adverse slippage
        if order.price_bound:
            order_body["priceBound"] = fmt(order.price_bound)

        # Trailing stop (set separately after fill)
        # OANDA doesn't support trailingStopLossOnFill with MARKET orders in all cases
        # We'll manage trailing stops post-fill

        # Tag trade with strategy metadata so it survives restarts.
        # OANDA tradeClientExtensions.tag is max 128 chars.
        import json
        tag_data: dict = {"s": order.strategy_name}
        if order.trailing_stop_distance:
            tag_data["td"] = order.trailing_stop_distance
        if order.trailing_activate_distance:
            tag_data["ta"] = order.trailing_activate_distance
        order_body["tradeClientExtensions"] = {
            "tag": json.dumps(tag_data, separators=(",", ":")),
        }

        return {"order": order_body}

    async def _process_order_result(self, order: OrderEvent, result: dict) -> None:
        """Parse OANDA order response and emit FillEvent."""
        # Check for fill
        fill_tx = result.get("orderFillTransaction")
        if fill_tx:
            trade_ids = fill_tx.get("tradeOpened", {}).get("tradeID") or fill_tx.get("id", "")
            fill_price = float(fill_tx.get("price", 0))

            fill = FillEvent(
                instrument=order.instrument,
                direction=order.direction,
                units=order.units,
                fill_price=fill_price,
                trade_id=str(trade_ids),
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                strategy_name=order.strategy_name,
                trailing_stop_distance=order.trailing_stop_distance,
                trailing_activate_distance=order.trailing_activate_distance,
            )

            log.info(
                "order_filled",
                instrument=fill.instrument,
                direction=fill.direction.name,
                units=fill.units,
                price=fill.fill_price,
                trade_id=fill.trade_id,
            )

            await self._bus.publish(fill)

            # Trailing stop activation is deferred: if an activate threshold
            # is set, PortfolioTracker arms the trailing stop once the trade
            # reaches that profit level.  Only set immediately when there is
            # a trailing distance but NO activation threshold (trail from entry).
            if order.trailing_stop_distance and not order.trailing_activate_distance:
                await self._set_trailing_stop(
                    trade_id=fill.trade_id,
                    distance=order.trailing_stop_distance,
                    instrument=order.instrument,
                )
            return

        # Check for rejection
        cancel_tx = result.get("orderCancelTransaction")
        if cancel_tx:
            reason = cancel_tx.get("reason", "unknown")
            log.warning("order_rejected", instrument=order.instrument, reason=reason)
            await self._bus.publish(ErrorEvent(
                source="execution",
                message=f"Order rejected: {reason}",
            ))
            return

        log.warning("order_unknown_response", result=str(result)[:500])

    async def _set_trailing_stop(
        self, trade_id: str, distance: float, instrument: str = "",
    ) -> None:
        """Set a trailing stop loss on an open trade.

        Raises on failure so callers (deferred activation in PortfolioTracker)
        can detect the error and retry on the next tick.
        """
        dist_str = format_price(distance, instrument) if instrument else f"{distance:.5f}"
        data = {
            "trailingStopLoss": {
                "distance": dist_str,
                "timeInForce": "GTC",
            }
        }
        await self._client.modify_trade(trade_id, data)
        log.info("trailing_stop_set", trade_id=trade_id, distance=distance)

    async def _close_trade_raw(self, trade_id: str, reason: str = "manual") -> None:
        """Close a specific trade — raises on failure (used by retry logic)."""
        result = await self._client.close_trade(trade_id)
        close_tx = result.get("orderFillTransaction", {})
        pnl = float(close_tx.get("pl", 0))
        instrument = close_tx.get("instrument", "")
        price = float(close_tx.get("price", 0))

        event = TradeCloseEvent(
            instrument=instrument,
            trade_id=trade_id,
            close_price=price,
            pnl=pnl,
            reason=reason,
        )
        await self._bus.publish(event)

        log.info(
            "trade_closed",
            trade_id=trade_id,
            instrument=instrument,
            pnl=pnl,
            reason=reason,
        )

    async def close_trade(self, trade_id: str, reason: str = "manual") -> None:
        """Close a specific trade by ID. Swallows exceptions (use _close_trade_raw for retry)."""
        try:
            await self._close_trade_raw(trade_id, reason)
        except Exception as e:
            log.error("trade_close_failed", trade_id=trade_id, error=str(e))

    async def close_all_positions(
        self, reason: str = "circuit_breaker", max_retries: int = 3, retry_delay: float = 5.0,
    ) -> bool:
        """Emergency: close all open positions with retry.

        Retries up to max_retries times if any trades fail to close.
        Returns True only if zero positions remain open after all attempts.
        """
        log.warning("closing_all_positions", reason=reason)

        for attempt in range(1, max_retries + 1):
            try:
                trades = await self._client.get_open_trades()
                if not trades:
                    log.info("close_all_complete", attempt=attempt)
                    return True

                failed = []
                for trade in trades:
                    trade_id = trade.get("id", "")
                    if trade_id:
                        try:
                            await self._close_trade_raw(trade_id, reason=reason)
                        except Exception as e:
                            log.warning("close_trade_retry_failed", trade_id=trade_id, attempt=attempt, error=str(e))
                            failed.append(trade_id)

                if not failed:
                    log.info("close_all_complete", attempt=attempt)
                    return True

                log.warning("close_all_partial", attempt=attempt, failed=failed, remaining=len(failed))

            except Exception as e:
                log.error("close_all_failed", attempt=attempt, error=str(e))

            if attempt < max_retries:
                import asyncio
                await asyncio.sleep(retry_delay)

        # Final check — did we actually flatten?
        try:
            remaining = await self._client.get_open_trades()
            if not remaining:
                return True
            log.error("close_all_exhausted", remaining=len(remaining), trade_ids=[t.get("id") for t in remaining])
            return False
        except Exception:
            return False
