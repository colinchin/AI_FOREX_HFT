"""Transaction poller — detects broker-side SL/TP/trailing stop closures.

Without this, when OANDA closes a trade via server-side SL/TP, the system's
RiskManager and PortfolioTracker remain stale (thinking positions are open).
This poller reconciles by polling OANDA's transaction history and emitting
TradeCloseEvent for any detected closures.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.core.events import TradeCloseEvent, TradeReducedEvent
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.api.client import OANDAClient
    from src.core.bus import EventBus

log = get_logger(__name__)

# Transaction types that indicate a trade was closed by the broker
_CLOSE_REASONS = {
    "STOP_LOSS_ORDER": "sl_hit",
    "TAKE_PROFIT_ORDER": "tp_hit",
    "TRAILING_STOP_LOSS_ORDER": "trailing_sl_hit",
    "MARKET_ORDER_TRADE_CLOSE": "manual",
    "LINKED_TRADE_CLOSED": "linked_close",
}


class TransactionPoller:
    """Polls OANDA transaction history to detect broker-side trade closures.

    Runs as a background task, checking for new transactions at a
    configurable interval. Emits TradeCloseEvent for any ORDER_FILL
    transactions that close trades (SL/TP/trailing stop fills).
    """

    def __init__(
        self,
        client: OANDAClient,
        event_bus: EventBus,
        poll_interval: float = 5.0,
    ) -> None:
        self._client = client
        self._bus = event_bus
        self._poll_interval = poll_interval
        self._last_tx_id: int = 0
        self._running = False
        self._task: asyncio.Task | None = None
        self._known_trade_ids: set[str] = set()
        self._emitted_tx_ids: set[str] = set()  # dedup: don't re-emit for same tx
        self._polls: int = 0

    @property
    def stats(self) -> dict:
        return {
            "polls": self._polls,
            "last_tx_id": str(self._last_tx_id),
            "running": self._running,
        }

    async def start(self) -> None:
        """Start polling for transactions."""
        if self._running:
            return
        # Get the current latest transaction ID as our starting point
        self._last_tx_id = int(await self._client.get_latest_transaction_id())

        # Seed known trades from currently open positions (survives restarts)
        try:
            open_trades = await self._client.get_open_trades()
            for trade in open_trades:
                tid = trade.get("id", "")
                if tid:
                    self._known_trade_ids.add(tid)
            if open_trades:
                log.info("poller_seeded_open_trades", count=len(open_trades),
                         trade_ids=sorted(self._known_trade_ids))
        except Exception as e:
            log.warning("poller_seed_failed", error=str(e))

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("transaction_poller_started", since_id=self._last_tx_id)

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("transaction_poller_stopped", polls=self._polls)

    def register_trade(self, trade_id: str) -> None:
        """Register a trade ID that we opened (to track it)."""
        self._known_trade_ids.add(trade_id)

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._check_transactions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("transaction_poll_error", error=str(e))
                await asyncio.sleep(self._poll_interval * 2)

    async def _check_transactions(self) -> None:
        """Fetch new transactions and process any trade closures."""
        self._polls += 1
        try:
            txns = await self._client.get_transactions_since(str(self._last_tx_id))
        except Exception as e:
            log.debug("transaction_fetch_failed", error=str(e))
            return

        if not txns:
            return

        for tx in txns:
            tx_id_str = tx.get("id", "0")
            tx_type = tx.get("type", "")

            # Update watermark using integer comparison
            try:
                tx_id_int = int(tx_id_str)
            except (ValueError, TypeError):
                continue
            if tx_id_int > self._last_tx_id:
                self._last_tx_id = tx_id_int

            # Only interested in ORDER_FILL that closes a trade
            if tx_type != "ORDER_FILL":
                continue

            # Dedup: skip if we already processed this transaction
            if tx_id_str in self._emitted_tx_ids:
                continue

            # Check if this fill closed a trade
            trades_closed = tx.get("tradesClosed", [])
            trade_reduced = tx.get("tradeReduced")

            # Determine the reason from the order's type/reason
            reason_key = tx.get("reason", "")
            close_reason = _CLOSE_REASONS.get(reason_key, "broker_close")

            for closed in trades_closed:
                trade_id = closed.get("tradeID", "")

                # Only process trades we opened (ignore external trades)
                if trade_id not in self._known_trade_ids:
                    log.debug("poller_ignoring_unknown_trade", trade_id=trade_id)
                    continue

                pnl = float(closed.get("realizedPL", 0))

                await self._emit_close(
                    tx=tx,
                    trade_id=trade_id,
                    pnl=pnl,
                    reason=close_reason,
                )
                self._emitted_tx_ids.add(tx_id_str)

            if trade_reduced:
                trade_id = trade_reduced.get("tradeID", "")
                if trade_id in self._known_trade_ids:
                    pnl = float(trade_reduced.get("realizedPL", 0))
                    remaining = abs(int(trade_reduced.get("currentUnits", 0)))
                    units_reduced = abs(int(trade_reduced.get("units", 0)))
                    instrument = tx.get("instrument", "")

                    if remaining == 0:
                        # Fully reduced — emit only TradeCloseEvent (not both,
                        # which would double-count PnL).
                        log.info(
                            "trade_fully_reduced_to_close",
                            trade_id=trade_id,
                            pnl=pnl,
                            reason=close_reason,
                        )
                        await self._emit_close(
                            tx=tx,
                            trade_id=trade_id,
                            pnl=pnl,
                            reason=close_reason,
                        )
                        self._emitted_tx_ids.add(tx_id_str)
                        self._known_trade_ids.discard(trade_id)
                    else:
                        # Genuine partial — emit TradeReducedEvent
                        log.info(
                            "trade_partially_closed",
                            trade_id=trade_id,
                            units_reduced=units_reduced,
                            remaining=remaining,
                            pnl=pnl,
                            reason=close_reason,
                        )
                        event = TradeReducedEvent(
                            instrument=instrument,
                            trade_id=trade_id,
                            units_reduced=units_reduced,
                            remaining_units=remaining,
                            pnl=pnl,
                            reason=close_reason,
                        )
                        await self._bus.publish(event)
                        self._emitted_tx_ids.add(tx_id_str)

    async def _emit_close(
        self, tx: dict, trade_id: str, pnl: float, reason: str,
    ) -> None:
        """Emit a TradeCloseEvent for a detected closure."""
        instrument = tx.get("instrument", "")
        price = float(tx.get("price", 0))

        # Parse timestamp
        time_str = tx.get("time", "")
        try:
            cleaned = time_str.rstrip("Z").split(".")[0]
            ts = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc,
            )
        except (ValueError, AttributeError):
            from src.utils.helpers import utc_now
            ts = utc_now()

        event = TradeCloseEvent(
            instrument=instrument,
            trade_id=trade_id,
            close_price=price,
            pnl=pnl,
            timestamp=ts,
            reason=reason,
        )

        log.info(
            "broker_trade_closed",
            trade_id=trade_id,
            instrument=instrument,
            pnl=pnl,
            reason=reason,
        )

        await self._bus.publish(event)
        self._known_trade_ids.discard(trade_id)
