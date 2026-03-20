"""Account state manager — polls OANDA for account updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.helpers import utc_now
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.api.client import OANDAClient

log = get_logger(__name__)


class AccountManager:
    """Tracks account state: balance, equity, margin, open trades."""

    def __init__(self, client: OANDAClient) -> None:
        self._client = client
        self._balance: float = 0.0
        self._equity: float = 0.0  # balance + unrealised P&L
        self._unrealised_pnl: float = 0.0
        self._margin_used: float = 0.0
        self._margin_available: float = 0.0
        self._open_trade_count: int = 0
        self._currency: str = "USD"
        self._last_transaction_id: str = ""
        self._last_update = utc_now()

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def unrealised_pnl(self) -> float:
        return self._unrealised_pnl

    @property
    def margin_used(self) -> float:
        return self._margin_used

    @property
    def margin_available(self) -> float:
        return self._margin_available

    @property
    def open_trade_count(self) -> int:
        return self._open_trade_count

    @property
    def currency(self) -> str:
        return self._currency

    async def get_equity(self) -> float:
        """Return current equity (for position sizing)."""
        return self._equity

    async def initialize(self) -> None:
        """Fetch initial account snapshot."""
        summary = await self._client.get_account_summary()
        account = summary.get("account", {})
        self._update_from_account(account)
        log.info(
            "account_initialized",
            balance=self._balance,
            equity=self._equity,
            currency=self._currency,
            open_trades=self._open_trade_count,
        )

    async def refresh(self) -> bool:
        """Poll for account state updates.

        Returns:
            True if refresh succeeded, False on failure.
        """
        try:
            summary = await self._client.get_account_summary()
            account = summary.get("account", {})
            self._update_from_account(account)
            self._last_update = utc_now()
            return True
        except Exception as e:
            log.warning("account_refresh_failed", error=str(e))
            return False

    def _update_from_account(self, account: dict) -> None:
        self._balance = float(account.get("balance", 0))
        self._unrealised_pnl = float(account.get("unrealizedPL", 0))
        self._equity = self._balance + self._unrealised_pnl
        self._margin_used = float(account.get("marginUsed", 0))
        self._margin_available = float(account.get("marginAvailable", 0))
        self._open_trade_count = int(account.get("openTradeCount", 0))
        self._currency = account.get("currency", "USD")
        tx_id = account.get("lastTransactionID", "")
        if tx_id:
            self._last_transaction_id = tx_id

    @property
    def summary(self) -> dict:
        return {
            "balance": self._balance,
            "equity": self._equity,
            "unrealised_pnl": self._unrealised_pnl,
            "margin_used": self._margin_used,
            "margin_available": self._margin_available,
            "open_trades": self._open_trade_count,
            "currency": self._currency,
            "last_update": self._last_update.isoformat(),
        }
