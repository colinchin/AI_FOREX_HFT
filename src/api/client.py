"""OANDA API client wrapper — auth, rate limiting, exponential backoff."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments_ep
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as transactions

from src.utils.config import OANDAConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

# OANDA rate limits: ~30 requests/second for REST
_MAX_REQUESTS_PER_SECOND = 25  # conservative
_MIN_REQUEST_INTERVAL = 1.0 / _MAX_REQUESTS_PER_SECOND
_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5  # seconds


class OANDAClient:
    """Async-friendly OANDA v20 API client with rate limiting and retry."""

    def __init__(self, config: OANDAConfig) -> None:
        self._config = config
        self._api = oandapyV20.API(
            access_token=config.access_token,
            environment="practice" if config.environment == "practice" else "live",
        )
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()
        self._consecutive_errors: int = 0

    @property
    def api(self) -> oandapyV20.API:
        return self._api

    @property
    def account_id(self) -> str:
        return self._config.account_id

    @property
    def stream_url(self) -> str:
        return self._config.stream_url

    @property
    def access_token(self) -> str:
        return self._config.access_token

    async def _rate_limit(self) -> None:
        """Enforce minimum interval between API requests."""
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.monotonic()

    async def request(self, endpoint: Any, max_retries: int = _MAX_RETRIES) -> dict:
        """Execute an OANDA API request with rate limiting and exponential backoff.

        Runs the synchronous oandapyV20 call in a thread executor to avoid
        blocking the asyncio event loop.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries):
            await self._rate_limit()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._api.request, endpoint
                )
                self._consecutive_errors = 0
                return result
            except oandapyV20.exceptions.V20Error as e:
                last_error = e
                status = getattr(e, "code", 0)

                # Don't retry client errors (4xx) except 429 (rate limit)
                if 400 <= status < 500 and status != 429:
                    log.error(
                        "oanda_client_error",
                        status=status,
                        message=str(e),
                        endpoint=str(endpoint),
                    )
                    raise

                self._consecutive_errors += 1
                backoff = _BASE_BACKOFF * (2 ** attempt)
                log.warning(
                    "oanda_request_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    backoff=backoff,
                    status=status,
                    error=str(e),
                )
                await asyncio.sleep(backoff)

            except Exception as e:
                last_error = e
                self._consecutive_errors += 1
                backoff = _BASE_BACKOFF * (2 ** attempt)
                log.warning(
                    "oanda_request_error",
                    attempt=attempt + 1,
                    error=str(e),
                    backoff=backoff,
                )
                await asyncio.sleep(backoff)

        log.error("oanda_request_exhausted", retries=max_retries, error=str(last_error))
        raise last_error  # type: ignore[misc]

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    # ── Convenience methods ─────────────────────────────────────

    async def get_account_summary(self) -> dict:
        ep = accounts.AccountSummary(self.account_id)
        return await self.request(ep)

    async def get_account_details(self) -> dict:
        ep = accounts.AccountDetails(self.account_id)
        return await self.request(ep)

    async def get_instruments(self) -> list[dict]:
        ep = accounts.AccountInstruments(self.account_id)
        result = await self.request(ep)
        return result.get("instruments", [])

    async def get_pricing(self, instrument_list: list[str]) -> dict:
        params = {"instruments": ",".join(instrument_list)}
        ep = pricing.PricingInfo(self.account_id, params=params)
        return await self.request(ep)

    async def get_candles(
        self,
        instrument: str,
        granularity: str = "M5",
        count: int | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
        price: str = "MBA",
    ) -> list[dict]:
        params: dict[str, Any] = {"granularity": granularity, "price": price}
        # OANDA rule: cannot combine 'count' with BOTH 'from' AND 'to'.
        # Allowed combos: count, from+count, from+to, to+count
        has_from = from_time is not None
        has_to = to_time is not None
        has_count = count is not None

        if has_from:
            params["from"] = from_time
        if has_to:
            params["to"] = to_time
        if has_count and not (has_from and has_to):
            params["count"] = min(count, 5000)

        ep = instruments_ep.InstrumentsCandles(instrument, params=params)
        result = await self.request(ep)
        return result.get("candles", [])

    async def place_order(self, order_data: dict) -> dict:
        ep = orders.OrderCreate(self.account_id, data=order_data)
        return await self.request(ep)

    async def get_open_trades(self) -> list[dict]:
        ep = trades.OpenTrades(self.account_id)
        result = await self.request(ep)
        return result.get("trades", [])

    async def close_trade(self, trade_id: str, units: str | None = None) -> dict:
        data: dict[str, Any] = {}
        if units is not None:
            data["units"] = units
        ep = trades.TradeClose(self.account_id, trade_id, data=data)
        return await self.request(ep)

    async def modify_trade(self, trade_id: str, data: dict) -> dict:
        ep = trades.TradeCRCDO(self.account_id, trade_id, data=data)
        return await self.request(ep)

    async def get_transactions_since(self, since_id: str) -> list[dict]:
        """Get all transactions since a given transaction ID."""
        params = {"id": since_id}
        ep = transactions.TransactionsSinceID(self.account_id, params=params)
        result = await self.request(ep)
        return result.get("transactions", [])

    async def get_latest_transaction_id(self) -> str:
        """Get the most recent transaction ID for this account."""
        summary = await self.get_account_summary()
        return str(summary.get("account", {}).get("lastTransactionID", "0"))

    async def health_check(self) -> dict[str, Any]:
        """Verify connectivity and return account info + latency."""
        start = time.monotonic()
        summary = await self.get_account_summary()
        latency_ms = (time.monotonic() - start) * 1000

        account = summary.get("account", {})
        return {
            "connected": True,
            "latency_ms": round(latency_ms, 1),
            "account_id": account.get("id"),
            "balance": account.get("balance"),
            "currency": account.get("currency"),
            "open_trade_count": account.get("openTradeCount"),
            "environment": self._config.environment,
        }
