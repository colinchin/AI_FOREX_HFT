"""Price stream manager — persistent connection, heartbeat monitoring, auto-reconnect."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp

from src.core.events import TickEvent
from src.utils.helpers import utc_now
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.core.bus import EventBus

log = get_logger(__name__)

_MAX_RECONNECT_DELAY = 60.0
_BASE_RECONNECT_DELAY = 1.0
_HEARTBEAT_TIMEOUT = 30.0  # seconds — matches design doc


class StreamManager:
    """Manages a persistent OANDA price stream with heartbeat monitoring."""

    def __init__(
        self,
        stream_url: str,
        account_id: str,
        access_token: str,
        instruments: list[str],
        event_bus: EventBus,
        heartbeat_timeout: float = _HEARTBEAT_TIMEOUT,
    ) -> None:
        self._stream_url = stream_url
        self._account_id = account_id
        self._access_token = access_token
        self._instruments = instruments
        self._bus = event_bus
        self._heartbeat_timeout = heartbeat_timeout

        self._running = False
        self._last_heartbeat: datetime = utc_now()
        self._last_tick_time: datetime | None = None
        self._ticks_received: int = 0
        self._reconnect_count: int = 0
        self._stream_healthy = False
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def is_healthy(self) -> bool:
        if not self._stream_healthy:
            return False
        elapsed = (utc_now() - self._last_heartbeat).total_seconds()
        return elapsed < self._heartbeat_timeout

    @property
    def stats(self) -> dict:
        return {
            "ticks_received": self._ticks_received,
            "reconnect_count": self._reconnect_count,
            "healthy": self.is_healthy,
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "last_tick": self._last_tick_time.isoformat() if self._last_tick_time else None,
        }

    async def start(self) -> None:
        """Start the streaming connection."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_with_reconnect())
        log.info("stream_manager_started", instruments=self._instruments)

    async def stop(self) -> None:
        """Stop the streaming connection gracefully."""
        self._running = False
        self._stream_healthy = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        log.info("stream_manager_stopped", ticks=self._ticks_received)

    async def _run_with_reconnect(self) -> None:
        """Connect with exponential backoff on failures."""
        delay = _BASE_RECONNECT_DELAY

        while self._running:
            try:
                connect_time = asyncio.get_event_loop().time()
                await self._stream_prices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't set _stream_healthy = False here — let the heartbeat
                # timeout (60s) detect truly dead streams.  Transient disconnects
                # that reconnect in 1-2s should NOT trigger the circuit breaker.
                self._reconnect_count += 1

                # Reset delay if stream was healthy for a meaningful period (>30s)
                # Prevents delay spiraling to 60s after transient disconnects
                elapsed = asyncio.get_event_loop().time() - connect_time
                if elapsed > 30:
                    delay = _BASE_RECONNECT_DELAY

                log.warning(
                    "stream_disconnected",
                    error=str(e),
                    reconnect_in=delay,
                    reconnect_count=self._reconnect_count,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY)
            else:
                # Clean disconnect (stream ended normally)
                delay = _BASE_RECONNECT_DELAY

    async def _stream_prices(self) -> None:
        """Open and process the OANDA price stream."""
        url = f"{self._stream_url}/v3/accounts/{self._account_id}/pricing/stream"
        params = {"instruments": ",".join(self._instruments)}
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ConnectionError(f"Stream HTTP {resp.status}: {body}")

                self._stream_healthy = True
                self._last_heartbeat = utc_now()
                log.info("stream_connected", instruments=self._instruments)

                async for line in resp.content:
                    if not self._running:
                        break

                    decoded = line.decode("utf-8").strip()
                    if not decoded:
                        continue

                    try:
                        msg = json.loads(decoded)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "HEARTBEAT":
                        self._last_heartbeat = utc_now()
                        continue

                    if msg_type == "PRICE":
                        await self._handle_price(msg)

        finally:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    async def _handle_price(self, msg: dict) -> None:
        """Parse a PRICE message and publish a TickEvent."""
        try:
            instrument = msg["instrument"]
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])

            if not bids or not asks:
                return

            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])

            # Parse OANDA timestamp
            time_str = msg.get("time", "")
            if time_str:
                # "2024-01-15T10:30:00.123456789Z"
                cleaned = time_str.rstrip("Z").split(".")[0]
                ts = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            else:
                ts = utc_now()

            tick = TickEvent(instrument=instrument, bid=bid, ask=ask, timestamp=ts)
            self._bus.publish_nowait(tick)

            self._ticks_received += 1
            self._last_tick_time = ts
            self._last_heartbeat = utc_now()  # price = alive

        except (KeyError, ValueError, IndexError) as e:
            log.warning("stream_parse_error", error=str(e), raw=str(msg)[:200])
