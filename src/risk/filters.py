"""Risk filters — spread, session, volatility, news.

Each filter returns (allowed: bool, reason: str).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.events import SignalEvent
from src.data.tick_buffer import TickBuffer
from src.strategy.indicators import IncrementalATR
from src.utils.helpers import is_within_session, spread_pips, utc_now
from src.utils.logger import get_logger

log = get_logger(__name__)


class SpreadFilter:
    """Reject signals when current spread exceeds threshold."""

    def __init__(self, max_spread_pips: float, tick_buffer: TickBuffer) -> None:
        self._max_pips = max_spread_pips
        self._tick_buffer = tick_buffer

    def check(self, signal: SignalEvent) -> tuple[bool, str]:
        tick = self._tick_buffer.latest(signal.instrument)
        if tick is None:
            return False, "no_tick_data"

        current_spread = spread_pips(tick.bid, tick.ask, signal.instrument)
        if current_spread > self._max_pips:
            return False, f"spread_too_wide:{current_spread:.1f}>{self._max_pips:.1f}"

        return True, ""


class SpreadCostFilter:
    """Reject when spread eats too much of the TP distance."""

    def __init__(self, max_cost_pct: float, tick_buffer: TickBuffer) -> None:
        self._max_pct = max_cost_pct
        self._tick_buffer = tick_buffer

    def check(self, signal: SignalEvent) -> tuple[bool, str]:
        tp_distance = signal.metadata.get("tp_distance")
        if tp_distance is None or tp_distance <= 0:
            return True, ""  # Can't check, allow

        tick = self._tick_buffer.latest(signal.instrument)
        if tick is None:
            return False, "no_tick_data"

        cost_pct = tick.spread / tp_distance
        if cost_pct > self._max_pct:
            return False, f"spread_cost_too_high:{cost_pct:.0%}>{self._max_pct:.0%}"

        return True, ""


class SessionFilter:
    """Only allow signals during configured trading sessions."""

    def __init__(self, sessions: dict[str, Any], active_sessions: list[str]) -> None:
        self._sessions = sessions
        self._active = active_sessions

    def check(self, signal: SignalEvent) -> tuple[bool, str]:
        now = signal.timestamp if signal.timestamp.tzinfo else utc_now()

        for session_name in self._active:
            session = self._sessions.get(session_name, {})
            start = session.get("start")
            end = session.get("end")
            if start and end and is_within_session(now, start, end):
                return True, ""

        return False, f"outside_active_sessions"


class VolatilityFilter:
    """Reject when ATR is abnormally high (potential news event / flash crash)."""

    def __init__(self, max_std_devs: float = 2.0, lookback: int = 50) -> None:
        self._max_std = max_std_devs
        self._lookback = lookback
        # Track ATR history per instrument
        self._atr_history: dict[str, list[float]] = {}

    def record_atr(self, instrument: str, atr_value: float) -> None:
        """Record an ATR observation for the instrument."""
        if instrument not in self._atr_history:
            self._atr_history[instrument] = []
        hist = self._atr_history[instrument]
        hist.append(atr_value)
        if len(hist) > self._lookback:
            self._atr_history[instrument] = hist[-self._lookback:]

    def check(self, signal: SignalEvent) -> tuple[bool, str]:
        atr_val = signal.metadata.get("atr")
        if atr_val is None:
            return True, ""  # Can't check, allow

        hist = self._atr_history.get(signal.instrument, [])
        if len(hist) < 10:
            return True, ""  # Not enough history

        import numpy as np
        mean = np.mean(hist)
        std = np.std(hist)

        if std > 0 and atr_val > mean + self._max_std * std:
            return False, f"volatility_too_high:atr={atr_val:.6f}>threshold={mean + self._max_std * std:.6f}"

        return True, ""


class NewsFilter:
    """Block trading around high-impact economic news events.

    Fetches the ForexFactory weekly calendar via the Fair Economy mirror API,
    caches locally, and blocks signals when an event affecting the traded
    currencies falls within a configurable blackout window.
    """

    _CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self, config: dict[str, Any]) -> None:
        self._enabled: bool = config.get("enabled", True)
        self._hours_before: float = config.get("hours_before", 1.0)
        self._hours_after: float = config.get("hours_after", 1.0)
        self._impact_levels: list[str] = [
            lvl.strip().capitalize() for lvl in config.get("impact_levels", ["High"])
        ]
        self._cache_seconds: int = max(config.get("cache_seconds", 3600), 60)

        # Cached calendar events and fetch state
        self._events: list[dict[str, Any]] = []
        self._last_fetch: datetime | None = None

    async def refresh(self) -> None:
        """Fetch calendar from API if cache is stale. Call periodically."""
        if not self._enabled:
            return

        now = utc_now()
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._cache_seconds:
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._CALENDAR_URL,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.json(content_type=None)
                        # Pre-filter to only relevant impact levels and parse dates
                        self._events = self._parse_events(raw)
                        self._last_fetch = utc_now()
                        log.info(
                            "news_calendar_refreshed",
                            total_fetched=len(raw),
                            filtered_events=len(self._events),
                        )
                    else:
                        log.warning("news_calendar_fetch_failed", status=resp.status)
        except Exception as e:
            log.warning("news_calendar_fetch_error", error=str(e))

    def _parse_events(self, raw: list[dict]) -> list[dict[str, Any]]:
        """Parse and filter calendar events to relevant impact levels."""
        parsed = []
        for ev in raw:
            impact = ev.get("impact", "").strip().capitalize()
            if impact not in self._impact_levels:
                continue

            date_str = ev.get("date", "")
            if not date_str:
                continue

            try:
                # FF dates: "2026-03-20T08:30:00-04:00" (US Eastern with offset)
                event_time = datetime.fromisoformat(date_str)
                # Normalise to UTC
                if event_time.tzinfo is not None:
                    event_time = event_time.astimezone(timezone.utc)
                else:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            parsed.append({
                "title": ev.get("title", "unknown"),
                "country": ev.get("country", "").upper(),
                "time_utc": event_time,
                "impact": impact,
            })
        return parsed

    def check(self, signal: SignalEvent) -> tuple[bool, str]:
        if not self._enabled:
            return True, ""

        # Fail closed: block if calendar was never fetched or is stale.
        # Grace period = 2x cache TTL (one full missed refresh cycle).
        if self._last_fetch is None:
            return False, "news_calendar_unavailable"

        # Use wall-clock time, not signal creation time, so queued signals
        # are still checked against the current blackout window.
        now = utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        stale_limit = self._cache_seconds * 2
        if (now - self._last_fetch).total_seconds() > stale_limit:
            return False, "news_calendar_stale"

        # Currencies from instrument (EUR_USD → {"EUR", "USD"})
        currencies = set(signal.instrument.split("_"))

        before = timedelta(hours=self._hours_before)
        after = timedelta(hours=self._hours_after)

        for ev in self._events:
            if ev["country"] not in currencies:
                continue

            ev_time = ev["time_utc"]
            if ev_time - before <= now <= ev_time + after:
                return (
                    False,
                    f"news_blackout:{ev['country']}:{ev['title']}",
                )

        return True, ""
