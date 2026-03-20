"""Tests for the NewsFilter — calendar parsing, blackout windows, fail-closed."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import Direction, SignalEvent, SignalStrength
from src.risk.filters import NewsFilter


def _make_signal(
    instrument: str = "EUR_USD",
    ts: datetime | None = None,
) -> SignalEvent:
    return SignalEvent(
        instrument=instrument,
        direction=Direction.LONG,
        strength=SignalStrength.MODERATE,
        strategy_name="test",
        timestamp=ts or datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
        metadata={"sl_distance": 0.001, "tp_distance": 0.002, "atr": 0.0008},
    )


def _make_calendar_events() -> list[dict]:
    """Simulated ForexFactory calendar JSON payload."""
    return [
        {
            "title": "Non-Farm Payrolls",
            "country": "USD",
            "date": "2026-03-20T08:30:00-04:00",  # 12:30 UTC
            "impact": "High",
            "forecast": "170K",
            "previous": "199K",
        },
        {
            "title": "ECB Rate Decision",
            "country": "EUR",
            "date": "2026-03-20T13:45:00+01:00",  # 12:45 UTC
            "impact": "High",
            "forecast": "",
            "previous": "4.50%",
        },
        {
            "title": "German Ifo",
            "country": "EUR",
            "date": "2026-03-21T09:00:00+01:00",  # 08:00 UTC next day
            "impact": "Medium",
            "forecast": "86.5",
            "previous": "85.2",
        },
        {
            "title": "AU Employment Change",
            "country": "AUD",
            "date": "2026-03-20T00:30:00+11:00",  # 13:30 UTC prev day
            "impact": "High",
            "forecast": "25K",
            "previous": "30K",
        },
        {
            "title": "FOMC Minutes",
            "country": "USD",
            "date": "2026-03-20T18:00:00-04:00",  # 22:00 UTC
            "impact": "Low",
            "forecast": "",
            "previous": "",
        },
    ]


class TestNewsFilterParsing:
    def test_parse_filters_by_impact(self):
        """Only High impact events should be retained by default config."""
        nf = NewsFilter({"enabled": True, "impact_levels": ["High"]})
        parsed = nf._parse_events(_make_calendar_events())

        titles = [e["title"] for e in parsed]
        assert "Non-Farm Payrolls" in titles
        assert "ECB Rate Decision" in titles
        assert "AU Employment Change" in titles
        assert "German Ifo" not in titles  # Medium
        assert "FOMC Minutes" not in titles  # Low

    def test_parse_multiple_impact_levels(self):
        """Should include multiple configured impact levels."""
        nf = NewsFilter({"enabled": True, "impact_levels": ["High", "Medium"]})
        parsed = nf._parse_events(_make_calendar_events())

        titles = [e["title"] for e in parsed]
        assert "German Ifo" in titles
        assert "FOMC Minutes" not in titles  # Low

    def test_parse_normalises_to_utc(self):
        """All parsed event times should be in UTC."""
        nf = NewsFilter({"enabled": True})
        parsed = nf._parse_events(_make_calendar_events())

        for ev in parsed:
            assert ev["time_utc"].tzinfo == timezone.utc

    def test_parse_nfp_time_correct(self):
        """NFP at 08:30 ET (UTC-4) should parse to 12:30 UTC."""
        nf = NewsFilter({"enabled": True})
        parsed = nf._parse_events(_make_calendar_events())

        nfp = next(e for e in parsed if e["title"] == "Non-Farm Payrolls")
        assert nfp["time_utc"] == datetime(2026, 3, 20, 12, 30, tzinfo=timezone.utc)

    def test_parse_country_uppercased(self):
        """Country codes should be uppercased."""
        nf = NewsFilter({"enabled": True})
        raw = [{"title": "Test", "country": "usd", "date": "2026-03-20T12:00:00+00:00", "impact": "High"}]
        parsed = nf._parse_events(raw)
        assert parsed[0]["country"] == "USD"

    def test_parse_skips_bad_dates(self):
        """Events with invalid or missing dates should be skipped."""
        nf = NewsFilter({"enabled": True})
        raw = [
            {"title": "No Date", "country": "USD", "date": "", "impact": "High"},
            {"title": "Bad Date", "country": "USD", "date": "not-a-date", "impact": "High"},
            {"title": "Good", "country": "USD", "date": "2026-03-20T12:00:00+00:00", "impact": "High"},
        ]
        parsed = nf._parse_events(raw)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Good"

    def test_parse_case_insensitive_impact(self):
        """Impact matching should be case-insensitive."""
        nf = NewsFilter({"enabled": True, "impact_levels": ["High"]})
        raw = [{"title": "Test", "country": "USD", "date": "2026-03-20T12:00:00+00:00", "impact": "high"}]
        parsed = nf._parse_events(raw)
        assert len(parsed) == 1


class TestNewsFilterCheck:
    def _loaded_filter(self, hours_before=0.5, hours_after=0.5, now=None) -> NewsFilter:
        """Return a filter with pre-loaded events (simulating a successful fetch).

        Pass the same datetime used for mock_now so the cache is always fresh.
        """
        nf = NewsFilter({
            "enabled": True,
            "hours_before": hours_before,
            "hours_after": hours_after,
            "impact_levels": ["High"],
        })
        nf._events = nf._parse_events(_make_calendar_events())
        nf._last_fetch = now or datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        return nf

    @patch("src.risk.filters.utc_now")
    def test_blocks_during_blackout_before(self, mock_now):
        """Signal 10 min before NFP (12:30 UTC) should be blocked with 30min window."""
        now = datetime(2026, 3, 20, 12, 20, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)
        signal = _make_signal("EUR_USD")

        allowed, reason = nf.check(signal)
        assert not allowed
        assert "USD" in reason
        assert "Non-Farm Payrolls" in reason

    @patch("src.risk.filters.utc_now")
    def test_blocks_during_blackout_after(self, mock_now):
        """Signal 10 min after NFP should be blocked."""
        now = datetime(2026, 3, 20, 12, 40, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)
        signal = _make_signal("EUR_USD")

        allowed, reason = nf.check(signal)
        assert not allowed

    @patch("src.risk.filters.utc_now")
    def test_allows_outside_blackout(self, mock_now):
        """Signal well outside any blackout window should pass."""
        now = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)
        signal = _make_signal("EUR_USD")

        allowed, reason = nf.check(signal)
        assert allowed

    @patch("src.risk.filters.utc_now")
    def test_currency_matching_both_sides(self, mock_now):
        """EUR_USD should match both EUR and USD events."""
        # During ECB (12:45 UTC), EUR_USD should be blocked via EUR
        now = datetime(2026, 3, 20, 12, 45, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert not allowed
        # Could match either NFP (USD) or ECB (EUR) — both are in window

    @patch("src.risk.filters.utc_now")
    def test_unrelated_currency_not_blocked(self, mock_now):
        """GBP_AUD should not be blocked by USD/EUR events."""
        now = datetime(2026, 3, 20, 12, 30, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.25, hours_after=0.25, now=now)

        allowed, reason = nf.check(_make_signal("GBP_AUD"))
        assert allowed

    @patch("src.risk.filters.utc_now")
    def test_aud_event_blocks_aud_pairs(self, mock_now):
        """AUD Employment at 13:30 UTC should block AUD_USD."""
        now = datetime(2026, 3, 19, 13, 30, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.25, hours_after=0.25, now=now)

        allowed, reason = nf.check(_make_signal("AUD_USD"))
        assert not allowed
        assert "AUD" in reason

    @patch("src.risk.filters.utc_now")
    def test_boundary_exact_start(self, mock_now):
        """Signal exactly at blackout start boundary should be blocked."""
        # NFP at 12:30, 30 min before = 12:00
        now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)

        allowed, reason = nf.check(_make_signal("USD_JPY"))
        assert not allowed

    @patch("src.risk.filters.utc_now")
    def test_boundary_exact_end(self, mock_now):
        """Signal exactly at blackout end boundary should be blocked (<=)."""
        # NFP at 12:30, 30 min after = 13:00
        now = datetime(2026, 3, 20, 13, 0, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)

        allowed, reason = nf.check(_make_signal("USD_CHF"))
        assert not allowed

    @patch("src.risk.filters.utc_now")
    def test_boundary_one_second_after_end(self, mock_now):
        """Signal 1 second past blackout end should be allowed."""
        now = datetime(2026, 3, 20, 13, 0, 1, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)

        allowed, reason = nf.check(_make_signal("USD_CHF"))
        assert allowed

    @patch("src.risk.filters.utc_now")
    def test_uses_wall_clock_not_signal_time(self, mock_now):
        """check() should use utc_now(), not signal.timestamp."""
        # Signal was created at 10:00 (safe), but wall clock is 12:30 (NFP time)
        now = datetime(2026, 3, 20, 12, 30, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = self._loaded_filter(hours_before=0.5, hours_after=0.5, now=now)
        signal = _make_signal("EUR_USD", ts=datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))

        allowed, reason = nf.check(signal)
        assert not allowed  # Wall clock is in blackout

    @patch("src.risk.filters.utc_now")
    def test_no_events_this_week_allows(self, mock_now):
        """If calendar fetched successfully but has no relevant events, allow."""
        now = datetime(2026, 3, 20, 12, 30, tzinfo=timezone.utc)
        mock_now.return_value = now
        nf = NewsFilter({"enabled": True})
        nf._events = []  # Successfully fetched, but nothing this week
        nf._last_fetch = now

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert allowed


class TestNewsFilterFailClosed:
    def test_blocks_when_never_fetched(self):
        """Enabled filter that never successfully fetched should block all signals."""
        nf = NewsFilter({"enabled": True})
        # _last_fetch is None — never fetched

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert not allowed
        assert reason == "news_calendar_unavailable"

    def test_disabled_filter_allows(self):
        """Disabled filter should always allow."""
        nf = NewsFilter({"enabled": False})

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert allowed

    def test_empty_config_defaults_enabled(self):
        """Empty config should default to enabled=True and fail closed."""
        nf = NewsFilter({})

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert not allowed
        assert reason == "news_calendar_unavailable"

    @patch("src.risk.filters.utc_now")
    def test_blocks_when_cache_stale(self, mock_now):
        """Should block when last fetch is older than 2x cache_seconds."""
        nf = NewsFilter({"enabled": True, "cache_seconds": 3600})
        # Last fetch was 3 hours ago (> 2x 3600s grace)
        nf._last_fetch = datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc)
        nf._events = []
        mock_now.return_value = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert not allowed
        assert reason == "news_calendar_stale"

    @patch("src.risk.filters.utc_now")
    def test_allows_when_cache_within_grace(self, mock_now):
        """Should allow when last fetch is within 2x cache_seconds."""
        nf = NewsFilter({"enabled": True, "cache_seconds": 3600})
        # Last fetch was 1.5 hours ago (< 2x 3600s grace)
        nf._last_fetch = datetime(2026, 3, 20, 10, 30, tzinfo=timezone.utc)
        nf._events = []  # No events this week
        mock_now.return_value = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        allowed, reason = nf.check(_make_signal("EUR_USD"))
        assert allowed


class TestNewsFilterRefresh:
    @pytest.mark.asyncio
    async def test_successful_refresh_populates_events(self):
        """Successful API call should populate events and set last_fetch."""
        nf = NewsFilter({"enabled": True, "cache_seconds": 3600})

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=_make_calendar_events())

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)
            await nf.refresh()

        assert nf._last_fetch is not None
        assert len(nf._events) > 0

    @pytest.mark.asyncio
    async def test_failed_refresh_keeps_stale_events(self):
        """Failed API call should not clear existing cached events."""
        nf = NewsFilter({"enabled": True, "cache_seconds": 0})
        old_events = [{"title": "Old", "country": "USD", "time_utc": datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc), "impact": "High"}]
        nf._events = old_events
        nf._last_fetch = datetime(2026, 3, 19, 0, 0, tzinfo=timezone.utc)  # stale

        with patch("aiohttp.ClientSession", side_effect=Exception("DNS failure")):
            await nf.refresh()

        # Events should be unchanged
        assert nf._events == old_events

    @pytest.mark.asyncio
    async def test_cache_prevents_refetch(self):
        """Should not refetch when cache is fresh."""
        nf = NewsFilter({"enabled": True, "cache_seconds": 3600})
        nf._last_fetch = datetime.now(timezone.utc)  # just fetched
        nf._events = [{"title": "Cached"}]

        with patch("aiohttp.ClientSession") as mock_cs:
            await nf.refresh()
            mock_cs.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_skips_fetch(self):
        """Disabled filter should not make any HTTP calls."""
        nf = NewsFilter({"enabled": False})

        with patch("aiohttp.ClientSession") as mock_cs:
            await nf.refresh()
            mock_cs.assert_not_called()
