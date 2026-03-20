"""Common helpers — pip conversion, time utilities, instrument metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from src.utils.logger import get_logger

log = get_logger(__name__)

# Pip multipliers per instrument category
_PIP_LOCATIONS: dict[str, int] = {
    # JPY pairs: 1 pip = 0.01
    "USD_JPY": -2,
    "EUR_JPY": -2,
    "GBP_JPY": -2,
    "AUD_JPY": -2,
    "CAD_JPY": -2,
    "CHF_JPY": -2,
    "NZD_JPY": -2,
}
_DEFAULT_PIP_LOCATION = -4  # Most pairs: 1 pip = 0.0001


def pip_location(instrument: str) -> int:
    """Return the pip location exponent for an instrument."""
    return _PIP_LOCATIONS.get(instrument, _DEFAULT_PIP_LOCATION)


def pip_value(instrument: str) -> float:
    """Return the value of 1 pip for an instrument (e.g. 0.0001)."""
    return 10.0 ** pip_location(instrument)


def price_to_pips(price_diff: float, instrument: str) -> float:
    """Convert a raw price difference to pips."""
    return price_diff / pip_value(instrument)


def pips_to_price(pips: float, instrument: str) -> float:
    """Convert pips to a raw price difference."""
    return pips * pip_value(instrument)


def spread_pips(bid: float, ask: float, instrument: str) -> float:
    """Calculate spread in pips from bid/ask."""
    return price_to_pips(ask - bid, instrument)


def price_precision(instrument: str) -> int:
    """Return the number of decimal places for formatting prices.

    JPY pairs use 3 decimal places (1 pip = 0.01, price like 150.123).
    Most other pairs use 5 decimal places (1 pip = 0.0001, price like 1.09123).
    """
    loc = pip_location(instrument)
    # Precision = |pip_location| + 1 (sub-pip precision)
    return abs(loc) + 1


def format_price(price: float, instrument: str) -> str:
    """Format a price with instrument-appropriate decimal precision."""
    prec = price_precision(instrument)
    return f"{price:.{prec}f}"


def quote_currency(instrument: str) -> str:
    """Extract quote currency from instrument (e.g., 'EUR_USD' → 'USD')."""
    parts = instrument.split("_")
    return parts[1] if len(parts) == 2 else "USD"


def _direct_or_inverse_rate(
    from_ccy: str, to_ccy: str, rate_lookup: callable,
) -> float | None:
    """Try to find a direct or inverse conversion rate from live tick data."""
    r = rate_lookup(f"{from_ccy}_{to_ccy}")
    if r and r > 0:
        return r
    r = rate_lookup(f"{to_ccy}_{from_ccy}")
    if r and r > 0:
        return 1.0 / r
    return None


def get_conversion_rate(
    from_ccy: str, to_ccy: str, rate_lookup: callable,
) -> float | None:
    """Get conversion rate from one currency to another using available tick data.

    Tries direct/inverse pairs first, then single-hop triangulation through
    common intermediaries, and finally two-hop triangulation (3 legs).

    Args:
        from_ccy: Source currency code (e.g. "GBP")
        to_ccy: Target currency code (e.g. "AUD")
        rate_lookup: Callable(instrument) -> mid_price or None

    Returns:
        Conversion rate, or None if no path found.
    """
    if from_ccy == to_ccy:
        return 1.0

    # Direct or inverse
    r = _direct_or_inverse_rate(from_ccy, to_ccy, rate_lookup)
    if r:
        return r

    _CCYS = ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD")

    # Single-hop triangulation (2 legs)
    for mid in _CCYS:
        if mid in (from_ccy, to_ccy):
            continue
        leg1 = _direct_or_inverse_rate(from_ccy, mid, rate_lookup)
        if not leg1:
            continue
        leg2 = _direct_or_inverse_rate(mid, to_ccy, rate_lookup)
        if leg2:
            return leg1 * leg2

    # Two-hop triangulation (3 legs) — needed for e.g. CAD→AUD via NZD+USD
    for mid1 in _CCYS:
        if mid1 in (from_ccy, to_ccy):
            continue
        leg1 = _direct_or_inverse_rate(from_ccy, mid1, rate_lookup)
        if not leg1:
            continue
        for mid2 in _CCYS:
            if mid2 in (from_ccy, to_ccy, mid1):
                continue
            leg2 = _direct_or_inverse_rate(mid1, mid2, rate_lookup)
            if not leg2:
                continue
            leg3 = _direct_or_inverse_rate(mid2, to_ccy, rate_lookup)
            if leg3:
                return leg1 * leg2 * leg3

    return None


def sl_distance_in_account_ccy(
    sl_distance: float,
    instrument: str,
    current_price: float,
    account_currency: str = "AUD",
    rate_lookup: callable | None = None,
) -> float:
    """Convert SL distance from quote currency to account currency.

    Uses live tick data to find proper conversion rates, including for
    cross pairs (EUR_GBP, NZD_CAD, etc.) where simple price division
    is incorrect.

    Args:
        sl_distance: Stop loss distance in quote currency price units
        instrument: Instrument name (e.g. "EUR_GBP")
        current_price: Current mid price of the instrument
        account_currency: Account base currency (e.g. "AUD")
        rate_lookup: Callable(instrument) -> mid_price or None.
            If None, falls back to legacy approximation.
    """
    quote = quote_currency(instrument)
    if quote == account_currency:
        return sl_distance

    # If we have a rate lookup, use proper conversion
    if rate_lookup is not None:
        rate = get_conversion_rate(quote, account_currency, rate_lookup)
        if rate is not None:
            return sl_distance * rate
        # Conversion unavailable — return 0 so the caller rejects the trade
        # rather than sizing on a mathematically wrong approximation.
        log.warning(
            "sl_conversion_unavailable",
            instrument=instrument,
            quote=quote,
            account_currency=account_currency,
        )
        return 0.0

    # No rate_lookup at all (e.g. backtest without conversion cache) —
    # fall back only when quote is USD (a reasonable approximation for
    # USD-quoted pairs with an AUD account).
    return sl_distance / current_price if current_price > 0 else 0.0


def utc_now() -> datetime:
    """Current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def parse_oanda_time(time_str: str) -> datetime:
    """Parse OANDA RFC3339 timestamp to timezone-aware datetime."""
    # OANDA uses format like "2024-01-15T10:30:00.000000000Z"
    cleaned = time_str.rstrip("Z").split(".")[0]
    return datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def is_within_session(
    dt: datetime,
    session_start: str,
    session_end: str,
) -> bool:
    """Check if a datetime falls within a trading session (UTC HH:MM strings)."""
    start_h, start_m = map(int, session_start.split(":"))
    end_h, end_m = map(int, session_end.split(":"))
    current_minutes = dt.hour * 60 + dt.minute
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    if start_minutes <= end_minutes:
        return start_minutes <= current_minutes < end_minutes
    # Wraps midnight
    return current_minutes >= start_minutes or current_minutes < end_minutes
