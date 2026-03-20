"""Position sizing — fixed fractional and ATR-based methods.

Ensures each trade risks exactly the configured percentage of equity,
accounting for stop-loss distance and spread.
"""

from __future__ import annotations

from src.utils.helpers import sl_distance_in_account_ccy
from src.utils.logger import get_logger

log = get_logger(__name__)


class PositionSizer:
    """Calculate position size based on risk parameters."""

    def __init__(
        self,
        method: str = "atr_based",
        min_units: int = 1,
        max_units: int = 100_000,
        max_position_pct: float = 0.10,
        account_currency: str = "AUD",
    ) -> None:
        self._method = method
        self._min_units = min_units
        self._max_units = max_units
        self._max_position_pct = max_position_pct
        self._account_currency = account_currency

    def calculate(
        self,
        equity: float,
        risk_pct: float,
        sl_distance: float,
        instrument: str,
        current_price: float,
        spread: float = 0.0,
        rate_lookup: callable | None = None,
    ) -> int:
        """Calculate position size in units.

        Args:
            equity: Current account equity
            risk_pct: Fraction of equity to risk (e.g. 0.01 = 1%)
            sl_distance: Stop loss distance in price (not pips)
            instrument: Instrument name for pip value lookup
            current_price: Current market price
            spread: Current spread in price

        Returns:
            Position size in units (always >= min_units, <= max_units)
        """
        if sl_distance <= 0 or equity <= 0 or risk_pct <= 0:
            return 0

        # Dollar amount we're willing to risk
        risk_amount = equity * risk_pct

        # Effective SL distance includes spread cost
        effective_sl = sl_distance + spread

        if effective_sl <= 0:
            return 0

        # Convert SL distance to account currency.
        # Uses live tick data for proper cross-rate conversion.
        effective_sl_usd = sl_distance_in_account_ccy(
            effective_sl, instrument, current_price,
            account_currency=self._account_currency,
            rate_lookup=rate_lookup,
        )

        if effective_sl_usd <= 0:
            return 0

        # Units = risk_amount / (risk per unit in account currency)
        units = int(risk_amount / effective_sl_usd)

        # Apply position cap (max % of equity at current price)
        # Convert notional to account currency for accurate cap
        max_notional = equity * self._max_position_pct
        if current_price > 0:
            # Notional per unit = base_ccy price in account_ccy terms
            parts = instrument.split("_")
            base_ccy = parts[0]
            quote_ccy = parts[1] if len(parts) > 1 else ""
            notional_rate = None
            if rate_lookup is not None:
                from src.utils.helpers import get_conversion_rate
                notional_rate = get_conversion_rate(
                    base_ccy, self._account_currency, rate_lookup,
                )
            if notional_rate:
                unit_value = notional_rate
            elif quote_ccy == self._account_currency:
                # Quote == account currency: current_price is correct
                unit_value = current_price
            else:
                # Can't convert — skip notional cap rather than use a wrong value.
                # The SL-based sizing above is the primary risk control.
                unit_value = 0
            if unit_value > 0:
                max_by_notional = int(max_notional / unit_value)
                units = min(units, max_by_notional)

        # Reject if below minimum (don't inflate tiny positions)
        if units < self._min_units:
            log.debug(
                "position_too_small",
                instrument=instrument,
                calculated_units=units,
                min_units=self._min_units,
            )
            return 0

        # Clamp to max
        units = min(units, self._max_units)

        log.debug(
            "position_sized",
            instrument=instrument,
            equity=equity,
            risk_pct=risk_pct,
            sl_distance=sl_distance,
            spread=spread,
            units=units,
        )

        return units
