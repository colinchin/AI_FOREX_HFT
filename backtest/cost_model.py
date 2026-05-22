"""Cost-corrected backtest accounting for the OANDA-vs-ECN broker experiment.

Why this module exists
----------------------
The base ``BacktestSimulator`` charges only **half** the configured spread —
applied to the entry fill price (``simulator.py:106``) — and lets positions
close at the exact SL/TP price with no spread re-applied. That under-counts
the round-trip cost by half.

For the ECN experiment that asymmetry is fatal: the ECN cost model legitimately
charges full round-trip commission, but the OANDA leg under-counts its
round-trip spread by half. The comparison would be structurally biased toward
OANDA on tight-spread pairs (commission > half-spread savings even when full
round-trip ECN cost is strictly lower than full round-trip OANDA cost).

``CostAwareSimulator`` fixes both halves of the asymmetry:

  1. Adds the missing half-spread cost at position close (so round-trip
     spread is charged in full, matching real broker behaviour).
  2. Deducts a round-trip ECN commission as bps of notional, in the same
     quote currency the base simulator books PnL in.

Set ``commission_bps_round_trip = 0`` for the corrected-OANDA accounting
(spread only, no commission). Set it to ``COMMISSION_BPS_ROUND_TRIP`` for
the ECN accounting (corrected spread + commission). Same simulator class,
same accounting framework, only the cost inputs differ — that is the
controlled-experiment requirement.

Cost components
---------------
1. **ECN raw spread** — derived per-pair from the existing OANDA SPREAD_TABLE.
   ``ecn_raw_spread_pips = max(0.2, oanda_spread_pips * 0.30)``
   Real ECN raw spread on majors is often 10–20% of OANDA standard; the 30%
   factor with a 0.2-pip floor deliberately over-charges ECN so the
   experiment cannot fool itself in ECN's favour.

2. **Commission** — bps of notional, round-trip, deducted in quote currency.
   IC Markets Raw / Pepperstone Razor charge ~USD $3.50 per side per 100k
   notional ≈ 0.7 bps round-trip. We use 0.8 bps round-trip (slightly worse
   than real).

       ``COMMISSION_BPS_ROUND_TRIP = 0.8``  (= 0.4 bps per side)

   commission_quote = units * entry_price * (COMMISSION_BPS_ROUND_TRIP / 10_000)
                    = units * entry_price * 0.00008

   Sanity: 100k EUR_USD at price 1.08 → 100_000 * 1.08 * 0.00008 = $8.64 USD
   per round-trip. In the $7–9 ballpark of real ECN commission.

3. **Close-side half-spread** — pip_value(instrument) * (spread_pips / 2) per
   unit, deducted from PnL at close. This pairs with the half-spread the
   base simulator already applies at entry to produce a full round-trip
   spread cost.
"""

from __future__ import annotations

from backtest.engine import ConversionRateCache
from backtest.simulator import BacktestSimulator, SimulatedPosition
from src.core.events import CandleEvent, Direction, TradeCloseEvent
from src.utils.helpers import pip_value


# ── Fixed cost inputs ────────────────────────────────────────────────────────
COMMISSION_BPS_ROUND_TRIP = 0.8           # bps of notional, round-trip
COMMISSION_FRACTION = COMMISSION_BPS_ROUND_TRIP / 10_000.0   # 0.00008

ECN_SPREAD_FRACTION = 0.30                # ECN raw spread = OANDA * this
ECN_SPREAD_FLOOR_PIPS = 0.2               # but never below 0.2 pips


def ecn_spread_for(oanda_spread_pips: float) -> float:
    """Derive ECN raw spread (pips) from the OANDA spread used in the baseline."""
    return max(ECN_SPREAD_FLOOR_PIPS, oanda_spread_pips * ECN_SPREAD_FRACTION)


def ecn_spread_override(instrument: str, oanda_spread_pips: float) -> float:
    """SPREAD_OVERRIDE callable for walk_forward_optimize.py (ECN mode)."""
    return ecn_spread_for(oanda_spread_pips)


class CostAwareSimulator(BacktestSimulator):
    """BacktestSimulator with full round-trip spread + optional ECN commission.

    Difference vs the parent class — both applied inside ``_close_position``:
      * subtract the close-side half-spread (the base sim only applies the
        entry-side half), so round-trip spread is correctly charged in full;
      * subtract the round-trip commission (bps of notional in quote ccy).

    Set ``commission_bps_round_trip = 0`` for corrected-OANDA accounting,
    ``COMMISSION_BPS_ROUND_TRIP`` for ECN accounting.

    Entry, SL/TP detection, trailing stops, and equity-curve bookkeeping are
    unchanged from the parent so the OANDA-vs-ECN comparison stays controlled
    (only the cost inputs vary).

    ``conversion_cache`` is accepted to preserve the factory's API surface and
    for potential future AUD-side reporting; the deduction itself operates in
    quote currency (matching the parent's PnL booking).
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        commission_per_unit: float = 0.0,
        commission_bps_round_trip: float = COMMISSION_BPS_ROUND_TRIP,
        conversion_cache: ConversionRateCache | None = None,
    ) -> None:
        super().__init__(
            initial_equity=initial_equity,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_unit=commission_per_unit,
        )
        self._commission_fraction = commission_bps_round_trip / 10_000.0
        self._conversion_cache = conversion_cache

    def _close_position(
        self,
        pos: SimulatedPosition,
        close_price: float,
        candle: CandleEvent,
        reason: str,
    ) -> TradeCloseEvent:
        """Close, book PnL, then deduct close-side half-spread + commission.

        Reimplements (not super-calls) the parent's ``_close_position`` because
        ``TradeCloseEvent`` is a frozen dataclass — we cannot mutate
        ``event.pnl`` after construction, and downstream DSR must see the
        cost-adjusted PnL series.
        """
        if pos.direction is Direction.LONG:
            gross_pnl = (close_price - pos.entry_price) * pos.units
        else:
            gross_pnl = (pos.entry_price - close_price) * pos.units

        # Close-side half-spread (pairs with the entry-side half the parent
        # already applied at fill, producing a full round-trip spread charge).
        pv = pip_value(pos.instrument)
        half_spread_per_unit = (self._spread_pips / 2.0) * pv
        spread_cost_quote = half_spread_per_unit * pos.units

        # Round-trip ECN commission as bps of notional in quote currency.
        commission_quote = pos.units * pos.entry_price * self._commission_fraction

        net_pnl = gross_pnl - spread_cost_quote - commission_quote

        self._balance += net_pnl

        event = TradeCloseEvent(
            instrument=pos.instrument,
            trade_id=pos.trade_id,
            close_price=close_price,
            pnl=net_pnl,
            timestamp=candle.timestamp,
            reason=reason,
        )
        self._closes.append(event)
        return event


def cost_aware_simulator_factory_ecn(
    initial_equity: float,
    spread_pips: float,
    slippage_pips: float,
    conversion_cache: ConversionRateCache | None,
) -> CostAwareSimulator:
    """SIMULATOR_FACTORY for the ECN cost model — full round-trip spread + commission."""
    return CostAwareSimulator(
        initial_equity=initial_equity,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_bps_round_trip=COMMISSION_BPS_ROUND_TRIP,
        conversion_cache=conversion_cache,
    )


def cost_aware_simulator_factory_oanda(
    initial_equity: float,
    spread_pips: float,
    slippage_pips: float,
    conversion_cache: ConversionRateCache | None,
) -> CostAwareSimulator:
    """SIMULATOR_FACTORY for the corrected-OANDA accounting — full round-trip
    spread, no commission. The right comparison-baseline for the ECN run."""
    return CostAwareSimulator(
        initial_equity=initial_equity,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_bps_round_trip=0.0,
        conversion_cache=conversion_cache,
    )


# Back-compat alias for the original ECN factory name.
cost_aware_simulator_factory = cost_aware_simulator_factory_ecn
