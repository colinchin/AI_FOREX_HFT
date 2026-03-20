"""Event dataclasses for the trading system event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from src.utils.helpers import utc_now


class Direction(Enum):
    LONG = auto()
    SHORT = auto()

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> Direction:
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


class SignalStrength(Enum):
    WEAK = auto()
    MODERATE = auto()
    STRONG = auto()


@dataclass(frozen=True, slots=True)
class TickEvent:
    instrument: str
    bid: float
    ask: float
    timestamp: datetime
    spread: float = field(init=False)
    mid: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spread", self.ask - self.bid)
        object.__setattr__(self, "mid", (self.bid + self.ask) / 2.0)


@dataclass(frozen=True, slots=True)
class CandleEvent:
    instrument: str
    timeframe: str  # M1, M5, H1, etc.
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime  # candle open time
    complete: bool = True

    @property
    def body(self) -> float:
        """Absolute candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class SignalEvent:
    instrument: str
    direction: Direction
    strength: SignalStrength
    strategy_name: str
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict = field(default_factory=dict)  # indicator values, reasons


@dataclass(frozen=True, slots=True)
class OrderEvent:
    instrument: str
    direction: Direction
    units: int
    order_type: OrderType
    stop_loss: float
    take_profit: float
    timestamp: datetime = field(default_factory=utc_now)
    price_bound: float | None = None  # max slippage for market orders
    trailing_stop_distance: float | None = None
    trailing_activate_distance: float | None = None
    strategy_name: str = ""


@dataclass(frozen=True, slots=True)
class FillEvent:
    instrument: str
    direction: Direction
    units: int
    fill_price: float
    trade_id: str
    timestamp: datetime = field(default_factory=utc_now)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_name: str = ""
    trailing_stop_distance: float | None = None
    trailing_activate_distance: float | None = None


@dataclass(frozen=True, slots=True)
class TradeCloseEvent:
    instrument: str
    trade_id: str
    close_price: float
    pnl: float
    timestamp: datetime = field(default_factory=utc_now)
    reason: str = ""  # tp_hit, sl_hit, time_exit, manual, circuit_breaker


@dataclass(frozen=True, slots=True)
class TradeReducedEvent:
    """A trade was partially closed (broker-side or manual)."""
    instrument: str
    trade_id: str
    units_reduced: int        # absolute units removed
    remaining_units: int      # absolute units still open
    pnl: float                # realised PnL from the reduced portion
    timestamp: datetime = field(default_factory=utc_now)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    source: str
    message: str
    timestamp: datetime = field(default_factory=utc_now)
    critical: bool = False


# Union type for event bus
Event = (
    TickEvent | CandleEvent | SignalEvent | OrderEvent | FillEvent
    | TradeCloseEvent | TradeReducedEvent | ErrorEvent
)
