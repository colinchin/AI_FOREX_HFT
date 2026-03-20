"""Data models for candles, positions, and trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.core.events import Direction
from src.utils.helpers import utc_now


@dataclass
class Candle:
    """OHLCV candle."""
    instrument: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool = True

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class Position:
    """An open position in the portfolio."""
    trade_id: str
    instrument: str
    direction: Direction
    units: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy_name: str
    unrealised_pnl: float = 0.0
    trailing_stop: float | None = None
    highest_pnl: float = 0.0  # For trailing stop tracking
    trailing_stop_distance: float | None = None      # distance to trail by
    trailing_activate_distance: float | None = None   # profit distance before arming
    trailing_armed: bool = False                      # True once activation API called

    def update_pnl(self, current_price: float) -> None:
        """Update unrealised P&L based on current market price."""
        if self.direction is Direction.LONG:
            self.unrealised_pnl = (current_price - self.entry_price) * self.units
        else:
            self.unrealised_pnl = (self.entry_price - current_price) * self.units
        self.highest_pnl = max(self.highest_pnl, self.unrealised_pnl)

    @property
    def duration_seconds(self) -> float:
        return (utc_now() - self.entry_time).total_seconds()


@dataclass
class ClosedTrade:
    """A completed trade record."""
    trade_id: str
    instrument: str
    direction: Direction
    units: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    strategy_name: str
    exit_reason: str  # tp_hit, sl_hit, time_exit, manual, circuit_breaker
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def risk_reward(self) -> float | None:
        """Actual R:R achieved."""
        risk = abs(self.entry_price - self.stop_loss) if self.stop_loss else None
        if not risk or risk == 0:
            return None
        reward = abs(self.exit_price - self.entry_price)
        return reward / risk


@dataclass
class DailyStats:
    """Aggregated daily trading statistics."""
    date: str
    entries_today: int = 0  # incremented on fill (entry) — used for daily trade cap
    trades_taken: int = 0   # incremented on close — used for P&L stats
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0  # max streak seen today (for reporting)
    _current_streak: int = field(default=0, repr=False)

    @property
    def current_loss_streak(self) -> int:
        """Current consecutive loss streak (resets on a win)."""
        return self._current_streak

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades_taken if self.trades_taken > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss != 0 else float("inf")

    def record_entry(self) -> None:
        """Record a new trade entry (fill). Used for daily trade cap."""
        self.entries_today += 1

    def record_trade(self, pnl: float) -> None:
        self.trades_taken += 1
        self.net_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
            self._current_streak = 0
        else:
            self.losses += 1
            self.gross_loss += pnl
            self._current_streak += 1
            self.consecutive_losses = max(self.consecutive_losses, self._current_streak)
        self.max_drawdown = min(self.max_drawdown, self.net_pnl)
