"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.events import CandleEvent, SignalEvent, TickEvent

if TYPE_CHECKING:
    from src.core.bus import EventBus


class Strategy(ABC):
    """Base class for all trading strategies.

    Strategies receive candle events, compute indicators, and emit signals.
    The same strategy class is used for both live trading and backtesting.
    """

    def __init__(self, name: str, config: dict, event_bus: EventBus | None = None) -> None:
        self._name = name
        self._config = config
        self._bus = event_bus
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    async def on_candle(self, candle: CandleEvent) -> SignalEvent | None:
        """Process a completed candle and optionally generate a signal.

        Returns a SignalEvent if conditions are met, None otherwise.
        """
        ...

    async def on_tick(self, tick: TickEvent) -> None:
        """Process a tick for intra-candle logic (trailing stops, etc.).

        Default implementation does nothing — override if needed.
        """
        pass

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Number of historical candles needed before the strategy can generate signals."""
        ...

    @abstractmethod
    def warmup(self, candles: list) -> None:
        """Seed indicators with historical candle data."""
        ...

    async def _emit_signal(self, signal: SignalEvent) -> None:
        """Publish a signal to the event bus."""
        if self._bus is not None:
            await self._bus.publish(signal)
