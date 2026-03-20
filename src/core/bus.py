"""Async event bus — publish/subscribe with type-based routing."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from src.core.events import Event
from src.utils.logger import get_logger

log = get_logger(__name__)

# Handler signature: async def handler(event: SomeEvent) -> None
EventHandler = Callable[[Any], Coroutine[Any, Any, None]]


class EventBus:
    """AsyncIO event bus with type-based pub/sub.

    Handlers are dispatched concurrently per event. The bus processes
    events sequentially from its internal queue to maintain ordering.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers: dict[type, list[EventHandler]] = defaultdict(list)
        self._running = False
        self._task: asyncio.Task | None = None
        self._processed: int = 0
        self._errors: int = 0

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        log.debug("bus_subscribe", event_type=event_type.__name__, handler=handler.__qualname__)

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        """Remove a handler for a specific event type."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Put an event onto the bus queue."""
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> None:
        """Non-blocking publish — drops event if queue is full."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("bus_queue_full", event_type=type(event).__name__)
            self._errors += 1

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribed handlers concurrently."""
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])

        if not handlers:
            return

        # Run all handlers for this event concurrently
        tasks = [asyncio.create_task(h(event)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._errors += 1
                log.error(
                    "bus_handler_error",
                    handler=handlers[i].__qualname__,
                    event_type=event_type.__name__,
                    error=str(result),
                )

    async def _run(self) -> None:
        """Main event processing loop."""
        log.info("bus_started")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._processed += 1
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                log.error("bus_loop_error", error=str(e))

    def start(self) -> None:
        """Start the event processing loop as an asyncio task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the event bus gracefully, draining remaining events."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("bus_stopped", processed=self._processed, errors=self._errors)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "processed": self._processed,
            "errors": self._errors,
            "pending": self._queue.qsize(),
        }
