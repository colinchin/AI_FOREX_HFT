"""Tests for deferred trailing-stop activation and partial trade reduction."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import (
    Direction,
    FillEvent,
    SignalStrength,
    TickEvent,
    TradeCloseEvent,
    TradeReducedEvent,
)
from src.core.models import DailyStats, Position
from src.data.store import TradeStore
from src.portfolio.tracker import PortfolioTracker
from src.risk.manager import RiskManager
from src.utils.config import RiskConfig


# ── helpers ──

def _make_risk_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_risk_per_trade=0.01,
        max_daily_loss=0.03,
        max_open_positions=3,
        max_consecutive_losses=5,
        cooldown_seconds=900,
        max_trades_per_day=50,
        max_spread_pips=2.0,
        max_spread_cost_pct=0.30,
        min_equity=100.0,
        position_sizing={"method": "atr_based", "min_units": 1, "max_units": 100000, "max_position_pct": 0.10},
        circuit_breaker={"stream_timeout_seconds": 30, "max_api_errors": 10, "equity_drawdown_pct": 0.05},
        weekend_close={},
        volatility_filter={"max_atr_std_devs": 2.0, "atr_lookback": 50},
        news_filter={"enabled": False},
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _tick(instrument: str, price: float) -> TickEvent:
    return TickEvent(
        instrument=instrument,
        bid=price - 0.00005,
        ask=price + 0.00005,
        timestamp=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
    )


# ── Trailing-stop activation tests ──

class TestTrailingStopActivation:
    @pytest.mark.asyncio
    async def test_trailing_armed_after_profit_threshold(self):
        """Trailing stop should only be set once profit reaches activation distance."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        execution = AsyncMock()
        execution._set_trailing_stop = AsyncMock()

        tracker = PortfolioTracker(bus, store, execution)

        # Simulate a fill with deferred trailing config
        fill = FillEvent(
            instrument="EUR_USD",
            direction=Direction.LONG,
            units=1000,
            fill_price=1.1000,
            trade_id="T1",
            trailing_stop_distance=0.0005,
            trailing_activate_distance=0.0010,  # arm after 10 pip profit
        )
        await tracker.on_fill(fill)

        pos = tracker.open_positions["T1"]
        assert not pos.trailing_armed

        # Tick at 1.1005 — only 5 pips profit, below threshold
        await tracker.on_tick(_tick("EUR_USD", 1.1005))
        assert not pos.trailing_armed
        execution._set_trailing_stop.assert_not_called()

        # Tick at 1.1011 — above threshold
        await tracker.on_tick(_tick("EUR_USD", 1.1011))
        assert pos.trailing_armed
        execution._set_trailing_stop.assert_called_once_with(
            trade_id="T1",
            distance=0.0005,
            instrument="EUR_USD",
        )

    @pytest.mark.asyncio
    async def test_trailing_retries_on_api_failure(self):
        """If the API call fails, trailing_armed should stay False for retry."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        execution = AsyncMock()
        execution._set_trailing_stop = AsyncMock(side_effect=Exception("API error"))

        tracker = PortfolioTracker(bus, store, execution)

        fill = FillEvent(
            instrument="EUR_USD",
            direction=Direction.LONG,
            units=1000,
            fill_price=1.1000,
            trade_id="T1",
            trailing_stop_distance=0.0005,
            trailing_activate_distance=0.0010,
        )
        await tracker.on_fill(fill)

        # Tick above threshold — API fails
        await tracker.on_tick(_tick("EUR_USD", 1.1015))
        pos = tracker.open_positions["T1"]
        assert not pos.trailing_armed  # should retry

        # Fix the API, next tick retries
        execution._set_trailing_stop = AsyncMock()
        await tracker.on_tick(_tick("EUR_USD", 1.1015))
        assert pos.trailing_armed
        execution._set_trailing_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_trailing_without_activation_distance(self):
        """Fill without trailing config should not trigger any trailing logic."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        execution = AsyncMock()

        tracker = PortfolioTracker(bus, store, execution)

        fill = FillEvent(
            instrument="EUR_USD",
            direction=Direction.LONG,
            units=1000,
            fill_price=1.1000,
            trade_id="T1",
        )
        await tracker.on_fill(fill)

        await tracker.on_tick(_tick("EUR_USD", 1.2000))  # huge profit
        execution._set_trailing_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_trailing_activation(self):
        """Trailing stop for SHORT should activate on price drop."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        execution = AsyncMock()
        execution._set_trailing_stop = AsyncMock()

        tracker = PortfolioTracker(bus, store, execution)

        fill = FillEvent(
            instrument="EUR_USD",
            direction=Direction.SHORT,
            units=1000,
            fill_price=1.1000,
            trade_id="T1",
            trailing_stop_distance=0.0005,
            trailing_activate_distance=0.0010,
        )
        await tracker.on_fill(fill)

        # Price goes up (loss) — no activation
        await tracker.on_tick(_tick("EUR_USD", 1.1010))
        assert not tracker.open_positions["T1"].trailing_armed

        # Price drops 10 pips from entry — activate
        await tracker.on_tick(_tick("EUR_USD", 1.0990))
        assert tracker.open_positions["T1"].trailing_armed


# ── Partial reduction tests ──

class TestTradeReduction:
    @pytest.mark.asyncio
    async def test_partial_reduction_updates_units(self):
        """on_trade_reduced should update the position's unit count."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        fill = FillEvent("EUR_USD", Direction.LONG, 1000, 1.1, "T1")
        await tracker.on_fill(fill)
        assert tracker.open_positions["T1"].units == 1000

        event = TradeReducedEvent(
            instrument="EUR_USD",
            trade_id="T1",
            units_reduced=400,
            remaining_units=600,
            pnl=50.0,
        )
        await tracker.on_trade_reduced(event)

        assert tracker.open_positions["T1"].units == 600
        assert tracker._total_pnl == 50.0
        assert tracker._total_trades == 1
        assert tracker._total_wins == 1

    @pytest.mark.asyncio
    async def test_partial_loss_updates_stats(self):
        """Partial reduction with a loss should update loss stats."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        fill = FillEvent("EUR_USD", Direction.LONG, 1000, 1.1, "T1")
        await tracker.on_fill(fill)

        event = TradeReducedEvent(
            instrument="EUR_USD",
            trade_id="T1",
            units_reduced=500,
            remaining_units=500,
            pnl=-30.0,
        )
        await tracker.on_trade_reduced(event)

        assert tracker._gross_loss == -30.0
        assert tracker._total_wins == 0

    @pytest.mark.asyncio
    async def test_unknown_trade_reduction_ignored(self):
        """Reduction for unknown trade_id should be silently ignored."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        event = TradeReducedEvent(
            instrument="EUR_USD",
            trade_id="UNKNOWN",
            units_reduced=100,
            remaining_units=900,
            pnl=10.0,
        )
        await tracker.on_trade_reduced(event)
        assert tracker._total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_risk_manager_reduction_records_trade(self):
        """RiskManager should route partial PnL through record_trade()."""
        from src.core.bus import EventBus
        from src.data.tick_buffer import TickBuffer

        bus = EventBus()
        tick_buffer = TickBuffer()
        config = _make_risk_config(max_consecutive_losses=2, cooldown_seconds=60)
        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(config, bus, tick_buffer,
                         {"london": {"start": "08:00", "end": "17:00"},
                          "active_sessions": ["london"], "news_blackouts": []},
                         equity_provider)

        # Two partial losses should trigger cooldown
        for i in range(2):
            event = TradeReducedEvent(
                instrument="EUR_USD",
                trade_id=f"T{i}",
                units_reduced=500,
                remaining_units=500,
                pnl=-100.0,
            )
            await rm.on_trade_reduced(event)

        assert rm.daily_stats.trades_taken == 2
        assert rm.daily_stats.losses == 2
        assert rm.daily_stats.net_pnl == -200.0
        assert rm._cooldown_until is not None  # cooldown triggered

    @pytest.mark.asyncio
    async def test_partial_win_resets_loss_streak(self):
        """A partial-close profit should reset the consecutive loss streak."""
        from src.core.bus import EventBus
        from src.data.tick_buffer import TickBuffer

        bus = EventBus()
        tick_buffer = TickBuffer()
        config = _make_risk_config(max_consecutive_losses=3)
        equity_provider = AsyncMock(return_value=100000.0)

        rm = RiskManager(config, bus, tick_buffer,
                         {"london": {"start": "08:00", "end": "17:00"},
                          "active_sessions": ["london"], "news_blackouts": []},
                         equity_provider)

        # One loss, then a partial win
        await rm.on_trade_reduced(TradeReducedEvent(
            instrument="EUR_USD", trade_id="T0",
            units_reduced=500, remaining_units=500, pnl=-50.0,
        ))
        assert rm.daily_stats.current_loss_streak == 1

        await rm.on_trade_reduced(TradeReducedEvent(
            instrument="EUR_USD", trade_id="T1",
            units_reduced=500, remaining_units=500, pnl=30.0,
        ))
        assert rm.daily_stats.current_loss_streak == 0


# ── Reconciliation tests ──

class TestReconciliation:
    def _broker_trade(self, **overrides) -> dict:
        """Minimal OANDA open trade dict."""
        trade = {
            "id": "100",
            "instrument": "EUR_USD",
            "currentUnits": "1000",
            "price": "1.10000",
            "openTime": "2026-03-20T10:00:00Z",
            "stopLossOrder": {"price": "1.09500"},
            "takeProfitOrder": {"price": "1.10500"},
        }
        trade.update(overrides)
        return trade

    def test_strategy_name_restored_from_tag(self):
        """reconcile should parse strategy name from clientExtensions.tag."""
        import json
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        tag = json.dumps({"s": "momentum_scalp"}, separators=(",", ":"))
        trade = self._broker_trade(clientExtensions={"tag": tag})
        tracker.reconcile_open_trades([trade])

        pos = tracker.open_positions["100"]
        assert pos.strategy_name == "momentum_scalp"

    def test_strategy_defaults_to_reconciled_without_tag(self):
        """Without clientExtensions, strategy_name should be 'reconciled'."""
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        tracker.reconcile_open_trades([self._broker_trade()])

        pos = tracker.open_positions["100"]
        assert pos.strategy_name == "reconciled"

    def test_deferred_trailing_restored_from_tag(self):
        """Pending deferred trailing activation should be restored from tag."""
        import json
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        tag = json.dumps(
            {"s": "momentum_scalp", "td": 0.0005, "ta": 0.0010},
            separators=(",", ":"),
        )
        trade = self._broker_trade(clientExtensions={"tag": tag})
        tracker.reconcile_open_trades([trade])

        pos = tracker.open_positions["100"]
        assert pos.trailing_stop_distance == 0.0005
        assert pos.trailing_activate_distance == 0.0010
        assert not pos.trailing_armed  # not yet armed

    def test_armed_trailing_from_broker_overrides_tag(self):
        """If broker already has a trailing stop, it should be marked armed."""
        import json
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        tracker = PortfolioTracker(bus, store)

        tag = json.dumps(
            {"s": "momentum_scalp", "td": 0.0005, "ta": 0.0010},
            separators=(",", ":"),
        )
        trade = self._broker_trade(
            clientExtensions={"tag": tag},
            trailingStopLossOrder={"distance": "0.00050"},
        )
        tracker.reconcile_open_trades([trade])

        pos = tracker.open_positions["100"]
        assert pos.trailing_armed
        assert pos.trailing_stop_distance == 0.0005

    @pytest.mark.asyncio
    async def test_deferred_trailing_activates_after_reconcile(self):
        """Reconciled trade with pending trailing should arm on next profit tick."""
        import json
        bus = MagicMock()
        store = AsyncMock(spec=TradeStore)
        execution = AsyncMock()
        execution._set_trailing_stop = AsyncMock()
        tracker = PortfolioTracker(bus, store, execution)

        tag = json.dumps(
            {"s": "momentum_scalp", "td": 0.0005, "ta": 0.0010},
            separators=(",", ":"),
        )
        trade = self._broker_trade(clientExtensions={"tag": tag})
        tracker.reconcile_open_trades([trade])

        # Tick below activation threshold
        await tracker.on_tick(_tick("EUR_USD", 1.1005))
        assert not tracker.open_positions["100"].trailing_armed

        # Tick above threshold (entry 1.1000 + 0.0010 = 1.1010)
        await tracker.on_tick(_tick("EUR_USD", 1.1011))
        assert tracker.open_positions["100"].trailing_armed
        execution._set_trailing_stop.assert_called_once()
