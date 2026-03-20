"""Backtest performance metrics — Sharpe, Sortino, drawdown, profit factor, etc."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.events import TradeCloseEvent


@dataclass
class PerformanceReport:
    """Comprehensive backtest performance report."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annualized_return: float
    avg_trade_duration_minutes: float
    total_duration_days: float
    trades_per_day: float
    close_reason_breakdown: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"  Total Trades:           {self.total_trades}",
            f"  Winning Trades:         {self.winning_trades}",
            f"  Losing Trades:          {self.losing_trades}",
            f"  Win Rate:               {self.win_rate:.1%}",
            "",
            f"  Total P&L:              {self.total_pnl:,.2f}",
            f"  Gross Profit:           {self.gross_profit:,.2f}",
            f"  Gross Loss:             {self.gross_loss:,.2f}",
            f"  Profit Factor:          {self.profit_factor:.3f}",
            f"  Expectancy/Trade:       {self.expectancy:,.4f}",
            "",
            f"  Average Win:            {self.avg_win:,.4f}",
            f"  Average Loss:           {self.avg_loss:,.4f}",
            f"  Largest Win:            {self.largest_win:,.4f}",
            f"  Largest Loss:           {self.largest_loss:,.4f}",
            "",
            f"  Max Consecutive Wins:   {self.max_consecutive_wins}",
            f"  Max Consecutive Losses: {self.max_consecutive_losses}",
            f"  Max Drawdown:           {self.max_drawdown:,.2f}",
            f"  Max Drawdown %:         {self.max_drawdown_pct:.2%}",
            "",
            f"  Sharpe Ratio:           {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio:          {self.sortino_ratio:.3f}",
            f"  Calmar Ratio:           {self.calmar_ratio:.3f}",
            f"  Annualized Return:      {self.annualized_return:.2%}",
            "",
            f"  Avg Trade Duration:     {self.avg_trade_duration_minutes:.1f} min",
            f"  Trading Period:         {self.total_duration_days:.1f} days",
            f"  Trades/Day:             {self.trades_per_day:.1f}",
            "",
        ]
        if self.close_reason_breakdown:
            lines.append("  Close Reason Breakdown:")
            for reason, data in sorted(self.close_reason_breakdown.items()):
                count = data["count"]
                avg_pnl = data["avg_pnl"]
                pct = count / self.total_trades * 100 if self.total_trades else 0
                lines.append(f"    {reason:20s}  {count:4d} ({pct:5.1f}%)  avg PnL: {avg_pnl:+.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_metrics(
    closes: list[TradeCloseEvent],
    equity_curve: list[tuple],
    initial_equity: float,
) -> PerformanceReport:
    """Calculate comprehensive performance metrics from backtest results."""
    if not closes:
        return _empty_report()

    pnls = [c.pnl for c in closes]
    pnl_array = np.array(pnls)
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array <= 0]

    total_pnl = float(np.sum(pnl_array))
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.sum(losses)) if len(losses) > 0 else 0.0

    # Win rate
    win_rate = len(wins) / len(pnls) if pnls else 0.0

    # Profit factor
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    # Averages
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    # Extremes
    largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
    largest_loss = float(np.min(losses)) if len(losses) > 0 else 0.0

    # Consecutive streaks
    max_consec_wins = _max_consecutive(pnls, positive=True)
    max_consec_losses = _max_consecutive(pnls, positive=False)

    # Drawdown from equity curve
    max_dd, max_dd_pct = _calculate_drawdown(equity_curve, initial_equity)

    # Risk-adjusted returns
    sharpe = _sharpe_ratio(pnls)
    sortino = _sortino_ratio(pnls)

    # Time calculations
    if len(closes) >= 2:
        first_time = closes[0].timestamp
        last_time = closes[-1].timestamp
        total_days = max((last_time - first_time).total_seconds() / 86400, 1)
    else:
        total_days = 1

    # Trade durations
    durations = []
    for c in closes:
        # Duration is approximate from close events
        durations.append(0)  # We don't have entry time in close events alone

    annualized_return = (total_pnl / initial_equity) * (365 / total_days) if total_days > 0 else 0
    calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0

    # Close reason breakdown
    reason_groups: dict[str, list[float]] = {}
    for c in closes:
        reason = c.reason or "unknown"
        reason_groups.setdefault(reason, []).append(c.pnl)
    close_reason_breakdown = {
        reason: {"count": len(pnl_list), "avg_pnl": float(np.mean(pnl_list))}
        for reason, pnl_list in reason_groups.items()
    }

    return PerformanceReport(
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        total_pnl=total_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        expectancy=total_pnl / len(pnls) if pnls else 0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        annualized_return=annualized_return,
        avg_trade_duration_minutes=0,  # Requires entry time tracking
        total_duration_days=total_days,
        trades_per_day=len(pnls) / total_days if total_days > 0 else 0,
        close_reason_breakdown=close_reason_breakdown,
    )


def _max_consecutive(pnls: list[float], positive: bool) -> int:
    """Count maximum consecutive wins or losses."""
    max_streak = 0
    current = 0
    for pnl in pnls:
        if (positive and pnl > 0) or (not positive and pnl <= 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _calculate_drawdown(
    equity_curve: list[tuple], initial_equity: float
) -> tuple[float, float]:
    """Calculate max drawdown in absolute and percentage terms."""
    if not equity_curve:
        return 0.0, 0.0

    equities = [e[1] for e in equity_curve]
    peak = initial_equity
    max_dd = 0.0

    for eq in equities:
        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)

    max_dd_pct = max_dd / initial_equity if initial_equity > 0 else 0
    return max_dd, max_dd_pct


def _sharpe_ratio(pnls: list[float], risk_free: float = 0.0, periods_per_year: float = 252 * 24) -> float:
    """Annualized Sharpe ratio from trade P&Ls."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    mean_return = np.mean(arr) - risk_free
    std_return = np.std(arr, ddof=1)
    if std_return == 0:
        return 0.0
    return float(mean_return / std_return * np.sqrt(min(len(pnls), periods_per_year)))


def _sortino_ratio(pnls: list[float], risk_free: float = 0.0) -> float:
    """Sortino ratio — only penalizes downside volatility."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    mean_return = np.mean(arr) - risk_free
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float("inf") if mean_return > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(mean_return / downside_std * np.sqrt(min(len(pnls), 252 * 24)))


def _empty_report() -> PerformanceReport:
    return PerformanceReport(
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
        total_pnl=0, gross_profit=0, gross_loss=0, profit_factor=0,
        expectancy=0, avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
        max_consecutive_wins=0, max_consecutive_losses=0,
        max_drawdown=0, max_drawdown_pct=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        annualized_return=0, avg_trade_duration_minutes=0,
        total_duration_days=0, trades_per_day=0,
    )
