"""Backtest runner CLI — run strategies against historical data."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.data_loader import BacktestDataLoader
from backtest.engine import BacktestEngine, ConversionRateCache, load_conversion_series
from backtest.simulator import BacktestSimulator
from src.strategy.momentum_scalp import MomentumScalpStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.multi_tf import MultiTimeframeFilter
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

log = get_logger(__name__)


STRATEGIES = {
    "momentum_scalp": MomentumScalpStrategy,
    "mean_reversion": MeanReversionStrategy,
}


async def run_backtest(
    strategy_name: str,
    instrument: str,
    from_date: str | None,
    to_date: str | None,
    initial_equity: float,
    spread_pips: float,
    slippage_pips: float,
) -> None:
    config = load_config()
    setup_logging(level="INFO", log_format="console")

    loader = BacktestDataLoader(config.data.get("parquet_dir", "data/parquet"))

    # Load M5 data
    print(f"\nLoading {instrument} M5 data...")
    m5_df = loader.load_candles(instrument, "M5", from_date, to_date)
    if m5_df.empty:
        print("No M5 data found. Run download_history.py first.")
        return
    m5_events = loader.df_to_candle_events(m5_df, instrument, "M5")
    print(f"  Loaded {len(m5_events)} M5 candles")

    # Load H1 data for multi-TF
    h1_events = None
    try:
        h1_df = loader.load_candles(instrument, "H1", from_date, to_date)
        if not h1_df.empty:
            h1_events = loader.df_to_candle_events(h1_df, instrument, "H1")
            print(f"  Loaded {len(h1_events)} H1 candles")
    except FileNotFoundError:
        print("  No H1 data available (running without multi-TF filter)")

    # Initialize strategy
    strategy_cls = STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        print(f"Unknown strategy: {strategy_name}")
        return

    strategy_config = config.strategy.get(strategy_name, config.strategy)
    strategy = strategy_cls(config=strategy_config)

    # Initialize simulator
    simulator = BacktestSimulator(
        initial_equity=initial_equity,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
    )

    # Multi-TF filter
    mtf = MultiTimeframeFilter() if h1_events else None

    # Load conversion rate time series for cross-currency sizing (no look-ahead)
    data_dir = config.data.get("parquet_dir", "data/parquet")
    conversion_cache = ConversionRateCache(load_conversion_series(data_dir))

    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        simulator=simulator,
        config=config.raw,
        multi_tf_filter=mtf,
        account_currency=config.raw.get("account_currency", "AUD"),
        conversion_cache=conversion_cache,
    )

    print(f"\nRunning backtest: {strategy_name} on {instrument}")
    print(f"  Period: {from_date or 'start'} to {to_date or 'end'}")
    print(f"  Equity: {initial_equity:,.0f}")
    print(f"  Spread: {spread_pips} pips, Slippage: {slippage_pips} pips")
    print()

    report = await engine.run(m5_events, h1_events)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    # Default to primary strategy from config
    _cfg = load_config()
    _default_strategy = _cfg.strategy.get("primary", "mean_reversion")
    parser.add_argument("-s", "--strategy", default=_default_strategy, choices=STRATEGIES.keys())
    parser.add_argument("-i", "--instrument", default="EUR_USD")
    parser.add_argument("-f", "--from-date", default=None)
    parser.add_argument("-t", "--to-date", default=None)
    parser.add_argument("-e", "--equity", type=float, default=100_000)
    parser.add_argument("--spread", type=float, default=1.5, help="Spread in pips")
    parser.add_argument("--slippage", type=float, default=0.5, help="Slippage in pips")

    args = parser.parse_args()
    asyncio.run(run_backtest(
        args.strategy, args.instrument,
        args.from_date, args.to_date,
        args.equity, args.spread, args.slippage,
    ))


if __name__ == "__main__":
    main()
