"""Bulk historical data download — CLI for fetching OANDA candle data."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger
from src.api.client import OANDAClient
from src.data.history import HistoryFetcher

log = get_logger(__name__)


async def download(
    instruments: list[str],
    granularity: str,
    from_date: str,
    to_date: str | None,
) -> None:
    config = load_config()
    setup_logging(level="INFO", log_format="console")

    client = OANDAClient(config.oanda)
    fetcher = HistoryFetcher(client, cache_dir=config.data.get("parquet_dir", "data/parquet"))

    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = (
        datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if to_date
        else datetime.now(timezone.utc)
    )

    for instrument in instruments:
        print(f"\nDownloading {instrument} {granularity} from {from_date} to {to_date or 'now'}...")
        df = await fetcher.fetch_candles(
            instrument=instrument,
            granularity=granularity,
            from_time=from_dt,
            to_time=to_dt,
            use_cache=True,
        )
        print(f"  Got {len(df)} candles")
        if not df.empty:
            print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical OANDA candle data")
    parser.add_argument(
        "-i", "--instruments",
        nargs="+",
        default=["EUR_USD", "USD_JPY", "GBP_USD"],
        help="Instruments to download",
    )
    parser.add_argument("-g", "--granularity", default="M5", help="Candle granularity")
    parser.add_argument("-f", "--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-t", "--to-date", default=None, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    asyncio.run(download(args.instruments, args.granularity, args.from_date, args.to_date))


if __name__ == "__main__":
    main()
