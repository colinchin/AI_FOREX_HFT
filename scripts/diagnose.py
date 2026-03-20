"""Connectivity diagnostics — verify OANDA practice account setup."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger
from src.api.client import OANDAClient

log = get_logger(__name__)


async def run_diagnostics() -> None:
    config = load_config()
    setup_logging(level="INFO", log_format="console")

    print("=" * 60)
    print("  OANDA Practice Account Diagnostics")
    print("=" * 60)

    client = OANDAClient(config.oanda)

    # 1. Health check
    print("\n[1] Connection Health Check...")
    try:
        health = await client.health_check()
        print(f"    Connected:       {health['connected']}")
        print(f"    Environment:     {health['environment']}")
        print(f"    Account ID:      {health['account_id']}")
        print(f"    Balance:         {health['balance']} {health['currency']}")
        print(f"    Open Trades:     {health['open_trade_count']}")
        print(f"    Latency:         {health['latency_ms']:.1f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # 2. Instruments
    print("\n[2] Available Instruments (configured)...")
    try:
        all_instruments = await client.get_instruments()
        instrument_names = {i["name"] for i in all_instruments}
        for inst in config.instruments:
            available = inst in instrument_names
            status = "OK" if available else "NOT FOUND"
            print(f"    {inst}: {status}")
    except Exception as e:
        print(f"    FAILED: {e}")

    # 3. Current pricing
    print("\n[3] Current Prices...")
    try:
        pricing_data = await client.get_pricing(config.instruments)
        for price in pricing_data.get("prices", []):
            instrument = price["instrument"]
            bids = price.get("bids", [{}])
            asks = price.get("asks", [{}])
            bid = float(bids[0].get("price", 0)) if bids else 0
            ask = float(asks[0].get("price", 0)) if asks else 0
            spread = ask - bid
            from src.utils.helpers import price_to_pips
            spread_p = price_to_pips(spread, instrument)
            print(f"    {instrument}: Bid={bid:.5f}  Ask={ask:.5f}  Spread={spread_p:.1f} pips")
    except Exception as e:
        print(f"    FAILED: {e}")

    # 4. Historical data test
    print("\n[4] Historical Data (last 5 M5 candles, EUR_USD)...")
    try:
        candles = await client.get_candles("EUR_USD", granularity="M5", count=5)
        for c in candles:
            mid = c.get("mid", {})
            t = c["time"][:19]
            print(f"    {t}  O={mid['o']}  H={mid['h']}  L={mid['l']}  C={mid['c']}  Vol={c.get('volume', 0)}")
    except Exception as e:
        print(f"    FAILED: {e}")

    print("\n" + "=" * 60)
    print("  Diagnostics Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
