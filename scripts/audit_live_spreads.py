"""Audit live OANDA spreads against scripts/backtest_all_pairs.py:SPREAD_TABLE.

Polls OANDA's current pricing endpoint N times spaced T seconds apart for all
28 G10 pairs and reports:
  * min, mean, max spread observed across the polls (pips)
  * the SPREAD_TABLE value used by the broker-cost experiment
  * a "realistic" estimate = max(observed) — pessimistic by design so the
    audit cannot flatter the backtest

Output: a YAML block ready to paste into a corrected SPREAD_TABLE, plus a
JSON dump for downstream scripts.

This is read-only on the broker side — only the pricing endpoint, no orders.

Usage:
    # Default: 6 samples spaced 30s apart over ~3 minutes
    python scripts/audit_live_spreads.py

    # Wider sample, e.g. 12 samples over 6 minutes:
    python scripts/audit_live_spreads.py --samples 12 --interval 30

    # Quick single-shot (least reliable):
    python scripts/audit_live_spreads.py --samples 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.client import OANDAClient
from src.utils.config import load_config
from src.utils.helpers import pip_value
from src.utils.logger import setup_logging
from scripts.backtest_all_pairs import G10_PAIRS, SPREAD_TABLE


async def sample_once(client: OANDAClient, instruments: list[str]) -> dict[str, float]:
    """Return {instrument: spread_pips} for one snapshot."""
    out: dict[str, float] = {}
    resp = await client.get_pricing(instruments)
    for p in resp.get("prices", []):
        instr = p["instrument"]
        try:
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
        except (KeyError, IndexError, ValueError):
            continue
        pv = pip_value(instr)
        out[instr] = (ask - bid) / pv
    return out


async def run_audit(samples: int, interval: float) -> None:
    config = load_config()
    client = OANDAClient(config.oanda)

    print(f"\nLive-spread audit — {samples} samples × {interval:.0f}s interval"
          f" = ~{samples * interval / 60:.1f} min")
    print(f"Pairs: {len(G10_PAIRS)} G10\n")

    history: dict[str, list[float]] = {p: [] for p in G10_PAIRS}
    for i in range(samples):
        t0 = time.monotonic()
        snap = await sample_once(client, G10_PAIRS)
        for p in G10_PAIRS:
            v = snap.get(p)
            if v is not None:
                history[p].append(v)
        print(f"  sample {i+1}/{samples}: {len(snap)} pairs returned "
              f"({time.monotonic()-t0:.1f}s)")
        if i < samples - 1:
            await asyncio.sleep(interval)

    # Aggregate
    rows = []
    for p in G10_PAIRS:
        obs = history[p]
        if not obs:
            rows.append({
                "pair": p, "table": SPREAD_TABLE.get(p),
                "min": None, "mean": None, "max": None, "n": 0, "realistic": None,
            })
            continue
        rows.append({
            "pair": p,
            "table": SPREAD_TABLE.get(p, 2.0),
            "min": min(obs),
            "mean": sum(obs) / len(obs),
            "max": max(obs),
            "n": len(obs),
            "realistic": max(obs),
        })

    # Sort by |delta| descending to put the biggest mismatches first
    def _delta(r):
        if r["realistic"] is None:
            return 0
        return abs(r["realistic"] - r["table"])
    rows.sort(key=_delta, reverse=True)

    print("\n" + "=" * 96)
    print("  LIVE SPREAD AUDIT — observed vs SPREAD_TABLE")
    print("=" * 96)
    print(f"  {'Pair':<10} {'Table':>6} {'Live min':>9} {'Live mean':>10} {'Live max':>9} "
          f"{'Realistic':>10} {'Δ vs table':>11} {'Note':<14}")
    print("-" * 96)
    big_delta = 0
    for r in rows:
        if r["realistic"] is None:
            print(f"  {r['pair']:<10} {r['table']:>6.1f} {'—':>9} {'—':>10} {'—':>9}"
                  f" {'—':>10} {'—':>11} {'no data':<14}")
            continue
        delta = r["realistic"] - r["table"]
        ratio = r["realistic"] / r["table"] if r["table"] else 0
        note = ""
        if abs(delta) >= 1.0 and ratio >= 1.5:
            note = "MATERIAL"
            big_delta += 1
        elif abs(delta) >= 0.5:
            note = "modest"
        print(f"  {r['pair']:<10} {r['table']:>6.1f} {r['min']:>9.2f} {r['mean']:>10.2f}"
              f" {r['max']:>9.2f} {r['realistic']:>10.2f} {delta:>+11.2f} {note:<14}")
    print("-" * 96)
    print(f"  Pairs with MATERIAL deviation (>=1 pip AND >=1.5x table): {big_delta}/{len(rows)}")
    print(f"  Realistic = max observed across {samples} samples (deliberately pessimistic).")
    print("=" * 96)

    # Persist
    out_dir = Path("data/spread_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "spread_audit.json"
    yaml_path = out_dir / "corrected_spread_table.yaml"
    with open(json_path, "w") as f:
        json.dump({
            "samples": samples,
            "interval_seconds": interval,
            "results": rows,
        }, f, indent=2, default=str)
    with open(yaml_path, "w") as f:
        f.write("# Corrected SPREAD_TABLE — realistic = max of live samples (pessimistic)\n")
        f.write("# Generated by scripts/audit_live_spreads.py\n")
        f.write("# Compare against scripts/backtest_all_pairs.py:SPREAD_TABLE\n\n")
        f.write("SPREAD_TABLE_CORRECTED:\n")
        # Re-sort by pair name for stable diffing
        for r in sorted(rows, key=lambda x: x["pair"]):
            if r["realistic"] is None:
                f.write(f"  {r['pair']}: {r['table']}    # no live data, keeping table value\n")
            else:
                f.write(f"  {r['pair']}: {round(r['realistic'], 1)}\n")
    print(f"\n  JSON: {json_path}")
    print(f"  YAML: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live OANDA spreads vs SPREAD_TABLE.")
    parser.add_argument("--samples", type=int, default=6,
                        help="Number of pricing polls (default 6)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between polls (default 30)")
    args = parser.parse_args()
    setup_logging(level="WARNING", log_format="console")
    asyncio.run(run_audit(args.samples, args.interval))


if __name__ == "__main__":
    main()
