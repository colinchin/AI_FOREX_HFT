# Setup Guide — Running on a New Machine

## Prerequisites

- Python 3.11+ installed
- VS Code with the Python extension
- OANDA practice (or live) account with API access

## Step 1: Open the Project

Open the `AI_FOREX_HFT` folder in VS Code.

## Step 2: Create a Virtual Environment

Open the VS Code terminal (`Ctrl + backtick`) and run:

```bash
python -m venv .venv
```

Activate it:

- **Windows**: `.venv\Scripts\activate`
- **Mac/Linux**: `source .venv/bin/activate`

VS Code should auto-detect the venv. If prompted, select it as the Python interpreter.

## Step 3: Install Dependencies

```bash
pip install -e ".[dev]"
```

This installs all runtime and test dependencies from `pyproject.toml`.

## Step 4: Configure OANDA Credentials

Edit `config/.env` with your credentials:

```env
OANDA_ACCOUNT_ID=your-account-id
OANDA_ACCESS_TOKEN=your-api-token
```

If you copied the entire folder, the `.env` file should already be present. Verify the token is correct for the new machine.

## Step 5: Verify Connectivity

```bash
python scripts/diagnose.py
```

You should see output with `connected: true` and your account balance.

## Step 6: Run the System

```bash
python -m src.main
```

You should see the startup sequence:

1. OANDA connected (account balance, currency)
2. News calendar refreshed (ForexFactory events)
3. Indicator warmup (M5 + H1 history per instrument)
4. Stream connected (tick data flowing)
5. Status reports every 60 seconds

Stop with `Ctrl+C` for graceful shutdown.

## Step 7: Run Tests (Optional)

```bash
python -m pytest tests/ -v
```

Should show 92/92 passing.

## Notes

- **Historical data cache**: The `data/parquet/` folder carries over, so warmup won't re-download history on first run.
- **Trade journal**: `data/trades.db` (SQLite) carries over with historical trade records.
- **Logs**: Written to `logs/trading.log` in JSON format.
- **Timezone**: All times are UTC internally. The machine's local timezone does not matter.
- **Cross-platform**: Works on Windows, Mac, and Linux without changes.
- **No trades outside sessions**: If you start during off-hours, the session filter will block signals until London (08:00 UTC) or New York (13:00 UTC) opens.
