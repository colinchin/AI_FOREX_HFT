"""Configuration loader — YAML settings + .env integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class OANDAConfig:
    account_id: str
    access_token: str
    environment: str  # practice | live
    rest_url: str
    stream_url: str


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    max_consecutive_losses: int
    cooldown_seconds: int
    max_trades_per_day: int
    max_spread_pips: float
    max_spread_cost_pct: float
    min_equity: float
    position_sizing: dict[str, Any]
    circuit_breaker: dict[str, Any]
    weekend_close: dict[str, Any]
    volatility_filter: dict[str, Any]
    news_filter: dict[str, Any]


@dataclass
class AppConfig:
    """Top-level application configuration."""

    raw: dict[str, Any] = field(repr=False)
    oanda: OANDAConfig = field(init=False)
    risk: RiskConfig = field(init=False)

    # Convenience accessors
    instruments: list[str] = field(init=False)
    granularities: dict[str, str] = field(init=False)
    strategy: dict[str, Any] = field(init=False)
    sessions: dict[str, Any] = field(init=False)
    data: dict[str, Any] = field(init=False)
    logging_cfg: dict[str, Any] = field(init=False)
    monitoring: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        env = self.raw.get("environment", os.getenv("OANDA_ENVIRONMENT", "practice"))
        is_practice = env == "practice"
        oanda_cfg = self.raw.get("oanda", {})

        self.oanda = OANDAConfig(
            account_id=os.environ["OANDA_ACCOUNT_ID"],
            access_token=os.environ["OANDA_ACCESS_TOKEN"],
            environment=env,
            rest_url=oanda_cfg.get("practice_url") if is_practice else oanda_cfg.get("live_url"),
            stream_url=oanda_cfg.get("stream_practice_url") if is_practice else oanda_cfg.get("stream_live_url"),
        )

        risk_raw = self.raw.get("risk", self.raw)  # handle nested or flat
        if "risk" in risk_raw:
            risk_raw = risk_raw["risk"]
        self.risk = RiskConfig(
            max_risk_per_trade=risk_raw["max_risk_per_trade"],
            max_daily_loss=risk_raw["max_daily_loss"],
            max_open_positions=risk_raw["max_open_positions"],
            max_consecutive_losses=risk_raw["max_consecutive_losses"],
            cooldown_seconds=risk_raw["cooldown_seconds"],
            max_trades_per_day=risk_raw["max_trades_per_day"],
            max_spread_pips=risk_raw["max_spread_pips"],
            max_spread_cost_pct=risk_raw.get("max_spread_cost_pct", 0.30),
            min_equity=risk_raw.get("min_equity", 100.0),
            position_sizing=risk_raw.get("position_sizing", {}),
            circuit_breaker=risk_raw.get("circuit_breaker", {}),
            weekend_close=risk_raw.get("weekend_close", {}),
            volatility_filter=risk_raw.get("volatility_filter", {}),
            news_filter=risk_raw.get("news_filter", {}),
        )

        self.instruments = self.raw.get("instruments", ["EUR_USD"])
        self.granularities = self.raw.get("granularities", {"signal": "M5"})
        self.strategy = self.raw.get("strategy", {})
        self.sessions = self.raw.get("sessions", {})
        self.data = self.raw.get("data", {})
        self.logging_cfg = self.raw.get("logging", {})
        self.monitoring = self.raw.get("monitoring", {})


def load_config(
    settings_path: Path | None = None,
    risk_path: Path | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    """Load and merge settings + risk YAML, then resolve env vars."""
    if env_path is None:
        env_path = _CONFIG_DIR / ".env"
    load_dotenv(env_path, override=False)

    if settings_path is None:
        settings_path = _CONFIG_DIR / "settings.yaml"
    if risk_path is None:
        risk_path = _CONFIG_DIR / "risk.yaml"

    with open(settings_path) as f:
        settings = yaml.safe_load(f) or {}
    with open(risk_path) as f:
        risk = yaml.safe_load(f) or {}

    merged = _deep_merge(settings, risk)
    return AppConfig(raw=merged)
