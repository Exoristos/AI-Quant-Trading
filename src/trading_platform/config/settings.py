"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global app configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    eodhd_api_key: Optional[str] = Field(default=None, validation_alias="EODHD_API_KEY")
    fred_api_key: Optional[str] = Field(default=None, validation_alias="FRED_API_KEY")
    evds_api_key: Optional[str] = Field(default=None, validation_alias="EVDS_API_KEY")
    # EVDS 3: default API path mirrors legacy ``/service/evds/`` on evds3 host; override if TCMB changes docs.
    evds_base_url: str = Field(
        default="https://evds3.tcmb.gov.tr/service/evds/",
        validation_alias="EVDS_BASE_URL",
    )

    @field_validator("evds_base_url", mode="before")
    @classmethod
    def _normalize_evds_base(cls, v: object) -> str:
        """Treat empty env override as missing; ensure trailing slash."""
        default = "https://evds3.tcmb.gov.tr/service/evds/"
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return str(v).strip().rstrip("/") + "/"
    artifacts_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    )


class DataSettings(BaseModel):
    """Defaults for data ingestion."""

    default_us_exchange_suffix: str = ""
    bist_exchange_suffix: str = ".IS"
    trading_days_per_year: int = 252


class BacktestSettings(BaseModel):
    """Default simulation parameters."""

    initial_cash: float = 100_000.0
    commission: float = 0.001
    slippage_bps: float = 5.0
    position_size_pct: float = 0.10
    risk_free_daily: float = 0.0
