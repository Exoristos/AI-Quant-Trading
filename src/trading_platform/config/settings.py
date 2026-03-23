from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    eodhd_api_key: Optional[str] = Field(default=None, validation_alias="EODHD_API_KEY")
    fred_api_key: Optional[str] = Field(default=None, validation_alias="FRED_API_KEY")
    evds_api_key: Optional[str] = Field(default=None, validation_alias="EVDS_API_KEY")
    evds_base_url: str = Field(
        default="https://evds3.tcmb.gov.tr/service/evds/",
        validation_alias="EVDS_BASE_URL",
    )
    public_ui: bool = Field(default=False, validation_alias="PUBLIC_UI")
    ui_page_title: str = Field(default="AI-Quant-Trading", validation_alias="UI_PAGE_TITLE")
    lstm_seq_len: int = Field(default=20, ge=5, le=120, validation_alias="LSTM_SEQ_LEN")
    lstm_epochs: int = Field(default=15, ge=1, le=200, validation_alias="LSTM_EPOCHS")
    label_horizon: int = Field(default=1, ge=1, le=20, validation_alias="LABEL_HORIZON")
    hold_epsilon: float = Field(default=0.002, ge=0.0001, validation_alias="HOLD_EPSILON")
    evds_series_codes: str = Field(
        default="TP.DK.USD.A.YTL",
        validation_alias="EVDS_SERIES_CODES",
    )
    public_train_each_run: bool = Field(default=True, validation_alias="PUBLIC_TRAIN_EACH_RUN")
    macro_csv_path: Optional[str] = Field(default=None, validation_alias="MACRO_CSV_PATH")
    bist_membership_csv: str = Field(default="", validation_alias="BIST_MEMBERSHIP_CSV")

    @field_validator("bist_membership_csv", mode="before")
    @classmethod
    def _strip_membership_csv(cls, v: object) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return ""
        return str(v).strip()

    @field_validator("macro_csv_path", mode="before")
    @classmethod
    def _blank_macro_csv(cls, v: object) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return str(v).strip()

    @field_validator("evds_series_codes", mode="before")
    @classmethod
    def _strip_evds_codes(cls, v: object) -> str:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            return "TP.DK.USD.A.YTL"
        return str(v).strip()

    @field_validator("evds_base_url", mode="before")
    @classmethod
    def _normalize_evds_base(cls, v: object) -> str:
        default = "https://evds3.tcmb.gov.tr/service/evds/"
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return str(v).strip().rstrip("/") + "/"

    artifacts_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    )


class DataSettings(BaseModel):

    default_us_exchange_suffix: str = ""
    bist_exchange_suffix: str = ".IS"
    trading_days_per_year: int = 252


class BacktestSettings(BaseModel):

    initial_cash: float = 100_000.0
    commission: float = 0.001
    slippage_bps: float = 5.0
    position_size_pct: float = 0.10
    risk_free_daily: float = 0.0
