from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]


def _split_env_list(env_name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(env_name, "")
    if not raw_value.strip():
        return default

    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or default


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "SAM3 Studio API"
    api_prefix: str = "/api"
    frontend_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://faiht.ai",
            "https://faiht.ai",
            "http://www.faiht.ai",
            "https://www.faiht.ai",
        ],
        alias="FRONTEND_ORIGINS",
    )
    data_dir: Path = ROOT_DIR / "data"
    uploads_dir: Path = data_dir / "uploads"
    max_upload_size_mb: int = 32

    sam3_device: str = Field(default="auto", alias="SAM3_DEVICE")
    sam3_checkpoint_path: str | None = Field(default=None, alias="SAM3_CHECKPOINT_PATH")
    sam3_bpe_path: str | None = Field(default=None, alias="SAM3_BPE_PATH")
    sam3_load_from_hf: bool = Field(default=True, alias="SAM3_LOAD_FROM_HF")
    sam3_compile: bool = Field(default=False, alias="SAM3_COMPILE")
    sam3_worker_count: int = Field(default=3, alias="SAM3_WORKER_COUNT", ge=1)
    sam3_queue_max_size: int = Field(default=0, alias="SAM3_QUEUE_MAX_SIZE", ge=0)
    sam3_request_timeout_seconds: float = Field(
        default=300.0,
        alias="SAM3_REQUEST_TIMEOUT_SECONDS",
        gt=0,
    )
    sam3_confidence_threshold: float = Field(
        default=0.5,
        alias="SAM3_CONFIDENCE_THRESHOLD",
    )

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def parse_frontend_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
