"""Backend settings from environment variables / .env (never hard-coded secrets)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from predictops.config import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'database' / 'predictops.db'}"
    upload_dir: str = str(PROJECT_ROOT / "data" / "uploads")
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    api_key: str | None = None          # if set, every request (except /health) needs header X-API-Key
    max_predict_minutes: int = 60 * 24 * 14   # guard against huge synchronous requests (14 days)

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
