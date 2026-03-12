from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "ai-lawyer-backend"
    log_level: str = "INFO"

    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"

    supabase_url: str | None = None
    supabase_key: str | None = None

    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

