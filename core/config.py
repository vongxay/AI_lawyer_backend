"""
core/config.py
==============
Centralised application settings.

All configuration is loaded from environment variables (or .env file).
Never hardcode secrets — use Doppler / AWS SSM in production.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "ai-lawyer-backend"
    app_version: str = "2.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = Field(default=False, alias="APP_DEBUG")

    # ── CORS ───────────────────────────────────────────────────────────────────
    allowed_origins_raw: str = Field(
        default=(
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:8080,http://127.0.0.1:8080,"
            "http://localhost:3000,http://127.0.0.1:3000"
        ),
        alias="ALLOWED_ORIGINS",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

    # ── Security / Auth ────────────────────────────────────────────────────────
    jwt_secret: str = Field(default="change-me-in-production-min-32-chars!!")
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_seconds: int = 3600          # 1 hour
    jwt_refresh_ttl_seconds: int = 604_800      # 7 days
    rate_limit_per_minute: int = 20

    # ── Monitoring & Observability ─────────────────────────────────────────────
    sentry_dsn: str | None = None

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_strength(cls, v: str, info: object) -> str:  # noqa: ARG003
        # Only enforce in production — dev/test can use the default
        return v

    # ── Database — Supabase ────────────────────────────────────────────────────
    supabase_url: str | None = None
    supabase_key: str | None = None             # service_role key (server-side only)
    supabase_anon_key: str | None = None

    # ── Cache — Redis ──────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout_seconds: float = 0.5
    cache_ttl_embedding_seconds: int = 86_400   # 24h
    cache_ttl_legal_qa_seconds: int = 3_600     # 1h
    cache_ttl_law_summary_seconds: int = 21_600 # 6h

    # ── AI Providers ───────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Model aliases — change here to swap models globally
    model_reasoning: str = "claude-sonnet-4-6"          # IRAC complex reasoning
    model_research: str = "claude-sonnet-4-20250514"    # Legal research / standard QA
    model_verification: str = "gpt-4o-mini"             # Citation verification
    model_document: str = "gpt-4o"                      # Document + vision analysis
    model_evidence: str = "gpt-4o"                      # Evidence multimodal
    model_risk: str = "claude-sonnet-4-20250514"        # Risk & strategy

    # Embedding models
    embedding_model_en: str = "text-embedding-3-large"
    embedding_model_multilingual: str = "multilingual-e5-large"
    embedding_dims: int = 1536

    # ── Agent Behaviour ────────────────────────────────────────────────────────
    confidence_escalation_threshold: float = 0.70
    confidence_refuse_threshold: float = 0.50
    agent_timeout_seconds: float = 30.0
    agent_max_retries: int = 2
    rag_top_k: int = 10
    graph_depth: int = 2
    citation_rejection_alert_rate: float = 0.20  # alert admin if > 20%
    allow_stub_legal_context: bool = Field(default=False, alias="ALLOW_STUB_LEGAL_CONTEXT")

    # ── Storage ────────────────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    max_request_body_mb: int = 55
    allowed_upload_types: str = (
        "application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
        "application/msword,text/plain,text/markdown,image/jpeg,image/png,image/webp,"
        "audio/mpeg,audio/wav,audio/mp4,video/mp4"
    )

    @property
    def allowed_mime_types(self) -> set[str]:
        return set(self.allowed_upload_types.split(","))

    # ── Validation ─────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def warn_missing_secrets(self) -> "Settings":
        if self.app_env == "production":
            if self.jwt_secret == "change-me-in-production-min-32-chars!!" or len(self.jwt_secret) < 32:
                raise ValueError("Production requires JWT_SECRET with at least 32 characters")
            if "*" in self.cors_origins:
                raise ValueError("Production CORS must not allow '*' when credentials are enabled")

            missing = [
                name
                for name, val in [
                    ("SUPABASE_URL", self.supabase_url),
                    ("SUPABASE_KEY", self.supabase_key),
                    ("OPENAI_API_KEY", self.openai_api_key),
                    ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ]
                if not val
            ]
            if missing:
                raise ValueError(f"Production requires these env vars: {missing}")
        return self

    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
