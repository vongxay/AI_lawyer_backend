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
    rate_limit_enabled: bool = True
    rate_limit_query_per_minute: int = 10
    rate_limit_stream_per_minute: int = 30
    rate_limit_document_per_minute: int = 5
    rate_limit_evidence_per_minute: int = 5
    rate_limit_backend: Literal["auto", "redis", "memory"] = "auto"

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
    redis_required: bool = False
    redis_max_connections: int = 20
    redis_socket_timeout_seconds: float = 0.5
    cache_ttl_embedding_seconds: int = 86_400   # 24h
    cache_ttl_legal_qa_seconds: int = 3_600     # 1h
    cache_ttl_law_summary_seconds: int = 21_600 # 6h

    # ── AI Providers ───────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_title: str = "AI Lawyer"
    anthropic_api_key: str | None = None

    # Model aliases — change here to swap models globally
    # Low-cost Claude API default. Override only when a workflow needs a stronger model.
    model_economy: str = "openai/gpt-4o-mini"               # OpenRouter/OpenAI-compatible fallback
    model_reasoning: str = "claude-sonnet-4-20250514"       # IRAC reasoning
    model_research: str = "openai/gpt-4o-mini"              # Legal research / standard QA
    model_verification: str = "openai/gpt-4o-mini"          # Citation verification
    model_document: str = "openai/gpt-4o-mini"              # Document analysis
    model_evidence: str = "openai/gpt-4o-mini"              # Evidence analysis
    model_risk: str = "claude-sonnet-4-20250514"            # Risk & strategy

    # LLM budget controls. Defaults are tuned for a small prepaid balance.
    llm_economy_mode: bool = True
    llm_fallback_to_economy_model: bool = True
    llm_verify_citations_with_llm: bool = False
    llm_max_tokens_reasoning: int = 1600
    llm_max_tokens_document: int = 1400
    llm_max_tokens_evidence: int = 1000
    llm_max_tokens_risk: int = 1000
    llm_max_tokens_verification: int = 350
    llm_max_tokens_draft: int = 1500
    llm_max_tokens_absolute: int = 1800

    # Embedding models
    embedding_model_en: str = "text-embedding-3-large"
    embedding_model_multilingual: str = "text-embedding-3-large"
    embedding_dims: int = 1536

    # ── Agent Behaviour ────────────────────────────────────────────────────────
    confidence_escalation_threshold: float = 0.70
    confidence_refuse_threshold: float = 0.50
    agent_timeout_seconds: float = 30.0
    agent_max_retries: int = 2
    rag_top_k: int = 10
    graph_depth: int = 2
    citation_rejection_alert_rate: float = 0.20  # alert admin if > 20%
    rag_chunk_max_chars: int = 2600
    rag_chunk_overlap_chars: int = 350
    rag_embedding_batch_size: int = 48
    rag_plan_max_queries: int = 3
    reasoning_context_top_k: int = 6
    reasoning_context_chunk_chars: int = 1200
    case_analysis_min_fact_chars: int = 120

    # ── Storage ────────────────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    max_request_body_mb: int = 55
    supabase_documents_bucket: str = "documents"
    supabase_evidence_bucket: str = "evidence"
    storage_signed_url_ttl_seconds: int = 3600
    url_ingest_max_documents: int = 10
    pdf_ocr_enabled: bool = True
    pdf_detect_garbled_text: bool = True
    pdf_ocr_max_pages: int = 0
    pdf_ocr_dpi: int = 300
    pdf_ocr_languages: str = "lao+tha+eng"
    tesseract_cmd: str | None = None
    tessdata_prefix: str | None = None
    allowed_upload_types: str = (
        "application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
        "application/msword,text/plain,text/csv,text/markdown,image/jpeg,image/png,image/webp,"
        "audio/mpeg,audio/wav,audio/x-wav,audio/mp4,video/mp4,application/zip,application/x-zip-compressed"
    )

    @property
    def allowed_mime_types(self) -> set[str]:
        return set(self.allowed_upload_types.split(","))

    # ── Validation ─────────────────────────────────────────────────────────────
    @staticmethod
    def _looks_configured_secret(value: str | None) -> bool:
        if not value:
            return False
        stripped = value.strip()
        return bool(stripped) and "..." not in stripped and not stripped.endswith("-")

    @staticmethod
    def _looks_openrouter_key(value: str | None) -> bool:
        return bool(value and value.strip().startswith("sk-or-"))

    @staticmethod
    def _looks_openrouter_base_url(value: str | None) -> bool:
        return bool(value and "openrouter.ai" in value.lower())

    @staticmethod
    def _is_openrouter_model(model: str) -> bool:
        return "/" in model

    @staticmethod
    def _is_openai_model(model: str) -> bool:
        return (
            model.startswith("gpt")
            or model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("openai/")
        )

    @staticmethod
    def _is_anthropic_model(model: str) -> bool:
        return model.startswith("claude")

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
                ]
                if not self._looks_configured_secret(val)
            ]
            llm_models = [
                self.model_economy,
                self.model_reasoning,
                self.model_research,
                self.model_verification,
                self.model_document,
                self.model_evidence,
                self.model_risk,
            ]
            has_anthropic_key = self._looks_configured_secret(self.anthropic_api_key)
            has_openai_key = self._looks_configured_secret(self.openai_api_key)
            has_openrouter_key = self._looks_configured_secret(self.openrouter_api_key) or (
                has_openai_key
                and (
                    self._looks_openrouter_key(self.openai_api_key)
                    or self._looks_openrouter_base_url(self.openai_base_url)
                )
            )
            has_openai_compatible_key = has_openai_key or has_openrouter_key

            if (
                any(self._is_anthropic_model(model) for model in llm_models)
                and not has_anthropic_key
                and not has_openrouter_key
            ):
                missing.append("ANTHROPIC_API_KEY")
            if (
                any(self._is_openai_model(model) for model in llm_models)
                and not has_openai_compatible_key
                and not (self.llm_fallback_to_economy_model and has_anthropic_key)
            ):
                missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
            if any(self._is_openrouter_model(model) for model in llm_models) and not has_openai_compatible_key:
                missing.append("OPENROUTER_API_KEY or OPENAI_API_KEY")
            if missing:
                raise ValueError(f"Production requires these env vars: {missing}")
        return self

    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
