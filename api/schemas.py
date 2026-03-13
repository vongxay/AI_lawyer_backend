"""
api/schemas.py
==============
All Pydantic v2 request/response schemas.

Design rules:
- Request schemas: validate incoming data (strict types)
- Response schemas: match IRAC output exactly (no validation failures)
- All Thai text fields allow any unicode string
- Confidence fields bounded [0.0, 1.0]
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Shared primitives ──────────────────────────────────────────────────────────

class CitationItem(BaseModel):
    ref: str
    status: Literal["VERIFIED", "OUTDATED", "UNVERIFIED", "REJECTED"] = "UNVERIFIED"
    note: str | None = None
    db_match: str | None = None
    year: int | None = None
    reason: str | None = None
    source_links: list[str] = Field(default_factory=list)


# ── IRAC sub-schemas ───────────────────────────────────────────────────────────

class IracIssue(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)


class IracStatute(BaseModel):
    name: str
    section: str
    text: str
    status: str = "ACTIVE"
    year: int | None = None


class IracPrecedent(BaseModel):
    case_no: str
    court: str | None = None
    relevance: str | None = None
    outcome: str | None = None
    graph_path: str | None = None


class IracRule(BaseModel):
    statutes: list[IracStatute] = Field(default_factory=list)
    precedents: list[IracPrecedent] = Field(default_factory=list)


class IracApplication(BaseModel):
    analysis: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    counter_args: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)


class IracConclusion(BaseModel):
    recommendation: str
    action_steps: list[str] = Field(default_factory=list)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    win_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    settlement_note: str | None = None


class IracPayload(BaseModel):
    issue: IracIssue
    rule: IracRule
    application: IracApplication
    conclusion: IracConclusion


# ── Risk/Strategy sub-schemas ──────────────────────────────────────────────────

class StrategicOption(BaseModel):
    name: str
    description: str | None = None
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    est_days: int | None = None
    success_likelihood: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"


class RiskAnalysis(BaseModel):
    win_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    strategic_options: list[StrategicOption] = Field(default_factory=list)
    recommended_option: str | None = None
    immediate_actions: list[str] = Field(default_factory=list)


# ── Request schemas ────────────────────────────────────────────────────────────

class LegalQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=5000)
    case_id: str | None = None
    jurisdiction: str | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class DraftRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=3000)
    document_type: str | None = None
    jurisdiction: str | None = None
    language: Literal["TH", "EN", "LA"] = "TH"


class VerifyCitationsRequest(BaseModel):
    citations: list[CitationItem] = Field(min_length=1, max_length=50)


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    corrected_answer: str | None = Field(default=None, max_length=5000)


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)
    document_type: Literal["law", "case", "form", "guideline"] = "law"
    jurisdiction: str = "TH"
    content: str | None = None
    url: str | None = None


# ── Response schemas ───────────────────────────────────────────────────────────

class LegalQueryResponse(BaseModel):
    """
    Main response schema — mirrors IRAC output from reasoning agent.
    Uses model_config to handle extra fields from LLM (forward-compatible).
    """
    model_config = {"extra": "ignore"}

    irac: IracPayload | dict[str, Any]    # dict fallback when LLM gives partial output
    citations: list[CitationItem] = Field(default_factory=list)
    citations_verified: bool = True
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    agents_used: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    escalated_to_expert: bool = False
    risk: RiskAnalysis | dict | None = None
    disclaimer: str


class DocumentAnalysisResponse(BaseModel):
    file_name: str
    file_type: str
    analysis: dict[str, Any]


class EvidenceAnalysisResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    overall_strength: str = "UNKNOWN"
    gaps: list[str] = Field(default_factory=list)
    evidence_summary: str | None = None


class CaseMemoryResponse(BaseModel):
    case_id: str
    facts_summary: str | None = None
    jurisdiction: str = "TH"
    status: str = "active"
    irac_count: int = 0
    key_citations_count: int = 0


class TimelineEntry(BaseModel):
    ts: int
    question: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    supabase: bool
    redis: bool
    version: str
