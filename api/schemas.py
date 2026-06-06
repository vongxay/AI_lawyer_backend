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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


ChatQueryMode = Literal["general", "serious_case", "evidence", "document", "draft"]
ResponseStyle = Literal["plain", "irac", "action_plan"]
UrgencyLevel = Literal["normal", "urgent", "critical"]


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
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    question: str = Field(
        min_length=3,
        max_length=5000,
        validation_alias=AliasChoices("question", "query"),
    )
    case_id: str | None = None
    session_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "sessionId"),
    )
    jurisdiction: str | None = None
    model_id: str | None = None
    query_mode: ChatQueryMode = Field(
        default="general",
        validation_alias=AliasChoices("query_mode", "queryMode", "mode"),
    )
    response_style: ResponseStyle = Field(
        default="irac",
        validation_alias=AliasChoices("response_style", "responseStyle"),
    )
    urgency: UrgencyLevel = "normal"
    include_irac: bool = True
    include_citations: bool = True

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class ChatSessionCreate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str | None = Field(default=None, max_length=200)
    legal_case_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("legal_case_id", "case_id", "caseId"),
    )
    query_type: str | None = Field(
        default=None,
        max_length=60,
        validation_alias=AliasChoices("query_type", "queryType"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "query_type")
    @classmethod
    def strip_optional_text(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        return value or None


class ChatSessionUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str | None = Field(default=None, max_length=200)
    status: Literal["active", "closed", "escalated", "archived"] | None = None
    query_type: str | None = Field(
        default=None,
        max_length=60,
        validation_alias=AliasChoices("query_type", "queryType"),
    )
    metadata: dict[str, Any] | None = None

    @field_validator("title", "query_type")
    @classmethod
    def strip_update_text(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        return value or None


class ChatMessageCreate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    role: Literal["user", "assistant", "ai", "system"]
    content: str = Field(min_length=1, max_length=20000)
    irac_output: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("irac_output", "irac"),
    )
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    agents_used: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("agents_used", "agentsUsed"),
    )
    model_used: str | None = Field(
        default=None,
        max_length=120,
        validation_alias=AliasChoices("model_used", "modelUsed"),
    )
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("latency_ms", "processing_time_ms", "processingTime"),
    )
    escalated: bool = False
    escalation_reason: str | None = Field(
        default=None,
        max_length=1000,
        validation_alias=AliasChoices("escalation_reason", "escalationReason"),
    )

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        return v.strip()


class ChatMessageUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    content: str | None = Field(default=None, min_length=1, max_length=20000)
    irac_output: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("irac_output", "irac"),
    )
    citations: list[dict[str, Any]] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    agents_used: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("agents_used", "agentsUsed"),
    )
    model_used: str | None = Field(
        default=None,
        max_length=120,
        validation_alias=AliasChoices("model_used", "modelUsed"),
    )
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("latency_ms", "processing_time_ms", "processingTime"),
    )
    escalated: bool | None = None
    escalation_reason: str | None = Field(
        default=None,
        max_length=1000,
        validation_alias=AliasChoices("escalation_reason", "escalationReason"),
    )

    @field_validator("content")
    @classmethod
    def strip_updated_content(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        return value or None


class CaseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str | None = Field(default=None, max_length=200)
    case_type: str = Field(
        default="general",
        max_length=80,
        validation_alias=AliasChoices("case_type", "type", "caseType"),
    )
    description: str | None = Field(default=None, max_length=5000)
    jurisdiction: str = "laos"

    @field_validator("title", "case_type", "description", "jurisdiction")
    @classmethod
    def strip_case_text(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        return value or None


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    prompt: str = Field(
        min_length=10,
        max_length=3000,
        validation_alias=AliasChoices("prompt", "context"),
    )
    document_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("document_type", "type"),
    )
    jurisdiction: str | None = None
    language: Literal["TH", "EN", "LA"] = "LA"


class VerifyCitationsRequest(BaseModel):
    citations: list[CitationItem] = Field(min_length=1, max_length=50)

    @field_validator("citations", mode="before")
    @classmethod
    def normalise_citations(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [{"ref": item} if isinstance(item, str) else item for item in v]
        return v


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    session_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("session_id", "query_id"),
    )
    message_id: str | None = None
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    corrected_answer: str | None = Field(default=None, max_length=5000)


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)
    document_type: Literal["law", "case", "form", "guideline"] = "law"
    jurisdiction: str = "laos"
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
    answer: str | None = None
    query_type: str | None = None
    query_mode: str | None = None
    response_style: str | None = None
    response_language: str | None = None
    selected_model_id: str | None = None
    session_id: str | None = None
    citations: list[CitationItem] = Field(default_factory=list)
    citations_verified: bool = True
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    agents_used: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    escalated_to_expert: bool = False
    risk: RiskAnalysis | dict | None = None
    document: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    answer_quality: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str


class DocumentAnalysisResponse(BaseModel):
    file_name: str
    file_type: str
    text_length: int | None = None
    analysis: dict[str, Any]


class EvidenceAnalysisResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    overall_strength: str = "UNKNOWN"
    gaps: list[str] = Field(default_factory=list)
    evidence_summary: str | None = None


class CaseRecordResponse(BaseModel):
    id: str
    title: str
    type: str = "general"
    status: Literal["active", "closed", "settled"] = "active"
    created_at: str | None = None
    last_accessed: str | None = None


class CaseMemoryResponse(BaseModel):
    case_id: str
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    irac_history: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    facts_summary: str | None = None
    jurisdiction: str = "laos"
    status: str = "active"
    irac_count: int = 0
    key_citations_count: int = 0


class TimelineEntry(BaseModel):
    id: str
    event: str
    date: str
    type: Literal["query", "evidence", "document", "milestone"] = "query"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    supabase: bool
    redis: bool
    version: str
