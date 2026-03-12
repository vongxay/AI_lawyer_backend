from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LegalQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    case_id: str | None = None


class CitationItem(BaseModel):
    ref: str
    status: Literal["VERIFIED", "OUTDATED", "UNVERIFIED", "REJECTED"] = "UNVERIFIED"
    source_links: list[str] = []


class IracIssue(BaseModel):
    primary: str
    secondary: list[str] = []


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
    statutes: list[IracStatute] = []
    precedents: list[IracPrecedent] = []


class IracApplication(BaseModel):
    analysis: str
    strengths: list[str] = []
    weaknesses: list[str] = []
    counter_args: list[str] = []
    rebuttals: list[str] = []


class IracConclusion(BaseModel):
    recommendation: str
    action_steps: list[str] = []
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    win_probability: float = 0.5
    settlement_note: str | None = None


class IracPayload(BaseModel):
    issue: IracIssue
    rule: IracRule
    application: IracApplication
    conclusion: IracConclusion


class LegalQueryResponse(BaseModel):
    irac: IracPayload
    citations_verified: bool = True
    citations: list[CitationItem] = []
    confidence: float = 0.75
    agents_used: list[str] = []
    processing_time_ms: int = 0
    escalated_to_expert: bool = False
    disclaimer: str
    risk: dict | None = None


class DraftRequest(BaseModel):
    prompt: str
    jurisdiction: str | None = None


class VerifyCitationsRequest(BaseModel):
    citations: list[CitationItem]


class EvidenceAnalyzeResponse(BaseModel):
    items: list[dict] = []
    overall_strength: str = "UNKNOWN"
    gaps: list[str] = []

