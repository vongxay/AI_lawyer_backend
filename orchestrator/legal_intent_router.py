"""
orchestrator/legal_intent_router.py
====================================
Structured legal intent routing for the AI lawyer workflow.

This router is intentionally deterministic. It decides whether the request is
ordinary conversation, a legal research question, a document/evidence task, or
an under-specified legal matter that should ask clarifying questions before RAG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.jurisdiction import contains_lao_script, contains_thai_script, infer_response_language


@dataclass
class LegalIntentRoute:
    query_type: str
    confidence: float
    legal_domain: str | None
    issue_type: str | None
    needs_clarification: bool
    clarification_questions: list[str]
    recommended_tools: list[str]
    reason: str
    should_use_rag: bool
    response_style: str = "plain"
    language: str = "en"
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "confidence": round(self.confidence, 3),
            "legal_domain": self.legal_domain,
            "issue_type": self.issue_type,
            "needs_clarification": self.needs_clarification,
            "clarification_questions": self.clarification_questions,
            "recommended_tools": self.recommended_tools,
            "reason": self.reason,
            "should_use_rag": self.should_use_rag,
            "response_style": self.response_style,
            "language": self.language,
            "signals": self.signals,
        }


_SHORT_CONVERSATION_REPLIES = {
    "ok",
    "okay",
    "alright",
    "hi",
    "hey",
    "hello",
    "\u0e42\u0e2d\u0e40\u0e04",
    "\u0e04\u0e23\u0e31\u0e1a",
    "\u0e04\u0e48\u0e30",
    "\u0ec2\u0ead\u0ec0\u0e84",
    "\u0ec0\u0e88\u0ebb\u0ec9\u0eb2",
}

_CONVERSATION_MARKERS = (
    "\u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0e94\u0eb5",
    "\u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0e94\u0eb5\u0e9a\u0ecd",
    "\u0e82\u0ead\u0e9a\u0ec3\u0e88",
    "\u0ec0\u0e88\u0ebb\u0ec9\u0eb2\u0ec1\u0ea1\u0ec8\u0e99\u0ec3\u0e9c",
    "\u0e88\u0eb7\u0ec8\u0ec4\u0e94\u0ec9\u0e9a\u0ecd",
    "\u0eaa\u0eb2\u0ea1\u0eb2\u0e94\u0e8a\u0ec8\u0ea7\u0e8d",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
    "\u0e2a\u0e1a\u0e32\u0e22\u0e14\u0e35",
    "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13",
    "\u0e04\u0e38\u0e13\u0e04\u0e37\u0e2d\u0e43\u0e04\u0e23",
    "\u0e08\u0e33\u0e44\u0e14\u0e49\u0e44\u0e2b\u0e21",
    "\u0e17\u0e33\u0e2d\u0e30\u0e44\u0e23\u0e44\u0e14\u0e49\u0e1a\u0e49\u0e32\u0e07",
    "hello",
    "hi",
    "hey",
    "how are you",
    "thanks",
    "thank you",
    "who are you",
    "what can you do",
    "remember",
)

_LEGAL_MARKERS = (
    "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
    "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
    "\u0eaa\u0eb4\u0e94",
    "\u0e84\u0eb0\u0e94\u0eb5",
    "\u0e9f\u0ec9\u0ead\u0e87",
    "\u0eab\u0ebc\u0eb1\u0e81\u0e96\u0eb2\u0e99",
    "\u0ec0\u0ead\u0e81\u0eb0\u0eaa\u0eb2\u0e99",
    "\u0ec2\u0e81\u0e87",
    "\u0ec0\u0eaa\u0e8d\u0eab\u0eb2\u0e8d",
    "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22",
    "\u0e21\u0e32\u0e15\u0e23\u0e32",
    "\u0e2a\u0e34\u0e17\u0e18\u0e34",
    "\u0e04\u0e14\u0e35",
    "\u0e1f\u0e49\u0e2d\u0e07",
    "\u0e2b\u0e25\u0e31\u0e01\u0e10\u0e32\u0e19",
    "\u0e40\u0e2d\u0e01\u0e2a\u0e32\u0e23",
    "\u0e42\u0e01\u0e07",
    "\u0e40\u0e2a\u0e35\u0e22\u0e2b\u0e32\u0e22",
    "law",
    "legal",
    "article",
    "section",
    "right",
    "rights",
    "case",
    "sue",
    "evidence",
    "document",
    "fraud",
    "damage",
)

_DOMAIN_MARKERS: dict[str, tuple[str, ...]] = {
    "land": (
        "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
        "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
        "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99\u0ead\u0eb8\u0e94\u0eaa\u0eb2\u0eab\u0eb0\u0e81\u0eb3",
        "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19",
        "land",
    ),
    "contract": (
        "\u0eaa\u0eb1\u0e99\u0e8d\u0eb2",
        "\u0e82\u0ecd\u0ec9\u0e95\u0ebb\u0e81\u0ea5\u0ebb\u0e87",
        "\u0e2a\u0e31\u0e0d\u0e0d\u0e32",
        "\u0e02\u0e49\u0e2d\u0e15\u0e01\u0e25\u0e07",
        "contract",
        "agreement",
    ),
    "labor": (
        "\u0ec1\u0eae\u0e87\u0e87\u0eb2\u0e99",
        "\u0e99\u0eb2\u0e8d\u0e88\u0ec9\u0eb2\u0e87",
        "\u0ea5\u0eb9\u0e81\u0e88\u0ec9\u0eb2\u0e87",
        "\u0e40\u0e25\u0e34\u0e01\u0e08\u0e49\u0e32\u0e07",
        "\u0e19\u0e32\u0e22\u0e08\u0e49\u0e32\u0e07",
        "\u0e25\u0e39\u0e01\u0e08\u0e49\u0e32\u0e07",
        "employer",
        "employee",
        "labor",
        "labour",
    ),
    "family": (
        "\u0e84\u0ead\u0e9a\u0e84\u0ebb\u0ea7",
        "\u0ec1\u0e95\u0ec8\u0e87\u0e87\u0eb2\u0e99",
        "\u0ea1\u0ecd\u0ea5\u0eb0\u0e94\u0ebb\u0e81",
        "\u0e04\u0e23\u0e2d\u0e1a\u0e04\u0e23\u0e31\u0e27",
        "\u0e41\u0e15\u0e48\u0e07\u0e07\u0e32\u0e19",
        "\u0e21\u0e23\u0e14\u0e01",
        "family",
        "marriage",
        "inheritance",
    ),
    "criminal": (
        "\u0ec2\u0e81\u0e87",
        "\u0eaa\u0ecd\u0ec9\u0ec2\u0e81\u0e87",
        "\u0ea5\u0eb1\u0e81",
        "\u0ead\u0eb2\u0e8d\u0eb2",
        "\u0e42\u0e01\u0e07",
        "\u0e09\u0e49\u0e2d\u0e42\u0e01\u0e07",
        "\u0e25\u0e31\u0e01",
        "\u0e2d\u0e32\u0e0d\u0e32",
        "fraud",
        "crime",
        "criminal",
    ),
    "company": (
        "\u0e9a\u0ecd\u0ea5\u0eb4\u0eaa\u0eb1\u0e94",
        "\u0e97\u0eb8\u0ea5\u0eb0\u0e81\u0eb4\u0e94",
        "\u0e1a\u0e23\u0e34\u0e29\u0e31\u0e17",
        "\u0e18\u0e38\u0e23\u0e01\u0e34\u0e08",
        "company",
        "business",
    ),
    "tax": (
        "\u0ead\u0eb2\u0e81\u0ead\u0e99",
        "\u0e9e\u0eb2\u0eaa\u0eb5",
        "\u0e20\u0e32\u0e29\u0e35",
        "\u0e2d\u0e32\u0e01\u0e23",
        "tax",
    ),
    "lease": (
        "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
        "\u0e40\u0e0a\u0e48\u0e32",
        "lease",
        "rent",
    ),
}

_STATUTE_MARKERS = (
    "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
    "\u0e21\u0e32\u0e15\u0e23\u0e32",
    "article",
    "section",
)

_RIGHTS_MARKERS = (
    "\u0eaa\u0eb4\u0e94",
    "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87",
    "\u0e2a\u0e34\u0e17\u0e18\u0e34",
    "\u0e1b\u0e01\u0e1b\u0e49\u0e2d\u0e07",
    "right",
    "rights",
    "protected",
)

_PROCEDURE_MARKERS = (
    "\u0e95\u0ec9\u0ead\u0e87\u0ec0\u0eae\u0eb1\u0e94\u0ec1\u0e99\u0ea7\u0ec3\u0e94",
    "\u0e82\u0eb1\u0ec9\u0e99\u0e95\u0ead\u0e99",
    "\u0eae\u0ec9\u0ead\u0e87\u0e97\u0eb8\u0e81",
    "\u0e15\u0e49\u0e2d\u0e07\u0e17\u0e33\u0e22\u0e31\u0e07\u0e44\u0e07",
    "\u0e17\u0e33\u0e22\u0e31\u0e07\u0e44\u0e07",
    "\u0e02\u0e31\u0e49\u0e19\u0e15\u0e2d\u0e19",
    "\u0e41\u0e08\u0e49\u0e07\u0e04\u0e27\u0e32\u0e21",
    "what should i do",
    "how to",
    "procedure",
    "complaint",
)

_STRATEGY_MARKERS = (
    "\u0e84\u0ea7\u0e99\u0e9f\u0ec9\u0ead\u0e87",
    "\u0ec0\u0e88\u0ea5\u0eb0\u0e88\u0eb2",
    "\u0ec2\u0ead\u0e81\u0eb2\u0e94\u0e8a\u0eb0\u0e99\u0eb0",
    "\u0e04\u0e27\u0e23\u0e1f\u0e49\u0e2d\u0e07",
    "\u0e40\u0e08\u0e23\u0e08\u0e32",
    "\u0e42\u0e2d\u0e01\u0e32\u0e2a\u0e0a\u0e19\u0e30",
    "should i sue",
    "settle",
    "win probability",
    "strategy",
)

_DOCUMENT_MARKERS = (
    "\u0ec0\u0ead\u0e81\u0eb0\u0eaa\u0eb2\u0e99",
    "\u0e95\u0ea7\u0e94",
    "\u0eaa\u0eb1\u0e99\u0e8d\u0eb2",
    "\u0e40\u0e2d\u0e01\u0e2a\u0e32\u0e23",
    "\u0e15\u0e23\u0e27\u0e08",
    "\u0e2a\u0e31\u0e0d\u0e0d\u0e32",
    "document",
    "review",
    "contract",
)

_EVIDENCE_MARKERS = (
    "\u0eab\u0ebc\u0eb1\u0e81\u0e96\u0eb2\u0e99",
    "\u0e9e\u0eb0\u0e8d\u0eb2\u0e99",
    "\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
    "\u0e2b\u0e25\u0e31\u0e01\u0e10\u0e32\u0e19",
    "\u0e1e\u0e22\u0e32\u0e19",
    "\u0e41\u0e0a\u0e17",
    "evidence",
    "witness",
    "screenshot",
    "chat",
)

_DRAFT_MARKERS = (
    "\u0eae\u0ec8\u0eb2\u0e87",
    "\u0e82\u0ebd\u0e99",
    "\u0e23\u0e48\u0e32\u0e07",
    "\u0e40\u0e02\u0e35\u0e22\u0e19",
    "draft",
    "write",
)

_PERSONAL_CASE_MARKERS = (
    "\u0e82\u0ec9\u0ead\u0e8d",
    "\u0e9c\u0ebb\u0ea1",
    "\u0e82\u0ead\u0e87\u0e82\u0ec9\u0ead\u0e8d",
    "\u0e1c\u0e21",
    "\u0e09\u0e31\u0e19",
    "\u0e02\u0e2d\u0e07\u0e1c\u0e21",
    "\u0e02\u0e2d\u0e07\u0e09\u0e31\u0e19",
    "i ",
    "my ",
    "me ",
)

_FACT_DETAIL_MARKERS = (
    "\u0eaa\u0eb1\u0e99\u0e8d\u0eb2",
    "\u0eab\u0ebc\u0eb1\u0e81\u0e96\u0eb2\u0e99",
    "\u0ea7\u0eb1\u0e99\u0e97\u0eb5",
    "\u0e88\u0eb3\u0e99\u0ea7\u0e99",
    "\u0e81\u0eb5\u0e9a",
    "\u0e2a\u0e31\u0e0d\u0e0d\u0e32",
    "\u0e2b\u0e25\u0e31\u0e01\u0e10\u0e32\u0e19",
    "\u0e27\u0e31\u0e19\u0e17\u0e35\u0e48",
    "\u0e08\u0e33\u0e19\u0e27\u0e19",
    "\u0e01\u0e35\u0e1a",
    "contract",
    "evidence",
    "date",
    "amount",
    "kip",
)


class LegalIntentRouter:
    def route(
        self,
        text: str,
        *,
        query_mode: str = "general",
        has_document: bool = False,
        has_evidence: bool = False,
        memory: dict[str, Any] | None = None,
        forced_query_type: str | None = None,
    ) -> LegalIntentRoute:
        clean = (text or "").strip()
        lowered = clean.casefold()
        language = infer_response_language(clean)
        memory_used = bool(memory and not memory.get("empty", True))

        domain = self._detect_domain(lowered)
        issue_type = self._detect_issue_type(lowered)
        signals = {
            "has_legal_marker": self._has_any(lowered, _LEGAL_MARKERS),
            "has_conversation_marker": self._has_any(lowered, _CONVERSATION_MARKERS),
            "has_statute_marker": self._has_any(lowered, _STATUTE_MARKERS),
            "has_personal_case_marker": self._has_any(lowered, _PERSONAL_CASE_MARKERS),
            "has_fact_detail_marker": self._has_any(lowered, _FACT_DETAIL_MARKERS) or bool(re.search(r"\d", clean)),
            "character_count": len(clean),
            "memory_used": memory_used,
        }

        mode_route = self._route_for_mode(
            query_mode=query_mode,
            has_document=has_document,
            has_evidence=has_evidence,
            domain=domain,
            issue_type=issue_type,
            language=language,
            signals=signals,
        )
        if mode_route:
            return mode_route

        if forced_query_type:
            return self._route_for_query_type(
                forced_query_type,
                domain=domain,
                issue_type=issue_type,
                language=language,
                reason="legacy_keyword_match",
                signals=signals,
            )

        if not clean or lowered in _SHORT_CONVERSATION_REPLIES:
            return self._conversation(language, "short_or_empty_conversation", signals)

        looks_legal = bool(signals["has_legal_marker"] or domain or issue_type)
        looks_conversation = bool(signals["has_conversation_marker"])
        if looks_conversation and not looks_legal:
            return self._conversation(language, "conversation_marker_without_legal_signal", signals)

        if self._needs_clarification(clean, lowered, domain, issue_type, signals, memory_used):
            return LegalIntentRoute(
                query_type="clarification",
                confidence=0.88,
                legal_domain=domain,
                issue_type=issue_type or "case_facts",
                needs_clarification=True,
                clarification_questions=self._clarification_questions(language),
                recommended_tools=["ask_clarifying_questions"],
                reason="legal_case_under_specified",
                should_use_rag=False,
                response_style="plain",
                language=language,
                signals=signals,
            )

        if issue_type == "document_review":
            return self._route_for_query_type("document_review", domain, issue_type, language, "document_review_signal", signals)
        if issue_type == "evidence_analysis":
            return self._route_for_query_type("evidence_analysis", domain, issue_type, language, "evidence_signal", signals)
        if issue_type == "draft_document":
            return self._route_for_query_type("draft_document", domain, issue_type, language, "draft_signal", signals)
        if issue_type == "case_strategy":
            return self._route_for_query_type("case_strategy", domain, issue_type, language, "strategy_signal", signals)

        return LegalIntentRoute(
            query_type="legal_question",
            confidence=0.82 if looks_legal else 0.62,
            legal_domain=domain,
            issue_type=issue_type or "general_legal",
            needs_clarification=False,
            clarification_questions=[],
            recommended_tools=["legal_research", "irac_reasoning", "citation_verification"],
            reason="legal_question_rag_required" if looks_legal else "default_legal_fallback",
            should_use_rag=True,
            response_style="plain",
            language=language,
            signals=signals,
        )

    def _route_for_mode(
        self,
        *,
        query_mode: str,
        has_document: bool,
        has_evidence: bool,
        domain: str | None,
        issue_type: str | None,
        language: str,
        signals: dict[str, Any],
    ) -> LegalIntentRoute | None:
        if has_evidence or query_mode == "evidence":
            return self._route_for_query_type("evidence_analysis", domain, "evidence_analysis", language, "evidence_mode", signals)
        if has_document or query_mode == "document":
            return self._route_for_query_type("document_review", domain, "document_review", language, "document_mode", signals)
        if query_mode == "draft":
            return self._route_for_query_type("draft_document", domain, "draft_document", language, "draft_mode", signals)
        if query_mode == "serious_case":
            return self._route_for_query_type("case_strategy", domain, issue_type or "case_strategy", language, "serious_case_mode", signals)
        return None

    def _route_for_query_type(
        self,
        query_type: str,
        domain: str | None,
        issue_type: str | None,
        language: str,
        reason: str,
        signals: dict[str, Any],
    ) -> LegalIntentRoute:
        tools = ["legal_research", "irac_reasoning", "citation_verification"]
        style = "plain"
        if query_type == "document_review":
            tools.append("document_analysis")
            style = "irac"
            issue_type = issue_type or "document_review"
        elif query_type == "evidence_analysis":
            tools.append("evidence_analysis")
            style = "action_plan"
            issue_type = issue_type or "evidence_analysis"
        elif query_type == "case_strategy":
            tools.append("risk_strategy")
            style = "action_plan"
            issue_type = issue_type or "case_strategy"
        elif query_type == "draft_document":
            tools.append("document_drafting")
            style = "irac"
            issue_type = issue_type or "draft_document"
        return LegalIntentRoute(
            query_type=query_type,
            confidence=0.9,
            legal_domain=domain,
            issue_type=issue_type,
            needs_clarification=False,
            clarification_questions=[],
            recommended_tools=tools,
            reason=reason,
            should_use_rag=True,
            response_style=style,
            language=language,
            signals=signals,
        )

    def _conversation(self, language: str, reason: str, signals: dict[str, Any]) -> LegalIntentRoute:
        return LegalIntentRoute(
            query_type="conversation",
            confidence=0.96,
            legal_domain=None,
            issue_type="small_talk",
            needs_clarification=False,
            clarification_questions=[],
            recommended_tools=[],
            reason=reason,
            should_use_rag=False,
            response_style="plain",
            language=language,
            signals=signals,
        )

    def _detect_domain(self, lowered: str) -> str | None:
        matches: list[tuple[str, int]] = []
        for domain, markers in _DOMAIN_MARKERS.items():
            score = sum(1 for marker in markers if marker in lowered)
            if score:
                matches.append((domain, score))
        if not matches:
            return None
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[0][0]

    def _detect_issue_type(self, lowered: str) -> str | None:
        if self._has_any(lowered, _STATUTE_MARKERS):
            return "statute_lookup"
        if self._has_any(lowered, _RIGHTS_MARKERS):
            return "rights_explanation"
        if self._has_any(lowered, _DRAFT_MARKERS):
            return "draft_document"
        if self._has_any(lowered, _EVIDENCE_MARKERS):
            return "evidence_analysis"
        if self._has_any(lowered, _DOCUMENT_MARKERS):
            return "document_review"
        if self._has_any(lowered, _STRATEGY_MARKERS):
            return "case_strategy"
        if self._has_any(lowered, _PROCEDURE_MARKERS):
            return "procedure"
        return None

    def _needs_clarification(
        self,
        clean: str,
        lowered: str,
        domain: str | None,
        issue_type: str | None,
        signals: dict[str, Any],
        memory_used: bool,
    ) -> bool:
        if memory_used:
            return False
        if issue_type in {"statute_lookup", "rights_explanation", "document_review", "evidence_analysis", "draft_document"}:
            return False
        if signals["has_statute_marker"]:
            return False
        if domain == "land" and issue_type in {None, "procedure"} and not signals["has_personal_case_marker"]:
            return False

        is_personal_case = bool(signals["has_personal_case_marker"])
        asks_action = issue_type in {"procedure", "case_strategy"} or self._has_any(lowered, _PROCEDURE_MARKERS)
        has_case_harm = domain in {"criminal", "contract", "labor"} or self._has_any(
            lowered,
            (
                "\u0ec2\u0e81\u0e87",
                "\u0e42\u0e01\u0e07",
                "\u0ec0\u0eaa\u0e8d\u0eab\u0eb2\u0e8d",
                "\u0e40\u0e2a\u0e35\u0e22\u0e2b\u0e32\u0e22",
                "fraud",
                "damage",
            ),
        )
        has_detail = bool(signals["has_fact_detail_marker"])
        short_case = len(clean) < 120
        return bool((is_personal_case or has_case_harm) and asks_action and short_case and not has_detail)

    def _clarification_questions(self, language: str) -> list[str]:
        if language == "lo":
            return [
                "ເຫດການເກີດຂຶ້ນເມື່ອໃດ ແລະ ຢູ່ແຂວງ/ເມືອງໃດ?",
                "ມີເອກະສານ ຫຼື ຫຼັກຖານໃດແດ່?",
                "ຄູ່ກໍລະນີແມ່ນໃຜ ແລະ ກ່ຽວຂ້ອງກັບທ່ານແນວໃດ?",
                "ທ່ານຕ້ອງການຜົນລັບໃດ: ເຈລະຈາ, ຮ້ອງທຸກ, ຟ້ອງຄະດີ ຫຼື ຮ່າງເອກະສານ?",
            ]
        if language == "th":
            return [
                "เหตุการณ์เกิดขึ้นเมื่อไร และเกิดที่แขวง/เมืองใดในลาว?",
                "มีเอกสารหรือหลักฐานอะไรบ้าง เช่น สัญญา แชท ใบเสร็จ หรือพยาน?",
                "คู่กรณีคือใคร และเกี่ยวข้องกับคุณอย่างไร?",
                "คุณต้องการผลลัพธ์แบบไหน: เจรจา แจ้งความ ฟ้องคดี หรือร่างเอกสาร?",
            ]
        return [
            "When did the event happen, and in which Lao province or district?",
            "What documents or evidence do you have, such as a contract, chat records, receipts, or witnesses?",
            "Who is the other party, and what is their relationship to you?",
            "What outcome do you want: negotiation, complaint, lawsuit, or a drafted document?",
        ]

    def _has_any(self, lowered: str, markers: tuple[str, ...]) -> bool:
        return any(marker in lowered for marker in markers)
