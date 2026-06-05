"""
orchestrator/query_classifier.py
==================================
Query classifier — determines which agent plan to execute.

Simple rule-based classifier for v1. Can be replaced with a fine-tuned
LLM classifier (GPT-4o-mini) in production for higher accuracy.
"""
from __future__ import annotations

from typing import Any, Literal

from core.logging import get_logger
from orchestrator.legal_intent_router import LegalIntentRoute, LegalIntentRouter

log = get_logger(__name__)

QueryType = Literal[
    "conversation",
    "legal_question",
    "document_review",
    "case_strategy",
    "evidence_analysis",
    "draft_document",
    "clarification",
]

_LAO_LEGAL_MARKERS = (
    "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",  # law
    "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",      # article
    "\u0eaa\u0eb4\u0e94",                  # right
    "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",  # land
    "\u0e94\u0eb4\u0e99",                  # land/soil
    "\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9\u0e94\u0eb4\u0e99",  # land use
    "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e94\u0eb4\u0e99",        # land use
    "\u0eaa\u0eb1\u0e99\u0e8d\u0eb2",      # contract
    "\u0e84\u0eb0\u0e94\u0eb5",            # case
    "\u0e9f\u0ec9\u0ead\u0e87",            # sue
    "\u0eab\u0ebc\u0eb1\u0e81\u0e96\u0eb2\u0e99",  # evidence
    "\u0ec0\u0ead\u0e81\u0eb0\u0eaa\u0eb2\u0e99",  # document
    "\u0ec2\u0e81\u0e87",                  # fraud
    "\u0ec0\u0eaa\u0e8d\u0eab\u0eb2\u0e8d",  # damage
)

_THAI_LEGAL_MARKERS = (
    "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22",
    "\u0e21\u0e32\u0e15\u0e23\u0e32",
    "\u0e2a\u0e34\u0e17\u0e18\u0e34",
    "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19",
    "\u0e2a\u0e31\u0e0d\u0e0d\u0e32",
    "\u0e04\u0e14\u0e35",
    "\u0e1f\u0e49\u0e2d\u0e07",
    "\u0e2b\u0e25\u0e31\u0e01\u0e10\u0e32\u0e19",
    "\u0e40\u0e2d\u0e01\u0e2a\u0e32\u0e23",
    "\u0e42\u0e01\u0e07",
    "\u0e40\u0e2a\u0e35\u0e22\u0e2b\u0e32\u0e22",
    "\u0e19\u0e32\u0e22\u0e08\u0e49\u0e32\u0e07",
    "\u0e25\u0e39\u0e01\u0e08\u0e49\u0e32\u0e07",
)

_ENGLISH_LEGAL_MARKERS = (
    "law",
    "legal",
    "article",
    "section",
    "right",
    "rights",
    "land",
    "contract",
    "case",
    "sue",
    "evidence",
    "document",
    "fraud",
    "damage",
    "employer",
    "employee",
)

_CONVERSATION_MARKERS = (
    "\u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0e94\u0eb5",  # Lao greeting
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
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "thanks",
    "thank you",
    "who are you",
    "what can you do",
    "remember",
)

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
    "\u0e08\u0ec9\u0eb2",
    "\u0ec2\u0ead\u0ec0\u0e84",
}

# ── Keyword sets per type (Thai + English) ─────────────────────────────────────
_KEYWORDS: dict[QueryType, list[str]] = {
    "document_review": [
        "สัญญา", "contract", "clause", "agreement", "ข้อสัญญา", "ตรวจสัญญา",
        "เอกสาร", "document review", "หนังสือบริคณห์", "lease agreement", "ข้อตกลง",
    ],
    "evidence_analysis": [
        "หลักฐาน", "evidence", "รูปภาพ", "เสียง", "audio", "video", "วิดีโอ",
        "screenshot", "อีเมล", "email", "พยาน", "witness", "ภาพถ่าย",
    ],
    "case_strategy": [
        "กลยุทธ์", "strategy", "ชนะคดี", "win probability", "โอกาสชนะ",
        "ควรฟ้อง", "should i sue", "เจรจา", "settle", "ประนีประนอม",
        "แนวทาง", "approach", "วางแผน", "plan case",
    ],
    "draft_document": [
        "ร่าง", "draft", "เขียน", "write", "หนังสือ", "notice",
        "จดหมาย", "letter", "สัญญาเช่า", "ใบมอบอำนาจ", "power of attorney",
        "บันทึกข้อตกลง", "mou", "สัญญาจ้าง",
    ],
}


class QueryClassifier:
    def __init__(self) -> None:
        self._router = LegalIntentRouter()

    def route(
        self,
        text: str,
        *,
        query_mode: str = "general",
        has_document: bool = False,
        has_evidence: bool = False,
        memory: dict[str, Any] | None = None,
    ) -> LegalIntentRoute:
        clean = (text or "").strip()
        lowered = clean.casefold()
        legacy_type = self._legacy_keyword_match(lowered)
        route = self._router.route(
            clean,
            query_mode=query_mode,
            has_document=has_document,
            has_evidence=has_evidence,
            memory=memory,
            forced_query_type=legacy_type,
        )
        log.debug(
            "classifier.routed",
            query_type=route.query_type,
            legal_domain=route.legal_domain,
            issue_type=route.issue_type,
            needs_clarification=route.needs_clarification,
            reason=route.reason,
        )
        return route

    async def classify(self, text: str) -> QueryType:
        return self.route(text).query_type  # type: ignore[return-value]

    def _legacy_keyword_match(self, lowered: str) -> QueryType | None:
        if not lowered:
            return None

        for query_type, keywords in _KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                log.debug("classifier.matched", query_type=query_type)
                return query_type
        return None

    def _looks_legal(self, lowered: str) -> bool:
        return any(marker in lowered for marker in (*_LAO_LEGAL_MARKERS, *_THAI_LEGAL_MARKERS, *_ENGLISH_LEGAL_MARKERS))

    def _looks_conversational(self, clean: str, lowered: str) -> bool:
        if not clean:
            return True
        if lowered in _SHORT_CONVERSATION_REPLIES:
            return True
        if len(clean) <= 32 and any(marker in lowered for marker in _CONVERSATION_MARKERS):
            return True
        return any(marker in lowered for marker in _CONVERSATION_MARKERS)
