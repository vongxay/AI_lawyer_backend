"""
orchestrator/query_classifier.py
==================================
Query classifier — determines which agent plan to execute.

Simple rule-based classifier for v1. Can be replaced with a fine-tuned
LLM classifier (GPT-4o-mini) in production for higher accuracy.
"""
from __future__ import annotations

from typing import Literal

from backend.core.logging import get_logger

log = get_logger(__name__)

QueryType = Literal[
    "legal_question",
    "document_review",
    "case_strategy",
    "evidence_analysis",
    "draft_document",
]

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
    async def classify(self, text: str) -> QueryType:
        lowered = text.lower()

        for query_type, keywords in _KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                log.debug("classifier.matched", query_type=query_type)
                return query_type

        log.debug("classifier.default", query_type="legal_question")
        return "legal_question"
