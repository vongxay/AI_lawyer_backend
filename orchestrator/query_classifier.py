from __future__ import annotations

from typing import Literal


QueryType = Literal[
    "legal_question",
    "document_review",
    "case_strategy",
    "evidence_analysis",
    "draft_document",
]


class QueryClassifier:
    async def classify(self, text: str) -> QueryType:
        lowered = text.lower()
        if any(k in lowered for k in ["สัญญา", "contract", "clause", "agreement"]):
            return "document_review"
        if any(k in lowered for k in ["หลักฐาน", "evidence", "รูป", "เสียง", "audio", "video"]):
            return "evidence_analysis"
        if any(k in lowered for k in ["กลยุทธ์", "strategy", "ชนะคดี", "win probability"]):
            return "case_strategy"
        if any(k in lowered for k in ["ร่าง", "draft", "หนังสือ", "notice"]):
            return "draft_document"
        return "legal_question"

