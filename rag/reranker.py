"""
Deterministic reranker for legal RAG.

This is a production-safe baseline while a managed cross-encoder/rerank model is
not configured. It favours statutory authority, official sources, clean text,
and chunks with section/article metadata.
"""
from __future__ import annotations

import re
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


class Reranker:
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        scored: list[tuple[dict[str, Any], float]] = []
        for chunk in chunks:
            score = self._score(query, chunk)
            scored.append(({**chunk, "_rerank_score": round(score, 4)}, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [chunk for chunk, _ in scored[:top_k]]

        log.debug("reranker.done", input=len(chunks), output=len(result))
        return result

    def _score(self, query: str, chunk: dict[str, Any]) -> float:
        base_score = self._safe_float(chunk.get("final_score"), default=0.25)
        keyword_boost = self._keyword_boost(query, chunk)
        authority_boost = self._authority_boost(chunk)
        structure_boost = self._structure_boost(chunk)
        quality_penalty = self._quality_penalty(chunk)
        graph_boost = 0.08 if chunk.get("type") == "precedent" else 0.0
        return base_score + keyword_boost + authority_boost + structure_boost + graph_boost - quality_penalty

    def _keyword_boost(self, query: str, chunk: dict[str, Any]) -> float:
        content = " ".join(
            str(value or "")
            for value in (
                chunk.get("title"),
                chunk.get("section"),
                chunk.get("section_ref"),
                chunk.get("content"),
            )
        ).casefold()
        terms = self._query_terms(query)
        if not terms:
            return 0.0

        matches = 0
        for term in terms:
            if term.isascii():
                if re.search(rf"\b{re.escape(term)}\b", content):
                    matches += 1
            elif term in content:
                matches += 1
        return min(0.35, matches * 0.045)

    def _query_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        for token in query.casefold().split():
            cleaned = token.strip(".,;:()[]{}\"'!?")
            if len(cleaned) >= 2:
                terms.append(cleaned)
        return list(dict.fromkeys(terms))

    def _authority_boost(self, chunk: dict[str, Any]) -> float:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        values = " ".join(
            str(value or "")
            for value in (
                chunk.get("source_url"),
                chunk.get("official_source_url"),
                chunk.get("source_authority"),
                metadata.get("source_url"),
                metadata.get("official_source_url"),
                metadata.get("source_authority"),
            )
        ).casefold()
        source_table = str(chunk.get("source_table") or "").casefold()
        doc_type = str(chunk.get("doc_type") or chunk.get("document_type") or chunk.get("type") or "").casefold()
        jurisdiction = str(chunk.get("jurisdiction") or metadata.get("jurisdiction") or "").casefold()

        boost = 0.0
        if "laoofficialgazette.gov.la" in values:
            boost += 0.35
        elif "official" in values:
            boost += 0.2
        if jurisdiction in {"laos", "la", "lao pdr"} and doc_type in {"law", "statute", "regulation", "decree"}:
            boost += 0.18
        if source_table == "laws":
            boost += 0.14
        if source_table == "cases" and doc_type in {"law", "statute", "regulation"}:
            boost -= 0.08
        return min(boost, 0.45)

    def _structure_boost(self, chunk: dict[str, Any]) -> float:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        section = str(chunk.get("section") or chunk.get("section_ref") or metadata.get("article") or "")
        title = str(chunk.get("title") or "")
        boost = 0.0
        if section.strip():
            boost += 0.12
        if any(word in section.casefold() for word in ("article", "section", "sec")):
            boost += 0.05
        if "\u0ea1\u0eb2\u0e94\u0e95\u0eb2" in section or "\u0e21\u0e32\u0e15\u0e23\u0e32" in section:
            boost += 0.05
        if title.strip():
            boost += 0.04
        return min(boost, 0.2)

    def _quality_penalty(self, chunk: dict[str, Any]) -> float:
        text = str(chunk.get("content") or "")
        if not text.strip():
            return 0.35
        sample = text[:1600]
        chars = [ch for ch in sample if not ch.isspace()]
        if len(chars) < 40:
            return 0.2
        suspicious = sum(1 for ch in chars if ch == "\ufffd" or 0x00C0 <= ord(ch) <= 0x00FF)
        suspicious_ratio = suspicious / len(chars)
        if suspicious_ratio > 0.45:
            return 0.45
        if suspicious_ratio > 0.25:
            return 0.25
        if suspicious_ratio > 0.12:
            return 0.1
        return 0.0

    def _safe_float(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
