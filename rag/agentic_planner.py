"""
Agentic retrieval planning for legal RAG.

The planner is intentionally deterministic. It expands a user's question into a
small set of high-signal retrieval queries, then asks for a second pass when the
first pass does not cover the legal authority needed for a grounded answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.jurisdiction import canonical_jurisdiction, contains_lao_script, contains_thai_script, infer_jurisdiction


LAO_LAW_TERMS = (
    "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d "  # law
    "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 "      # article
    "\u0e94\u0eb3\u0ea5\u0eb1\u0e94 "      # decree
    "\u0e84\u0eb3\u0eaa\u0eb1\u0ec8\u0e87"  # order
)
LAO_OFFICIAL_TERMS = "Lao PDR law article decree regulation official gazette"

THAI_LAW_TERMS = (
    "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22 "
    "\u0e21\u0e32\u0e15\u0e23\u0e32 "
    "\u0e1e\u0e23\u0e30\u0e23\u0e32\u0e0a\u0e1a\u0e31\u0e0d\u0e0d\u0e31\u0e15\u0e34"
)


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    purpose: str
    jurisdiction: str | None
    priority: int = 1
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCoverage:
    enough_results: bool
    has_statute: bool
    has_official_source: bool
    has_clean_text: bool
    should_second_pass: bool
    reason: str | None
    metrics: dict[str, Any]


class AgenticRetrievalPlanner:
    """Plans multi-pass retrieval for Lao-first legal research."""

    min_results: int = 5

    def plan(
        self,
        question: str,
        jurisdiction: str | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> list[RetrievalQuery]:
        analysis = analysis or {}
        canonical = infer_jurisdiction(question, jurisdiction) or canonical_jurisdiction(analysis.get("jurisdiction"))
        intent = str(analysis.get("practice_area") or self._classify_intent(question))
        queries = [
            RetrievalQuery(
                query=question,
                purpose="original_user_question",
                jurisdiction=canonical,
                priority=1,
                required=True,
                metadata={"intent": intent, "analysis_confidence": analysis.get("confidence")},
            )
        ]
        queries.extend(self._authority_hint_queries(question, canonical, intent, analysis, priority=0))
        queries.extend(self._analysis_search_phrase_queries(canonical, intent, analysis, priority=1))

        if canonical == "laos":
            queries.extend(self._lao_expansions(question, intent, priority=1))
        elif canonical == "thailand":
            queries.extend(self._thai_expansions(question, intent, priority=1))
        else:
            queries.extend(self._generic_expansions(question, intent, priority=1))

        return self._dedupe(queries)

    def second_pass(
        self,
        question: str,
        jurisdiction: str | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> list[RetrievalQuery]:
        analysis = analysis or {}
        canonical = canonical_jurisdiction(jurisdiction) or infer_jurisdiction(question) or canonical_jurisdiction(analysis.get("jurisdiction"))
        intent = str(analysis.get("practice_area") or self._classify_intent(question))
        hint_queries = self._authority_hint_queries(question, canonical, intent, analysis, priority=2)
        if canonical == "laos":
            return self._dedupe([*hint_queries, *self._lao_expansions(question, intent, priority=2, second_pass=True)])
        if canonical == "thailand":
            return self._dedupe([*hint_queries, *self._thai_expansions(question, intent, priority=2, second_pass=True)])
        return self._dedupe([*hint_queries, *self._generic_expansions(question, intent, priority=2, second_pass=True)])

    def assess_coverage(self, chunks: list[dict[str, Any]], jurisdiction: str | None = None) -> RetrievalCoverage:
        canonical = canonical_jurisdiction(jurisdiction)
        count = len(chunks)
        statute_count = sum(1 for chunk in chunks if self._is_statute_like(chunk))
        official_count = sum(1 for chunk in chunks if self._is_official_source(chunk))
        clean_count = sum(1 for chunk in chunks if self._looks_clean(chunk))
        enough_results = count >= self.min_results
        has_statute = statute_count > 0
        has_official_source = official_count > 0
        has_clean_text = clean_count > 0

        reason: str | None = None
        if not enough_results:
            reason = "too_few_results"
        elif not has_statute:
            reason = "no_statutory_authority"
        elif canonical == "laos" and not has_official_source:
            reason = "no_official_lao_source"
        elif not has_clean_text:
            reason = "retrieved_text_quality_low"

        recoverable_by_second_pass = reason in {
            "too_few_results",
            "no_statutory_authority",
            "no_official_lao_source",
            "retrieved_text_quality_low",
        }

        return RetrievalCoverage(
            enough_results=enough_results,
            has_statute=has_statute,
            has_official_source=has_official_source,
            has_clean_text=has_clean_text,
            should_second_pass=recoverable_by_second_pass,
            reason=reason,
            metrics={
                "count": count,
                "statute_count": statute_count,
                "official_source_count": official_count,
                "clean_text_count": clean_count,
            },
        )

    def should_second_pass(self, chunks: list[dict[str, Any]], min_results: int | None = None) -> bool:
        threshold = self.min_results if min_results is None else min_results
        if len(chunks) < threshold:
            return True
        return not any(self._is_statute_like(chunk) for chunk in chunks)

    def _lao_expansions(
        self,
        question: str,
        intent: str,
        *,
        priority: int,
        second_pass: bool = False,
    ) -> list[RetrievalQuery]:
        domain_terms = self._lao_domain_terms(intent)
        official_terms = f"{LAO_LAW_TERMS} {domain_terms}".strip()
        english_terms = f"{LAO_OFFICIAL_TERMS} {self._english_domain_terms(intent)}".strip()
        section_terms = self._section_terms(question)

        queries = [
            RetrievalQuery(
                query=f"{question} {official_terms}",
                purpose=f"lao_{intent}_statutory_terms",
                jurisdiction="laos",
                priority=priority,
                required=True,
                metadata={"intent": intent, "authority": "statute"},
            ),
            RetrievalQuery(
                query=f"{question} {english_terms}",
                purpose=f"lao_{intent}_english_authority_terms",
                jurisdiction="laos",
                priority=priority,
                metadata={"intent": intent, "authority": "translation"},
            ),
        ]
        if section_terms:
            queries.append(
                RetrievalQuery(
                    query=f"{section_terms} {LAO_LAW_TERMS} {english_terms}",
                    purpose="lao_specific_article_lookup",
                    jurisdiction="laos",
                    priority=priority,
                    required=True,
                    metadata={"intent": intent, "authority": "article"},
                )
            )
        if second_pass:
            queries.append(
                RetrievalQuery(
                    query=f"{self._english_domain_terms(intent)} Lao PDR ministry decree decision instruction regulation",
                    purpose=f"lao_{intent}_authority_second_pass",
                    jurisdiction="laos",
                    priority=priority,
                    metadata={"intent": intent, "authority": "official_secondary"},
                )
            )
        return queries

    def _thai_expansions(
        self,
        question: str,
        intent: str,
        *,
        priority: int,
        second_pass: bool = False,
    ) -> list[RetrievalQuery]:
        domain_terms = self._thai_domain_terms(intent)
        queries = [
            RetrievalQuery(
                query=f"{question} {THAI_LAW_TERMS} {domain_terms}",
                purpose=f"thai_{intent}_statutory_terms",
                jurisdiction="thailand",
                priority=priority,
                required=True,
                metadata={"intent": intent, "authority": "statute"},
            )
        ]
        if second_pass:
            queries.append(
                RetrievalQuery(
                    query=f"{question} Thai law section regulation {self._english_domain_terms(intent)}",
                    purpose=f"thai_{intent}_english_second_pass",
                    jurisdiction="thailand",
                    priority=priority,
                    metadata={"intent": intent, "authority": "translation"},
                )
            )
        return queries

    def _generic_expansions(
        self,
        question: str,
        intent: str,
        *,
        priority: int,
        second_pass: bool = False,
    ) -> list[RetrievalQuery]:
        terms = self._english_domain_terms(intent)
        suffix = "law article statute regulation legal test"
        if second_pass:
            suffix = f"{suffix} official source primary authority"
        return [
            RetrievalQuery(
                query=f"{question} {suffix} {terms}".strip(),
                purpose=f"generic_{intent}_legal_terms",
                jurisdiction=None,
                priority=priority,
                metadata={"intent": intent},
            )
        ]

    def _authority_hint_queries(
        self,
        question: str,
        jurisdiction: str | None,
        intent: str,
        analysis: dict[str, Any],
        *,
        priority: int,
    ) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        hints = analysis.get("authority_hints") if isinstance(analysis.get("authority_hints"), list) else []
        for index, hint in enumerate(hints[:5]):
            if not isinstance(hint, dict):
                continue
            law_name = str(hint.get("law_name") or "").strip()
            article = str(hint.get("article") or "").strip()
            search_terms = hint.get("search_terms") if isinstance(hint.get("search_terms"), list) else []
            terms = " ".join(str(term).strip() for term in search_terms[:6] if str(term).strip())
            query = " ".join(part for part in (question, law_name, article, terms) if part).strip()
            if not query:
                continue
            raw_priority = hint.get("priority")
            hint_priority = 0 if raw_priority is None or raw_priority == "" else max(0, int(raw_priority) - 1)
            queries.append(
                RetrievalQuery(
                    query=query,
                    purpose=f"authority_hint_{index + 1}",
                    jurisdiction=canonical_jurisdiction(hint.get("jurisdiction")) or jurisdiction,
                    priority=priority + hint_priority,
                    required=index == 0 or bool(article),
                    metadata={
                        "intent": intent,
                        "authority": "candidate_authority",
                        "law_name": law_name,
                        "article": article or None,
                        "reason": hint.get("reason"),
                    },
                )
            )
        return queries

    def _analysis_search_phrase_queries(
        self,
        jurisdiction: str | None,
        intent: str,
        analysis: dict[str, Any],
        *,
        priority: int,
    ) -> list[RetrievalQuery]:
        phrases = analysis.get("search_phrases") if isinstance(analysis.get("search_phrases"), list) else []
        queries: list[RetrievalQuery] = []
        for index, phrase in enumerate(phrases[1:4], start=1):
            query = str(phrase).strip()
            if not query:
                continue
            queries.append(
                RetrievalQuery(
                    query=query,
                    purpose=f"legal_issue_search_phrase_{index}",
                    jurisdiction=jurisdiction,
                    priority=priority,
                    metadata={"intent": intent, "authority": "issue_analysis"},
                )
            )
        return queries

    def _classify_intent(self, question: str) -> str:
        lowered = question.casefold()
        markers = {
            "land": (
                "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
                "\u0e81\u0eb3\u0ea1\u0eb0\u0eaa\u0eb4\u0e94",
                "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
                "\u0ead\u0eb0\u0eaa\u0eb1\u0e87\u0eab\u0eb2",
                "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19",
                "land",
                "property",
                "ownership",
                "usufruct",
                "immovable",
            ),
            "lease": (
                "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                "\u0e84\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                "\u0e9c\u0eb9\u0ec9\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                "\u0e40\u0e0a\u0e48\u0e32",
                "lease",
                "rent",
                "tenant",
            ),
            "company": ("company", "shareholder", "director", "enterprise", "business", "investment"),
            "labor": ("labor", "labour", "employment", "employee", "termination", "wage", "salary"),
            "tax": ("tax", "vat", "customs", "excise", "income"),
            "family": ("marriage", "divorce", "child", "inheritance", "family", "spouse"),
            "criminal": ("criminal", "police", "bail", "detention", "offence", "offense", "penalty"),
        }
        for intent, words in markers.items():
            if any(word in lowered for word in words):
                return intent
        if contains_lao_script(question) or contains_thai_script(question):
            return "general"
        return "general"

    def _lao_domain_terms(self, intent: str) -> str:
        terms = {
            "land": "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 \u0e81\u0eb3\u0ea1\u0eb0\u0eaa\u0eb4\u0e94 \u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9 \u0ead\u0eb0\u0eaa\u0eb1\u0e87\u0eab\u0eb2",
            "lease": "\u0eaa\u0eb1\u0e99\u0e8d\u0eb2 \u0ec0\u0e8a\u0ebb\u0ec8\u0eb2 \u0e84\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2 \u0e9c\u0eb9\u0ec9\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
            "labor": "\u0ec1\u0eae\u0e87\u0e87\u0eb2\u0e99 \u0e84\u0ec8\u0eb2\u0ec1\u0eae\u0e87 \u0eaa\u0eb1\u0e99\u0e8d\u0eb2\u0ec1\u0eae\u0e87\u0e87\u0eb2\u0e99",
        }
        return terms.get(intent, "")

    def _thai_domain_terms(self, intent: str) -> str:
        terms = {
            "land": "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19 \u0e01\u0e23\u0e23\u0e21\u0e2a\u0e34\u0e17\u0e18\u0e34\u0e4c \u0e2d\u0e2a\u0e31\u0e07\u0e2b\u0e32",
            "lease": "\u0e2a\u0e31\u0e0d\u0e0d\u0e32\u0e40\u0e0a\u0e48\u0e32 \u0e04\u0e48\u0e32\u0e40\u0e0a\u0e48\u0e32 \u0e1c\u0e39\u0e49\u0e40\u0e0a\u0e48\u0e32",
        }
        return terms.get(intent, "")

    def _english_domain_terms(self, intent: str) -> str:
        terms = {
            "land": "land ownership land use right immovable property concession title deed",
            "lease": "lease contract rent tenant landlord immovable property",
            "company": "enterprise company shareholder director investment license",
            "labor": "labour employment termination wage severance work permit",
            "tax": "tax VAT income customs excise declaration",
            "family": "family marriage divorce child inheritance succession",
            "criminal": "criminal offence penalty detention police prosecutor",
        }
        return terms.get(intent, "civil code obligation contract right duty")

    def _section_terms(self, question: str) -> str | None:
        words = question.replace("\n", " ").split()
        section_markers = {
            "article",
            "art",
            "section",
            "sec",
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
            "\u0e21\u0e32\u0e15\u0e23\u0e32",
        }
        picked: list[str] = []
        for index, word in enumerate(words):
            cleaned = word.strip(".,;:()[]{}")
            if cleaned.casefold().rstrip(".") in section_markers:
                picked.extend(words[index:index + 3])
        return " ".join(picked).strip() or None

    def _is_statute_like(self, chunk: dict[str, Any]) -> bool:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        values = " ".join(
            str(value or "")
            for value in (
                chunk.get("type"),
                chunk.get("doc_type"),
                chunk.get("document_type"),
                chunk.get("source_table"),
                metadata.get("document_type"),
                chunk.get("title"),
                chunk.get("section"),
                chunk.get("section_ref"),
                metadata.get("law_no"),
                metadata.get("article"),
            )
        ).casefold()
        markers = (
            "law",
            "laws",
            "statute",
            "regulation",
            "decree",
            "article",
            "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
            "\u0e94\u0eb3\u0ea5\u0eb1\u0e94",
            "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22",
            "\u0e21\u0e32\u0e15\u0e23\u0e32",
        )
        return any(marker in values for marker in markers)

    def _is_official_source(self, chunk: dict[str, Any]) -> bool:
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
        return "laoofficialgazette.gov.la" in values or "official" in values

    def _looks_clean(self, chunk: dict[str, Any]) -> bool:
        text = str(chunk.get("content") or "")
        if len(text.strip()) < 80:
            return False
        sample = text[:1200]
        chars = [ch for ch in sample if not ch.isspace()]
        if not chars:
            return False
        suspicious = sum(1 for ch in chars if ch == "\ufffd" or 0x00C0 <= ord(ch) <= 0x00FF)
        return suspicious / len(chars) < 0.25

    def _dedupe(self, queries: list[RetrievalQuery]) -> list[RetrievalQuery]:
        seen: set[tuple[str, str | None]] = set()
        unique: list[RetrievalQuery] = []
        for item in sorted(queries, key=lambda q: (q.priority, not q.required)):
            key = (item.query.casefold().strip(), item.jurisdiction)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
