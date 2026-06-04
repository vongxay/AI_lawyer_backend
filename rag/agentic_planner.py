"""
Agentic retrieval planning for legal RAG.

This module keeps the agentic behaviour deterministic and testable: it plans
multiple retrieval attempts, expands Lao legal queries, and requests a follow-up
search when the first pass does not return enough grounded legal sources.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.jurisdiction import canonical_jurisdiction, infer_jurisdiction

LAO_GENERIC_TERMS = "ກົດໝາຍ ມາດຕາ ສປປ ລາວ"
LAO_AUTHORITY_TERMS = "ກົດໝາຍວ່າດ້ວຍ ມາດຕາ ດຳລັດ ຄຳສັ່ງ"
LAO_LAND_TERMS = "ທີ່ດິນ ກຳມະສິດ ສິດນຳໃຊ້ ອະສັງຫາ ໂອນ"
LAO_LEASE_TERMS = "ສັນຍາ ໜີ້ ຄ່າເຊົ່າ ເຊົ່າ ຜູ້ເຊົ່າ"


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    purpose: str
    jurisdiction: str | None
    priority: int = 1


class AgenticRetrievalPlanner:
    def plan(self, question: str, jurisdiction: str | None = None) -> list[RetrievalQuery]:
        canonical = infer_jurisdiction(question, jurisdiction)
        queries = [
            RetrievalQuery(
                query=question,
                purpose="original_user_question",
                jurisdiction=canonical,
                priority=1,
            )
        ]

        if canonical == "laos":
            queries.extend(self._lao_expansions(question, canonical))
        elif canonical == "thailand":
            queries.extend(self._thai_expansions(question, canonical))

        return self._dedupe(queries)

    def second_pass(self, question: str, jurisdiction: str | None = None) -> list[RetrievalQuery]:
        canonical = canonical_jurisdiction(jurisdiction) or infer_jurisdiction(question)
        if canonical == "laos":
            if self._is_land_query(question):
                return self._dedupe([
                    RetrievalQuery(
                        query=f"{question} {LAO_AUTHORITY_TERMS} {LAO_LAND_TERMS}",
                        purpose="lao_land_statute_second_pass",
                        jurisdiction=canonical,
                        priority=2,
                    ),
                    RetrievalQuery(
                        query=f"{question} Lao PDR land law ownership usufruct land use right immovable property official gazette",
                        purpose="lao_land_english_second_pass",
                        jurisdiction=canonical,
                        priority=2,
                    ),
                ])

            if self._is_lease_query(question):
                return self._dedupe([
                    RetrievalQuery(
                        query=f"{question} {LAO_AUTHORITY_TERMS} {LAO_LEASE_TERMS}",
                        purpose="lao_lease_statute_second_pass",
                        jurisdiction=canonical,
                        priority=2,
                    ),
                    RetrievalQuery(
                        query=f"{question} Lao PDR civil code lease contract rent tenant official gazette",
                        purpose="lao_lease_english_second_pass",
                        jurisdiction=canonical,
                        priority=2,
                    ),
                ])

            return self._dedupe([
                RetrievalQuery(
                    query=f"{question} {LAO_AUTHORITY_TERMS}",
                    purpose="lao_statute_second_pass",
                    jurisdiction=canonical,
                    priority=2,
                ),
                RetrievalQuery(
                    query=f"{question} Lao PDR law article decree regulation official gazette",
                    purpose="lao_english_translation_second_pass",
                    jurisdiction=canonical,
                    priority=2,
                ),
            ])

            return self._dedupe([
                RetrievalQuery(
                    query=f"{question} ກົດໝາຍວ່າດ້ວຍ ມາດຕາ ດຳລັດ ຄຳສັ່ງ",
                    purpose="lao_statute_second_pass",
                    jurisdiction=canonical,
                    priority=2,
                ),
                RetrievalQuery(
                    query=f"{question} Lao PDR law article decree regulation official gazette",
                    purpose="lao_english_translation_second_pass",
                    jurisdiction=canonical,
                    priority=2,
                ),
            ])

        return self._dedupe([
            RetrievalQuery(
                query=f"{question} law section regulation legal article",
                purpose="generic_legal_second_pass",
                jurisdiction=canonical,
                priority=2,
            )
        ])

    def should_second_pass(self, chunks: list[dict], min_results: int = 5) -> bool:
        if len(chunks) < min_results:
            return True
        law_like = [c for c in chunks if c.get("type") in {"law", "statute", "regulation"}]
        return len(law_like) == 0

    def _lao_expansions(self, question: str, jurisdiction: str) -> list[RetrievalQuery]:
        if self._is_land_query(question):
            return [
                RetrievalQuery(
                    query=f"{question} {LAO_GENERIC_TERMS} {LAO_LAND_TERMS}",
                    purpose="lao_land_terms",
                    jurisdiction=jurisdiction,
                    priority=1,
                ),
                RetrievalQuery(
                    query=f"{question} Lao PDR land law ownership property usufruct land use right official gazette",
                    purpose="lao_land_english_terms",
                    jurisdiction=jurisdiction,
                    priority=1,
                ),
            ]

        if self._is_lease_query(question):
            return [
                RetrievalQuery(
                    query=f"{question} {LAO_GENERIC_TERMS} {LAO_LEASE_TERMS}",
                    purpose="lao_lease_terms",
                    jurisdiction=jurisdiction,
                    priority=1,
                ),
                RetrievalQuery(
                    query=f"{question} Lao PDR law article decree regulation official gazette lease rent tenant",
                    purpose="lao_lease_english_terms",
                    jurisdiction=jurisdiction,
                    priority=1,
                ),
            ]

        return [
            RetrievalQuery(
                query=f"{question} {LAO_GENERIC_TERMS}",
                purpose="lao_statute_terms",
                jurisdiction=jurisdiction,
                priority=1,
            ),
            RetrievalQuery(
                query=f"{question} Lao PDR law article decree regulation official gazette",
                purpose="lao_official_gazette_terms",
                jurisdiction=jurisdiction,
                priority=1,
            ),
        ]

        return [
            RetrievalQuery(
                query=f"{question} ກົດໝາຍ ມາດຕາ ສປປ ລາວ",
                purpose="lao_statute_terms",
                jurisdiction=jurisdiction,
                priority=1,
            ),
            RetrievalQuery(
                query=f"{question} Lao PDR legislation article official gazette",
                purpose="lao_official_gazette_terms",
                jurisdiction=jurisdiction,
                priority=1,
            ),
        ]

    def _thai_expansions(self, question: str, jurisdiction: str) -> list[RetrievalQuery]:
        return [
            RetrievalQuery(
                query=f"{question} กฎหมาย มาตรา พระราชบัญญัติ สัญญา เช่า ค่าเช่า ผู้เช่า ผู้ให้เช่า",
                purpose="thai_statute_terms",
                jurisdiction=jurisdiction,
                priority=1,
            )
        ]

        return [
            RetrievalQuery(
                query=f"{question} กฎหมาย มาตรา พระราชบัญญัติ",
                purpose="thai_statute_terms",
                jurisdiction=jurisdiction,
                priority=1,
            )
        ]

    def _is_land_query(self, question: str) -> bool:
        lowered = question.casefold()
        markers = (
            "ທີ່ດິນ",
            "ກຳມະສິດ",
            "ສິດນຳໃຊ້",
            "ອະສັງຫາ",
            "ที่ดิน",
            "กรรมสิทธิ์",
            "land",
            "property",
            "ownership",
            "usufruct",
            "immovable",
        )
        return any(marker in lowered for marker in markers)

    def _is_lease_query(self, question: str) -> bool:
        lowered = question.casefold()
        markers = (
            "ເຊົ່າ",
            "ຄ່າເຊົ່າ",
            "ຜູ້ເຊົ່າ",
            "เช่า",
            "ค่าเช่า",
            "lease",
            "rent",
            "tenant",
        )
        return any(marker in lowered for marker in markers)

    def _dedupe(self, queries: list[RetrievalQuery]) -> list[RetrievalQuery]:
        seen: set[tuple[str, str | None]] = set()
        unique: list[RetrievalQuery] = []
        for item in queries:
            key = (item.query.casefold().strip(), item.jurisdiction)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
