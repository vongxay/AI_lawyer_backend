"""
rag/smart_query_expander.py
===========================
LLM-powered query understanding for Lao legal RAG.

The deterministic LegalQueryAnalyzer only recognises a question when it contains
hard-coded keywords. Real users ask colloquially ("ຂ້ອຍຢາກຂາຍດິນ ຕ້ອງເຮັດແນວໃດ")
without naming the statute. This module asks a cheap LLM to read the question
like a Lao lawyer and produce a structured research brief:

  * practice_area        — normalised domain bucket
  * legal_concepts       — Lao legal concepts the question is really about
  * candidate_laws       — likely Lao law names to search
  * candidate_articles   — likely article numbers (if inferable)
  * search_phrases       — Lao search phrases (statute-style wording)
  * hyde_passage         — a short hypothetical Lao statutory answer used to
                           build a much stronger semantic embedding (HyDE)

The output is merged into the analysis dict that drives the agentic planner, so
existing retrieval machinery automatically benefits. It NEVER produces legal
conclusions or citations — only search guidance.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.config import get_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from services.llm_service import LlmService

log = get_logger(__name__)

_PRACTICE_AREAS = (
    "land", "lease", "labor", "company", "tax", "family", "criminal",
    "contract", "administrative", "investment", "immigration", "environment",
    "general",
)

# Map Lao domain words the LLM sometimes returns to the canonical English buckets.
_LAO_AREA_ALIASES = {
    "\u0e94\u0eb4\u0e99": "land",                       # ດິນ
    "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99": "land",        # ທີ່ດິນ
    "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2": "lease",            # ເຊົ່າ
    "\u0ec1\u0eae\u0e87\u0e87\u0eb2\u0e99": "labor",       # ແຮງງານ
    "\u0e84\u0ead\u0e9a\u0e84\u0ebb\u0ea7": "family",      # ຄອບຄົວ
    "\u0ead\u0eb2\u0e8d\u0eb2": "criminal",               # ອາຍາ
    "\u0eaa\u0eb1\u0e8d\u0eb2": "contract",               # ສັນຍາ
    "\u0e9e\u0eb2\u0eaa\u0eb5": "tax",                    # ພາສີ
    "\u0ea7\u0eb4\u0eaa\u0eb2\u0eab\u0eb0\u0e81\u0eb4\u0e94": "company",  # ວິສາຫະກິດ
}

_SYSTEM_PROMPT = (
    "ເຈົ້າເປັນຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ. ໜ້າທີ່ຂອງເຈົ້າແມ່ນວິເຄາະຄຳຖາມຂອງຜູ້ໃຊ້ "
    "ແລະ ສ້າງ 'ແຜນຄົ້ນຫາ' ເພື່ອຊອກຫາມາດຕາກົດໝາຍລາວທີ່ກ່ຽວຂ້ອງ. "
    "ຫ້າມໃຫ້ຄຳຕັດສິນທາງກົດໝາຍ ຫຼື ອ້າງອີງມາດຕາທີ່ບໍ່ແນ່ໃຈ. "
    "ໃຫ້ຄິດວ່າຄຳຖາມເວົ້າເຖິງບັນຫາກົດໝາຍຫຍັງ ແລະ ກົດໝາຍສະບັບໃດ/ມາດຕາໃດໜ້າຈະກ່ຽວຂ້ອງ. "
    "ຕອບເປັນ JSON ເທົ່ານັ້ນ."
)

_USER_TEMPLATE = """ຄຳຖາມຂອງຜູ້ໃຊ້:
\"\"\"{question}\"\"\"

ສ້າງ JSON ຕາມ schema ນີ້ (ຄ່າຂໍ້ຄວາມເປັນພາສາລາວ ເວັ້ນແຕ່ຊື່ພາສາອັງກິດ):
{{
  "practice_area": "ໜຶ່ງໃນ: land, lease, labor, company, tax, family, criminal, contract, administrative, investment, immigration, environment, general",
  "legal_concepts": ["ແນວຄິດກົດໝາຍ 2-5 ຂໍ້ ເປັນພາສາລາວ"],
  "candidate_laws": ["ຊື່ກົດໝາຍລາວທີ່ໜ້າຈະກ່ຽວຂ້ອງ ເຊັ່ນ: ກົດໝາຍວ່າດ້ວຍທີ່ດິນ"],
  "candidate_articles": ["ເລກມາດຕາ ຖ້າພໍຄາດເດົາໄດ້ ບໍ່ດັ່ງນັ້ນເປັນ array ວ່າງ"],
  "search_phrases": ["ປະໂຫຍກຄົ້ນຫາແບບກົດໝາຍ 3-6 ຂໍ້ ເປັນພາສາລາວ"],
  "hyde_passage": "ຫຍໍ້ຄຳຕອບສົມມຸດແບບມາດຕາກົດໝາຍ 2-4 ປະໂຫຍກ ເປັນພາສາລາວ (ໃຊ້ສຳລັບການຄົ້ນຫາເທົ່ານັ້ນ)"
}}

ຕອບເປັນ JSON ດຽວ ບໍ່ມີຄຳອະທິບາຍອື່ນ."""


@dataclass
class QueryUnderstanding:
    practice_area: str | None = None
    legal_concepts: list[str] = field(default_factory=list)
    candidate_laws: list[str] = field(default_factory=list)
    candidate_articles: list[str] = field(default_factory=list)
    search_phrases: list[str] = field(default_factory=list)
    hyde_passage: str = ""

    def is_empty(self) -> bool:
        return not (
            self.legal_concepts
            or self.candidate_laws
            or self.search_phrases
            or self.hyde_passage
        )


class SmartQueryExpander:
    """Turns a colloquial question into a structured legal search brief via LLM."""

    def __init__(self, llm: "LlmService | None" = None) -> None:
        self._llm = llm
        self._settings = get_settings()

    def _llm_service(self) -> "LlmService | None":
        if self._llm is not None:
            return self._llm
        try:
            from services.llm_service import LlmService

            self._llm = LlmService()
        except Exception as exc:  # noqa: BLE001
            log.debug("smart_query_expander.llm_init_failed", error=str(exc))
            self._llm = None
        return self._llm

    @staticmethod
    def should_expand(question: str) -> bool:
        text = (question or "").strip()
        if len(text) < 6:
            return False
        # Skip pure greetings / thanks — they are not legal research questions.
        greetings = ("ສະບາຍດີ", "ຂອບໃຈ", "hello", "hi", "thank")
        lowered = text.casefold()
        if any(lowered.startswith(g) for g in greetings) and len(text) < 25:
            return False
        return True

    async def expand(self, question: str) -> QueryUnderstanding:
        """Best-effort LLM expansion. Never raises — returns empty on any failure."""
        if not self.should_expand(question):
            return QueryUnderstanding()

        llm = self._llm_service()
        if llm is None:
            return QueryUnderstanding()

        from services.llm_service import Message

        try:
            result = await asyncio.wait_for(
                llm.generate(
                    model=self._settings.model_research,
                    messages=[Message(role="user", content=_USER_TEMPLATE.format(question=question.strip()))],
                    system=_SYSTEM_PROMPT,
                    max_tokens=600,
                    temperature=0.0,
                ),
                timeout=15.0,
            )
        except Exception as exc:  # noqa: BLE001
            log.info("smart_query_expander.generate_failed", error=str(exc))
            return QueryUnderstanding()

        understanding = self._parse(result.text)
        if not understanding.is_empty():
            log.info(
                "smart_query_expander.ok",
                practice_area=understanding.practice_area,
                concepts=len(understanding.legal_concepts),
                laws=len(understanding.candidate_laws),
                phrases=len(understanding.search_phrases),
                has_hyde=bool(understanding.hyde_passage),
            )
        return understanding

    def _parse(self, raw: str) -> QueryUnderstanding:
        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            return QueryUnderstanding()

        practice_area = str(payload.get("practice_area") or "").strip().lower()
        if practice_area not in _PRACTICE_AREAS:
            practice_area = _LAO_AREA_ALIASES.get(payload.get("practice_area", "").strip(), None)

        return QueryUnderstanding(
            practice_area=practice_area,
            legal_concepts=self._clean_list(payload.get("legal_concepts")),
            candidate_laws=self._clean_list(payload.get("candidate_laws")),
            candidate_articles=self._clean_articles(payload.get("candidate_articles")),
            search_phrases=self._clean_list(payload.get("search_phrases")),
            hyde_passage=str(payload.get("hyde_passage") or "").strip()[:600],
        )

    @staticmethod
    def _extract_json(raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            return None
        # Strip ```json fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if brace:
                text = brace.group(0)
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _clean_list(value: Any, *, limit: int = 8) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text[:160])
            if len(cleaned) >= limit:
                break
        return cleaned

    @staticmethod
    def _clean_articles(value: Any, *, limit: int = 6) -> list[str]:
        if not isinstance(value, list):
            return []
        articles: list[str] = []
        for item in value:
            match = re.search(r"\d{1,4}", str(item or ""))
            if match and match.group(0) not in articles:
                articles.append(match.group(0))
            if len(articles) >= limit:
                break
        return articles


def merge_understanding_into_analysis(
    analysis: dict[str, Any],
    understanding: QueryUnderstanding,
) -> dict[str, Any]:
    """Fold LLM understanding into the deterministic analysis dict that drives
    the agentic planner. Deterministic results are kept; LLM results augment."""
    if understanding.is_empty():
        return analysis

    merged = dict(analysis)

    # Upgrade practice_area only if the deterministic analyzer gave up ("general").
    if understanding.practice_area and (merged.get("practice_area") in (None, "", "general")):
        merged["practice_area"] = understanding.practice_area

    # Merge articles.
    articles = list(merged.get("articles") or [])
    for art in understanding.candidate_articles:
        if art not in articles:
            articles.append(art)
    merged["articles"] = articles[:8]

    # Prepend LLM search phrases (concepts + phrases) ahead of deterministic ones.
    existing_phrases = list(merged.get("search_phrases") or [])
    llm_phrases = [*understanding.search_phrases, *understanding.legal_concepts]
    seen = {p.casefold() for p in llm_phrases}
    combined = llm_phrases + [p for p in existing_phrases if p.casefold() not in seen]
    merged["search_phrases"] = combined[:14]

    # Add candidate laws as high-priority authority hints.
    hints = list(merged.get("authority_hints") or [])
    existing_law_names = {
        str(h.get("law_name") or "").casefold() for h in hints if isinstance(h, dict)
    }
    llm_hints: list[dict[str, Any]] = []
    for law in understanding.candidate_laws:
        if law.casefold() in existing_law_names:
            continue
        llm_hints.append({
            "law_name": law,
            "search_terms": [law, *understanding.legal_concepts[:3]],
            "reason": "LLM-inferred candidate statute",
            "jurisdiction": merged.get("jurisdiction") or "laos",
            "article": None,
            "priority": 1,
        })
    merged["authority_hints"] = (llm_hints + hints)[:6]

    # Record the HyDE passage for the embedding builder.
    merged["hyde_passage"] = understanding.hyde_passage
    merged["llm_legal_concepts"] = understanding.legal_concepts
    return merged
