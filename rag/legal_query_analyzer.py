"""
Legal question understanding for agentic RAG.

This module turns a raw user question into a structured legal research brief:
practice area, legal issues, facts to verify, missing facts, and candidate
authorities to search. It does not generate legal conclusions and it never
treats authority hints as citations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.jurisdiction import canonical_jurisdiction, infer_jurisdiction, infer_response_language


LAO_LAND = "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99"
LAO_LAW = "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d"
LAO_ARTICLE = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2"
LAO_RIGHT = "\u0eaa\u0eb4\u0e94"
LAO_LAND_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_ALT = "\u0eaa\u0eb4\u0e94\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_OCR = "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ecd\u0ec3\u0e8a\u0ec9"
LAO_PROTECTION = "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87"
LAO_GUARD_RIGHT = "\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81\u0eae\u0eb1\u0e81\u0eaa\u0eb2"
LAO_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9"
LAO_BENEFIT_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a"
LAO_BENEFITS = "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94"
LAO_TRANSFER_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99"
LAO_INHERIT_RIGHT = "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94"
THAI_LAND = "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19"
THAI_LAW = "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22"
THAI_ARTICLE = "\u0e21\u0e32\u0e15\u0e23\u0e32"


@dataclass(frozen=True)
class AuthorityHint:
    law_name: str
    search_terms: list[str]
    reason: str
    jurisdiction: str | None = "laos"
    article: str | None = None
    priority: int = 1


@dataclass(frozen=True)
class LegalQueryAnalysis:
    original_question: str
    normalized_question: str
    jurisdiction: str | None
    response_language: str
    practice_area: str
    issue_type: str
    legal_issues: list[str]
    material_facts: list[str]
    missing_facts: list[str]
    parties: list[str]
    requested_outcome: str | None
    authority_hints: list[AuthorityHint] = field(default_factory=list)
    search_phrases: list[str] = field(default_factory=list)
    confidence: float = 0.65

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["authority_hints"] = [asdict(item) for item in self.authority_hints]
        return data


class LegalQueryAnalyzer:
    """Deterministic legal issue analyzer with Lao jurisdiction defaults."""

    def analyze(
        self,
        question: str,
        *,
        jurisdiction: str | None = None,
        memory: dict[str, Any] | None = None,
    ) -> LegalQueryAnalysis:
        normalized = self._normalize(question)
        canonical = infer_jurisdiction(normalized, jurisdiction) or canonical_jurisdiction(jurisdiction) or "laos"
        language = infer_response_language(question)
        practice_area = self._practice_area(normalized)
        issue_type = self._issue_type(normalized)
        articles = self._article_refs(normalized)
        facts = self._material_facts(normalized)
        missing_facts = self._missing_facts(practice_area, issue_type, normalized, memory=memory)
        parties = self._parties(normalized)
        requested_outcome = self._requested_outcome(normalized, issue_type)
        legal_issues = self._legal_issues(practice_area, issue_type, normalized, requested_outcome)
        authority_hints = self._authority_hints(
            practice_area=practice_area,
            issue_type=issue_type,
            jurisdiction=canonical,
            articles=articles,
            question=normalized,
        )
        search_phrases = self._search_phrases(normalized, practice_area, issue_type, authority_hints)
        confidence = self._confidence(practice_area, issue_type, facts, articles)

        return LegalQueryAnalysis(
            original_question=question,
            normalized_question=normalized,
            jurisdiction=canonical,
            response_language=language,
            practice_area=practice_area,
            issue_type=issue_type,
            legal_issues=legal_issues,
            material_facts=facts,
            missing_facts=missing_facts,
            parties=parties,
            requested_outcome=requested_outcome,
            authority_hints=authority_hints,
            search_phrases=search_phrases,
            confidence=confidence,
        )

    def _normalize(self, question: str) -> str:
        return re.sub(r"\s+", " ", question or "").strip()

    def _practice_area(self, question: str) -> str:
        lowered = question.casefold()
        areas = {
            "land": (
                LAO_LAND,
                "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
                "\u0e81\u0eb3\u0ea1\u0eb0\u0eaa\u0eb4\u0e94",
                THAI_LAND,
                "land",
                "property",
                "ownership",
                "usufruct",
                "immovable",
                "title deed",
            ),
            "lease": (
                "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                "\u0e84\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                "\u0e40\u0e0a\u0e48\u0e32",
                "lease",
                "rent",
                "tenant",
                "landlord",
            ),
            "labor": ("labor", "labour", "employment", "employee", "termination", "wage", "salary", "severance"),
            "company": ("company", "enterprise", "shareholder", "director", "investment", "business license"),
            "tax": ("tax", "vat", "customs", "excise", "income tax", "withholding"),
            "family": ("marriage", "divorce", "child", "custody", "inheritance", "succession", "spouse"),
            "criminal": ("criminal", "police", "detention", "bail", "offence", "offense", "penalty", "prosecutor"),
            "contract": ("contract", "agreement", "breach", "debt", "obligation", "damages"),
        }
        for area, markers in areas.items():
            if any(marker in lowered for marker in markers):
                return area
        return "general"

    def _issue_type(self, question: str) -> str:
        lowered = question.casefold()
        if any(word in lowered for word in ("right", "rights", "\u0eaa\u0eb4\u0e94", "\u0e2a\u0e34\u0e17\u0e18\u0e34")):
            return "rights"
        if any(word in lowered for word in ("can i sue", "claim", "compensation", "damages", "remedy")):
            return "remedy"
        if any(word in lowered for word in ("deadline", "limitation", "appeal", "file", "procedure", "process")):
            return "procedure"
        if any(word in lowered for word in ("risk", "chance", "win", "strategy", "negotiate", "settle")):
            return "strategy"
        if any(word in lowered for word in ("legal", "valid", "void", "enforce", "terminate", "cancel")):
            return "validity"
        if any(word in lowered for word in ("must", "required", "permit", "license", "register", "compliance")):
            return "compliance"
        return "analysis"

    def _article_refs(self, question: str) -> list[str]:
        patterns = [
            r"\b(?:article|art\.?|section|sec\.?)\s*([0-9A-Za-z./-]+)",
            rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE})\s*([0-9A-Za-z./-]+)",
        ]
        refs: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, question, flags=re.IGNORECASE):
                value = match.group(1).strip(".,;:()[]{}")
                if value and value not in refs:
                    refs.append(value)
        return refs[:5]

    def _material_facts(self, question: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|[;]|(?:\s+-\s+)", question)
        facts: list[str] = []
        for part in parts:
            clean = part.strip()
            if len(clean) < 12:
                continue
            if self._looks_like_question_only(clean):
                continue
            facts.append(clean[:260])
        return facts[:6]

    def _looks_like_question_only(self, text: str) -> bool:
        lowered = text.casefold()
        return (
            text.endswith("?")
            and not any(word in lowered for word in ("because", "after", "signed", "paid", "received", "own", "rent"))
        )

    def _missing_facts(
        self,
        practice_area: str,
        issue_type: str,
        question: str,
        *,
        memory: dict[str, Any] | None,
    ) -> list[str]:
        missing: list[str] = []
        memory_has_facts = bool(memory and (memory.get("facts_summary") or memory.get("conversation_summary")))
        if not memory_has_facts and len(question) < 120:
            missing.append("timeline and key dates")
            missing.append("documents or notices already received")

        domain_missing = {
            "land": ["land title/use-right document", "land location", "holder/owner name", "transaction or dispute date"],
            "lease": ["lease contract terms", "rent/payment history", "notice of termination", "breach alleged by each side"],
            "labor": ["employment contract", "termination date", "salary and unpaid benefits", "employer notice/reason"],
            "company": ["company registration details", "shareholder/director role", "resolution or contract at issue"],
            "tax": ["tax period", "assessment notice", "taxpayer status", "amount disputed"],
            "family": ["marriage/family registration documents", "child/property details", "relevant dates"],
            "criminal": ["charge or alleged offence", "police/prosecutor stage", "detention/bail status", "evidence summary"],
            "contract": ["signed contract", "performance history", "breach date", "loss/damages evidence"],
        }
        missing.extend(domain_missing.get(practice_area, ["relevant official documents", "specific legal outcome requested"]))
        if issue_type == "procedure":
            missing.append("current procedural stage")
        return list(dict.fromkeys(missing))[:7]

    def _parties(self, question: str) -> list[str]:
        lowered = question.casefold()
        parties = []
        markers = {
            "tenant": ("tenant", "\u0e9c\u0eb9\u0ec9\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2", "\u0e1c\u0e39\u0e49\u0e40\u0e0a\u0e48\u0e32"),
            "landlord": ("landlord", "\u0e9c\u0eb9\u0ec9\u0ec3\u0eab\u0ec9\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2", "\u0e1c\u0e39\u0e49\u0e43\u0e2b\u0e49\u0e40\u0e0a\u0e48\u0e32"),
            "employee": ("employee", "worker", "employer", "staff"),
            "company": ("company", "director", "shareholder"),
            "state_authority": ("ministry", "district", "village", "tax office", "police", "court"),
        }
        for party, words in markers.items():
            if any(word in lowered for word in words):
                parties.append(party)
        return parties[:6]

    def _requested_outcome(self, question: str, issue_type: str) -> str | None:
        lowered = question.casefold()
        outcomes = [
            ("compensation", ("compensation", "damages", "money", "refund")),
            ("validity assessment", ("valid", "void", "legal", "enforce")),
            ("termination advice", ("terminate", "cancel", "evict", "end contract")),
            ("filing/procedure guidance", ("file", "appeal", "deadline", "process")),
            ("rights explanation", ("rights", "right", "\u0eaa\u0eb4\u0e94", "\u0e2a\u0e34\u0e17\u0e18\u0e34")),
        ]
        for label, words in outcomes:
            if any(word in lowered for word in words):
                return label
        if issue_type == "rights":
            return "rights explanation"
        if issue_type == "remedy":
            return "available remedies"
        return None

    def _legal_issues(
        self,
        practice_area: str,
        issue_type: str,
        question: str,
        requested_outcome: str | None,
    ) -> list[str]:
        labels = {
            "land": "land ownership/use-right issue",
            "lease": "lease and contractual obligation issue",
            "labor": "employment/labour rights issue",
            "company": "enterprise/company governance issue",
            "tax": "tax compliance or assessment issue",
            "family": "family or inheritance issue",
            "criminal": "criminal liability or procedure issue",
            "contract": "contract validity/breach issue",
            "general": "general Lao legal issue",
        }
        issues = [labels.get(practice_area, "legal issue")]
        if issue_type != "analysis":
            issues.append(f"{issue_type} question")
        if self._is_land_use_right_protection_question(question):
            issues.append("land-use-right protection under Law on Land Article 5")
        if requested_outcome:
            issues.append(f"requested outcome: {requested_outcome}")
        articles = self._article_refs(question)
        if articles:
            issues.append("specific article lookup: " + ", ".join(articles))
        return issues[:5]

    def _authority_hints(
        self,
        *,
        practice_area: str,
        issue_type: str,
        jurisdiction: str | None,
        articles: list[str],
        question: str,
    ) -> list[AuthorityHint]:
        if canonical_jurisdiction(jurisdiction) != "laos":
            return [
                AuthorityHint(
                    law_name="Relevant primary legislation",
                    search_terms=[question, "law", "article", practice_area, issue_type],
                    reason="Non-Lao or unspecified jurisdiction primary authority lookup",
                    jurisdiction=jurisdiction,
                )
            ]

        base: dict[str, list[AuthorityHint]] = {
            "land": [
                AuthorityHint("Law on Land", ["Lao PDR land law", LAO_LAND, "land use right", "ownership"], "Primary land statute", priority=1),
                AuthorityHint("Civil Code", ["immovable property", "ownership", "obligations", "contract"], "Civil-law background for property/obligations", priority=2),
            ],
            "lease": [
                AuthorityHint("Civil Code", ["lease contract", "rent", "tenant", "obligations"], "Primary contract and lease authority", priority=1),
                AuthorityHint("Law on Land", ["land lease", LAO_LAND, "land use right"], "Needed when the leased asset is land", priority=2),
            ],
            "labor": [
                AuthorityHint("Labour Law", ["employment contract", "termination", "wage", "severance"], "Primary labour rights authority", priority=1),
            ],
            "company": [
                AuthorityHint("Enterprise Law", ["enterprise", "company", "director", "shareholder"], "Primary enterprise governance authority", priority=1),
                AuthorityHint("Investment Promotion Law", ["investment", "license", "foreign investor"], "Investment-specific authority", priority=2),
            ],
            "tax": [
                AuthorityHint("Tax Law", ["tax", "assessment", "taxpayer", "income"], "Primary tax authority", priority=1),
                AuthorityHint("Value Added Tax Law", ["VAT", "invoice", "input tax", "output tax"], "VAT-specific authority", priority=2),
            ],
            "family": [
                AuthorityHint("Family Law", ["marriage", "divorce", "child", "spouse"], "Primary family-law authority", priority=1),
                AuthorityHint("Civil Code", ["inheritance", "succession", "property"], "Civil-law authority where family property/succession is involved", priority=2),
            ],
            "criminal": [
                AuthorityHint("Penal Code", ["offence", "penalty", "criminal liability"], "Primary criminal offence authority", priority=1),
                AuthorityHint("Criminal Procedure Law", ["police", "prosecutor", "court", "detention", "bail"], "Procedure and enforcement authority", priority=2),
            ],
            "contract": [
                AuthorityHint("Civil Code", ["contract", "obligation", "breach", "damages"], "Primary contract authority", priority=1),
            ],
            "general": [
                AuthorityHint("Civil Code", ["rights", "obligations", "civil law", "article"], "General private-law authority", priority=2),
            ],
        }
        hints = list(base.get(practice_area, base["general"]))
        if (
            practice_area == "land"
            and issue_type == "rights"
            and self._is_land_use_right_protection_question(question)
        ):
            hints.insert(
                0,
                AuthorityHint(
                    law_name="Law on Land",
                    search_terms=[
                        f"{LAO_ARTICLE} 5",
                        "Article 5",
                        LAO_PROTECTION,
                        LAO_LAND_USE_RIGHT,
                        LAO_LAND_USE_RIGHT_ALT,
                        LAO_GUARD_RIGHT,
                        LAO_USE_RIGHT,
                        LAO_BENEFITS,
                        LAO_TRANSFER_RIGHT,
                        LAO_INHERIT_RIGHT,
                    ],
                    reason="Land-use-right protection is governed by Law on Land Article 5",
                    jurisdiction="laos",
                    article="5",
                    priority=0,
                ),
            )
        for article in articles:
            hints.insert(
                0,
                AuthorityHint(
                    law_name=hints[0].law_name if hints else "Relevant Lao law",
                    search_terms=[f"{LAO_ARTICLE} {article}", f"Article {article}", question],
                    reason="User asked about a specific article/section",
                    jurisdiction="laos",
                    article=article,
                    priority=0,
                ),
            )
        return hints[:5]

    def _search_phrases(
        self,
        question: str,
        practice_area: str,
        issue_type: str,
        authority_hints: list[AuthorityHint],
    ) -> list[str]:
        phrases = [question]
        for hint in authority_hints:
            parts = [hint.law_name, hint.article or "", *hint.search_terms[:4], practice_area, issue_type]
            phrase = " ".join(part for part in parts if part).strip()
            if phrase:
                phrases.append(phrase)
        return list(dict.fromkeys(phrases))[:8]

    def _confidence(self, practice_area: str, issue_type: str, facts: list[str], articles: list[str]) -> float:
        score = 0.45
        if practice_area != "general":
            score += 0.2
        if issue_type != "analysis":
            score += 0.1
        if facts:
            score += 0.1
        if articles:
            score += 0.1
        return min(score, 0.9)

    def _is_land_use_right_protection_question(self, question: str) -> bool:
        lowered = question.casefold()
        has_land = LAO_LAND in lowered or "land" in lowered
        has_right = LAO_RIGHT in lowered or "right" in lowered
        has_use_right = any(
            marker in lowered
            for marker in (
                LAO_LAND_USE_RIGHT,
                LAO_LAND_USE_RIGHT_ALT,
                LAO_LAND_USE_RIGHT_OCR,
                "land use right",
                "use right",
            )
        )
        has_protection_intent = any(
            marker in lowered
            for marker in (
                LAO_PROTECTION,
                "\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81",
                "\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0e81\u0eb2\u0e99\u0e9b\u0ebb\u0e81",
                "\u0eaa\u0eb4\u0e94\u0ec3\u0e94",
                "\u0ec3\u0e94\u0ec1\u0e94\u0ec8",
                "protected",
                "protection",
                "which rights",
            )
        )
        return has_land and has_right and (has_use_right or has_protection_intent) and has_protection_intent
