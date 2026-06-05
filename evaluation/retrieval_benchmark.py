"""
Retrieval benchmark runner for Lao legal RAG.

Run from ``AI_lawyer_backend``:

    py -m evaluation.retrieval_benchmark --json

The runner intentionally uses keyword-only retrieval by default. This keeps the
benchmark stable when the OpenAI embedding quota/key is unavailable, while still
testing the production retriever, query analyzer, planner, and reranker.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.database import get_supabase
from rag.agentic_planner import AgenticRetrievalPlanner
from rag.legal_query_analyzer import LegalQueryAnalyzer
from rag.reranker import Reranker
from rag.retriever import Retriever


LAO_ARTICLE = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2"
THAI_ARTICLE = "\u0e21\u0e32\u0e15\u0e23\u0e32"


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    id: str
    question: str
    jurisdiction: str = "laos"
    language: str = "lo"
    category: str = "general"
    difficulty: str = "medium"
    expected_law: str | None = None
    expected_articles: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalBenchmarkCase":
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            jurisdiction=str(data.get("jurisdiction") or "laos"),
            language=str(data.get("language") or "lo"),
            category=str(data.get("category") or "general"),
            difficulty=str(data.get("difficulty") or "medium"),
            expected_law=str(data.get("expected_law")) if data.get("expected_law") else None,
            expected_articles=[str(item) for item in data.get("expected_articles", [])],
            must_contain_any=[str(item) for item in data.get("must_contain_any", [])],
        )


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    passed: bool
    expected_articles: list[str]
    article_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    term_coverage: float
    retrieved_count: int
    top_sections: list[str]
    top_titles: list[str]
    plan: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "expected_articles": self.expected_articles,
            "article_rank": self.article_rank,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "reciprocal_rank": self.reciprocal_rank,
            "term_coverage": self.term_coverage,
            "retrieved_count": self.retrieved_count,
            "top_sections": self.top_sections,
            "top_titles": self.top_titles,
            "plan": self.plan,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RetrievalBenchmarkReport:
    dataset: str
    case_count: int
    pass_rate: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    average_term_coverage: float
    results: list[RetrievalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_count": self.case_count,
            "pass_rate": self.pass_rate,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
            "average_term_coverage": self.average_term_coverage,
            "results": [item.to_dict() for item in self.results],
        }


class RetrievalBenchmarkRunner:
    def __init__(self, *, supabase: Any | None = None, top_k: int = 10, plan_queries: int = 3) -> None:
        self._analyzer = LegalQueryAnalyzer()
        self._planner = AgenticRetrievalPlanner()
        self._retriever = Retriever(supabase=supabase)
        self._reranker = Reranker()
        self._top_k = top_k
        self._plan_queries = max(1, plan_queries)

    async def run(self, cases: list[RetrievalBenchmarkCase], *, dataset_name: str) -> RetrievalBenchmarkReport:
        results = [await self.run_case(case) for case in cases]
        count = len(results)
        if count == 0:
            return RetrievalBenchmarkReport(dataset_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

        return RetrievalBenchmarkReport(
            dataset=dataset_name,
            case_count=count,
            pass_rate=round(sum(item.passed for item in results) / count, 4),
            hit_at_1=round(sum(item.hit_at_1 for item in results) / count, 4),
            hit_at_3=round(sum(item.hit_at_3 for item in results) / count, 4),
            hit_at_5=round(sum(item.hit_at_5 for item in results) / count, 4),
            mrr=round(sum(item.reciprocal_rank for item in results) / count, 4),
            average_term_coverage=round(sum(item.term_coverage for item in results) / count, 4),
            results=results,
        )

    async def run_case(self, case: RetrievalBenchmarkCase) -> RetrievalCaseResult:
        analysis = self._analyzer.analyze(case.question, jurisdiction=case.jurisdiction)
        plan = self._planner.plan(case.question, case.jurisdiction, analysis=analysis.to_dict())
        chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        for item in plan[:self._plan_queries]:
            rows = await self._retriever.retrieve(
                query=item.query,
                embedding=None,
                jurisdiction=item.jurisdiction,
                top_k=max(self._top_k, self._top_k * 2),
            )
            chunks.extend(rows)
            trace.append({
                "purpose": item.purpose,
                "jurisdiction": item.jurisdiction,
                "query": item.query,
                "results": len(rows),
            })

        deduped = _dedupe_chunks(chunks)
        reranked = await self._reranker.rerank(query=case.question, chunks=deduped, top_k=self._top_k)
        return evaluate_retrieval_case(case, reranked, plan_trace=trace)


def load_cases(path: Path) -> list[RetrievalBenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Retrieval benchmark dataset must be a JSON list.")
    return [RetrievalBenchmarkCase.from_dict(item) for item in raw if isinstance(item, dict)]


def evaluate_retrieval_case(
    case: RetrievalBenchmarkCase,
    chunks: list[dict[str, Any]],
    *,
    plan_trace: list[dict[str, Any]] | None = None,
) -> RetrievalCaseResult:
    article_rank = _first_matching_article_rank(chunks, case.expected_articles)
    term_coverage = _term_coverage(chunks, case.must_contain_any)
    warnings = _quality_warnings(chunks)
    passed = bool(article_rank and article_rank <= 3)
    if not case.expected_articles and case.must_contain_any:
        passed = term_coverage > 0

    return RetrievalCaseResult(
        case_id=case.id,
        passed=passed,
        expected_articles=case.expected_articles,
        article_rank=article_rank,
        hit_at_1=bool(article_rank and article_rank <= 1),
        hit_at_3=bool(article_rank and article_rank <= 3),
        hit_at_5=bool(article_rank and article_rank <= 5),
        reciprocal_rank=round(1.0 / article_rank, 4) if article_rank else 0.0,
        term_coverage=term_coverage,
        retrieved_count=len(chunks),
        top_sections=[_chunk_section(chunk) for chunk in chunks[:5]],
        top_titles=[str(chunk.get("title") or "")[:120] for chunk in chunks[:5]],
        plan=plan_trace or [],
        warnings=warnings,
    )


def _first_matching_article_rank(chunks: list[dict[str, Any]], expected_articles: list[str]) -> int | None:
    targets = [_normalise_article(value) for value in expected_articles if _normalise_article(value)]
    if not targets:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if any(_chunk_matches_article(chunk, target) for target in targets):
            return index
    return None


def _chunk_matches_article(chunk: dict[str, Any], target: str) -> bool:
    text = _chunk_text(chunk)
    patterns = (
        rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|art\.?|section|sec\.?)\s*0*{re.escape(target)}(?:\D|$)",
        rf"^0*{re.escape(target)}(?:\.|\s)",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def _term_coverage(chunks: list[dict[str, Any]], terms: list[str]) -> float:
    clean_terms = [term.casefold().strip() for term in terms if term.strip()]
    if not clean_terms:
        return 1.0
    haystack = "\n".join(_chunk_text(chunk) for chunk in chunks[:5]).casefold()
    hits = sum(1 for term in clean_terms if term in haystack)
    return round(hits / len(clean_terms), 4)


def _quality_warnings(chunks: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not chunks:
        return ["no_results"]
    for index, chunk in enumerate(chunks[:3], start=1):
        text = str(chunk.get("content") or "")
        if _suspicious_text_ratio(text) > 0.25:
            warnings.append(f"top_{index}_text_quality_low")
    return warnings


def _suspicious_text_ratio(text: str) -> float:
    sample = text[:1200]
    chars = [ch for ch in sample if not ch.isspace()]
    if not chars:
        return 1.0
    suspicious = sum(1 for ch in chars if ch == "\ufffd" or 0x00C0 <= ord(ch) <= 0x00FF)
    return suspicious / len(chars)


def _chunk_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return " ".join(
        str(value or "")
        for value in (
            chunk.get("title"),
            chunk.get("section"),
            chunk.get("section_ref"),
            chunk.get("content"),
            metadata.get("section"),
            metadata.get("article"),
            metadata.get("law_no"),
        )
    )


def _chunk_section(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    value = chunk.get("section") or chunk.get("section_ref") or metadata.get("section") or metadata.get("article")
    return str(value or "")


def _normalise_article(value: str) -> str:
    match = re.search(r"0*([0-9]{1,4})", str(value))
    return match.group(1) if match else ""


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    unique: list[dict[str, Any]] = []
    for chunk in chunks:
        key = str(chunk.get("chunk_id") or chunk.get("id") or f"{chunk.get('title')}|{_chunk_section(chunk)}")
        if key not in seen:
            seen[key] = len(unique)
            unique.append(chunk)
            continue
        current_index = seen[key]
        if _score(chunk) > _score(unique[current_index]):
            unique[current_index] = {**unique[current_index], **chunk}
    return unique


def _score(chunk: dict[str, Any]) -> float:
    for key in ("_rerank_score", "final_score", "score"):
        try:
            return float(chunk.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


async def _run_cli(args: argparse.Namespace) -> RetrievalBenchmarkReport:
    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path)
    supabase = await get_supabase()
    if supabase is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env.")
    runner = RetrievalBenchmarkRunner(supabase=supabase, top_k=args.top_k, plan_queries=args.plan_queries)
    return await runner.run(cases, dataset_name=str(dataset_path))


def _parse_args() -> argparse.Namespace:
    default_dataset = Path(__file__).with_name("lao_retrieval_benchmark.json")
    parser = argparse.ArgumentParser(description="Run Lao legal RAG retrieval benchmark.")
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to retrieval benchmark JSON dataset.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of final reranked chunks per case.")
    parser.add_argument("--plan-queries", type=int, default=3, help="Number of agentic retrieval queries per case.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--unicode", action="store_true", help="Print unicode characters instead of JSON escapes.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run_cli(args))
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=not args.unicode, indent=2))
        return
    summary = {key: payload[key] for key in ("case_count", "pass_rate", "hit_at_1", "hit_at_3", "hit_at_5", "mrr")}
    print(json.dumps(summary, ensure_ascii=not args.unicode, indent=2))


if __name__ == "__main__":
    main()
