"""
Legal evaluation benchmark service.

Runs curated legal QA cases through the same orchestration path as production
chat, then scores groundedness, citations, and confidence calibration.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    jurisdiction: str = "LA"
    language: str = "lo"
    category: str = "general"
    expected_answer: str | None = None
    required_citations: list[str] = field(default_factory=list)
    difficulty: str = "medium"


class LegalEvaluationService:
    def __init__(self, supabase: Any | None = None) -> None:
        self._supabase = supabase
        self._memory_cases: list[dict[str, Any]] = []

    async def list_cases(self, *, jurisdiction: str = "laos", limit: int = 100) -> list[dict[str, Any]]:
        if self._supabase:
            try:
                result = await (
                    self._supabase.table("legal_eval_cases")
                    .select("*")
                    .eq("jurisdiction", jurisdiction)
                    .eq("status", "active")
                    .limit(limit)
                    .execute()
                )
                return result.data or []
            except Exception as exc:
                log.warning("evaluation.list_cases.failed", error=str(exc))
        return self._memory_cases[:limit]

    async def create_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        case = {
            "id": str(uuid.uuid4()),
            "jurisdiction": payload.get("jurisdiction", "laos"),
            "language": payload.get("language", "lo"),
            "category": payload.get("category", "general"),
            "question": payload["question"],
            "expected_answer": payload.get("expected_answer"),
            "required_citations": payload.get("required_citations", []),
            "fact_pattern": payload.get("fact_pattern", {}),
            "difficulty": payload.get("difficulty", "medium"),
            "status": "active",
        }

        if self._supabase:
            try:
                result = await self._supabase.table("legal_eval_cases").insert(case).execute()
                data = result.data[0] if isinstance(result.data, list) and result.data else result.data
                return data or case
            except Exception as exc:
                log.warning("evaluation.create_case.failed", error=str(exc))

        self._memory_cases.append(case)
        return case

    async def run(self, *, workflow: Any, cases: list[dict[str, Any]], user_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []

        for case in cases:
            result = await workflow.orchestrate(
                question=case["question"],
                case_id=None,
                jurisdiction=case.get("jurisdiction", "laos"),
                user_id=user_id,
                tenant_id=None,
            )
            response = result.response
            score = self._score_response(case, response)
            results.append({
                "case_id": case.get("id"),
                "question": case["question"],
                "confidence": response.get("confidence", result.confidence),
                "score": score,
                "passed": score["passed"],
                "citations": response.get("citations", []),
                "answer_quality": response.get("answer_quality", {}),
            })

        total = len(results)
        passed = sum(1 for item in results if item["passed"])
        citation_pass = sum(1 for item in results if item["score"]["citation_ok"]) / total if total else 0.0
        avg_conf = sum(float(item["confidence"]) for item in results) / total if total else 0.0
        run = {
            "id": str(uuid.uuid4()),
            "run_name": f"lao-eval-{int(time.time())}",
            "jurisdiction": "laos",
            "total_cases": total,
            "passed_cases": passed,
            "avg_confidence": round(avg_conf, 3),
            "citation_pass_rate": round(citation_pass, 3),
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "results": results,
        }

        if self._supabase:
            try:
                await self._supabase.table("legal_eval_runs").insert({
                    **run,
                    "created_by": user_id,
                }).execute()
            except Exception as exc:
                log.warning("evaluation.persist_run.failed", error=str(exc))

        return run

    def _score_response(self, case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        citations = response.get("citations") or []
        required = [str(item).casefold() for item in (case.get("required_citations") or [])]
        refs = [str(c.get("ref", "")).casefold() for c in citations if isinstance(c, dict)]
        verified_count = sum(1 for c in citations if isinstance(c, dict) and c.get("status") == "VERIFIED")

        required_ok = all(any(req in ref or ref in req for ref in refs) for req in required) if required else True
        citation_ok = bool(citations) and verified_count > 0 and required_ok
        quality = response.get("answer_quality") or {}
        grounded_ok = quality.get("level") in {"strong", "limited"}
        confidence = float(response.get("confidence", 0.0))
        confidence_ok = confidence >= 0.55 if citation_ok else confidence <= 0.5

        return {
            "citation_ok": citation_ok,
            "grounded_ok": grounded_ok,
            "confidence_ok": confidence_ok,
            "verified_citations": verified_count,
            "passed": citation_ok and grounded_ok and confidence_ok,
        }
