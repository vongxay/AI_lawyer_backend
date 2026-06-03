"""
Admin evaluation endpoints for Lao legal QA benchmarks.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.dependencies import WorkflowDep
from core.database import get_supabase
from core.security import CurrentUser, get_admin_user
from services.evaluation_service import LegalEvaluationService

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


class EvalCaseCreate(BaseModel):
    question: str = Field(min_length=5, max_length=5000)
    jurisdiction: str = "laos"
    language: str = "lo"
    category: str = "general"
    expected_answer: str | None = None
    required_citations: list[str] = Field(default_factory=list)
    fact_pattern: dict[str, Any] = Field(default_factory=dict)
    difficulty: str = "medium"


@router.get("/cases", summary="List legal evaluation cases")
async def list_eval_cases(
    user: AdminUser,
    jurisdiction: str = Query(default="laos"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    service = LegalEvaluationService(await get_supabase())
    cases = await service.list_cases(jurisdiction=jurisdiction, limit=limit)
    return {"items": cases, "total": len(cases)}


@router.post("/cases", summary="Create a legal evaluation case")
async def create_eval_case(payload: EvalCaseCreate, user: AdminUser) -> dict:
    service = LegalEvaluationService(await get_supabase())
    item = await service.create_case(payload.model_dump())
    return item


@router.post("/runs", summary="Run legal QA evaluation benchmark")
async def run_evaluation(
    workflow: WorkflowDep,
    user: AdminUser,
    jurisdiction: str = Query(default="laos"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    service = LegalEvaluationService(await get_supabase())
    cases = await service.list_cases(jurisdiction=jurisdiction, limit=limit)
    return await service.run(workflow=workflow, cases=cases, user_id=user.sub)
