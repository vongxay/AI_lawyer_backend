from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from backend.api.schemas import EvidenceAnalyzeResponse
from backend.orchestrator.workflow_manager import WorkflowManager


router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])
_workflow = WorkflowManager()


@router.post("/analyze", response_model=EvidenceAnalyzeResponse)
async def analyze_evidence(files: list[UploadFile] = File(...)) -> dict:
    _ = files
    result = await _workflow.orchestrate(question="evidence analysis", case_id=None, has_evidence=True)
    return {
        "items": result.response.get("risk", []) or [],
        "overall_strength": "UNKNOWN",
        "gaps": ["stub: evidence files received"],
    }

