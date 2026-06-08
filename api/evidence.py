"""
api/evidence.py
===============
Evidence upload and multimodal analysis endpoint.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from agents.evidence_agent import EvidenceFile
from api.dependencies import WorkflowDep
from api.schemas import EvidenceAnalysisResponse
from api.upload_utils import read_upload_with_limit, upload_limit_bytes
from core.config import get_settings
from core.exceptions import UnsupportedFileTypeError
from core.security import CurrentUser, require_roles

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.post(
    "/analyze",
    response_model=EvidenceAnalysisResponse,
    summary="Upload evidence files (image/audio/email) for analysis",
)
async def analyze_evidence(
    workflow: WorkflowDep,
    user: AuthUser,
    files: list[UploadFile] = File(...),
    question: str = Form("Analyze this evidence and assess its legal relevance."),
    case_id: str | None = Form(default=None),
) -> dict:
    settings = get_settings()
    evidence_files: list[EvidenceFile] = []

    for upload in files:
        content_type = upload.content_type or "application/octet-stream"
        if content_type not in settings.allowed_mime_types:
            raise UnsupportedFileTypeError(
                f"'{upload.filename}' has unsupported type '{content_type}'",
                details={"allowed": sorted(settings.allowed_mime_types)},
            )

        content = await read_upload_with_limit(upload, max_bytes=upload_limit_bytes(settings))

        evidence_files.append(EvidenceFile(
            filename=upload.filename or "unnamed",
            content_type=content_type,
            content=content,
        ))

    result = await workflow.orchestrate(
        question=question.strip(),
        case_id=case_id,
        evidence_files=evidence_files,
        user_id=user.sub,
        tenant_id=user.tenant_id,
    )

    evidence = _normalise_evidence_data(result.response.get("evidence"), len(evidence_files))
    return evidence


def _normalise_evidence_data(data: Any, file_count: int) -> dict[str, Any]:
    if isinstance(data, dict) and data:
        return {
            "items": data.get("items") or [],
            "overall_strength": data.get("overall_strength") or "UNKNOWN",
            "gaps": data.get("gaps") or [],
            "evidence_summary": data.get("evidence_summary") or f"{file_count} file(s) analysed.",
        }

    return {
        "items": [],
        "overall_strength": "UNKNOWN",
        "gaps": ["Evidence agent did not return structured analysis."],
        "evidence_summary": "Evidence analysis unavailable; review the original files manually.",
    }
