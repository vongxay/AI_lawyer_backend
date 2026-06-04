"""
api/documents.py
================
Document upload and analysis endpoint.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import WorkflowDep
from api.schemas import DocumentAnalysisResponse
from core.config import get_settings
from core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from core.security import CurrentUser, require_roles
from services.ingestion_service import extract_text

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    summary="Upload and analyse a legal document (PDF, DOCX)",
)
async def analyze_document(
    workflow: WorkflowDep,
    user: AuthUser,
    file: UploadFile = File(...),
    question: str = Form("Analyze this legal document and identify the key legal issues."),
    case_id: str | None = Form(default=None),
) -> dict:
    settings = get_settings()
    content_type = file.content_type or "application/octet-stream"

    if content_type not in settings.allowed_mime_types:
        raise UnsupportedFileTypeError(
            f"File type '{content_type}' is not supported.",
            details={"allowed": sorted(settings.allowed_mime_types)},
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(
            f"File size {size_mb:.1f}MB exceeds limit of {settings.max_upload_size_mb}MB"
        )

    document_text = extract_text(content, content_type, file.filename or "uploaded-document")

    result = await workflow.orchestrate(
        question=question.strip(),
        case_id=case_id,
        document_text=document_text,
        user_id=user.sub,
        tenant_id=user.tenant_id,
    )

    return {
        "file_name": file.filename or "uploaded-document",
        "file_type": content_type,
        "text_length": len(document_text),
        "analysis": result.response,
    }
