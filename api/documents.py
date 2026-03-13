"""
api/documents.py
================
Document upload and analysis endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.dependencies import WorkflowDep
from api.schemas import DocumentAnalysisResponse
from core.config import get_settings
from core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from core.security import get_optional_user
from fastapi import Depends
from typing import Annotated
from core.security import CurrentUser

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
OptionalUser = Annotated[CurrentUser | None, Depends(get_optional_user)]


@router.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    summary="Upload and analyse a legal document (PDF, DOCX)",
)
async def analyze_document(
    workflow: WorkflowDep,
    user: OptionalUser,
    file: UploadFile = File(...),
    question: str = "วิเคราะห์เอกสารนี้และระบุประเด็นกฎหมายสำคัญ",
    case_id: str | None = None,
) -> dict:
    settings = get_settings()

    # Validate file type
    if file.content_type not in settings.allowed_mime_types:
        raise UnsupportedFileTypeError(
            f"File type '{file.content_type}' is not supported.",
            details={"allowed": list(settings.allowed_mime_types)},
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(
            f"File size {size_mb:.1f}MB exceeds limit of {settings.max_upload_size_mb}MB"
        )

    # Extract text from PDF (basic — production would use PyMuPDF)
    document_text = _extract_text(content, file.content_type)

    result = await workflow.orchestrate(
        question=question,
        case_id=case_id,
        document_text=document_text,
        user_id=user.sub if user else None,
        tenant_id=user.tenant_id if user else None,
    )

    return {
        "file_name": file.filename,
        "file_type": file.content_type,
        "analysis": result.response,
    }


def _extract_text(content: bytes, content_type: str) -> str:
    """Basic text extraction. In production use PyMuPDF for PDF, python-docx for DOCX."""
    if "text" in content_type:
        return content.decode("utf-8", errors="replace")
    # For PDF/DOCX: return placeholder — production uses full extraction pipeline
    return f"[Document content — {len(content)} bytes. Full extraction requires PyMuPDF.]"
