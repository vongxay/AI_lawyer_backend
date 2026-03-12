from __future__ import annotations

from fastapi import APIRouter, File, UploadFile


router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> dict:
    return {
        "file_name": file.filename,
        "file_type": file.content_type,
        "analysis_result": {"summary": "stub document analysis"},
    }

