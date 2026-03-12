from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class IngestRequest(BaseModel):
    source: str = Field(min_length=1)
    jurisdiction: str | None = None


@router.post("/ingest")
async def ingest(payload: IngestRequest) -> dict:
    return {"status": "queued", "source": payload.source, "jurisdiction": payload.jurisdiction}


@router.get("/audit-log")
async def audit_log() -> list[dict]:
    return []


@router.get("/expert-queue")
async def expert_queue() -> list[dict]:
    return []

