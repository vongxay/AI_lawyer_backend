from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    corrected_answer: str | None = None


@router.post("/")
async def submit_feedback(payload: FeedbackRequest) -> dict:
    return {"status": "ok", "received": payload.model_dump()}

