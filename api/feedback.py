"""
api/feedback.py
===============
User feedback collection endpoint.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas import FeedbackRequest
from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, require_roles

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])
log = get_logger(__name__)
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.post("/", summary="Submit feedback on a legal query response")
async def submit_feedback(
    payload: FeedbackRequest,
    user: AuthUser,
) -> dict:
    supabase = await get_supabase()

    entry = {
        "tenant_id": user.tenant_id,
        "session_id": payload.session_id,
        "message_id": payload.message_id,
        "user_id": user.sub,
        "rating": payload.rating,
        "feedback_type": "positive" if payload.rating >= 4 else "negative" if payload.rating <= 2 else "neutral",
        "comment": payload.comment,
        "corrected_answer": payload.corrected_answer,
    }

    if supabase and payload.message_id:
        try:
            await supabase.table("feedback").insert(entry).execute()
        except Exception as exc:
            log.warning("feedback.persist.failed", error=str(exc))

    log.info("feedback.received", session_id=payload.session_id, rating=payload.rating)
    return {"status": "ok", "session_id": payload.session_id}
