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
from core.security import CurrentUser, get_optional_user

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])
log = get_logger(__name__)
OptionalUser = Annotated[CurrentUser | None, Depends(get_optional_user)]


@router.post("/", summary="Submit feedback on a legal query response")
async def submit_feedback(
    payload: FeedbackRequest,
    user: OptionalUser,
) -> dict:
    supabase = await get_supabase()

    entry = {
        "session_id": payload.session_id,
        "user_id": user.sub if user else None,
        "rating": payload.rating,
        "comment": payload.comment,
        "corrected_answer": payload.corrected_answer,
    }

    if supabase:
        try:
            await supabase.table("feedback").insert(entry).execute()
        except Exception as exc:
            log.warning("feedback.persist.failed", error=str(exc))

    log.info("feedback.received", session_id=payload.session_id, rating=payload.rating)
    return {"status": "ok", "session_id": payload.session_id}
