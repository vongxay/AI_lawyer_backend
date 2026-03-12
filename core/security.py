from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jwt

from backend.core.config import get_settings


@dataclass(frozen=True)
class TokenPayload:
    sub: str
    exp: int
    role: str | None = None
    tenant_id: str | None = None


def create_access_token(subject: str, role: str | None = None, tenant_id: str | None = None, ttl_seconds: int = 3600) -> str:
    settings = get_settings()
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": now + ttl_seconds,
        "iat": now,
    }
    if role:
        payload["role"] = role
    if tenant_id:
        payload["tenant_id"] = tenant_id

    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> TokenPayload:
    settings = get_settings()
    decoded = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return TokenPayload(
        sub=str(decoded.get("sub")),
        exp=int(decoded.get("exp")),
        role=decoded.get("role"),
        tenant_id=decoded.get("tenant_id"),
    )

