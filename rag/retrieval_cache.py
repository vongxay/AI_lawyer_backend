"""Redis cache helpers for legal retrieval results."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def retrieval_cache_key(
    *,
    query: str,
    jurisdiction: str | None,
    tenant_id: str | None,
    top_k: int,
    embedded: bool,
) -> str:
    raw = "|".join([
        jurisdiction or "any",
        tenant_id or "default",
        str(top_k),
        "1" if embedded else "0",
        query.strip().casefold()[:1200],
    ])
    return f"cache:retrieval:{hashlib.sha256(raw.encode()).hexdigest()}"


def serialise_chunks(chunks: list[dict[str, Any]]) -> str:
    return json.dumps(chunks, ensure_ascii=False, default=str)


def deserialise_chunks(payload: str) -> list[dict[str, Any]]:
    data = json.loads(payload)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]
