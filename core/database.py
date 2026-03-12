from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from redis import Redis

from backend.core.config import get_settings


if TYPE_CHECKING:
    from supabase import Client  # pragma: no cover


@dataclass(frozen=True)
class DbClients:
    supabase: "Client | None"
    redis: Redis


_clients: DbClients | None = None


def get_db_clients() -> DbClients:
    global _clients
    if _clients is not None:
        return _clients

    settings = get_settings()

    supabase_client = None
    if settings.supabase_url and settings.supabase_key:
        try:
            from supabase import create_client  # type: ignore

            supabase_client = create_client(settings.supabase_url, settings.supabase_key)
        except Exception:
            # Supabase client is optional in dev environments where wheels are unavailable.
            supabase_client = None

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)

    _clients = DbClients(supabase=supabase_client, redis=redis_client)
    return _clients

