"""
workers.py
==========
ARQ worker settings for background jobs.

The project does not enqueue background jobs yet, but keeping a valid worker
entrypoint prevents deployment from starting a broken container.
"""
from __future__ import annotations

from urllib.parse import urlparse

from arq.connections import RedisSettings

from core.config import get_settings


async def healthcheck_task(ctx: dict) -> str:
    _ = ctx
    return "ok"


def _redis_settings_from_url(url: str) -> RedisSettings:
    parsed = urlparse(url)
    database = int((parsed.path or "/0").lstrip("/") or "0")
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=database,
        password=parsed.password,
        ssl=parsed.scheme == "rediss",
    )


class WorkerSettings:
    functions = [healthcheck_task]
    redis_settings = _redis_settings_from_url(get_settings().redis_url)
    max_jobs = 10
    job_timeout = 300
