from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine


@dataclass(frozen=True)
class AgentResult:
    data: dict[str, Any]


class BaseAgent:
    name: str = "base"

    async def _with_timeout(self, coro: Coroutine[Any, Any, Any], timeout_s: float = 30.0) -> Any:
        return await asyncio.wait_for(coro, timeout=timeout_s)

    async def _retry(self, fn: Callable[[], Coroutine[Any, Any, Any]], retries: int = 2) -> Any:
        last_exc: Exception | None = None
        for _ in range(retries + 1):
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 - boundary retry
                last_exc = exc
                await asyncio.sleep(0.2)
        if last_exc:
            raise last_exc
        raise RuntimeError("Retry failed without exception")

