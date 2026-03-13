"""
agents/base_agent.py
=====================
Base class for all legal agents.

Every agent inherits:
- Async timeout enforcement
- Exponential-backoff retry via tenacity
- Structured logging with agent name and timing
- Fallback result on terminal failure
- Input PII redaction before any LLM call
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import get_settings
from backend.core.exceptions import AgentTimeoutError, ExternalServiceError
from backend.core.logging import get_logger
from backend.services.llm_service import LlmResult, LlmService, Message

T = TypeVar("T")
log = get_logger(__name__)


@dataclass(frozen=True)
class AgentResult:
    """
    Typed wrapper for every agent output.

    `data`       — the agent's primary output dict
    `confidence` — agent-specific confidence [0, 1]
    `agent_name` — set automatically by BaseAgent.run()
    `duration_ms`— wall-clock time for this agent
    `tokens_used`— sum of LLM tokens consumed
    """
    data: dict[str, Any]
    confidence: float = 1.0
    agent_name: str = ""
    duration_ms: int = 0
    tokens_used: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseAgent(ABC):
    """
    Abstract base for all legal agents.

    Subclasses implement `_execute` which is called with PII-clean input.
    The public `run` method handles the orchestration shell.
    """

    #: Override in subclasses
    name: str = "base"

    def __init__(self, llm: LlmService | None = None) -> None:
        self._llm = llm or LlmService()
        self._settings = get_settings()

    # ── Public interface ────────────────────────────────────────────────────────

    async def run(self, **kwargs) -> AgentResult:
        """
        Execute the agent with timeout + retry.

        Callers always get an AgentResult back — never a raw exception.
        On unrecoverable failure, error field is populated.
        """
        started = time.perf_counter()
        try:
            result = await self._with_timeout_and_retry(lambda: self._execute(**kwargs))
            duration = int((time.perf_counter() - started) * 1000)
            return AgentResult(
                data=result,
                confidence=result.pop("_confidence", 1.0),
                agent_name=self.name,
                duration_ms=duration,
                tokens_used=result.pop("_tokens", 0),
            )
        except AgentTimeoutError as exc:
            log.error("agent.timeout", agent=self.name, error=str(exc))
            return self._error_result(f"Agent '{self.name}' timed out", started)
        except ExternalServiceError as exc:
            log.error("agent.external_error", agent=self.name, error=str(exc))
            return self._error_result(str(exc), started)
        except Exception as exc:  # noqa: BLE001
            log.error("agent.unexpected_error", agent=self.name, error=str(exc), exc_info=True)
            return self._error_result(f"Unexpected error in agent '{self.name}'", started)

    # ── Protected interface for subclasses ──────────────────────────────────────

    @abstractmethod
    async def _execute(self, **kwargs) -> dict[str, Any]:
        """
        Core agent logic. Return a dict with at minimum:
            {"result": ...}

        Optionally include:
            {"_confidence": 0.85, "_tokens": 1234, ...}
        These keys are extracted by AgentResult and removed from `data`.
        """
        ...

    async def _call_llm(
        self,
        *,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> LlmResult:
        messages = [Message(role="user", content=user_message)]
        return await self._llm.generate(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )

    # ── Internal helpers ────────────────────────────────────────────────────────

    async def _with_timeout_and_retry(self, fn: Callable[[], Coroutine]) -> Any:
        timeout = self._settings.agent_timeout_seconds
        retries = self._settings.agent_max_retries

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(ExternalServiceError),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            stop=stop_after_attempt(retries + 1),
            reraise=True,
        ):
            with attempt:
                try:
                    return await asyncio.wait_for(fn(), timeout=timeout)
                except asyncio.TimeoutError:
                    raise AgentTimeoutError(
                        f"Agent '{self.name}' exceeded {timeout}s timeout"
                    )

    def _error_result(self, error_msg: str, started: float) -> AgentResult:
        duration = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            data={},
            confidence=0.0,
            agent_name=self.name,
            duration_ms=duration,
            error=error_msg,
        )
