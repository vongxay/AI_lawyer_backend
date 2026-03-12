from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlmResult:
    text: str


class LlmService:
    """
    Thin abstraction layer for OpenAI/Anthropic (stubbed for now).
    Replace internals with real provider clients without changing callers.
    """

    async def generate(self, *, prompt: str, model: str) -> LlmResult:
        return LlmResult(text=f"[stub:{model}] {prompt[:200]}")

