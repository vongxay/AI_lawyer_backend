from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBundle:
    retrieved_documents: str
    case_memory_summary: str


class ContextBuilder:
    async def build(self, *, retrieved_documents: str, case_memory_summary: str) -> ContextBundle:
        return ContextBundle(retrieved_documents=retrieved_documents, case_memory_summary=case_memory_summary)

