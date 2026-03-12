from __future__ import annotations


class ContextAssembler:
    async def assemble(self, chunks: list[dict]) -> str:
        return "\n\n".join([c.get("content", "") for c in chunks])

