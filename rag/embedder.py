from __future__ import annotations


class Embedder:
    async def embed(self, text: str) -> list[float]:
        # Stub embedding: deterministic small vector
        return [float((sum(text.encode("utf-8")) % 1000) / 1000.0)]

