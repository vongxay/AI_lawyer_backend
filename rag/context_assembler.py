from __future__ import annotations


class ContextAssembler:
    async def assemble(self, chunks: list[dict], *, max_chunks: int = 12) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks[:max_chunks], start=1):
            chunk_type = str(chunk.get("type") or chunk.get("doc_type") or "document").upper()
            title = str(chunk.get("title") or chunk.get("law_title") or "Untitled").strip()
            section = str(chunk.get("section") or chunk.get("article") or chunk.get("section_ref") or "").strip()
            law_no = str(chunk.get("law_no") or chunk.get("metadata", {}).get("law_no") or "").strip()
            source = str(chunk.get("source_url") or chunk.get("official_source_url") or "").strip()
            header_bits = [f"[{index}] {chunk_type}: {title}"]
            if law_no:
                header_bits.append(f"Law No. {law_no}")
            if section:
                header_bits.append(f"Section/Article {section}")
            if source:
                header_bits.append(f"Source: {source}")
            content = str(chunk.get("content") or "").strip()
            parts.append("\n".join([", ".join(header_bits), content]))
        return "\n\n---\n\n".join(parts)
