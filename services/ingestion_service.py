"""
services/ingestion_service.py
=============================
Admin legal document ingestion pipeline.
"""
from __future__ import annotations

import io
import os
import re
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

from core.config import get_settings
from core.exceptions import ExternalServiceError, ProviderNotConfiguredError, UnsupportedFileTypeError
from core.jurisdiction import canonical_jurisdiction, needs_multilingual_embedding, short_jurisdiction
from core.logging import get_logger
from services.llm_service import EmbeddingService

log = get_logger(__name__)

LAO_LAW_CATEGORY_IDS = {
    "constitution_justice",
    "state_security",
    "economy",
    "social_culture",
    "foreign_affairs",
}
DEFAULT_LAO_LAW_CATEGORY = "constitution_justice"
INGESTION_PIPELINE_VERSION = 6


@dataclass(frozen=True)
class IngestionInput:
    filename: str
    content_type: str
    content: bytes
    document_type: str
    jurisdiction: str
    law_category: str | None = None
    title: str | None = None
    year: int | None = None
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None
    law_no: str | None = None
    article: str | None = None
    gazette_date: str | None = None
    effective_date: str | None = None
    language: str | None = None
    review_status: str = "pending_review"
    tenant_id: str | None = None
    user_id: str | None = None
    allow_short_text: bool = False


@dataclass(frozen=True)
class IngestionResult:
    job_id: str
    document_id: str | None
    source_table: str
    title: str
    status: str
    chunks: int
    chunks_indexed: int
    chunks_embedded: int
    text_length: int
    embedding_model: str | None
    review_status: str
    document_type: str
    law_category: str
    jurisdiction: str
    extraction_method: str | None = None
    language: str | None = None
    text_quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalTextChunk:
    index: int
    content: str
    section_ref: str | None
    token_count: int


@dataclass(frozen=True)
class LegalStructureReport:
    article_count: int
    max_article_number: int | None
    missing_articles: tuple[str, ...] = ()
    duplicate_sections: tuple[str, ...] = ()
    out_of_order_sections: int = 0
    warnings: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "article_count": self.article_count,
            "max_article_number": self.max_article_number,
            "missing_articles": list(self.missing_articles),
            "duplicate_sections": list(self.duplicate_sections),
            "out_of_order_sections": self.out_of_order_sections,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TextQualityReport:
    score: float
    language: str
    char_count: int
    lao_ratio: float
    thai_ratio: float
    latin_ratio: float
    symbol_ratio: float
    mojibake_ratio: float
    repeated_symbol_runs: int
    suspicious_latin_tokens: int
    legal_marker_count: int
    warnings: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        if self.char_count < 80:
            return True
        if (
            self.language == "lo"
            and self.char_count >= 300
            and (self.thai_ratio >= 0.15 or self.thai_ratio > max(0.05, self.lao_ratio * 0.45))
        ):
            return False
        return self.score >= 0.58 and self.mojibake_ratio <= 0.12

    @property
    def needs_review(self) -> bool:
        return self.char_count >= 80 and self.score < 0.78

    def to_metadata(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "language": self.language,
            "char_count": self.char_count,
            "lao_ratio": round(self.lao_ratio, 3),
            "thai_ratio": round(self.thai_ratio, 3),
            "latin_ratio": round(self.latin_ratio, 3),
            "symbol_ratio": round(self.symbol_ratio, 3),
            "mojibake_ratio": round(self.mojibake_ratio, 3),
            "repeated_symbol_runs": self.repeated_symbol_runs,
            "suspicious_latin_tokens": self.suspicious_latin_tokens,
            "legal_marker_count": self.legal_marker_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ExtractedLegalText:
    text: str
    method: str
    quality: TextQualityReport
    warnings: list[str] = field(default_factory=list)


class LegalDocumentIngestionService:
    def __init__(self, supabase: Any | None) -> None:
        self._supabase = supabase
        self._settings = get_settings()
        self._embedder = EmbeddingService()

    async def ingest(self, item: IngestionInput) -> IngestionResult:
        if item.content_type not in self._settings.allowed_mime_types:
            raise UnsupportedFileTypeError(
                f"File type '{item.content_type}' is not supported.",
                details={"allowed": sorted(self._settings.allowed_mime_types)},
            )

        title = item.title or _title_from_filename(item.filename)
        extracted_text = extract_text_with_metadata(
            item.content,
            item.content_type,
            item.filename,
            jurisdiction=item.jurisdiction,
            language_hint=item.language,
        )
        text = extracted_text.text
        warnings: list[str] = list(extracted_text.warnings)
        if len(text.strip()) < 20:
            if not item.allow_short_text:
                raise UnsupportedFileTypeError(
                    "Could not extract enough text from this document.",
                    details={"file_name": item.filename, "content_type": item.content_type},
                )
            warnings.append(
                "Document text is very short; add the full legal text before relying on search or AI answers."
            )
        if not item.allow_short_text and not extracted_text.quality.is_usable:
            raise UnsupportedFileTypeError(
                "Extracted text quality is too low for trusted Lao legal search.",
                details={
                    "file_name": item.filename,
                    "content_type": item.content_type,
                    "extraction_method": extracted_text.method,
                    "text_quality": extracted_text.quality.to_metadata(),
                    "warnings": extracted_text.warnings,
                    "hint": (
                        "Upload a Unicode text-searchable PDF/DOCX/TXT, or enable Tesseract OCR "
                        "with Lao language data for scanned PDFs."
                    ),
                },
            )
        if extracted_text.quality.needs_review:
            warnings.append(
                "Extracted text quality is below the preferred threshold; review the indexed chunks before approval."
            )

        chunks = chunk_legal_text(
            text,
            max_chars=self._settings.rag_chunk_max_chars,
            overlap=self._settings.rag_chunk_overlap_chars,
        )
        structure = assess_legal_structure(chunks)
        warnings.extend(structure.warnings)
        embedding = await self._embed_text(text, jurisdiction=item.jurisdiction)
        source_table = _source_table(item.document_type)
        law_category = normalise_lao_law_category(item.law_category)
        legal_metadata = infer_lao_legal_metadata(text, title=title, source_url=item.source_url)
        job_id = str(uuid.uuid4())

        if not self._supabase:
            warnings = [
                *warnings,
                *embedding["warnings"],
                "Supabase is not configured; document was processed but not persisted.",
            ]
            return IngestionResult(
                job_id=job_id,
                document_id=None,
                source_table=source_table,
                title=title,
                status="processed_without_database",
                chunks=len(chunks),
                chunks_indexed=0,
                chunks_embedded=0,
                text_length=len(text),
                embedding_model=embedding["model"],
                review_status=item.review_status,
                document_type=item.document_type,
                law_category=law_category,
                jurisdiction=_short_jurisdiction(item.jurisdiction),
                extraction_method=extracted_text.method,
                language=extracted_text.quality.language,
                text_quality=extracted_text.quality.to_metadata(),
                warnings=warnings,
            )

        document_id = await self._insert_document(
            source_table=source_table,
            title=title,
            text=text,
            item=item,
            embedding=embedding["vector"],
            chunks=chunks,
            extraction=extracted_text,
            legal_metadata=legal_metadata,
            legal_structure=structure,
        )
        chunk_embedding = await self._embed_chunks(chunks, jurisdiction=item.jurisdiction)
        chunks_indexed, chunk_index_warnings = await self._insert_chunks(
            source_table=source_table,
            document_id=document_id,
            title=title,
            item=item,
            chunks=chunks,
            embeddings=chunk_embedding["vectors"],
            extraction=extracted_text,
            legal_metadata=legal_metadata,
            law_category=law_category,
        )

        log.info(
            "ingestion.completed",
            job_id=job_id,
            document_id=document_id,
            source_table=source_table,
            chunks=len(chunks),
            chunks_indexed=chunks_indexed,
            text_length=len(text),
        )

        return IngestionResult(
            job_id=job_id,
            document_id=document_id,
            source_table=source_table,
            title=title,
            status="indexed",
            chunks=len(chunks),
            chunks_indexed=chunks_indexed,
            chunks_embedded=chunk_embedding["embedded"],
            text_length=len(text),
            embedding_model=chunk_embedding["model"] or embedding["model"],
            review_status=item.review_status,
            document_type=item.document_type,
            law_category=law_category,
            jurisdiction=_short_jurisdiction(item.jurisdiction),
            extraction_method=extracted_text.method,
            language=extracted_text.quality.language,
            text_quality=extracted_text.quality.to_metadata(),
            warnings=[*warnings, *embedding["warnings"], *chunk_embedding["warnings"], *chunk_index_warnings],
        )

    async def _embed_text(self, text: str, *, jurisdiction: str) -> dict[str, Any]:
        embedding_text = text[:8000]
        try:
            result = await self._embedder.embed(
                embedding_text,
                multilingual=_needs_multilingual_embedding(embedding_text, jurisdiction),
            )
            return {"vector": result.vector, "model": result.model, "warnings": []}
        except ProviderNotConfiguredError as exc:
            log.warning("ingestion.embedding.skipped", error=str(exc))
            return {
                "vector": None,
                "model": None,
                "warnings": ["Embedding provider is not configured; vector embedding was not generated."],
            }
        except Exception as exc:
            shorter_text = text[:2500]
            if shorter_text and shorter_text != embedding_text:
                try:
                    result = await self._embedder.embed(
                        shorter_text,
                        multilingual=_needs_multilingual_embedding(shorter_text, jurisdiction),
                    )
                    log.warning("ingestion.embedding.retried_shorter_text", error=str(exc))
                    return {
                        "vector": result.vector,
                        "model": result.model,
                        "warnings": [
                            "Full-document embedding needed a shorter excerpt; "
                            "chunk embeddings are still the primary RAG index."
                        ],
                    }
                except Exception as retry_exc:
                    log.warning(
                        "ingestion.embedding.shorter_retry_failed",
                        error=str(retry_exc),
                    )
            log.warning("ingestion.embedding.failed", error=str(exc))
            return {
                "vector": None,
                "model": None,
                "warnings": [f"Embedding provider failed; document was indexed for keyword search only: {exc}"],
            }

    async def _embed_chunks(self, chunks: list[LegalTextChunk], *, jurisdiction: str) -> dict[str, Any]:
        if not chunks:
            return {"vectors": [], "embedded": 0, "model": None, "warnings": []}

        texts = [chunk.content[:8000] for chunk in chunks]
        multilingual = any(_needs_multilingual_embedding(text, jurisdiction) for text in texts[:20])
        batch_size = max(1, self._settings.rag_embedding_batch_size)
        vectors: list[list[float] | None] = []
        model: str | None = None

        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                results = await self._embedder.embed_many(batch, multilingual=multilingual)
                vectors.extend(result.vector for result in results)
                model = model or (results[0].model if results else None)
            return {
                "vectors": vectors,
                "embedded": len([vector for vector in vectors if vector]),
                "model": model,
                "warnings": [],
            }
        except ProviderNotConfiguredError as exc:
            log.warning("ingestion.chunk_embedding.skipped", error=str(exc))
            return {
                "vectors": [None for _ in chunks],
                "embedded": 0,
                "model": None,
                "warnings": ["Embedding provider is not configured; chunk vectors were not generated."],
            }
        except Exception as exc:
            log.warning("ingestion.chunk_embedding.failed", error=str(exc))
            return {
                "vectors": [None for _ in chunks],
                "embedded": 0,
                "model": None,
                "warnings": [f"Embedding provider failed; chunks were indexed for keyword search only: {exc}"],
            }

    async def _insert_document(
        self,
        *,
        source_table: str,
        title: str,
        text: str,
        item: IngestionInput,
        embedding: list[float] | None,
        chunks: list[LegalTextChunk],
        extraction: ExtractedLegalText,
        legal_metadata: dict[str, Any],
        legal_structure: LegalStructureReport,
    ) -> str:
        law_category = normalise_lao_law_category(item.law_category)
        document_article = _document_article_metadata(
            explicit_article=item.article,
            inferred_article=legal_metadata.get("article"),
            chunks=chunks,
        )
        metadata = {
            "file_name": item.filename,
            "content_type": item.content_type,
            "source_url": item.source_url,
            "official_source_url": item.source_url if _is_official_lao_source(item.source_url) else None,
            "source_authority": _source_authority(item.source_url),
            "language": item.language or legal_metadata.get("language"),
            "law_category": law_category,
            "law_no": item.law_no or legal_metadata.get("law_no"),
            "article": document_article,
            "gazette_date": item.gazette_date or legal_metadata.get("gazette_date"),
            "effective_date": item.effective_date or legal_metadata.get("effective_date"),
            "review_status": item.review_status,
            "chunks": len(chunks),
            "text_length": len(text),
            "extraction_method": extraction.method,
            "text_quality": extraction.quality.to_metadata(),
            "legal_structure": legal_structure.to_metadata(),
            "extraction_warnings": extraction.warnings,
            "ingestion_version": INGESTION_PIPELINE_VERSION,
        }
        vector = _vector_literal(embedding) if embedding else None

        payloads = _build_insert_payloads(
            source_table=source_table,
            title=title,
            text=text,
            item=item,
            metadata=metadata,
            vector=vector,
        )
        payloads = payloads + [_legacy_payload(payload) for payload in payloads]

        last_error: Exception | None = None
        for payload in payloads:
            try:
                result = await self._supabase.table(source_table).insert(payload).execute()
                data = result.data[0] if isinstance(result.data, list) and result.data else result.data
                doc_id = data.get("id") if isinstance(data, dict) else None
                if doc_id:
                    return str(doc_id)
            except Exception as exc:
                last_error = exc
                log.warning(
                    "ingestion.insert_variant.failed",
                    source_table=source_table,
                    error=str(exc),
                    columns=list(payload.keys()),
                )

        raise ExternalServiceError(f"Could not persist document in {source_table}: {last_error}")

    async def _insert_chunks(
        self,
        *,
        source_table: str,
        document_id: str,
        title: str,
        item: IngestionInput,
        chunks: list[LegalTextChunk],
        embeddings: list[list[float] | None],
        extraction: ExtractedLegalText,
        legal_metadata: dict[str, Any],
        law_category: str,
    ) -> tuple[int, list[str]]:
        if not chunks:
            return 0, []

        payloads: list[dict[str, Any]] = []
        status = _active_status_for_review(item.review_status)
        language = item.language or legal_metadata.get("language") or extraction.quality.language
        law_no = item.law_no or legal_metadata.get("law_no")
        document_article = item.article or legal_metadata.get("article")
        for chunk in chunks:
            embedding = embeddings[chunk.index] if chunk.index < len(embeddings) else None
            chunk_quality = assess_lao_legal_text_quality(chunk.content)
            chunk_article, chunk_article_source = _chunk_article_metadata(
                chunk,
                document_article=document_article,
                total_chunks=len(chunks),
            )
            payloads.append({
                "tenant_id": item.tenant_id,
                "source_table": source_table,
                "source_id": document_id,
                "document_type": item.document_type,
                "law_category": law_category,
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "title": title,
                "chunk_index": chunk.index,
                "section_ref": chunk.section_ref,
                "chapter_ref": _chapter_ref_from_section(chunk.section_ref),
                "law_no": law_no,
                "article": chunk_article,
                "language": language,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "status": status,
                "review_status": item.review_status,
                "metadata": {
                    "file_name": item.filename,
                    "source_url": item.source_url,
                    "tags": item.tags,
                    "law_category": law_category,
                    "law_no": law_no,
                    "article": chunk_article,
                    "article_source": chunk_article_source,
                    "document_article": document_article,
                    "language": language,
                    "extraction_method": extraction.method,
                    "document_text_quality": extraction.quality.to_metadata(),
                    "chunk_text_quality": chunk_quality.to_metadata(),
                    "ingestion_version": INGESTION_PIPELINE_VERSION,
                    "chunking_strategy": "lao_legal_section_paragraph_v2",
                },
                "embedding": _vector_literal(embedding) if embedding else None,
            })

        inserted = 0
        try:
            for start in range(0, len(payloads), 100):
                batch = payloads[start:start + 100]
                result = await self._supabase.table("document_chunks").insert(batch).execute()
                inserted += len(result.data or batch)
        except Exception as exc:
            log.warning("ingestion.chunk_insert.failed", error=str(exc))
            legacy_payloads = [_legacy_chunk_payload(payload) for payload in payloads]
            if legacy_payloads != payloads:
                try:
                    for start in range(0, len(legacy_payloads), 100):
                        batch = legacy_payloads[start:start + 100]
                        result = await self._supabase.table("document_chunks").insert(batch).execute()
                        inserted += len(result.data or batch)
                    return inserted, [
                        "document_chunks table accepted a legacy schema; "
                        "apply supabase_lao_law_categories.sql for category-level RAG."
                    ]
                except Exception as retry_exc:
                    log.warning("ingestion.chunk_insert_legacy.failed", error=str(retry_exc))
            return inserted, [
                "document_chunks table is not available; apply supabase_agentic_rag_chunks.sql "
                "and supabase_lao_law_categories.sql to enable chunk-level RAG."
            ]

        return inserted, []


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    return extract_text_with_metadata(content, content_type, filename).text


def extract_text_with_metadata(
    content: bytes,
    content_type: str,
    filename: str,
    *,
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> ExtractedLegalText:
    lower_name = filename.lower()

    if content_type in {"text/plain", "text/markdown"} or lower_name.endswith((".txt", ".md")):
        raw_text, decode_warnings = _decode_text_bytes(content)
        text = normalise_lao_legal_text(raw_text)
        quality = assess_lao_legal_text_quality(text)
        return ExtractedLegalText(
            text=text,
            method="plain_text",
            quality=quality,
            warnings=[*decode_warnings, *_quality_warnings(quality)],
        )

    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return _extract_pdf(content, jurisdiction=jurisdiction, language_hint=language_hint)

    if (
        content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or lower_name.endswith(".docx")
    ):
        text = normalise_lao_legal_text(_extract_docx(content))
        quality = assess_lao_legal_text_quality(text)
        return ExtractedLegalText(
            text=text,
            method="docx",
            quality=quality,
            warnings=_quality_warnings(quality),
        )

    if content_type == "application/msword" or lower_name.endswith(".doc"):
        raw_text, decode_warnings = _decode_text_bytes(content)
        text = normalise_lao_legal_text(raw_text)
        quality = assess_lao_legal_text_quality(text)
        return ExtractedLegalText(
            text=text,
            method="legacy_doc_text_decode",
            quality=quality,
            warnings=[
                "Legacy .doc extraction is best-effort; upload DOCX, Unicode TXT, or searchable PDF when possible.",
                *decode_warnings,
                *_quality_warnings(quality),
            ],
        )

    raise UnsupportedFileTypeError(f"Unsupported legal document type '{content_type}'.")


ARTICLE_LABEL_PATTERN = (
    r"(?:มาตรา|ມາດຕາ|Article|Art\.?|Section|Sec\.?)"
)

SECTION_HEADING_RE = re.compile(
    r"(?im)^(?P<section>"
    r"(?:มาตรา|ข้อ|หมวด|บทที่|ມາດຕາ|ຂໍ້|ຫມວດ|ໝວດ|ພາກ|ບົດທີ|Article|Art\.|Section|Sec\.|Chapter|Part)"
    r"\s+[0-9A-Za-zก-๙ກ-ຮ./-]+"
    r")"
)
SPLIT_ARTICLE_NUMBER_RE = re.compile(
    rf"(?im)^(\s*{ARTICLE_LABEL_PATTERN}[ \t]+)([1-9])[ \t]+([0-9]{{2,3}})(?=[ \t]+[^\d\s]|[ \t]*$)"
)
ARTICLE_REF_RE = re.compile(
    rf"(?i){ARTICLE_LABEL_PATTERN}\s*([0-9A-Za-zก-๙ກ-ຮ./-]+)"
)


THAI_BLOCK_RE = re.compile(r"[\u0e00-\u0e7f]")
LAO_BLOCK_RE = re.compile(r"[\u0e80-\u0eff]")
LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,}\b")
REPEATED_SYMBOL_RUN_RE = re.compile(r"(?:\. ?){6,}|[-_]{8,}|[\"'“”+]{4,}|[=|<>\\/]{4,}")
MOJIBAKE_CHAR_RE = re.compile(r"[\ufffd\u00c0-\u00ff\u201a\u201e\u2212]")
LAO_LEGAL_MARKER_RE = re.compile(
    r"(?:ກົດໝາຍ|ມາດຕາ|ດຳລັດ|ຄຳສັ່ງ|ລະບຽບ|ສິດ|ພັນທະ|ລັດ|ສປປ|ສານ|ທີ່ດິນ)"
)
PDF_SYMBOL_CHARS = set(".-_*+=|\\/\"'“”<>[]{}:;")
LATIN_NOISE_ALLOWLIST = {
    "article",
    "art",
    "chapter",
    "part",
    "section",
    "sec",
    "no",
    "law",
    "decree",
    "regulation",
    "page",
    "lao",
    "laos",
    "pdf",
    "spp",
}
MIN_PDF_TEXT_QUALITY = 0.62
PREFERRED_PDF_TEXT_QUALITY = 0.78


@dataclass(frozen=True)
class _ExtractionCandidate:
    method: str
    text: str
    quality: TextQualityReport
    warnings: list[str]


def normalise_lao_legal_text(text: str, *, source: str | None = None) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\u200b-\u200f\ufeff]", "", normalized)
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)
    if source == "pdf":
        normalized = _drop_pdf_artifact_lines(normalized)
    normalized = _repair_split_article_heading_numbers(normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _repair_split_article_heading_numbers(text: str) -> str:
    """Repair OCR headings like 'ມາດຕາ 1 14' so section parsing sees article 114."""

    def replace(match: re.Match[str]) -> str:
        number = f"{match.group(2)}{match.group(3)}"
        if int(number) > 999:
            return match.group(0)
        return f"{match.group(1)}{number}"

    return SPLIT_ARTICLE_NUMBER_RE.sub(replace, text or "")


def _decode_text_bytes(content: bytes) -> tuple[str, list[str]]:
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig", errors="replace"), []
    if content.startswith(b"\xff\xfe") or content.startswith(b"\xfe\xff"):
        return content.decode("utf-16", errors="replace"), []

    try:
        text = content.decode("utf-8")
        quality = assess_lao_legal_text_quality(text)
        if quality.mojibake_ratio <= 0.01:
            return text, []
    except UnicodeDecodeError:
        pass

    null_ratio = content.count(b"\x00") / max(1, len(content))
    if null_ratio > 0.05:
        for encoding in ("utf-16-le", "utf-16-be", "utf-16"):
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                continue
            quality = assess_lao_legal_text_quality(text)
            if quality.mojibake_ratio <= 0.01 and quality.char_count >= 20:
                return text, [f"Decoded text as {encoding}; verify original file encoding."]

    text = content.decode("utf-8", errors="replace")
    return text, ["Text encoding could not be detected cleanly; decoded as UTF-8 with replacement characters."]


def _drop_pdf_artifact_lines(text: str) -> str:
    cleaned: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        previous_blank = False
        if _is_pdf_artifact_line(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _is_pdf_artifact_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    chars = [ch for ch in compact if not ch.isspace()]
    if not chars:
        return False

    if re.fullmatch(r"[\d\s.,;:()|/\\_\-+]+", compact):
        return True

    lao_count = len(LAO_BLOCK_RE.findall(compact))
    latin_tokens = LATIN_TOKEN_RE.findall(compact)
    suspicious_latin = [
        token for token in latin_tokens if token.lower() not in LATIN_NOISE_ALLOWLIST
    ]
    symbol_ratio = sum(1 for ch in chars if ch in PDF_SYMBOL_CHARS) / len(chars)
    has_repeated_symbols = bool(REPEATED_SYMBOL_RUN_RE.search(compact))

    if has_repeated_symbols and (re.search(r"\d+\s*$", compact) or symbol_ratio > 0.18):
        return True
    if len(chars) > 24 and symbol_ratio > 0.52:
        return True
    if suspicious_latin and lao_count < 8 and len(suspicious_latin) >= 2:
        return True
    return bool(len(chars) <= 6 and lao_count == 0 and latin_tokens)


def assess_lao_legal_text_quality(text: str) -> TextQualityReport:
    sample = normalise_lao_legal_text(text)[:12000]
    chars = [ch for ch in sample if not ch.isspace()]
    if not chars:
        return TextQualityReport(
            score=0.0,
            language="unknown",
            char_count=0,
            lao_ratio=0.0,
            thai_ratio=0.0,
            latin_ratio=0.0,
            symbol_ratio=0.0,
            mojibake_ratio=0.0,
            repeated_symbol_runs=0,
            suspicious_latin_tokens=0,
            legal_marker_count=0,
            warnings=("No readable text was extracted.",),
        )

    char_count = len(chars)
    lao_count = len(LAO_BLOCK_RE.findall(sample))
    thai_count = len(THAI_BLOCK_RE.findall(sample))
    latin_count = sum(1 for ch in chars if "A" <= ch <= "Z" or "a" <= ch <= "z")
    mojibake_count = len(MOJIBAKE_CHAR_RE.findall(sample))
    symbol_count = sum(
        1
        for ch in chars
        if not ch.isalnum() and not ("\u0e00" <= ch <= "\u0eff")
    )
    repeated_symbol_runs = len(REPEATED_SYMBOL_RUN_RE.findall(sample))
    legal_marker_count = len(LAO_LEGAL_MARKER_RE.findall(sample))

    latin_tokens = LATIN_TOKEN_RE.findall(sample)
    looks_lao = lao_count / char_count > 0.10 or legal_marker_count > 0
    suspicious_latin_tokens = 0
    if looks_lao:
        suspicious_latin_tokens = len([
            token for token in latin_tokens if token.lower() not in LATIN_NOISE_ALLOWLIST
        ])

    lao_ratio = lao_count / char_count
    thai_ratio = thai_count / char_count
    latin_ratio = latin_count / char_count
    symbol_ratio = symbol_count / char_count
    mojibake_ratio = mojibake_count / char_count
    looks_thai = thai_ratio > 0.10 and lao_ratio < 0.08

    score = 1.0
    score -= min(0.70, mojibake_ratio * 2.2)
    score -= min(0.38, repeated_symbol_runs * 0.035)
    score -= min(0.30, max(0.0, symbol_ratio - 0.25) * 1.4)
    if looks_lao:
        score -= min(0.45, max(0.0, thai_ratio - 0.015) * 3.0)
        if thai_ratio > max(0.05, lao_ratio * 0.35) and char_count > 250:
            score -= 0.12
        score -= min(0.35, max(0.0, latin_ratio - 0.08) * 1.8)
        score -= min(0.28, suspicious_latin_tokens / 90)
        if lao_ratio < 0.22 and char_count > 500:
            score -= 0.12
        if legal_marker_count == 0 and char_count > 700:
            score -= 0.08
    elif looks_thai:
        score -= min(0.25, max(0.0, latin_ratio - 0.08) * 1.2)
    elif latin_ratio < 0.18 and char_count > 500:
        score -= 0.08

    score += min(0.08, legal_marker_count * 0.01)
    score = max(0.0, min(1.0, score))
    if looks_lao:
        language = "lo"
    elif looks_thai:
        language = "th"
    else:
        language = "en"

    warnings: list[str] = []
    if mojibake_ratio > 0.03:
        warnings.append("Text contains mojibake/replacement characters; source PDF text layer may be corrupt.")
    if repeated_symbol_runs >= 4:
        warnings.append("Text contains many repeated symbol runs; PDF table-of-contents or scan artifacts may remain.")
    if looks_lao and suspicious_latin_tokens >= 10:
        warnings.append("Lao text contains many unexpected Latin OCR tokens.")
    if looks_lao and thai_ratio >= 0.02 and char_count >= 80:
        warnings.append(
            "Lao text contains Thai-script characters; OCR may have used Thai language data."
        )
    if looks_lao and thai_ratio > max(0.05, lao_ratio * 0.35) and char_count > 250:
        warnings.append("Thai-script contamination is high for a Lao legal document.")
    if looks_lao and lao_ratio < 0.22 and char_count > 500:
        warnings.append("Lao character coverage is low for a Lao legal document.")
    if score < PREFERRED_PDF_TEXT_QUALITY:
        warnings.append("Extracted text should be reviewed before approval.")

    return TextQualityReport(
        score=score,
        language=language,
        char_count=char_count,
        lao_ratio=lao_ratio,
        thai_ratio=thai_ratio,
        latin_ratio=latin_ratio,
        symbol_ratio=symbol_ratio,
        mojibake_ratio=mojibake_ratio,
        repeated_symbol_runs=repeated_symbol_runs,
        suspicious_latin_tokens=suspicious_latin_tokens,
        legal_marker_count=legal_marker_count,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _quality_warnings(quality: TextQualityReport) -> list[str]:
    return list(quality.warnings)


def _build_extraction_candidate(method: str, raw_text: str, *, source: str) -> _ExtractionCandidate | None:
    text = normalise_lao_legal_text(raw_text, source=source)
    if not text.strip():
        return None
    quality = assess_lao_legal_text_quality(text)
    return _ExtractionCandidate(
        method=method,
        text=text,
        quality=quality,
        warnings=_quality_warnings(quality),
    )


def _choose_best_candidate(candidates: list[_ExtractionCandidate]) -> _ExtractionCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=_extraction_candidate_rank)


def _extraction_candidate_rank(candidate: _ExtractionCandidate) -> tuple[float, int, int, int]:
    adjusted_score = candidate.quality.score - _thai_contamination_penalty(candidate.quality)
    method_priority = {
        "pymupdf_text_layer": 3,
        "pypdf_text_layer": 2,
        "tesseract_ocr": 1,
    }.get(candidate.method, 0)
    return (
        adjusted_score,
        candidate.quality.legal_marker_count,
        method_priority,
        len(candidate.text),
    )


def _thai_contamination_penalty(quality: TextQualityReport) -> float:
    if quality.language != "lo":
        return 0.0
    penalty = max(0.0, quality.thai_ratio - 0.015) * 1.8
    if quality.thai_ratio > max(0.05, quality.lao_ratio * 0.35) and quality.char_count > 250:
        penalty += 0.12
    return min(0.35, penalty)


def chunk_text(text: str, *, max_chars: int = 3200, overlap: int = 300) -> list[str]:
    return [chunk.content for chunk in chunk_legal_text(text, max_chars=max_chars, overlap=overlap)]


def assess_legal_structure(chunks: list[LegalTextChunk]) -> LegalStructureReport:
    articles: list[int] = []
    section_refs: list[str] = []

    for chunk in chunks:
        section_ref = chunk.section_ref or ""
        if not section_ref or section_ref == "Preamble":
            continue
        article = _article_number_as_int(_article_from_section(section_ref))
        if article is None:
            continue
        articles.append(article)
        section_refs.append(section_ref)

    unique_articles = _ordered_unique_ints(articles)
    max_article = max(unique_articles) if unique_articles else None
    missing: list[str] = []
    if max_article and max_article <= 1000:
        article_set = set(unique_articles)
        missing = [str(number) for number in range(1, max_article + 1) if number not in article_set]

    duplicate_sections = _duplicate_values(section_refs)
    out_of_order = sum(
        1
        for previous, current in zip(articles, articles[1:], strict=False)
        if current < previous
    )

    warnings: list[str] = []
    if missing:
        warnings.append(
            "Legal article sequence appears to have missing headings: "
            f"{', '.join(missing[:30])}{'...' if len(missing) > 30 else ''}."
        )
    if duplicate_sections:
        warnings.append(
            "Legal article headings appear more than once; review OCR/chunking before approval."
        )
    if out_of_order:
        warnings.append(
            "Legal article headings are out of order; scanned PDF OCR may have mixed columns or amendments."
        )

    return LegalStructureReport(
        article_count=len(unique_articles),
        max_article_number=max_article,
        missing_articles=tuple(missing),
        duplicate_sections=tuple(duplicate_sections[:50]),
        out_of_order_sections=out_of_order,
        warnings=tuple(warnings),
    )


def chunk_legal_text(text: str, *, max_chars: int = 2600, overlap: int = 350) -> list[LegalTextChunk]:
    text = _normalise_chunk_source_text(text)
    if not text:
        return []

    chunks: list[LegalTextChunk] = []
    max_chars = max(400, int(max_chars))
    overlap = max(0, int(overlap))
    raw_sections = _split_legal_sections(text)
    chunk_index = 0

    for section_ref, section_text in raw_sections:
        for part_index, part in enumerate(_split_with_overlap(section_text, max_chars=max_chars, overlap=overlap)):
            section = section_ref
            if section_ref and part_index:
                section = f"{section_ref} (continued {part_index + 1})"
            chunks.append(LegalTextChunk(
                index=chunk_index,
                content=part,
                section_ref=section,
                token_count=max(1, len(part) // 4),
            ))
            chunk_index += 1

    return chunks


def _normalise_chunk_source_text(text: str) -> str:
    """Keep legal structure readable while removing PDF line-wrap noise."""
    normalized = normalise_lao_legal_text(text)
    if not normalized:
        return ""

    blocks: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        blocks.append(re.sub(r"\s+", " ", " ".join(paragraph_lines)).strip())
        paragraph_lines.clear()

    for raw_line in normalized.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            flush_paragraph()
            continue
        if _is_legal_heading_line(line):
            flush_paragraph()
            blocks.append(line)
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return _trim_chunk_text("\n\n".join(block for block in blocks if block))


def _is_legal_heading_line(line: str) -> bool:
    return bool(SECTION_HEADING_RE.match(line.strip()))


def _split_legal_sections(text: str) -> list[tuple[str | None, str]]:
    text = _trim_chunk_text(text)
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)] if text else []

    sections: list[tuple[str | None, str]] = []
    first_start = matches[0].start()
    if first_start > 0:
        preamble = _trim_chunk_text(text[:first_start])
        if preamble:
            sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_ref = re.sub(r"\s+", " ", match.group("section")).strip()
        section_text = _trim_chunk_text(text[start:end])
        if section_text:
            sections.append((section_ref, section_text))

    return sections or ([(None, text)] if text else [])


def _split_with_overlap(text: str, *, max_chars: int, overlap: int) -> list[str]:
    clean = _trim_chunk_text(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    safe_overlap = min(overlap, max_chars // 2)
    current_blocks: list[str] = []

    for block in _paragraph_blocks(clean):
        if len(block) > max_chars:
            long_text = _join_chunk_blocks([*current_blocks, block]) if current_blocks else block
            chunks.extend(_split_long_block(long_text, max_chars=max_chars, overlap=safe_overlap))
            current_blocks = []
            continue

        if current_blocks and len(_join_chunk_blocks([*current_blocks, block])) > max_chars:
            chunks.append(_join_chunk_blocks(current_blocks))
            overlap_blocks = _tail_overlap_blocks(current_blocks, safe_overlap)
            current_blocks = (
                overlap_blocks
                if overlap_blocks and len(_join_chunk_blocks([*overlap_blocks, block])) <= max_chars
                else []
            )

        current_blocks.append(block)

    if current_blocks:
        chunks.append(_join_chunk_blocks(current_blocks))

    return [chunk for chunk in chunks if chunk]


def _paragraph_blocks(text: str) -> list[str]:
    return [
        _trim_chunk_text(part)
        for part in re.split(r"\n{2,}", text)
        if _trim_chunk_text(part)
    ]


def _join_chunk_blocks(blocks: list[str]) -> str:
    return _trim_chunk_text("\n\n".join(blocks))


def _tail_overlap_blocks(blocks: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    for block in reversed(blocks):
        candidate = [block, *tail]
        candidate_text = _join_chunk_blocks(candidate)
        if len(candidate_text) > overlap and tail:
            break
        if len(candidate_text) > overlap:
            break
        tail = candidate
    return tail


def _split_long_block(text: str, *, max_chars: int, overlap: int) -> list[str]:
    clean = _trim_chunk_text(text)
    if len(clean) <= max_chars:
        return [clean]

    parts: list[str] = []
    safe_overlap = min(overlap, max_chars // 2)
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        end = _find_chunk_boundary(clean, start=start, end=end, max_chars=max_chars)
        part = _trim_chunk_text(clean[start:end])
        if part:
            parts.append(part)
        if end >= len(clean):
            break
        next_start = max(0, end - safe_overlap)
        if next_start <= start:
            next_start = end
        start = _advance_past_combining_marks(clean, next_start)
    return parts


def _find_chunk_boundary(text: str, *, start: int, end: int, max_chars: int) -> int:
    if end >= len(text):
        return len(text)

    minimum = start + int(max_chars * 0.58)
    boundary = text.rfind("\n\n", start, end)
    if boundary >= minimum:
        return _advance_past_combining_marks(text, boundary)

    for separator in (". ", "। ", "。 ", "! ", "? ", "; ", " "):
        boundary = text.rfind(separator, start, end)
        if boundary >= minimum:
            return _advance_past_combining_marks(text, boundary + len(separator))

    return _advance_past_combining_marks(text, end)


def _advance_past_combining_marks(text: str, index: int) -> int:
    while 0 < index < len(text) and unicodedata.category(text[index]).startswith("M"):
        index += 1
    return index


def _trim_chunk_text(text: str) -> str:
    trimmed = re.sub(r"[ \t]+", " ", text or "")
    trimmed = re.sub(r" *\n *", "\n", trimmed)
    trimmed = re.sub(r"\n{3,}", "\n\n", trimmed)
    return trimmed.strip()


def _extract_pdf(
    content: bytes,
    *,
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> ExtractedLegalText:
    errors: list[str] = []
    settings = get_settings()
    candidates: list[_ExtractionCandidate] = []

    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            text = "\n\n".join(page.get_text("text") for page in doc)
        candidate = _build_extraction_candidate("pymupdf_text_layer", text, source="pdf")
        if candidate:
            candidates.append(candidate)
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(candidate.text):
                errors.append("PyMuPDF extracted low-quality Lao text from a PDF text layer.")
        else:
            errors.append("PyMuPDF extracted no text; the PDF may be scanned or image-only.")
    except ModuleNotFoundError:
        errors.append("PyMuPDF is not installed. Install it with: py -m pip install PyMuPDF")
    except Exception as exc:
        errors.append(f"PyMuPDF failed: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        candidate = _build_extraction_candidate("pypdf_text_layer", text, source="pdf")
        if candidate:
            candidates.append(candidate)
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(candidate.text):
                errors.append("pypdf extracted low-quality Lao text from a PDF text layer.")
        else:
            errors.append("pypdf extracted no text; the PDF may be scanned or image-only.")
    except ModuleNotFoundError:
        errors.append("pypdf is not installed. Install it with: py -m pip install pypdf")
    except Exception as exc:
        errors.append(f"pypdf failed: {exc}")

    best_before_ocr = _choose_best_candidate(candidates)
    should_try_ocr = (
        settings.pdf_ocr_enabled
        and (
            best_before_ocr is None
            or best_before_ocr.quality.score < PREFERRED_PDF_TEXT_QUALITY
            or _looks_like_garbled_pdf_text(best_before_ocr.text)
        )
    )

    ocr_error_start = len(errors)
    if should_try_ocr:
        text = _extract_pdf_with_ocr(
            content,
            errors=errors,
            jurisdiction=jurisdiction,
            language_hint=language_hint,
        )
        candidate = _build_extraction_candidate("tesseract_ocr", text, source="pdf")
        if candidate:
            candidates.append(candidate)
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(candidate.text):
                errors.append("OCR fallback extracted text that still looks low quality.")
    else:
        if not settings.pdf_ocr_enabled:
            errors.append("Local OCR fallback is disabled. Set PDF_OCR_ENABLED=true to enable scanned PDF OCR.")

    best = _choose_best_candidate(candidates)
    if best and best.text.strip():
        if settings.pdf_detect_garbled_text and (_looks_like_garbled_pdf_text(best.text) or not best.quality.is_usable):
            errors.append(
                f"Best PDF extraction candidate '{best.method}' is below quality threshold "
                f"(score={best.quality.score:.2f})."
            )
        else:
            warnings = list(best.warnings)
            if best.method == "tesseract_ocr":
                warnings.extend(errors[ocr_error_start:])
            if best.quality.needs_review:
                warnings.append(
                    f"PDF text quality score is {best.quality.score:.2f}; admin review is recommended before approval."
                )
            return ExtractedLegalText(
                text=best.text,
                method=best.method,
                quality=best.quality,
                warnings=list(dict.fromkeys(warnings)),
            )

    details = {
        "extractors": errors,
        "candidates": [
            {
                "method": candidate.method,
                "text_length": len(candidate.text),
                "quality": candidate.quality.to_metadata(),
            }
            for candidate in candidates
        ],
    }
    raise UnsupportedFileTypeError(
        "PDF text extraction failed. The PDF text layer appears scanned, missing, or garbled. "
        "Install Tesseract OCR with the document jurisdiction's language data, increase OCR quality, "
        "or upload a Unicode text-searchable PDF.",
        details=details,
    )


def _looks_like_garbled_pdf_text(text: str) -> bool:
    sample = re.sub(r"\s+", " ", text).strip()[:6000]
    chars = [ch for ch in sample if not ch.isspace()]
    if len(chars) < 80:
        return False

    quality = assess_lao_legal_text_quality(sample)
    if not quality.is_usable:
        return True
    if quality.score < MIN_PDF_TEXT_QUALITY:
        return True
    if quality.language == "lo" and quality.thai_ratio >= 0.06 and quality.char_count > 250:
        return True
    if (
        quality.language == "lo"
        and quality.needs_review
        and (
            quality.repeated_symbol_runs >= 3
            or quality.suspicious_latin_tokens >= 8
            or (quality.lao_ratio < 0.18 and quality.char_count > 500)
            or quality.thai_ratio >= 0.025
            or quality.symbol_ratio > 0.42
        )
    ):
        return True

    replacement_ratio = sample.count("\ufffd") / len(chars)
    if replacement_ratio > 0.01:
        return True

    if (
        quality.language == "lo"
        and quality.lao_ratio > 0.08
        and quality.thai_ratio < 0.025
        and quality.score >= PREFERRED_PDF_TEXT_QUALITY
    ):
        return False
    if quality.language == "th" and quality.score >= PREFERRED_PDF_TEXT_QUALITY:
        return False

    suspicious = 0
    weird_symbols = set("®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„")
    for ch in chars:
        code = ord(ch)
        if ch in weird_symbols or 0x00C0 <= code <= 0x00FF or code in {0x201A, 0x201E, 0x2212}:
            suspicious += 1

    suspicious_ratio = suspicious / len(chars)
    if suspicious_ratio > 0.45:
        return True

    mojibake_token_chars = r"A-Za-zÀ-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„"
    hyphen_noise = len(re.findall(rf"(?:[{mojibake_token_chars}]{{1,6}}-){{2,}}", sample))
    extended_tokens = len(
        re.findall(
            rf"[{mojibake_token_chars}]*[À-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„]"
            rf"[{mojibake_token_chars}]*",
            sample,
        )
    )

    if suspicious_ratio > 0.18 and hyphen_noise >= 2:
        return True
    return bool(suspicious_ratio > 0.28 and extended_tokens >= 20)


def _extract_pdf_with_ocr(
    content: bytes,
    *,
    errors: list[str],
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> str:
    try:
        import fitz
    except ModuleNotFoundError:
        errors.append("OCR fallback needs PyMuPDF. Install it with: py -m pip install PyMuPDF")
        return ""

    try:
        from PIL import Image
    except ModuleNotFoundError:
        errors.append("OCR fallback needs Pillow. Install it with: py -m pip install pillow")
        return ""

    try:
        import pytesseract
    except ModuleNotFoundError:
        errors.append("OCR fallback needs pytesseract. Install it with: py -m pip install pytesseract")
        return ""

    settings = get_settings()
    _configure_tesseract(pytesseract, errors=errors)
    ocr_language = _resolve_tesseract_languages(
        pytesseract,
        errors=errors,
        jurisdiction=jurisdiction,
        language_hint=language_hint,
    )
    if not ocr_language:
        return ""

    text_parts: list[str] = []

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            max_pages = int(settings.pdf_ocr_max_pages)
            page_count = len(doc) if max_pages <= 0 else min(len(doc), max(1, max_pages))
            scale = max(72, settings.pdf_ocr_dpi) / 72
            matrix = fitz.Matrix(scale, scale)

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                image = _prepare_ocr_image(image)
                text = _ocr_image_text(
                    image,
                    pytesseract,
                    language=ocr_language,
                    page_number=page_index + 1,
                    errors=errors,
                )
                if text.strip():
                    text_parts.append(text.strip())

            if len(doc) > page_count:
                errors.append(
                    f"OCR scanned only the first {page_count} of {len(doc)} pages. "
                    "Increase PDF_OCR_MAX_PAGES for longer scanned PDFs."
                )
    except Exception as exc:
        errors.append(f"OCR fallback failed: {exc}")
        return ""

    text = "\n\n".join(text_parts)
    if not text.strip():
        errors.append(
            "OCR fallback extracted no text; the scan may be low quality "
            "or Tesseract language data may be missing."
        )
    return text


def _prepare_ocr_image(image: Any) -> Any:
    try:
        from PIL import ImageEnhance, ImageOps

        prepared = ImageOps.grayscale(image)
        prepared = ImageOps.autocontrast(prepared)
        return ImageEnhance.Sharpness(prepared).enhance(1.25)
    except Exception:
        return image


def _configure_tesseract(pytesseract: Any, *, errors: list[str]) -> None:
    settings = get_settings()
    tesseract_cmd = settings.tesseract_cmd or _default_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        errors.append(
            "Tesseract executable was not found. Set TESSERACT_CMD or install Tesseract OCR."
        )

    tessdata_prefix = settings.tessdata_prefix or _default_tessdata_prefix()
    if tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = tessdata_prefix


def _resolve_tesseract_languages(
    pytesseract: Any,
    *,
    errors: list[str],
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> str:
    requested = [
        part.strip()
        for part in (get_settings().pdf_ocr_languages or "eng").split("+")
        if part.strip()
    ]
    if not requested:
        requested = ["eng"]
    selected = _select_ocr_languages(
        requested,
        jurisdiction=jurisdiction,
        language_hint=language_hint,
    )
    skipped = [language for language in requested if language not in selected]
    if skipped and ("tha" in skipped or "lao" in skipped):
        errors.append(
            "OCR language list was narrowed to "
            + "+".join(selected)
            + " for jurisdiction/language context; skipped: "
            + ", ".join(skipped)
            + "."
        )

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        errors.append(f"Could not inspect Tesseract language data: {exc}")
        return "+".join(selected)

    usable = [language for language in selected if language in available]
    missing = [language for language in selected if language not in available]
    if missing:
        errors.append(
            "Tesseract language data missing for: "
            + ", ".join(missing)
            + ". Install the traineddata files or set TESSDATA_PREFIX to the project tessdata directory."
        )
    if _requires_ocr_language("lao", jurisdiction=jurisdiction, language_hint=language_hint) and "lao" not in usable:
        errors.append("Lao OCR language data is required for Lao legal documents; refusing wrong-language OCR.")
        return ""
    if _requires_ocr_language("tha", jurisdiction=jurisdiction, language_hint=language_hint) and "tha" not in usable:
        errors.append("Thai OCR language data is required for Thai legal documents; refusing wrong-language OCR.")
        return ""
    if not usable:
        errors.append(
            "No requested OCR languages are available; refusing to OCR with the wrong language model."
        )
        return ""
    return "+".join(usable)


def _select_ocr_languages(
    requested: list[str],
    *,
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> list[str]:
    canonical = canonical_jurisdiction(jurisdiction)
    normalized_hint = _normalize_language_hint(language_hint)

    if normalized_hint == "lo" or canonical == "laos":
        return _dedupe_languages(["lao", "eng", *_other_requested_ocr_languages(requested)])
    if normalized_hint == "th" or canonical == "thailand":
        return _dedupe_languages(["tha", "eng", *_other_requested_ocr_languages(requested)])
    if normalized_hint == "en":
        return _dedupe_languages(["eng", *_other_requested_ocr_languages(requested)])
    return _dedupe_languages(requested or ["eng"])


def _other_requested_ocr_languages(requested: list[str]) -> list[str]:
    return [language for language in requested if language not in {"lao", "tha", "eng"}]


def _requires_ocr_language(
    language: str,
    *,
    jurisdiction: str | None = None,
    language_hint: str | None = None,
) -> bool:
    canonical = canonical_jurisdiction(jurisdiction)
    normalized_hint = _normalize_language_hint(language_hint)
    return (
        (language == "lao" and (normalized_hint == "lo" or canonical == "laos"))
        or (language == "tha" and (normalized_hint == "th" or canonical == "thailand"))
    )


def _normalize_language_hint(language_hint: str | None) -> str | None:
    if not language_hint:
        return None
    normalized = language_hint.strip().casefold().replace("_", "-")
    if normalized in {"lo", "la", "lao", "laos", "lao-pdr"}:
        return "lo"
    if normalized in {"th", "tha", "thai", "thailand"}:
        return "th"
    if normalized in {"en", "eng", "english"}:
        return "en"
    return normalized or None


def _dedupe_languages(languages: list[str]) -> list[str]:
    deduped: list[str] = []
    for language in languages:
        normalized = language.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped or ["eng"]


def _default_tesseract_cmd() -> str | None:
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _default_tessdata_prefix() -> str | None:
    project_tessdata = Path(__file__).resolve().parents[1] / "tessdata"
    candidates = [
        project_tessdata,
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _ocr_image_text(
    image: Any,
    pytesseract: Any,
    *,
    language: str,
    page_number: int,
    errors: list[str],
) -> str:
    try:
        return pytesseract.image_to_string(image, lang=language)
    except Exception as exc:
        errors.append(f"Tesseract OCR failed on page {page_number} with languages '{language}': {exc}")
        return ""


def _extract_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as docx:
            xml = docx.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            parts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            text = "".join(parts).strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise UnsupportedFileTypeError(f"DOCX text extraction failed: {exc}") from exc


def _source_table(document_type: str) -> str:
    normalized = document_type.strip().lower()
    if normalized in {"case", "case_law", "judgment"}:
        return "cases"
    if normalized in {"form", "template"}:
        return "legal_forms"
    return "laws"


def normalise_lao_law_category(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in LAO_LAW_CATEGORY_IDS:
        return normalized
    return DEFAULT_LAO_LAW_CATEGORY


def _chapter_ref_from_section(section_ref: str | None) -> str | None:
    if not section_ref:
        return None
    match = re.search(r"(?i)(?:chapter|part|หมวด|ພາກ|ຫມວດ|ໝວດ)\s*([0-9A-Za-zก-๙ກ-ຮ./-]+)", section_ref)
    return match.group(0).strip() if match else None


def _article_from_section(section_ref: str | None) -> str | None:
    if not section_ref:
        return None
    repaired = _repair_split_article_heading_numbers(section_ref)
    match = ARTICLE_REF_RE.search(repaired)
    if not match:
        return None
    return match.group(1).strip().rstrip(".,;:)")


def _chunk_article_metadata(
    chunk: LegalTextChunk,
    *,
    document_article: str | None,
    total_chunks: int,
) -> tuple[str | None, str | None]:
    section_article = _article_from_section(chunk.section_ref)
    if section_article and chunk.section_ref != "Preamble":
        return section_article, "section_ref"

    content_article = None if chunk.section_ref == "Preamble" else _article_from_section(chunk.content)
    if content_article:
        return content_article, "content"

    if total_chunks == 1 and document_article:
        return document_article, "document"
    return None, None


def _document_article_metadata(
    *,
    explicit_article: str | None,
    inferred_article: str | None,
    chunks: list[LegalTextChunk],
) -> str | None:
    explicit_article = (explicit_article or "").strip()
    if explicit_article:
        return explicit_article

    article_numbers = _ordered_unique_ints(
        _article_number_as_int(_article_from_section(chunk.section_ref))
        for chunk in chunks
        if chunk.section_ref and chunk.section_ref != "Preamble"
    )
    if len(article_numbers) > 1:
        return None
    if len(article_numbers) == 1:
        return str(article_numbers[0])

    inferred_article = (inferred_article or "").strip()
    return inferred_article or None


def _article_number_as_int(article: str | None) -> int | None:
    if not article:
        return None
    match = re.match(r"0*([0-9]{1,4})(?:\D|$)", article.strip())
    if not match:
        return None
    return int(match.group(1))


def _ordered_unique_ints(values: Any) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _duplicate_values(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    duplicates: list[str] = []
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if counts[value] == 2:
            duplicates.append(value)
    return duplicates


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return re.sub(r"\.[^.]+$", "", stem).replace("_", " ").replace("-", " ").strip() or filename


def _db_jurisdiction(value: str) -> str:
    return canonical_jurisdiction(value) or value.lower()


def _short_jurisdiction(value: str) -> str:
    return short_jurisdiction(value) or value


def _needs_multilingual_embedding(text: str, jurisdiction: str) -> bool:
    return needs_multilingual_embedding(text, jurisdiction)


def infer_lao_legal_metadata(text: str, *, title: str, source_url: str | None = None) -> dict[str, Any]:
    sample = f"{title}\n{text[:5000]}"
    law_no = _first_match(sample, [
        r"(?:Law\s+No\.?|No\.?)\s*([0-9A-Za-z/.-]+)",
        r"(?:ເລກທີ|ສະບັບເລກທີ)\s*([0-9A-Za-z/.-]+)",
    ])
    article = _first_match(sample, [
        r"(?:Article|Art\.?)\s*([0-9A-Za-z/.-]+)",
        r"(?:ມາດຕາ)\s*([0-9A-Za-z/.-]+)",
    ])
    return {
        "language": "lo" if any("\u0e80" <= ch <= "\u0eff" for ch in sample) else "en",
        "law_no": law_no,
        "article": article,
        "source_authority": _source_authority(source_url),
    }


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _source_authority(source_url: str | None) -> str:
    host = _source_host(source_url)
    if host in {"na.gov.la", "www.na.gov.la"} or host.endswith(".na.gov.la"):
        return "lao_national_assembly"
    if host in {"laoofficialgazette.gov.la", "www.laoofficialgazette.gov.la"}:
        return "lao_official_gazette"
    return "uploaded"


def _is_official_lao_source(source_url: str | None) -> bool:
    return _source_authority(source_url) in {"lao_national_assembly", "lao_official_gazette"}


def _source_host(source_url: str | None) -> str:
    if not source_url:
        return ""
    parsed = urlparse(source_url)
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    return host.lower().strip()


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def _active_status_for_review(review_status: str) -> str:
    return "active" if review_status == "approved" else "pending"


def _build_insert_payloads(
    *,
    source_table: str,
    title: str,
    text: str,
    item: IngestionInput,
    metadata: dict[str, Any],
    vector: str,
) -> list[dict[str, Any]]:
    tags = item.tags or []
    modern_status = _active_status_for_review(item.review_status)

    if source_table == "cases":
        return [
            {
                "tenant_id": item.tenant_id,
                "case_no": title,
                "court": "Unknown",
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "year_be": item.year,
                "language": metadata.get("language"),
                "law_category": metadata.get("law_category"),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "status": modern_status,
                "review_status": item.review_status,
                "ruling": text,
                "full_text": text,
                "summary": text[:1000],
                "tags": tags,
                "metadata": metadata,
                "embedding": vector,
                "ingested_by": item.user_id,
            },
            {
                "tenant_id": item.tenant_id,
                "case_no": title,
                "court": "Unknown",
                "year": item.year or 0,
                "language": metadata.get("language"),
                "law_category": metadata.get("law_category"),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "review_status": item.review_status,
                "summary": text[:1000],
                "ruling": text,
                "ratio_decidendi": text[:3000],
                "jurisdiction": _short_jurisdiction(item.jurisdiction),
                "metadata": metadata,
                "embedding": vector,
            },
        ]

    if source_table == "legal_forms":
        return [
            {
                "tenant_id": item.tenant_id,
                "title": title,
                "form_type": "general",
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "law_category": metadata.get("law_category"),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "review_status": item.review_status,
                "content": text,
                "tags": tags,
                "metadata": metadata,
                "embedding": vector,
                "is_active": item.review_status == "approved",
            },
            {
                "tenant_id": item.tenant_id,
                "title": title,
                "form_type": "general",
                "jurisdiction": _short_jurisdiction(item.jurisdiction),
                "language": _short_jurisdiction(item.jurisdiction),
                "law_category": metadata.get("law_category"),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "review_status": item.review_status,
                "content": text,
                "metadata": metadata,
                "embedding": vector,
                "is_active": item.review_status == "approved",
            },
        ]

    return [
        {
            "tenant_id": item.tenant_id,
            "title": title,
            "doc_type": "regulation" if item.document_type == "regulation" else "statute",
            "jurisdiction": _db_jurisdiction(item.jurisdiction),
            "year_be": item.year,
            "language": metadata.get("language"),
            "law_category": metadata.get("law_category"),
            "official_source_url": metadata.get("official_source_url"),
            "source_authority": metadata.get("source_authority"),
            "law_no": metadata.get("law_no"),
            "article": metadata.get("article"),
            "gazette_date": metadata.get("gazette_date"),
            "effective_date": metadata.get("effective_date"),
            "status": modern_status,
            "review_status": item.review_status,
            "full_text": text,
            "summary": text[:1000],
            "source_url": item.source_url,
            "tags": tags,
            "metadata": metadata,
            "embedding": vector,
            "ingested_by": item.user_id,
        },
        {
            "tenant_id": item.tenant_id,
            "title": title,
            "jurisdiction": _short_jurisdiction(item.jurisdiction),
            "year": item.year,
            "language": metadata.get("language"),
            "law_category": metadata.get("law_category"),
            "official_source_url": metadata.get("official_source_url"),
            "source_authority": metadata.get("source_authority"),
            "law_no": metadata.get("law_no"),
            "article": metadata.get("article"),
            "gazette_date": metadata.get("gazette_date"),
            "effective_date": metadata.get("effective_date"),
            "review_status": item.review_status,
            "full_text": text,
            "status": "ACTIVE",
            "metadata": metadata,
            "embedding": vector,
        },
    ]


def _legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    optional_columns = {
        "language",
        "law_category",
        "official_source_url",
        "source_authority",
        "law_no",
        "article",
        "gazette_date",
        "effective_date",
        "amended_by",
        "repealed_by",
        "review_status",
        "reviewed_by",
        "reviewed_at",
        "review_notes",
        "is_active",
        "metadata",
    }
    legacy = {key: value for key, value in payload.items() if key not in optional_columns}
    if payload.get("review_status") and legacy.get("status") == "pending":
        legacy["status"] = "active"
    return legacy


def _legacy_chunk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    optional_columns = {
        "law_category",
        "chapter_ref",
        "law_no",
        "article",
        "language",
    }
    return {key: value for key, value in payload.items() if key not in optional_columns}
