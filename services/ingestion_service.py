"""
services/ingestion_service.py
=============================
Admin legal document ingestion pipeline.
"""
from __future__ import annotations

import io
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

from core.config import get_settings
from core.exceptions import ExternalServiceError, ProviderNotConfiguredError, UnsupportedFileTypeError
from core.jurisdiction import canonical_jurisdiction, needs_multilingual_embedding, short_jurisdiction
from core.logging import get_logger
from services.llm_service import EmbeddingService

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestionInput:
    filename: str
    content_type: str
    content: bytes
    document_type: str
    jurisdiction: str
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


@dataclass(frozen=True)
class IngestionResult:
    job_id: str
    document_id: str | None
    source_table: str
    title: str
    status: str
    chunks: int
    text_length: int
    embedding_model: str | None
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
        text = extract_text(item.content, item.content_type, item.filename)
        if len(text.strip()) < 20:
            raise UnsupportedFileTypeError(
                "Could not extract enough text from this document.",
                details={"file_name": item.filename, "content_type": item.content_type},
            )

        chunks = chunk_text(text)
        embedding = await self._embed_text(text, jurisdiction=item.jurisdiction)
        source_table = _source_table(item.document_type)
        job_id = str(uuid.uuid4())

        if not self._supabase:
            warnings = [
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
                text_length=len(text),
                embedding_model=embedding["model"],
                warnings=warnings,
            )

        document_id = await self._insert_document(
            source_table=source_table,
            title=title,
            text=text,
            item=item,
            embedding=embedding["vector"],
            chunks=chunks,
        )

        log.info(
            "ingestion.completed",
            job_id=job_id,
            document_id=document_id,
            source_table=source_table,
            chunks=len(chunks),
            text_length=len(text),
        )

        return IngestionResult(
            job_id=job_id,
            document_id=document_id,
            source_table=source_table,
            title=title,
            status="indexed",
            chunks=len(chunks),
            text_length=len(text),
            embedding_model=embedding["model"],
            warnings=embedding["warnings"],
        )

    async def _embed_text(self, text: str, *, jurisdiction: str) -> dict[str, Any]:
        try:
            embedding_text = text[:8000]
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
            log.warning("ingestion.embedding.failed", error=str(exc))
            raise ExternalServiceError(f"Embedding failed: {exc}") from exc

    async def _insert_document(
        self,
        *,
        source_table: str,
        title: str,
        text: str,
        item: IngestionInput,
        embedding: list[float] | None,
        chunks: list[str],
    ) -> str:
        extracted = infer_lao_legal_metadata(text, title=title, source_url=item.source_url)
        metadata = {
            "file_name": item.filename,
            "content_type": item.content_type,
            "source_url": item.source_url,
            "official_source_url": item.source_url if _is_official_lao_source(item.source_url) else None,
            "source_authority": _source_authority(item.source_url),
            "language": item.language or extracted.get("language"),
            "law_no": item.law_no or extracted.get("law_no"),
            "article": item.article or extracted.get("article"),
            "gazette_date": item.gazette_date or extracted.get("gazette_date"),
            "effective_date": item.effective_date or extracted.get("effective_date"),
            "review_status": item.review_status,
            "chunks": len(chunks),
            "text_length": len(text),
            "ingestion_version": 2,
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


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    lower_name = filename.lower()

    if content_type in {"text/plain", "text/markdown"} or lower_name.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="replace")

    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return _extract_pdf(content)

    if (
        content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or lower_name.endswith(".docx")
    ):
        return _extract_docx(content)

    if content_type == "application/msword" or lower_name.endswith(".doc"):
        return content.decode("utf-8", errors="replace")

    raise UnsupportedFileTypeError(f"Unsupported legal document type '{content_type}'.")


def chunk_text(text: str, *, max_chars: int = 3200, overlap: int = 300) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        boundary = clean.rfind(" ", start, end)
        if boundary > start + max_chars * 0.6:
            end = boundary
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_pdf(content: bytes) -> str:
    try:
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        return "\n\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        raise UnsupportedFileTypeError(f"PDF text extraction failed: {exc}") from exc


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
    if _is_official_lao_source(source_url):
        return "lao_official_gazette"
    return "uploaded"


def _is_official_lao_source(source_url: str | None) -> bool:
    return bool(source_url and "laoofficialgazette.gov.la" in source_url.lower())


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


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

    if source_table == "cases":
        return [
            {
                "case_no": title,
                "court": "Unknown",
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "year_be": item.year,
                "language": metadata.get("language"),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
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
                "case_no": title,
                "court": "Unknown",
                "year": item.year or 0,
                "language": metadata.get("language"),
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
                "title": title,
                "form_type": "general",
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "review_status": item.review_status,
                "content": text,
                "tags": tags,
                "embedding": vector,
            },
            {
                "title": title,
                "form_type": "general",
                "jurisdiction": _short_jurisdiction(item.jurisdiction),
                "language": _short_jurisdiction(item.jurisdiction),
                "official_source_url": metadata.get("official_source_url"),
                "source_authority": metadata.get("source_authority"),
                "review_status": item.review_status,
                "content": text,
                "embedding": vector,
            },
        ]

    return [
        {
            "title": title,
            "doc_type": "regulation" if item.document_type == "regulation" else "statute",
            "jurisdiction": _db_jurisdiction(item.jurisdiction),
            "year_be": item.year,
            "language": metadata.get("language"),
            "official_source_url": metadata.get("official_source_url"),
            "source_authority": metadata.get("source_authority"),
            "law_no": metadata.get("law_no"),
            "article": metadata.get("article"),
            "gazette_date": metadata.get("gazette_date"),
            "effective_date": metadata.get("effective_date"),
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
            "title": title,
            "jurisdiction": _short_jurisdiction(item.jurisdiction),
            "year": item.year,
            "language": metadata.get("language"),
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
    }
    return {key: value for key, value in payload.items() if key not in optional_columns}
