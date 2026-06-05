"""
services/ingestion_service.py
=============================
Admin legal document ingestion pipeline.
"""
from __future__ import annotations

import io
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
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
    jurisdiction: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalTextChunk:
    index: int
    content: str
    section_ref: str | None
    token_count: int


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
        warnings: list[str] = []
        if len(text.strip()) < 20:
            if not item.allow_short_text:
                raise UnsupportedFileTypeError(
                    "Could not extract enough text from this document.",
                    details={"file_name": item.filename, "content_type": item.content_type},
                )
            warnings.append(
                "Document text is very short; add the full legal text before relying on search or AI answers."
            )

        chunks = chunk_legal_text(
            text,
            max_chars=self._settings.rag_chunk_max_chars,
            overlap=self._settings.rag_chunk_overlap_chars,
        )
        embedding = await self._embed_text(text, jurisdiction=item.jurisdiction)
        source_table = _source_table(item.document_type)
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
                jurisdiction=_short_jurisdiction(item.jurisdiction),
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
        chunk_embedding = await self._embed_chunks(chunks, jurisdiction=item.jurisdiction)
        chunks_indexed, chunk_index_warnings = await self._insert_chunks(
            source_table=source_table,
            document_id=document_id,
            title=title,
            item=item,
            chunks=chunks,
            embeddings=chunk_embedding["vectors"],
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
            jurisdiction=_short_jurisdiction(item.jurisdiction),
            warnings=[*warnings, *embedding["warnings"], *chunk_embedding["warnings"], *chunk_index_warnings],
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

    async def _insert_chunks(
        self,
        *,
        source_table: str,
        document_id: str,
        title: str,
        item: IngestionInput,
        chunks: list[LegalTextChunk],
        embeddings: list[list[float] | None],
    ) -> tuple[int, list[str]]:
        if not chunks:
            return 0, []

        payloads: list[dict[str, Any]] = []
        status = _active_status_for_review(item.review_status)
        for chunk in chunks:
            embedding = embeddings[chunk.index] if chunk.index < len(embeddings) else None
            payloads.append({
                "tenant_id": item.tenant_id,
                "source_table": source_table,
                "source_id": document_id,
                "document_type": item.document_type,
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "title": title,
                "chunk_index": chunk.index,
                "section_ref": chunk.section_ref,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "status": status,
                "review_status": item.review_status,
                "metadata": {
                    "file_name": item.filename,
                    "source_url": item.source_url,
                    "tags": item.tags,
                    "law_no": item.law_no,
                    "article": item.article,
                    "ingestion_version": 3,
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
            return inserted, [
                "document_chunks table is not available; apply supabase_agentic_rag_chunks.sql to enable chunk-level RAG."
            ]

        return inserted, []


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


SECTION_HEADING_RE = re.compile(
    r"(?im)^(?P<section>"
    r"(?:มาตรา|ข้อ|หมวด|บทที่|ມາດຕາ|Article|Art\.|Section|Sec\.|Chapter|Part)"
    r"\s+[0-9A-Za-zก-๙ກ-ຮ./-]+"
    r")"
)


def chunk_text(text: str, *, max_chars: int = 3200, overlap: int = 300) -> list[str]:
    return [chunk.content for chunk in chunk_legal_text(text, max_chars=max_chars, overlap=overlap)]


def chunk_legal_text(text: str, *, max_chars: int = 2600, overlap: int = 350) -> list[LegalTextChunk]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: list[LegalTextChunk] = []
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


def _split_legal_sections(text: str) -> list[tuple[str | None, str]]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [(None, re.sub(r"\s+", " ", text).strip())]

    sections: list[tuple[str | None, str]] = []
    first_start = matches[0].start()
    if first_start > 100:
        preamble = re.sub(r"\s+", " ", text[:first_start]).strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_ref = re.sub(r"\s+", " ", match.group("section")).strip()
        section_text = re.sub(r"\s+", " ", text[start:end]).strip()
        if section_text:
            sections.append((section_ref, section_text))

    return sections or [(None, re.sub(r"\s+", " ", text).strip())]


def _split_with_overlap(text: str, *, max_chars: int, overlap: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    safe_overlap = min(overlap, max_chars // 2)
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        boundary = clean.rfind(" ", start, end)
        if boundary > start + max_chars * 0.6:
            end = boundary
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(0, end - safe_overlap)
    return chunks


def _extract_pdf(content: bytes) -> str:
    errors: list[str] = []
    settings = get_settings()

    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            text = "\n\n".join(page.get_text("text") for page in doc)
        if text.strip():
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(text):
                errors.append("PyMuPDF extracted garbled text from a PDF text layer; trying another extractor/OCR.")
            else:
                return text
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
        if text.strip():
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(text):
                errors.append("pypdf extracted garbled text from a PDF text layer; trying OCR.")
            else:
                return text
        else:
            errors.append("pypdf extracted no text; the PDF may be scanned or image-only.")
    except ModuleNotFoundError:
        errors.append("pypdf is not installed. Install it with: py -m pip install pypdf")
    except Exception as exc:
        errors.append(f"pypdf failed: {exc}")

    if settings.pdf_ocr_enabled:
        text = _extract_pdf_with_ocr(content, errors=errors)
        if text.strip():
            if settings.pdf_detect_garbled_text and _looks_like_garbled_pdf_text(text):
                errors.append("OCR fallback extracted text that still looks garbled.")
            else:
                return text
    else:
        errors.append("Local OCR fallback is disabled. Set PDF_OCR_ENABLED=true to enable scanned PDF OCR.")

    details = {"extractors": errors}
    raise UnsupportedFileTypeError(
        "PDF text extraction failed. The PDF text layer appears scanned, missing, or garbled. "
        "Install Tesseract OCR with Lao/Thai language data, or upload a Unicode text-searchable PDF.",
        details=details,
    )


def _looks_like_garbled_pdf_text(text: str) -> bool:
    sample = re.sub(r"\s+", " ", text).strip()[:6000]
    chars = [ch for ch in sample if not ch.isspace()]
    if len(chars) < 80:
        return False

    replacement_ratio = sample.count("\ufffd") / len(chars)
    if replacement_ratio > 0.01:
        return True

    lao_thai = sum(1 for ch in chars if ("\u0e00" <= ch <= "\u0eff"))
    lao_thai_ratio = lao_thai / len(chars)
    if lao_thai_ratio > 0.08:
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

    hyphen_noise = len(re.findall(r"(?:[A-Za-zÀ-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„]{1,6}-){2,}", sample))
    extended_tokens = len(re.findall(r"[A-Za-zÀ-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„]*[À-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„][A-Za-zÀ-ÿ®©ªº¯½¾¿¡§¦£¥µ±²³´ð−†™‰„]*", sample))

    if suspicious_ratio > 0.18 and hyphen_noise >= 2:
        return True
    if suspicious_ratio > 0.28 and extended_tokens >= 20:
        return True
    return False


def _extract_pdf_with_ocr(content: bytes, *, errors: list[str]) -> str:
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

    text_parts: list[str] = []

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = min(len(doc), max(1, settings.pdf_ocr_max_pages))
            scale = max(72, settings.pdf_ocr_dpi) / 72
            matrix = fitz.Matrix(scale, scale)

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                text = _ocr_image_text(
                    image,
                    pytesseract,
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
        errors.append("OCR fallback extracted no text; the scan may be low quality or Tesseract language data may be missing.")
    return text


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


def _ocr_image_text(image: Any, pytesseract: Any, *, page_number: int, errors: list[str]) -> str:
    language = (get_settings().pdf_ocr_languages or "eng").strip() or "eng"
    try:
        return pytesseract.image_to_string(image, lang=language)
    except Exception as exc:
        if language != "eng":
            errors.append(
                f"Tesseract OCR failed on page {page_number} with languages '{language}'; retrying with 'eng': {exc}"
            )
            try:
                return pytesseract.image_to_string(image, lang="eng")
            except Exception as fallback_exc:
                errors.append(f"Tesseract OCR failed on page {page_number} with 'eng': {fallback_exc}")
                return ""
        errors.append(f"Tesseract OCR failed on page {page_number}: {exc}")
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
                "case_no": title,
                "court": "Unknown",
                "jurisdiction": _db_jurisdiction(item.jurisdiction),
                "year_be": item.year,
                "language": metadata.get("language"),
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
                "is_active": item.review_status == "approved",
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
                "is_active": item.review_status == "approved",
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
        "is_active",
    }
    legacy = {key: value for key, value in payload.items() if key not in optional_columns}
    if payload.get("review_status") and legacy.get("status") == "pending":
        legacy["status"] = "active"
    return legacy
