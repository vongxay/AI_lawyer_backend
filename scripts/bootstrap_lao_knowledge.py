"""
Bootstrap Lao legal knowledge base from manifest URLs.

Downloads official/public legal PDFs and ingests them with review_status=approved
so the production retriever (review_status=approved) can use them immediately.

Run from AI_lawyer_backend:

    py -m scripts.knowledge_base_audit
    py -m scripts.bootstrap_lao_knowledge
    py -m scripts.bootstrap_lao_knowledge --dry-run
    py -m scripts.bootstrap_lao_knowledge --only lao_land_law_70_na_2019_lo
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from core.config import get_settings
from core.database import get_supabase
from core.logging import get_logger
from services.ingestion_service import IngestionInput, LegalDocumentIngestionService

log = get_logger(__name__)

DEFAULT_MANIFEST = Path(__file__).with_name("lao_law_manifest.json")


async def _source_already_indexed(supabase: Any, source_url: str) -> bool:
    try:
        result = (
            await supabase.table("document_chunks")
            .select("id")
            .contains("metadata", {"source_url": source_url})
            .eq("review_status", "approved")
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception:
        try:
            result = (
                await supabase.table("document_chunks")
                .select("id, metadata")
                .eq("review_status", "approved")
                .limit(2000)
                .execute()
            )
            for row in result.data or []:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                if metadata.get("source_url") == source_url:
                    return True
        except Exception as exc:
            log.warning("bootstrap.duplicate_check_failed", error=str(exc))
    return False


async def _download(url: str, *, timeout_seconds: float = 120.0) -> tuple[bytes, str]:
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type or content_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(urlparse(url).path)
            content_type = guessed or "application/pdf"
        return response.content, content_type


async def bootstrap_manifest(
    manifest_path: Path,
    *,
    dry_run: bool = False,
    only_ids: set[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    if supabase is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env.")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Manifest must be a JSON list.")

    service = LegalDocumentIngestionService(supabase=supabase)
    results: list[dict[str, Any]] = []

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if entry.get("enabled") is False:
            results.append({"id": entry.get("id"), "status": "skipped_disabled"})
            continue

        entry_id = str(entry.get("id") or "")
        if only_ids and entry_id not in only_ids:
            continue

        url = str(entry.get("url") or "").strip()
        if not url:
            results.append({"id": entry_id, "status": "skipped_missing_url"})
            continue

        if skip_existing and await _source_already_indexed(supabase, url):
            results.append({"id": entry_id, "status": "skipped_existing", "url": url})
            continue

        title = str(entry.get("title") or entry_id)
        if dry_run:
            results.append({"id": entry_id, "status": "dry_run", "url": url, "title": title})
            continue

        log.info("bootstrap.download.start", entry_id=entry_id, url=url)
        content, content_type = await _download(url)
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            results.append({
                "id": entry_id,
                "status": "failed",
                "error": f"Remote document ({size_mb:.1f}MB) exceeds MAX_UPLOAD_SIZE_MB={settings.max_upload_size_mb}",
            })
            continue

        filename = Path(urlparse(url).path).name or f"{entry_id}.pdf"
        ingest_result = await service.ingest(
            IngestionInput(
                filename=filename,
                content_type=content_type,
                content=content,
                document_type=str(entry.get("document_type") or "law"),
                jurisdiction=str(entry.get("jurisdiction") or "laos"),
                law_category=entry.get("law_category"),
                title=title,
                year=entry.get("year"),
                tags=[str(tag) for tag in entry.get("tags", []) if tag],
                source_url=url,
                law_no=entry.get("law_no"),
                language=entry.get("language"),
                review_status=str(entry.get("review_status") or "approved"),
                tenant_id=settings.default_tenant_id,
            )
        )
        results.append({
            "id": entry_id,
            "status": ingest_result.status,
            "document_id": ingest_result.document_id,
            "chunks": ingest_result.chunks,
            "chunks_indexed": ingest_result.chunks_indexed,
            "chunks_embedded": ingest_result.chunks_embedded,
            "review_status": ingest_result.review_status,
            "warnings": ingest_result.warnings[:5],
            "url": url,
        })
        log.info(
            "bootstrap.ingest.completed",
            entry_id=entry_id,
            chunks_indexed=ingest_result.chunks_indexed,
            status=ingest_result.status,
        )

    succeeded = sum(1 for item in results if item.get("status") in {"indexed", "processed_without_database"})
    skipped = sum(1 for item in results if str(item.get("status", "")).startswith("skipped"))
    failed = sum(1 for item in results if item.get("status") == "failed")

    return {
        "manifest": str(manifest_path),
        "dry_run": dry_run,
        "processed": len(results),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Lao legal knowledge base.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest JSON.")
    parser.add_argument("--dry-run", action="store_true", help="List planned ingestions without downloading.")
    parser.add_argument("--only", action="append", default=[], help="Only ingest manifest entry id(s).")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source_url already exists.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--unicode", action="store_true", help="Print unicode instead of JSON escapes.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    only_ids = set(args.only) if args.only else None
    report = asyncio.run(
        bootstrap_manifest(
            Path(args.manifest),
            dry_run=args.dry_run,
            only_ids=only_ids,
            skip_existing=not args.force,
        )
    )
    print(json.dumps(report, ensure_ascii=not args.unicode, indent=2))


if __name__ == "__main__":
    main()
