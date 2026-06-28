"""
Scrape Lao laws from the Official Gazette and build/update lao_law_manifest.json.

Source: https://laoofficialgazette.gov.la (legaltype=16 = ກົດໝາຍ in force)
The National Assembly lists ~176 laws currently in force; the Gazette index shows ~180.

Run from AI_lawyer_backend:

    py -m scripts.fetch_lao_gazette_manifest --scrape-only
    py -m scripts.fetch_lao_gazette_manifest --bootstrap
    py -m scripts.fetch_lao_gazette_manifest --bootstrap --limit 5
    py -m scripts.fetch_lao_gazette_manifest --bootstrap --force
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from core.logging import get_logger
from scripts.bootstrap_lao_knowledge import DEFAULT_MANIFEST, bootstrap_manifest

log = get_logger(__name__)

GAZETTE_BASE = "https://laoofficialgazette.gov.la"
LIST_URL = (
    f"{GAZETTE_BASE}/index.php?r=site/list&legaltype=16&old=0&Document_page={{page}}"
)
USER_AGENT = "Mozilla/5.0 (compatible; AILawyerBootstrap/1.0; +https://github.com/)"

ROW_RE = re.compile(
    r'<tr class="(?:odd|even)">\s*'
    r"<td>(?P<title>[^<]+)</td>\s*"
    r"<td>(?P<agency>[^<]+)</td>\s*"
    r'<td[^>]*>(?P<effective>[^<]*)</td>\s*'
    r'<td[^>]*>(?P<issue>[^<]*)</td>\s*'
    r"<td>(?P<legal_type>[^<]+)</td>\s*"
    r"<td>(?P<status>[^<]+)</td>\s*"
    r'<td><a href="(?P<display>[^"]+)">[^<]+</a></td>\s*'
    r"<td[^>]*>.*?</td>\s*"
    r'<td[^>]*>\s*(?:<a[^>]+href="(?P<pdf>[^"]+\.pdf)"[^>]*>.*?</a>)?\s*</td>',
    re.DOTALL | re.IGNORECASE,
)

DISPLAY_ID_RE = re.compile(r"id=(\d+)")


def infer_law_category(title: str, agency: str) -> str:
    text = f"{title} {agency}".casefold()
    if any(k in text for k in ("ຕ່າງປະເທດ", "diplomat", "foreign", "ຊາຍແດນ", "ນັກທູດ")):
        return "foreign_affairs"
    if any(k in text for k in ("ປ້ອງກັນ", "ກອງທັບ", "ຄວາມສະຫງົບ", "security", "defence", "defense")):
        return "state_security"
    if any(k in text for k in (
        "ສຶກສາ", "education", "ສາທາ", "health", "ແຮງງານ", "labor", "labour",
        "ຄອບຄົວ", "family", "ວັດທະ", "culture", "ກິລາ", "sport",
    )):
        return "social_culture"
    if any(k in text for k in (
        "ຍຸຕິທຳ", "justice", "ສພາ", "assembly", "constitution", "ລັດຖະທ",
        "election", "ເລືອກຕັ້ງ", "ສານ", "court", "prosecut",
    )):
        return "constitution_justice"
    return "economy"


def parse_year(effective: str, issue: str) -> int | None:
    for value in (effective, issue):
        match = re.search(r"(\d{4})", value or "")
        if match:
            year = int(match.group(1))
            if 1975 <= year <= 2035:
                return year
    return None


def slugify_id(display_href: str, title: str) -> str:
    match = DISPLAY_ID_RE.search(display_href)
    if match:
        return f"lao_gazette_{match.group(1)}"
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"lao_gazette_{digest}"


def parse_list_page(html: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for match in ROW_RE.finditer(html):
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        agency = re.sub(r"\s+", " ", match.group("agency")).strip()
        pdf_path = (match.group("pdf") or "").strip()
        if not pdf_path:
            continue
        display = match.group("display").strip()
        effective = match.group("effective").strip()
        issue = match.group("issue").strip()
        pdf_url = urljoin(GAZETTE_BASE, pdf_path)
        display_url = urljoin(GAZETTE_BASE, display)
        entry_id = slugify_id(display, title)
        entries.append({
            "id": entry_id,
            "title": title,
            "url": pdf_url,
            "document_type": "law",
            "jurisdiction": "laos",
            "law_category": infer_law_category(title, agency),
            "year": parse_year(effective, issue),
            "language": "lo",
            "tags": ["official_gazette", "lao_law", agency[:40] if agency else "unknown"],
            "review_status": "approved",
            "source_url": display_url,
            "agency": agency,
            "effective_date": effective,
            "issue_date": issue,
            "notes": f"Scraped from Lao Official Gazette display {display_url}",
        })
    return entries


def detect_last_page(html: str) -> int:
    match = re.search(r'Document_page=(\d+)">[^<]*</a></li></ul></div><div class="keys"', html)
    if match:
        return int(match.group(1))
    pages = [int(n) for n in re.findall(r"Document_page=(\d+)", html)]
    return max(pages) if pages else 1


async def scrape_gazette_laws(*, max_pages: int | None = None) -> list[dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(60.0, connect=15.0)
    all_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        first = await client.get(LIST_URL.format(page=1))
        first.raise_for_status()
        last_page = detect_last_page(first.text)
        if max_pages is not None:
            last_page = min(last_page, max_pages)

        for page in range(1, last_page + 1):
            url = LIST_URL.format(page=page)
            if page == 1:
                html = first.text
            else:
                log.info("gazette.scrape.page", page=page, total_pages=last_page)
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
                await asyncio.sleep(0.4)

            page_entries = parse_list_page(html)
            for entry in page_entries:
                entry_id = entry["id"]
                if entry_id in seen_ids:
                    continue
                seen_ids.add(entry_id)
                all_entries.append(entry)

    log.info("gazette.scrape.done", count=len(all_entries), pages=last_page)
    return all_entries


def merge_manifest(
    scraped: list[dict[str, Any]],
    existing_path: Path,
    *,
    keep_manual: bool = True,
) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if existing_path.exists():
        raw = json.loads(existing_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            existing = raw

    scraped_urls = {entry["url"] for entry in scraped}
    merged = list(scraped)

    if keep_manual:
        for entry in existing:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "")
            if url and url not in scraped_urls:
                merged.append(entry)

    return merged


def write_manifest(entries: list[dict[str, Any]], path: Path) -> None:
    # Strip scrape-only metadata not needed by bootstrap
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        cleaned.append({
            "id": entry["id"],
            "title": entry["title"],
            "url": entry["url"],
            "document_type": entry.get("document_type", "law"),
            "jurisdiction": entry.get("jurisdiction", "laos"),
            "law_category": entry.get("law_category"),
            "year": entry.get("year"),
            "language": entry.get("language", "lo"),
            "tags": entry.get("tags", []),
            "review_status": entry.get("review_status", "approved"),
            "notes": entry.get("notes"),
        })
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Lao laws from Official Gazette.")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape and write manifest.")
    parser.add_argument("--bootstrap", action="store_true", help="Ingest manifest entries into Supabase.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest output path.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit gazette pages (debug).")
    parser.add_argument("--limit", type=int, default=None, help="Ingest only first N manifest entries.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source_url exists.")
    parser.add_argument("--dry-run", action="store_true", help="Bootstrap dry-run (no download/ingest).")
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    scraped = await scrape_gazette_laws(max_pages=args.max_pages)
    merged = merge_manifest(scraped, manifest_path, keep_manual=True)
    write_manifest(merged, manifest_path)

    report: dict[str, Any] = {
        "scraped": len(scraped),
        "manifest_total": len(merged),
        "manifest_path": str(manifest_path),
    }

    if args.bootstrap or (not args.scrape_only):
        only_ids = None
        if args.limit:
            only_ids = {entry["id"] for entry in merged[: args.limit]}
        bootstrap_report = await bootstrap_manifest(
            manifest_path,
            dry_run=args.dry_run,
            only_ids=only_ids,
            skip_existing=not args.force,
        )
        report["bootstrap"] = bootstrap_report

    return report


def main() -> None:
    args = _parse_args()
    if not args.scrape_only and not args.bootstrap:
        args.scrape_only = True
    report = asyncio.run(_async_main(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
