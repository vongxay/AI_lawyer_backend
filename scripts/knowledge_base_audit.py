"""
Knowledge base audit — report document/chunk coverage in Supabase.

Run from AI_lawyer_backend:

    py -m scripts.knowledge_base_audit
    py -m scripts.knowledge_base_audit --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from typing import Any

from core.config import get_settings
from core.database import get_supabase


async def audit_knowledge_base(*, sample_limit: int = 5000) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    if supabase is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env.")

    report: dict[str, Any] = {
        "tenant_id": settings.default_tenant_id,
        "tables": {},
        "document_chunks": {},
        "gaps": [],
    }

    for table in ("document_chunks", "laws", "cases", "legal_forms", "ingestion_jobs"):
        try:
            result = await supabase.table(table).select("id", count="exact").limit(1).execute()
            report["tables"][table] = {"count": result.count or 0}
        except Exception as exc:
            report["tables"][table] = {"count": None, "error": str(exc)}

    chunk_count = report["tables"].get("document_chunks", {}).get("count") or 0
    if chunk_count == 0:
        report["gaps"].append("document_chunks_empty")
        report["gaps"].append("run_bootstrap: py -m scripts.bootstrap_lao_knowledge")
        return report

    try:
        rows = (
            await supabase.table("document_chunks")
            .select("review_status, status, jurisdiction, law_category, document_type, title")
            .limit(sample_limit)
            .execute()
        ).data or []
    except Exception as exc:
        report["document_chunks"]["error"] = str(exc)
        return report

    report["document_chunks"] = {
        "sampled": len(rows),
        "review_status": dict(Counter(row.get("review_status") for row in rows)),
        "status": dict(Counter(row.get("status") for row in rows)),
        "jurisdiction": dict(Counter(row.get("jurisdiction") for row in rows)),
        "law_category": dict(Counter(row.get("law_category") for row in rows)),
        "document_type": dict(Counter(row.get("document_type") for row in rows)),
        "unique_titles": len({row.get("title") for row in rows if row.get("title")}),
    }

    approved = report["document_chunks"]["review_status"].get("approved", 0)
    if approved == 0:
        report["gaps"].append("no_approved_chunks — RAG retriever filters review_status=approved")

    laos_approved = sum(
        1 for row in rows
        if row.get("jurisdiction") in {"laos", "LA", "la"} and row.get("review_status") == "approved"
    )
    if laos_approved == 0:
        report["gaps"].append("no_approved_lao_chunks")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AI Lawyer knowledge base coverage.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--unicode", action="store_true", help="Print unicode instead of JSON escapes.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = asyncio.run(audit_knowledge_base())
    if args.json:
        print(json.dumps(report, ensure_ascii=not args.unicode, indent=2))
        return

    print(json.dumps(
        {
            "tables": report["tables"],
            "document_chunks": report["document_chunks"],
            "gaps": report["gaps"],
        },
        ensure_ascii=not args.unicode,
        indent=2,
    ))


if __name__ == "__main__":
    main()
