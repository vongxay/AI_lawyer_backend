"""
rag/graph_expander.py
=====================
Case law graph traversal using the `get_precedent_chain` SQL function.
Expands top case results to include their precedent chains.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)


class GraphExpander:
    def __init__(self, supabase: "AsyncClient | None" = None) -> None:
        self._supabase = supabase

    async def expand(self, *, case_ids: list[str], depth: int = 2) -> list[dict[str, Any]]:
        if not case_ids or not self._supabase:
            return []

        all_results: list[dict] = []
        for case_id in case_ids[:5]:   # Limit to top-5 to control token cost
            try:
                result = await self._supabase.rpc(
                    "get_precedent_chain",
                    {"start_case_id": case_id, "max_depth": depth},
                ).execute()
                if result.data:
                    for row in result.data:
                        all_results.append({
                            **row,
                            "type": "precedent",
                            "source_case": case_id,
                            "content": f"Precedent at depth {row.get('depth', 1)}: {row.get('relationship', 'cites')}",
                        })
            except Exception as exc:
                log.debug("graph.expand.failed", case_id=case_id, error=str(exc))

        log.info("graph.expanded", input_cases=len(case_ids), expanded_nodes=len(all_results))
        return all_results
