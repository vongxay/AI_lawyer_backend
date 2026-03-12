from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class CitationVerificationAgent(BaseAgent):
    name = "verification"

    async def verify(self, citations: list[dict]) -> dict:
        verified: list[dict] = []
        for c in citations:
            ref = c.get("ref")
            status = c.get("status", "UNVERIFIED")
            if isinstance(ref, str) and ref.strip():
                # Stub: mark as VERIFIED if contains "มาตรา"
                status = "VERIFIED" if "มาตรา" in ref else status
            verified.append({**c, "status": status})
        citations_verified = all(v.get("status") == "VERIFIED" for v in verified) if verified else True
        return {"citations": verified, "citations_verified": citations_verified}

