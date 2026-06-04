"""
Tests for CitationVerificationAgent.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.verification_agent import CitationVerificationAgent
from core.config import get_settings
from services.llm_service import LlmResult


@pytest.fixture
def agent_no_db():
    """Agent without Supabase — exercises LLM fallback path."""
    return CitationVerificationAgent(supabase=None)


@pytest.fixture
def mock_llm_plausible():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LlmResult(
        text=json.dumps([
            {"ref": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420", "plausible": True, "reason": "Known Thai tort law"},
        ]),
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
    ))
    return llm


class TestCitationVerificationAgent:
    async def test_empty_citations_returns_verified(self, agent_no_db):
        result = await agent_no_db.run(citations=[])
        assert result.data["citations_verified"] is True
        assert result.data["citations"] == []
        assert result.data["rejection_rate"] == 0.0

    async def test_marks_unverified_when_no_db(self, agent_no_db, mock_llm_plausible):
        agent_no_db._llm = mock_llm_plausible
        citations = [{"ref": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420", "status": "UNVERIFIED"}]
        result = await agent_no_db.run(citations=citations)
        assert result.ok
        assert len(result.data["citations"]) == 1
        assert result.data["citations"][0]["status"] == "UNVERIFIED"
        mock_llm_plausible.generate.assert_not_called()

    async def test_rejection_rate_computed(self, agent_no_db):
        """When LLM service is stub, citations should stay UNVERIFIED (not REJECTED)."""
        agent_no_db._llm = MagicMock()
        agent_no_db._llm.generate = AsyncMock(side_effect=Exception("LLM down"))

        citations = [{"ref": "มาตรา 999", "status": "UNVERIFIED"}]
        result = await agent_no_db.run(citations=citations)
        # Should degrade gracefully — not raise
        assert result.ok
        verified = result.data["citations"]
        assert all(c["status"] == "UNVERIFIED" for c in verified)

    async def test_confidence_decreases_with_rejections(self, agent_no_db, mock_llm_plausible):
        settings = get_settings()
        previous = settings.llm_verify_citations_with_llm
        settings.llm_verify_citations_with_llm = True
        mock_llm_plausible.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps([
                {"ref": "fake citation 123", "plausible": False, "reason": "Does not exist"},
            ]),
            model="stub", provider="stub",
        ))
        try:
            agent_no_db._llm = mock_llm_plausible
            citations = [{"ref": "fake citation 123", "status": "UNVERIFIED"}]
            result = await agent_no_db.run(citations=citations)
            assert result.ok
            assert result.data["citations"][0]["status"] == "REJECTED"
            assert result.data["rejection_rate"] == 1.0
            assert result.confidence == 0.0
        finally:
            settings.llm_verify_citations_with_llm = previous
