"""
Tests for IracReasoningAgent — covers JSON parsing, fallback, and LLM integration.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.reasoning_agent import IracReasoningAgent
from backend.services.llm_service import LlmResult


VALID_IRAC_JSON = json.dumps({
    "irac": {
        "issue": {"primary": "test issue", "secondary": []},
        "rule": {"statutes": [], "precedents": []},
        "application": {
            "analysis": "test analysis",
            "strengths": ["strong point"],
            "weaknesses": [],
            "counter_args": [],
            "rebuttals": [],
        },
        "conclusion": {
            "recommendation": "test recommendation",
            "action_steps": ["step 1"],
            "risk_level": "MEDIUM",
            "win_probability": 0.72,
            "settlement_note": None,
        },
    },
    "confidence": 0.85,
    "citations": [{"ref": "มาตรา 420", "status": "UNVERIFIED"}],
})


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LlmResult(
        text=VALID_IRAC_JSON,
        model="claude-sonnet-4-20250514",
        input_tokens=500,
        output_tokens=300,
        provider="anthropic",
    ))
    return llm


@pytest.fixture
def agent(mock_llm):
    return IracReasoningAgent(llm=mock_llm)


class TestIracReasoningAgent:
    async def test_returns_valid_irac_structure(self, agent):
        result = await agent.run(
            question="ฉันถูกนายจ้างไล่ออกไม่เป็นธรรม ต้องทำอย่างไร",
            research={"retrieved_documents": [], "case_graph_context": [], "memory_highlights": {}},
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        assert result.ok
        assert "irac" in result.data
        assert result.data["irac"]["issue"]["primary"] == "test issue"
        assert result.confidence == 0.85

    async def test_parses_confidence_correctly(self, agent):
        result = await agent.run(
            question="test",
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        assert 0.0 <= result.confidence <= 1.0

    async def test_fallback_on_invalid_json(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text="This is not JSON at all — plain text response",
            model="stub",
            provider="stub",
        ))
        result = await agent.run(
            question="test question",
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        assert result.ok  # Should not raise — fallback kicks in
        assert "irac" in result.data
        assert result.confidence == 0.5  # fallback confidence

    async def test_insufficient_context_response(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps({"insufficient_context": True, "reason": "No laws found"}),
            model="stub",
            provider="stub",
        ))
        result = await agent.run(
            question="obscure question",
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        assert result.ok
        assert result.confidence == 0.3

    async def test_strips_markdown_fences(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=f"```json\n{VALID_IRAC_JSON}\n```",
            model="stub",
            provider="stub",
        ))
        result = await agent.run(
            question="test",
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        assert result.ok
        assert result.data.get("confidence") == 0.85

    async def test_builds_context_with_research(self, agent):
        """Research docs are injected into LLM prompt context."""
        research = {
            "retrieved_documents": [
                {"type": "law", "title": "ประมวลกฎหมายแพ่ง", "section": "มาตรา 420", "content": "test content"}
            ],
            "case_graph_context": [],
            "memory_highlights": {},
        }
        context = agent._build_context(
            research=research, document=None, evidence=None,
            memory={"empty": True}
        )
        assert "ประมวลกฎหมายแพ่ง" in context["docs"]

    async def test_agent_result_has_name(self, agent):
        result = await agent.run(
            question="test", research=None, document=None, evidence=None, memory={}
        )
        assert result.agent_name == "reasoning"
