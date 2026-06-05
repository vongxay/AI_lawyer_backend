"""
Tests for IracReasoningAgent — covers JSON parsing, fallback, and LLM integration.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.reasoning_agent import IracReasoningAgent
from services.llm_service import LlmResult


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
        model="claude-haiku-4-5-20251001",
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

    async def test_fallback_does_not_surface_raw_structured_json(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text='```json\n{"irac":{"issue":{"primary":"test","secondary":[]},"application":{"analysis":"partial"',
            model="stub",
            provider="stub",
        ))
        result = await agent.run(
            question="ປະເພດດິນອຸດສາຫະກຳສາມາດປຸກສ້າງໄດ້ບໍ?",
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )
        analysis = result.data["irac"]["application"]["analysis"]
        assert result.ok
        assert "```json" not in analysis
        assert '"irac"' not in analysis

    async def test_repairs_truncated_json_when_possible(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text='{"irac":{"issue":{"primary":"test issue","secondary":[]},"rule":{"statutes":[],"precedents":[]},"application":{"analysis":"test analysis","strengths":[],"weaknesses":[],"counter_args":[],"rebuttals":[]},"conclusion":{"recommendation":"test recommendation"',
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
        assert result.data["irac"]["conclusion"]["recommendation"] == "test recommendation"

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
            question="test",
            research=research, document=None, evidence=None,
            memory={"empty": True}
        )
        assert "ประมวลกฎหมายแพ่ง" in context["docs"]

    async def test_prioritises_explicit_lao_article_number(self, agent):
        chunks = [
            {
                "type": "law",
                "title": "land law",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 14",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 14. other",
                "final_score": 9.0,
            },
            {
                "type": "law",
                "title": "land law",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 25.",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 25. water land rules",
                "final_score": 1.0,
            },
        ]

        ordered = agent._prioritise_target_sections(
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 25 \u0e95\u0ec9\u0ead\u0e87\u0ea5\u0eb0\u0ea7\u0eb1\u0e87\u0eab\u0e8d\u0eb1\u0e87?",
            chunks,
        )

        assert ordered[0]["section"] == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 25."
        assert ordered[0]["_target_section_match"] is True

    async def test_agent_result_has_name(self, agent):
        result = await agent.run(
            question="test", research=None, document=None, evidence=None, memory={}
        )
        assert result.agent_name == "reasoning"
