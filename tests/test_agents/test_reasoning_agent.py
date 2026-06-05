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

LAND_RIGHT_PROTECTION_Q = (
    "\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0eaa\u0eb4\u0e94"
    "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
    "\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0e81\u0eb2\u0e99\u0e9b\u0ebb\u0e81"
    "\u0e9b\u0ec9\u0ead\u0e87\u0eaa\u0eb4\u0e94\u0ec3\u0e94\u0ec1\u0e94\u0ec8"
)

LAND_LEASE_DURATION_Q = (
    "\u0e9e\u0ebb\u0e99\u0ea5\u0eb0\u0ec0\u0ea1\u0eb7\u0ead\u0e87\u0ea5\u0eb2\u0ea7"
    "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99"
    "\u0e82\u0ead\u0e87\u0ea5\u0eb1\u0e94\u0ec4\u0e94\u0ec9"
    "\u0eaa\u0eb9\u0e87\u0eaa\u0eb8\u0e94\u0e88\u0eb1\u0e81\u0e9b\u0eb5"
)


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
        assert "plain text response" not in result.data["irac"]["application"]["analysis"]
        assert result.confidence == 0.3  # unsafe raw model output is withheld

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

    async def test_insufficient_context_reason_suppresses_prompt_leak(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps({
                "insufficient_context": True,
                "reason": "No answer can be given without using the CONTEXT block.",
            }),
            model="stub",
            provider="stub",
        ))
        result = await agent.run(
            question=LAND_RIGHT_PROTECTION_Q,
            research=None,
            document=None,
            evidence=None,
            memory={"empty": True},
        )

        analysis = result.data["irac"]["application"]["analysis"]
        assert "CONTEXT block" not in analysis
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

    async def test_does_not_hardcode_article_for_land_use_right_question(self, agent):
        chunks = [
            {
                "type": "law",
                "title": "Law on Land",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 12",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 12. other land content",
                "final_score": 9.0,
            },
            {
                "type": "law",
                "title": "Law on Land",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5.",
                "content": (
                    "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5. "
                    "\u0e81\u0eb2\u0e99\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87"
                    "\u0eaa\u0eb4\u0e94 \u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99 "
                    "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94"
                ),
                "final_score": 1.0,
            },
        ]

        ordered = agent._prioritise_target_sections(LAND_RIGHT_PROTECTION_Q, chunks)

        assert agent._section_numbers_from_question(LAND_RIGHT_PROTECTION_Q) == []
        assert ordered[0]["section"] == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 12"
        assert "_target_section_match" not in ordered[0]

    async def test_focused_statutory_answer_uses_retrieved_article_thirteen(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps({
                "recommendation": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 13 \u0e81\u0eb3\u0e99\u0ebb\u0e94\u0ea7\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2\u0ec4\u0e94\u0ec9\u0eaa\u0eb9\u0e87\u0eaa\u0eb8\u0e94\u0e9a\u0ecd\u0ec8\u0ec0\u0e81\u0eb5\u0e99 30 \u0e9b\u0eb5",
                "analysis": "\u0e84\u0eb3\u0e95\u0ead\u0e9a\u0e99\u0eb5\u0ec9\u0ead\u0eb5\u0e87\u0e88\u0eb2\u0e81\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1\u0ea1\u0eb2\u0e94\u0e95\u0eb2 13",
                "action_steps": ["\u0e81\u0eb3\u0e99\u0ebb\u0e94\u0ec4\u0ea5\u0e8d\u0eb0\u0ec0\u0ea7\u0ea5\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2\u0e9a\u0ecd\u0ec8\u0ec0\u0e81\u0eb5\u0e99 30 \u0e9b\u0eb5"],
                "confidence": 0.88,
            }),
            model="stub",
            provider="stub",
        ))
        research = {
            "retrieved_documents": [
                {
                    "type": "law",
                    "title": "Law on Land",
                    "section": "Article 13",
                    "content": (
                        "Article 13. The State allows Lao citizens to lease state land "
                        "for no more than thirty years."
                    ),
                    "final_score": 10.0,
                }
            ],
            "query_analysis": {
                "authority_hints": [{"law_name": "Law on Land", "article": "13"}],
            },
        }

        result = await agent.run(
            question=LAND_LEASE_DURATION_Q,
            research=research,
            document=None,
            evidence=None,
            memory={"empty": True},
            response_language="lo",
            query_mode="general",
            response_style="plain",
            query_type="legal_question",
        )

        recommendation = result.data["irac"]["conclusion"]["recommendation"]
        assert result.ok
        assert result.confidence == 0.88
        assert "30" in recommendation
        assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 13" in recommendation
        mock_llm.generate.assert_called_once()

    async def test_focused_statutory_answer_uses_retrieved_article_five(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps({
                "recommendation": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5 \u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9 \u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99 \u0ec1\u0ea5\u0eb0 \u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94",
                "analysis": "\u0e84\u0eb3\u0e95\u0ead\u0e9a\u0ead\u0eb5\u0e87\u0e88\u0eb2\u0e81\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5",
                "action_steps": [],
                "confidence": 0.87,
            }),
            model="stub",
            provider="stub",
        ))
        research = {
            "retrieved_documents": [
                {
                    "type": "law",
                    "title": "Law on Land",
                    "section": "Article 5",
                    "content": (
                        "Article 5. The State protects lawful benefits of land-use-right holders "
                        "and guarantees rights to protect, use, receive benefits, transfer, and inherit."
                    ),
                    "final_score": 10.0,
                }
            ],
            "query_analysis": {
                "authority_hints": [{"law_name": "Law on Land", "article": "5"}],
            },
        }

        result = await agent.run(
            question=LAND_RIGHT_PROTECTION_Q,
            research=research,
            document=None,
            evidence=None,
            memory={"empty": True},
            response_language="lo",
            query_mode="general",
            response_style="plain",
            query_type="legal_question",
        )

        recommendation = result.data["irac"]["conclusion"]["recommendation"]
        assert result.ok
        assert result.confidence == 0.87
        assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5" in recommendation
        assert "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99" in recommendation
        assert "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94" in recommendation
        mock_llm.generate.assert_called_once()

    async def test_focused_statutory_answer_supports_any_retrieved_article(self, mock_llm, agent):
        mock_llm.generate = AsyncMock(return_value=LlmResult(
            text=json.dumps({
                "recommendation": "Article 77 answers the user question from the retrieved statute.",
                "analysis": "The answer is grounded only in Article 77 text.",
                "action_steps": ["Apply the Article 77 conditions to the facts."],
                "confidence": 0.86,
            }),
            model="stub",
            provider="stub",
        ))
        research = {
            "retrieved_documents": [
                {
                    "type": "law",
                    "title": "Example Lao Law",
                    "section": "Article 77",
                    "content": "Article 77. A generic statutory rule with conditions that answer the question.",
                    "final_score": 10.0,
                }
            ],
            "query_analysis": {
                "authority_hints": [{"law_name": "Example Lao Law", "article": "77"}],
            },
        }

        result = await agent.run(
            question="What does Article 77 require?",
            research=research,
            document=None,
            evidence=None,
            memory={"empty": True},
            response_language="en",
            query_mode="general",
            response_style="plain",
            query_type="legal_question",
        )

        assert result.ok
        assert result.data["irac"]["rule"]["statutes"][0]["section"] == "Article 77"
        assert "Article 77" in result.data["irac"]["conclusion"]["recommendation"]
        mock_llm.generate.assert_called_once()

    async def test_fallback_suppresses_prompt_leak_text(self, agent):
        response = agent._fallback_response(
            LAND_RIGHT_PROTECTION_Q,
            "Answer using the CONTEXT block only. STRICT GENERATION RULES apply.",
        )

        analysis = response["irac"]["application"]["analysis"]
        assert "CONTEXT block" not in analysis
        assert "STRICT GENERATION" not in analysis
        assert response["_confidence"] == 0.3

    async def test_agent_result_has_name(self, agent):
        result = await agent.run(
            question="test", research=None, document=None, evidence=None, memory={}
        )
        assert result.agent_name == "reasoning"
