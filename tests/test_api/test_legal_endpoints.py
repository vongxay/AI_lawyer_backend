"""
Integration tests for legal API endpoints using FastAPI TestClient.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agents.base_agent import AgentResult
from main import create_app
from orchestrator.workflow_manager import OrchestrationResult


def _mock_orchestration_result() -> OrchestrationResult:
    return OrchestrationResult(
        response={
            "irac": {
                "issue": {"primary": "test issue", "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": "test",
                    "strengths": [], "weaknesses": [],
                    "counter_args": [], "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": "test",
                    "action_steps": [],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.7,
                    "settlement_note": None,
                },
            },
            "citations": [],
            "citations_verified": True,
            "confidence": 0.85,
            "agents_used": ["research", "reasoning", "verification"],
            "processing_time_ms": 1500,
            "escalated_to_expert": False,
            "risk": None,
            "disclaimer": "For informational purposes only.",
        },
        confidence=0.85,
        agents_used=["research", "reasoning", "verification"],
        processing_time_ms=1500,
        escalated_to_expert=False,
    )


@pytest.fixture
def client():
    app = create_app()

    # Override the WorkflowManager dependency
    mock_workflow = MagicMock()
    mock_workflow.orchestrate = AsyncMock(return_value=_mock_orchestration_result())
    mock_workflow._verification_agent = MagicMock()
    mock_workflow._verification_agent.run = AsyncMock(return_value=AgentResult(
        data={"citations": [], "citations_verified": True, "rejection_rate": 0.0},
        agent_name="verification",
    ))

    from api.dependencies import get_workflow_manager
    app.dependency_overrides[get_workflow_manager] = lambda: mock_workflow

    # Skip DB connections
    with patch("backend.core.database.ping_redis", AsyncMock(return_value=True)), \
         patch("backend.core.database.ping_supabase", AsyncMock(return_value=False)):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestLegalQueryEndpoint:
    def test_post_query_returns_200(self, client):
        response = client.post(
            "/api/v1/legal/query",
            json={"question": "สัญญาเช่าถูกยกเลิกโดยไม่แจ้งล่วงหน้า มีสิทธิอะไรบ้าง"},
        )
        assert response.status_code == 200

    def test_response_has_irac_structure(self, client):
        response = client.post(
            "/api/v1/legal/query",
            json={"question": "สัญญาเช่าถูกยกเลิก"},
        )
        data = response.json()
        assert "irac" in data
        assert "confidence" in data
        assert "disclaimer" in data
        assert "agents_used" in data

    def test_empty_question_returns_422(self, client):
        response = client.post("/api/v1/legal/query", json={"question": ""})
        assert response.status_code == 422

    def test_too_short_question_returns_422(self, client):
        response = client.post("/api/v1/legal/query", json={"question": "ab"})
        assert response.status_code == 422

    def test_missing_question_returns_422(self, client):
        response = client.post("/api/v1/legal/query", json={})
        assert response.status_code == 422

    def test_with_case_id(self, client):
        response = client.post(
            "/api/v1/legal/query",
            json={"question": "อัปเดตคดีของฉัน", "case_id": "case-abc-123"},
        )
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded", "down")
        assert "version" in data

    def test_verify_citations_endpoint(self, client):
        response = client.post(
            "/api/v1/legal/citations/verify",
            json={"citations": [{"ref": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420", "status": "UNVERIFIED"}]},
        )
        assert response.status_code == 200

    def test_precedent_graph_endpoint(self, client):
        response = client.get("/api/v1/legal/graph/ฎ.1234%2F2560")
        assert response.status_code == 200


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client):
        response = client.post(
            "/api/v1/feedback/",
            json={"session_id": "sess-001", "rating": 4, "comment": "ดีมาก"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_rating_out_of_range_rejected(self, client):
        response = client.post(
            "/api/v1/feedback/",
            json={"session_id": "sess-001", "rating": 6},
        )
        assert response.status_code == 422
