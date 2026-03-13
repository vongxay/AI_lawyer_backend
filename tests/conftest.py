"""
tests/conftest.py
=================
Shared pytest fixtures for all test modules.

Fixtures provided:
    anyio_backend     — force asyncio backend (required for pytest-asyncio)
    mock_llm          — LlmService stub returning predictable output
    mock_supabase     — AsyncMock Supabase client
    mock_redis        — AsyncMock Redis client
    sample_irac       — Complete valid IRAC dict for reuse across tests
    sample_citations  — List of citation dicts with mixed statuses
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.llm_service import LlmResult


# ── Async backend ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ── LLM mock ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LlmService that returns a valid IRAC JSON string."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LlmResult(
        text=json.dumps({
            "irac": {
                "issue": {"primary": "stub issue", "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": "stub analysis",
                    "strengths": ["strength 1"],
                    "weaknesses": [],
                    "counter_args": [],
                    "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": "stub recommendation",
                    "action_steps": ["action 1"],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.72,
                    "settlement_note": None,
                },
            },
            "confidence": 0.85,
            "citations": [{"ref": "มาตรา 420", "status": "UNVERIFIED"}],
        }),
        model="claude-sonnet-4-20250514",
        input_tokens=400,
        output_tokens=250,
        provider="anthropic",
    ))
    return llm


# ── DB mocks ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_supabase():
    """Minimal AsyncMock Supabase client — returns empty data by default."""
    sb = MagicMock()
    # Table chain: .table().select().eq().execute() → data=[]
    table_mock = MagicMock()
    table_mock.select = MagicMock(return_value=table_mock)
    table_mock.eq = MagicMock(return_value=table_mock)
    table_mock.ilike = MagicMock(return_value=table_mock)
    table_mock.upsert = MagicMock(return_value=table_mock)
    table_mock.insert = MagicMock(return_value=table_mock)
    table_mock.update = MagicMock(return_value=table_mock)
    table_mock.order = MagicMock(return_value=table_mock)
    table_mock.range = MagicMock(return_value=table_mock)
    table_mock.limit = MagicMock(return_value=table_mock)
    table_mock.single = MagicMock(return_value=table_mock)
    table_mock.execute = AsyncMock(return_value=MagicMock(data=[]))
    sb.table = MagicMock(return_value=table_mock)

    # RPC chain: .rpc().execute() → data=[]
    rpc_mock = MagicMock()
    rpc_mock.execute = AsyncMock(return_value=MagicMock(data=[]))
    sb.rpc = MagicMock(return_value=rpc_mock)

    return sb


@pytest.fixture
def mock_redis():
    """Minimal AsyncMock Redis client."""
    r = MagicMock()
    r.get = AsyncMock(return_value=None)       # Cache miss by default
    r.setex = AsyncMock(return_value=True)
    r.ping = AsyncMock(return_value=True)
    return r


# ── Sample data ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_irac() -> dict:
    return {
        "irac": {
            "issue": {
                "primary": "นายจ้างเลิกจ้างโดยไม่จ่ายค่าชดเชยตามกฎหมายหรือไม่",
                "secondary": ["การเลิกจ้างชอบด้วยกฎหมายหรือไม่", "สิทธิได้รับค่าชดเชย"],
            },
            "rule": {
                "statutes": [
                    {
                        "name": "พระราชบัญญัติคุ้มครองแรงงาน",
                        "section": "มาตรา 118",
                        "text": "นายจ้างซึ่งเลิกจ้างลูกจ้างต้องจ่ายค่าชดเชย",
                        "status": "ACTIVE",
                        "year": 2541,
                    }
                ],
                "precedents": [
                    {
                        "case_no": "ฎ. 1234/2560",
                        "court": "ศาลฎีกา",
                        "relevance": "วางหลักเรื่องการเลิกจ้างไม่เป็นธรรม",
                        "outcome": "ลูกจ้างชนะ",
                        "graph_path": None,
                    }
                ],
            },
            "application": {
                "analysis": "จากข้อเท็จจริง นายจ้างเลิกจ้างโดยไม่แจ้งล่วงหน้าและไม่จ่ายค่าชดเชย",
                "strengths": ["มีหลักฐานการทำงานครบ 3 ปี", "ไม่มีการกระทำผิดของลูกจ้าง"],
                "weaknesses": ["อาจต้องพิสูจน์ว่าไม่ได้ลาออกเอง"],
                "counter_args": ["นายจ้างอาจอ้างว่าลูกจ้างละทิ้งหน้าที่"],
                "rebuttals": ["มีพยานยืนยันว่าถูกไล่ออก"],
            },
            "conclusion": {
                "recommendation": "ยื่นคำร้องต่อพนักงานตรวจแรงงานหรือฟ้องต่อศาลแรงงาน",
                "action_steps": [
                    "รวบรวมสัญญาจ้าง สลิปเงินเดือน",
                    "ยื่นคำร้องภายใน 30 วัน",
                    "ปรึกษาทนายแรงงาน",
                ],
                "risk_level": "LOW",
                "win_probability": 0.78,
                "settlement_note": "อาจเจรจาให้นายจ้างจ่ายค่าชดเชยโดยสมัครใจก่อนฟ้อง",
            },
        },
        "confidence": 0.87,
        "citations": [
            {"ref": "พระราชบัญญัติคุ้มครองแรงงาน มาตรา 118", "status": "VERIFIED"},
            {"ref": "ฎ. 1234/2560", "status": "VERIFIED"},
        ],
    }


@pytest.fixture
def sample_citations() -> list[dict]:
    return [
        {"ref": "ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420", "status": "UNVERIFIED"},
        {"ref": "ฎ. 5678/2562", "status": "UNVERIFIED"},
        {"ref": "fake citation that does not exist", "status": "UNVERIFIED"},
    ]
