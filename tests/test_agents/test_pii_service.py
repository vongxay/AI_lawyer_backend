"""
Tests for PiiService — critical for data protection compliance.
"""
from __future__ import annotations

import pytest

from backend.services.pii_service import PiiService


@pytest.fixture
def svc() -> PiiService:
    return PiiService()


class TestPiiService:
    def test_redacts_thai_phone_mobile(self, svc):
        result = svc.redact("โทร 0812345678 ด่วน")
        assert "0812345678" not in result.redacted_text
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "PHONE_TH" in result.found_types

    def test_redacts_thai_national_id(self, svc):
        result = svc.redact("บัตรประชาชน 1234567890123")
        assert "1234567890123" not in result.redacted_text
        assert "THAI_ID" in result.found_types

    def test_redacts_email(self, svc):
        result = svc.redact("ส่งมาที่ user@example.com นะ")
        assert "user@example.com" not in result.redacted_text
        assert "EMAIL" in result.found_types

    def test_no_pii_unchanged(self, svc):
        text = "ฉันต้องการคำปรึกษาเรื่องสัญญาเช่าบ้าน"
        result = svc.redact(text)
        assert result.redacted_text == text
        assert result.found_types == []
        assert result.redaction_count == 0

    def test_multiple_pii_types_in_one_string(self, svc):
        text = "ติดต่อ 0912345678 หรือ email@test.com"
        result = svc.redact(text)
        assert "0912345678" not in result.redacted_text
        assert "email@test.com" not in result.redacted_text
        assert result.redaction_count >= 2

    def test_convenience_method(self, svc):
        redacted = svc.redact_text("call me at 0891234567")
        assert "0891234567" not in redacted
