"""
services/pii_service.py
========================
PII detection and redaction layer.

MUST be applied to all user input BEFORE it is sent to any external LLM API.
Redacted text is logged with a hash so it can be reconstructed if needed
(by an authorised admin — never sent to LLM).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from backend.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    found_types: list[str]     # categories of PII found (no actual values)
    redaction_count: int


class PiiService:
    """
    Rule-based PII redaction for Thai / English legal text.

    Patterns covered:
    - Thai national ID (13 digits)
    - Thai phone numbers (0X-XXXX-XXXX)
    - Thai bank account numbers
    - Email addresses
    - Passport numbers (Thai + common formats)
    - IP addresses
    - Generic credit card numbers (Luhn-like patterns)
    """

    _PATTERNS: list[tuple[str, re.Pattern]] = [
        ("THAI_ID",       re.compile(r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b")),       # 1-2345-67890-12-3
        ("THAI_ID",       re.compile(r"\b\d{13}\b")),                                 # raw 13 digits
        ("PHONE_TH",      re.compile(r"\b0[689]\d[-\s]?\d{3}[-\s]?\d{4}\b")),        # mobile
        ("PHONE_TH",      re.compile(r"\b0[2-9]\d{1}[-\s]?\d{3}[-\s]?\d{4}\b")),    # landline
        ("EMAIL",         re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
        ("BANK_ACCOUNT",  re.compile(r"\b\d{3}-\d{1}-\d{5}-\d{1}\b")),              # Thai bank format
        ("PASSPORT",      re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
        ("IP_ADDRESS",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
        ("CREDIT_CARD",   re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
    ]

    def redact(self, text: str) -> RedactionResult:
        result = text
        found_types: list[str] = []
        total_count = 0

        for pii_type, pattern in self._PATTERNS:
            matches = pattern.findall(result)
            if matches:
                found_types.append(pii_type)
                total_count += len(matches)
                result = pattern.sub(f"[REDACTED_{pii_type}]", result)

        if found_types:
            log.info(
                "pii.redacted",
                types=list(set(found_types)),
                count=total_count,
                text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            )

        return RedactionResult(
            redacted_text=result,
            found_types=list(set(found_types)),
            redaction_count=total_count,
        )

    def redact_text(self, text: str) -> str:
        """Convenience method — returns redacted string only."""
        return self.redact(text).redacted_text
