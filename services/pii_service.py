from __future__ import annotations

import re


class PiiService:
    _phone = re.compile(r"\b0\d{8,9}\b")
    _id = re.compile(r"\b\d{13}\b")

    def redact(self, text: str) -> str:
        text = self._phone.sub("[REDACTED_PHONE]", text)
        text = self._id.sub("[REDACTED_ID]", text)
        return text

