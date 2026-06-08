from __future__ import annotations

from io import BytesIO

import pytest

from api.upload_utils import (
    LAW_CODE_UPLOAD_TARGET_KB,
    read_upload_with_limit,
    request_body_limit_bytes,
    supports_law_code_upload_target,
    upload_limit_bytes,
)
from core.config import Settings
from core.exceptions import FileTooLargeError


class FakeUpload:
    def __init__(self, content: bytes, *, filename: str = "law.pdf", size: int | None = None) -> None:
        self.filename = filename
        self.size = size
        self._stream = BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_default_upload_limits_support_50000kb_law_code_pdf() -> None:
    settings = Settings(max_upload_size_mb=50, max_request_body_mb=55)

    assert supports_law_code_upload_target(settings) is True
    assert upload_limit_bytes(settings) >= LAW_CODE_UPLOAD_TARGET_KB * 1024
    assert request_body_limit_bytes(settings) > upload_limit_bytes(settings)


async def test_read_upload_with_limit_allows_exact_boundary() -> None:
    content = b"x" * 16
    upload = FakeUpload(content, size=len(content))

    assert await read_upload_with_limit(upload, max_bytes=len(content)) == content


async def test_read_upload_with_limit_rejects_declared_oversize_before_reading() -> None:
    upload = FakeUpload(b"", size=17)

    with pytest.raises(FileTooLargeError):
        await read_upload_with_limit(upload, max_bytes=16)


async def test_read_upload_with_limit_rejects_streamed_oversize() -> None:
    upload = FakeUpload(b"x" * 17, size=None)

    with pytest.raises(FileTooLargeError):
        await read_upload_with_limit(upload, max_bytes=16)
