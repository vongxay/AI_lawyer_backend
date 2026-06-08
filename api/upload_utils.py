"""
Shared upload helpers for bounded in-memory file reads.
"""
from __future__ import annotations

from fastapi import UploadFile

from core.config import Settings
from core.exceptions import FileTooLargeError

READ_CHUNK_BYTES = 1024 * 1024
LAW_CODE_UPLOAD_TARGET_KB = 50_000


def upload_limit_bytes(settings: Settings) -> int:
    return settings.max_upload_size_bytes


def request_body_limit_bytes(settings: Settings) -> int:
    return settings.max_request_body_bytes


def supports_law_code_upload_target(settings: Settings) -> bool:
    return upload_limit_bytes(settings) >= LAW_CODE_UPLOAD_TARGET_KB * 1024


async def read_upload_with_limit(upload: UploadFile, *, max_bytes: int, label: str | None = None) -> bytes:
    file_name = label or upload.filename or "uploaded file"
    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, int) and declared_size > max_bytes:
        raise _file_too_large(file_name, declared_size, max_bytes)

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _file_too_large(file_name, total, max_bytes)
        chunks.append(chunk)

    return b"".join(chunks)


def _file_too_large(file_name: str, actual_bytes: int, max_bytes: int) -> FileTooLargeError:
    actual_mb = actual_bytes / (1024 * 1024)
    max_mb = max_bytes / (1024 * 1024)
    return FileTooLargeError(f"'{file_name}' ({actual_mb:.1f}MB) exceeds limit of {max_mb:.0f}MB")
