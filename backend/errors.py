"""Error mappings and exceptions for the FastAPI backend."""

from __future__ import annotations

from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse

from common.schemas import ErrorBody, ErrorCode, ErrorResponse

ERROR_STATUS_MAP: dict[ErrorCode, int] = {
    "UNSUPPORTED_FORMAT": 415,
    "FILE_TOO_LARGE": 413,
    "EMPTY_DOCUMENT": 422,
    "PARSE_FAILED": 422,
    "INVALID_REQUEST": 422,
    "DOCUMENT_NOT_FOUND": 404,
    "INDEX_EMPTY": 409,
    "INTERNAL_ERROR": 500,
}


class APIError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code or ERROR_STATUS_MAP.get(code, 500)
        self.details = details or {}


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    content = ErrorResponse(
        error=ErrorBody(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
    ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=content)
