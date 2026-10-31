"""Ingestion failures. Owner: Member 2. backend/errors.py maps code -> HTTP status."""

from common.schemas import ErrorCode


class IngestionError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
