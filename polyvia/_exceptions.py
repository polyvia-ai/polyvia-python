"""Polyvia SDK exceptions."""

from __future__ import annotations


class PolyviaError(Exception):
    """Base exception for all Polyvia SDK errors."""


class APIError(PolyviaError):
    """An error response from the Polyvia API."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AuthenticationError(APIError):
    """401 — missing or invalid API key."""


class ForbiddenError(APIError):
    """403 — the resource belongs to another user."""


class NotFoundError(APIError):
    """404 — document, group, or task not found."""


class RateLimitError(APIError):
    """429 — too many requests."""


class ServiceUnavailableError(APIError):
    """503 — database or upstream service is temporarily unavailable."""


class IngestionError(PolyviaError):
    """A document ingestion task finished with status='failed'."""

    def __init__(self, task_id: str, error: str | None) -> None:
        self.task_id = task_id
        self.error = error
        super().__init__(f"Ingestion task {task_id!r} failed: {error}")


class IngestionTimeout(PolyviaError):
    """ingest.wait() exceeded its timeout before the task completed."""

    def __init__(self, task_id: str, timeout: float) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id!r} did not complete within {timeout}s")
