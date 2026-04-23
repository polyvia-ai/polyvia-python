"""Sync and async HTTP transports built on httpx."""

from __future__ import annotations

from typing import Any

import httpx

from ._exceptions import (
    APIError,
    AuthenticationError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

DEFAULT_BASE_URL = "https://app.polyvia.ai"
DEFAULT_TIMEOUT = 60.0


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        detail: str = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text

    code = resp.status_code
    exc_cls: type[APIError]
    if code == 401:
        exc_cls = AuthenticationError
    elif code == 403:
        exc_cls = ForbiddenError
    elif code == 404:
        exc_cls = NotFoundError
    elif code == 429:
        exc_cls = RateLimitError
    elif code == 503:
        exc_cls = ServiceUnavailableError
    else:
        exc_cls = APIError
    raise exc_cls(code, detail)


class SyncTransport:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=True,
        )

    def get(self, path: str, **kwargs: Any) -> Any:
        resp = self._http.get(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    def post(self, path: str, **kwargs: Any) -> Any:
        resp = self._http.post(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    def patch(self, path: str, **kwargs: Any) -> Any:
        resp = self._http.patch(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    def delete(self, path: str, **kwargs: Any) -> Any:
        resp = self._http.delete(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SyncTransport":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncTransport:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=True,
        )

    async def get(self, path: str, **kwargs: Any) -> Any:
        resp = await self._http.get(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    async def post(self, path: str, **kwargs: Any) -> Any:
        resp = await self._http.post(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    async def patch(self, path: str, **kwargs: Any) -> Any:
        resp = await self._http.patch(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    async def delete(self, path: str, **kwargs: Any) -> Any:
        resp = await self._http.delete(path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncTransport":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
