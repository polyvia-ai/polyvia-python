"""
Polyvia Python SDK — sync and async clients.

Typical usage::

    from polyvia import Polyvia

    client = Polyvia(api_key="poly_...")

    # Ingest a document and wait for it to be ready
    result = client.ingest.file("report.pdf", name="Q4 Report")
    doc    = client.ingest.wait(result.task_id)

    # Query it
    answer = client.query("What are the key findings?", document_id=doc.document_id)
    print(answer.answer)

    # Connect an AI assistant via MCP
    client.mcp.print_claude_desktop_snippet()
"""

from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from ._exceptions import IngestionError, IngestionTimeout, NotFoundError
from ._models import (
    BatchIngestItem,
    BatchIngestResult,
    Document,
    Group,
    IngestionStatus,
    IngestResult,
    QueryResult,
    RateLimits,
    Usage,
)
from ._tools import as_anthropic_tools, as_langchain_tools, as_openai_tools
from ._transport import AsyncTransport, SyncTransport
from .mcp import MCPConfig

# Direct uploads to Convex storage need a generous timeout — large PDFs over
# slow connections can take a while. Separate from the SDK's API-call timeout.
_DIRECT_UPLOAD_TIMEOUT = 300.0

# The ingest-status endpoint tracks tasks in memory, so on serverless backends a
# poll can land on a different instance and 404 even for a live task. When that
# happens we fall back to the document's persisted status (the source of truth),
# mapping its document-status vocabulary onto the ingestion-status one.
_DOC_STATUS_TO_TASK = {
    "uploading": "pending",
    "parsing": "parsing",
    "completed": "completed",
    "failed": "failed",
}


def _mime_for(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# Accepts a group by human name (str) or a Group object. Most callers should
# just pass the name — the SDK resolves it to the backend group id for you, so
# you never have to track the opaque id yourself.
GroupRef = Union[str, "Group"]


def _resolve_group_id(
    t: SyncTransport,
    group: Optional[GroupRef],
    group_id: Optional[str],
    *,
    create: bool,
) -> Optional[str]:
    """Resolve a ``group`` (name or Group) / explicit ``group_id`` to a group id.

    ``group_id`` wins if given. A ``group`` name is looked up by exact match;
    with ``create=True`` (ingest) it's created when missing, otherwise (query)
    a missing name raises a clear error.
    """
    if group_id is not None:
        return group_id
    if group is None:
        return None
    if isinstance(group, Group):
        return group.id
    name = str(group)
    for g in t.get("/api/v1/groups").get("groups", []):
        if g.get("name") == name:
            return g.get("id")
    if create:
        return t.post("/api/v1/groups", json={"name": name})["group_id"]
    raise ValueError(
        f"No group named {name!r}. Pass group_id=..., or create it first with "
        f"client.groups.get_or_create({name!r})."
    )


async def _resolve_group_id_async(
    t: AsyncTransport,
    group: Optional[GroupRef],
    group_id: Optional[str],
    *,
    create: bool,
) -> Optional[str]:
    """Async twin of :func:`_resolve_group_id`."""
    if group_id is not None:
        return group_id
    if group is None:
        return None
    if isinstance(group, Group):
        return group.id
    name = str(group)
    listed = await t.get("/api/v1/groups")
    for g in listed.get("groups", []):
        if g.get("name") == name:
            return g.get("id")
    if create:
        created = await t.post("/api/v1/groups", json={"name": name})
        return created["group_id"]
    raise ValueError(
        f"No group named {name!r}. Pass group_id=..., or create it first with "
        f"client.groups.get_or_create({name!r})."
    )


# ── Sync resource namespaces ──────────────────────────────────────────────────


class IngestResource:
    """client.ingest — document ingestion methods."""

    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def _upload_one(
        self,
        path: Path,
        *,
        name: Optional[str],
        group_id: Optional[str],
    ) -> Dict[str, Any]:
        """Direct-to-storage upload + finalize for one file. Returns the raw
        ``{document_id, task_id, status}`` dict from the API."""
        mime = _mime_for(path)
        url_resp = self._t.post("/api/v1/ingest/upload-url")
        upload_url = url_resp["upload_url"]

        # PUT bytes directly to Convex storage so the upload never touches
        # the API server (and isn't subject to Vercel's 4.5 MB body limit).
        # Use a fresh httpx client so we don't leak our Bearer token to Convex.
        with path.open("rb") as fh, httpx.Client(
            timeout=_DIRECT_UPLOAD_TIMEOUT, follow_redirects=True
        ) as http:
            # Convex storage upload URLs require POST (PUT returns 405).
            resp = http.post(upload_url, content=fh.read(), headers={"Content-Type": mime})
            resp.raise_for_status()
            storage_id = resp.json()["storageId"]

        body: Dict[str, Any] = {
            "storage_id": storage_id,
            "file_type": mime,
            "name": name or path.name,
        }
        if group_id:
            body["group_id"] = group_id
        return self._t.post("/api/v1/ingest/finalize", json=body)

    def file(
        self,
        path: str | Path,
        *,
        name: Optional[str] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
    ) -> IngestResult:
        """Upload a single file and queue it for parsing.

        Uploads the file bytes directly to Polyvia's storage backend (no
        API-server proxy), so there is no practical file-size cap from the
        SDK itself.

        Parameters
        ----------
        path:
            Path to the file on disk.
        name:
            Display name in Polyvia. Defaults to the filename.
        group:
            Group to file the document under — pass the group **name** (created
            if it doesn't exist yet) or a :class:`Group`. Prefer this over
            ``group_id``; you rarely need the backend id.
        group_id:
            Explicit backend group id (advanced). Takes precedence over ``group``.

        Returns
        -------
        IngestResult
            Contains ``document_id`` and ``task_id``. Poll
            :meth:`status` or call :meth:`wait` to track progress.
        """
        gid = _resolve_group_id(self._t, group, group_id, create=True)
        raw = self._upload_one(Path(path), name=name, group_id=gid)
        return IngestResult(**raw)

    def batch(
        self,
        paths: List[str | Path],
        *,
        names: Optional[List[str]] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
    ) -> BatchIngestResult:
        """Upload multiple files. Each file is uploaded directly to storage
        and finalized independently — a failure on one file doesn't affect
        the others.

        Parameters
        ----------
        paths:
            List of file paths.
        names:
            Optional list of display names aligned to ``paths``.
        group:
            Group for every document — pass the group **name** (created once if
            it doesn't exist) or a :class:`Group`. Resolved a single time for the
            whole batch. Prefer this over ``group_id``.
        group_id:
            Explicit backend group id (advanced). Takes precedence over ``group``.
        """
        gid = _resolve_group_id(self._t, group, group_id, create=True)
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for i, raw_path in enumerate(paths):
            p = Path(raw_path)
            display_name = names[i] if names and i < len(names) else None
            try:
                results.append(self._upload_one(p, name=display_name, group_id=gid))
            except Exception as e:
                err = {"file": p.name, "error": str(e)}
                results.append(err)
                errors.append(err)

        items = [BatchIngestItem(**r) for r in results]
        return BatchIngestResult(results=items, errors=errors or None)

    def status(self, task_id: str) -> IngestionStatus:
        """Return the current status of an ingestion task.

        Falls back to the document's persisted status if the in-memory
        ingest-status endpoint can't find the task (e.g. the poll hit a
        different serverless instance than the one that started it).
        """
        try:
            raw = self._t.get(f"/api/v1/ingest/{task_id}")
            return IngestionStatus(**raw)
        except NotFoundError:
            doc = self._t.get(f"/api/v1/documents/{task_id}")
            return IngestionStatus(
                task_id=task_id,
                document_id=doc.get("id"),
                status=_DOC_STATUS_TO_TASK.get(doc.get("status", ""), doc.get("status", "")),
                error=None,
            )

    def wait(
        self,
        task_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> IngestionStatus:
        """Block until the ingestion task reaches a terminal state.

        Parameters
        ----------
        task_id:
            From :meth:`file` or :meth:`batch`.
        poll_interval:
            Seconds between status checks (default 3).
        timeout:
            Maximum seconds to wait before raising :exc:`IngestionTimeout`.

        Returns
        -------
        IngestionStatus
            With ``status='completed'``.

        Raises
        ------
        IngestionError
            If the task finishes with ``status='failed'``.
        IngestionTimeout
            If ``timeout`` is exceeded.
        """
        deadline = time.monotonic() + timeout
        while True:
            st = self.status(task_id)
            if st.status == "completed":
                return st
            if st.status == "failed":
                raise IngestionError(task_id, st.error)
            if time.monotonic() >= deadline:
                raise IngestionTimeout(task_id, timeout)
            time.sleep(poll_interval)


class DocumentsResource:
    """client.documents — document CRUD methods."""

    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def list(
        self,
        *,
        status: Optional[str] = None,
        group_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """List documents in the workspace."""
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if group_ids:
            params["group_ids"] = ",".join(group_ids)
        elif group_id:
            params["group_id"] = group_id
        raw = self._t.get("/api/v1/documents", params=params or None)
        return [Document(**d) for d in raw["documents"]]

    def get(self, document_id: str) -> Document:
        """Get metadata for a single document."""
        raw = self._t.get(f"/api/v1/documents/{document_id}")
        return Document(**raw)

    def update(self, document_id: str, *, group_id: Optional[str] = None) -> Dict[str, Any]:
        """Update a document's metadata (currently: group assignment).

        Pass ``group_id=None`` explicitly to remove it from its current group.
        """
        return self._t.patch(f"/api/v1/documents/{document_id}", json={"group_id": group_id})

    def delete(self, document_id: str) -> Dict[str, Any]:
        """Permanently delete a document and its stored file."""
        return self._t.delete(f"/api/v1/documents/{document_id}")


class GroupsResource:
    """client.groups — group CRUD methods."""

    def __init__(self, transport: SyncTransport) -> None:
        self._t = transport

    def list(self) -> List[Group]:
        """List all groups in the workspace."""
        raw = self._t.get("/api/v1/groups")
        return [Group(**g) for g in raw["groups"]]

    def create(self, name: str) -> Dict[str, Any]:
        """Create a new group and return ``{group_id, name}``.

        Note: this always creates a new group, even if one with the same name
        already exists. Prefer :meth:`get_or_create` unless you specifically
        want a fresh group every time.
        """
        return self._t.post("/api/v1/groups", json={"name": name})

    def find(self, name: str) -> Optional[Group]:
        """Return the group with this exact ``name``, or ``None`` if there isn't one."""
        for g in self.list():
            if g.name == name:
                return g
        return None

    def get_or_create(self, name: str) -> Group:
        """Return the group named ``name``, creating it if it doesn't exist.

        Idempotent and matched by exact name, so it never makes duplicate
        groups — the easy way to turn a human name into a usable group without
        ever handling the backend group id yourself::

            group = client.groups.get_or_create("Q4 Earnings")
            client.ingest.file("10k.pdf", group=group)   # or group="Q4 Earnings"
        """
        existing = self.find(name)
        if existing is not None:
            return existing
        raw = self.create(name)
        return Group(id=raw["group_id"], name=raw.get("name", name))

    def delete_documents(self, group_id: str) -> Dict[str, Any]:
        """Delete all documents in the group. The group itself is kept."""
        return self._t.delete(f"/api/v1/groups/{group_id}/documents")

    def delete(self, group_id: str, *, delete_documents: bool = False) -> Dict[str, Any]:
        """Delete a group.

        Parameters
        ----------
        delete_documents:
            If ``True``, first delete all documents in the group.
            If ``False`` (default) and the group still has documents, raises a
            :exc:`~polyvia.APIError` 400.
        """
        if delete_documents:
            self.delete_documents(group_id)
        return self._t.delete(f"/api/v1/groups/{group_id}")


class ToolsResource:
    """client.tools — agent tool adapters."""

    def __init__(self, client: "Polyvia") -> None:
        self._client = client

    def openai(self) -> Tuple[List[Dict[str, Any]], Any]:
        """Return ``(tools, call_tool)`` for the OpenAI ChatCompletion API.

        Example::

            import json, openai
            tools, call = client.tools.openai()

            response = openai.chat.completions.create(
                model="gpt-4o", messages=[...], tools=tools
            )
            for tc in response.choices[0].message.tool_calls or []:
                result = call(tc.function.name, json.loads(tc.function.arguments))
        """
        return as_openai_tools(self._client)

    def anthropic(self) -> Tuple[List[Dict[str, Any]], Any]:
        """Return ``(tools, call_tool)`` for the Anthropic Messages API.

        Example::

            import anthropic as ant
            tools, call = client.tools.anthropic()

            response = ant.Anthropic().messages.create(
                model="claude-opus-4-5", messages=[...], tools=tools
            )
            for block in response.content:
                if block.type == "tool_use":
                    result = call(block.name, block.input)
        """
        return as_anthropic_tools(self._client)

    def langchain(self) -> List[Any]:
        """Return LangChain ``BaseTool`` instances.

        Requires ``pip install polyvia[langchain]``.

        Example::

            from langchain_openai import ChatOpenAI
            from langchain.agents import AgentExecutor, create_tool_calling_agent

            tools = client.tools.langchain()
            agent = create_tool_calling_agent(ChatOpenAI(model="gpt-4o"), tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools)
            executor.invoke({"input": "What do my documents say about Q4?"})
        """
        return as_langchain_tools(self._client)


# ── Main sync client ──────────────────────────────────────────────────────────


class Polyvia:
    """Synchronous Polyvia client.

    Parameters
    ----------
    api_key:
        Your ``poly_...`` API key. If omitted, the ``POLYVIA_API_KEY``
        environment variable is used.
    base_url:
        Override the API base URL (default: ``https://app.polyvia.ai``).
    timeout:
        HTTP request timeout in seconds (default: 60).

    Example
    -------
    ::

        import os
        from polyvia import Polyvia

        client = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])

        result = client.ingest.file("report.pdf")
        client.ingest.wait(result.task_id)

        print(client.query("Summarise the report.").answer)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://app.polyvia.ai",
        timeout: float = 60.0,
    ) -> None:
        import os

        resolved_key = api_key or os.environ.get("POLYVIA_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "api_key is required. Pass it explicitly or set the POLYVIA_API_KEY env var."
            )

        self._transport = SyncTransport(resolved_key, base_url=base_url, timeout=timeout)
        self._api_key = resolved_key
        self._base_url = base_url

        # Resource namespaces
        self.ingest = IngestResource(self._transport)
        self.documents = DocumentsResource(self._transport)
        self.groups = GroupsResource(self._transport)
        self.tools = ToolsResource(self)

    # ── Top-level methods ─────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        document_id: Optional[str] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ) -> QueryResult:
        """Ask a natural-language question about your documents.

        Parameters
        ----------
        question:
            Your question (max 2 000 characters).
        document_id:
            Scope to a single document (fastest).
        group:
            Scope to one group by **name** (or pass a :class:`Group`). The group
            must already exist. Prefer this over ``group_id``.
        group_id:
            Scope to one group by backend id.
        group_ids:
            Scope to multiple groups (takes precedence over ``group`` / ``group_id``).

        Returns
        -------
        QueryResult
            Contains ``answer`` and optionally ``document_id`` / ``group_ids``.
        """
        if group is not None and group_id is None and group_ids is None:
            group_id = _resolve_group_id(self._transport, group, None, create=False)
        body: Dict[str, Any] = {"query": question}
        if document_id:
            body["document_id"] = document_id
        elif group_ids:
            body["group_ids"] = group_ids
        elif group_id:
            body["group_id"] = group_id
        raw = self._transport.post("/api/v1/query", json=body)
        return QueryResult(**raw)

    def usage(self) -> Usage:
        """Return usage statistics for the current API key."""
        raw = self._transport.get("/api/v1/usage")
        return Usage(**raw)

    def rate_limits(self) -> RateLimits:
        """Return rate-limit configuration and current window usage."""
        raw = self._transport.get("/api/v1/rate-limits")
        return RateLimits(**raw)

    # ── MCP property ──────────────────────────────────────────

    @property
    def mcp(self) -> MCPConfig:
        """MCP server connection configuration.

        Use to connect Claude Desktop, OpenAI Agents, or any MCP-compatible
        client to the Polyvia hosted MCP server.

        Example::

            client.mcp.print_claude_desktop_snippet()
        """
        return MCPConfig(
            url=f"{self._base_url}/mcp",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    # ── Context manager ───────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._transport.close()

    def __enter__(self) -> "Polyvia":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Async resource namespaces ─────────────────────────────────────────────────


class AsyncIngestResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def _upload_one(
        self,
        path: Path,
        *,
        name: Optional[str],
        group_id: Optional[str],
    ) -> Dict[str, Any]:
        mime = _mime_for(path)
        url_resp = await self._t.post("/api/v1/ingest/upload-url")
        upload_url = url_resp["upload_url"]

        with path.open("rb") as fh:
            data = fh.read()
        async with httpx.AsyncClient(
            timeout=_DIRECT_UPLOAD_TIMEOUT, follow_redirects=True
        ) as http:
            # Convex storage upload URLs require POST (PUT returns 405).
            resp = await http.post(upload_url, content=data, headers={"Content-Type": mime})
            resp.raise_for_status()
            storage_id = resp.json()["storageId"]

        body: Dict[str, Any] = {
            "storage_id": storage_id,
            "file_type": mime,
            "name": name or path.name,
        }
        if group_id:
            body["group_id"] = group_id
        return await self._t.post("/api/v1/ingest/finalize", json=body)

    async def file(
        self,
        path: str | Path,
        *,
        name: Optional[str] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
    ) -> IngestResult:
        gid = await _resolve_group_id_async(self._t, group, group_id, create=True)
        raw = await self._upload_one(Path(path), name=name, group_id=gid)
        return IngestResult(**raw)

    async def batch(
        self,
        paths: List[str | Path],
        *,
        names: Optional[List[str]] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
    ) -> BatchIngestResult:
        gid = await _resolve_group_id_async(self._t, group, group_id, create=True)
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for i, raw_path in enumerate(paths):
            p = Path(raw_path)
            display_name = names[i] if names and i < len(names) else None
            try:
                results.append(await self._upload_one(p, name=display_name, group_id=gid))
            except Exception as e:
                err = {"file": p.name, "error": str(e)}
                results.append(err)
                errors.append(err)

        items = [BatchIngestItem(**r) for r in results]
        return BatchIngestResult(results=items, errors=errors or None)

    async def status(self, task_id: str) -> IngestionStatus:
        try:
            raw = await self._t.get(f"/api/v1/ingest/{task_id}")
            return IngestionStatus(**raw)
        except NotFoundError:
            doc = await self._t.get(f"/api/v1/documents/{task_id}")
            return IngestionStatus(
                task_id=task_id,
                document_id=doc.get("id"),
                status=_DOC_STATUS_TO_TASK.get(doc.get("status", ""), doc.get("status", "")),
                error=None,
            )

    async def wait(
        self,
        task_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> IngestionStatus:
        import asyncio

        deadline = time.monotonic() + timeout
        while True:
            st = await self.status(task_id)
            if st.status == "completed":
                return st
            if st.status == "failed":
                raise IngestionError(task_id, st.error)
            if time.monotonic() >= deadline:
                raise IngestionTimeout(task_id, timeout)
            await asyncio.sleep(poll_interval)


class AsyncDocumentsResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list(
        self,
        *,
        status: Optional[str] = None,
        group_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if group_ids:
            params["group_ids"] = ",".join(group_ids)
        elif group_id:
            params["group_id"] = group_id
        raw = await self._t.get("/api/v1/documents", params=params or None)
        return [Document(**d) for d in raw["documents"]]

    async def get(self, document_id: str) -> Document:
        raw = await self._t.get(f"/api/v1/documents/{document_id}")
        return Document(**raw)

    async def update(self, document_id: str, *, group_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._t.patch(
            f"/api/v1/documents/{document_id}", json={"group_id": group_id}
        )

    async def delete(self, document_id: str) -> Dict[str, Any]:
        return await self._t.delete(f"/api/v1/documents/{document_id}")


class AsyncGroupsResource:
    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def list(self) -> List[Group]:
        raw = await self._t.get("/api/v1/groups")
        return [Group(**g) for g in raw["groups"]]

    async def create(self, name: str) -> Dict[str, Any]:
        return await self._t.post("/api/v1/groups", json={"name": name})

    async def find(self, name: str) -> Optional[Group]:
        """Return the group with this exact ``name``, or ``None``."""
        for g in await self.list():
            if g.name == name:
                return g
        return None

    async def get_or_create(self, name: str) -> Group:
        """Return the group named ``name``, creating it if it doesn't exist (idempotent)."""
        existing = await self.find(name)
        if existing is not None:
            return existing
        raw = await self.create(name)
        return Group(id=raw["group_id"], name=raw.get("name", name))

    async def delete_documents(self, group_id: str) -> Dict[str, Any]:
        return await self._t.delete(f"/api/v1/groups/{group_id}/documents")

    async def delete(self, group_id: str, *, delete_documents: bool = False) -> Dict[str, Any]:
        if delete_documents:
            await self.delete_documents(group_id)
        return await self._t.delete(f"/api/v1/groups/{group_id}")


# ── Main async client ─────────────────────────────────────────────────────────


class AsyncPolyvia:
    """Asynchronous Polyvia client — same API as :class:`Polyvia` but all
    resource methods are coroutines.

    Example
    -------
    ::

        import asyncio
        from polyvia import AsyncPolyvia

        async def main():
            async with AsyncPolyvia(api_key="poly_...") as client:
                result = await client.ingest.file("report.pdf")
                await client.ingest.wait(result.task_id)
                answer = await client.query("Key findings?")
                print(answer.answer)

        asyncio.run(main())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://app.polyvia.ai",
        timeout: float = 60.0,
    ) -> None:
        import os

        resolved_key = api_key or os.environ.get("POLYVIA_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "api_key is required. Pass it explicitly or set the POLYVIA_API_KEY env var."
            )

        self._transport = AsyncTransport(resolved_key, base_url=base_url, timeout=timeout)
        self._api_key = resolved_key
        self._base_url = base_url

        self.ingest = AsyncIngestResource(self._transport)
        self.documents = AsyncDocumentsResource(self._transport)
        self.groups = AsyncGroupsResource(self._transport)

    async def query(
        self,
        question: str,
        *,
        document_id: Optional[str] = None,
        group: Optional[GroupRef] = None,
        group_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ) -> QueryResult:
        if group is not None and group_id is None and group_ids is None:
            group_id = await _resolve_group_id_async(self._transport, group, None, create=False)
        body: Dict[str, Any] = {"query": question}
        if document_id:
            body["document_id"] = document_id
        elif group_ids:
            body["group_ids"] = group_ids
        elif group_id:
            body["group_id"] = group_id
        raw = await self._transport.post("/api/v1/query", json=body)
        return QueryResult(**raw)

    async def usage(self) -> Usage:
        raw = await self._transport.get("/api/v1/usage")
        return Usage(**raw)

    async def rate_limits(self) -> RateLimits:
        raw = await self._transport.get("/api/v1/rate-limits")
        return RateLimits(**raw)

    @property
    def mcp(self) -> MCPConfig:
        return MCPConfig(
            url=f"{self._base_url}/mcp",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> "AsyncPolyvia":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
