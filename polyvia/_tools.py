"""
Agent tool definitions and executor for the Polyvia API.

These mirror the MCP server tools so they work with any LLM agent framework
that accepts JSON-schema tool definitions (OpenAI, Anthropic, LangChain, etc.).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from ._client import Polyvia

# ── Schema definitions ────────────────────────────────────────────────────────

_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "polyvia_ingest_document",
        "description": (
            "Upload a document to Polyvia for parsing and indexing. "
            "Returns a task_id — poll polyvia_check_ingestion_status until status='completed' "
            "before querying the document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file on disk.",
                },
                "name": {
                    "type": "string",
                    "description": "Display name in Polyvia. Defaults to the filename.",
                },
                "group_id": {
                    "type": "string",
                    "description": "Optional group to assign the document to on creation.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "polyvia_check_ingestion_status",
        "description": (
            "Check the processing status of a document ingestion task. "
            "Poll until status='completed' before querying."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "task_id returned by polyvia_ingest_document.",
                }
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "polyvia_list_groups",
        "description": "List all document groups in the Polyvia workspace.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "polyvia_create_group",
        "description": "Create a new document group.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Display name for the group."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "polyvia_list_documents",
        "description": (
            "List documents in the Polyvia workspace. "
            "Filter by status and/or group(s)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["uploading", "parsing", "completed", "failed"],
                    "description": "Filter by document status.",
                },
                "group_id": {"type": "string", "description": "Filter by a single group ID."},
                "group_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by multiple group IDs.",
                },
            },
        },
    },
    {
        "name": "polyvia_get_document",
        "description": "Get metadata and summary for a single document.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID."}
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "polyvia_update_document",
        "description": "Update a document's metadata (currently: group assignment).",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID to update."},
                "group_id": {
                    "type": ["string", "null"],
                    "description": "Group to assign; null to remove from group.",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "polyvia_delete_document",
        "description": "Permanently delete a document and its stored file.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID to delete."}
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "polyvia_delete_group",
        "description": (
            "Delete a document group. Fails if documents are still assigned unless "
            "delete_documents=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Group ID to delete."},
                "delete_documents": {
                    "type": "boolean",
                    "description": "Delete all group documents first (default false).",
                },
            },
            "required": ["group_id"],
        },
    },
    {
        "name": "polyvia_query",
        "description": (
            "Ask a natural-language question about documents. "
            "Scope to a single document (document_id), a group (group_id / group_ids), "
            "or omit to search all completed documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Your question (max 2000 chars)."},
                "document_id": {
                    "type": "string",
                    "description": "Restrict to one document (highest priority).",
                },
                "group_id": {"type": "string", "description": "Restrict to one group."},
                "group_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to multiple groups.",
                },
            },
            "required": ["query"],
        },
    },
]


def _make_executor(client: "Polyvia") -> Callable[[str, Dict[str, Any]], Any]:
    """Return a callable that dispatches tool calls to the REST API."""

    def execute(name: str, args: Dict[str, Any]) -> Any:
        if name == "polyvia_ingest_document":
            result = client.ingest.file(
                args["file_path"],
                name=args.get("name"),
                group_id=args.get("group_id"),
            )
            return result.model_dump()

        if name == "polyvia_check_ingestion_status":
            return client.ingest.status(args["task_id"]).model_dump()

        if name == "polyvia_list_groups":
            return [g.model_dump() for g in client.groups.list()]

        if name == "polyvia_create_group":
            return client.groups.create(args["name"])

        if name == "polyvia_list_documents":
            docs = client.documents.list(
                status=args.get("status"),
                group_id=args.get("group_id"),
                group_ids=args.get("group_ids"),
            )
            return [d.model_dump() for d in docs]

        if name == "polyvia_get_document":
            return client.documents.get(args["document_id"]).model_dump()

        if name == "polyvia_update_document":
            return client.documents.update(
                args["document_id"], group_id=args.get("group_id")
            )

        if name == "polyvia_delete_document":
            return client.documents.delete(args["document_id"])

        if name == "polyvia_delete_group":
            return client.groups.delete(
                args["group_id"],
                delete_documents=args.get("delete_documents", False),
            )

        if name == "polyvia_query":
            result = client.query(
                args["query"],
                document_id=args.get("document_id"),
                group_id=args.get("group_id"),
                group_ids=args.get("group_ids"),
            )
            return result.model_dump()

        raise ValueError(f"Unknown tool: {name!r}")

    return execute


# ── Format adapters ───────────────────────────────────────────────────────────

def as_openai_tools(
    client: "Polyvia",
) -> Tuple[List[Dict[str, Any]], Callable[[str, Dict[str, Any]], Any]]:
    """Return (tools, executor) in OpenAI ChatCompletion / Responses API format.

    Usage::

        tools, call = client.tools.openai()
        response = openai.chat.completions.create(model="gpt-4o", tools=tools, ...)
        for tc in response.choices[0].message.tool_calls or []:
            result = call(tc.function.name, json.loads(tc.function.arguments))
    """
    openai_tools = [
        {"type": "function", "function": {**t}} for t in _TOOLS
    ]
    return openai_tools, _make_executor(client)


def as_anthropic_tools(
    client: "Polyvia",
) -> Tuple[List[Dict[str, Any]], Callable[[str, Dict[str, Any]], Any]]:
    """Return (tools, executor) in Anthropic Messages API format.

    Usage::

        tools, call = client.tools.anthropic()
        response = anthropic.messages.create(model="claude-opus-4-5", tools=tools, ...)
        for block in response.content:
            if block.type == "tool_use":
                result = call(block.name, block.input)
    """
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in _TOOLS
    ]
    return anthropic_tools, _make_executor(client)


def as_langchain_tools(client: "Polyvia") -> List[Any]:
    """Return a list of LangChain ``BaseTool`` instances.

    Requires ``pip install polyvia[langchain]``.

    Usage::

        tools = client.tools.langchain()
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain support requires `pip install polyvia[langchain]`"
        ) from exc

    executor = _make_executor(client)
    lc_tools = []
    for t in _TOOLS:
        name = t["name"]
        schema = t["parameters"]

        def _run(executor=executor, name=name, **kwargs: Any) -> str:
            result = executor(name, kwargs)
            return json.dumps(result, default=str)

        lc_tools.append(
            StructuredTool.from_function(
                func=_run,
                name=name,
                description=t["description"],
                args_schema=None,  # LangChain will infer from signature
            )
        )
    return lc_tools
