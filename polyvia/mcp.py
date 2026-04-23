"""MCP server connection helpers for the Polyvia hosted MCP endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

MCP_URL = "https://app.polyvia.ai/mcp"


@dataclass
class MCPConfig:
    """Connection details for the Polyvia MCP server.

    Pass to any MCP-compatible client (Claude Desktop, OpenAI Agents SDK,
    LangChain MCP adapters, etc.).

    Examples
    --------
    Connect from Claude Desktop::

        import json, pathlib
        cfg_path = pathlib.Path.home() / ".claude" / "claude_desktop_config.json"
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        config.setdefault("mcpServers", {})["polyvia"] = client.mcp.to_claude_desktop_config()
        cfg_path.write_text(json.dumps(config, indent=2))
    """

    url: str = MCP_URL
    headers: Dict[str, str] = field(default_factory=dict)

    # ── Claude Desktop ─────────────────────────────────────────

    def to_claude_desktop_config(self) -> Dict[str, Any]:
        """Return the JSON object for a ``mcpServers`` entry in
        ``~/.claude/claude_desktop_config.json``."""
        return {
            "type": "http",
            "url": self.url,
            "headers": self.headers,
        }

    # ── OpenAI Responses API (remote MCP tool) ────────────────

    def to_openai_responses_tool(
        self,
        *,
        server_label: str = "polyvia",
        require_approval: str = "never",
    ) -> Dict[str, Any]:
        """Return a tool entry for the OpenAI **Responses API** remote MCP support.

        Drop the result straight into the ``tools`` list of
        ``client.responses.create()``.  OpenAI connects to the Polyvia MCP
        server on your behalf — no manual tool-dispatch loop needed.

        Parameters
        ----------
        server_label:
            Label shown in the response for tool calls (default ``"polyvia"``).
        require_approval:
            ``"never"`` (default) lets OpenAI call tools automatically.
            Set to ``"always"`` to review each call first.

        Example
        -------
        ::

            from openai import OpenAI
            from polyvia import Polyvia

            polyvia = Polyvia(api_key="poly_...")
            oai     = OpenAI()

            response = oai.responses.create(
                model="gpt-4o",
                tools=[polyvia.mcp.to_openai_responses_tool()],
                input="What are my Q4 findings?",
            )
            print(response.output_text)
        """
        return {
            "type": "mcp",
            "server_label": server_label,
            "server_url": self.url,
            "headers": self.headers,
            "require_approval": require_approval,
        }

    # ── Anthropic beta MCP client ─────────────────────────────

    def to_anthropic_mcp_server(
        self,
        *,
        name: str = "polyvia",
    ) -> Dict[str, Any]:
        """Return an entry for the ``mcp_servers`` list in
        ``client.beta.messages.create()``.

        Parameters
        ----------
        name:
            Identifier for this server in the response (default ``"polyvia"``).

        Example
        -------
        ::

            from anthropic import Anthropic
            from polyvia import Polyvia

            polyvia = Polyvia(api_key="poly_...")
            ant     = Anthropic()

            response = ant.beta.messages.create(
                model="claude-opus-4-5",
                max_tokens=1000,
                messages=[{"role": "user", "content": "What are my Q4 findings?"}],
                mcp_servers=[polyvia.mcp.to_anthropic_mcp_server()],
                betas=["mcp-client-2025-04-04"],
            )
            print(response.content[0].text)
        """
        return {
            "type": "url",
            "url": self.url,
            "name": name,
            "headers": self.headers,
        }

    # ── OpenAI Agents SDK (MCPServerStreamableHTTP) ───────────

    def to_openai_mcp_server(self) -> Dict[str, Any]:
        """Return kwargs suitable for ``openai.agents.MCPServerStreamableHTTP``."""
        return {
            "url": self.url,
            "headers": self.headers,
        }

    # ── Generic / custom clients ──────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict representation."""
        return {"url": self.url, "headers": self.headers}

    def __repr__(self) -> str:
        masked = {k: (v[:10] + "…" if k.lower() == "authorization" else v)
                  for k, v in self.headers.items()}
        return f"MCPConfig(url={self.url!r}, headers={masked})"

    # ── Pretty-print for notebooks ─────────────────────────────

    def print_claude_desktop_snippet(self) -> None:
        """Print a copy-pasteable JSON snippet for Claude Desktop."""
        snippet = {"mcpServers": {"polyvia": self.to_claude_desktop_config()}}
        print(json.dumps(snippet, indent=2))
