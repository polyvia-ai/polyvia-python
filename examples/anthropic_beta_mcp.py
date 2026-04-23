"""
Polyvia SDK — Anthropic beta MCP client
========================================

Anthropic connects directly to the Polyvia MCP server via the
beta MCP client feature — no manual tool-dispatch loop needed.

Requirements:
    pip install polyvia anthropic

Run:
    POLYVIA_API_KEY=poly_... ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_beta_mcp.py
"""

import os

from anthropic import Anthropic
from polyvia import Polyvia

polyvia = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])
ant     = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = ant.beta.messages.create(
    model="claude-opus-4-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": "What documents do I have, and what are the key themes across them?"}],
    mcp_servers=[polyvia.mcp.to_anthropic_mcp_server()],
    betas=["mcp-client-2025-04-04"],
)

print(response.content[0].text)
