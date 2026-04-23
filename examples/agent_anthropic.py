"""
Polyvia SDK — Anthropic Agent Example
=======================================

Uses the Anthropic Messages API with Polyvia tools so Claude can
ingest documents, search the workspace, and answer questions autonomously.

Requirements:
    pip install polyvia anthropic

Run:
    POLYVIA_API_KEY=poly_... ANTHROPIC_API_KEY=sk-ant-... python examples/agent_anthropic.py
"""

import json
import os

import anthropic
from polyvia import Polyvia

# ── Setup ──────────────────────────────────────────────────────────────────────
polyvia = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])
ant = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

tools, call_tool = polyvia.tools.anthropic()

SYSTEM = (
    "You are a helpful research assistant with access to the user's Polyvia "
    "document workspace. Use the available tools to find documents and answer "
    "questions from their content. Always cite the source document when possible."
)

# ── ReAct agent loop ──────────────────────────────────────────────────────────
def run_agent(user_message: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = ant.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )

        # Collect any tool uses and the text content
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # No tool calls — final answer
        if not tool_uses:
            return " ".join(b.text for b in text_blocks)

        # Add assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and build the user (tool_result) turn
        tool_results = []
        for tu in tool_uses:
            print(f"  → {tu.name}({tu.input})")
            result = call_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached."


# ── MCP alternative ───────────────────────────────────────────────────────────
def show_mcp_config():
    """Show how to connect Claude Desktop directly to the Polyvia MCP server."""
    print("\nAlternatively, connect via MCP (no code needed):")
    polyvia.mcp.print_claude_desktop_snippet()


if __name__ == "__main__":
    print("Ask a question about your Polyvia workspace.")
    print("Example: Compare the Q3 and Q4 reports.\n")
    question = input("Question: ").strip() or "What documents do I have?"
    answer = run_agent(question)
    print(f"\nAnswer:\n{answer}")
    show_mcp_config()
