"""
Polyvia SDK — OpenAI Agent Example
====================================

Uses the OpenAI ChatCompletion API with Polyvia tools so the model can
ingest documents, search the workspace, and answer questions autonomously.

Requirements:
    pip install polyvia openai

Run:
    POLYVIA_API_KEY=poly_... OPENAI_API_KEY=sk-... python examples/agent_openai.py
"""

import json
import os

import openai
from polyvia import Polyvia

# ── Setup ──────────────────────────────────────────────────────────────────────
polyvia = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])
oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

tools, call_tool = polyvia.tools.openai()

# ── ReAct agent loop ──────────────────────────────────────────────────────────
def run_agent(user_message: str, max_steps: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "You have access to the user's Polyvia document workspace. "
                "Use the polyvia_list_documents and polyvia_query tools to find "
                "and answer questions from their documents."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        response = oai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # No tool calls — final answer
        if not msg.tool_calls:
            return msg.content or ""

        # Execute each tool call and feed results back
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → {tc.function.name}({args})")
            result = call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return "Max steps reached."


if __name__ == "__main__":
    print("Ask a question about your Polyvia workspace.")
    print("Example: What are the key risks mentioned across my reports?\n")
    question = input("Question: ").strip() or "What documents do I have?"
    answer = run_agent(question)
    print(f"\nAnswer:\n{answer}")
