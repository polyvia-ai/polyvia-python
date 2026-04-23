"""
Polyvia SDK — OpenAI Responses API + remote MCP
================================================

OpenAI connects directly to the Polyvia MCP server and calls tools
automatically — no manual tool-dispatch loop needed.

Requirements:
    pip install polyvia openai

Run:
    POLYVIA_API_KEY=poly_... OPENAI_API_KEY=sk-... python examples/openai_responses_mcp.py
"""

import os

from openai import OpenAI
from polyvia import Polyvia

polyvia = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = oai.responses.create(
    model="gpt-4o",
    tools=[polyvia.mcp.to_openai_responses_tool()],
    input="What documents do I have, and what are the key themes across them?",
)

print(response.output_text)
