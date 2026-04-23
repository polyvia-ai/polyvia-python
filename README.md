# polyvia-python

Official Python SDK for the [Polyvia](https://polyvia.ai) Polyvia AI platform API and MCP server.

```python
from polyvia import Polyvia

client = Polyvia(api_key="poly_...")

# Ingest → wait → query
result = client.ingest.file("report.pdf", name="Q4 Report")
client.ingest.wait(result.task_id)
print(client.query("What are the key findings?").answer)
```

---

## Table of Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [REST API](#rest-api)
  - [Ingest](#ingest)
  - [Query](#query)
  - [Groups](#groups)
  - [Documents](#documents)
  - [Usage & Rate Limits](#usage--rate-limits)
- [MCP Server](#mcp-server)
  - [Anthropic beta MCP client](#anthropic-beta-mcp-client)
  - [OpenAI Responses API](#openai-responses-api)
  - [OpenAI Agents SDK](#openai-agents-sdk)
  - [Claude Desktop](#claude-desktop)
- [Agent Tools (programmatic)](#agent-tools-programmatic)
  - [OpenAI ChatCompletion](#openai-chatcompletion)
  - [Anthropic Messages API](#anthropic-messages-api)
  - [LangChain](#langchain)
- [Async Client](#async-client)
- [Error Handling](#error-handling)
- [Development](#development)

---

## Installation

```bash
pip install polyvia
```

LangChain agent support:

```bash
pip install "polyvia[langchain]"
```

Requires Python 3.9+.

---

## Authentication

Generate an API key at **[app.polyvia.ai → Settings → API](https://app.polyvia.ai/settings)**.
All keys start with `poly_`.

```python
# Pass explicitly
client = Polyvia(api_key="poly_...")

# Or set the environment variable and omit the argument
# export POLYVIA_API_KEY=poly_...
client = Polyvia()
```

---

## REST API

### Ingest

```python
# Single file — returns immediately with a task_id to poll
result = client.ingest.file("report.pdf", name="Q4 Report", group_id="g_...")
# IngestResult(document_id='...', task_id='...', status='pending')

# Multiple files in one request
batch = client.ingest.batch(
    ["q3.pdf", "q4.pdf"],
    names=["Q3 Report", "Q4 Report"],
    group_id="g_...",
)

# Check status
status = client.ingest.status(result.task_id)
# IngestionStatus(status='parsing', ...)

# Block until done — raises IngestionError on failure, IngestionTimeout on timeout
done = client.ingest.wait(result.task_id, poll_interval=5, timeout=300)
```

### Query

```python
# All completed documents
answer = client.query("What risks are mentioned across all reports?")

# Single document (fastest)
answer = client.query("Summarise section 3.", document_id="doc_...")

# Scoped to a group
answer = client.query("Key findings?", group_id="g_...")

# Scoped to multiple groups
answer = client.query("Compare results.", group_ids=["g_...", "g_..."])

print(answer.answer)
```

### Groups

```python
# Create
group = client.groups.create("Finance")
group_id = group["group_id"]

# List
for g in client.groups.list():
    print(g.name, g.id, g.color)

# Delete all documents in a group, then the group itself
client.groups.delete(group_id, delete_documents=True)

# Or separately
client.groups.delete_documents(group_id)   # wipe documents, keep group
client.groups.delete(group_id)             # remove empty group
```

### Documents

```python
# List — filter by status and/or group
docs = client.documents.list(status="completed", group_id="g_...")
docs = client.documents.list(group_ids=["g_...", "g_..."])

# Get one
doc = client.documents.get("doc_...")

# Move to a different group / remove from group
client.documents.update("doc_...", group_id="g_other")
client.documents.update("doc_...", group_id=None)

# Delete
client.documents.delete("doc_...")
```

### Usage & Rate Limits

```python
usage = client.usage()
print(usage.usage.requests.period)    # requests this calendar month
print(usage.usage.requests.total)     # all-time
print(usage.usage.documents_stored)  # live document count

limits = client.rate_limits()
print(limits.limits["requests_per_minute"])
print(limits.current["remaining_this_minute"])
print(limits.resets_at.month)         # ISO timestamp of next monthly reset
```

---

## MCP Server

Polyvia runs a hosted [Model Context Protocol](https://modelcontextprotocol.io) server at
`https://app.polyvia.ai/mcp`. Connect your AI client once and it can ingest, search,
and query documents without any manual tool-dispatch code.

`client.mcp` returns an `MCPConfig` object with a helper for every major client:

| Method | Use with |
|--------|----------|
| `to_anthropic_mcp_server()` | `ant.beta.messages.create(mcp_servers=[...])` |
| `to_openai_responses_tool()` | `oai.responses.create(tools=[...])` |
| `to_openai_mcp_server()` | OpenAI Agents SDK `MCPServerStreamableHTTP` |
| `to_claude_desktop_config()` | `~/.claude/claude_desktop_config.json` |

---

### Anthropic beta MCP client

```python
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
```

`to_anthropic_mcp_server()` produces:

```python
{
    "type": "url",
    "url": "https://app.polyvia.ai/mcp",
    "name": "polyvia",            # customise with name="my-docs"
    "headers": {"Authorization": "Bearer poly_..."},
}
```

---

### OpenAI Responses API

```python
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
```

`to_openai_responses_tool()` produces:

```python
{
    "type": "mcp",
    "server_label": "polyvia",        # customise with server_label="my-docs"
    "server_url": "https://app.polyvia.ai/mcp",
    "headers": {"Authorization": "Bearer poly_..."},
    "require_approval": "never",      # or "always" to confirm each call
}
```

---

### OpenAI Agents SDK

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHTTP
from polyvia import Polyvia

polyvia = Polyvia(api_key="poly_...")
cfg = polyvia.mcp.to_openai_mcp_server()

server = MCPServerStreamableHTTP(url=cfg["url"], headers=cfg["headers"])
agent  = Agent(name="Research", mcp_servers=[server])
result = Runner.run_sync(agent, "What do my Q4 reports say about revenue?")
print(result.final_output)
```

---

### Claude Desktop

```python
# Print a snippet to copy-paste into ~/.claude/claude_desktop_config.json
client.mcp.print_claude_desktop_snippet()
```

Or wire it up programmatically:

```python
import json, pathlib

cfg_path = pathlib.Path.home() / ".claude" / "claude_desktop_config.json"
config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
config.setdefault("mcpServers", {})["polyvia"] = client.mcp.to_claude_desktop_config()
cfg_path.write_text(json.dumps(config, indent=2))
print("Restart Claude Desktop to activate.")
```

`to_claude_desktop_config()` produces:

```json
{
  "type": "http",
  "url": "https://app.polyvia.ai/mcp",
  "headers": { "Authorization": "Bearer poly_..." }
}
```

---

## Agent Tools (programmatic)

If you'd rather manage the tool-dispatch loop yourself — or your framework
doesn't support remote MCP — use `client.tools` to get JSON-schema tool
definitions and an executor that calls the REST API directly.

All 10 Polyvia tools are included: ingest, status, list/get/update/delete
documents, list/create/delete groups, and query.

### OpenAI ChatCompletion

```python
import json
from openai import OpenAI
from polyvia import Polyvia

client = Polyvia(api_key="poly_...")
oai    = OpenAI()

tools, call = client.tools.openai()

response = oai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are my Q4 findings?"}],
    tools=tools,
)

for tc in response.choices[0].message.tool_calls or []:
    result = call(tc.function.name, json.loads(tc.function.arguments))
    print(result)
```

### Anthropic Messages API

```python
import anthropic
from polyvia import Polyvia

client = Polyvia(api_key="poly_...")
ant    = anthropic.Anthropic()

tools, call = client.tools.anthropic()

response = ant.messages.create(
    model="claude-opus-4-5",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Summarise my Finance documents."}],
    tools=tools,
)

for block in response.content:
    if block.type == "tool_use":
        result = call(block.name, block.input)
        print(result)
```

### LangChain

Requires `pip install "polyvia[langchain]"`.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from polyvia import Polyvia

client = Polyvia(api_key="poly_...")
tools  = client.tools.langchain()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to a document workspace."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent    = create_tool_calling_agent(ChatOpenAI(model="gpt-4o"), tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "What risks are mentioned in my reports?"})
```

---

## Async Client

Every method on `AsyncPolyvia` is a coroutine — same API surface as the sync client.

```python
import asyncio
from polyvia import AsyncPolyvia

async def main():
    async with AsyncPolyvia(api_key="poly_...") as client:
        result = await client.ingest.file("report.pdf")
        await client.ingest.wait(result.task_id)
        answer = await client.query("Key findings?")
        print(answer.answer)

asyncio.run(main())
```

---

## Error Handling

```python
from polyvia import (
    AuthenticationError,  # 401 — bad or missing API key
    ForbiddenError,        # 403 — document belongs to another user
    NotFoundError,         # 404 — document, group, or task not found
    RateLimitError,        # 429 — too many requests
    IngestionError,        # task finished with status='failed'
    IngestionTimeout,      # ingest.wait() exceeded its timeout
)

try:
    done = client.ingest.wait(task_id, timeout=60)
except IngestionError as e:
    print(f"Parsing failed: {e.error}")
except IngestionTimeout:
    print("Timed out — document may still be processing")
except RateLimitError:
    print("Rate limit hit — back off and retry")
except NotFoundError:
    print("Document or task not found")
except AuthenticationError:
    print("Invalid API key")
```

---

## Development

```bash
git clone https://github.com/polyvia-ai/polyvia-python
cd polyvia-python
pip install -e ".[dev]"
pytest
```

---

## License

MIT
