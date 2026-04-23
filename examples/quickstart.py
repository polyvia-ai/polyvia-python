"""
Polyvia SDK — Quickstart
========================

Covers: ingest → wait → query → groups → usage & rate limits.

Run:
    POLYVIA_API_KEY=poly_... python examples/quickstart.py
"""

import os
from polyvia import Polyvia

client = Polyvia(api_key=os.environ["POLYVIA_API_KEY"])

# ── 1. Create a group ──────────────────────────────────────────────────────────
print("Creating group...")
group = client.groups.create("Q4 Reports")
group_id = group["group_id"]
print(f"  group_id = {group_id}")

# ── 2. Ingest a single file ────────────────────────────────────────────────────
print("\nIngesting report.pdf...")
result = client.ingest.file("report.pdf", name="Q4 2024 Report", group_id=group_id)
print(f"  document_id = {result.document_id}")
print(f"  task_id     = {result.task_id}")

# ── 3. Wait for parsing to complete ───────────────────────────────────────────
print("\nWaiting for parsing...")
status = client.ingest.wait(result.task_id, poll_interval=5)
print(f"  status = {status.status}")

# ── 4. Query the document ──────────────────────────────────────────────────────
print("\nQuerying document...")
answer = client.query("What are the key financial highlights?", document_id=result.document_id)
print(f"  {answer.answer}")

# ── 5. Query across the whole group ───────────────────────────────────────────
print("\nQuerying group...")
answer = client.query("Summarise the Q4 findings.", group_id=group_id)
print(f"  {answer.answer}")

# ── 6. List documents in the group ────────────────────────────────────────────
print("\nDocuments in group:")
docs = client.documents.list(group_id=group_id, status="completed")
for doc in docs:
    print(f"  [{doc.status}] {doc.title}  ({doc.id})")

# ── 7. Usage & rate limits ────────────────────────────────────────────────────
print("\nUsage this month:")
usage = client.usage()
u = usage.usage
print(f"  requests : {u.requests.period} / {u.requests.total} all-time")
print(f"  ingests  : {u.ingests.period}  / {u.ingests.total} all-time")
print(f"  queries  : {u.queries.period}  / {u.queries.total} all-time")
print(f"  stored   : {u.documents_stored} documents")

limits = client.rate_limits()
print("\nRate limits:")
print(f"  {limits.current['requests_this_minute']} / {limits.limits['requests_per_minute']} req/min")
print(f"  {limits.current['requests_this_month']} / {limits.limits['requests_per_month']} req/month")

# ── 8. MCP connection snippet ─────────────────────────────────────────────────
print("\nClaude Desktop MCP config:")
client.mcp.print_claude_desktop_snippet()
