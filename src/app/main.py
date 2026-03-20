"""Async subagents graphs for LangSmith deployment."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx
from deepagents import AsyncSubAgent, create_deep_agent
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ASYNC_SUBAGENTS_MODEL", "anthropic:claude-sonnet-4-6")

_http_client = httpx.AsyncClient(
    headers={"User-Agent": "async-subagents-template/0.1"},
    timeout=10,
    follow_redirects=True,
)


RESEARCHER_SYSTEM_PROMPT = """
You are a focused researcher.

- Gather evidence using available tools.
- List assumptions.
- Report contradictions clearly.
- Output should be concise and source-grounded.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


# !!! warning "SSRF Risk for Self-Hosted Deployments"
#     This tool is included as an example since it doesn't require secrets.
#     It's safe for LangSmith deployments, but for self-hosted deployments,
#     you should either rewrite this tool to prevent SSRF attacks (e.g.,
#     block private IP ranges, validate DNS resolution) or configure
#     network-level controls to restrict inbound access.
@tool
async def web_fetch(url: str) -> str:
    """Fetch a URL and return its body as text.

    Args:
        url: The URL to fetch (must be http or https).
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        resp = await _http_client.get(url)
        resp.raise_for_status()
        return resp.text[:50_000]
    except httpx.HTTPError as exc:
        return f"Error fetching {url}: {exc}"


researcher_graph = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, web_fetch],
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    name="researcher",
)


ASYNC_SUBAGENTS: list[AsyncSubAgent] = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        # graph_id must match the key in langgraph.json's "graphs" object,
        # which tells LangGraph which graph to invoke for this subagent.
        "graph_id": "researcher",
    },
]

supervisor_agent = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, web_fetch],
    subagents=ASYNC_SUBAGENTS,
    interrupt_on={
        "execute": True,
        "write_file": True,
    },
    name="supervisor_agent",
)
