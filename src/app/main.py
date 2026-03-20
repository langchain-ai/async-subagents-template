"""Async subagents graphs for LangSmith deployment."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from deepagents import AsyncSubAgent, create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import tool
from langgraph._internal._constants import CONF
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph_sdk import get_client

DEFAULT_MODEL = os.getenv("ASYNC_SUBAGENTS_MODEL", "anthropic:claude-sonnet-4-6")
logger = logging.getLogger(__name__)

_http_client = httpx.AsyncClient(
    headers={"User-Agent": "async-subagents-template/0.1"},
    timeout=10,
    follow_redirects=True,
)

SYSTEM_PROMPT = """
You are an async subagent supervisor.

Workflow:
1. Write and maintain a todo list for non-trivial requests.
2. Delegate focused fact-finding to subagents when helpful.
3. Store intermediate drafts in files when the task is long.
4. Before finalizing, critique your work for risks, gaps, and missing constraints.
5. Return concise, actionable output.

- Prefer concrete evidence over assumptions.
- State unresolved uncertainty explicitly.
- Keep output compact unless the user asks for depth.
""".strip()

RESEARCHER_SYSTEM_PROMPT = """
You are a focused researcher.

- Gather evidence using available tools.
- List assumptions.
- Report contradictions clearly.
- Output should be concise and source-grounded.
""".strip()

CRITIC_SYSTEM_PROMPT = """
You are a critical reviewer.

- Find weak logic and untested assumptions.
- Identify missing constraints and edge cases.
- Suggest specific improvements.
- Keep feedback concise and actionable.
""".strip()


class CompletionNotifierMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Notifier."""
        config = get_config()
        configurable = config.get(CONF, {})
        parent_thread_id = configurable.get("parent_thread_id")
        parent_assistant_id = configurable.get("parent_assistant_id")
        subagent_name = configurable.get("subagent_name") or "general-purpose"
        parent_url = configurable.get("parent_url")
        parent_headers = configurable.get("parent_headers")

        if not parent_thread_id or not parent_assistant_id:
            return None

        messages = state.get("messages", [])
        last_msg = ""
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                content = last.content
                last_msg = content if isinstance(content, str) else str(content)
            elif isinstance(last, dict):
                content = last.get("content", "")
                last_msg = content if isinstance(content, str) else str(content)

        summary = last_msg[:500] if last_msg else "(completed)"
        notification = f"[Async subagent '{subagent_name}' has completed] Result: {summary}"

        try:
            client = get_client(url=parent_url, headers=parent_headers)
            await client.runs.create(
                thread_id=parent_thread_id,
                assistant_id=parent_assistant_id,
                input={"messages": [{"role": "user", "content": notification}]},
            )
            logger.info(
                "Notified parent thread %s that subagent '%s' completed",
                parent_thread_id,
                subagent_name,
            )
        except Exception:
            logger.warning(
                "Failed to notify parent thread %s",
                parent_thread_id,
                exc_info=True,
            )
        return None


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


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
    middleware=[CompletionNotifierMiddleware()],
    interrupt_on={
        "execute": True,
        "write_file": True,
    },
    name="researcher",
)


ASYNC_SUBAGENTS: list[AsyncSubAgent] = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        "graph_id": "researcher",
    },
]

main_agent = create_deep_agent(
    model=DEFAULT_MODEL,
    tools=[utc_now, web_fetch],
    system_prompt=SYSTEM_PROMPT,
    subagents=ASYNC_SUBAGENTS,
    interrupt_on={
        "execute": True,
        "write_file": True,
    },
    name="main_agent",
)
