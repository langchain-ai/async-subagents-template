import os

import pytest

from app.main import supervisor_agent

pytestmark = pytest.mark.anyio

if not os.getenv("ANTHROPIC_API_KEY"):
    pytest.skip(
        "Set ANTHROPIC_API_KEY to run integration tests.", allow_module_level=True
    )


async def test_async_subagents_smoke() -> None:
    result = await supervisor_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello in one sentence.",
                }
            ]
        }
    )
    assert result is not None
    assert result.get("messages")
