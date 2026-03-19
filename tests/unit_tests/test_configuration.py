from langgraph.pregel import Pregel

from app.main import supervisor_agent


def test_async_subagents_graph_compiles() -> None:
    assert isinstance(supervisor_agent, Pregel)
