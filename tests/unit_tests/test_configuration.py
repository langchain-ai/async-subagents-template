from langgraph.pregel import Pregel

from app.main import main_agent


def test_async_subagents_graph_compiles() -> None:
    assert isinstance(main_agent, Pregel)
