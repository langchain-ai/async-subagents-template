# Async Subagents Template

A minimal template for building and deploying an async subagent pattern with Deep Agents, LangGraph, and LangSmith deployment.

## What's in this repo

### Application code

- `src/app/main.py` defines the deployed graphs.
- `src/app/__init__.py` marks the Python package.

The main graph is a supervisor-style agent that demonstrates:
- async tools
- async subagents (`researcher` and `critic`)
- LangSmith/LangGraph deployment-ready graph exports
- human-in-the-loop interrupts for `execute` and `write_file`

### Configuration

- `langgraph.json` registers the deployed supervisor and subagent graphs.
- `pyproject.toml` defines dependencies and the `src/` package layout.
- `.env.example` shows the expected local environment variables.

### Tests

- `tests/unit_tests/test_configuration.py` checks that the main async subagent graph compiles.
- `tests/integration_tests/test_graph.py` runs a basic async smoke test.
- `tests/conftest.py` contains shared pytest setup.

Integration tests are skipped unless `ANTHROPIC_API_KEY` is set.

### Local workflow

- `Makefile` provides `install`, `dev`, `serve`, `test`, `integration-tests`, `lint`, and `format`.
- `uv.lock` pins the resolved dependency set.

## Quickstart

1. Sync dependencies and create a local environment file:

```bash
uv sync
cp .env.example .env
```

2. Start the local LangGraph dev server:

```bash
uv run langgraph dev
```

3. Run the unit tests:

```bash
make test
```

4. Deploy to LangSmith when ready:

```bash
uv run langgraph deploy
```

See the [LangSmith CLI docs](https://docs.langchain.com/langsmith/cli#deploy) for deployment details.

## Reference docs

- Deep Agents async subagents: https://docs.langchain.com/oss/python/deepagents/subagents
- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- LangSmith CLI: https://docs.langchain.com/langsmith/cli
