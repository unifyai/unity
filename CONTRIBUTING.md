# Contributing to Unity

## Getting started

Unity depends on two sibling repositories. Clone all three as siblings:

```bash
git clone https://github.com/unifyai/unity.git
git clone https://github.com/unifyai/unify.git
git clone https://github.com/unifyai/unillm.git

cd unity
pip install uv && uv sync --all-groups
```

This installs Unity and its sibling dependencies (linked via `[tool.uv.sources]` in `pyproject.toml`).

## Running tests

Tests use real LLM calls with cached responses. After the first run, cached responses replay instantly.

```bash
source .venv/bin/activate

# Run all tests
tests/parallel_run.sh tests/

# Run a specific module
tests/parallel_run.sh tests/contact_manager/

# Run a specific test
tests/parallel_run.sh tests/contact_manager/test_ask.py::test_name
```

Tests require a running backend (Orchestra). The test runner starts a local instance automatically via Docker (requires Docker Desktop). See [tests/README.md](tests/README.md) for the full testing philosophy.

## Code style

We use `black` for formatting and `autoflake` for unused import removal. Pre-commit hooks run automatically on commit:

```bash
.venv/bin/python -m pre_commit run --all-files
```

## CI on forks

The full test suite requires org-level secrets (API keys, backend access). Fork PRs run lint checks only. A maintainer will trigger the full test suite on your PR after review.

## Pull requests

- Open PRs against the `staging` branch
- Keep PRs focused — one concern per PR
- Tests should pass locally before opening a PR
- We don't require backward compatibility (see the project's aggressive refactoring philosophy)

## Design principles

- **No regex routing.** If the system handles something wrong, fix the prompt or tool docstring, not a hardcoded rule.
- **No defensive coding.** Don't wrap things in try/except unless you're handling a specific, recoverable error. Fail loud.
- **English as API.** Managers communicate through natural-language interfaces. The Actor orchestrates through English-language primitives.
- **Real LLMs in tests.** We never mock the LLM client. Responses are cached for speed, not faked.

## Questions?

Open a [Discussion](https://github.com/unifyai/unity/discussions) or join our [Discord](https://discord.com/invite/sXyFF8tDtm).
