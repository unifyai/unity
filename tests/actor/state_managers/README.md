# Actor State Manager Tests

These tests verify that `CodeActActor` correctly routes to state manager primitives (contacts, tasks, knowledge, etc.) via code execution.

## Overview

The Actor state manager tests ensure that `CodeActActor` can:
- Route to the correct state manager primitives via `execute_code`
- Compose multiple manager operations
- Select and reuse memoized functions from `FunctionManager` via semantic search
- Execute plans that interact with both simulated and real state managers

## Test Structure

| Directory | Purpose |
|-----------|---------|
| `test_simulated/` | Tests using simulated state managers (fast, deterministic) |
| `test_real/` | Tests using real state managers (integration-level) |
| `conftest.py` | Shared fixtures and configuration |
| `utils.py` | Assertion helpers and test utilities |

## Running the Tests

```bash
# Run all Actor state manager tests
tests/parallel_run.sh tests/actor/state_managers/

# Run only simulated tests (fast)
tests/parallel_run.sh tests/actor/state_managers/simulated/

# Run only real tests (integration)
tests/parallel_run.sh tests/actor/state_managers/real/

# Run specific manager tests
tests/parallel_run.sh tests/actor/state_managers/simulated/contacts/
```
