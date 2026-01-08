# Actor State Manager Tests

These tests verify that `HierarchicalActor` correctly orchestrates state manager primitives (contacts, tasks, knowledge, etc.) by generating executable Python plans.

## Overview

The Actor state manager tests ensure that `HierarchicalActor` can:
- Generate Python plans that correctly call state manager primitives
- Compose multiple manager operations in a single plan
- Select and reuse memoized functions from `FunctionManager` via semantic search
- Execute plans that interact with both simulated and real state managers

Actor tests verify code-first plan generation and execution, reflecting the Actor's role as the central intelligence of the Unity system.

## Test Structure

| Directory | Purpose |
|-----------|---------|
| `test_simulated/` | Tests using simulated state managers (fast, deterministic) |
| `test_real/` | Tests using real state managers (integration-level) |
| `conftest.py` | Shared fixtures and configuration |
| `utils.py` | Assertion helpers and test utilities |

## Testing Patterns

The test suite uses two complementary patterns:

| Pattern | Description | When to Use | Example |
|---------|-------------|-------------|---------|
| **On-the-fly Planning** | Actor generates Python plans dynamically from natural language goals | Testing Actor's ability to reason about and compose primitives | `test_contacts/test_ask.py` |
| **Memoized Functions** | Actor selects pre-seeded functions from FunctionManager via semantic search | Testing Actor's ability to reuse existing skills | `test_contacts/test_ask_memoized.py` |

### On-the-fly Planning

In these tests, the Actor receives a natural language goal and must generate a complete Python plan from scratch. This tests the Actor's reasoning and code generation capabilities.

**Example**: `test_contacts/test_ask.py`
- Actor receives: "Which of our contacts prefers to be contacted by phone?"
- Actor generates: Python plan calling `primitives.contacts.ask(...)`
- Verification: Checks `handle.plan_source_code` contains the expected primitive calls

### Memoized Functions

In these tests, a function is pre-seeded into `FunctionManager` (via `FunctionManager.add()`), and the Actor must select it via semantic search when `can_compose=True`. This tests the Actor's ability to discover and reuse existing skills.

**Example**: `test_contacts/test_ask_memoized.py`
- Pre-seed: Function that calls `primitives.contacts.ask(...)`
- Actor receives: Similar natural language goal
- Actor selects: Pre-seeded function via semantic search
- Verification: Checks `handle.action_log` for "Generating plan from goal..." and verifies function was called

## Key Fixtures

### `configure_simulated_managers` (autouse)

**Location**: `conftest.py`

Forces simulated implementations for all tests under `test_simulated/`. This fixture:
- Sets `UNITY_*_IMPL=simulated` environment variables
- Updates `SETTINGS` singleton to use simulated managers
- Clears `ManagerRegistry` to ensure fresh instances

**Usage**: Automatic—no explicit fixture request needed for tests in `test_simulated/`

### `configure_real_managers` (autouse)

**Location**: `conftest.py`

Forces real implementations for all tests under `test_real/`. This fixture:
- Sets `UNITY_*_IMPL=real` environment variables
- Stubs network access to prevent initialization errors
- Updates `SETTINGS` singleton to use real managers
- Clears `ManagerRegistry` to ensure fresh instances

**Usage**: Automatic—no explicit fixture request needed for tests in `test_real/`

### `mock_verification`

**Location**: `conftest.py`

Monkeypatches `HierarchicalActor._check_state_against_goal` to always return success, bypassing verification LLM calls while preserving plan generation and execution.

**Why**: These tests verify plan generation and execution, not verification logic. Verification is tested separately in `test_verification_bypass.py`.

**Usage**: Request as a fixture parameter:
```python
async def test_something(mock_verification):
    # Verification is bypassed automatically
```

## Assertion Helpers

All helpers are in `utils.py`:

### `assert_tool_called(handle, tool_name)`

Verifies a specific primitive was called by checking `handle.idempotency_cache`.

**Example**:
```python
assert_tool_called(handle, "primitives.contacts.ask")
```

### `assert_memoized_function_used(handle, function_name=None)`

Verifies Actor used LLM-generated plan (can_compose=True path) by checking `handle.action_log` for "Generating plan from goal...". Optionally verifies a specific function was called in the plan.

**Example**:
```python
assert_memoized_function_used(handle, "find_contact_email")
```

### `get_state_manager_tools(handle)`

Extracts all state manager tool calls from idempotency cache. Returns a list of tool names (e.g., `["primitives.contacts.ask", "primitives.tasks.update"]`).

**Example**:
```python
tools = get_state_manager_tools(handle)
assert "primitives.contacts.ask" in tools
```

### `make_actor(impl, can_compose, can_store)`

Context manager for creating `HierarchicalActor` with browser mocks. Automatically mocks browser primitives to prevent Keychain prompts and network access.

**Example**:
```python
async with make_actor(impl="simulated", can_compose=True) as actor:
    handle = await actor.act("goal", persist=False)
    result = await handle.result()
```

**Note**: The `impl` parameter is for documentation/assertion purposes only. Actual implementation selection is controlled by autouse fixtures based on test path.

## Test Coverage by Manager

| Manager | Operations | Simulated Tests | Real Tests |
|---------|-----------|-----------------|------------|
| Contacts | ask, update, ask_and_update | ✅ (6 tests) | ✅ |
| Tasks | ask, update, execute, ask_and_update | ✅ (6 tests) | ✅ |
| Knowledge | ask, update, ask_and_update | ✅ (6 tests) | ✅ |
| Transcripts | ask | ✅ (2 tests) | ✅ |
| Files | ask, organize, ask_and_organize | ✅ (6 tests) | ❌ (not yet ported) |
| Guidance | ask, update | ✅ (4 tests) | ✅ |
| WebSearch | ask | ✅ (2 tests) | ✅ |
| Session | (cross-manager) | ❌ | ✅ |

**Test counts**: Each operation typically has both on-the-fly and memoized variants, plus combined operations (e.g., `ask_and_update`).

## Test Organization Notes

Actor tests focus on code-first plan generation and execution. Some infrastructure-level tests (like handle serialization, pause/resume behavior, and steering passthrough) are tested at the Actor level rather than state manager level, reflecting Actor's role as the central orchestrator.

## Running the Tests

```bash
# Run all Actor state manager tests
tests/parallel_run.sh tests/test_actor/test_state_managers/

# Run only simulated tests (fast)
tests/parallel_run.sh tests/test_actor/test_state_managers/test_simulated/

# Run only real tests (integration)
tests/parallel_run.sh tests/test_actor/test_state_managers/test_real/

# Run specific manager tests
tests/parallel_run.sh tests/test_actor/test_state_managers/test_simulated/test_contacts/

# Run with fresh LLM calls (no cache)
tests/parallel_run.sh --env UNIFY_CACHE=false tests/test_actor/test_state_managers/
```

## Debugging Tips

### Inspect Plan Generation

Check `handle.plan_source_code` to see the generated Python plan:
```python
print(handle.plan_source_code)
```

### Inspect Action Log

Check `handle.action_log` for Actor's reasoning steps:
```python
for entry in handle.action_log:
    print(entry)
```

### Verify Tool Calls

Use `handle.idempotency_cache` to see which primitives were called:
```python
from tests.test_actor.test_state_managers.utils import get_state_manager_tools
tools = get_state_manager_tools(handle)
print(tools)
```

### Memoized Function Tests

For memoized tests, verify "Generating plan from goal..." appears in action log:
```python
log_text = "\n".join(handle.action_log)
assert "Generating plan from goal..." in log_text
```

### Verification Bypass

Remember that verification is bypassed via `mock_verification` fixture. Failures indicate plan generation or execution issues, not verification problems. If you need to test verification, see `test_verification_bypass.py`.

## Test Design Notes

- **All tests marked with `pytestmark = pytest.mark.eval`** as they involve real LLM calls for plan generation
- Tests use idempotency cache and action log for verification, reflecting Actor's code-first execution model
- Both simulated and real manager implementations are tested to ensure Actor works correctly across different environments
