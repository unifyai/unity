# Test Logging & Data

This document covers the logging infrastructure for tests: local log files, remote telemetry, and how to find and analyze test data.

---

## Log Directory Structure

Test logs are organized into **datetime-prefixed directories** for natural time-based ordering and to prevent confusion when multiple agents or users run tests concurrently.

### Directory Layout

Directory names follow the format: `YYYY-MM-DDTHH-MM-SS_{socket_name}`
- The datetime prefix enables chronological sorting in filesystem listings
- The socket name identifies the terminal session for isolation

```
logs/pytest/
├── 2025-12-05T09-15-22_unity_dev_ttys042/    # Run at 09:15 from Terminal A
│   ├── test_contact-test_ask.txt
│   └── test_task-test_update.txt
├── 2025-12-05T10-30-45_unity_dev_ttys099/    # Run at 10:30 from Terminal B (agent)
│   └── test_contact-test_ask.txt
├── 2025-12-05T14-22-18_unity_dev_ttys042/    # Run at 14:22 from Terminal A (new run)
│   └── test_foo.txt
├── 2025-12-05T14-35-00_unity_pid12345/       # Non-interactive shell
│   └── ...
└── 2025-12-05T14-40-00_unity_dev_ttys042/    # Direct pytest from Terminal A (same ID as parallel_run)
    └── test_foo.txt

logs/llm/
├── 2025-12-05T09-15-22_unity_dev_ttys042/    # Same datetime-prefixed structure
│   └── {session_id}/
│       └── *.txt
└── ...
```

### Log File Naming

All log files use **semantic naming** within datetime-prefixed directories:

| Command | Log File |
|---------|----------|
| `pytest tests/test_contact_manager/test_ask.py` | `test_contact_manager-test_ask.txt` |
| `pytest tests/test_contact_manager/test_ask.py::test_foo` | `test_contact_manager-test_ask--test_foo.txt` |
| `pytest tests/test_contact_manager/` | `test_contact_manager.txt` |
| `pytest tests/` | `tests.txt` |
| `pytest` (no args) | `all.txt` |

### Finding Your Logs

At the end of every test run, a banner shows exactly where logs are:

```
========================================================================
📄 Test log: /Users/you/unity/logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/test_foo.txt
📁 This run's logs: /Users/you/unity/logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/
📂 All log directories:  /Users/you/unity/logs/pytest/*/
========================================================================
```

**Finding recent runs:** Directories are sorted chronologically, so recent runs appear at the bottom of `ls` output:
```bash
ls logs/pytest/              # Oldest first, newest last
ls -r logs/pytest/           # Newest first
```

**For agents:** Read the terminal output to find the exact log path. The directory name (e.g., `2025-12-05T14-30-22_unity_dev_ttys042`) is printed when tests start via `parallel_run`.

**For cross-run analysis:** Use glob patterns to search across all runs:
```bash
ls logs/pytest/*/            # List all run directories
ls logs/pytest/*/*.txt       # List all log files across all runs
```

---

## Test Data Logging (Remote)

Tests log rich telemetry to the Unify backend, enabling post-hoc analysis of test runs, LLM behavior, and performance. Data is organized into two layers: a **global summary context** and **per-test contexts**.

### Combined Context (Global Summary)

Every test logs a summary record to the shared `Combined` context within the `UnityTests` project. This provides a unified view across all tests in a session.

**Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `test_fpath` | `str` | Test path: `folder/file.py::test_name` |
| `tags` | `list` | Session-level tags (via `--tags` or `UNIFY_TEST_TAGS`) |
| `duration` | `float` | Wall-clock time in seconds |
| `llm_io` | `list` | Full LLM request/response logs (from `logs/llm/` files) |
| `settings` | `dict` | Complete settings snapshot (production + test-only) |

**Use cases:**

- **Duration analysis**: Compare runtimes across tests, identify slow tests, track performance regressions
- **Settings ablation**: Filter by settings values to compare behavior with different configurations
- **LLM debugging**: Inspect the full LLM I/O for any test without re-running it
- **Tagging experiments**: Use `--tags "experiment-A,gpt-4"` to label runs for later filtering

**Example query:** "Show all tests where `UNIFY_CACHE=false` and duration > 10s"

### Per-Test Contexts (State Manager Data)

Each test decorated with `@_handle_project` gets its own isolated context. Within this context, state managers store their domain data in sub-contexts.

**Context hierarchy example:**

```
tests/test_contact_manager/test_basic/test_create          # Root test context
├── Contacts                                                # ContactManager data
├── Transcripts                                             # TranscriptManager data
├── Knowledge                                               # KnowledgeManager data
├── Tasks                                                   # TaskScheduler data
├── Events/_callbacks/                                      # EventBus subscriptions
└── ...                                                     # Other manager contexts
```

**How it works:**

1. The `@_handle_project` decorator (in `tests/helpers.py`) creates a unique context path derived from the test's file path and function name:
   - `tests/test_contact_manager/test_basic.py::test_create` → `tests/test_contact_manager/test_basic/test_create`

2. Before the test runs, the decorator:
   - Sets this as the active Unify context
   - Clears the EventBus to ensure isolation
   - Records which LLM I/O files exist (to detect new ones during the test)

3. During the test, state managers automatically create their sub-contexts (e.g., `Contacts`, `Transcripts`) under the active context.

4. After the test completes:
   - Duration is calculated
   - New LLM I/O files are collected
   - A summary record is logged to `Combined`
   - Optionally, the test context is deleted (if `UNIFY_DELETE_CONTEXT_ON_EXIT=true`)

**Decorator usage:**

```python
from tests.helpers import _handle_project

@_handle_project
def test_create_contact():
    cm = ContactManager()
    cm._create_contact(first_name="Alice")
    # ... assertions
```

For async tests:

```python
@_handle_project
async def test_async_operation():
    result = await some_async_call()
    # ... assertions
```

---

## Scenario Fixtures (Shared Seed Data)

Some test suites use session-scoped fixtures to create shared seed data once, then reset to that state before each test. This is more efficient than recreating data for every test.

**Example from `test_contact_manager/conftest.py`:**

```python
@pytest_asyncio.fixture(scope="session")
async def contact_scenario(request):
    """Create seeded contacts once per session."""
    ctx = "tests/test_contact/Scenario"
    unify.set_context(ctx, relative=False)

    # Seed data (idempotent - skips if already exists)
    builder = await ScenarioBuilderContacts.create()

    # Commit the initial state for rollback
    unify.commit_context(ctx, commit_message="Initial seed data")

    return builder.cm, id_mapping

@pytest.fixture(scope="function")
def contact_manager_scenario(contact_scenario):
    """Rollback to clean state before each test."""
    cm, id_map = contact_scenario

    # Reset to committed state
    unify.rollback_context(ctx, commit_hash=initial_commit_hash)

    yield cm, id_map
```

**Key concepts:**

- **Session fixture** (`scope="session"`): Creates seed data once per pytest session
- **Commit/rollback**: Uses Unify's git-like versioning to snapshot and restore state
- **Function fixture** (`scope="function"`): Rolls back before each test for isolation

---

## Inspecting Logged Data

**Via Unify Dashboard:**

Browse to the `UnityTests` project and explore:
- `Combined` context for summary records
- Individual test contexts (e.g., `tests/test_contact_manager/test_basic/test_create`) for detailed state

**Via Python:**

```python
import unify

unify.activate("UnityTests")

# Query Combined context
logs = unify.get_logs(context="Combined")
for log in logs:
    print(f"{log['test_fpath']}: {log['duration']:.2f}s")

# Query a specific test's contacts
unify.set_context("tests/test_contact_manager/test_basic/test_create/Contacts")
contacts = unify.get_logs()
```

---

## Context Cleanup

By default, test contexts persist across runs (useful for debugging). To auto-delete:

```bash
# Delete context after each test
parallel_run --env UNIFY_DELETE_CONTEXT_ON_EXIT=true tests

# Or delete entire project before session (clean slate)
parallel_run --env UNIFY_TESTS_DELETE_PROJ_ON_START=true tests

# Or delete entire project after session
parallel_run --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```
