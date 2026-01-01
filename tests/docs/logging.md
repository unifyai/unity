# Test Logging & Data

This document covers the logging infrastructure for tests: local log files, remote telemetry, and how to find and analyze test data.

---

## Log Directory Overview

All logs are organized under `logs/` with five main subdirectories:

| Directory | Purpose | Structure | Control |
|-----------|---------|-----------|---------|
| `logs/pytest/` | Test output (stdout/stderr) | One `.txt` per test | Test-only |
| `logs/unity/` | Unity LOGGER output (async tool loop, managers) | `unity.log` per session | `UNITY_LOG` + `UNITY_LOG_DIR` |
| `logs/llm/` | Raw LLM request/response traces | `.txt` files per request | `UNILLM_LOG` + `UNILLM_LOG_DIR` |
| `logs/unify/` | Unify SDK HTTP traces | JSON files per request | `UNIFY_LOG` + `UNIFY_LOG_DIR` |
| `logs/orchestra/` | Orchestra API traces | Per-request JSON with OpenTelemetry spans | `ORCHESTRA_LOG_DIR` |

**Cross-correlation:** When `UNITY_TEST_TRACING=true` (default), each test logs a `TRACE_ID` that links pytest output → Unity logs → Unify HTTP logs → Orchestra traces. See [Correlating Logs Across Systems](#correlating-logs-across-systems).

---

## Pytest Logs (`logs/pytest/`)

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

## Unity Logs (`logs/unity/`)

Unity LOGGER output captures async tool loop events, manager operations, tool scheduling, and other runtime logging. This is production-ready logging controlled by the `UNITY_LOG_DIR` environment variable.

```
logs/unity/
├── 2025-12-05T09-15-22_unity_dev_ttys042/
│   └── unity.log        # All LOGGER output for this session
└── ...
```

**Sample content:**
```
2026-01-01 14:26:46,175    INFO 🧑‍💻 [ContactManager.ask(ca3e)] User Message: What is Alice Smith's...
2026-01-01 14:26:46,535    INFO 🔄 [ContactManager.ask(ca3e)] LLM thinking…
2026-01-01 14:26:50,664    INFO 🛠️  ToolCall Scheduled [ContactManager.ask(ca3e)] search_contacts
```

**Production usage:** Set `UNITY_LOG_DIR=/path/to/logs` to enable file logging in production. If not set, logs go to console only.

---

## Unify Logs (`logs/unify/`)

Unify SDK HTTP traces capture all requests to the Orchestra API with OpenTelemetry trace correlation.

```
logs/unify/
├── 2025-12-05T09-15-22_unity_dev_ttys042/
│   ├── 14-26-27.611_POST_project-UnityTests-contexts_210ms_200_no-trace.json
│   ├── 14-26-46.175_GET_logs_331ms_200_f124f0d3.json   # trace_id suffix!
│   └── ...
└── ...
```

**Filename format:** `{timestamp}_{METHOD}_{route}_{duration}ms_{status}_{trace_id}.json`

The `trace_id` suffix (last 8 chars) enables correlation with pytest `[TRACE] TRACE_ID=...` output and Orchestra traces.

**Environment variables:**
- `UNIFY_LOG=true` (default) - Enable logging (console + file if directory set)
- `UNIFY_LOG_DIR=/path/to/logs` - Directory for file logging

---

## LLM Logs (`logs/llm/`)

LLM request/response traces are handled directly by the `unillm` package. These contain the raw I/O for each LLM call made during tests.

```
logs/llm/
├── 2025-12-05T09-15-22_unity_dev_ttys042/
│   └── *.txt  (e.g., 142536_123456789_hit.txt, 142537_987654321_miss.txt)
└── ...
```

**Log file format:**
- `{HHMMSS}_{nanoseconds}_pending.txt` - Written immediately when LLM call starts
- `{HHMMSS}_{nanoseconds}_hit.txt` - Finalized after call completes (cache hit)
- `{HHMMSS}_{nanoseconds}_miss.txt` - Finalized after call completes (cache miss)

If an LLM call hangs or crashes, the `_pending.txt` file remains as evidence.

**Environment variables:**
- `UNILLM_LOG=true` (default) - Enable logging (console + file if directory set)
- `UNILLM_LOG_DIR=/path/to/logs` - Directory for file logging

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

---

## Orchestra Logs (`logs/orchestra/`)

Orchestra logs capture detailed API request traces using OpenTelemetry. Each Orchestra session (started via `orchestra.sh`) creates a timestamped directory.

### Directory Structure

```
logs/orchestra/
└── 2025-12-30T18-27-43/                      # Session (one per orchestra.sh start/restart)
    └── requests/                              # Per-request API traces
        ├── 2025-12-30T18-28-03.852_DELETE_project-name_81ms_5cc61e5f.json
        ├── 2025-12-30T18-28-03.934_GET_projects_20ms_8e6fb277.json
        ├── 2025-12-30T18-46-55.980_GET_projects_PENDING_7be454fc.json
        └── 2025-12-30T18-47-01.234_POST_logs_143ms_a1b2c3d4.json
```

### Trace File Naming

Each request generates a JSON file with a descriptive filename:

```
{datetime}_{METHOD}_{route}_{duration|PENDING}_{trace_id_short}.json
```

| Component | Example | Description |
|-----------|---------|-------------|
| `datetime` | `2025-12-30T18-28-03.852` | Request start time (millisecond precision) |
| `METHOD` | `GET`, `POST`, `DELETE` | HTTP method |
| `route` | `projects`, `project-name` | API route (path params replaced with placeholders) |
| `duration` | `81ms`, `PENDING` | Request duration (or `PENDING` while in-flight) |
| `trace_id_short` | `5cc61e5f` | Last 8 chars of the OpenTelemetry trace_id |

### Trace File Contents

Each JSON file contains:

```json
{
  "trace_id": "344797cc597872b6f9a1e8675cc61e5f",
  "status": "complete",
  "spans": [
    {
      "name": "GET /v0/projects",
      "span_id": "a1b2c3d4e5f6a7b8",
      "parent_span_id": null,
      "start_time": "2025-12-30T18:28:03.852Z",
      "end_time": "2025-12-30T18:28:03.933Z",
      "duration_ms": 81,
      "attributes": {
        "http.method": "GET",
        "http.route": "/v0/projects",
        "http.status_code": 200,
        "http.request.query_params": "{}",
        "http.request.body": "{...}"
      }
    },
    {
      "name": "SELECT projects",
      "span_id": "...",
      "parent_span_id": "a1b2c3d4e5f6a7b8",
      "attributes": { "db.statement": "SELECT ..." }
    }
  ]
}
```

**Key fields:**
- `trace_id`: Full 32-character OpenTelemetry trace ID (matches pytest's `TRACE_ID`)
- `status`: `"complete"` or `"in_progress"` (for long-running requests)
- `spans`: All OpenTelemetry spans for this request (HTTP, database, OpenAI calls, etc.)

### In-Progress Traces

For long-running requests, trace files are written incrementally:
1. File created immediately with `status: "in_progress"` and `PENDING` in filename
2. Updated periodically (every 500ms) as new spans complete
3. Renamed on completion with actual duration

This allows debugging long-running requests before they complete.

---

## Correlating Logs Across Systems

OpenTelemetry trace IDs link all log types together, enabling end-to-end debugging across the full stack.

### How It Works

1. **Pytest creates a trace span** for each test (via `conftest.py`'s `_trace_test` fixture)
2. **Unity LOGGER output** goes to `logs/unity/` with timestamps for correlation
3. **Unify SDK HTTP calls** include the trace ID in filenames (last 8 chars)
4. **Orchestra uses the same trace ID** and writes it to server-side trace files
5. **Trace IDs appear in all logs**, enabling correlation

### Log Correlation Chain

```
logs/pytest/         →  [TRACE] TRACE_ID=...7be454fc test=test_ask
    ↓ (same session)
logs/unity/          →  🧑‍💻 [ContactManager.ask(ca3e)] User Message: ...
    ↓ (same trace_id)
logs/unify/          →  14-26-46.175_GET_logs_331ms_200_7be454fc.json
    ↓ (same trace_id)
logs/orchestra/      →  2025-12-30T18-46-55.980_GET_projects_43ms_7be454fc.json
```

### Finding Correlated Traces

**Step 1: Find the trace_id in pytest output**

Each test logs its trace_id to stdout:
```
[TRACE] TRACE_ID=099b207f89222185695d25977be454fc test=test_create_contact
```

Or grep the log file:
```bash
grep "TRACE_ID=" logs/pytest/2025-12-30T18-30-00_unity_dev_ttys042/test_contact_manager-test_ask.txt
```

**Step 2: Find matching Unify HTTP traces**

The last 8 characters of the trace_id appear in the filename:
```bash
# trace_id=099b207f89222185695d25977be454fc → search for *7be454fc*
ls logs/unify/2025-12-30T18-30-00_unity_dev_ttys042/*7be454fc*
```

**Step 3: Find matching Orchestra trace files**

```bash
ls logs/orchestra/2025-12-30T18-27-43/requests/*7be454fc*
```

**Step 4: Read the trace files**

```bash
# Unify SDK request/response
cat logs/unify/2025-12-30T18-30-00_unity_dev_ttys042/*7be454fc*.json | jq .

# Orchestra server-side trace
cat logs/orchestra/2025-12-30T18-27-43/requests/*7be454fc*.json | jq .
```

### Example Workflow

```bash
# 1. Test fails - check pytest log for trace_id
grep TRACE_ID logs/pytest/2025-12-30T18-30-00_unity_dev/test_contact_manager-test_ask.txt
# Found: [TRACE] TRACE_ID=099b207f89222185695d25977be454fc test=test_ask

# 2. Check Unity LOGGER output for the session
cat logs/unity/2025-12-30T18-30-00_unity_dev/unity.log | grep "ContactManager"

# 3. Find Unify SDK HTTP calls with matching trace
ls logs/unify/2025-12-30T18-30-00_unity_dev/*7be454fc*

# 4. Find Orchestra server traces
ls logs/orchestra/*/requests/*7be454fc*

# 5. Inspect the full request chain
cat logs/orchestra/2025-12-30T18-27-43/requests/*7be454fc*.json | jq .
# See full request params, DB queries, response status, timing, etc.
```

### Disabling Trace Correlation

To disable OpenTelemetry tracing in tests:
```bash
parallel_run --env UNITY_TEST_TRACING=false tests/
```

This removes the `[TRACE] TRACE_ID=...` output and disables `traceparent` header injection.
