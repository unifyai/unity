# Test Logging & Data

This document covers the logging infrastructure for tests: local log files, remote telemetry, and how to find and analyze test data.

---

## Downloading CI Logs

When debugging CI failures, use the `download_ci_logs.sh` utility to download log artifacts from GitHub Actions.

### Quick Start

```bash
# From an artifact URL (copy from GitHub Actions UI)
./logs/download_ci_logs.sh "https://github.com/unifyai/unity/actions/runs/20882540406/artifacts/5086119156"

# From a run URL + pattern match
./logs/download_ci_logs.sh "https://github.com/unifyai/unity/actions/runs/20882540406" --pattern "function_manager"

# From just a run ID + pattern
./logs/download_ci_logs.sh 20882540406 --pattern "function_manager"

# List all artifacts for a run (discovery mode)
./logs/download_ci_logs.sh 20882540406 --list
```

### How to Get the URL

1. Go to the GitHub Actions run page (e.g., from a failed CI notification)
2. Click on the failed job to expand it
3. Scroll down to **Artifacts** section
4. Right-click the artifact name → "Copy link address"
5. Pass that URL to the script

Alternatively, just copy the run URL from your browser and use `--pattern` to filter.

### Output Location

Downloaded logs are extracted to `logs/ci/<run_id>/<artifact_name>/`:

```
logs/ci/
├── latest -> 20882540406/logs-function_manager/   # Symlink to most recent
├── 20882540406/
│   └── logs-function_manager/
│       ├── pytest/
│       │   └── 2026-01-10T18-32-52_unitypid7578/
│       │       ├── duration_summary.txt
│       │       ├── function_manager-test_basics-test_add_single_success.txt
│       │       └── ...
│       ├── unity/
│       ├── unify/
│       └── all/
└── 20881234567/
    └── logs-web_searcher/
        └── ...
```

### Script Options

| Option | Description |
|--------|-------------|
| `--pattern <pattern>` | Filter artifacts by name (case-insensitive substring match) |
| `--list` | List available artifacts without downloading |
| `--force` | Re-download even if artifact already exists locally |
| `--repo <owner/repo>` | Override repository (default: `unifyai/unity`) |
| `--help` | Show usage help |

### Features

- **Flexible input**: Accepts full artifact URLs, run URLs, or just run IDs
- **Smart waiting**: If the CI run is still in progress, waits for artifacts to become available (up to 10 minutes)
- **Progress display**: Shows download progress for large artifacts
- **Idempotent**: Skips re-downloading if artifact already exists (use `--force` to override)
- **Latest symlink**: `logs/ci/latest` always points to the most recently downloaded artifact

### Troubleshooting

**"Not authenticated with GitHub CLI"**
```bash
gh auth login
```

**"No artifacts matching pattern"**
```bash
# List all available artifacts first
./logs/download_ci_logs.sh <run_id> --list
```

**"Run still in progress"**
The script waits up to 10 minutes for artifacts to be uploaded. If the run takes longer, re-run the script after the CI job completes.

**Large artifacts (100MB+)**
Expect ~1-2 minutes per 100MB on typical connections. The script shows progress during download.

---

## Log Directory Overview

All logs are organized under `logs/` with seven main subdirectories:

| Directory | Purpose | Structure | Control |
|-----------|---------|-----------|---------|
| `logs/ci/` | **Downloaded CI artifacts** | `<run_id>/<artifact>/` | `download_ci_logs.sh` |
| `logs/pytest/` | Test output (stdout/stderr) | One `.txt` per test | Test-only |
| `logs/unity/` | Unity LOGGER output (async tool loop, managers) | `unity.log` per session | `UNITY_LOG` + `UNITY_LOG_DIR` |
| `logs/unillm/` | Raw LLM request/response traces | `.txt` files per request | `UNILLM_LOG_DIR` (+ `UNILLM_TERMINAL_LOG` for console) |
| `logs/unify/` | Unify SDK HTTP traces | JSON files per request | `UNIFY_LOG_DIR` (+ `UNIFY_TERMINAL_LOG` for console) |
| `logs/orchestra/` | Orchestra API traces | Per-request JSON with OpenTelemetry spans | `ORCHESTRA_LOG_DIR` |
| `logs/magnitude/` | **Magnitude agent debug** (screenshots, act traces, coordinates) | Per-act bundles with PNGs | `MAGNITUDE_LOG_DIR` + `MAGNITUDE_DEBUG` |
| `logs/all/` | **Cross-repo OTEL traces** | `{trace_id}.jsonl` per test | `*_OTEL` + `*_OTEL_LOG_DIR` |

**Cross-correlation:** When running tests via `parallel_run.sh`, OTEL tracing is enabled by default across all four repos (unity, unify, unillm, orchestra). All spans are written to `logs/all/`, enabling full-stack trace analysis. See [Cross-Repo OTEL Traces](#cross-repo-otel-traces-logsall).

**Worktree note:** When running from a git worktree (e.g., Cursor Background Agents), `logs/orchestra/` and `logs/all/` are symlinked to the main repo's directories. This ensures orchestra logs (from the shared server) and OTEL traces (for cross-repo correlation) all go to one place. Other log types (`pytest/`, `unity/`, `unify/`, `unillm/`) remain in the worktree.

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
| `pytest tests/contact_manager/test_ask.py` | `contact_manager-test_ask.txt` |
| `pytest tests/contact_manager/test_ask.py::test_foo` | `contact_manager-test_ask--test_foo.txt` |
| `pytest tests/contact_manager/` | `contact_manager.txt` |
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
2026-01-01 14:26:46,175    INFO ➡️ [ContactManager.ask(ca3e)] Request: What is Alice Smith's...
2026-01-01 14:26:46,535    INFO 🧠 [ContactManager.ask(ca3e)] LLM thinking…
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
- `UNIFY_TERMINAL_LOG=true` (default) - Enable terminal (console) output
- `UNIFY_LOG_DIR=/path/to/logs` - Directory for file-based traces (independent of terminal)

---

## Unillm Logs (`logs/unillm/`)

LLM request/response traces are handled directly by the `unillm` package. These contain the raw I/O for each LLM call made during tests.

```
logs/unillm/
├── 2025-12-05T09-15-22_unity_dev_ttys042/
│   └── *.txt  (e.g., 142536_123456789.cache_hit.txt, 142537_987654321.cache_miss.txt)
└── ...
```

**Log file format:**
- `{HHMMSS}_{nanoseconds}.cache_pending.txt` - Written immediately when LLM call starts
- `{HHMMSS}_{nanoseconds}.cache_hit.txt` - Finalized after call completes (cache hit)
- `{HHMMSS}_{nanoseconds}.cache_miss.txt` - Finalized after call completes (cache miss)

If an LLM call hangs or crashes, the `.cache_pending.txt` file remains as evidence.

**Environment variables:**
- `UNILLM_TERMINAL_LOG=true` (default) - Enable terminal (console) output
- `UNILLM_LOG_DIR=/path/to/logs` - Directory for file-based traces (independent of terminal)

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
| `llm_io` | `list` | Full LLM request/response logs (from `logs/unillm/` files) |
| `settings` | `dict` | Complete settings snapshot (production + test-only) |

**Use cases:**

- **Duration analysis**: Compare runtimes across tests, identify slow tests, track performance regressions
- **Settings ablation**: Filter by settings values to compare behavior with different configurations
- **LLM debugging**: Inspect the full LLM I/O for any test without re-running it
- **Tagging experiments**: Use `--tags "experiment-A,gpt-4"` to label runs for later filtering

**Example query:** "Show all tests where `UNILLM_CACHE=false` and duration > 10s"

### Per-Test Contexts (State Manager Data)

Each test decorated with `@_handle_project` gets its own isolated context. Within this context, state managers store their domain data in sub-contexts.

**Context hierarchy example:**

```
tests/contact_manager/test_basic/test_create          # Root test context
├── Contacts                                                # ContactManager data
├── Transcripts                                             # TranscriptManager data
├── Knowledge                                               # KnowledgeManager data
├── Tasks                                                   # TaskScheduler data
├── Events/_callbacks/                                      # EventBus subscriptions
└── ...                                                     # Other manager contexts
```

**How it works:**

1. The `@_handle_project` decorator (in `tests/helpers.py`) creates a unique context path derived from the test's file path and function name:
   - `tests/contact_manager/test_basic.py::test_create` → `tests/contact_manager/test_basic/test_create`

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

**Example from `contact_manager/conftest.py`:**

```python
@pytest_asyncio.fixture(scope="session")
async def contact_scenario(request):
    """Create seeded contacts once per session."""
    ctx = "tests/contact/Scenario"
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
- Individual test contexts (e.g., `tests/contact_manager/test_basic/test_create`) for detailed state

**Via Python:**

```python
import unify

unify.activate("UnityTests")

# Query Combined context
logs = unify.get_logs(context="Combined")
for log in logs:
    print(f"{log['test_fpath']}: {log['duration']:.2f}s")

# Query a specific test's contacts
unify.set_context("tests/contact_manager/test_basic/test_create/Contacts")
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

## Magnitude Logs (`logs/magnitude/`)

Magnitude logs capture exhaustive debug data from the browser/desktop automation agent (magnitude-core + agent-service). Controlled by two environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAGNITUDE_DEBUG` | `false` | Enables debug payload emission from agent-service. When `false`, zero overhead. |
| `MAGNITUDE_LOG_DIR` | `""` | Directory for persisting magnitude logs. If unset, debug payloads are silently dropped. |

In production: `MAGNITUDE_DEBUG=false` by default (zero overhead). Set `MAGNITUDE_DEBUG=true` on a specific job when investigating an issue. `MAGNITUDE_LOG_DIR=/var/log/magnitude` is pre-configured in both the main container and the desktop VM.

### Directory Structure

Each `desktop.act()` or `session.act()` call creates a timestamped bundle:

```
/var/log/magnitude/
├── magnitude.log                          # Text debug lines from magnitude-core (pino)
└── acts/
    └── 2026-03-02T01-25-58_drag_the_Jack_of_Spades/
        ├── act_trace.json                 # Full structured trace
        ├── planning_screenshot.png        # What the magnitude LLM saw (Playwright screenshot)
        ├── native_screenshot.png          # Actual desktop state (native OS capture, desktop mode only)
        └── post_action/
            ├── 001_mouse_click_512_384.png
            ├── 002_keyboard_type.png
            └── ...
```

### act_trace.json

Contains the complete act trajectory:

- `task`: the natural language instruction
- `lineage`: Unity call chain (e.g. `["CodeActActor.act(ab12)", "execute_code(9693)"]`)
- `sessionMode`: `desktop`, `web`, or `web-vm`
- `reasoning`: LLM's reasoning text
- `plannedActions`: full action list with coordinates/params
- `actionTraces`: per-action execution timing and errors
- `planningMs`, `totalMs`: timing breakdown

### Debugging with Magnitude Logs

The most common debugging scenario: desktop actions not registering.

1. Open the `acts/` directory for the failing act
2. Compare `planning_screenshot.png` (what the LLM saw) vs `native_screenshot.png` (actual desktop)
3. If they differ: noVNC rendering issue (stale frame, disconnected WebSocket, scaling mismatch)
4. If they match: check `act_trace.json` for coordinate precision (raw vs transformed coords)
5. Check `post_action/` screenshots to see whether each action had visible effect

### Where Logs Are Saved

Screenshots and traces are saved **locally by the agent-service process** — they never
travel over the WebSocket.  This avoids serializing large base64 images on the critical
path of `act()` calls.

| Mode | Agent-service runs in | Logs saved to | Synced by |
|------|----------------------|---------------|-----------|
| **web** | Main Unity pod | `/var/log/magnitude/` (same container) | `stream_logs.py` and `upload_pod_logs.py` |
| **desktop** | Desktop VM container | `/var/log/magnitude/` (VM container) | Access via VM supervisor logs or SSH |

### Production Streaming

For the main pod, magnitude logs are automatically included in:
- `stream_logs.py`: mirrored to `logs/prod_logs/<job>/magnitude/`
- `upload_pod_logs.py`: uploaded to GCS on pod shutdown

For the desktop VM, logs are on the VM filesystem at `/var/log/magnitude/`.

---

## Orchestra Logs (`logs/orchestra/`)

Orchestra logs capture detailed API request traces using OpenTelemetry. Each Orchestra session (started via `parallel_run.sh` or the `orchestra` shell function) creates a timestamped directory.

### Directory Structure

```
logs/orchestra/
└── 2025-12-30T18-27-43/                      # Session (one per orchestra start/restart)
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

## Cross-Repo OTEL Traces (`logs/all/`)

The `logs/all/` directory contains **unified OpenTelemetry traces** spanning all four repositories (unity, unify, unillm, orchestra). Each test creates a single `{trace_id}.jsonl` file containing spans from the entire call stack.

### How It Works

When tests run via `parallel_run.sh`, OTEL tracing is automatically enabled for all repos:

1. **Unity** creates the root span and TracerProvider
2. **Unillm** creates child spans for LLM calls (using Unity's provider)
3. **Unify** creates child spans for HTTP requests (using Unity's provider)
4. **Orchestra** receives the `traceparent` header and creates server-side spans

All repos write to the same directory, and since files are keyed by `trace_id`, each test's spans are aggregated into a single file.

### Directory Structure

```
logs/all/
└── 2026-01-01T14-30-22_unity_dev_ttys042/
    ├── 099b207f89222185695d25977be454fc.jsonl   # Full trace for test_ask
    ├── a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.jsonl   # Full trace for test_update
    └── ...
```

### Trace File Format (JSONL)

Each `.jsonl` file contains one JSON object per line, representing a span:

```json
{"service": "unity", "trace_id": "099b207f...", "span_id": "a1b2c3d4", "parent_span_id": null, "name": "ContactManager.ask", "start_time": "2026-01-01T14:30:22.123Z", "end_time": "2026-01-01T14:30:25.456Z", "duration_ms": 3333, "attributes": {...}}
{"service": "unillm", "trace_id": "099b207f...", "span_id": "e5f6g7h8", "parent_span_id": "a1b2c3d4", "name": "LLM gpt-5.2@openai", "start_time": "...", ...}
{"service": "unify", "trace_id": "099b207f...", "span_id": "i9j0k1l2", "parent_span_id": "e5f6g7h8", "name": "POST /v0/chat/completions", "start_time": "...", ...}
{"service": "orchestra", "trace_id": "099b207f...", "span_id": "m3n4o5p6", "parent_span_id": "i9j0k1l2", "name": "POST /v0/chat/completions", "start_time": "...", ...}
```

### Reading Trace Files

```bash
# View all spans for a trace (pretty-printed)
cat logs/all/2026-01-01T14-30-22_unity_dev_ttys042/099b207f89222185695d25977be454fc.jsonl | jq -s .

# Filter by service
cat logs/all/*/*.jsonl | jq -s '[.[] | select(.service == "orchestra")]'

# Find slow spans (>1s)
cat logs/all/*/*.jsonl | jq -s '[.[] | select(.duration_ms > 1000)]'
```

### Environment Variables

OTEL is enabled automatically by `parallel_run.sh`. To customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `UNITY_OTEL` | `true` (via parallel_run.sh) | Enable Unity OTEL tracing |
| `UNIFY_OTEL` | `true` (via parallel_run.sh) | Enable Unify SDK OTEL tracing |
| `UNILLM_OTEL` | `true` (via parallel_run.sh) | Enable Unillm OTEL tracing |
| `UNITY_OTEL_LOG_DIR` | `logs/all/` | Unity span output directory |
| `UNIFY_OTEL_LOG_DIR` | `logs/all/` | Unify span output directory |
| `UNILLM_OTEL_LOG_DIR` | `logs/all/` | Unillm span output directory |
| `ORCHESTRA_OTEL_LOG_DIR` | `logs/all/` | Orchestra span output directory |

To disable OTEL tracing:
```bash
parallel_run.sh --env UNITY_OTEL=false --env UNIFY_OTEL=false --env UNILLM_OTEL=false tests/
```

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
logs/unity/          →  ➡️ [ContactManager.ask(ca3e)] Request: ...
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
grep "TRACE_ID=" logs/pytest/2025-12-30T18-30-00_unity_dev_ttys042/contact_manager-test_ask.txt
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
grep TRACE_ID logs/pytest/2025-12-30T18-30-00_unity_dev/contact_manager-test_ask.txt
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

---

## Trace Upload to Test Context

At the end of each test, OTEL spans are automatically uploaded to a `Trace` child context under the test's context. This enables viewing distributed traces directly in the Orchestra UI alongside test data.

### Where Traces Go

```
tests/contact_manager/test_ask/test_ask_time_check/DefaultUser/DefaultAssistant/
├── Contacts          # ContactManager data
├── Transcripts       # TranscriptManager data
├── Trace             # ← OTEL spans uploaded here
└── ...
```

### Trace Context Schema

Each span is uploaded as a row with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | 32-char hex trace ID |
| `span_id` | `str` | 16-char hex span ID |
| `parent_span_id` | `str` | Parent span ID (or null for root) |
| `name` | `str` | Span name (e.g., `ContactManager.ask`, `POST /v0/logs`) |
| `service` | `str` | Service name: `unity`, `unify`, `unillm`, or `orchestra` |
| `start_time` | `datetime` | Span start timestamp |
| `end_time` | `datetime` | Span end timestamp |
| `duration_ms` | `float` | Duration in milliseconds |
| `status` | `str` | Span status (`UNSET`, `OK`, `ERROR`) |
| `attributes` | `dict` | Span attributes (HTTP params, DB queries, etc.) |

### Configuration

Control trace upload behavior via settings in `tests/settings.py` or environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `UNITY_TRACE_UPLOAD` | `true` | Enable/disable trace upload entirely |
| `UNITY_TRACE_SERVICES` | `all` | Services to include: `all` or comma-separated (e.g., `unity,orchestra`) |
| `UNITY_TRACE_EXCLUDE_PATTERNS` | `""` | Comma-separated span name patterns to exclude |

### Example Configurations

```bash
# Disable trace upload entirely
UNITY_TRACE_UPLOAD=false pytest tests/

# Unity spans only (~30 spans per test)
UNITY_TRACE_SERVICES=unity pytest tests/

# Unity + Orchestra HTTP spans, skip DB internals (~100 spans)
UNITY_TRACE_SERVICES=unity,orchestra
UNITY_TRACE_EXCLUDE_PATTERNS=connect,db.query pytest tests/

# Everything except auth overhead (~200 spans)
UNITY_TRACE_EXCLUDE_PATTERNS=db.query.select.users,db.query.select.api_key,db.query.select.team_member pytest tests/

# Full detail (default, ~600 spans per test)
UNITY_TRACE_SERVICES=all pytest tests/
```

### Understanding Span Counts

A typical test generates spans from multiple sources:

| Service | Typical Count | Contents |
|---------|---------------|----------|
| `unity` | ~30 | Manager methods, tool loops, internal operations |
| `unify` | ~10 | SDK HTTP client calls |
| `unillm` | ~5 | LLM request/response |
| `orchestra` | ~550 | HTTP handlers, DB queries (very granular) |

Orchestra's instrumentation is particularly detailed, capturing every SQL query and connection. Use `UNITY_TRACE_SERVICES=unity` for a clean view of test logic, or `UNITY_TRACE_EXCLUDE_PATTERNS=connect,db.query` to keep HTTP-level Orchestra spans while filtering DB noise.

### Querying Trace Data

```python
import unify

unify.activate("UnityTests")

# Get trace spans for a specific test
trace_ctx = "tests/contact_manager/test_ask/test_ask_time_check/DefaultUser/DefaultAssistant/Trace"
logs = unify.get_logs(context=trace_ctx, limit=100)

# Filter by service
unity_spans = [log for log in logs if log.entries.get("service") == "unity"]

# Find slow spans
slow_spans = [log for log in logs if log.entries.get("duration_ms", 0) > 1000]
```
