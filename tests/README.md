# Tests

This directory contains the test suite for Unity. Before diving into how to run tests, it's important to understand the philosophy behind how tests are structured.

## Test Philosophy: Symbolic ↔ Eval Spectrum

Tests in this codebase fall on a spectrum between two paradigms:

### Symbolic Tests

At one end of the spectrum, **symbolic tests** use the LLM purely as a stub. The LLM receives minimal "dummy" instructions designed to trigger specific code paths, allowing us to verify that core async logic, tools, and state management work correctly.

**Key characteristics:**
- LLM behavior is deterministic and predictable
- Focus is on testing the *infrastructure*: async tool loops, steering, pausing/resuming, state mutations
- The LLM's "intelligence" is irrelevant—we just need it to call the right tools in the right order
- Failures indicate regressions in our symbolic/programmatic logic

### Eval Tests

At the other end, **eval tests** exercise the system end-to-end. We ask a high-level question or give a directive, then verify the outcome—regardless of how many internal tool calls or LLM steps occurred.

**Key characteristics:**
- Focus is on *capability*: "Did the assistant correctly answer the question?" or "Did it complete the task?"
- Internal implementation details (tool call order, number of steps) don't matter
- Tests the LLM's reasoning and decision-making in realistic scenarios
- Failures may indicate prompt issues, tool design problems, or genuine capability gaps

### The Spectrum (Not Binary)

Most tests sit somewhere between these extremes. A test might:
- Use realistic prompts but only verify specific tool calls were made
- Test end-to-end behavior but with constrained, predictable inputs
- Combine symbolic infrastructure checks with high-level outcome assertions

Think of each test as having a "slider" between symbolic and eval—not a binary classification.

### Caching and Determinism (`UNIFY_CACHE`)

When `UNIFY_CACHE="true"` (the default), all LLM responses are cached in `.cache.ndjson` files:

1. **First run**: The LLM executes normally; responses are stored in the cache
2. **Subsequent runs**: Cached responses are replayed—no actual LLM calls occur

This means:
- **Symbolic tests** behave identically on every run (cache acts as a deterministic stub)
- **Eval tests** *also* become deterministic after the first run—they replay the same LLM "thinking" that produced the original passing result
- Both test types effectively verify that *symbolic logic has not regressed* once the cache is populated
- Tests run fast on CI (milliseconds vs seconds/minutes for real LLM calls)
- To re-evaluate LLM behavior, delete the relevant `.cache.ndjson`, set `UNIFY_CACHE="false"`, or use `--env UNIFY_CACHE=false` with the parallel runner

### Tagging Tests as Eval

To mark a test file as eval (end-to-end LLM reasoning), add a module-level pytest marker:

```python
import pytest

# All tests in this file exercise end-to-end LLM reasoning
pytestmark = pytest.mark.eval
```

For mixed files where only some tests are eval, use test-level markers:

```python
@pytest.mark.eval
@pytest.mark.asyncio
async def test_natural_language_query():
    ...
```

### Running Test Categories

Use the parallel runner flags to filter by test category:

```bash
# Run only eval tests (end-to-end LLM reasoning)
./parallel_run.sh --eval-only tests

# Run only symbolic tests (infrastructure/deterministic)
./parallel_run.sh --symbolic-only tests

# Standard pytest also works
pytest -m eval tests/
pytest -m "not eval" tests/
```

---

## Running Tests

### Shell Setup (Recommended)

Add these aliases to your `~/.zshrc` or `~/.bashrc` for convenient access to test helpers from anywhere:

```bash
# Unity test helper aliases
alias parallel_run='~/unity/tests/parallel_run.sh'
alias watch_tests='~/unity/tests/watch_tests.sh'
alias attach='~/unity/tests/attach.sh'
alias kill_failed='~/unity/tests/kill_failed.sh'
alias kill_server='~/unity/tests/kill_server.sh'
alias list_runs='~/unity/tests/list_runs.sh'
alias monitor_resources='~/unity/tests/monitor_resources.sh'
```

After adding, run `source ~/.zshrc` (or restart your terminal). You can then run `parallel_run`, `watch_tests`, etc. from any directory.

### Quick Start

```bash
# Run all tests sequentially
pytest tests/

# Run a specific test file
pytest tests/test_contact_manager/test_create_contact.py

# Run a specific test
pytest tests/test_contact_manager/test_create_contact.py::test_create_single_contact
```

### Parallel Execution

For faster runs, use either:

1. **pytest-xdist** (simple, built-in):
   ```bash
   pytest -n auto tests/
   ```

2. **`parallel_run.sh`** (better debugging experience—see below)

> **⚡ Speed Tip:** When running a small number of tests (1-20), always use `-t` with `parallel_run.sh` to run each test in its own tmux session. Without `-t`, tests within the same file run serially, which can block for 10+ minutes unnecessarily. See [Per-Test Mode](#per-test-mode--t-for-maximum-parallelism) below.

---

## Parallel Test Runner (`parallel_run.sh`)

This helper script launches one tmux session per test file (or per test function with `-t`) and runs `pytest` in its own window. It searches recursively and can be restricted to specific folders, files, or tests.

### Terminal Isolation (Automatic)

Each terminal session automatically gets its own **isolated tmux server**. This means:

- **Cursor agents don't interfere with each other**: Each agent's tests run in their own isolated tmux server
- **`tmux kill-server` is safe**: It only kills sessions from the terminal that ran it
- **No configuration needed**: Isolation is automatic based on the terminal's TTY device

**How it works:** The script derives a unique socket name from your terminal's TTY (e.g., `/dev/ttys042` → socket `unity_dev_ttys042`). All tmux commands use this socket automatically.

**Monitoring your tests:**

```bash
# Watch YOUR terminal's tests (automatic isolation)
tests/watch_tests.sh

# Watch ALL terminals' tests
tests/watch_tests.sh --all

# Attach to a specific session to see its output
tests/attach.sh '<session-name>'
```

**Recovering orphaned runs (when you close the original terminal):**

```bash
# List all active test runs across all terminals
tests/list_runs.sh

# Watch tests from a specific socket (orphaned run)
tests/watch_tests.sh --socket unity_dev_ttys042

# Attach to a session in a specific socket
tests/attach.sh --socket unity_dev_ttys042 'f ❌ test_actor-test_code_act'

# Kill failed sessions in a specific socket
tests/kill_failed.sh --socket unity_dev_ttys042

# Kill a specific socket's server
tests/kill_server.sh --socket unity_dev_ttys042
```

**Cleanup:**

```bash
# Kill failed sessions from THIS terminal
tests/kill_failed.sh

# Kill failed sessions from ALL terminals
tests/kill_failed.sh --all

# Kill the entire tmux server for THIS terminal
tests/kill_server.sh

# Kill ALL unity test tmux servers
tests/kill_server.sh --all
```

### Per-Test Mode (`-t`) for Maximum Parallelism

**IMPORTANT:** By default, the script creates one tmux session per *file*. If a single file contains 15 tests, they run serially within that session—potentially blocking for 10+ minutes.

Use `-t/--per-test` to create one session per *test function*, enabling full parallelism:

```bash
# WITHOUT -t: 15 tests in one file run serially (~10 min)
./parallel_run.sh --wait tests/test_contact_manager/test_ask.py

# WITH -t: 15 tests run concurrently in 15 sessions (~1 min)
./parallel_run.sh -t --wait tests/test_contact_manager/test_ask.py
```

**When to use `-t`:**
- Running a single test file with multiple tests
- Running a small number of specific tests
- Running a small directory (< 20 tests total)
- Anytime you want maximum speed and don't mind many tmux sessions

**When to omit `-t`:**
- Running the entire test suite (hundreds of tests)
- Running many files where per-file grouping helps organization
- When you prefer fewer, more manageable tmux sessions

### Why not just pytest-xdist?

pytest-xdist works fine for basic parallel execution. However, `parallel_run.sh` provides a significantly better **debugging experience** for our LLM-heavy async tests:

| Feature | `parallel_run.sh` | pytest-xdist |
|---------|-------------------|--------------|
| **Interactive debugging** | `tmux attach -t <session>` to any running/failed test | Output multiplexed across workers; hard to isolate |
| **Post-failure inspection** | Failed sessions stay open with full scrollback | Just a failure message in terminal |
| **Visual status** | Real-time `p ✅` / `f ❌` / `r ⏳` per test file | Single progress bar |
| **Log isolation** | Per-run folders in `.pytest_logs/{datetime}_{socket}/` | Merged output (requires extra config) |
| **Load balancing** | Static (1 session = 1 target) | Dynamic redistribution |

**When tmux shines:** Our tests involve complex async LLM tool loops with steering, pausing, resuming, and interjections. When something fails, you need the complete context—the LLM I/O, the async flow, the interleaved logs. Being able to `tmux attach` to a failing test, scroll through its full history, and even interact with it is invaluable.

**When to use xdist instead:** For quick parallel runs where you don't need debugging (`pytest -n auto`), or when dynamic load balancing matters (tests with highly variable durations).

**TL;DR:** This script prioritizes **developer experience** over raw parallelization efficiency. Both approaches achieve parallelism; this one makes debugging failures much easier.

### Shared Project Mode (Default)

By default, `parallel_run.sh` uses a **shared project mode** where all parallel test sessions log to the same `UnityTests` project. This enables:

- **Unified duration logging**: All test durations and LLM I/O are recorded in a single `Combined` context, making it easy to compare runtimes and review LLM calls across different test files.
- **Race-free parallel execution**: The script automatically runs an internal prepare module (`_prepare_shared_project.py`) before spawning sessions. This module idempotently creates the shared project and contexts once, eliminating race conditions.
- **Faster startup**: Sessions skip redundant project/context creation since it's already done.

When a session starts in shared mode, it executes roughly:

```bash
export UNIFY_SKIP_SESSION_SETUP=True
source ~/unity/.venv/bin/activate
pytest <target>
```

### Random Projects Mode

For isolation purposes, you can use `--env` to give each tmux session its own isolated project:

```bash
./parallel_run.sh --env UNIFY_TESTS_RAND_PROJ=true --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```

In this mode, each session gets a unique project like `UnityTests_aB3xY9zQ` which is deleted on exit. The script auto-detects when `UNIFY_TESTS_RAND_PROJ=true` is set and skips the shared project preparation.

### Live Status and Auto-Close

- **Status prefix**: Each tmux session name is prefixed with a typeable marker and emoji: `r ⏳` while the test runs, `p ✅` on success (passed), or `f ❌` on failure. The letters are chosen to sort alphabetically as failed→passed→running, so failing tests appear first in listings. This also makes tab-completion easy in shells like zsh.
- **Auto-close on success**: Sessions that pass are automatically killed about 10 seconds after completion. Failing sessions remain open for inspection.
- You can still attach before auto-close; you'll see the final message (e.g., `pytest exited with code: 0`) and a short notice that auto-close is scheduled.

### Installation

Save the script at the repository root as a hidden file and make it executable:

```bash
chmod +x parallel_run.sh
```

### Requirements

- **tmux** and **pytest** must be installed (e.g., `brew install tmux`).
- **Virtualenv** is assumed to live at `~/unity/.venv/`. If yours differs, update the `source ~/unity/.venv/bin/activate` line inside the script.
- Optional: create an `.env` file at the repository root (i.e., `~/unity/.env`). Both helper scripts will auto-load it if present via `tests/../.env`.

### Basic Usage

From the repository root, run:

```bash
./parallel_run.sh
```

What happens:

1. **Prepare**: The shared `UnityTests` project and `Combined` context are created (if not already present).
2. **Discovery**: Recursively finds all `test_*.py` files (excluding caches/venvs; see excludes below).
3. **Sessions**: Creates one tmux session per file.
4. **Window name**: The file's basename without `.py`.
5. **Session name**: Status-prefixed and derived from the file path, e.g., `tests/unit/test_math.py` → `r ⏳ unit-test_math` (then `p ✅ unit-test_math` or `f ❌ unit-test_math`).

Common tmux actions:

```bash
tmux ls                                # list sessions
tmux attach -t <session-name>          # attach to a session
tmux switch-client -t <session-name>   # switch sessions (when already inside tmux)
```

### Targeting Specific Folders/Files/Tests

Limit the search by passing directories and/or `.py` files. Examples:

```bash
# Only run files under a single folder
./parallel_run.sh tests/integration

# Multiple roots
./parallel_run.sh tests/unit tests/integration

# Specific files
./parallel_run.sh tests/foo_test.py tests/bar_test.py

# Specific tests (pytest node ids)
./parallel_run.sh tests/foo_test.py::TestClass::test_something tests/bar_test.py::test_case

# Per-test mode (create a session per test for all inputs)
./parallel_run.sh -t                         # per-test across the whole repo
./parallel_run.sh -t tests                   # per-test across a folder
./parallel_run.sh -t tests/foo_test.py       # per-test across a single file
./parallel_run.sh -t tests tests/foo_test.py # mix folders and files, all per-test

# Mix files and directories
./parallel_run.sh tests/api tests/db/test_migrations.py

# Wait for completion and log to files (CI / Agent mode)
./parallel_run.sh --wait tests/unit

# Set environment variables (see "Environment Variable Overrides" below)
./parallel_run.sh --env UNIFY_CACHE=false tests
./parallel_run.sh -e UNIFY_CACHE=false -e UNIFY_DELETE_CONTEXT_ON_EXIT=true tests

# Use isolated random projects
./parallel_run.sh --env UNIFY_TESTS_RAND_PROJ=true --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests

# Run only eval tests (end-to-end LLM reasoning tests)
./parallel_run.sh --eval-only tests

# Run only symbolic tests (infrastructure/deterministic tests)
./parallel_run.sh --symbolic-only tests

# Repeat tests for statistical sampling (see below)
./parallel_run.sh --env UNIFY_CACHE=false --repeat 10 --eval-only tests/test_contact_manager

# Tag test runs for filtering (logged to Combined context)
./parallel_run.sh --tags "experiment-1" tests
./parallel_run.sh --tags "model-compare,gpt-4o" tests

# Combine with other options
./parallel_run.sh --eval-only --wait tests/test_contact_manager
./parallel_run.sh --env UNIFY_CACHE=false --eval-only tests/test_contact_manager
```

How it interprets arguments:

- **Directories**: Recursed (respecting excludes) to find `*.py`.
- **Files**: Run exactly as provided (no recursion).
- **Tests**: Pytest node ids like `path/to/test_file.py::TestClass::test_case` or `path/to/test_file.py::test_case` are run exactly as provided (one session per node id).
  - If you specify individual tests, only those tests are run (one session per test).
  - When you do not specify individual tests, the script creates one session per file.
  - With `-t/--per-test`, the script collects node ids via `pytest --collect-only` and creates one session per test for every directory/file you pass (plus any explicit node ids).

### Wait Mode and Logs (`--wait [N]`)

Use `-w/--wait` to block until all tests finish. This is useful for CI/CD pipelines or automated agents.

```bash
# Wait indefinitely until all tests complete
./parallel_run.sh --wait tests/my_tests

# Wait up to 120 seconds, then timeout
./parallel_run.sh --wait 120 tests/my_tests
```

**Behavior:**
- Blocks until all tmux sessions complete (or timeout is reached).
- If all pass, exits with code `0`.
- If any fail, exits with code `1` and lists the failed sessions.
- If timeout is reached before completion, exits with code `2`.
- **Logs**: Each session writes its full pytest output to a datetime-prefixed folder in `.pytest_logs/{datetime}_{socket}/` with semantic naming (e.g., `.pytest_logs/2025-12-05T14-30-22_unity_dev_ttys042/test_contact_manager-test_ask.txt`).
- **Debugging**: When running with `--wait`, inspect these log files to diagnose failures instead of attaching to tmux sessions (though sessions remain open for inspection if they fail).

**Timeout examples:**
```bash
# Quick sanity check with 60s timeout
./parallel_run.sh --wait 60 -t tests/test_basic.py

# Long-running tests with 5 minute timeout
./parallel_run.sh --wait 300 tests/test_slow_suite/
```

### Match Tests by Filename (Glob-Style)

Use `-m/--match` to run tests whose basenames match a simple glob pattern. The pattern is matched against the filename only (not the full path). Quote the pattern to prevent your shell from expanding it.

Examples:

```bash
# Run all "docstring" focused tests (each in its own tmux session)
./parallel_run.sh -m "*_tool_docstring*"
```

This one-liner matches files such as:

- `tests/test_contact/test_contact_tool_docstrings.py`
- `tests/test_transcript_manager/test_transcript_tool_docstrings.py`
- `tests/test_task_scheduler/test_task_tool_docstrings.py`
- `tests/test_conductor/test_conductor_tool_docstrings.py`
- `tests/test_file_manager/test_file_tool_docstrings.py`
- `tests/test_guidance/test_guidance_tool_docstring.py`
- `tests/test_knowledge/test_knowledge_tool_docstrings.py`
- `tests/test_secret_manager/test_secret_manager_tool_docstrings.py`
- `tests/test_skill_manager/test_skill_tool_docstrings.py`
- `tests/test_web_searcher/test_web_tool_docstrings.py`

Notes:

- `*` means "anything before/after" in the filename. You can combine it with other characters (e.g., `test_*_tool_docstring*.py`).
- When using `-m/--match`, the default behavior still applies: one tmux session per matching test file.

### Environment Variable Overrides (`--env`)

The `-e/--env KEY=VALUE` flag sets environment variables for all pytest sessions. This is the primary way to configure test behavior—any environment variable recognized by `TestingSettings` (see `tests/settings.py`) can be overridden.

**Usage:**

```bash
# Single override
./parallel_run.sh --env UNIFY_CACHE=false tests

# Multiple overrides (flag can be repeated)
./parallel_run.sh -e UNIFY_CACHE=false -e UNIFY_DELETE_CONTEXT_ON_EXIT=true tests

# Disable caching for fresh LLM calls
./parallel_run.sh --env UNIFY_CACHE=false tests

# Use isolated random projects (each session gets its own project)
./parallel_run.sh --env UNIFY_TESTS_RAND_PROJ=true --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```

**Available Variables:**

Settings are organized in two classes with inheritance:
- `ProductionSettings` (`unity/settings.py`) - used in deployed system AND tests
- `TestingSettings` (`tests/settings.py`) - inherits production + adds test-only settings

The `--env` approach is intentionally generic. Any variable from either class is available via `--env` without modifying the shell script.

**Production Settings** (also used in tests):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UNIFY_MODEL` | str | `gpt-5.1@openai` | LLM model to use |
| `UNIFY_CACHE` | bool/str | `true` | Enable/disable LLM response caching |
| `LLM_IO_DEBUG` | bool | `true` | Log full LLM request/response payloads |
| `ASYNCIO_DEBUG` | bool | `false` | Enable asyncio debug mode |
| `ASYNCIO_VERBOSE_DEBUG` | bool | `false` | Verbose asyncio logging with task/thread breadcrumbs |
| `PYTEST_LOG_TO_FILE` | bool | `true` | Log pytest output to files |
| `UNITY_SEMANTIC_CACHE` | bool | `false` | Enable semantic cache mode |
| `UNITY_READONLY_ASK_GUARD` | bool | `true` | Enable read-only ask guard (mutation-intent classifier) |
| `FIRST_ASK_TOOL_IS_SEARCH` | bool | `true` | Force semantic search tool on first step of `ask` methods |
| `FIRST_MUTATION_TOOL_IS_ASK` | bool | `true` | Force `ask` tool on first step of mutation methods (`update`, `refactor`, `organize`) |
| `UNITY_SILENCE_HTTPX` | bool | `true` | Silence httpx library logging |
| `UNITY_SILENCE_URLLIB3` | bool | `true` | Silence urllib3 library logging |
| `UNITY_SILENCE_OPENAI` | bool | `true` | Silence openai library logging |
| `UNITY_LOG_ONLY_PROJECT` | bool | `true` | Only log unity project messages |
| `UNITY_LOG_INCLUDE_PREFIXES` | str | `"unity"` | Comma-separated logger prefixes to include |

**Test-Only Settings**:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UNIFY_DELETE_CONTEXT_ON_EXIT` | bool | `false` | Delete test context after each test |
| `UNIFY_OVERWRITE_PROJECT` | bool | `false` | Overwrite project on activation |
| `UNIFY_REGISTER_SUMMARY_CALLBACKS` | bool | `false` | Register summary callbacks |
| `UNIFY_REGISTER_UPDATE_CALLBACKS` | bool | `false` | Register update callbacks |
| `UNIFY_TESTS_RAND_PROJ` | bool | `false` | Use random project names (isolated per session) |
| `UNIFY_TESTS_DELETE_PROJ_ON_EXIT` | bool | `false` | Delete random project when session exits |
| `UNIFY_CACHE_BENCHMARK` | bool | `false` | Enable cache hit/miss benchmarking |
| `UNIFY_PRETEST_CONTEXT_CREATE` | bool | `false` | Pre-create contexts before tests |
| `UNIFY_TEST_TAGS` | str | `""` | Comma-separated tags for duration logging (use `--tags` shorthand) |
| `UNIFY_SKIP_SESSION_SETUP` | bool | `false` | Skip project/context creation (pre-done) |

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-w [N]`, `--wait [N]` | Block until all tests complete; exit 0 on success, 1 on failure, 2 on timeout. Optional `N` sets timeout in seconds. |
| `-t`, `--per-test` | Create one session per test function instead of per file |
| `-m PATTERN`, `--match PATTERN` | Only run files matching the glob pattern |
| `-e KEY=VALUE`, `--env KEY=VALUE` | Set environment variable for all sessions (repeatable) |
| `--tags TAG` | Tag test runs for filtering (shorthand for `--env UNIFY_TEST_TAGS=...`; repeatable, comma-separated) |
| `--eval-only` | Run only tests marked with `pytest.mark.eval` (end-to-end LLM tests) |
| `--symbolic-only` | Run only tests NOT marked with `pytest.mark.eval` (infrastructure tests) |
| `--repeat N` | Run each test N times; useful for statistical sampling (see below) |

### Statistical Sampling with `--repeat`

The `--repeat N` flag runs each test target N times, creating N separate tmux sessions per target. While this works with any test, **the primary use case is for eval tests with `UNIFY_CACHE=false`**.

**Why this matters:**

- **Symbolic tests** are deterministic—running them multiple times yields identical results (especially with caching enabled). There's no statistical value.
- **Eval tests with caching** are also deterministic after the first run—the cached LLM responses are replayed exactly.
- **Eval tests with `UNIFY_CACHE=false`** make fresh LLM calls each run. The LLM may reason differently, take more/fewer steps, or even fail occasionally. Each run is an independent sample.

**Use cases for repeated eval runs:**

1. **Pass rate estimation**: Run an eval test 20 times to measure reliability (e.g., "passes 18/20 = 90%")
2. **Runtime distribution**: Plot test durations across runs to understand variance
3. **LLM step analysis**: Compare how many tool calls or reasoning steps the model takes
4. **Thinking time metrics**: Measure average LLM response latency across samples
5. **Regression detection**: A test that was 100% reliable but now fails 5% of the time indicates a problem

**Example workflow:**

```bash
# Run a specific eval test 10 times without caching
./parallel_run.sh --env UNIFY_CACHE=false --repeat 10 --eval-only tests/test_contact_manager/test_ask.py

# Run all eval tests 5 times each, wait for completion
./parallel_run.sh --env UNIFY_CACHE=false --repeat 5 --eval-only --wait tests
```

Each repeated run gets its own tmux session (with `-2`, `-3`, etc. suffixes to avoid name collisions) and its own log file in `.pytest_logs/{datetime}_{socket}/`. After completion, you can analyze the logs to compute statistics.

### Defaults & Conventions

- **Environment**:
  - If `../.env` exists relative to the `tests` directory (i.e., `~/unity/.env`), it will be sourced automatically so you can define `UNIFY_KEY`, `UNIFY_BASE_URL`, or other variables once.
  - By default, exports `UNIFY_SKIP_SESSION_SETUP=True` for shared project mode.
  - Use `--env` to override any `TestingSettings` variable (see table above).
- **Virtualenv**: Assumes `~/unity/.venv/bin/activate`.
- **Excludes**: Skips directories: `.git`, `.hg`, `.svn`, `.venv`, `venv`, `.mypy_cache`, `.pytest_cache`, `__pycache__`, `.idea`, `.vscode`.
  - You can edit the `EXCLUDE_DIRS` array in the script to add/remove entries.
- **Names**:
  - Session: `<status-prefix> <relative-path-with-slashes-replaced-by-dashes>` (without `.py`). Example: `r ⏳ unit-test_math` → `p ✅ unit-test_math` or `f ❌ unit-test_math`.
  - Window: `<filename-without-.py>`.
  - If a session name already exists, the script appends `-2`, `-3`, … to avoid collisions.

### Tips

- **Watch session statuses live**:

  ```bash
  tests/watch_tests.sh        # Watch THIS terminal's tests
  tests/watch_tests.sh --all  # Watch ALL terminals' tests
  ```

  As tests start, sessions show a `r ⏳` prefix. They flip to `p ✅` or `f ❌` when pytest exits. Successful sessions auto-close ~10s later.

- **Kill all failed sessions** at once:

  ```bash
  tests/kill_failed.sh        # Kill failed sessions from THIS terminal
  tests/kill_failed.sh --all  # Kill failed sessions from ALL terminals
  tests/kill_failed.sh -n     # Dry run - show what would be killed
  ```

  > **Best Practice:** Always clean up failed sessions after you've extracted the failure info from `.pytest_logs/`. Logs are persisted there, so keeping sessions open just clutters the output. Run `tests/kill_failed.sh` after investigating failures.

- **Kill a single session** once a test finishes (the socket name is printed when tests are launched):

  ```bash
  tmux -L <socket> kill-session -t <session-name>
  ```

  Note: sessions that pass auto-close within ~10 seconds; you typically only need to kill failing sessions.

- **Run in the background** (script exits immediately; sessions keep running):

  ```bash
  nohup ./parallel_run.sh tests &>/dev/null &
  ```

- **See test output later**: The socket name is printed when tests are launched. Use it to attach:

  ```bash
  tmux -L <socket> attach -t <session-name>
  ```

### Troubleshooting

- **"tmux: command not found"**
  - Install tmux (e.g., `brew install tmux`, `apt-get install tmux`).

- **Virtualenv not found / wrong Python**
  - Update the activation line in the script:

    ```bash
    source /path/to/your/venv/bin/activate
    ```

- **No sessions created**
  - Ensure there are `.py` files under the provided paths and that excludes aren't hiding your files.

- **Permission denied**
  - Make the script executable:

    ```bash
    chmod +x parallel_run.sh
    ```

### Customization

Open `parallel_run.sh` and tweak as needed:

- **`EXCLUDE_DIRS=( ... )`** — add/remove directories to skip.
- **`run_cmd()`** — change the command chain (e.g., add flags: `pytest -q -x`).
- **Session naming** — adjust `session_basename_for()` to your taste.

### Quick Reference (tmux with isolation)

Each terminal uses its own tmux socket (printed when tests launch). Common commands:

```bash
# List sessions for THIS terminal
tmux -L <socket> ls

# Attach to a session
tmux -L <socket> attach -t <name>

# Kill a session
tmux -L <socket> kill-session -t <name>

# Inside tmux, switch sessions
tmux switch-client -t <name>
```

**Helper scripts (recommended):**

```bash
tests/list_runs.sh          # List all active test runs (all sockets)
tests/watch_tests.sh        # Watch this terminal's tests
tests/attach.sh '<name>'    # Attach to a session
tests/kill_failed.sh        # Kill failed sessions
tests/kill_server.sh        # Kill all sessions (entire server)
```

**Recovering orphaned runs:**

```bash
# If you close a terminal, the tests keep running. Use list_runs to find them:
tests/list_runs.sh

# Then use --socket to target the orphaned run from any terminal:
tests/watch_tests.sh --socket <socket-name>
tests/attach.sh --socket <socket-name> '<session-name>'
tests/kill_failed.sh --socket <socket-name>
tests/kill_server.sh --socket <socket-name>
```

That's it! Run tests, use the helpers to monitor, and jump into whichever test you want to watch.

---

## Log Directory Structure

Test logs are organized into **datetime-prefixed directories** for natural time-based ordering and to prevent confusion when multiple agents or users run tests concurrently.

### Directory Layout

Directory names follow the format: `YYYY-MM-DDTHH-MM-SS_{socket_name}`
- The datetime prefix enables chronological sorting in filesystem listings
- The socket name identifies the terminal session for isolation

```
.pytest_logs/
├── 2025-12-05T09-15-22_unity_dev_ttys042/    # Run at 09:15 from Terminal A
│   ├── test_contact-test_ask.txt
│   └── test_task-test_update.txt
├── 2025-12-05T10-30-45_unity_dev_ttys099/    # Run at 10:30 from Terminal B (agent)
│   └── test_contact-test_ask.txt
├── 2025-12-05T14-22-18_unity_dev_ttys042/    # Run at 14:22 from Terminal A (new run)
│   └── test_foo.txt
├── 2025-12-05T14-35-00_unity_pid12345/       # Non-interactive shell
│   └── ...
└── standalone/                               # Direct pytest runs (no parallel_run.sh)
    └── test_foo_2025-12-05_14-35-00.txt

.llm_io_debug/
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
📄 Test log: /Users/you/unity/.pytest_logs/2025-12-05T14-30-22_unity_dev_ttys042/test_foo.txt
📁 This run's logs: /Users/you/unity/.pytest_logs/2025-12-05T14-30-22_unity_dev_ttys042/
📂 All log directories:  /Users/you/unity/.pytest_logs/*/
========================================================================
```

**Finding recent runs:** Directories are sorted chronologically, so recent runs appear at the bottom of `ls` output:
```bash
ls .pytest_logs/              # Oldest first, newest last
ls -r .pytest_logs/           # Newest first
```

**For agents:** Read the terminal output to find the exact log path. The directory name (e.g., `2025-12-05T14-30-22_unity_dev_ttys042`) is printed when tests start via `parallel_run.sh`.

**For cross-run analysis:** Use glob patterns to search across all runs:
```bash
ls .pytest_logs/*/            # List all run directories
ls .pytest_logs/*/*.txt       # List all log files across all runs
```

---

## Test Data Logging

Tests log rich telemetry to the Unify backend, enabling post-hoc analysis of test runs, LLM behavior, and performance. Data is organized into two layers: a **global summary context** and **per-test contexts**.

### Combined Context (Global Summary)

Every test logs a summary record to the shared `Combined` context within the `UnityTests` project. This provides a unified view across all tests in a session.

**Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `test_fpath` | `str` | Test path: `folder/file.py::test_name` |
| `tags` | `list` | Session-level tags (via `--tags`, `--test-tags`, or `UNIFY_TEST_TAGS`) |
| `duration` | `float` | Wall-clock time in seconds |
| `llm_io` | `list` | Full LLM request/response logs (from `.llm_io_debug/` files) |
| `settings` | `dict` | Complete settings snapshot (production + test-only) |

**Use cases:**

- **Duration analysis**: Compare runtimes across tests, identify slow tests, track performance regressions
- **Settings ablation**: Filter by settings values to compare behavior with different configurations (e.g., `UNIFY_CACHE=true` vs `UNIFY_CACHE=false`)
- **LLM debugging**: Inspect the full LLM I/O for any test without re-running it
- **Tagging experiments**: Use `--test-tags "experiment-A,gpt-4"` to label runs for later filtering

**Example query:** "Show all tests where `UNIFY_CACHE=false` and duration > 10s"

### Per-Test Contexts (State Manager Data)

Each test decorated with `@_handle_project` gets its own isolated context. Within this context, state managers (`ContactManager`, `TranscriptManager`, `TaskScheduler`, etc.) store their domain data in sub-contexts.

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

### Scenario Fixtures (Shared Seed Data)

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

### Inspecting Logged Data

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

### Context Cleanup

By default, test contexts persist across runs (useful for debugging). To auto-delete:

```bash
# Delete context after each test
./parallel_run.sh --env UNIFY_DELETE_CONTEXT_ON_EXIT=true tests

# Or delete entire project after session
./parallel_run.sh --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```

---

## Grid Search (`grid_search.sh`)

Run tests across all combinations of settings values. This is useful for:

- **Model comparisons**: Compare behavior across different LLMs
- **Feature flag ablations**: Test with settings enabled/disabled
- **Configuration sweeps**: Find optimal settings combinations

### Basic Usage

```bash
# Make executable (first time only)
chmod +x tests/grid_search.sh

# Grid search across models
./grid_search.sh --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" tests/test_contact_manager/

# Grid search across models AND cache settings (2×2 = 4 combinations)
./grid_search.sh --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" --env UNIFY_CACHE="true|false" tests/
```

### How It Works

1. **Parse grid variables**: Settings with `|` separators define the search space
2. **Generate combinations**: Full Cartesian product of all values
3. **Launch runs**: Each combination spawns a separate `parallel_run.sh` invocation
4. **Auto-tag**: Each run is automatically tagged with its `--env` values for easy filtering
5. **Log results**: Each run logs both tags and full settings dict to `Combined`

### Auto-Tagging

Each run is **automatically tagged** with all `--env` values passed to `grid_search.sh`. This makes post-hoc analysis trivial—you can filter results by the exact configuration used for each run.

**How it works:**

- Tags are formatted as `KEY1=val1,KEY2=val2,...` (comma-separated)
- **Grid variables** (with `|`): The specific value selected for that run is tagged
- **Constant variables** (no `|`): Included in tags for all runs
- **Background variables** (from `.env` file): NOT included in tags

**Why this design?**

When running a grid search, you want to filter results by the variables you're actively experimenting with. Variables from `.env` or other sources are held constant across all runs and don't help distinguish between grid cells. They still appear in the full `settings` dict for completeness, but the `tags` field contains only what you passed on the command line.

**Example:**

```bash
./grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNIFY_CACHE="true|false" tests/
```

Generates 4 runs with these auto-tags:

| Run | Tags |
|-----|------|
| 1 | `UNIFY_MODEL=gpt-4o,UNIFY_CACHE=true` |
| 2 | `UNIFY_MODEL=gpt-4o,UNIFY_CACHE=false` |
| 3 | `UNIFY_MODEL=claude-3,UNIFY_CACHE=true` |
| 4 | `UNIFY_MODEL=claude-3,UNIFY_CACHE=false` |

With a constant variable:

```bash
./grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env EXPERIMENT_ID="exp-42" tests/
```

Generates 2 runs:

| Run | Tags |
|-----|------|
| 1 | `UNIFY_MODEL=gpt-4o,EXPERIMENT_ID=exp-42` |
| 2 | `UNIFY_MODEL=claude-3,EXPERIMENT_ID=exp-42` |

### Syntax

```bash
./grid_search.sh [options] --env KEY=val1|val2|val3 [--env KEY2=a|b] [targets...]
```

- **Pipe (`|`)**: Separates values to grid over
- **No pipe**: Single value passed through to all runs
- **Targets**: Test files/directories (same as `parallel_run.sh`)

### Options

| Option | Description |
|--------|-------------|
| `--env KEY=val1\|val2` | Grid variable (multiple values, pipe-separated); each value becomes a separate run |
| `--env KEY=value` | Constant variable (single value for all runs, included in auto-tags) |
| `-n`, `--dry-run` | Show generated commands without executing (including auto-tags) |
| `--wait-all` | Run combinations sequentially (with `--wait` per run) |
| `-h`, `--help` | Show help |

All other options are passed through to `parallel_run.sh`.

### Examples

**Model comparison with eval tests:**

```bash
./grid_search.sh \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic|gemini-2.5-pro@google" \
  --env UNIFY_CACHE="false" \
  --eval-only \
  tests/test_contact_manager/
```

This generates 3 runs (one per model), each with fresh LLM calls.

**Feature flag ablation:**

```bash
./grid_search.sh \
  --env FIRST_ASK_TOOL_IS_SEARCH="true|false" \
  --env FIRST_MUTATION_TOOL_IS_ASK="true|false" \
  tests/test_conductor/
```

This generates 4 runs (2×2 grid) testing all combinations of these two feature flags.

**Dry run to preview:**

```bash
./grid_search.sh -n \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" \
  --env UNIFY_CACHE="true|false" \
  tests/
```

Output:

```
Grid Search Configuration
=========================
Grid variables:
  UNIFY_MODEL: gpt-4o@openai | claude-sonnet-4-20250514@anthropic
  UNIFY_CACHE: true | false

Total combinations: 4

Generated runs:
  [1/4] UNIFY_MODEL=gpt-4o@openai UNIFY_CACHE=true
  [2/4] UNIFY_MODEL=gpt-4o@openai UNIFY_CACHE=false
  [3/4] UNIFY_MODEL=claude-sonnet-4-20250514@anthropic UNIFY_CACHE=true
  [4/4] UNIFY_MODEL=claude-sonnet-4-20250514@anthropic UNIFY_CACHE=false

Dry run - commands that would be executed:

  tests/parallel_run.sh --env UNIFY_MODEL=gpt-4o@openai --env UNIFY_CACHE=true --tags UNIFY_MODEL=gpt-4o@openai,UNIFY_CACHE=true tests/
  tests/parallel_run.sh --env UNIFY_MODEL=gpt-4o@openai --env UNIFY_CACHE=false --tags UNIFY_MODEL=gpt-4o@openai,UNIFY_CACHE=false tests/
  ...
```

Note that each run includes `--tags` with the specific configuration values—this happens automatically.

**Sequential execution (resource-constrained):**

```bash
./grid_search.sh --wait-all \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" \
  tests/
```

Runs combinations one at a time instead of all concurrently.

### Analyzing Results

After a grid search, query the `Combined` context to compare results. The auto-generated tags make filtering straightforward:

```python
import unify

unify.activate("UnityTests")
logs = unify.get_logs(context="Combined")

# Filter by tags (contains the exact --env values from the grid search)
for log in logs:
    tags = log.get("tags", [])
    duration = log.get("duration", 0)
    # Tags are like ["UNIFY_MODEL=gpt-4o", "UNIFY_CACHE=true"]
    print(f"{tags}: {duration:.2f}s")

# Or filter by specific tag values
gpt4_runs = [log for log in logs if "UNIFY_MODEL=gpt-4o" in log.get("tags", [])]
```

The full `settings` dict is also available for variables not passed via `--env` (e.g., values from `.env` files).

Or use the Unify dashboard to filter by `tags` (exact match) or `settings.UNIFY_MODEL` (for all values).

### Combining with Other Features

Grid search composes with all `parallel_run.sh` features:

```bash
# Grid + eval-only + repeat for statistical sampling
./grid_search.sh \
  --env UNIFY_MODEL="gpt-4o@openai|claude-sonnet-4-20250514@anthropic" \
  --env UNIFY_CACHE="false" \
  --eval-only \
  --repeat 5 \
  tests/test_contact_manager/test_ask.py
```

This generates 2 models × 5 repeats = 10 runs, useful for comparing pass rates across models.

---

## Resource Monitor Dashboard (`monitor_resources.sh`)

When running parallel tests with heavy network I/O, it's useful to monitor system resources to understand bottlenecks, detect connection leaks, and ensure you're not hitting OS limits.

### Quick Start

```bash
# Make executable (first time only)
chmod +x tests/monitor_resources.sh

# Launch the dashboard
tests/monitor_resources.sh

# Or add an alias to ~/.zshrc
alias monitor_resources='~/unity/tests/monitor_resources.sh'
```

This launches a tmux-based dashboard with four panes:

```
┌──────────────────────────────────────────────┐
│                    htop                      │
│          (CPU, Memory, Processes)            │
├──────────────────────────────────────────────┤
│                   nettop                     │
│         (Per-process Network I/O)            │
├──────────────────────┬───────────────────────┤
│   File Descriptors   │   TCP Connections     │
│  (Python processes)  │   (Active sockets)    │
└──────────────────────┴───────────────────────┘
```

### What Each Pane Shows

| Pane | Tool | What to Watch |
|------|------|---------------|
| **Top** | `htop` | CPU per core, memory usage, process list. Moderate CPU (20-50%) is normal for async tests. |
| **Middle** | `nettop` | Per-process network I/O (bytes in/out). Look for Python processes with high bandwidth. |
| **Bottom Left** | File Descriptors | Count of open FDs for Python processes. Each TCP connection = 1 FD. Watch for growth approaching `ulimit`. |
| **Bottom Right** | TCP Connections | ESTABLISHED connections (active), TIME_WAIT (closing), LISTEN (servers). High TIME_WAIT is normal after connection bursts. |

### Interpreting Metrics During Test Runs

**CPU:**
- Expect 20-50% usage from async event loops and SSL handshakes
- High CPU with low network I/O = potential bottleneck (not our tests, which are network-bound)

**Memory:**
- Watch for continuous growth during long test runs (potential memory leak)
- Cached memory is fine—the kernel releases it when needed

**File Descriptors:**
- Default macOS limit is ~256 per process
- If approaching limit, you'll see connection failures
- Fix: run `ulimit -n 4096` before `parallel_run.sh`

**TIME_WAIT Connections:**
- Normal after connections close (lasts ~60s on macOS)
- Very high counts indicate excessive connection churn
- Not usually a problem unless >1000s

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl-b + arrow` | Move between panes |
| `Ctrl-b + z` | Zoom current pane (toggle fullscreen) |
| `Ctrl-b + d` | Detach from session (keeps running in background) |
| `Ctrl-c` | Stop the current pane's command |

### Managing the Dashboard

```bash
# Re-attach to a running dashboard
tmux attach -t unity-monitor

# Kill the dashboard
tmux kill-session -t unity-monitor

# Check if dashboard is running
tmux has-session -t unity-monitor && echo "Running"
```

### Pre-Test Tuning (Heavy Parallelism)

Before running many parallel tests:

```bash
# Increase file descriptor limit (resets on terminal close)
ulimit -n 4096

# Check current limit
ulimit -n
```

For extreme parallelism (hundreds of concurrent connections):

```bash
# macOS kernel tuning (requires sudo, resets on reboot)
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=65536
```

### Requirements

- **tmux**: `brew install tmux` (required)
- **htop**: `brew install htop` (recommended, falls back to `top`)
- **nettop**: Built into macOS (no install needed)

---

## Cleanup Unify Test Projects

Use the cleanup helper to delete test projects from the Unify backend. By default, it deletes **both** the shared `UnityTests` project and any random `UnityTests_*` projects:

```bash
# first time only, ensure it's executable
chmod +x tests/project_cleanup.sh

# show what would be deleted (no changes), prompt env if needed
tests/project_cleanup.sh --dry-run

# delete all test projects (shared + random) interactively
tests/project_cleanup.sh

# delete without prompts
tests/project_cleanup.sh -y

# only delete random projects (UnityTests_*), keep the shared one
tests/project_cleanup.sh --random-only

# only delete the shared project (UnityTests), keep random ones
tests/project_cleanup.sh --shared-only

# force environment without prompt
tests/project_cleanup.sh -s   # staging
tests/project_cleanup.sh -p   # production
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Show matching projects without deleting |
| `-y`, `--yes` | Do not prompt for confirmation |
| `--shared-only` | Only delete the shared `UnityTests` project |
| `--random-only` | Only delete random `UnityTests_*` projects |
| `--prefix PREFIX` | Override prefix for random projects (default: `UnityTests_`) |
| `-s`, `--staging` | Use staging environment |
| `-p`, `--production` | Use production environment |

Requirements:

- `UNIFY_KEY` must be set in your environment (you can place it in `~/unity/.env` which is auto-sourced by the script)
- `jq` and `curl` must be installed
- To skip the environment prompt, either pass `-s/--staging` or `-p/--production`,
  or set `UNIFY_BASE_URL` (e.g., `https://api.unify.ai/v0` for production or
  `https://orchestra-staging-lz5fmz6i7q-ew.a.run.app/v0` for staging).
