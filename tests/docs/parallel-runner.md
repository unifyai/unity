# Parallel Test Runner (`parallel_run.sh`)

This helper script launches one tmux session per test function by default (or per file with `-s/--serial`) and runs `pytest` in its own window. It searches recursively and can be restricted to specific folders, files, or tests.

> **Quick start:** `parallel_run tests/` — runs all tests in parallel tmux sessions.

## Terminal Isolation (Automatic)

Each terminal session automatically gets its own **isolated tmux server**. This means:

- **Cursor agents don't interfere with each other**: Each agent's tests run in their own isolated tmux server
- **`tmux kill-server` is safe**: It only kills sessions from the terminal that ran it
- **No configuration needed**: Isolation is automatic based on the terminal's TTY device

**How it works:** The script derives a unique socket name from your terminal's TTY (e.g., `/dev/ttys042` → socket `unity_dev_ttys042`). All tmux commands use this socket automatically.

**Monitoring your tests:**

```bash
# Watch YOUR terminal's tests (automatic isolation)
watch_tests

# Watch ALL terminals' tests
watch_tests --all

# Attach to a specific session to see its output
attach '<session-name>'
```

**Recovering orphaned runs (when you close the original terminal):**

```bash
# List all active test runs across all terminals
list_runs

# Watch tests from a specific socket (orphaned run)
watch_tests --socket unity_dev_ttys042

# Attach to a session in a specific socket (two equivalent syntaxes)
attach --socket unity_dev_ttys042 'f ❌ actor-test_code_act'
attach 'unity_dev_ttys042:f ❌ actor-test_code_act'  # shorthand

# Kill failed sessions in a specific socket
kill_failed --socket unity_dev_ttys042

# Kill a specific socket's server
kill_server --socket unity_dev_ttys042
```

**Cleanup:**

```bash
# Kill failed sessions from THIS terminal
kill_failed

# Kill failed sessions from ALL terminals
kill_failed --all

# Kill the entire tmux server for THIS terminal (+ orphaned processes)
kill_server

# Kill ALL unity* tmux servers (+ orphaned processes)
kill_server --all

# Kill ALL tmux servers for this user (+ orphaned processes)
kill_server --global
```

> **Note:** `kill_server` automatically purges orphaned pytest processes that may have been left behind from crashed test runs. This prevents silent resource exhaustion (file descriptors, memory, network connections). Use `--no-purge` to skip this if needed.

---

## Command-Line Options

The script **always blocks** until all tests complete, streaming pass/fail results inline.

| Option | Description |
|--------|-------------|
| `-t N`, `--timeout N` | Abort if tests don't complete within N seconds. |
| `-s`, `--serial` | Create one session per file instead of per test (tests within a file run serially) |
| `-j N`, `--jobs N` | Limit concurrent tmux sessions (default: 25). Use `-j 0` or `-j none` for unlimited. |
| `-m PATTERN`, `--match PATTERN` | Only run files matching the glob pattern |
| `-e KEY=VALUE`, `--env KEY=VALUE` | Set environment variable for all sessions (repeatable) |
| `--tags TAG` | Tag test runs for filtering (repeatable, comma-separated) |
| `--eval-only` | Run only tests marked with `pytest.mark.eval` |
| `--symbolic-only` | Run only tests NOT marked with `pytest.mark.eval` |
| `--repeat N` | Run each test N times; useful for statistical sampling |
| `-h`, `--help` | Show help |

---

## Serial Mode (`-s`)

By default, the script creates one tmux session per *test function* for maximum parallelism. If a file contains 15 tests, all 15 run concurrently in separate sessions.

Use `-s/--serial` to create one session per *file* instead (tests within a file run serially):

```bash
# DEFAULT: 15 tests run concurrently in 15 sessions (~1 min)
parallel_run tests/contact_manager/test_ask.py

# WITH -s: 15 tests in one file run serially (~10 min)
parallel_run -s tests/contact_manager/test_ask.py
```

**When to use `-s`:**
- Running the entire test suite (hundreds of tests) where per-file grouping helps organization
- When you prefer fewer, more manageable tmux sessions
- Debugging scenarios where you want related tests grouped together

**When to omit `-s` (default behavior):**
- Running a single test file with multiple tests
- Running a small number of specific tests
- Running a small directory (< 100 tests total)
- Anytime you want maximum speed

---

## Concurrency Limits

By default, `parallel_run` limits concurrent sessions to **25** to prevent resource exhaustion. Use `-j N` to adjust:

```bash
# Lower concurrency for resource-constrained systems
parallel_run -j 8 tests/contact_manager/

# Higher concurrency for powerful machines
parallel_run -j 100 tests/

# Unlimited (not recommended for large test suites)
parallel_run -j 0 tests/
parallel_run -j none tests/      # equivalent
parallel_run -j unlimited tests/ # equivalent
```

---

## Blocking Behavior and Logs

The script always blocks until all tests complete, streaming pass/fail results inline as tests finish.

```bash
# Run tests (blocks until all complete)
parallel_run tests/my_tests

# Run with 120 second timeout
parallel_run --timeout 120 tests/my_tests
```

**Behavior:**
- Blocks until all tmux sessions complete (or timeout is reached).
- Streams pass/fail results inline as each test finishes.
- If all pass, exits with code `0`.
- If any fail, exits with code `1` and lists the failed sessions.
- If timeout is reached before completion, exits with code `2`.
- **Logs**: Each session writes its full pytest output to `logs/pytest/{datetime}_{socket}/`.

**Timeout examples:**
```bash
# Quick sanity check with 60s timeout
parallel_run --timeout 60 tests/basic.py

# Long-running tests with 5 minute timeout
parallel_run --timeout 300 tests/slow_suite/
```

---

## Environment Variable Overrides (`--env`)

The `-e/--env KEY=VALUE` flag sets environment variables for all pytest sessions:

```bash
# Single override
parallel_run --env UNILLM_CACHE=false tests

# Multiple overrides (flag can be repeated)
parallel_run -e UNILLM_CACHE=false -e UNIFY_DELETE_CONTEXT_ON_EXIT=true tests

# Use isolated random projects (each session gets its own project)
parallel_run --env UNIFY_TESTS_RAND_PROJ=true --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```

### Available Variables

Settings are organized in two classes with inheritance:
- `ProductionSettings` (`unity/settings.py`) - used in deployed system AND tests
- `TestingSettings` (`tests/settings.py`) - inherits production + adds test-only settings

**Production Settings** (also used in tests):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UNIFY_MODEL` | str | `gpt-5.2@openai` | LLM model to use |
| `UNILLM_CACHE` | bool/str | `true` | Enable/disable LLM response caching |
| `LLM_IO_DEBUG` | bool | `true` | Log full LLM request/response payloads |
| `UNITY_TERMINAL_LOG` | bool | `true` | Enable/disable terminal (console) logging |
| `UNITY_ASYNCIO_DEBUG` | bool | `false` | Enable Python asyncio debug mode |
| `PYTEST_LOG_TO_FILE` | bool | `true` | Log pytest output to files |
| `UNITY_READONLY_ASK_GUARD` | bool | `true` | Enable read-only ask guard |
| `FIRST_ASK_TOOL_IS_SEARCH` | bool | `false` | Force semantic search on first `ask` step |
| `FIRST_MUTATION_TOOL_IS_ASK` | bool | `false` | Force `ask` on first mutation step |

**Test-Only Settings**:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UNIFY_DELETE_CONTEXT_ON_EXIT` | bool | `false` | Delete test context after each test |
| `UNIFY_OVERWRITE_PROJECT` | bool | `false` | Overwrite project on activation |
| `UNIFY_TESTS_RAND_PROJ` | bool | `false` | Use random project names (isolated per session) |
| `UNIFY_TESTS_DELETE_PROJ_ON_START` | bool | `true` | Delete project before session starts (clean slate) |
| `UNIFY_TESTS_DELETE_PROJ_ON_EXIT` | bool | `false` | Delete random project when session exits |
| `UNIFY_TEST_TAGS` | str | `""` | Comma-separated tags for duration logging |
| `UNIFY_SKIP_SESSION_SETUP` | bool | `false` | Skip project/context creation (pre-done) |

---

## Match Tests by Filename (`--match`)

Use `-m/--match` to run tests whose basenames match a glob pattern:

```bash
# Run all "docstring" focused tests
parallel_run -m "*_tool_docstring*"
```

This matches files like `test_contact_tool_docstrings.py`, `test_guidance_tool_docstring.py`, etc.

---

## Statistical Sampling (`--repeat`)

The `--repeat N` flag runs each test target N times. **The primary use case is for eval tests with `UNILLM_CACHE=false`**:

```bash
# Run a specific eval test 10 times without caching
parallel_run --env UNILLM_CACHE=false --repeat 10 --eval-only tests/contact_manager/test_ask.py

# Run all eval tests 5 times each
parallel_run --env UNILLM_CACHE=false --repeat 5 --eval-only tests
```

**Use cases:**
1. **Pass rate estimation**: Run an eval test 20 times to measure reliability (e.g., "passes 18/20 = 90%")
2. **Runtime distribution**: Plot test durations across runs to understand variance
3. **Regression detection**: A test that was 100% reliable but now fails 5% of the time indicates a problem

---

## Why Not pytest-xdist?

pytest-xdist works fine for basic parallel execution. However, `parallel_run` provides a significantly better **debugging experience** for our LLM-heavy async tests:

| Feature | `parallel_run` | pytest-xdist |
|---------|----------------|--------------|
| **Interactive debugging** | `tmux attach -t <session>` to any running/failed test | Output multiplexed across workers; hard to isolate |
| **Post-failure inspection** | Failed sessions stay open with full scrollback | Just a failure message in terminal |
| **Visual status** | Real-time `p ✅` / `f ❌` / `r ⏳` per test | Single progress bar |
| **Log isolation** | Per-run folders in `logs/pytest/` | Merged output |

**When tmux shines:** Our tests involve complex async LLM tool loops with steering, pausing, resuming, and interjections. When something fails, you need the complete context. Being able to `tmux attach` to a failing test and scroll through its full history is invaluable.

**When to use xdist instead:** For quick parallel runs where you don't need debugging (`pytest -n auto`), or when dynamic load balancing matters.

---

## Shared Project Mode (Default)

By default, all parallel test sessions log to the same `UnityTests` project. This enables:

- **Unified duration logging**: All test durations and LLM I/O are recorded in a single `Combined` context
- **Race-free parallel execution**: The script automatically runs `_prepare_shared_project.py` before spawning sessions
- **Faster startup**: Sessions skip redundant project/context creation

### Random Projects Mode

For isolation purposes, give each session its own project:

```bash
parallel_run --env UNIFY_TESTS_RAND_PROJ=true --env UNIFY_TESTS_DELETE_PROJ_ON_EXIT=true tests
```

Each session gets a unique project like `UnityTests_aB3xY9zQ` which is deleted on exit.

---

## Live Status and Auto-Close

- **Status prefix**: Each tmux session name is prefixed with `r ⏳` (running), `p ✅` (passed), or `f ❌` (failed). Letters sort alphabetically as failed→passed→running.
- **Inline pass/fail feedback**: When using a job limit, results print inline as sessions complete.
- **Auto-close on success**: Sessions that pass are automatically killed ~10 seconds after completion. Failing sessions remain open for inspection.

---

## Usage Examples

```bash
# Run all tests in a folder
parallel_run tests/integration

# Multiple roots
parallel_run tests/unit tests/integration

# Specific files
parallel_run tests/foo_test.py tests/bar_test.py

# Specific tests (pytest node ids)
parallel_run tests/foo_test.py::TestClass::test_something

# Serial mode (one session per file)
parallel_run -s tests

# With timeout (abort after 5 minutes)
parallel_run --timeout 300 tests/unit

# Run only eval tests
parallel_run --eval-only tests

# Run only symbolic tests
parallel_run --symbolic-only tests

# Tag test runs for filtering
parallel_run --tags "experiment-1" tests

# Combine options
parallel_run --eval-only tests/contact_manager
parallel_run --env UNILLM_CACHE=false --eval-only tests/contact_manager
```

---

## Troubleshooting

- **"tmux: command not found"**
  - Install tmux: `brew install tmux` (macOS) or `apt-get install tmux` (Linux)

- **Virtualenv not found / wrong Python**
  - `parallel_run.sh` uses the repo-local `.venv/` (created by `uv sync --all-groups`).
  - If `.venv/` is missing, ensure `python3` + `pip` are available, then run: `pip install uv && uv sync --all-groups`
  - If `uv` was installed with `pip --user`, ensure `~/.local/bin` is on your `PATH`.

- **No sessions created**
  - Ensure there are `.py` files under the provided paths and that excludes aren't hiding your files.

- **Permission denied**
  - Make the script executable: `chmod +x parallel_run.sh`

- **High resource usage after tests (FDs, memory, swap)**
  - Run: `kill_server --global` to kill servers and purge orphaned processes
  - Check manually: `ps aux | grep -E "unity.*pytest" | grep -v grep`
  - Use `monitor_resources` to check file descriptor counts

- **"error connecting to ... (No such file or directory)"**
  - The tmux socket file was deleted while tests were running
  - Solution: Wait for tests to complete, or re-run them

---

## Quick Reference (tmux)

Each terminal uses its own tmux socket (printed when tests launch):

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
list_runs           # List all active test runs (all sockets)
watch_tests         # Watch this terminal's tests
attach '<name>'     # Attach to a session
kill_failed         # Kill failed sessions
kill_server         # Kill server + purge orphaned processes
kill_server --all   # Kill all unity* servers + purge orphans
kill_server --global  # Kill ALL tmux servers + purge orphans
```

---

## Requirements

- **tmux** must be installed (`brew install tmux`)
- **coreutils** (recommended on macOS): Provides `timeout` command. Install with `brew install coreutils`.
- **Python environment**: `parallel_run.sh` uses the repo-local `.venv/` and will bootstrap it via `uv sync --all-groups` if needed.
- Optional: `.env` file at repo root (`.env`) for `UNIFY_KEY`, etc.

---

## Customization

Open `parallel_run.sh` and tweak:

- **`EXCLUDE_DIRS=( ... )`** — add/remove directories to skip
- **`run_cmd()`** — change the command chain (e.g., add flags: `pytest -q -x`)
- **Session naming** — adjust `session_basename_for()` to your taste
