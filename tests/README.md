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
- To re-evaluate LLM behavior, delete the relevant `.cache.ndjson`, set `UNIFY_CACHE="false"`, or use `--no-cache` with the parallel runner

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
./.parallel_run.sh --eval-only tests

# Run only symbolic tests (infrastructure/deterministic)
./.parallel_run.sh --symbolic-only tests

# Standard pytest also works
pytest -m eval tests/
pytest -m "not eval" tests/
```

---

## Running Tests

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

2. **`.parallel_run.sh`** (better debugging experience—see below)

---

## Parallel Test Runner (`.parallel_run.sh`)

This helper script launches one tmux session per test file (or per test function with `-t`) and runs `pytest` in its own window. It searches recursively and can be restricted to specific folders, files, or tests.

### Why not just pytest-xdist?

pytest-xdist works fine for basic parallel execution. However, `.parallel_run.sh` provides a significantly better **debugging experience** for our LLM-heavy async tests:

| Feature | `.parallel_run.sh` | pytest-xdist |
|---------|-------------------|--------------|
| **Interactive debugging** | `tmux attach -t <session>` to any running/failed test | Output multiplexed across workers; hard to isolate |
| **Post-failure inspection** | Failed sessions stay open with full scrollback | Just a failure message in terminal |
| **Visual status** | Real-time `? ⏳` / `o ✅` / `x ❌` per test file | Single progress bar |
| **Log isolation** | Per-session files in `.pytest_logs/` | Merged output (requires extra config) |
| **Load balancing** | Static (1 session = 1 target) | Dynamic redistribution |

**When tmux shines:** Our tests involve complex async LLM tool loops with steering, pausing, resuming, and interjections. When something fails, you need the complete context—the LLM I/O, the async flow, the interleaved logs. Being able to `tmux attach` to a failing test, scroll through its full history, and even interact with it is invaluable.

**When to use xdist instead:** For quick parallel runs where you don't need debugging (`pytest -n auto`), or when dynamic load balancing matters (tests with highly variable durations).

**TL;DR:** This script prioritizes **developer experience** over raw parallelization efficiency. Both approaches achieve parallelism; this one makes debugging failures much easier.

### Shared Project Mode (Default)

By default, `.parallel_run.sh` uses a **shared project mode** where all parallel test sessions log to the same `UnityTests` project. This enables:

- **Unified duration logging**: All test durations and LLM I/O are recorded in a single `Combined` context, making it easy to compare runtimes and review LLM calls across different test files.
- **Race-free parallel execution**: The script automatically runs an internal prepare module (`_prepare_shared_project.py`) before spawning sessions. This module idempotently creates the shared project and contexts once, eliminating race conditions.
- **Faster startup**: Sessions skip redundant project/context creation since it's already done.

When a session starts in shared mode, it executes roughly:

```bash
export UNIFY_SKIP_SESSION_SETUP=True
source ~/unity/.venv/bin/activate
pytest <target>
```

### Random Projects Mode (Legacy)

For backward compatibility or isolation purposes, you can use the `--random-projects` flag to give each tmux session its own isolated project:

```bash
./.parallel_run.sh --random-projects tests
```

In this mode, each session gets a unique project like `UnityTests_aB3xY9zQ` which is deleted on exit:

```bash
export UNIFY_TESTS_RAND_PROJ=True
export UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True
source ~/unity/.venv/bin/activate
pytest <target>
```

### Live Status and Auto-Close

- **Status prefix**: Each tmux session name is prefixed with a typeable marker and emoji: `? ⏳` while the test runs, `o ✅` on success, or `x ❌` on failure. This makes it easy to tab-complete names in shells like zsh.
- **Auto-close on success**: Sessions that pass are automatically killed about 10 seconds after completion. Failing sessions remain open for inspection.
- You can still attach before auto-close; you'll see the final message (e.g., `pytest exited with code: 0`) and a short notice that auto-close is scheduled.

### Installation

Save the script at the repository root as a hidden file and make it executable:

```bash
chmod +x .parallel_run.sh
```

### Requirements

- **tmux** and **pytest** must be installed (e.g., `brew install tmux`).
- **Virtualenv** is assumed to live at `~/unity/.venv/`. If yours differs, update the `source ~/unity/.venv/bin/activate` line inside the script.
- Optional: create an `.env` file at the repository root (i.e., `~/unity/.env`). Both helper scripts will auto-load it if present via `tests/../.env`.

### Basic Usage

From the repository root, run:

```bash
./.parallel_run.sh
```

What happens:

1. **Prepare**: The shared `UnityTests` project and `Combined` context are created (if not already present).
2. **Discovery**: Recursively finds all `test_*.py` files (excluding caches/venvs; see excludes below).
3. **Sessions**: Creates one tmux session per file.
4. **Window name**: The file's basename without `.py`.
5. **Session name**: Status-prefixed and derived from the file path, e.g., `tests/unit/test_math.py` → `? ⏳ unit-test_math` (then `o ✅ unit-test_math` or `x ❌ unit-test_math`).

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
./.parallel_run.sh tests/integration

# Multiple roots
./.parallel_run.sh tests/unit tests/integration

# Specific files
./.parallel_run.sh tests/foo_test.py tests/bar_test.py

# Specific tests (pytest node ids)
./.parallel_run.sh tests/foo_test.py::TestClass::test_something tests/bar_test.py::test_case

# Per-test mode (create a session per test for all inputs)
./.parallel_run.sh -t                         # per-test across the whole repo
./.parallel_run.sh -t tests                   # per-test across a folder
./.parallel_run.sh -t tests/foo_test.py       # per-test across a single file
./.parallel_run.sh -t tests tests/foo_test.py # mix folders and files, all per-test

# Mix files and directories
./.parallel_run.sh tests/api tests/db/test_migrations.py

# Wait for completion and log to files (CI / Agent mode)
./.parallel_run.sh --wait tests/unit

# Use isolated random projects (legacy mode)
./.parallel_run.sh --random-projects tests

# Run only eval tests (end-to-end LLM reasoning tests)
./.parallel_run.sh --eval-only tests

# Run only symbolic tests (infrastructure/deterministic tests)
./.parallel_run.sh --symbolic-only tests

# Disable LLM response caching (re-evaluate LLM behavior)
./.parallel_run.sh --no-cache tests

# Combine with other options
./.parallel_run.sh --eval-only --wait tests/test_contact_manager
./.parallel_run.sh --no-cache --eval-only tests/test_contact_manager
```

How it interprets arguments:

- **Directories**: Recursed (respecting excludes) to find `*.py`.
- **Files**: Run exactly as provided (no recursion).
- **Tests**: Pytest node ids like `path/to/test_file.py::TestClass::test_case` or `path/to/test_file.py::test_case` are run exactly as provided (one session per node id).
  - If you specify individual tests, only those tests are run (one session per test).
  - When you do not specify individual tests, the script creates one session per file.
  - With `-t/--per-test`, the script collects node ids via `pytest --collect-only` and creates one session per test for every directory/file you pass (plus any explicit node ids).

### Wait Mode and Logs (`--wait`)

Use `-w/--wait` to block until all tests finish. This is useful for CI/CD pipelines or automated agents.

```bash
./.parallel_run.sh --wait tests/my_tests
```

**Behavior:**
- Blocks until all tmux sessions complete.
- If all pass, exits with code `0`.
- If any fail, exits with code `1` and lists the failed sessions.
- **Logs**: Each session writes its full pytest output to a file in `.pytest_logs/` named after the session (e.g., `.pytest_logs/unit-test_math.txt`).
- **Debugging**: When running with `--wait`, inspect these log files to diagnose failures instead of attaching to tmux sessions (though sessions remain open for inspection if they fail).

### Match Tests by Filename (Glob-Style)

Use `-m/--match` to run tests whose basenames match a simple glob pattern. The pattern is matched against the filename only (not the full path). Quote the pattern to prevent your shell from expanding it.

Examples:

```bash
# Run all "docstring" focused tests (each in its own tmux session)
./.parallel_run.sh -m "*_tool_docstring*"
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

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-w`, `--wait` | Block until all tests complete; exit 0 on success, 1 on any failure |
| `-t`, `--per-test` | Create one session per test function instead of per file |
| `-m PATTERN`, `--match PATTERN` | Only run files matching the glob pattern |
| `--random-projects` | Use isolated random project names (legacy mode) |
| `--eval-only` | Run only tests marked with `pytest.mark.eval` (end-to-end LLM tests) |
| `--symbolic-only` | Run only tests NOT marked with `pytest.mark.eval` (infrastructure tests) |
| `--no-cache` | Disable LLM response caching (`UNIFY_CACHE=false`); forces fresh LLM calls |

### Defaults & Conventions

- **Environment**:
  - If `../.env` exists relative to the `tests` directory (i.e., `~/unity/.env`), it will be sourced automatically so you can define `UNIFY_KEY`, `UNIFY_BASE_URL`, or other variables once.
  - By default, exports `UNIFY_SKIP_SESSION_SETUP=True` for shared project mode.
  - With `--random-projects`, exports `UNIFY_TESTS_RAND_PROJ=True` and `UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True`.
- **Virtualenv**: Assumes `~/unity/.venv/bin/activate`.
- **Excludes**: Skips directories: `.git`, `.hg`, `.svn`, `.venv`, `venv`, `.mypy_cache`, `.pytest_cache`, `__pycache__`, `.idea`, `.vscode`.
  - You can edit the `EXCLUDE_DIRS` array in the script to add/remove entries.
- **Names**:
  - Session: `<status-prefix> <relative-path-with-slashes-replaced-by-dashes>` (without `.py`). Example: `? ⏳ unit-test_math` → `o ✅ unit-test_math` or `x ❌ unit-test_math`.
  - Window: `<filename-without-.py>`.
  - If a session name already exists, the script appends `-2`, `-3`, … to avoid collisions.

### Tips

- **Watch session statuses live**:

  ```bash
  watch -n 0.5 'tmux ls'
  ```

  As tests start, sessions show a `? ⏳` prefix. They flip to `o ✅` or `x ❌` when pytest exits. Successful sessions auto-close ~10s later.

- **Kill a session** once a test finishes:

  ```bash
  tmux kill-session -t <session-name>
  ```

  Note: sessions that pass auto-close within ~10 seconds; you typically only need to kill failing sessions.

- **Run in the background** (script exits immediately; sessions keep running):

  ```bash
  nohup ./.parallel_run.sh tests &>/dev/null &
  ```

- **See test output later**: just `tmux attach -t <session-name>` — pytest output stays in the window buffer.

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
    chmod +x .parallel_run.sh
    ```

### Customization

Open `.parallel_run.sh` and tweak as needed:

- **`EXCLUDE_DIRS=( ... )`** — add/remove directories to skip.
- **`run_cmd()`** — change the command chain (e.g., add flags: `pytest -q -x`).
- **Session naming** — adjust `session_basename_for()` to your taste.

### Quick Reference (tmux)

- Next/prev session (inside tmux):
  - Open the command prompt: `Ctrl-b :`
  - Type: `switch-client -n` (next) / `switch-client -p` (prev)
- List sessions: `tmux ls`
- Attach: `tmux attach -t <name>`
- Switch (inside tmux): `tmux switch-client -t <name>`
- Kill: `tmux kill-session -t <name>`

That's it! Run it, list sessions, and jump into whichever test you want to watch.

---

## Cleanup Unify Test Projects

Use the cleanup helper to delete test projects from the Unify backend. By default, it deletes **both** the shared `UnityTests` project and any random `UnityTests_*` projects:

```bash
# first time only, ensure it's executable
chmod +x tests/.project_cleanup.sh

# show what would be deleted (no changes), prompt env if needed
tests/.project_cleanup.sh --dry-run

# delete all test projects (shared + random) interactively
tests/.project_cleanup.sh

# delete without prompts
tests/.project_cleanup.sh -y

# only delete random projects (UnityTests_*), keep the shared one
tests/.project_cleanup.sh --random-only

# only delete the shared project (UnityTests), keep random ones
tests/.project_cleanup.sh --shared-only

# force environment without prompt
tests/.project_cleanup.sh -s   # staging
tests/.project_cleanup.sh -p   # production
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
