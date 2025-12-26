# Tests

This directory contains the test suite for Unity.

## Quick Start

```bash
# Shell setup (add to ~/.zshrc for permanent aliases)
source /path/to/your/unity/clone/tests/shell_init.zsh

# Run all tests in parallel
parallel_run tests/

# Watch progress
watch_tests

# Attach to a failing test
attach 'f ❌ test_contact_manager-test_ask'

# Clean up
kill_failed        # Kill failed sessions
kill_server        # Kill tmux server + orphaned processes
```

---

## Worktree Support (Cursor Background Agents)

All test commands **automatically detect the current git repository** and use that repo's scripts. This means:

- ✅ Commands work correctly in **git worktrees** (e.g., Cursor Background Agents)
- ✅ Tests run against the **current repo's code**, not a hardcoded path
- ✅ Logs appear in the **current repo's** `pytest_logs/` directory
- ✅ No manual path adjustments needed

**How it works:** When you run `parallel_run`, the shell function checks `git rev-parse --show-toplevel` to find the current repo root, then uses that repo's `tests/parallel_run.sh`. If you're not in a git repo, it falls back to the originally configured path.

### Browsing All Worktree Logs from Main Repo

When tests run from a worktree (via any method - `parallel_run`, direct `pytest`, etc.), **symlinks are automatically created** in the main repo's log directories pointing to each worktree's logs:

```
/Users/you/unity/pytest_logs/
├── 2025-12-05T14-30-45_unity_dev_ttys042/   # main repo's own logs
├── worktree-oty/  →  ~/.cursor/worktrees/unity/oty/pytest_logs/
├── worktree-xyz/  →  ~/.cursor/worktrees/unity/xyz/pytest_logs/
└── ...

/Users/you/unity/llm_io_debug/
├── 2025-12-05T14-30-45_unity_dev_ttys042/    # main repo's logs (terminal A)
├── worktree-oty/  →  ~/.cursor/worktrees/unity/oty/llm_io_debug/
└── ...
```

This lets you browse **all logs from all worktrees** in one place (the main repo), while each worktree still maintains its own isolated log directories.

**Note:** Symlinks are created by `conftest.py` during pytest session start, so they work regardless of how pytest was invoked

---

## Test Philosophy: Symbolic ↔ Eval Spectrum

Tests fall on a spectrum between two paradigms:

**Symbolic Tests** use the LLM purely as a stub—minimal "dummy" instructions trigger specific code paths. Focus is on testing *infrastructure*: async tool loops, steering, state mutations. Failures indicate regressions in symbolic/programmatic logic.

**Eval Tests** exercise the system end-to-end. We ask a high-level question, then verify the outcome—regardless of internal tool calls. Focus is on *capability*: "Did the assistant complete the task?" Failures may indicate prompt issues or capability gaps.

Most tests sit somewhere between these extremes.

### Caching and Determinism

When `UNIFY_CACHE="true"` (the default), all LLM responses are cached:
- **First run**: LLM executes normally; responses stored in `.cache.ndjson`
- **Subsequent runs**: Cached responses replayed—no actual LLM calls

Both test types become deterministic after caching. To re-evaluate LLM behavior:
```bash
parallel_run --env UNIFY_CACHE=false tests
```

### Marking Tests as Eval

```python
import pytest

pytestmark = pytest.mark.eval  # All tests in file are eval

# Or per-test:
@pytest.mark.eval
async def test_natural_language_query():
    ...
```

### Running by Category

```bash
parallel_run --eval-only tests       # Only eval tests
parallel_run --symbolic-only tests   # Only symbolic tests
```

---

## Tools at a Glance

| Command | Purpose |
|---------|---------|
| `parallel_run <tests>` | Run tests in parallel tmux sessions |
| `watch_tests` | Monitor test progress in real-time |
| `attach '<name>'` | Attach to a tmux session |
| `list_runs` | List all active test runs across terminals |
| `kill_failed` | Kill all failed sessions |
| `kill_server` | Kill tmux server + purge orphaned processes |
| `monitor_resources` | Launch resource monitoring dashboard |
| `grid_search.sh` | Run tests across setting combinations |
| `project_cleanup.sh` | Delete test projects from Unify backend |

All commands support `--help` for usage details.

---

## Common Workflows

### Run tests and watch progress

```bash
# Terminal 1: Run tests
parallel_run tests/test_contact_manager/

# Terminal 2: Watch (optional - inline feedback is shown by default)
watch_tests
```

### Debug a failing test

```bash
# Find failing sessions
watch_tests                    # Look for f ❌ prefix

# Attach to see full output
attach 'f ❌ test_contact_manager-test_ask'

# Or check the log file
ls pytest_logs/*/             # Find the run directory
cat pytest_logs/2025-12-05T14-30-22_unity_dev_ttys042/test_contact_manager-test_ask.txt
```

### Clean up after tests

```bash
kill_failed           # Kill failed sessions (keep passing ones)
kill_server           # Kill this terminal's tmux server
kill_server --all     # Kill all unity* tmux servers
kill_server --global  # Kill ALL tmux servers
```

### Run with different settings

```bash
# Disable caching for fresh LLM calls
parallel_run --env UNIFY_CACHE=false tests

# Use isolated projects per test
parallel_run --env UNIFY_TESTS_RAND_PROJ=true tests

# Compare models (grid search)
grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" tests/
```

### Overwrite test scenarios

Some test suites (ContactManager, TranscriptManager, etc.) use pre-seeded scenario data that persists between runs for speed. To delete and recreate these scenarios from scratch:

```bash
parallel_run --overwrite-scenarios tests/test_contact_manager
```

Use this when scenario seed data has changed (e.g., new contacts, updated transcript exchanges) and you need to regenerate the cached scenario state.

### Pass extra args to pytest

Use `--` to pass any additional arguments directly to pytest:

```bash
# Verbose output with short tracebacks
parallel_run tests/ -- -v --tb=short

# Drop into debugger on first failure
parallel_run tests/ -- --pdb -x

# Re-run only last failed tests
parallel_run tests/ -- --lf

# Combine with parallel_run flags
parallel_run -w --overwrite-scenarios tests/test_contact_manager -- -v
```

Any pytest option or custom conftest option (like `--unify-stub`) can be passed this way.

---

## Cloud Test Runs (GitHub Actions)

For surgical test runs without straining your local machine, use GitHub Actions:

- **No local CPU load** — tests run on GitHub's infrastructure
- **No rate limiting** — GitHub runners have excellent network connectivity
- **24 parallel jobs** — one per test folder, all running simultaneously
- **Full `parallel_run.sh` support** — same flags work in CI as locally

### Triggering Tests

Tests are **off by default** to avoid unnecessary CI costs. Trigger them explicitly:

| Method | How to Trigger | What Runs |
|--------|----------------|-----------|
| **`[run-tests]`** | Include in commit message or PR title | All 24 test folders (parallel workers) |
| **`[parallel_run.sh ...]`** | Include in commit message or PR title | Specified paths/args (single worker) |
| **Manual** | GitHub Actions UI → "Run workflow" | Configurable via inputs |

**Examples:**

```bash
# Run ALL tests (24 parallel workers)
git commit -m "Fix contact manager bug [run-tests]"

# Run specific folder (single worker)
git commit -m "Fix contact manager bug [parallel_run.sh tests/test_contact_manager]"

# Run multiple folders (single worker, both run concurrently inside)
git commit -m "Fix bugs [parallel_run.sh tests/test_contact_manager tests/test_transcript_manager]"

# Run with extra args (single worker)
git commit -m "Eval check [parallel_run.sh --eval-only tests/test_actor]"

# Run specific test file
git commit -m "Fix test [parallel_run.sh tests/test_actor/test_code_act.py]"

# Regular commit (no tests)
git commit -m "Update documentation"
```

The `[parallel_run.sh ...]` syntax accepts the same arguments as the local script—paths, flags, everything. Both `tests/test_foo` and `test_foo` work (paths are resolved relative to the `tests/` directory).

### Manual Workflow Dispatch

For maximum control, use the GitHub Actions UI:

1. Go to **Actions** → **"Testing Unity with uv"**
2. Click **"Run workflow"** dropdown
3. Select your branch and configure inputs:

| Input | Default | Description |
|-------|---------|-------------|
| `test_path` | `.` (all) | Path to test folder, file, or specific test |
| `parallel_run_args` | *(empty)* | Extra args passed to `parallel_run.sh` |
| `timeout_minutes` | 120 | `parallel_run.sh` timeout (minutes) |

**Flexible Test Targeting:**

| Input Value | What Runs |
|-------------|-----------|
| *(blank or `.`)* | All 24 test folders in parallel |
| `tests/test_actor` | Only the `test_actor` folder |
| `tests/test_actor/test_code_act.py` | Only that specific file |
| `tests/test_actor/test_code_act.py::test_name` | Only that specific test |

**Advanced Options (`parallel_run_args`):**

| Flag | Example | Description |
|------|---------|-------------|
| `--eval-only` | `--eval-only` | Only `@pytest.mark.eval` tests |
| `--symbolic-only` | `--symbolic-only` | Only non-eval tests |
| `--repeat N` | `--repeat 5` | Run each test N times |
| `-s` | `-s` | Serial mode (one session per file) |
| `--tags` | `--tags exp-1` | Tag runs for filtering |
| `-j N` | `-j 10` | Limit concurrent sessions |
| `--env K=V` | `--env UNIFY_CACHE=false` | Set environment variable |

### Accessing Test Logs

After a CI run, logs are available in the GitHub Actions UI:

| Artifact | Contents |
|----------|----------|
| `all-logs-consolidated` | **One-click download** of all logs combined |
| `pytest-logs-{folder}` | Individual folder's pytest output |
| `llm-io-debug-{folder}` | Individual folder's LLM I/O traces |

**Inline Failure Summaries**: Failed jobs display collapsible failure excerpts directly in the Summary page—no download required for quick triage.

---

## Parallel Runner Quick Reference

```bash
parallel_run [options] <targets>

# Targeting
parallel_run tests/                              # Directory
parallel_run tests/test_foo.py                   # File
parallel_run tests/test_foo.py::test_bar         # Specific test

# Common flags
parallel_run -w tests/                           # Wait for completion
parallel_run -s tests/                           # Serial (per-file, not per-test)
parallel_run -j 8 tests/                         # Limit to 8 concurrent
parallel_run --eval-only tests/                  # Only eval tests
parallel_run --env KEY=VALUE tests/              # Set environment variable
parallel_run --repeat 5 tests/                   # Run each test 5 times
parallel_run --overwrite-scenarios tests/        # Delete and recreate test scenarios

# Pass extra args directly to pytest (after --)
parallel_run tests/ -- -v --tb=short            # Verbose with short tracebacks
parallel_run tests/ -- --pdb                    # Drop into debugger on failure
parallel_run tests/ -- --lf                     # Re-run last failed tests
```

See [Parallel Runner Guide](docs/parallel-runner.md) for full documentation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `tmux: command not found` | `brew install tmux` |
| High resource usage after tests | `kill_server --global` |
| "error connecting to ... (No such file or directory)" | Socket was deleted; re-run tests |
| Tests not found | Check that path exists and isn't in `EXCLUDE_DIRS` |
| Permission denied | `chmod +x tests/*.sh` |

---

## Detailed Documentation

- **[Parallel Runner](docs/parallel-runner.md)** — Full guide to `parallel_run`, tmux isolation, flags, and troubleshooting
- **[Grid Search](docs/grid-search.md)** — Running tests across setting combinations for model comparisons and ablations
- **[Resource Monitor](docs/resource-monitor.md)** — Dashboard for monitoring CPU, memory, network, and file descriptors
- **[Logging & Data](docs/logging.md)** — Log directory structure, remote telemetry, and analyzing test data

---

## Project Cleanup

Delete test projects from the Unify backend:

```bash
# Preview what would be deleted
project_cleanup.sh --dry-run

# Delete all test projects
project_cleanup.sh -y

# Only delete random projects (keep shared UnityTests)
project_cleanup.sh --random-only
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Show matching projects without deleting |
| `-y`, `--yes` | Skip confirmation prompt |
| `--shared-only` | Only delete `UnityTests` |
| `--random-only` | Only delete `UnityTests_*` |
| `-s`, `--staging` | Use staging environment |
| `-p`, `--production` | Use production environment |

---

## Requirements

- **tmux**: `brew install tmux`
- **coreutils** (macOS): `brew install coreutils` — provides `timeout` for helper scripts
- **Python virtualenv**: Repo-local `.venv/` (create/sync via `uv sync --all-groups`)
- **Environment**: Optional `.env` file at repo root (`.env`) for `UNIFY_KEY`, etc.
