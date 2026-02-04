# Tests

This directory contains the test suite for Unity.

## Table of Contents

- [Quick Start](#quick-start)
- [Tools at a Glance](#tools-at-a-glance)
- [Test Philosophy](#test-philosophy-symbolic--eval-spectrum)
- [Parallel Runner Reference](#parallel-runner-reference)
- [Common Workflows](#common-workflows)
- [Cloud Test Runs (GitHub Actions)](#cloud-test-runs-github-actions)
- [Worktree Support](#worktree-support-cursor-background-agents)
- [Project Cleanup](#project-cleanup)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Detailed Documentation](#detailed-documentation)

---

## Quick Start

```bash
# Run all tests in parallel (blocks until completion)
tests/parallel_run.sh tests/

# Run a specific folder
tests/parallel_run.sh tests/contact_manager/

# Run with a timeout (useful for CI)
tests/parallel_run.sh --timeout 300 tests/

# Run on CI instead (no local CPU load)
git commit -m "Fix bug [run-tests]"
```

**Optional shell aliases** (for convenience):

```bash
# Add to ~/.zshrc for permanent aliases
source /path/to/unity/tests/shell_init.zsh

# Then use shorter commands
parallel_run tests/
watch_tests
kill_failed
```

---

## Tools at a Glance

| Command | Purpose |
|---------|---------|
| `parallel_run <tests>` | Run tests in parallel tmux sessions (local) |
| `parallel_cloud_run.sh <tests>` | Run tests on GitHub Actions CI |
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

## Parallel Runner Reference

The script always blocks until all tests complete, streaming pass/fail results inline.

```bash
parallel_run [options] <targets>

# Targeting
parallel_run tests/                              # Directory
parallel_run tests/foo.py                   # File
parallel_run tests/foo.py::test_bar         # Specific test

# Common flags
parallel_run --timeout 300 tests/                # Abort after 5 minutes
parallel_run -s tests/                           # Serial (per-file, not per-test)
parallel_run -j 8 tests/                         # Limit to 8 concurrent
parallel_run --eval-only tests/                  # Only eval tests
parallel_run --symbolic-only tests/              # Only symbolic tests
parallel_run --env KEY=VALUE tests/              # Set environment variable
parallel_run --repeat 5 tests/                   # Run each test 5 times
parallel_run --overwrite-scenarios tests/        # Delete and recreate test scenarios

# Pass extra args directly to pytest (after --)
parallel_run tests/ -- -v --tb=short            # Verbose with short tracebacks
parallel_run tests/ -- --pdb                    # Drop into debugger on failure
parallel_run tests/ -- --lf                     # Re-run last failed tests
```

| Flag | Description |
|------|-------------|
| `-t`, `--timeout N` | Abort if tests don't complete within N seconds |
| `-s`, `--serial` | One session per file (default: one per test) |
| `-j N`, `--jobs N` | Limit concurrent sessions (default: 25) |
| `--eval-only` | Only `@pytest.mark.eval` tests |
| `--symbolic-only` | Only non-eval tests |
| `--env K=V` | Set environment variable (repeatable) |
| `--repeat N` | Run each test N times |
| `--tags TAG` | Tag runs for filtering |
| `--overwrite-scenarios` | Delete and recreate test scenarios |
| `--` | Pass remaining args to pytest |

See [Parallel Runner Guide](docs/parallel-runner.md) for full documentation.

---

## Common Workflows

### Run tests and watch progress

```bash
# Terminal 1: Run tests
parallel_run tests/contact_manager/

# Terminal 2: Watch (optional - inline feedback is shown by default)
watch_tests
```

### Debug a failing test

```bash
# Find failing sessions
watch_tests                    # Look for f ❌ prefix

# Attach to see full output
attach 'f ❌ contact_manager-test_ask'

# Or check the log file
ls logs/pytest/*/             # Find the run directory
cat logs/pytest/2025-12-05T14-30-22_unity_dev_ttys042/contact_manager-test_ask.txt
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
parallel_run --overwrite-scenarios tests/contact_manager
```

Use this when scenario seed data has changed (e.g., new contacts, updated transcript exchanges) and you need to regenerate the cached scenario state.

---

## Cloud Test Runs (GitHub Actions)

For surgical test runs without straining your local machine, use GitHub Actions:

- **No local CPU load** — tests run on GitHub's infrastructure
- **No rate limiting** — GitHub runners have excellent network connectivity
- **24 parallel jobs** — one per test folder, all running simultaneously
- **Full `parallel_run.sh` support** — same flags work in CI as locally

### Quick Cloud Run (`parallel_cloud_run.sh`)

The fastest way to run CI tests on your current code—even uncommitted changes:

```bash
# Test current code state (handles uncommitted/unpushed automatically)
parallel_cloud_run.sh tests/contact_manager

# Run all tests
parallel_cloud_run.sh .

# Override a setting from .env
parallel_cloud_run.sh --env UNIFY_CACHE=false tests/
```

The script automatically:
1. Loads your `.env` file and passes it securely to CI (sensitive values masked in logs)
2. Stashes uncommitted changes
3. Pushes to a unique staging branch (`ci-staging-{user}-{datetime}`)
4. Triggers the CI workflow and displays the direct run URL
5. Restores your local state (staged/unstaged preserved)

See [Cloud Runner Guide](docs/parallel-cloud-run.md) for details.

### Triggering Tests

Tests are **off by default** to avoid unnecessary CI costs. Trigger them explicitly:

| Method | How to Trigger | What Runs |
|--------|----------------|-----------|
| **`parallel_cloud_run.sh`** | Run script locally | Current code state (auto-pushes staging branch) |
| **`[run-tests]`** | Include in commit message or PR title | All 24 test folders (parallel workers) |
| **`[parallel_run.sh ...]`** | Include in commit message or PR title | Specified paths/args (single worker) |
| **Manual** | GitHub Actions UI or `gh` CLI | Configurable via inputs |

**Examples:**

```bash
# Run ALL tests (24 parallel workers)
git commit -m "Fix contact manager bug [run-tests]"

# Run specific folder (single worker)
git commit -m "Fix contact manager bug [parallel_run.sh tests/contact_manager]"

# Run multiple folders (single worker, both run concurrently inside)
git commit -m "Fix bugs [parallel_run.sh tests/contact_manager tests/transcript_manager]"

# Run with extra args (single worker)
git commit -m "Eval check [parallel_run.sh --eval-only tests/actor]"

# Run specific test file
git commit -m "Fix test [parallel_run.sh tests/actor/code_act.py]"

# Regular commit (no tests)
git commit -m "Update documentation"
```

The `[parallel_run.sh ...]` syntax accepts the same arguments as the local script—paths, flags, everything. Both `tests/foo` and `test_foo` work (paths are resolved relative to the `tests/` directory).

### CLI Trigger (`gh`)

The fastest way to trigger CI tests without commits or the web UI:

```bash
# Install GitHub CLI (one-time)
brew install gh
gh auth login

# Run all tests
gh workflow run tests.yml --repo unifyai/unity --ref main

# Run specific folder
gh workflow run tests.yml --repo unifyai/unity --ref main \
  -f test_path="tests/actor"

# Run with extra args
gh workflow run tests.yml --repo unifyai/unity --ref main \
  -f test_path="tests/actor" \
  -f parallel_run_args="--eval-only"

# Run on a different branch
gh workflow run tests.yml --repo unifyai/unity --ref my-feature-branch \
  -f test_path="tests/contact_manager"
```

**Available inputs:**

| Flag | Description |
|------|-------------|
| `-f test_path="..."` | Path to test folder/file (default: `.` for all) |
| `-f parallel_run_args="..."` | Extra args (see [Parallel Runner Reference](#parallel-runner-reference)) |
| `-f timeout_minutes="N"` | Timeout in minutes (default: 120) |
| `--ref <branch>` | Branch to run tests on |

**Watch the run:**

```bash
gh run list --repo unifyai/unity --workflow tests.yml
gh run watch --repo unifyai/unity <run-id>
gh run view --repo unifyai/unity <run-id> --log
```

### Manual Workflow Dispatch (UI)

For maximum control, use the GitHub Actions UI:

1. Go to **Actions** → **"Tests"**
2. Click **"Run workflow"** dropdown
3. Select your branch and configure inputs

| Input | Default | Description |
|-------|---------|-------------|
| `test_path` | `.` (all) | Path to test folder, file, or specific test |
| `parallel_run_args` | *(empty)* | Extra args (see [Parallel Runner Reference](#parallel-runner-reference)) |
| `timeout_minutes` | 120 | `parallel_run.sh` timeout (minutes) |

### Accessing Test Logs

After a CI run, logs are available in the GitHub Actions UI:

| Artifact | Contents |
|----------|----------|
| `pytest-logs-{folder}` | Test output logs for each folder |
| `llm-io-debug-{folder}` | LLM request/response traces for each folder |
| `cache-diff-{run}-{folder}` | Cache delta files (used internally) |

**Inline Failure Summaries**: Failed jobs display collapsible failure excerpts directly in the Summary page—no download required for quick triage.

---

## Worktree Support (Cursor Background Agents)

All test commands **automatically detect the current git repository** and use that repo's scripts. This means:

- ✅ Commands work correctly in **git worktrees** (e.g., Cursor Background Agents)
- ✅ Tests run against the **current repo's code**, not a hardcoded path
- ✅ Logs appear in the **current repo's** `logs/pytest/` directory
- ✅ No manual path adjustments needed
- ✅ **Concurrent worktree tests don't interfere** with each other's orchestra

**How it works:** When you run `parallel_run`, the shell function checks `git rev-parse --show-toplevel` to find the current repo root, then uses that repo's `tests/parallel_run.sh`. If you're not in a git repo, it falls back to the originally configured path.

**Orchestra and shared logs:** Since local orchestra is a shared server (one instance for all worktrees), its logs go to a single location. When running from a worktree, `parallel_run.sh` creates symlinks:
- `logs/orchestra/` → main repo's `logs/orchestra/`
- `logs/all/` → main repo's `logs/all/` (for OTEL trace correlation)

This means concurrent tests from different worktrees can run without restarting orchestra, while still having all logs accessible from each worktree's `logs/` directory.

### Browsing All Worktree Logs from Main Repo

When tests run from a worktree (via any method - `parallel_run`, direct `pytest`, etc.), **symlinks are automatically created** in the main repo's log directories pointing to each worktree's logs:

```
/Users/you/unity/logs/pytest/
├── 2025-12-05T14-30-45_unity_dev_ttys042/   # main repo's own logs
├── worktree-oty/  →  ~/.cursor/worktrees/unity/oty/logs/pytest/
├── worktree-xyz/  →  ~/.cursor/worktrees/unity/xyz/logs/pytest/
└── ...

/Users/you/unity/logs/unillm/
├── 2025-12-05T14-30-45_unity_dev_ttys042/    # main repo's logs (terminal A)
├── worktree-oty/  →  ~/.cursor/worktrees/unity/oty/logs/unillm/
└── ...
```

This lets you browse **all logs from all worktrees** in one place (the main repo), while each worktree still maintains its own isolated log directories.

**Note:** Symlinks are created by `conftest.py` during pytest session start, so they work regardless of how pytest was invoked.

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

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `tmux: command not found` | `brew install tmux` |
| High resource usage after tests | `kill_server --global` |
| "error connecting to ... (No such file or directory)" | Socket was deleted; re-run tests |
| Tests not found | Check that path exists and isn't in `EXCLUDE_DIRS` |
| Permission denied | `chmod +x tests/*.sh` |

---

## Requirements

- **tmux**: `brew install tmux`
- **coreutils** (macOS): `brew install coreutils` — provides `timeout` for helper scripts
- **Python virtualenv**: Repo-local `.venv/` (create/sync via `uv sync --all-groups`)
- **Environment**: Optional `.env` file at repo root (`.env`) for `UNIFY_KEY`, etc.

---

## Detailed Documentation

- **[Parallel Runner](docs/parallel-runner.md)** — Full guide to `parallel_run`, tmux isolation, flags, and troubleshooting
- **[Cloud Runner](docs/parallel-cloud-run.md)** — Trigger CI tests on current code state (handles uncommitted changes)
- **[Grid Search](docs/grid-search.md)** — Running tests across setting combinations for model comparisons and ablations
- **[Resource Monitor](docs/resource-monitor.md)** — Dashboard for monitoring CPU, memory, network, and file descriptors
- **[Logging & Data](docs/logging.md)** — Log directory structure, remote telemetry, and analyzing test data
