# Tests

This directory contains the test suite for Unity.

## Quick Start

```bash
# Shell setup (add to ~/.zshrc for permanent aliases)
source ~/unity/tests/shell_init.zsh

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
ls .pytest_logs/*/             # Find the run directory
cat .pytest_logs/2025-12-05T14-30-22_unity_dev_ttys042/test_contact_manager-test_ask.txt
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
- **Python virtualenv**: Assumed at `~/unity/.venv/`
- **Environment**: Optional `.env` file at `~/unity/.env` for `UNIFY_KEY`, etc.
