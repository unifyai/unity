# Cloud Runner (`parallel_cloud_run.sh`)

Trigger CI tests on GitHub Actions with the current local code state—even if changes aren't committed or pushed.

## Quick Start

```bash
# Run specific tests on CI
parallel_cloud_run.sh tests/test_contact_manager

# Run all tests on CI
parallel_cloud_run.sh .

# Run multiple folders
parallel_cloud_run.sh tests/test_actor tests/test_conductor

# Override a .env setting
parallel_cloud_run.sh --env UNIFY_CACHE=false tests/
```

The script automatically:
- **Loads your `.env`** and passes all settings to CI (API keys, `UNIFY_BASE_URL`, etc.)
- **Handles uncommitted changes** by pushing to a unique staging branch
- **Displays the direct run URL** after triggering (polls until the run appears)

---

## How It Works

The script detects your local code state and chooses the optimal path:

### Clean & Pushed Branch

If your branch is clean (no uncommitted changes) and fully pushed:

```
┌─────────────────────────────────────────────┐
│  Local: main @ abc123 (pushed, clean)       │
└─────────────────────────────────────────────┘
                    │
                    ▼
        gh workflow run --ref main
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  CI runs on: main                           │
└─────────────────────────────────────────────┘
```

### Local Changes Detected

If you have uncommitted changes or unpushed commits:

```
┌─────────────────────────────────────────────┐
│  Local: feature @ def456                    │
│  - Uncommitted: file1.py (staged)           │
│  - Uncommitted: file2.py (unstaged)         │
│  - Unpushed commits: 3                      │
└─────────────────────────────────────────────┘
                    │
     1. Stash uncommitted changes
     2. Create unique staging branch
     3. Apply stash + commit
     4. Push staging branch
     5. Trigger CI on staging branch
     6. Return to feature branch
     7. Restore stash (preserving staged/unstaged)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Local: feature @ def456 (unchanged!)       │
│  - file1.py still staged                    │
│  - file2.py still unstaged                  │
│                                             │
│  CI runs on: ci-staging-your-name-2025-...  │
└─────────────────────────────────────────────┘
```

---

## The Staging Branch

When local changes exist, the script creates a **unique staging branch** named `ci-staging-{username}-{datetime}`.

**Why unique branches?**

- **Natural isolation**: Each CI run has its own branch—no interference between runs
- **Agent-friendly**: Cloud-based agents (e.g., Cursor) can checkout the exact branch to debug failures
- **Clear history**: Easy to see what code each CI run tested

**Tradeoff**: Unique branches accumulate over time. Periodically clean up stale branches:

```bash
# Delete all remote ci-staging-* branches
git branch -r | grep 'ci-staging-' | sed 's|origin/||' | xargs -I{} git push origin --delete {}

# Or delete branches older than 7 days (requires gh CLI)
gh api repos/unifyai/unity/branches --paginate -q '.[].name' | \
  grep 'ci-staging-' | xargs -I{} git push origin --delete {}
```

---

## Usage

```bash
parallel_cloud_run.sh [--env KEY=VALUE ...] [test_paths...]
```

| Argument | Description |
|----------|-------------|
| `--env KEY=VALUE` | Override an environment variable (repeatable) |
| `test_paths` | Paths to test (default: `.` for all tests) |

**Examples:**

```bash
# Single folder
parallel_cloud_run.sh tests/test_contact_manager

# Multiple folders (run concurrently in single CI job)
parallel_cloud_run.sh tests/test_actor tests/test_conductor

# Specific test file
parallel_cloud_run.sh tests/test_actor/test_code_act.py

# All tests
parallel_cloud_run.sh .
parallel_cloud_run.sh   # equivalent

# Override settings
parallel_cloud_run.sh --env UNIFY_CACHE=false tests/
parallel_cloud_run.sh --env UNIFY_CACHE=false --env UNIFY_MODEL=gpt-4o tests/
```

---

## Environment Variables

The script automatically loads your local `.env` file and passes **all** values to the CI workflow via `--env` arguments. This means CI runs use your personal settings:

- `UNIFY_KEY` — your API key
- `UNIFY_BASE_URL` — your preferred backend (staging/production)
- `UNIFY_CACHE`, `UNIFY_MODEL`, etc.

**Override order**: `.env` values are loaded first, then explicit `--env` args are appended. Later values win, so command-line overrides take precedence.

```bash
# .env has UNIFY_CACHE=true, but this overrides it to false
parallel_cloud_run.sh --env UNIFY_CACHE=false tests/
```

---

## Requirements

- **GitHub CLI (`gh`)**: Install with `brew install gh`, then `gh auth login`
- **Git remote**: Must have `origin` remote configured
- **Branch**: Must be on a branch (not detached HEAD)

---

## What Gets Pushed

When local changes are detected, the staging branch includes:

1. **All commits on your current branch** (including unpushed ones)
2. **All uncommitted changes** (staged, unstaged, and untracked files)

Everything is bundled into a single commit on the staging branch with a message like:
```
CI: local changes from feature (2025-12-26 22:30)
```

---

## Comparison with Other Methods

| Method | Use Case | Requires Push? |
|--------|----------|----------------|
| `parallel_cloud_run.sh` | Test current code state | No (auto-handles) |
| `[parallel_run.sh ...]` in commit | Test with a commit | Yes |
| `gh workflow run` | Full control | Yes (branch must exist) |
| Manual UI dispatch | One-off runs | Yes (branch must exist) |

Use `parallel_cloud_run.sh` when you want to test uncommitted work without creating commits on your actual branch.

---

## Run URL

After triggering, the script polls GitHub (up to 30 seconds) until the workflow run appears, then displays its direct URL:

```
Waiting for run to appear....

✓ Workflow triggered!
  https://github.com/unifyai/unity/actions/runs/12345678
```

Click the link to go directly to your run—no need to search through the Actions list. CI also generates `pytest-logs-*` artifacts containing full test output, which can be shared with Cursor agents to help debug any failures.

---

## Workflow Example

```bash
# 1. Make some changes locally
vim unity/contact_manager/manager.py

# 2. Test them on CI without committing
parallel_cloud_run.sh tests/test_contact_manager
# Output includes direct link to the run

# 3. Click the link or watch via CLI
gh run watch --repo unifyai/unity <run-id>

# 4. If tests pass, commit your changes
git add -A && git commit -m "Fix contact manager bug"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `gh: command not found` | `brew install gh && gh auth login` |
| `Detached HEAD state` | `git checkout <branch-name>` |
| `Could not restore stash` | Check `git stash list`; apply manually |
| Push failed | Check network; ensure you have push access |

---

## See Also

- [Cloud Test Runs (GitHub Actions)](../README.md#cloud-test-runs-github-actions) — CI triggering overview
- [Parallel Runner](parallel-runner.md) — Local test runner documentation
