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
```

The script automatically handles uncommitted changes and unpushed commits, pushing them to a staging branch for CI while leaving your local state unchanged.

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
     2. Create/reset staging branch
     3. Apply stash + commit
     4. Force push staging branch
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
│  CI runs on: ci-staging-your-name           │
└─────────────────────────────────────────────┘
```

---

## The Staging Branch

When local changes exist, the script uses a persistent staging branch named `ci-staging-{username}` (derived from your git username).

**Why persistent?**

- **No timing issues**: If we created and immediately deleted a temp branch, the CI run might fail because GitHub hasn't finished cloning yet
- **No branch pollution**: One branch per developer, reused across runs
- **Fast**: Force-push overwrites previous state instantly

The staging branch is automatically created on first use and updated on each subsequent run.

---

## Usage

```bash
parallel_cloud_run.sh [test_paths...]
```

| Argument | Description |
|----------|-------------|
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

## Workflow Example

```bash
# 1. Make some changes locally
vim unity/contact_manager/manager.py

# 2. Test them on CI without committing
parallel_cloud_run.sh tests/test_contact_manager

# 3. Watch the run
gh run list --repo unifyai/unity --workflow tests.yml
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
