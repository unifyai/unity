#!/usr/bin/env bash
set -euo pipefail

# Trigger CI tests on the current code state via GitHub Actions.
#
# If the branch is clean and pushed, triggers CI directly on the current branch.
# If there are local changes (uncommitted or unpushed), creates a unique staging
# branch with a datetime suffix, pushes the current state, triggers CI, then
# restores the local state.
#
# Environment variables from .env are automatically passed to CI. Explicit --env
# args override .env values (later values win).
#
# Usage:
#   parallel_cloud_run.sh tests/test_contact_manager
#   parallel_cloud_run.sh tests/test_actor tests/test_conductor
#   parallel_cloud_run.sh .                    # All tests
#   parallel_cloud_run.sh -s                   # All tests, serial mode (implicit ".")
#   parallel_cloud_run.sh -s tests/            # Specific path, serial mode
#   parallel_cloud_run.sh --env UNIFY_CACHE=false tests/  # Override .env
#
# Each run creates a unique branch (ci-staging-{user}-{datetime}) for isolation.

REPO="unifyai/unity"
WORKFLOW="tests.yml"
POLL_TIMEOUT=30  # seconds to wait for run to appear

# Find repo root and cd into it (allows calling from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# Helper: Poll for the workflow run URL after triggering
# ============================================================================

get_run_url() {
  local branch="$1"
  local trigger_time="$2"  # ISO timestamp from before triggering
  local test_path="$3"     # For soft name matching

  local poll_start
  poll_start=$(date +%s)

  echo -n "Waiting for run to appear"

  while true; do
    # Get recent runs on this branch (fetch several to find the right one)
    local runs_json
    runs_json=$(gh run list \
      --repo "$REPO" \
      --workflow "$WORKFLOW" \
      --branch "$branch" \
      --limit 10 \
      --json databaseId,createdAt,name 2>/dev/null || echo "[]")

    if [[ -n "$runs_json" && "$runs_json" != "[]" ]]; then
      # Use Python to filter and select the best matching run:
      # 1. Must be created after trigger_time (required)
      # 2. Prefer runs whose name contains test_path (soft preference)
      # Note: Pipe JSON via stdin to avoid heredoc escaping issues with special chars
      local run_id
      run_id=$(echo "$runs_json" | python3 -c "
import json
import sys
from datetime import datetime

runs = json.load(sys.stdin)
trigger_time = datetime.fromisoformat('$trigger_time'.replace('Z', '+00:00'))
test_path = '$test_path'.lower()

# Filter to runs created after trigger
candidates = []
for run in runs:
    created = datetime.fromisoformat(run['createdAt'].replace('Z', '+00:00'))
    if created >= trigger_time:
        candidates.append((created, run))

if not candidates:
    sys.exit(0)

# Sort by creation time descending (most recent first)
candidates.sort(key=lambda x: x[0], reverse=True)

# Prefer runs whose name contains the test path (case-insensitive)
if test_path and test_path != '.':
    for created, run in candidates:
        name = (run.get('name') or '').lower()
        # Check if test path or its basename is in the name
        path_parts = test_path.replace('tests/', '').split('/')
        if any(part in name for part in path_parts if part):
            print(run['databaseId'])
            sys.exit(0)

# Fall back to most recent timestamp-matched run
print(candidates[0][1]['databaseId'])
"
      )

      if [[ -n "$run_id" ]]; then
        echo ""
        echo "https://github.com/$REPO/actions/runs/$run_id"
        return 0
      fi
    fi

    # Check timeout
    local elapsed=$(( $(date +%s) - poll_start ))
    if (( elapsed >= POLL_TIMEOUT )); then
      echo ""
      echo "https://github.com/$REPO/actions/workflows/$WORKFLOW"
      return 1
    fi

    echo -n "."
    sleep 1
  done
}

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$CURRENT_BRANCH" == "HEAD" ]]; then
  echo "Error: Detached HEAD state. Please checkout a branch first." >&2
  exit 1
fi

# Check for gh CLI
if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI is required. Install with: brew install gh" >&2
  exit 1
fi

# ============================================================================
# Parse arguments: separate flags from test paths
# ============================================================================

declare -a EXTRA_ENV_ARGS=()
declare -a PASSTHROUGH_ARGS=()
declare -a TEST_PATHS=()

while (( $# > 0 )); do
  case "$1" in
    --env)
      if [[ -n "${2:-}" ]]; then
        EXTRA_ENV_ARGS+=("--env" "$2")
        shift 2
      else
        echo "Error: --env requires a KEY=VALUE argument" >&2
        exit 1
      fi
      ;;
    --env=*)
      EXTRA_ENV_ARGS+=("--env" "${1#--env=}")
      shift
      ;;
    -*)
      # Any other flag (e.g., -s, --timeout, --no-cache) passes through to parallel_run.sh
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      TEST_PATHS+=("$1")
      shift
      ;;
  esac
done

# Build test_path from remaining arguments (default to "." for full suite)
if (( ${#TEST_PATHS[@]} == 0 )); then
  TEST_PATH="."
else
  TEST_PATH="${TEST_PATHS[*]}"
fi

# ============================================================================
# Load .env file content (will be passed securely to CI, not as CLI args)
# ============================================================================

ENV_FILE="$REPO_ROOT/.env"
ENV_FILE_CONTENT_B64=""
if [[ -f "$ENV_FILE" ]]; then
  # Base64 encode the entire .env file for safe transport to CI
  # CI will decode it, write to .env on runner, and mask sensitive values
  ENV_FILE_CONTENT_B64=$(base64 < "$ENV_FILE" | tr -d '\n')
fi

# Build parallel_run_args from passthrough flags and explicit --env CLI args
# These are intentionally visible in logs since user explicitly passed them
PARALLEL_RUN_ARGS=""
if (( ${#PASSTHROUGH_ARGS[@]} > 0 )); then
  PARALLEL_RUN_ARGS="${PASSTHROUGH_ARGS[*]}"
fi
if (( ${#EXTRA_ENV_ARGS[@]} > 0 )); then
  PARALLEL_RUN_ARGS="${PARALLEL_RUN_ARGS:+$PARALLEL_RUN_ARGS }${EXTRA_ENV_ARGS[*]}"
fi

# ============================================================================
# Check if we need the staging branch approach
# ============================================================================

has_uncommitted_changes() {
  ! git diff --cached --quiet || ! git diff --quiet || \
    [[ -n "$(git ls-files --others --exclude-standard)" ]]
}

has_unpushed_commits() {
  if git rev-parse --verify "@{u}" >/dev/null 2>&1; then
    [[ $(git rev-list --count "@{u}..HEAD" 2>/dev/null) -gt 0 ]]
  else
    # No upstream - check if branch exists on remote
    ! git ls-remote --exit-code --heads origin "$CURRENT_BRANCH" >/dev/null 2>&1
  fi
}

# If clean and pushed, use simple path
if ! has_uncommitted_changes && ! has_unpushed_commits; then
  echo "Branch is clean and pushed. Triggering CI on: $CURRENT_BRANCH"
  echo "Test path: $TEST_PATH"
  [[ -n "$PARALLEL_RUN_ARGS" ]] && echo "Explicit overrides: $PARALLEL_RUN_ARGS"
  [[ -n "$ENV_FILE_CONTENT_B64" ]] && echo "Environment: .env file (contents hidden)"
  echo ""

  # Capture timestamp before triggering (for accurate run detection)
  TRIGGER_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # Build workflow dispatch command
  # Note: env_file_content is base64-encoded and passed securely (not echoed)
  gh_args=(
    workflow run "$WORKFLOW"
    --repo "$REPO"
    --ref "$CURRENT_BRANCH"
    -f "test_path=$TEST_PATH"
  )
  [[ -n "$PARALLEL_RUN_ARGS" ]] && gh_args+=(-f "parallel_run_args=$PARALLEL_RUN_ARGS")
  [[ -n "$ENV_FILE_CONTENT_B64" ]] && gh_args+=(-f "env_file_content=$ENV_FILE_CONTENT_B64")

  gh "${gh_args[@]}"

  echo ""
  RUN_URL=$(get_run_url "$CURRENT_BRANCH" "$TRIGGER_TIME" "$TEST_PATH")
  echo ""
  echo "✓ Workflow triggered!"
  echo "  $RUN_URL"
  exit 0
fi

# ============================================================================
# Staging branch approach for local changes
# ============================================================================

echo "Local changes detected. Creating unique staging branch..."
echo ""

# Generate unique staging branch name with datetime suffix
TEMP_USER=$(git config user.name 2>/dev/null | tr ' ' '-' | tr '[:upper:]' '[:lower:]' || whoami)
DATETIME=$(date +%Y-%m-%dT%H-%M-%S)
STAGING_BRANCH="ci-staging-${TEMP_USER}-${DATETIME}"

# Save list of staged files for restoration (preserves staged vs unstaged distinction)
STAGED_FILES=$(git diff --cached --name-only)

# Track state for cleanup
NEED_RESTORE=""

cleanup() {
  local current
  current=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

  # Return to original branch if not already there
  if [[ "$current" != "$CURRENT_BRANCH" ]]; then
    git checkout "$CURRENT_BRANCH" 2>/dev/null || true
  fi

  # Restore stashed changes if we created a stash
  if [[ -n "$NEED_RESTORE" ]]; then
    if git stash pop 2>/dev/null; then
      # Re-stage files that were originally staged
      if [[ -n "$STAGED_FILES" ]]; then
        echo "$STAGED_FILES" | while IFS= read -r file; do
          [[ -n "$file" ]] && git add "$file" 2>/dev/null || true
        done
      fi
    else
      echo "Warning: Could not restore stash. Check 'git stash list'." >&2
    fi
  fi
}
trap cleanup EXIT

# Stash uncommitted changes if any
if has_uncommitted_changes; then
  echo "Stashing uncommitted changes..."
  git stash push -u -m "parallel_cloud_run staging"
  NEED_RESTORE="yes"
fi

# Create unique staging branch from current HEAD (includes unpushed commits)
echo "Creating branch: $STAGING_BRANCH"
git checkout -b "$STAGING_BRANCH"

# Apply and commit stashed changes if we stashed anything
if [[ -n "$NEED_RESTORE" ]]; then
  git stash apply stash@{0}
  git add -A
  git commit -m "CI: local changes from $CURRENT_BRANCH ($(date +%Y-%m-%d\ %H:%M))"
fi

# Push to remote (new unique branch, no force needed)
echo "Pushing to origin/$STAGING_BRANCH..."
git push origin "$STAGING_BRANCH"

# Return to original branch (cleanup will handle stash restoration)
git checkout "$CURRENT_BRANCH"

# Trigger CI
echo ""
echo "Triggering CI on staging branch: $STAGING_BRANCH"
echo "Test path: $TEST_PATH"
[[ -n "$PARALLEL_RUN_ARGS" ]] && echo "Explicit overrides: $PARALLEL_RUN_ARGS"
[[ -n "$ENV_FILE_CONTENT_B64" ]] && echo "Environment: .env file (contents hidden)"
echo ""

# Capture timestamp before triggering (for accurate run detection)
TRIGGER_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build workflow dispatch command
# Note: env_file_content is base64-encoded and passed securely (not echoed)
gh_args=(
  workflow run "$WORKFLOW"
  --repo "$REPO"
  --ref "$STAGING_BRANCH"
  -f "test_path=$TEST_PATH"
)
[[ -n "$PARALLEL_RUN_ARGS" ]] && gh_args+=(-f "parallel_run_args=$PARALLEL_RUN_ARGS")
[[ -n "$ENV_FILE_CONTENT_B64" ]] && gh_args+=(-f "env_file_content=$ENV_FILE_CONTENT_B64")

gh "${gh_args[@]}"

echo ""
RUN_URL=$(get_run_url "$STAGING_BRANCH" "$TRIGGER_TIME" "$TEST_PATH")
echo ""
echo "✓ Workflow triggered!"
echo "  $RUN_URL"
echo ""
echo "Branch: $STAGING_BRANCH"
echo "Note: Delete stale ci-staging-* branches periodically with:"
echo "  git branch -r | grep 'ci-staging-' | sed 's|origin/||' | xargs -I{} git push origin --delete {}"
