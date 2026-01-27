#!/usr/bin/env bash
set -euo pipefail

# Trigger CI tests via GitHub Actions.
#
# By default, tests whatever is currently on the remote branch (safe, non-invasive).
# If local changes exist (uncommitted or unpushed), a warning is printed but CI
# runs against the remote state.
#
# Use --push-local to explicitly opt-in to testing local state. This creates a
# unique staging branch (ci-staging-{user}-{datetime}), commits any uncommitted
# changes on top of unpushed commits, pushes, and triggers CI.
#
# Environment variables from .env are automatically passed to CI. Explicit --env
# args override .env values (later values win).
#
# Usage:
#   parallel_cloud_run.sh tests/contact_manager
#   parallel_cloud_run.sh tests/actor tests/contact_manager
#   parallel_cloud_run.sh .                    # All tests
#   parallel_cloud_run.sh -s                   # All tests, serial mode (implicit ".")
#   parallel_cloud_run.sh -s tests/            # Specific path, serial mode
#   parallel_cloud_run.sh --env UNIFY_CACHE=false tests/  # Override .env
#   parallel_cloud_run.sh --push-local tests/  # Include local uncommitted/unpushed changes

REPO="unifyai/unity"
WORKFLOW="tests.yml"
POLL_TIMEOUT=30  # seconds to wait for run to appear

# Find repo root and cd into it (allows calling from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Source shared argument parsing (used by both parallel_run.sh and parallel_cloud_run.sh)
source "$SCRIPT_DIR/_parse_args.sh"

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
# Parse arguments using shared helper
# ============================================================================
# First extract --push-local (cloud-specific), then parse the rest with shared helper.

PUSH_LOCAL=0
declare -a REMAINING_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--push-local" ]]; then
    PUSH_LOCAL=1
  else
    REMAINING_ARGS+=("$arg")
  fi
done

# Parse remaining arguments using shared helper
# Returns: 0=success, 1=help requested, 2=error
parse_test_args "${REMAINING_ARGS[@]}"
_parse_result=$?
if (( _parse_result == 1 )); then
  # Help requested - show cloud-specific help
  HELP_SCRIPT_NAME="parallel_cloud_run.sh"
  HELP_EXTRA_OPTIONS="  --push-local         Push local uncommitted/unpushed changes to CI
"
  print_help
  exit 0
elif (( _parse_result == 2 )); then
  # Error (already printed)
  exit 1
fi
unset _parse_result REMAINING_ARGS

# ============================================================================
# Resolve and validate test paths
# ============================================================================
# This runs after cd "$REPO_ROOT", so all paths are relative to repo root.

if ! resolve_test_paths "$REPO_ROOT"; then
  exit 1
fi

# Build test_path from resolved arguments (default to "." for full suite)
if (( ${#RESOLVED_TEST_PATHS[@]} == 0 )); then
  TEST_PATH="."
else
  TEST_PATH="${RESOLVED_TEST_PATHS[*]}"
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

# Build parallel_run_args from parsed flags using shared helper
# include-env ensures --env flags are included (visible in CI logs since user passed them)
PARALLEL_RUN_ARGS="$(reconstruct_parallel_run_args include-env)"

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

# Default: test against remote branch
if (( ! PUSH_LOCAL )); then
  # Check for local changes and warn (but proceed with remote)
  UNCOMMITTED=$(has_uncommitted_changes && echo 1 || echo 0)
  UNPUSHED=$(has_unpushed_commits && echo 1 || echo 0)

  if (( UNCOMMITTED || UNPUSHED )); then
    echo "Note: Local changes detected but not included in this CI run."
    (( UNCOMMITTED )) && echo "  - Uncommitted changes present"
    (( UNPUSHED )) && echo "  - Unpushed commits present"
    echo "Use --push-local to test local state instead."
    echo ""
  fi

  echo "Triggering CI on remote: origin/$CURRENT_BRANCH"
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
# --push-local: Push local state to a staging branch
# ============================================================================

echo "Pushing local state to staging branch (--push-local)..."
echo ""

# Generate unique staging branch name with datetime suffix
TEMP_USER=$(git config user.name 2>/dev/null | tr ' ' '-' | tr '[:upper:]' '[:lower:]' || whoami)
DATETIME=$(date +%Y-%m-%dT%H-%M-%S)
STAGING_BRANCH="ci-staging-${TEMP_USER}-${DATETIME}"

# Track if we made a temporary commit (for cleanup)
DID_TEMP_COMMIT=0

cleanup() {
  # If we made a temporary commit, undo it (returns changes to working directory)
  if (( DID_TEMP_COMMIT )); then
    git reset HEAD~1 >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Commit uncommitted changes on top of current HEAD (if any)
if has_uncommitted_changes; then
  echo "Committing local changes temporarily..."
  git add -A
  git commit -m "CI: local changes from $CURRENT_BRANCH ($(date +%Y-%m-%d\ %H:%M))"
  DID_TEMP_COMMIT=1
fi

# Push current HEAD to the staging branch (no local branch switch needed)
echo "Pushing to origin/$STAGING_BRANCH..."
git push origin "HEAD:refs/heads/$STAGING_BRANCH"

# Cleanup will reset the temporary commit, restoring uncommitted changes

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
