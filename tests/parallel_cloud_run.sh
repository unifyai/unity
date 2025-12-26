#!/usr/bin/env bash
set -euo pipefail

# Trigger CI tests on the current code state via GitHub Actions.
#
# If the branch is clean and pushed, triggers CI directly on the current branch.
# If there are local changes (uncommitted or unpushed), uses a persistent staging
# branch to push the current state, triggers CI, then restores the local state.
#
# Usage:
#   parallel_cloud_run.sh tests/test_contact_manager
#   parallel_cloud_run.sh tests/test_actor tests/test_conductor
#   parallel_cloud_run.sh .                    # All tests
#
# The staging branch (ci-staging-{username}) persists for CI reruns.

REPO="unifyai/unity"
WORKFLOW="tests.yml"

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

# Build test_path from arguments
if (( $# == 0 )); then
  TEST_PATH="."
else
  TEST_PATH="$*"
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
  echo ""

  gh workflow run "$WORKFLOW" \
    --repo "$REPO" \
    --ref "$CURRENT_BRANCH" \
    -f test_path="$TEST_PATH"

  echo ""
  echo "✓ Workflow triggered!"
  echo "  https://github.com/$REPO/actions/workflows/$WORKFLOW"
  exit 0
fi

# ============================================================================
# Staging branch approach for local changes
# ============================================================================

echo "Local changes detected. Using staging branch..."
echo ""

# Generate staging branch name based on git username
TEMP_USER=$(git config user.name 2>/dev/null | tr ' ' '-' | tr '[:upper:]' '[:lower:]' || whoami)
STAGING_BRANCH="ci-staging-${TEMP_USER}"

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

# Create/reset staging branch to current HEAD (includes unpushed commits)
echo "Creating staging branch: $STAGING_BRANCH"
git checkout -B "$STAGING_BRANCH"

# Apply and commit stashed changes if we stashed anything
if [[ -n "$NEED_RESTORE" ]]; then
  git stash apply stash@{0}
  git add -A
  git commit -m "CI: local changes from $CURRENT_BRANCH ($(date +%Y-%m-%d\ %H:%M))"
fi

# Force push to remote
echo "Pushing to origin/$STAGING_BRANCH..."
git push -f origin "$STAGING_BRANCH"

# Return to original branch (cleanup will handle stash restoration)
git checkout "$CURRENT_BRANCH"

# Trigger CI
echo ""
echo "Triggering CI on staging branch: $STAGING_BRANCH"
echo "Test path: $TEST_PATH"
echo ""

gh workflow run "$WORKFLOW" \
  --repo "$REPO" \
  --ref "$STAGING_BRANCH" \
  -f test_path="$TEST_PATH"

echo ""
echo "✓ Workflow triggered!"
echo "  https://github.com/$REPO/actions/workflows/$WORKFLOW"
echo ""
echo "Staging branch '$STAGING_BRANCH' persists for future runs."
