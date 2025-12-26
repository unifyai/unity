#!/usr/bin/env bash
set -euo pipefail

# Trigger CI tests on the current branch via GitHub Actions.
#
# Usage:
#   parallel_cloud_run.sh tests/test_contact_manager
#   parallel_cloud_run.sh tests/test_actor tests/test_conductor
#   parallel_cloud_run.sh .                    # All tests
#
# This is a thin wrapper around `gh workflow run` that:
# - Automatically uses the current branch
# - Passes test paths to the workflow

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

# Build test_path from arguments (space-separated paths become the value)
if (( $# == 0 )); then
  TEST_PATH="."
else
  TEST_PATH="$*"
fi

echo "Triggering CI tests on branch: $CURRENT_BRANCH"
echo "Test path: $TEST_PATH"
echo ""

gh workflow run "$WORKFLOW" \
  --repo "$REPO" \
  --ref "$CURRENT_BRANCH" \
  -f test_path="$TEST_PATH"

echo ""
echo "Workflow triggered! View at:"
echo "  https://github.com/$REPO/actions/workflows/$WORKFLOW"
