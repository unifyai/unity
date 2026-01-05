#!/bin/bash
# Creates a PR from staging to main with auto-merge enabled (merge strategy)
#
# After the PR merges, the sync-staging workflow will fast-forward staging
# to match main, keeping both branches at the same commit.

set -e

# Check if gh is available
if ! command -v gh &> /dev/null; then
  echo "Error: gh CLI is not installed"
  exit 1
fi

# Check if a PR already exists
existing_pr=$(gh pr list --base main --head staging --state open --json number,url --jq 'if length > 0 then .[0] | "\(.number) \(.url)" else "" end')

if [ -n "$existing_pr" ]; then
  pr_number=$(echo "$existing_pr" | cut -d' ' -f1)
  pr_url=$(echo "$existing_pr" | cut -d' ' -f2)
  echo "PR #$pr_number already exists: $pr_url"
  echo "Enabling auto-merge..."
  gh pr merge "$pr_number" --auto --merge
  echo "Done. PR will be merged when all checks pass."
  exit 0
fi

# Create the PR
echo "Creating PR: staging → main"
pr_url=$(gh pr create \
  --base main \
  --head staging \
  --title "Release: staging → main" \
  --body "Automated release PR from staging to main.")

pr_number=$(echo "$pr_url" | grep -oE '[0-9]+$')
echo "Created PR #$pr_number: $pr_url"

# Enable auto-merge
echo "Enabling auto-merge..."
gh pr merge "$pr_number" --auto --merge

echo "Done. PR will be merged when all checks pass."
