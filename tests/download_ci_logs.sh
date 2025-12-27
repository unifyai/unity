#!/usr/bin/env bash
set -euo pipefail

# Download pytest logs from a GitHub Actions artifact URL directly into pytest_logs/
#
# Usage:
#   download_ci_logs.sh <artifact_url>
#
# Example:
#   download_ci_logs.sh https://github.com/unifyai/unity/actions/runs/20537719146/artifacts/4974812109
#
# The script extracts the artifact ID from the URL and downloads directly via the GitHub API.
# Logs are placed in pytest_logs/ alongside any local test logs.

# Find repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Argument validation ----

if [[ $# -ne 1 ]]; then
  echo "Usage: download_ci_logs.sh <artifact_url>" >&2
  echo "" >&2
  echo "Example:" >&2
  echo "  download_ci_logs.sh https://github.com/unifyai/unity/actions/runs/20537719146/artifacts/4974812109" >&2
  exit 1
fi

ARTIFACT_URL="$1"

# ---- Extract components from URL ----
# URL format: https://github.com/{owner}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}

if [[ ! "$ARTIFACT_URL" =~ github\.com/([^/]+)/([^/]+)/actions/runs/([0-9]+)/artifacts/([0-9]+) ]]; then
  echo "Error: Invalid artifact URL format." >&2
  echo "" >&2
  echo "Expected format:" >&2
  echo "  https://github.com/{owner}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}" >&2
  echo "" >&2
  echo "Got: $ARTIFACT_URL" >&2
  exit 1
fi

OWNER="${BASH_REMATCH[1]}"
REPO="${BASH_REMATCH[2]}"
RUN_ID="${BASH_REMATCH[3]}"
ARTIFACT_ID="${BASH_REMATCH[4]}"

echo "Downloading artifact..."
echo "  Repository:  $OWNER/$REPO"
echo "  Run ID:      $RUN_ID"
echo "  Artifact ID: $ARTIFACT_ID"
echo ""

# ---- Check for gh CLI ----

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI is required. Install with: brew install gh" >&2
  exit 1
fi

# ---- Download and extract ----

PYTEST_LOGS_DIR="$REPO_ROOT/pytest_logs"
TEMP_ZIP=$(mktemp)

# Ensure pytest_logs directory exists
mkdir -p "$PYTEST_LOGS_DIR"

# Download artifact via GitHub API
echo "Fetching from GitHub API..."
if ! gh api "repos/$OWNER/$REPO/actions/artifacts/$ARTIFACT_ID/zip" > "$TEMP_ZIP" 2>&1; then
  echo "Error: Failed to download artifact. Check that:" >&2
  echo "  - You are authenticated with 'gh auth login'" >&2
  echo "  - The artifact URL is valid and not expired" >&2
  rm -f "$TEMP_ZIP"
  exit 1
fi

# Extract using Python (more portable than unzip which may not be installed)
echo "Extracting to pytest_logs/..."
if ! python3 -c "import zipfile; zipfile.ZipFile('$TEMP_ZIP').extractall('$PYTEST_LOGS_DIR')"; then
  echo "Error: Failed to extract artifact." >&2
  rm -f "$TEMP_ZIP"
  exit 1
fi

# Clean up temp file
rm -f "$TEMP_ZIP"

# Show what was extracted
echo ""
echo "✓ Downloaded CI logs to pytest_logs/"
echo ""

# List the extracted directories (CI logs are in datetime-prefixed subdirs)
echo "Extracted contents:"
# Find directories that were just created (modified in last minute)
find "$PYTEST_LOGS_DIR" -maxdepth 1 -type d -mmin -1 -name "20*" | while read -r dir; do
  dirname=$(basename "$dir")
  file_count=$(find "$dir" -type f | wc -l | tr -d ' ')
  echo "  $dirname/ ($file_count files)"
done

echo ""
echo "View logs with:"
echo "  ls pytest_logs/"
