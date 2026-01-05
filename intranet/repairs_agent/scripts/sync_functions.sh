#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# sync_functions.sh - Sync metric functions to FunctionManager
#
# Wrapper script for the Python sync module.
#
# Usage:
#   ./sync_functions.sh                    # Sync all metrics
#   ./sync_functions.sh --dry-run          # Preview what would be synced
#   ./sync_functions.sh --overwrite        # Overwrite existing functions
#   ./sync_functions.sh -v --overwrite     # Verbose sync with overwrite
# ==============================================================================

# Resolve script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"

# Load .env if exists
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

# Activate venv if not already in one
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV_PY="$REPO_ROOT/.venv/bin/python"
    if [[ ! -x "$VENV_PY" ]]; then
        echo "❌ Virtual environment not found at $REPO_ROOT/.venv"
        echo "   Run: pip install uv && uv sync --all-groups"
        exit 1
    fi
else
    VENV_PY="python"
fi

# Run the sync script
exec "$VENV_PY" -m intranet.repairs_agent.dynamic.sync "$@"
