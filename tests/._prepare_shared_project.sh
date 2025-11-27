#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Internal script to prepare the shared UnityTests project for parallel runs.
#
# This script is designed to be called automatically by .parallel_run.sh
# before spawning tmux sessions. It ensures the shared project and contexts
# exist, making subsequent parallel pytest sessions race-free.
#
# The script is idempotent: calling it multiple times has no adverse effects.
#
# Usage (internal - typically invoked by .parallel_run.sh):
#   tests/._prepare_shared_project.sh
#
# Environment:
#   UNIFY_KEY          Required. API key for the Unify backend.
#   UNIFY_BASE_URL     Optional. Override the default API endpoint.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Source environment from .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

# Activate virtualenv
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

# Run idempotent setup via the companion Python module
python "$SCRIPT_DIR/_prepare_shared_project.py"
