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

# Run idempotent setup via Python
python - <<'PYTHON'
import sys

try:
    import unify
except ImportError:
    print("Error: 'unify' package not found. Ensure the virtualenv is active.", file=sys.stderr)
    sys.exit(1)

PROJECT = "UnityTests"

# 1. Activate/create project (idempotent - does not overwrite if exists)
try:
    unify.activate(PROJECT, overwrite=False)
except Exception as e:
    # Tolerate activation failures (e.g., project already active in another process)
    print(f"Note: Project activation returned: {e}", file=sys.stderr)

unify.set_user_logging(False)

# 2. Ensure Durations context with fields (idempotent)
try:
    existing_contexts = unify.get_contexts(prefix="Durations")
except Exception:
    existing_contexts = []

if "Durations" not in existing_contexts:
    try:
        unify.create_context("Durations")
    except Exception as e:
        # Tolerate if already exists (race with another process)
        if "already exists" not in str(e).lower():
            print(f"Note: Durations context creation returned: {e}", file=sys.stderr)

# Ensure fields exist (idempotent - create_fields tolerates existing fields)
try:
    unify.create_fields(
        context="Durations",
        fields={
            "test_fpath": {"type": "str", "mutable": True},
            "tags": {"type": "list", "mutable": True},
            "duration": {"type": "float", "mutable": True},
        },
    )
except Exception as e:
    # Tolerate field creation errors (may already exist)
    pass

print(f"Shared project '{PROJECT}' is ready.")
PYTHON
