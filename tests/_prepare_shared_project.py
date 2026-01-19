#!/usr/bin/env python3
"""
Internal module to prepare the shared UnityTests project for parallel runs.

This script is called by parallel_run.sh before spawning tmux sessions.
It ensures the shared project and contexts exist, making subsequent
parallel pytest sessions race-free.

The script is idempotent: calling it multiple times has no adverse effects.

Usage (internal - typically invoked via parallel_run.sh):
    python3 tests/_prepare_shared_project.py
"""

# Suppress urllib3 connection pool warnings before any imports configure logging
import logging

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

import sys
from pathlib import Path

# Ensure repo root is in sys.path so we can import unity.
# This script is invoked directly by parallel_run.sh, not via pytest
# which handles path setup automatically.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv()

PROJECT = "UnityTests"


def prepare_shared_project() -> None:
    """Prepare the shared UnityTests project and Combined context."""
    try:
        import unify
    except ImportError:
        print(
            "Error: 'unify' package not found. Ensure the virtualenv is active.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Activate/create project (idempotent - does not overwrite if exists)
    try:
        unify.activate(PROJECT, overwrite=False)
    except Exception as e:
        # Tolerate activation failures (e.g., project already active in another process)
        print(f"Note: Project activation returned: {e}", file=sys.stderr)

    unify.set_user_logging(False)

    # 2. Ensure Combined context with fields (idempotent)
    unify.create_context("Combined")

    # Ensure fields exist (idempotent - create_fields tolerates existing fields)
    try:
        unify.create_fields(
            context="Combined",
            fields={
                "test_fpath": {"type": "str", "mutable": True},
                "tags": {"type": "list", "mutable": True},
                "duration": {"type": "float", "mutable": True},
                "llm_io": {"type": "list", "mutable": True},
                "settings": {"type": "dict", "mutable": True},
            },
        )
    except Exception:
        # Tolerate field creation errors (may already exist)
        pass

    # Pre-create assistant-derived contexts via unity.init() to avoid races
    # when parallel pytest sessions (xdist, tmux, CI) all call unity.init()
    try:
        import unity

        unity.init(PROJECT)
    except Exception as e:
        # Tolerate if contexts already exist (another process created them)
        if "already exists" not in str(e).lower():
            print(f"Note: unity.init() returned: {e}", file=sys.stderr)

    print(f"Prepared shared project '{PROJECT}'")


if __name__ == "__main__":
    prepare_shared_project()
