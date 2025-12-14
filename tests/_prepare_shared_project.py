#!/usr/bin/env python
"""
Internal module to prepare the shared UnityTests project for parallel runs.

This script is called by parallel_run.sh before spawning tmux sessions.
It ensures the shared project and contexts exist, making subsequent
parallel pytest sessions race-free.

The script is idempotent: calling it multiple times has no adverse effects.

Usage (internal - typically invoked via parallel_run.sh):
    python tests/_prepare_shared_project.py
"""

import logging
import sys

from dotenv import load_dotenv

# Suppress urllib3 connection pool warnings during parallel API calls
logging.getLogger("urllib3").setLevel(logging.ERROR)

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
    try:
        existing_contexts = unify.get_contexts(prefix="Combined")
    except Exception:
        existing_contexts = []

    if "Combined" not in existing_contexts:
        try:
            unify.create_context("Combined")
        except Exception as e:
            # Tolerate if already exists (race with another process)
            if "already exists" not in str(e).lower():
                print(
                    f"Note: Combined context creation returned: {e}",
                    file=sys.stderr,
                )

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

    print(f"Shared project '{PROJECT}' is ready.")


if __name__ == "__main__":
    prepare_shared_project()
