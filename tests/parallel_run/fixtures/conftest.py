"""
Minimal conftest for parallel_run.sh fixture tests.

These tests exist only to verify parallel_run.sh behavior (session creation,
status prefixes, exit codes, etc.), not to test unity functionality.

By overriding the global session hooks, we skip:
- Random project creation (API calls)
- Heavy unity imports
- Context setup

This makes fixture tests complete in milliseconds instead of seconds,
enabling high-concurrency testing of parallel_run.sh itself.
"""

import os


def pytest_configure(config):
    """Mark that we're in fixture test mode and skip heavy setup."""
    # Signal to the global conftest that we want minimal setup
    os.environ["UNIFY_SKIP_SESSION_SETUP"] = "True"
    # Don't create random projects - we don't need them
    os.environ["UNIFY_TESTS_RAND_PROJ"] = "False"
    os.environ["UNIFY_TESTS_DELETE_PROJ_ON_START"] = "False"
    os.environ["UNIFY_TESTS_DELETE_PROJ_ON_EXIT"] = "False"


def pytest_sessionstart(session):
    """Override global session start - skip project activation."""
    # Minimal setup: just ensure unify is importable but don't activate


def pytest_sessionfinish(session, exitstatus):
    """Override global session finish - skip project cleanup."""
