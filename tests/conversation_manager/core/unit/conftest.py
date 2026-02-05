"""
Minimal conftest for isolated unit tests.

Tests in this directory run with mocked dependencies and don't require
the full unify infrastructure (no API authentication, no Orchestra).

IMPORTANT: Run these tests with --confcutdir to skip parent conftest hooks:

    pytest tests/conversation_manager/core/unit/ \\
        --confcutdir=tests/conversation_manager/core/unit -v

The --confcutdir flag prevents pytest from loading conftest.py files
from parent directories, which would otherwise try to activate unify
and make API calls.
"""
