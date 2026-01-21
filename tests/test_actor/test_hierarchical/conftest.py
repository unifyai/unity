"""
Pytest configuration for HierarchicalActor tests.

We keep shared helpers in `helpers.py` and import them explicitly where needed.
Conftest should stay small and fixture-focused to avoid hidden magic.
"""

import pytest  # noqa: F401

