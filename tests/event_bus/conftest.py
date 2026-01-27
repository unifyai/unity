"""
tests/event_bus/conftest.py
================================

Configuration for EventBus tests.
All tests in this folder require EventBus publishing to be enabled.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Apply enable_eventbus marker to all tests in this directory.

    Note: pytestmark at module level in conftest.py doesn't reliably apply
    markers to tests in the directory. Using pytest_collection_modifyitems
    is the correct approach for directory-wide marker application.
    """
    marker = pytest.mark.enable_eventbus
    for item in items:
        # Check if the test is in this directory (test_event_bus)
        if "test_event_bus" in str(item.fspath):
            item.add_marker(marker)
