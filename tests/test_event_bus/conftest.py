"""
tests/test_event_bus/conftest.py
================================

Configuration for EventBus tests.
All tests in this folder require EventBus publishing to be enabled.
"""

import pytest

# Apply the enable_eventbus marker to all tests in this folder
pytestmark = pytest.mark.enable_eventbus
