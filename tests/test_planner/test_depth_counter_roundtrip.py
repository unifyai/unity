from planner.context import context
import pytest


def test_depth_counter_roundtrip():
    # Enter twice with dummy IDs
    context.enter_exploration("tab1", "main")
    context.enter_exploration("tab2", "main")
    assert context._exploration_depth == 2

    # Exit twice and verify cleanup
    tab = context.exit_exploration()
    assert tab is None
    tab = context.exit_exploration()
    assert tab == "main"
    assert context._exploration_depth == 0
    assert context.get_exploration_tab() is None
    assert context.get_main_tab() is None
