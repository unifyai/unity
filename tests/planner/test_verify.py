import pytest
import queue

from planner.verifier import verify, BubbleUp, Verifier
from planner.model import Primitive


# Helper function to set up queues for testing
def set_queues(broadcast_queue=None, reimplement_queue=None):
    """Set up the queues for testing."""
    from planner.context import context

    if broadcast_queue is None:
        broadcast_queue = queue.Queue()

    if reimplement_queue is None:
        reimplement_queue = queue.Queue()

    context.set_broadcast_queue(broadcast_queue)
    Verifier._reimplement_queue = reimplement_queue

    return broadcast_queue, reimplement_queue


def test_verify_reimplement(monkeypatch):
    """
    Test that the verify decorator correctly handles 'reimplement' verdicts.

    This test:
    1. Sets up the queues
    2. Mocks get_snapshot to return two identical snapshots then a changed one
    3. Mocks Verifier.check to return 'reimplement' then 'ok'
    4. Defines a decorated function and calls it
    5. Asserts it eventually returns a Primitive and at least one rewrite was attempted
    """
    # Set up queues
    broadcast_queue, reimplement_queue = set_queues(queue.Queue(), queue.Queue())

    # Create mock snapshots
    snapshot1 = {"url": "https://example.com", "title": "Example"}
    snapshot2 = {"url": "https://example.com", "title": "Example"}  # Identical to first
    snapshot3 = {
        "url": "https://example.com/changed",
        "title": "Changed Example",
    }  # Different

    # Mock get_snapshot to return different snapshots in sequence
    snapshots = [snapshot1, snapshot2, snapshot3]
    snapshot_index = 0

    def mock_get_snapshot():
        nonlocal snapshot_index
        result = snapshots[snapshot_index]
        snapshot_index = min(snapshot_index + 1, len(snapshots) - 1)
        return result

    monkeypatch.setattr("planner.context.get_snapshot", mock_get_snapshot)

    # Mock Verifier.check to return 'reimplement' then 'ok'
    check_results = ["reimplement", "ok"]
    check_index = 0

    def mock_check(*args, **kwargs):
        nonlocal check_index
        result = check_results[check_index]
        check_index = min(check_index + 1, len(check_results) - 1)
        return result

    monkeypatch.setattr("planner.verifier.Verifier.check", mock_check)

    # Define a decorated function
    @verify
    def foo():
        # Return a Primitive to simulate a browser action
        return Primitive("open_browser", {}, "open_browser")

    # Call the function
    result = foo()

    # Assert the function was eventually called successfully
    assert isinstance(result, Primitive)
    assert result.call_literal == "open_browser"

    # Assert the function object was enqueued
    fn_obj = reimplement_queue.get_nowait()
    assert fn_obj.__name__ == "foo"


def test_verify_push_up(monkeypatch):
    """
    Test that the verify decorator correctly handles 'push_up_stack' verdicts.

    This test:
    1. Sets up the queues
    2. Mocks get_snapshot to always return the same snapshot
    3. Mocks Verifier.check to return 'push_up_stack'
    4. Defines nested decorated functions
    5. Asserts that calling the outer function raises BubbleUp
    """
    # Set up queues
    set_queues()

    # Create a mock snapshot
    snapshot = {"url": "https://example.com", "title": "Example"}

    # Mock get_snapshot to always return the same snapshot
    def mock_get_snapshot():
        return snapshot

    monkeypatch.setattr("planner.context.get_snapshot", mock_get_snapshot)

    # Mock Verifier.check to return 'push_up_stack'
    def mock_check(*args, **kwargs):
        return "push_up_stack"

    monkeypatch.setattr("planner.verifier.Verifier.check", mock_check)

    # Define nested decorated functions
    @verify
    def low():
        return Primitive("low_level_action", {}, "low_level_action")

    @verify
    def high():
        return low()

    # Assert that calling high() raises BubbleUp
    with pytest.raises(BubbleUp):
        high()
