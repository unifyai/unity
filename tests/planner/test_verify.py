import pytest
import queue
from unittest.mock import patch, MagicMock

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


def test_verify_reimplement():
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
    snapshot3 = {"url": "https://example.com/changed", "title": "Changed Example"}  # Different
    
    # Track rewrite attempts
    rewrite_attempts = []
    
    # Mock get_snapshot to return different snapshots in sequence
    with patch('planner.context.get_snapshot', side_effect=[snapshot1, snapshot2, snapshot3]):
        # Mock Verifier.check to return 'reimplement' then 'ok'
        with patch('planner.verifier.Verifier.check', side_effect=['reimplement', 'ok']):
            # Mock CodeRewriter.rewrite_function to track calls
            with patch('planner.code_rewriter.rewrite_function') as mock_rewrite:
                def track_rewrite(fn):
                    rewrite_attempts.append(fn.__name__)
                
                mock_rewrite.side_effect = track_rewrite
                
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
                
                # Assert at least one rewrite was attempted
                assert len(rewrite_attempts) >= 1
                assert "foo" in rewrite_attempts


def test_verify_push_up():
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
    with patch('planner.context.get_snapshot', return_value=snapshot):
        # Mock Verifier.check to return 'push_up_stack'
        with patch('planner.verifier.Verifier.check', return_value='push_up_stack'):
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
