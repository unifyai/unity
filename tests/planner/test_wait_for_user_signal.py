import threading
import time
import queue
import pytest

from planner import primitives
from planner.model import Primitive


def test_wait_for_user_signal_blocks_until_event_set():
    # Initialize the pause event and call queue
    pause_event = threading.Event()
    call_queue = queue.Queue()
    # Set runtime controls in primitives module
    primitives.set_runtime_controls(pause_event=pause_event, call_queue=call_queue)
    # Ensure the event is cleared before starting
    pause_event.clear()

    # Launch a background thread that sets the event after a delay
    delay = 0.1

    def set_event_after_delay():
        time.sleep(delay)
        pause_event.set()

    setter_thread = threading.Thread(target=set_event_after_delay)
    setter_thread.start()

    # Record the start time
    start_time = time.monotonic()
    # Call the raw wait_for_user_signal primitive (blocks until event is set)
    primitive = primitives._raw_wait_for_user_signal(timeout=None)
    # Record the end time
    end_time = time.monotonic()
    setter_thread.join()

    duration = end_time - start_time
    # Assert that the call blocked at least as long as the delay
    assert duration >= delay, f"wait_for_user_signal returned too early: {duration}s"
    # Assert that the returned object is a Primitive with the expected name
    assert isinstance(primitive, Primitive)
    assert primitive.name == "wait_for_user_signal"
