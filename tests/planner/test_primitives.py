import queue
import threading
import time
import pytest

from planner.primitives import set_queues, open_browser
from planner.model import Primitive


def test_open_browser_blocks_until_ack():
    # Setup the command and acknowledgment queues
    text_q = queue.Queue()
    ack_q = queue.Queue()
    set_queues(text_q, ack_q)

    # Container to capture the call_literal from the queue
    ack_cmd = []

    # Background thread to simulate controller acknowledgment
    def ack_handler():
        # Wait for the command literal enqueued by open_browser
        cmd = text_q.get(timeout=1)
        ack_cmd.append(cmd)
        # Simulate processing delay
        time.sleep(0.1)
        # Acknowledge the command
        ack_q.put(cmd)

    thread = threading.Thread(target=ack_handler, daemon=True)
    thread.start()

    # Call the primitive and measure blocking duration
    start_time = time.time()
    primitive = open_browser()
    duration = time.time() - start_time

    # Ensure the call was blocked until acknowledgment
    assert duration >= 0.1, f"Primitive returned too quickly: {duration:.3f}s"

    # Validate the primitive returned correctly
    assert isinstance(primitive, Primitive)
    assert ack_cmd, "No command was captured by the ack handler"
    assert primitive.call_literal == ack_cmd[0]

    # Ensure queues are empty after processing
    assert text_q.empty(), "Text queue should be empty after primitive execution"
    assert ack_q.empty(), "Ack queue should be empty after primitive execution"

    # Cleanup
    thread.join(timeout=1)
