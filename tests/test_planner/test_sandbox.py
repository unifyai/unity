import pytest
import queue
from planner.sandbox import exec_plan, SecurityError
from planner.model import Primitive
from planner.primitives import set_queues
import threading
import time


def test_exec_plan_returns_callable_root_plan():
    src = (
        "from planner.primitives import open_browser"
        "\n"
        "root_plan = lambda: open_browser()"
    )

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

    plan_module = exec_plan(src)
    assert hasattr(plan_module, "root_plan"), "Module should have 'root_plan' attribute"
    assert callable(plan_module.root_plan), "'root_plan' should be callable"

    result = plan_module.root_plan()
    assert isinstance(result, Primitive), "Result of root_plan() should be a Primitive"

    # Cleanup
    thread.join(timeout=1)


@pytest.mark.parametrize(
    "malicious_code",
    [
        "import os",
        "__import__('socket')",
    ],
)
def test_exec_plan_raises_security_error(malicious_code):
    with pytest.raises(SecurityError):
        exec_plan(malicious_code)
