import queue
import threading
import time
import pytest

from planner.sandbox import exec_plan
from planner.primitives import set_queues
from planner.planner import Planner
from planner.verifier import verify


class DummyLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class DummyPlanner:
    def __init__(self, completion_q):
        self._task_completion_q = completion_q
        self._coms_asyncio_loop = DummyLoop()


def test_root_plan_executes_and_completes():
    # Set up command, ack, and completion queues
    text_q = queue.Queue()
    ack_q = queue.Queue()
    done_q = queue.Queue()

    # Initialize primitives to use our queues
    set_queues(text_q, ack_q)

    # Controller stub: echo back each command after a short delay
    def ack_worker():
        while True:
            cmd = text_q.get()
            time.sleep(0.01)
            ack_q.put(cmd)

    threading.Thread(target=ack_worker, daemon=True).start()

    # Create and compile a minimal plan
    src = """
from planner.verifier import verify
from planner.primitives import open_browser

@verify
def root_fn():
    open_browser()
"""
    module = exec_plan(src)
    root_fn = module.root_fn

    # Run the planner wrapper and verify completion
    planner = DummyPlanner(done_q)
    planner._run_plan_wrapper = Planner._run_plan_wrapper.__get__(planner)
    planner._run_plan_wrapper(root_fn)
    assert done_q.get_nowait() == "Task completed successfully"
