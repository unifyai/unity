import queue
import asyncio

from planner.planner import Planner
from planner.context import context
from planner import primitives


class DummyLoop:
    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


def test_exploration_done(monkeypatch):
    captured = []
    # Stub primitives to capture calls
    monkeypatch.setattr(primitives, "close_this_tab", lambda: captured.append("close"))
    monkeypatch.setattr(
        primitives, "select_tab", lambda tab: captured.append(f"select:{tab}")
    )
    # Stub context to return a specific main_tab
    monkeypatch.setattr(context, "exit_exploration", lambda: "mainTab")
    monkeypatch.setattr(context, "get_exploration_tab", lambda: "exploreTab")

    # Set up queues and planner
    task_update_queue = queue.Queue()
    text_action_q = queue.Queue()
    action_completion_q = queue.Queue()
    task_completion_q = asyncio.Queue()
    loop = DummyLoop()

    planner = Planner(
        task_update_queue,
        text_action_q,
        action_completion_q,
        task_completion_q,
        loop,
        browser_state_broadcast_q=None,
        daemon=False,
    )

    # Enqueue exploration done event and termination signal
    task_update_queue.put("__exploration_done__")
    task_update_queue.put(None)

    # Run planner for one iteration
    planner.run()

    # Verify calls order: close_this_tab then select_tab(mainTab)
    assert captured == ["close", "select:mainTab"]
