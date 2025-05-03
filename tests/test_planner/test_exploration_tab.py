import queue
import types
import pytest

from planner import primitives, context, update_handler
from planner.update_handler import _select_stack_function, context


@pytest.fixture(autouse=True)
def stub_generate_prompt(monkeypatch):
    """
    Stub the generate_prompt function to always return a simple exploratory_task.
    """
    code = (
        "def exploratory_task():\n"
        "    # Use primitives in the stub task\n"
        "    new_tab()\n"
        "    search('dummy query')\n"
        "    wait_for_user_signal()\n"
        "    return None\n"
    )
    monkeypatch.setattr(update_handler, "generate_prompt", lambda prompt: code)


@pytest.fixture
def stub_primitives_and_context(monkeypatch):
    """
    Stub primitives and context to control and count new_tab calls,
    and simulate exploration tab reuse behavior.
    """
    new_tab_calls = []

    # Count new_tab calls
    def fake_new_tab():
        new_tab_calls.append(True)
        return "explore_tab_id"

    monkeypatch.setattr(primitives, "new_tab", fake_new_tab)

    # Stub other browser primitives to no-ops
    monkeypatch.setattr(primitives, "search", lambda query: None)
    monkeypatch.setattr(primitives, "wait_for_user_signal", lambda: None)
    monkeypatch.setattr(primitives, "select_tab", lambda tab_id: None)

    # last_state_snapshot returns an original active tab
    monkeypatch.setattr(
        context, "last_state_snapshot", lambda: {"active_tab": "orig_tab"}
    )

    # get_exploration_tab returns None on first call, then reuse the same tab id
    call_count = {"n": 0}

    def fake_get_exploration_tab():
        call_count["n"] += 1
        return "explore_tab_id" if call_count["n"] > 1 else None

    monkeypatch.setattr(context, "get_exploration_tab", fake_get_exploration_tab)

    # enter_exploration and exit_exploration do nothing
    monkeypatch.setattr(context, "enter_exploration", lambda tab_id: None)
    monkeypatch.setattr(context, "exit_exploration", lambda: None)

    return new_tab_calls


def test_exploration_tab_reuse(stub_primitives_and_context):
    """
    Simulate two exploratory updates and verify that only one new_tab is called,
    ensuring exploration tab reuse.
    """

    class DummyPlanner:
        def __init__(self):
            self._plan_module = types.SimpleNamespace()
            self._call_queue = queue.Queue()

        def _pause(self):
            pass

        def _resume(self):
            pass

    planner = DummyPlanner()

    # First exploratory update
    update_handler.handle_update(planner, "explore something")
    # Second exploratory update
    update_handler.handle_update(planner, "explore another thing")

    # Execute the queued exploratory tasks
    for _ in range(2):
        fn = planner._call_queue.get()
        fn()

    # Assert new_tab was called only once
    assert len(stub_primitives_and_context) == 1
