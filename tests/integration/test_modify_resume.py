import pytest
import queue
import asyncio
import threading
import time
import types

from planner.planner import Planner
import planner.zero_shot as zs
import planner.primitives as primitives
import planner.update_handler as uh


def test_modify_resume(monkeypatch):
    # Stub zero_shot.create_initial_plan to return a simple plan
    def fake_create_initial_plan(task_description):
        module = types.ModuleType("fake_plan_module")

        def root_fn():
            # Simple plan actions
            primitives.dummy_action("open google")
            primitives.dummy_action("search cats")

        module.root_fn = root_fn
        return module, root_fn

    monkeypatch.setattr(zs, "create_initial_plan", fake_create_initial_plan)

    # Stub primitives.set_queues and define dummy_action
    queues = {}

    def fake_set_queues(text_action_q, action_completion_q):
        queues["text_action_q"] = text_action_q
        queues["action_completion_q"] = action_completion_q

    def dummy_action(action):
        # Send action and immediately acknowledge
        queues["text_action_q"].put(action)
        queues["action_completion_q"].put("ack")

    monkeypatch.setattr(primitives, "set_queues", fake_set_queues)
    monkeypatch.setattr(primitives, "dummy_action", dummy_action)

    # Stub update_handler.generate_prompt and rewrite_function
    def fake_generate_prompt(prompt):
        if "course_correction" in prompt:
            return "def course_correction():\n    pass"
        return ""

    monkeypatch.setattr(uh, "generate_prompt", fake_generate_prompt)
    monkeypatch.setattr(uh, "rewrite_function", lambda *args, **kwargs: None)

    # Spy on schedule_and_resume for course_correction
    scheduled = {}

    def fake_schedule_and_resume(planner, fn_name):
        scheduled["fn_name"] = fn_name

    monkeypatch.setattr(uh, "_schedule_and_resume", fake_schedule_and_resume)

    # Set up queues and event loop
    task_update_q = queue.Queue()
    text_action_q = queue.Queue()
    action_completion_q = queue.Queue()
    task_completion_q = asyncio.Queue()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    # Start planner thread
    planner = Planner(
        task_update_q, text_action_q, action_completion_q, task_completion_q, loop
    )
    planner.start()

    # Send initial plan request
    task_update_q.put("Open Google and search cats")

    # Verify plan actions executed
    assert queues["text_action_q"].get(timeout=5) == "open google"
    assert queues["text_action_q"].get(timeout=5) == "search cats"

    # Wait for task completion
    status = loop.run_until_complete(task_completion_q.get())
    assert "Task completed" in status

    # Send modification update
    task_update_q.put("Actually search dogs not cats")
    # Allow time for modification handler to schedule course_correction
    time.sleep(0.5)

    # Planner thread should still be alive
    assert planner.is_alive()

    # Confirm course_correction was scheduled
    assert scheduled.get("fn_name") == "course_correction"
