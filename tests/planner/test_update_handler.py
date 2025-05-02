import queue
import asyncio
import types
import threading

import pytest

from planner.planner import Planner
from planner import update_handler


def test_handle_update_modify_calls_rewrite_and_schedules_callable(monkeypatch):
    # Prepare dummy planner with default pause_event (set) and empty call_queue
    text_update_q = queue.Queue()
    text_action_q = queue.Queue()
    action_completion_q = queue.Queue()
    task_completion_q = asyncio.Queue()
    loop = asyncio.new_event_loop()

    planner = Planner(
        task_update_queue=text_update_q,
        text_action_q=text_action_q,
        action_completion_q=action_completion_q,
        task_completion_q=task_completion_q,
        coms_asyncio_loop=loop,
        daemon=True,
    )

    # Ensure initial state
    assert planner._pause_event.is_set()
    assert planner._call_queue.qsize() == 0

    # Create dummy module with a target function
    dummy_module = types.ModuleType("dummy_module")

    def dummy_function():
        pass

    dummy_module.dummy_function = dummy_function
    planner._plan_module = dummy_module

    # Patch context.get_call_stack to return empty stack
    monkeypatch.setattr(update_handler.context, "get_call_stack", lambda: [])

    # Force modify mode so we don't consume a fake prompt in classification
    monkeypatch.setattr(update_handler, "_classify_update", lambda txt: "modify")

    # Capture calls to rewrite_function
    rewritten = {}

    def fake_rewrite_function(func, new_code=None):
        rewritten["func"] = func

    monkeypatch.setattr(update_handler, "rewrite_function", fake_rewrite_function)

    # Patch generate_prompt to first return the target function name, then course correction code
    prompts = []

    def fake_generate_prompt(prompt_text):
        prompts.append(prompt_text)
        if len(prompts) == 1:
            return "dummy_function"
        # Second call returns a simple course_correction definition
        return "def course_correction():\n    pass"

    monkeypatch.setattr(update_handler, "generate_prompt", fake_generate_prompt)

    # Patch set_system_message and _set_stateful to no-op
    monkeypatch.setattr(update_handler, "set_system_message", lambda msg: None)
    monkeypatch.setattr(update_handler, "_set_stateful", lambda state: None)

    # Execute handle_update with a modify request
    update_handler.handle_update(planner, "modify request text")

    # Assertions
    # rewrite_function should have been called on the dummy_function
    assert rewritten.get("func") is dummy_function
    # Planner should remain paused until wrapper is executed
    assert planner.paused is True
    # The call_queue should contain exactly one callable (course_correction)
    assert planner._call_queue.qsize() == 1
    wrapper = planner._call_queue.get_nowait()
    assert callable(wrapper)

    # Execute the wrapper function
    wrapper()

    # After wrapper execution, planner should be unpaused
    assert planner.paused is False


def test_handle_update_exploration_schedules_and_resumes(monkeypatch):
    # Setup a Planner instance with its pause_event set and empty call_queue
    text_update_q = queue.Queue()
    text_action_q = queue.Queue()
    action_completion_q = queue.Queue()
    task_completion_q = asyncio.Queue()
    loop = asyncio.new_event_loop()

    planner = Planner(
        task_update_queue=text_update_q,
        text_action_q=text_action_q,
        action_completion_q=action_completion_q,
        task_completion_q=task_completion_q,
        coms_asyncio_loop=loop,
        daemon=True,
    )

    # Ensure initial state
    assert planner._pause_event.is_set()
    assert planner._call_queue.qsize() == 0

    # Create dummy module for the planner
    dummy_module = types.ModuleType("dummy_module")
    planner._plan_module = dummy_module

    # Force classification to exploratory
    monkeypatch.setattr(update_handler, "_classify_update", lambda txt: "exploratory")

    # Patch context.last_state_snapshot to return a dummy browser state
    monkeypatch.setattr(
        update_handler.context,
        "last_state_snapshot",
        lambda: {"url": "https://example.com", "active_tab": "tab1"},
    )

    # Patch generate_prompt to return valid exploratory_task code
    def fake_generate_prompt(prompt_text):
        return "def exploratory_task():\n    pass"

    monkeypatch.setattr(update_handler, "generate_prompt", fake_generate_prompt)

    # Patch set_system_message and _set_stateful to no-op
    monkeypatch.setattr(update_handler, "set_system_message", lambda msg: None)
    monkeypatch.setattr(update_handler, "_set_stateful", lambda state: None)

    # Perform the update
    update_handler.handle_update(planner, "some exploration")

    # After handling, root plan should remain paused
    assert planner.paused

    # Exactly one wrapper callable enqueued
    assert planner._call_queue.qsize() == 1
    wrapper = planner._call_queue.get_nowait()
    assert callable(wrapper)

    # Running the wrapper should resume the planner
    wrapper()
    assert not planner.paused
