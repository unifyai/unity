import pytest

import planner.update_handler as uh


def test_select_stack_function_valid_choice(monkeypatch):
    # Setup a known call stack
    monkeypatch.setattr(
        uh.context, "get_call_stack", lambda: ["func_a", "func_b", "func_c"]
    )
    # Stub generate_prompt to return a valid choice
    monkeypatch.setattr(uh, "generate_prompt", lambda prompt: "func_b")
    # Stub system and stateful calls
    monkeypatch.setattr(uh, "set_system_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(uh, "_set_stateful", lambda *args, **kwargs: None)
    # Assert valid choice is returned
    assert uh._select_stack_function("update text") == "func_b"


def test_select_stack_function_out_of_stack_choice(monkeypatch):
    # Setup a known call stack
    monkeypatch.setattr(
        uh.context, "get_call_stack", lambda: ["func_x", "func_y", "func_z"]
    )
    # Stub generate_prompt to return an out-of-stack name
    monkeypatch.setattr(uh, "generate_prompt", lambda prompt: "unknown_func")
    monkeypatch.setattr(uh, "set_system_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(uh, "_set_stateful", lambda *args, **kwargs: None)
    # Assert fallback to top-most frame
    assert uh._select_stack_function("update text") == "func_z"


def test_select_stack_function_non_integer_choice(monkeypatch):
    # Setup a known call stack
    monkeypatch.setattr(uh.context, "get_call_stack", lambda: ["x", "y", "z"])
    # Stub generate_prompt to return a non-integer string
    monkeypatch.setattr(uh, "generate_prompt", lambda prompt: "123")
    monkeypatch.setattr(uh, "set_system_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(uh, "_set_stateful", lambda *args, **kwargs: None)
    # Assert fallback to top-most frame
    assert uh._select_stack_function("update text") == "z"


def test_select_stack_function_empty_stack(monkeypatch):
    # Setup an empty call stack
    monkeypatch.setattr(uh.context, "get_call_stack", lambda: [])
    # Assert None is returned when stack is empty
    assert uh._select_stack_function("update text") is None
