import inspect
import pytest

from planner.zero_shot import create_initial_plan
import planner.zero_shot as zs


def test_single_primitive_and_stub(monkeypatch):
    # Mock the LLM response with multiple primitives in root_plan and one in helper
    snippet = """
    def root_plan():
        open_browser("url1")
        open_browser("url2")

    def helper_func():
        open_gmail()
    """
    monkeypatch.setattr(zs, "generate_prompt", lambda prompt: snippet)

    # Generate and process the plan
    module, root_fn = create_initial_plan("any task")

    # Retrieve the source code of the generated module
    src = inspect.getsource(module)
    # Only the first primitive call should remain in root_plan
    assert src.count("open_browser(") == 1
    # The helper without an initial primitive should be stubbed
    assert "raise NotImplementedError" in src
    # The helper function should be decorated with @verify
    assert src.startswith("@verify")
