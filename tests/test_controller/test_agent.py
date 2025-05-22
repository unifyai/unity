import pytest

from unity.controller import agent as agent_mod
from unity.controller.states import BrowserState


@pytest.mark.timeout(30)
def test_text_to_browser_action():
    """Smoke-test that agent.text_to_browser_action returns a dict with keys.
    Relies on online Unify backend; will skip when network/API not available."""
    try:
        result = agent_mod.text_to_browser_action(
            "open browser",
            screenshot=None,
            tabs=[],
            buttons=None,
            history=[],
            state=BrowserState(),
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(result, dict)
    assert "action" in result
    assert "rationale" in result
    assert "open" in result["rationale"] and "browser" in result["rationale"]
    assert "new_tab" in result["action"]


@pytest.mark.timeout(30)
def test_text_to_browser_action_multi_step():
    """Smoke-test multi-step command generation.
    Relies on online Unify backend; will skip when network/API not available."""
    try:
        # Set the state to be in a textbox as key presses only available in textbox
        test_state = BrowserState()
        test_state.in_textbox = True
        
        result = agent_mod.text_to_browser_action(
            "select 3 characters to the left",
            screenshot=None,
            tabs=[],
            buttons=None,
            history=[],
            state=test_state,
            multi_step_mode=True,
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")
    # Should return a dict with 'action' list and 'rationale'
    assert isinstance(result, dict)
    assert "action" in result and "rationale" in result
    actions = result["action"]
    assert isinstance(actions, list)
    assert len(actions) >= 4
    # Expect first command to hold_shift, then three cursor_left
    assert actions[0] == "hold_shift"
    assert actions[1:4] == ["cursor_left", "cursor_left", "cursor_left"]


@pytest.mark.timeout(30)
def test_ask_llm_bool():
    """Smoke-test ask_llm with boolean response_type. Skips when backend unavailable."""
    try:
        answer = agent_mod.ask_llm("Is 2+2 equal to 4?", response_type=bool)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, bool)


@pytest.mark.timeout(30)
def test_ask_llm_str():
    """ask_llm should return a plain string when response_type=str"""
    try:
        answer = agent_mod.ask_llm("Say hello in one word.", response_type=str)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, str)
    assert len(answer) > 0


from pydantic import BaseModel, Field


class _Coords(BaseModel):
    lat: float = Field(..., description="latitude")
    lon: float = Field(..., description="longitude")


@pytest.mark.timeout(30)
def test_ask_llm_custom_model():
    """ask_llm should handle arbitrary Pydantic response models."""
    try:
        ret = agent_mod.ask_llm(
            "Provide the coordinates of the Eiffel Tower as JSON with 'lat' and 'lon'.",
            response_type=_Coords,
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    # Because our custom model has no 'answer' attribute, ask_llm returns the model instance
    assert isinstance(ret, _Coords)
    assert -90 <= ret.lat <= 90
    assert -180 <= ret.lon <= 180


@pytest.mark.timeout(30)
def test_ask_llm_int():
    """ask_llm should return an int when response_type=int"""
    try:
        answer = agent_mod.ask_llm("What is 10 minus 3?", response_type=int)
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, int)


@pytest.mark.timeout(30)
def test_ask_llm_float():
    """ask_llm should return a float when response_type=float"""
    try:
        answer = agent_mod.ask_llm(
            "Provide 1 divided by 3 as a decimal.",
            response_type=float,
        )
    except Exception as exc:
        pytest.skip(f"Skipping – Unify backend unavailable: {exc}")

    assert isinstance(answer, float)
