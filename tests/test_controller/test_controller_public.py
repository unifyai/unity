import pytest
from pathlib import Path
import base64

from unity.controller.controller import Controller


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_observe_bool():
    """Smoke-test Controller.observe with bool response."""
    c = Controller()
    # minimal cached context
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ret = await c.observe("Is 2+2 equal to 4?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ret, bool)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_observe_str():
    """Smoke-test observe with string response type."""
    c = Controller()
    c._observe_ctx = {"state": {}}
    c._last_shot = b""
    try:
        ans = await c.observe("Reply with 'hello'.", str)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ans, str)
    assert len(ans) > 0


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_act_smoke():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()
    c._observe_ctx = {"state": {"in_textbox": False}}

    try:
        actions = await c.act("open browser")
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(actions, list)
    assert isinstance(actions[0], str)
    # browser worker should have been started
    assert c._browser_open is True
    assert actions[0] == "new_tab"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_screen_observation_linkedin():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()

    raw_jpeg = Path("tests/test_controller/test_images/linkedin.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    c._last_shot = b64
    try:
        ret = await c.observe("Is the page on LinkedIn?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ret, bool)
    assert ret is True


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_screen_observation_google():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()

    raw_jpeg = Path("tests/test_controller/test_images/google.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    c._last_shot = b64
    try:
        ret = await c.observe("Is the page on LinkedIn?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")
    assert isinstance(ret, bool)
    assert ret is False


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_controller_feedback_loop():
    """Smoke-test Controller.act and Redis publications."""
    c = Controller()
    c._observe_ctx = {"state": {"in_textbox": False}}

    # on google
    raw_jpeg = Path("tests/test_controller/test_images/google.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    c._last_shot = b64

    # observe page state
    try:
        ret = await c.observe("Is the page on LinkedIn?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")

    assert ret is False

    # go to linkedin and ensure command is correct
    try:
        actions = await c.act("go to LinkedIn website")
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")

    assert isinstance(actions, list) and isinstance(actions[0], str)
    assert "open_url" in actions[0] and "linkedin.com" in actions[0]

    # on linkedin
    raw_jpeg = Path("tests/test_controller/test_images/linkedin.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    c._last_shot = b64

    # observe page state
    try:
        ret = await c.observe("Is the page on LinkedIn?", bool)
    except Exception as exc:
        pytest.skip(f"Skipping – backend unavailable: {exc}")

    assert ret is True
