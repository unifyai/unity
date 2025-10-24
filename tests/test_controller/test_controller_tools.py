import os
import pytest
from pathlib import Path
import base64
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS
from unity.controller.controller import Controller
from unity.controller.playwright_utils.worker import BrowserWorker

# Use the same model as other tests (override via UNIFY_MODEL env)
MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


@pytest.mark.asyncio
@_handle_project
async def test_controller_act_tool_loop():
    """
    Verify that the Controller.act method can be used as a tool
    within the async-tool-use loop to perform a browser action.
    """
    client = unify.AsyncUnify(
        MODEL_NAME,
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    controller = Controller()
    controller._observe_ctx = {"state": {"in_textbox": True}}

    # Initialize the browser worker since we're not running the controller thread
    controller._browser_worker = BrowserWorker(
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        session_connect_url=controller.session_connect_url,
        headless=True,
        mode=controller._mode,
        debug=controller._debug,
    )

    # Run the loop with only the 'act' tool
    result = await start_async_tool_loop(
        client,
        message=f"Call `act` with request 'type hello in dutch', and return the executed action.",
        tools={"act": controller.act},
    ).result()

    # Expect the action command to appear in the result
    assert "enter_text" in result.lower()


@pytest.mark.asyncio
@_handle_project
async def test_controller_observe_tool_loop():
    """
    Verify that the Controller.observe method can be used as a tool
    within the async-tool-use loop to answer a simple question.
    """
    # Create a fresh AsyncUnify client
    client = unify.AsyncUnify(
        MODEL_NAME,
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    # Instantiate Controller and prime minimal context
    controller = Controller()
    controller._observe_ctx = {"state": {}}
    controller._last_shot = b""

    # Start the async tool-use loop with only the 'observe' tool
    answer = await start_async_tool_loop(
        client,
        message="Use the `observe` tool to determine if 2+2 equals 4, then return the result.",
        tools={"observe": controller.observe},
    ).result()

    # The tool returns a boolean; ensure the answer contains 'true'
    assert any(token in answer.lower() for token in ("true", "yes"))


@pytest.mark.slow
@pytest.mark.asyncio
@_handle_project
async def test_controller_complex_tool_loop():
    """
    Verify that the Controller.observe method can be used as a tool
    within the async-tool-use loop to answer a simple question.
    """
    # Create a fresh AsyncUnify client
    client = unify.AsyncUnify(
        MODEL_NAME,
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    # Instantiate Controller and prime minimal context
    controller = Controller()
    controller._observe_ctx = {"state": {"in_textbox": True}}

    # Initialize the browser worker since we're not running the controller thread
    controller._browser_worker = BrowserWorker(
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        session_connect_url=controller.session_connect_url,
        headless=True,
        mode=controller._mode,
        debug=controller._debug,
    )

    raw_jpeg = Path("tests/test_controller/test_images/google.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    controller._last_shot = b64

    # Start the async tool-use loop with only the 'observe' tool
    answer = await start_async_tool_loop(
        client,
        message="""
        Call `observe` to determine if the page is on Google.
        If false, call `act` with request 'go to google.com',
        else if true, call `act` with request 'type hello in dutch', wait for the call to complete.
        Then, call `act` with request 'hold ctrl, press cursor left twice, release ctrl, then hold ctrl, press delete, release ctrl', and return the executed action.
        """,
        tools={"observe": controller.observe, "act": controller.act},
    ).result()

    # The tool should execute commands involving ctrl, cursor_left, and delete
    # Check that the answer contains relevant keywords
    answer_lower = answer.lower()
    assert any(
        token in answer_lower
        for token in (
            "ctrl",
            "control",
            "cursor",
            "left",
            "delete",
            "hold",
            "release",
            "executed",
            "action",
        )
    )
