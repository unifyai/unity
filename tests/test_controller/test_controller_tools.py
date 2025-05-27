import types, sys  # noqa: E402 (stubs before imports)
import os
import pytest
from pathlib import Path
import base64
import unify
import unity.common.llm_helpers as llmh
from tests.helpers import _handle_project
import json


# --- Redis stub -----------------------------------------------------------
class _FakePubSub:
    def __init__(self):
        self._messages = []

    def subscribe(self, *_):
        pass

    def listen(self):
        # generator expected by Controller.run(); empty -> instant end if used
        while self._messages:
            yield self._messages.pop()
        while True:
            yield {"type": "noop"}

    def get_message(self):
        return None


class _FakeRedis:
    def __init__(self, *a, **k):
        self._pubsub = _FakePubSub()
        self.published: list[tuple[str, str]] = []

    def pubsub(self):
        return self._pubsub

    def publish(self, chan, msg):
        self.published.append((chan, msg))


sys.modules.setdefault("redis", types.ModuleType("redis"))
sys.modules["redis"].Redis = _FakeRedis


# --- BrowserWorker stub ---------------------------------------------------
class _DummyWorker:
    def __init__(self, *a, **k):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, *a, **k):
        pass


# ensure parent package module exists
pkg_path = "unity.controller.playwright"
if pkg_path not in sys.modules:
    sys.modules[pkg_path] = types.ModuleType("playwright_stub")
worker_mod = types.ModuleType("worker")
worker_mod.BrowserWorker = _DummyWorker
sys.modules["unity.controller.playwright.worker"] = worker_mod

from unity.controller.controller import Controller

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
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    controller = Controller()
    controller._observe_ctx = {"state": {"in_textbox": True}}

    # Run the loop with only the 'act' tool
    result = await llmh.start_async_tool_use_loop(
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
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    # Instantiate Controller and prime minimal context
    controller = Controller()
    controller._observe_ctx = {"state": {}}
    controller._last_shot = b""

    # Start the async tool-use loop with only the 'observe' tool
    answer = await llmh.start_async_tool_use_loop(
        client,
        message="Use the `observe` tool to determine if 2+2 equals 4, then return the result.",
        tools={"observe": controller.observe},
    ).result()

    # The tool returns a boolean; ensure the answer contains 'true'
    assert any(token in answer.lower() for token in ("true", "yes"))


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
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
    )
    client.set_system_message("Feel free to call multiple tools per turn.")

    # Instantiate Controller and prime minimal context
    controller = Controller()
    controller._observe_ctx = {"state": {"in_textbox": True}}

    raw_jpeg = Path("tests/test_controller/test_images/google.jpeg").read_bytes()
    b64 = base64.b64encode(raw_jpeg).decode("utf-8")
    controller._last_shot = b64

    # Start the async tool-use loop with only the 'observe' tool
    answer = await llmh.start_async_tool_use_loop(
        client,
        message="""
        Call `observe` to determine if the page is on Google.
        If false, call `act` with request 'go to google.com',
        else if true, call `act` with request 'type hello in dutch', wait for the call to complete.
        Then, call `act` with request 'move caret two words to the left and delete a word to the right', and return the executed action.
        """,
        tools={"observe": controller.observe, "act": controller.act},
    ).result()

    # The tool returns a boolean; ensure the answer contains 'true'
    assert all(
        token in answer.lower() for token in ("ctrl", "cursor left twice", "delete")
    )
