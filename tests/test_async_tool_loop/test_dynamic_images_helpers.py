from __future__ import annotations

import asyncio
import base64
import json
import pytest

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project


class _SpyClient:
    """Minimal AsyncUnify-compatible stub used by these tests.

    Mirrors the subset used by async tool loop tests: a `messages` list,
    `append_messages`, and a `system_message` property and setter.
    """

    def __init__(self):
        self.messages: list[dict] = []
        self._system: str = ""

    def append_messages(self, msgs):
        self.messages.extend(msgs)

    @property
    def system_message(self) -> str:
        return self._system

    def set_system_message(self, msg: str):
        self._system = msg
        return self


def _solid_png_bytes(r: int, g: int, b: int) -> bytes:
    from unity.image_manager.utils import make_solid_png_base64

    b64 = make_solid_png_base64(2, 2, (r, g, b))
    return base64.b64decode(b64)


class _DummyImage:
    def __init__(self, *, data: str):
        self.data = data


class DummyImageHandle:
    """Lightweight test double that mirrors the ImageHandle surface we use."""

    def __init__(self, *, image_id: int, caption: str | None, raw_bytes: bytes):
        # Non-URL data → forces raw() path in attach helper
        self._image = _DummyImage(data="")
        self._raw = bytes(raw_bytes)
        self._image_id = int(image_id)
        self._caption = caption

    @property
    def image_id(self) -> int:
        return self._image_id

    @property
    def caption(self) -> str | None:
        return self._caption

    def raw(self) -> bytes:
        return self._raw

    async def ask(self, question: str):
        # Deterministic canned answer for tests
        # We return BLUE irrespective of the question for simplicity
        return "BLUE"


def _find_tool_name(tools: list[dict], prefix: str) -> str | None:
    for t in tools or []:
        try:
            n = t.get("function", {}).get("name")
            if isinstance(n, str) and n.startswith(prefix):
                return n
        except Exception:
            continue
    return None


@pytest.mark.asyncio
@_handle_project
async def test_interject_dynamic_helper_appends_images(monkeypatch) -> None:
    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            # Schedule base tool `do_work`
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_WORK",
                        "type": "function",
                        "function": {"name": "do_work", "arguments": "{}"},
                    },
                ],
            }
        elif step["n"] == 1:
            # Let the LLM take a turn after the tool emits a notification;
            # on this second turn, request the interject helper.
            step["n"] += 1
            interject_name = _find_tool_name(tools, "interject_")
            assert interject_name, "interject helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_INTERJECT",
                        "type": "function",
                        "function": {
                            "name": interject_name,
                            "arguments": json.dumps(
                                {
                                    "content": "please proceed",
                                    "images": {"this[:]": 42},
                                },
                            ),
                        },
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool that emits a notification (to grant the LLM a turn),
    # then waits for interjection and finishes
    async def do_work(
        *,
        interject_queue: asyncio.Queue[str],
        notification_up_q: asyncio.Queue[dict],
    ):
        await notification_up_q.put({"message": "ready"})
        _ = await interject_queue.get()
        return {"ok": True}

    client = _SpyClient()
    images = {
        "[0:5]": DummyImageHandle(
            image_id=42,
            caption="blue square",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={"do_work": do_work},
        images=images,
    )

    await h.result()

    # After interjection helper with images, the overview doc should include an appended entry
    assert tools_snapshots, "No LLM call captured"
    last_tools = tools_snapshots[-1]
    live_tool = next(
        t for t in last_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "source=interjection" in desc
    assert "id=42" in desc


@pytest.mark.asyncio
@_handle_project
async def test_stop_dynamic_helper_appends_images(monkeypatch) -> None:
    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_WAIT",
                        "type": "function",
                        "function": {"name": "wait_forever", "arguments": "{}"},
                    },
                ],
            }
        elif step["n"] == 1:
            # After tool notification, LLM gets a turn to call stop helper
            step["n"] += 1
            stop_name = _find_tool_name(tools, "stop_")
            assert stop_name, "stop helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_STOP",
                        "type": "function",
                        "function": {
                            "name": stop_name,
                            "arguments": json.dumps({"images": {"this[:]": 42}}),
                        },
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool that emits a notification (to grant the LLM a turn),
    # then waits forever until stopped
    async def wait_forever(*, notification_up_q: asyncio.Queue[dict]):
        await notification_up_q.put({"message": "starting"})
        await asyncio.Event().wait()
        return {"ok": False}

    client = _SpyClient()
    images = {
        "[0:3]": DummyImageHandle(
            image_id=42,
            caption="blue tile",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hey",
        tools={"wait_forever": wait_forever},
        images=images,
    )

    await h.result()

    last_tools = tools_snapshots[-1]
    live_tool = next(
        t for t in last_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "source=stop" in desc
    assert "id=42" in desc


@pytest.mark.asyncio
@_handle_project
async def test_clarify_helpers_append_images_for_request_and_answer(
    monkeypatch,
) -> None:
    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_NEEDCLAR",
                        "type": "function",
                        "function": {"name": "need_clar", "arguments": "{}"},
                    },
                ],
            }
        elif step["n"] == 1:
            step["n"] += 1
            clarify_name = _find_tool_name(tools, "clarify_")
            assert clarify_name, "clarify helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_CLARIFY",
                        "type": "function",
                        "function": {
                            "name": clarify_name,
                            "arguments": json.dumps(
                                {"answer": "blue", "images": {"this[:]": 42}},
                            ),
                        },
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    async def need_clar(
        *,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ) -> dict:
        # Send a clarification request with images
        await clarification_up_q.put(
            {"question": "What is the dominant color?", "images": {"this[:]": 42}},
        )
        ans = await clarification_down_q.get()
        return {"answer": ans}

    client = _SpyClient()
    images = {
        "[0:5]": DummyImageHandle(
            image_id=42,
            caption="blue square",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello",
        tools={"need_clar": need_clar},
        images=images,
    )

    await h.result()

    last_tools = tools_snapshots[-1]
    live_tool = next(
        t for t in last_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "source=clar_request" in desc
    assert "source=clar_answer" in desc
    assert "id=42" in desc


@pytest.mark.asyncio
@_handle_project
async def test_notification_payload_appends_images(monkeypatch) -> None:
    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_NOTIFY",
                        "type": "function",
                        "function": {"name": "notify", "arguments": "{}"},
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    async def notify(*, notification_up_q: asyncio.Queue[dict]) -> dict:
        await notification_up_q.put({"message": "progress", "images": {"this[:]": 42}})
        # small yield to allow loop to observe notification while task is pending
        await asyncio.sleep(0.01)
        return {"ok": True}

    client = _SpyClient()
    images = {
        "[0:2]": DummyImageHandle(
            image_id=42,
            caption="blue tile",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Go",
        tools={"notify": notify},
        images=images,
    )

    await h.result()

    last_tools = tools_snapshots[-1]
    live_tool = next(
        t for t in last_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "source=notification" in desc
    assert "id=42" in desc


@pytest.mark.asyncio
@_handle_project
async def test_ask_image_with_images_param_appends_log(monkeypatch) -> None:
    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_IMG_ASK",
                        "type": "function",
                        "function": {
                            "name": "ask_image",
                            "arguments": json.dumps(
                                {
                                    "image_id": 42,
                                    "question": "What is the dominant color?",
                                    "images": {"this[:]": 42},
                                },
                            ),
                        },
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "final", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = _SpyClient()
    images = {
        "[0:5]": DummyImageHandle(
            image_id=42,
            caption="blue square",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={},
        images=images,
    )

    await h.result()

    # Expect a tool-result message for ask_image containing the BLUE answer
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert tool_msgs, "Expected a tool-result message for ask_image"
    assert any('"BLUE"' in (m.get("content") or "") for m in tool_msgs)

    last_tools = tools_snapshots[-1]
    live_tool = next(
        t for t in last_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "source=ask" in desc
    assert "id=42" in desc


@pytest.mark.asyncio
@_handle_project
async def test_dynamic_sources_multi_append_overview(monkeypatch) -> None:
    """
    Single session appends images from multiple dynamic sources and verifies
    the overview reflects each source: interjection, ask, clar_request,
    clar_answer, notification, stop.
    """

    tools_snapshots: list[list[dict]] = []
    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)

        if step["n"] == 0:
            step["n"] += 1
            # Start long-running tool that will emit a notification and a
            # clarification request with images.
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_RUN",
                        "type": "function",
                        "function": {"name": "do_run", "arguments": "{}"},
                    },
                ],
            }
        elif step["n"] == 1:
            step["n"] += 1
            # Ask about the image (appends source=ask)
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_IMG_ASK",
                        "type": "function",
                        "function": {
                            "name": "ask_image",
                            "arguments": json.dumps(
                                {
                                    "image_id": 42,
                                    "question": "What color?",
                                    "images": {"this[:]": 42},
                                },
                            ),
                        },
                    },
                ],
            }
        elif step["n"] == 2:
            step["n"] += 1
            # Interject into the running tool (appends source=interjection)
            interject_name = _find_tool_name(tools, "interject_")
            assert interject_name, "interject helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_INTERJECT",
                        "type": "function",
                        "function": {
                            "name": interject_name,
                            "arguments": json.dumps(
                                {
                                    "content": "keep going",
                                    "images": {"this[:]": 42},
                                },
                            ),
                        },
                    },
                ],
            }
        elif step["n"] == 3:
            step["n"] += 1
            # Answer the clarification (appends source=clar_answer)
            clarify_name = _find_tool_name(tools, "clarify_")
            assert clarify_name, "clarify helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_CLARIFY",
                        "type": "function",
                        "function": {
                            "name": clarify_name,
                            "arguments": json.dumps(
                                {"answer": "blue", "images": {"this[:]": 42}},
                            ),
                        },
                    },
                ],
            }
        elif step["n"] == 4:
            step["n"] += 1
            # Stop the long-running tool (appends source=stop)
            stop_name = _find_tool_name(tools, "stop_")
            assert stop_name, "stop helper not exposed"
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_STOP",
                        "type": "function",
                        "function": {
                            "name": stop_name,
                            "arguments": json.dumps({"images": {"this[:]": 42}}),
                        },
                    },
                ],
            }
        else:
            msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    # Base tool that emits notification and clarification with images, then
    # waits for interjection and clarify-answer.
    async def do_run(
        *,
        interject_queue: asyncio.Queue[str],
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
        notification_up_q: asyncio.Queue[dict],
    ) -> dict:
        # Emit a notification first so the loop attaches a notification entry
        await notification_up_q.put({"message": "progress", "images": {"this[:]": 42}})
        # Request clarification with images
        await clarification_up_q.put(
            {"question": "What is the dominant color?", "images": {"this[:]": 42}},
        )
        # Wait for interjection and clarify answer
        _ = await interject_queue.get()
        ans = await clarification_down_q.get()
        # Keep the tool running until an explicit stop so the stop helper can act
        await asyncio.Event().wait()
        return {"answer": ans}

    client = _SpyClient()
    images = {
        "[0:5]": DummyImageHandle(
            image_id=42,
            caption="blue square",
            raw_bytes=_solid_png_bytes(0, 0, 255),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={"do_run": do_run},
        images=images,
    )

    await h.result()

    # Verify the overview includes source lines for all dynamic events
    assert tools_snapshots, "No LLM call captured; expected at least one exposure set."
    names = [t.get("function", {}).get("name") for t in tools_snapshots[-1]]
    assert "live_images_overview" in names
    live_tool = next(
        t
        for t in tools_snapshots[-1]
        if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    # Initial seed is always present
    assert "source=user_message" in desc
    # Dynamic appends
    assert "source=notification" in desc
    assert "source=clar_request" in desc
    assert "source=interjection" in desc
    assert "source=clar_answer" in desc
    assert "source=ask" in desc
    assert "source=stop" in desc
