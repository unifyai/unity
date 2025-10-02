from __future__ import annotations

import base64
import pytest

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project


class _SpyClient:
    """Minimal AsyncUnify-compatible stub used by these tests.

    Exposes only the attributes the loop actually uses when we monkeypatch
    generate_with_preprocess: a `messages` list, `append_messages`, and a
    `system_message` property (plus an optional setter for completeness).
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


def _solid_png_bytes() -> bytes:
    """Make a tiny solid PNG as raw bytes for attachment tests."""
    from unity.image_manager.utils import make_solid_png_base64

    b64 = make_solid_png_base64(2, 2, (255, 0, 0))
    return base64.b64decode(b64)


class _DummyImage:
    def __init__(self, *, data: str):
        self.data = data


class DummyImageHandle:
    """Lightweight test double that mirrors the ImageHandle surface we use."""

    def __init__(self, *, image_id: int, caption: str | None, raw_bytes: bytes):
        self._image = _DummyImage(data="")  # non-URL → forces raw() path
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
        return "BLUE"


@pytest.mark.asyncio
@_handle_project
async def test_live_images_helpers_exposed_with_alignment_description(
    monkeypatch,
) -> None:
    """
    Verify that the loop exposes `live_images_overview`, `ask_image`, and
    `attach_image_raw` on the first LLM turn and that the overview docstring
    includes span-aligned substring and caption.
    """

    tools_snapshots: list[list[dict]] = []

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):  # noqa: D401
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)
        # Return a single assistant reply with no tool calls (finish early)
        msg = {"role": "assistant", "content": "done", "tool_calls": []}
        client.messages.append(msg)
        return msg

    # Spy at the callsite used by the loop
    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = _SpyClient()

    # Message whose first 5 chars are 'Hello' – used to compute substring
    message_text = "Hello world – please reason over the image if needed."
    images = {
        "[0:5]": DummyImageHandle(
            image_id=42,
            caption="cat on mat",
            raw_bytes=_solid_png_bytes(),
        ),
    }

    handle = start_async_tool_loop(
        client=client,
        message=message_text,
        tools={},
        images=images,  # ← new feature under test
    )

    await handle.result()

    # We must have recorded at least one tool exposure set
    assert tools_snapshots, "No LLM call captured; expected at least one exposure set."
    names = [t.get("function", {}).get("name") for t in tools_snapshots[0]]
    assert {"live_images_overview", "ask_image", "attach_image_raw"}.issubset(
        set(names),
    )

    # The overview description should include the span, substring, and caption
    live_tool = next(
        t for t in tools_snapshots[0] if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]
    assert "span=[0:5]" in desc
    assert "substring='Hello'" in desc
    assert "caption='cat on mat'" in desc


@pytest.mark.asyncio
@_handle_project
async def test_ask_image_dynamic_helper_executes_and_returns(monkeypatch) -> None:
    """
    Drive the loop to call `ask_image` and verify a tool result is inserted with
    the (dummy) image answer returned by the handle.
    """

    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        if step["n"] == 0:
            step["n"] += 1
            # Request the dynamic helper on the first assistant turn
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_IMG_1",
                        "type": "function",
                        "function": {
                            "name": "ask_image",
                            "arguments": '{"image_id": 42, "question": "What is the dominant color?"}',
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
            raw_bytes=_solid_png_bytes(),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={},
        images=images,
    )

    await h.result()

    # Find the tool message inserted for the helper; content should be a JSON string "BLUE"
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert tool_msgs, "Expected a tool-result message for ask_image"
    assert any('"BLUE"' in (m.get("content") or "") for m in tool_msgs)


@pytest.mark.asyncio
@_handle_project
async def test_attach_image_raw_appends_image_block(monkeypatch) -> None:
    """
    Drive the loop to call `attach_image_raw` and verify a user message with an
    image_url content block (data URL) was appended to the transcript.
    """

    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        if step["n"] == 0:
            step["n"] += 1
            # Request the attach helper on the first assistant turn
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_IMG_2",
                        "type": "function",
                        "function": {
                            "name": "attach_image_raw",
                            "arguments": '{"image_id": 99, "note": "please inspect"}',
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

    client = _SpyClient()
    images = {
        "[0:4]": DummyImageHandle(
            image_id=99,
            caption="red tile",
            raw_bytes=_solid_png_bytes(),
        ),
    }

    handle = start_async_tool_loop(
        client=client,
        message="This is a test",
        tools={},
        images=images,
    )

    await handle.result()

    # Look for a user message that contains an image_url with a data URL
    has_data_url = False
    for m in client.messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict)
            and b.get("type") == "image_url"
            and isinstance(b.get("image_url", {}).get("url"), str)
            and b["image_url"]["url"].startswith("data:image/")
            for b in content
        ):
            has_data_url = True
            break

    assert has_data_url, "Expected an attached data:image/* URL in a user message"
