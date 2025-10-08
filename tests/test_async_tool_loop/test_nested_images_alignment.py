from __future__ import annotations

import base64
import json
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS


# Removed stub client; tests use real AsyncUnify with spies only.


def _solid_png_bytes() -> bytes:
    from unity.image_manager.utils import make_solid_png_base64

    b64 = make_solid_png_base64(2, 2, (0, 0, 255))
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

    async def ask(self, question: str):  # pragma: no cover - not used here
        return "OK"


@pytest.mark.asyncio
@_handle_project
async def test_align_images_for_helper_builds_arg_scoped_mapping() -> None:
    """
    Calling align_images_for returns a dict suitable for a subsequent inner tool
    call: keys are arg-scoped spans like "question[15:28]" and values are ids.
    """

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the helper `align_images_for` "
        'with arguments: {\n  "args": { "question": "Please compare the Cairo skyline images for clarity" },\n  "hints": [{ "arg": "question", "substring": "Cairo skyline", "image_id": 42 }]\n}. '
        "After the helper returns, provide a short final reply.",
    )
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

    # Find the tool result for align_images_for
    result_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "align_images_for"
    ]
    assert result_msgs, "align_images_for tool result not found"
    payload = result_msgs[-1].get("content") or "{}"
    data = json.loads(payload)
    images_map = data.get("images") or {}
    # Expect a single arg-scoped key for the substring provided
    assert any(k.startswith("question[") and k.endswith("]") for k in images_map.keys())
    assert list(images_map.values()) == [42]


@pytest.mark.asyncio
@_handle_project
async def test_inner_tool_receives_and_resolves_arg_scoped_images() -> None:
    """
    A base tool that accepts `images` should receive a dict keyed by arg-scoped
    spans where values are resolved to live image handles.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        ids = []
        keys = []
        for k, v in (images or {}).items():
            keys.append(k)
            try:
                ids.append(int(getattr(v, "image_id", -1)))
            except Exception:
                ids.append(-1)
        return {"received": {"keys": keys, "ids": ids, "question": question}}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `analyze` with arguments: "
        '{\n  "question": "Hello world",\n  "images": { "question[0:5]": 42 }\n}. Then provide a short final reply.',
    )
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
        tools={"analyze": analyze},
        images=images,
    )

    await h.result()

    # The tool result should confirm keys and resolved ids
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"
    content = tool_msgs[-1].get("content") or "{}"
    obj = json.loads(content)
    rec = obj.get("received") or {}
    assert rec.get("keys") == ["question[0:5]"]
    assert rec.get("ids") == [42]
    assert rec.get("question") == "Hello world"


@pytest.mark.asyncio
@_handle_project
async def test_invalid_arg_or_span_entries_are_dropped() -> None:
    """
    Entries with invalid arg names or out-of-range spans are ignored by the scheduler.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        return {"received_keys": list((images or {}).keys())}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `analyze` with arguments: "
        '{\n  "question": "Hello",\n  "images": { "question[0:5]": 42, "text[0:4]": 42, "question[10:20]": 42 }\n}. '
        "Then provide a short final reply.",
    )
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
        tools={"analyze": analyze},
        images=images,
    )

    await h.result()

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"
    obj = json.loads(tool_msgs[-1].get("content") or "{}")
    keys = obj.get("received_keys") or []
    assert keys == ["question[0:5]"]


@pytest.mark.asyncio
@_handle_project
async def test_no_implicit_images_pass_when_omitted() -> None:
    """
    If `images` is omitted in the inner tool call, no images are passed implicitly
    even when the outer loop had live images.
    """

    def analyze(*, question: str) -> dict:
        return {"ok": True}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        'You are running inside an automated test. In your FIRST assistant turn, call the tool `analyze` with arguments: { "question": "Hello" }. '
        "Do NOT include an `images` field. Then provide a short final reply.",
    )
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
        tools={"analyze": analyze},
        images=images,
    )

    await h.result()

    # The tool result should exist; absence of errors confirms no implicit images
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"


@pytest.mark.asyncio
@_handle_project
async def test_images_value_may_be_handle_objects() -> None:
    """
    The `images` mapping can include live handle objects as values instead of ids.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        ids = [int(getattr(v, "image_id", -1)) for v in (images or {}).values()]
        return {"ids": ids}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the tool `analyze` with arguments: "
        '{\n  "question": "Hello world",\n  "images": { "question[0:5]": { "__handle__": true } }\n}. Then provide a short final reply.',
    )

    handle_obj = DummyImageHandle(
        image_id=99,
        caption="handle value",
        raw_bytes=_solid_png_bytes(),
    )

    images = {"[0:5]": handle_obj}

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={"analyze": analyze},
        images=images,
    )

    await h.result()

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"
    obj = json.loads(tool_msgs[-1].get("content") or "{}")
    assert obj.get("ids") == [99]
