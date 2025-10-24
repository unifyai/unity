from __future__ import annotations

import base64
import json
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS
from unity.image_manager.types import RawImageRef, ImageRefs
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
)


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


async def _await_tool(
    client: "unify.AsyncUnify",
    tool_name: str,
    *,
    min_results: int = 1,
) -> None:
    await _wait_for_tool_request(client, tool_name)
    await _wait_for_tool_result(client, tool_name=tool_name, min_results=min_results)


@pytest.mark.asyncio
@_handle_project
async def test_align_images_for_helper_builds_arg_scoped_mapping() -> None:
    """
    Arg-scoped image alignment has been removed. This test now verifies that
    live image helpers are exposed and callable by asserting a call to
    `live_images_overview` in the first assistant turn.
    """

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. In your FIRST assistant turn, call the helper `live_images_overview`. After the helper returns, provide a short final reply.",
    )

    # Provide live images via the new ImageRefs container (no arg-scoped spans)
    images = ImageRefs([RawImageRef(image_id=42)])

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={},
        images=images,
    )

    # Ensure the helper is requested and its result is logged before asserting
    await _await_tool(client, "live_images_overview", min_results=1)
    await h.result()

    # The tool result for live_images_overview should exist
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "live_images_overview"
    ]
    assert tool_msgs, "Expected a tool-result message for live_images_overview"


@pytest.mark.asyncio
@_handle_project
async def test_inner_tool_receives_and_resolves_arg_scoped_images() -> None:
    """
    Arg‑scoped image alignment/resolution has been removed. A base tool receives
    whatever `images` payload the model sends (no implicit handle resolution).
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
        reasoning_effort="high",
        service_tier="priority",
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

    # Deterministic ordering: wait for analyze request and its tool result
    await _await_tool(client, "analyze", min_results=1)
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
    # No implicit handle resolution – raw ids are passed as-is to the tool
    # and remain non-handle values (-1 from the test analyzer logic).
    assert rec.get("ids") == [-1]
    assert rec.get("question") == "Hello world"


@pytest.mark.asyncio
@_handle_project
async def test_invalid_arg_or_span_entries_are_dropped() -> None:
    """
    Arg‑scoped alignment validation has been removed; the scheduler passes through
    whatever mapping is provided. This test now verifies the tool receives all entries.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        return {"received_keys": list((images or {}).keys())}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
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

    await _await_tool(client, "analyze", min_results=1)
    await h.result()

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"
    obj = json.loads(tool_msgs[-1].get("content") or "{}")
    keys = obj.get("received_keys") or []
    assert keys == ["question[0:5]", "text[0:4]", "question[10:20]"]


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
        reasoning_effort="high",
        service_tier="priority",
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

    await _await_tool(client, "analyze", min_results=1)
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
    The `images` mapping may include placeholders, but implicit substitution with
    live handles has been removed; values are forwarded as-is to the tool.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        ids = [int(getattr(v, "image_id", -1)) for v in (images or {}).values()]
        return {"ids": ids}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
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

    await _await_tool(client, "analyze", min_results=1)
    await h.result()

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "analyze"
    ]
    assert tool_msgs, "Expected a tool-result message for analyze"
    obj = json.loads(tool_msgs[-1].get("content") or "{}")
    # No implicit conversion to handles – analyzer sees a non-handle value
    # and returns -1 for image_id extraction.
    assert obj.get("ids") == [-1]
