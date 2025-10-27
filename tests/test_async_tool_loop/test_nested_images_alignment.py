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
async def test_live_images_overview_is_injected_synthetically() -> None:
    """
    Verify that a synthetic call to `live_images_overview` is injected in the
    first assistant turn and that its tool result exists.
    """

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. Provide a short final reply.",
    )

    # Provide live images via the ImageRefs container
    images = ImageRefs([RawImageRef(image_id=42)])

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={},
        images=images,
    )

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
async def test_overview_after_clarification_images() -> None:
    """
    When a child tool requests clarification and supplies images with the question,
    the overview should be reinjected including those images.
    """

    async def need_clar(*, _clarification_up_q, _clarification_down_q):
        await _clarification_up_q.put(
            {"question": "q?", "images": [RawImageRef(image_id=rid)]},
        )
        _ = await _clarification_down_q.get()
        return {"ok": True}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Call `need_clar`. 2️⃣ When the tool asks a question, answer using the `_clarify_…` helper with the single word 'ok'.",
    )

    from unity.image_manager.utils import make_solid_png_base64
    from unity.image_manager.image_manager import ImageManager

    manager = ImageManager()
    [rid] = manager.add_images(
        [{"caption": "clar", "data": make_solid_png_base64(2, 2, (0, 0, 255))}],
    )

    h = start_async_tool_loop(
        client=client,
        message="go",
        tools={"need_clar": need_clar},
        images=[],
        tool_policy=lambda step, available: (
            ("required", {"need_clar": available["need_clar"]})
            if step == 0
            else ("auto", {})
        ),
    )

    # Wait for need_clar to be requested so the clarification gets raised
    await _wait_for_tool_request(client, "need_clar")

    # Scan for latest overview content containing the new image id
    import asyncio

    for _ in range(100):
        ov_msgs = [
            m
            for m in client.messages
            if m.get("role") == "tool" and m.get("name") == "live_images_overview"
        ]
        if ov_msgs and (f'"image_id": {rid}' in (ov_msgs[-1].get("content") or "")):
            break
        await asyncio.sleep(0.01)

    assert ov_msgs, "Expected overview reinjected after clarification images"
    assert f'"image_id": {rid}' in (ov_msgs[-1].get("content") or "")

    await h.result()


@pytest.mark.asyncio
@_handle_project
async def test_inner_tool_receives_images_mapping() -> None:
    """
    A base tool receives whatever `images` payload the model sends (no implicit handle resolution).
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
        '{\n  "question": "Hello world",\n  "images": { "img_key": 42 }\n}. Then provide a short final reply.',
    )
    images = {
        "img_key": DummyImageHandle(
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
    assert rec.get("keys") == ["img_key"]
    # No implicit handle resolution – raw ids are passed as-is to the tool
    # and remain non-handle values (-1 from the test analyzer logic).
    assert rec.get("ids") == [-1]
    assert rec.get("question") == "Hello world"


@pytest.mark.asyncio
@_handle_project
async def test_various_image_mapping_keys_are_preserved() -> None:
    """
    The scheduler passes through whatever mapping is provided. This test verifies
    the tool receives all entries.
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
        '{\n  "question": "Hello",\n  "images": { "k1": 42, "k2": 42, "k3": 42 }\n}. '
        "Then provide a short final reply.",
    )
    images = {
        "k1": DummyImageHandle(
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
    assert keys == ["k1", "k2", "k3"]


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
        "img_key": DummyImageHandle(
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
        '{\n  "question": "Hello world",\n  "images": { "img_key": { "__handle__": true } }\n}. Then provide a short final reply.',
    )

    handle_obj = DummyImageHandle(
        image_id=99,
        caption="handle value",
        raw_bytes=_solid_png_bytes(),
    )

    images = {"img_key": handle_obj}

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
