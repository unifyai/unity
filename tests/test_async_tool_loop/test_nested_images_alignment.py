from __future__ import annotations

import base64
import json
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS


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
async def test_align_images_for_helper_builds_arg_scoped_mapping(monkeypatch) -> None:
    """
    Calling align_images_for returns a dict suitable for a subsequent inner tool
    call: keys are arg-scoped spans like "question[15:28]" and values are ids.
    """

    snapshots: list[list[dict]] = []

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        snapshots.append(tools)

        # First turn: call helper
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ALIGN",
                    "type": "function",
                    "function": {
                        "name": "align_images_for",
                        "arguments": json.dumps(
                            {
                                "args": {
                                    "question": "Please compare the Cairo skyline images for clarity",
                                },
                                "hints": [
                                    {
                                        "arg": "question",
                                        "substring": "Cairo skyline",
                                        "image_id": 42,
                                    },
                                ],
                            },
                        ),
                    },
                },
            ],
        }

        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
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
async def test_inner_tool_receives_and_resolves_arg_scoped_images(monkeypatch) -> None:
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

    step = {"n": 0}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        if step["n"] == 0:
            step["n"] += 1
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_ANALYZE",
                        "type": "function",
                        "function": {
                            "name": "analyze",
                            "arguments": json.dumps(
                                {
                                    "question": "Hello world",
                                    "images": {"question[0:5]": 42},
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

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
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
async def test_invalid_arg_or_span_entries_are_dropped(monkeypatch) -> None:
    """
    Entries with invalid arg names or out-of-range spans are ignored by the scheduler.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        return {"received_keys": list((images or {}).keys())}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ANALYZE",
                    "type": "function",
                    "function": {
                        "name": "analyze",
                        "arguments": json.dumps(
                            {
                                "question": "Hello",
                                # valid, in-range
                                "images": {
                                    "question[0:5]": 42,
                                    # invalid arg name
                                    "text[0:4]": 42,
                                    # out-of-range span
                                    "question[10:20]": 42,
                                },
                            },
                        ),
                    },
                },
            ],
        }
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
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
async def test_no_implicit_images_pass_when_omitted(monkeypatch) -> None:
    """
    If `images` is omitted in the inner tool call, no images are passed implicitly
    even when the outer loop had live images.
    """

    def analyze(*, question: str) -> dict:
        return {"ok": True}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ANALYZE",
                    "type": "function",
                    "function": {
                        "name": "analyze",
                        "arguments": json.dumps({"question": "Hello"}),
                    },
                },
            ],
        }
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
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
async def test_images_value_may_be_handle_objects(monkeypatch) -> None:
    """
    The `images` mapping can include live handle objects as values instead of ids.
    """

    def analyze(*, question: str, images: dict[str, object]) -> dict:
        ids = [int(getattr(v, "image_id", -1)) for v in (images or {}).values()]
        return {"ids": ids}

    async def _fake_gwp(client, preprocess_msgs, **gen_kwargs):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ANALYZE",
                    "type": "function",
                    "function": {
                        "name": "analyze",
                        "arguments": json.dumps(
                            {
                                "question": "Hello world",
                                "images": {"question[0:5]": {"__handle__": True}},
                            },
                        ),
                    },
                },
            ],
        }
        client.messages.append(msg)
        return msg

    from unity.common._async_tool import loop as _loop

    monkeypatch.setattr(_loop, "generate_with_preprocess", _fake_gwp, raising=True)

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
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
