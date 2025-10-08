from __future__ import annotations

import base64
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS


# Removed stub client; tests use real AsyncUnify with spies only.


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

    from unity.common._async_tool import loop as _loop

    orig_gwp = getattr(_loop, "generate_with_preprocess")

    async def _spy_gwp(client, preprocess_msgs, **gen_kwargs):  # noqa: D401
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)
        return await orig_gwp(client, preprocess_msgs, **gen_kwargs)

    # Spy at the callsite used by the loop
    monkeypatch.setattr(_loop, "generate_with_preprocess", _spy_gwp, raising=True)

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Reply exactly with the word 'done'.")

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

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call the dynamic helper `ask_image` once for image_id=42 with the question 'What is the dominant color?'. "
        "Then provide a short final answer.",
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

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call the dynamic helper `attach_image_raw` for image_id=99 with note 'please inspect'. "
        "Then reply with exactly 'red'.",
    )
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

    final_reply = await handle.result()

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

    # Verify that the assistant could "see" and reason over the attached image
    # by answering with the correct colour.
    assert (
        final_reply.strip().lower().startswith("red")
    ), f"Assistant did not identify the image colour – got: {final_reply!r}"


@pytest.mark.asyncio
@_handle_project
async def test_semantic_alignment_and_ask_image(monkeypatch) -> None:
    """
    Motivating example: Given a message with two "this colour" spans – one for Susan and one
    for Emily – seed two live images aligned to those spans. Verify that:
      - the first exposure includes both spans with their substrings
      - the model calls ask_image on the Emily-aligned image id
      - the loop inserts the ask_image tool result and returns the final answer
    """

    tools_snapshots: list[list[dict]] = []
    from unity.common._async_tool import loop as _loop

    orig_gwp2 = getattr(_loop, "generate_with_preprocess")

    async def _spy_gwp2(client, preprocess_msgs, **gen_kwargs):
        tools = gen_kwargs.get("tools") or []
        tools_snapshots.append(tools)
        return await orig_gwp2(client, preprocess_msgs, **gen_kwargs)

    monkeypatch.setattr(_loop, "generate_with_preprocess", _spy_gwp2, raising=True)

    client = unify.AsyncUnify(
        "gpt-5@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )

    # Message with two "this colour" segments
    message_text = (
        "Susan likes this colour but Emily likes this colour, "
        "which colour does Emily like?"
    )
    seg = "this colour"
    pos1 = message_text.find(seg)
    assert pos1 >= 0, "first 'this colour' not found"
    pos2 = message_text.find(seg, pos1 + 1)
    assert pos2 >= 0, "second 'this colour' not found"

    # Two dummy image handles – first for Susan, second for Emily
    susan_id = 201
    emily_id = 202
    images = {
        f"[{pos1}:{pos1 + len(seg)}]": DummyImageHandle(
            image_id=susan_id,
            caption="red tile",
            raw_bytes=_solid_png_bytes(),
        ),
        f"[{pos2}:{pos2 + len(seg)}]": DummyImageHandle(
            image_id=emily_id,
            caption="blue tile",
            raw_bytes=_solid_png_bytes(),
        ),
    }
    # Instruct model to call ask_image for Emily's span and then answer 'blue'.
    client.set_system_message(
        "First, call the dynamic helper `ask_image` on the image aligned to Emily's 'this colour' span. "
        "Then reply with 'blue'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message=message_text,
        tools={},
        images=images,
    )

    final = await handle.result()

    # Verify the first exposure contains the overview with both spans and substrings
    assert tools_snapshots, "No LLM call captured; expected at least one exposure set."
    first_tools = tools_snapshots[0]
    live_tool = next(
        t for t in first_tools if t["function"]["name"] == "live_images_overview"
    )
    desc = live_tool["function"]["description"]

    # Both occurrences should appear with their exact spans and substrings
    assert f"span=[{pos1}:{pos1 + len(seg)}]" in desc
    assert f"span=[{pos2}:{pos2 + len(seg)}]" in desc
    assert "substring='this colour'" in desc

    # Confirm an ask_image tool-result message exists and contains the BLUE payload
    ask_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert ask_msgs, "Expected a tool-result message for ask_image"
    assert any('"BLUE"' in (m.get("content") or "") for m in ask_msgs)

    # Final answer should reflect Emily's colour
    assert final.strip().lower().startswith("blue")
