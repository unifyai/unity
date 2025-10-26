from __future__ import annotations

import base64
import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from unity.common._async_tool.loop_config import LIVE_IMAGES_REGISTRY
from unity.image_manager.types import RawImageRef, AnnotatedImageRef, ImageRefs
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
async def test_live_images_helpers_exposed_and_overview_injected(
    monkeypatch,
) -> None:
    """
    Verify that the loop exposes image helpers (`ask_image`, `attach_image_raw`)
    and injects a synthetic `live_images_overview` tool call/result that contains
    the image id and caption in its payload.
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
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Reply exactly with the word 'done'.")

    # Seed registry with a handle for id=42 so helpers can resolve it
    LIVE_IMAGES_REGISTRY.set(
        {
            42: DummyImageHandle(
                image_id=42,
                caption="cat on mat",
                raw_bytes=_solid_png_bytes(),
            ),
        },
    )

    # Provide typed ImageRefs with an annotation
    message_text = "Hello world – please reason over the image if needed."
    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=42),
                annotation="greeting",
            ),
        ],
    )

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
    # Only actionable helpers are exposed to the LLM
    assert {"ask_image", "attach_image_raw"}.issubset(set(names))

    # Synthetic overview must be injected as an assistant tool-call + tool result

    # Find the synthetic assistant tool-call
    calls = []
    for m in client.messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == "live_images_overview":
                    calls.append(tc)
    assert calls, "Expected a synthetic assistant tool call for live_images_overview"
    call_id = calls[0].get("id")

    # Corresponding tool result should contain image_id and caption
    tmsgs = [
        m
        for m in client.messages
        if m.get("role") == "tool"
        and m.get("name") == "live_images_overview"
        and (call_id is None or m.get("tool_call_id") == call_id)
    ]
    assert tmsgs, "Expected a tool-result message for live_images_overview"
    content = tmsgs[-1].get("content") or "{}"
    # Payload is JSON; assert id and caption appear
    assert '"image_id": 42' in content
    assert '"caption": "cat on mat"' in content


@pytest.mark.asyncio
@_handle_project
async def test_ask_image_dynamic_helper_executes_and_returns(monkeypatch) -> None:
    """
    Drive the loop to call `ask_image` and verify a tool result is inserted with
    the (dummy) image answer returned by the handle.
    """

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call the dynamic helper `ask_image` once for image_id=42 with the question 'What is the dominant color?'. "
        "Then provide a short final answer.",
    )
    LIVE_IMAGES_REGISTRY.set(
        {
            42: DummyImageHandle(
                image_id=42,
                caption="blue square",
                raw_bytes=_solid_png_bytes(),
            ),
        },
    )
    images = ImageRefs([RawImageRef(image_id=42)])

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
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "Call the dynamic helper `attach_image_raw` for image_id=99 with note 'please inspect'. "
        "Then reply with exactly 'red'.",
    )
    LIVE_IMAGES_REGISTRY.set(
        {
            99: DummyImageHandle(
                image_id=99,
                caption="red tile",
                raw_bytes=_solid_png_bytes(),
            ),
        },
    )
    images = ImageRefs([RawImageRef(image_id=99)])

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
async def test_images_and_ask_image(monkeypatch) -> None:
    """
    Motivating example: Given a message that refers to two colours – one for Susan and one
    for Emily – seed two live images. Verify that:
      - the first exposure includes both image ids and captions
      - the model calls ask_image on Emily's image id
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
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )

    # Two dummy image handles – first for Susan, second for Emily
    susan_id = 201
    emily_id = 202
    LIVE_IMAGES_REGISTRY.set(
        {
            susan_id: DummyImageHandle(
                image_id=susan_id,
                caption="red tile",
                raw_bytes=_solid_png_bytes(),
            ),
            emily_id: DummyImageHandle(
                image_id=emily_id,
                caption="blue tile",
                raw_bytes=_solid_png_bytes(),
            ),
        },
    )

    # Provide typed refs with annotations indicating who is who
    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=susan_id),
                annotation="Susan this colour",
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=emily_id),
                annotation="Emily this colour",
            ),
        ],
    )

    # Instruct model to call ask_image for Emily's id and then answer 'blue'.
    client.set_system_message(
        "First, call the dynamic helper `ask_image` on image_id=202. Then reply with 'blue'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message=(
            "Susan likes this colour but Emily likes this colour, which colour does Emily like?"
        ),
        tools={},
        images=images,
    )

    final = await handle.result()

    # Verify the synthetic overview tool result contains both ids and captions
    ov_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "live_images_overview"
    ]
    assert ov_msgs, "Expected a tool-result message for live_images_overview"
    ov_content = ov_msgs[-1].get("content") or "{}"
    assert '"image_id": 201' in ov_content
    assert '"image_id": 202' in ov_content
    assert ("red tile" in ov_content) and ("blue tile" in ov_content)

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


@pytest.mark.asyncio
@_handle_project
async def test_overview_injected_before_first_llm_step(monkeypatch) -> None:
    """
    Verify the synthetic `live_images_overview` assistant tool-call/result are injected
    before the first LLM thinking step (no model choice is used to call it).
    """

    import asyncio
    from unity.common._async_tool import loop as _loop

    # Spy on LLM generate to capture the log index at the first thinking step and block it
    llm_called = asyncio.Event()
    proceed = asyncio.Event()
    index_before_llm: int | None = None

    orig_gwp = getattr(_loop, "generate_with_preprocess")

    async def _spy_gwp(client, preprocess_msgs, **gen_kwargs):
        nonlocal index_before_llm
        llm_called.set()
        try:
            index_before_llm = len(client.messages)
        except Exception:
            index_before_llm = None
        # Block until the test allows progress to ensure we can assert ordering
        await proceed.wait()
        return await orig_gwp(client, preprocess_msgs, **gen_kwargs)

    monkeypatch.setattr(_loop, "generate_with_preprocess", _spy_gwp, raising=True)

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message("Say 'done'.")

    # Seed a single image in the live registry and align a typed ref
    LIVE_IMAGES_REGISTRY.set(
        {
            7: DummyImageHandle(
                image_id=7,
                caption="seven cats",
                raw_bytes=_solid_png_bytes(),
            ),
        },
    )

    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=7),
                annotation="group photo",
            ),
        ],
    )

    h = start_async_tool_loop(
        client=client,
        message="Hello",
        tools={},
        images=images,
        max_steps=5,
        timeout=120,
    )

    # Poll for the synthetic assistant tool-call/result BEFORE LLM is invoked
    async def _find_overview_msgs(timeout_s: float = 2.0):
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout_s:
            # Find assistant tool-call and result for overview
            asst_idx = None
            call_id = None
            for idx, m in enumerate(client.messages):
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    if fn.get("name") == "live_images_overview":
                        asst_idx = idx
                        call_id = tc.get("id")
                        break
                if asst_idx is not None:
                    break

            if asst_idx is not None:
                # Find the corresponding tool result and its index
                result_msg = None
                result_idx = None
                for idx, m in enumerate(client.messages):
                    if (
                        m.get("role") == "tool"
                        and m.get("name") == "live_images_overview"
                        and (call_id is None or m.get("tool_call_id") == call_id)
                    ):
                        result_msg = m
                        result_idx = idx
                if result_msg is not None:
                    return asst_idx, result_idx, result_msg
            await asyncio.sleep(0.01)
        return None

    found = await _find_overview_msgs()
    assert (
        found is not None
    ), "Expected synthetic overview tool result before first LLM step"
    asst_idx, result_idx, tool_msg = found
    content = tool_msg.get("content") or "{}"
    assert '"image_id": 7' in content
    assert '"caption": "seven cats"' in content

    # Ensure overview assistant tool-call and tool result are recorded before the first LLM step
    assert llm_called.is_set(), "LLM did not start; test spy did not fire"
    assert index_before_llm is not None, "Spy did not capture index before LLM call"
    assert asst_idx < index_before_llm
    assert result_idx < index_before_llm

    # Allow the LLM to proceed and complete
    proceed.set()
    await h.result()
