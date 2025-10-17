from __future__ import annotations

import asyncio
import pytest
import unify
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_message_prefix,
)


@pytest.mark.asyncio
@_handle_project
async def test_interject_dynamic_helper_appends_images() -> None:
    async def do_work(
        *,
        interject_queue: asyncio.Queue[str],
        notification_up_q: asyncio.Queue[dict],
    ):
        await notification_up_q.put({"message": "ready"})
        _ = await interject_queue.get()
        return {"ok": True}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Call `do_work`. 2️⃣ When the user says 'please proceed', call the helper whose name starts with `_interject_` "
        'passing `{ "content": "please proceed" }`. 3️⃣ Then finish and answer \'done\'.',
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={"do_work": do_work},
        images=images,
        max_steps=20,
        timeout=240,
        tool_policy=lambda step, available: (
            ("required", {"do_work": available["do_work"]})
            if step == 0
            else ("auto", {})
        ),
    )

    await _wait_for_tool_request(client, "do_work")
    await _wait_for_tool_message_prefix(client, "do_work")
    await h.interject("please proceed", image_refs=[RawImageRef(image_id=img_id)])
    await _wait_for_assistant_call_prefix(client, "interject_")
    await _wait_for_tool_message_prefix(client, "interject ")
    final = await h.result()
    assert final.strip().lower().endswith("done")


@pytest.mark.asyncio
@_handle_project
async def test_stop_dynamic_helper_appends_images() -> None:
    async def wait_forever(*, notification_up_q: asyncio.Queue[dict]):
        await notification_up_q.put({"message": "starting"})
        await asyncio.Event().wait()
        return {"ok": False}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Call `wait_forever`. 2️⃣ If the user later says 'stop', call the `_stop_…` helper to stop the running call. "
        "Then reply 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue tile", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Hey",
        tools={"wait_forever": wait_forever},
        images=images,
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "wait_forever")
    await _wait_for_tool_message_prefix(client, "wait_forever")
    await h.interject("stop", image_refs=[RawImageRef(image_id=img_id)])
    await _wait_for_assistant_call_prefix(client, "stop_")
    await _wait_for_tool_message_prefix(client, "stop ")
    assert any(
        m.get("role") == "tool"
        and isinstance(m.get("name"), str)
        and "stop" in m.get("name")
        and "stopped successfully" in (m.get("content") or "").lower()
        for m in client.messages
    )
    final = await h.result()
    assert final.strip().lower().endswith("done")


@pytest.mark.asyncio
@_handle_project
async def test_clarify_helpers_append_images_for_request_and_answer() -> None:
    async def need_clar(
        *,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ) -> dict:
        await clarification_up_q.put(
            {
                "question": "What is the dominant color?",
                "images": [RawImageRef(image_id=img_id)],
            },
        )
        ans = await clarification_down_q.get()
        return {"answer": ans}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Call `need_clar`. 2️⃣ When the tool asks a question, answer using the `_clarify_…` helper with the single word 'blue'. "
        "3️⃣ Finish by saying 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Hello",
        tools={"need_clar": need_clar},
        images=images,
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "need_clar")
    await _wait_for_assistant_call_prefix(client, "clarify_")
    await _wait_for_tool_message_prefix(client, "clarify_")
    final = await h.result()
    assert final.strip().lower().endswith("done")


@pytest.mark.asyncio
@_handle_project
async def test_notification_payload_appends_images() -> None:
    async def notify(*, notification_up_q: asyncio.Queue[dict]) -> dict:
        await notification_up_q.put(
            {"message": "progress", "images": [RawImageRef(image_id=img_id)]},
        )
        return {"ok": True}

    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Call `notify` and then finish with 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue tile", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Go",
        tools={"notify": notify},
        images=images,
        max_steps=10,
        timeout=240,
    )

    await _wait_for_tool_request(client, "notify")
    event = await asyncio.wait_for(h.next_notification(), timeout=60)
    assert event["type"] == "notification"
    assert event["tool_name"] == "notify"
    assert isinstance(event.get("message"), str)

    final = await h.result()
    assert final.strip().lower().endswith("done")


@pytest.mark.asyncio
@_handle_project
async def test_ask_image_with_images_param_appends_log() -> None:
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "1️⃣ Use the `ask_image` tool once for the aligned image to identify its color. 2️⃣ Then answer with any single word.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={},
        images=images,
        max_steps=10,
        timeout=240,
    )

    await h.result()

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert tool_msgs, "Expected a tool-result message for ask_image"


@pytest.mark.asyncio
@_handle_project
async def test_two_span_images_then_interjection_three_asks_real_llm() -> None:
    """
    Real-LLM flow:
    - Initial user message references two spans: "this paint" (John) and "that paint" (David)
      with live images aligned to those spans.
    - The assistant should call `ask_image` twice to identify both colours, compute a 50/50 mix,
      and wait.
    - A user interjection then introduces Jenny's paint (with an aligned image via `images` on
      interject). The assistant should call `ask_image` once more for Jenny, then mix again and
      answer with the single final colour word.
    - We assert three `ask_image` tool results and a final answer of "blue" (we seed all images
      to answer BLUE deterministically).
    """

    # Real ImageHandles created below

    # Initial message – two guests with span-aligned references
    user_msg = (
        "We're throwing an art party, with two guests. John has brought this paint, and David has "
        "brought that paint. We will now mix the colours together 50/50. What will be the resulting colour?"
    )
    seg_this = "this paint"
    seg_that = "that paint"
    pos_this = user_msg.find(seg_this)
    pos_that = user_msg.find(seg_that)
    assert pos_this >= 0 and pos_that >= 0, "Span substrings not found in user message"

    # Use stored images and typed refs
    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    [john_id, david_id, jenny_id] = manager.add_images(
        [
            {"caption": "john's blue paint", "data": b64_blue},
            {"caption": "david's blue paint", "data": b64_blue},
            {"caption": "jenny's blue paint", "data": b64_blue},
        ],
    )
    images = [RawImageRef(image_id=john_id), RawImageRef(image_id=david_id)]

    # Real client – drive the model to call ask_image 3 times and produce final colour
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. Follow these steps exactly:\n"
        "1️⃣  Use the `ask_image` tool to identify the colour of each image aligned to the user message.\n"
        "    First call `ask_image` for the span 'this paint' (John). Then call `ask_image` for the span 'that paint' (David).\n"
        "2️⃣  Compute the 50/50 paint mixture of those two colours mentally. Do not reply to the user yet.\n"
        "3️⃣  When the user interjects with a new paint ('this' referring to Jenny's paint), call `ask_image` once for that image.\n"
        "4️⃣  Compute a new 50/50 mixture of your earlier result with Jenny's colour.\n"
        "5️⃣  Finally, reply with exactly the single lowercase word representing the final resulting colour, and nothing else.",
    )

    handle = start_async_tool_loop(
        client=client,
        message=user_msg,
        tools={},
        images=images,
        max_steps=30,
        timeout=360,
    )

    # Wait deterministically for two ask_image tool results (John + David)
    await _wait_for_tool_result(client, tool_name="ask_image", min_results=2)

    # Interject with Jenny's paint and attach her image under the interjection source
    interjection_msg = (
        "Oh Jenny just arrived, her paint looks like this. We will mix her paint with the previous mix "
        "from John and David (again, 50/50). What will the final resultant colour be?"
    )
    await handle.interject(
        interjection_msg,
        image_refs=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=jenny_id),
                annotation="interjection",
            ),
        ],
    )

    # Wait until the third ask_image tool result is inserted (Jenny)
    await _wait_for_tool_result(client, tool_name="ask_image", min_results=3)

    # Finish and assert outcomes
    final = await handle.result()

    # Expect exactly a single colour word – seeded BLUE makes the final still blue
    assert final.strip().lower() == "blue"

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert (
        len(tool_msgs) == 3
    ), "Expected exactly three ask_image tool results (John, David, Jenny)"
