from __future__ import annotations

import asyncio
import pytest
import unify
from unity.image_manager.image_manager import ImageManager, ImageHandle
from unity.image_manager.types.image import Image
from unity.image_manager.utils import make_solid_png_base64

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_message_prefix,
)


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
    ih = ImageHandle(
        manager=manager,
        image=Image(image_id=42, caption="blue square", data=b64_blue),
    )
    images = {"[0:5]": ih}

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
            else ("auto", available)
        ),
    )

    await _wait_for_tool_request(client, "do_work")
    await h.interject("please proceed", images={"this[:]": ih})
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
    ih = ImageHandle(
        manager=manager,
        image=Image(image_id=42, caption="blue tile", data=b64_blue),
    )
    images = {"[0:3]": ih}

    h = start_async_tool_loop(
        client=client,
        message="Hey",
        tools={"wait_forever": wait_forever},
        images=images,
        max_steps=20,
        timeout=240,
    )

    await _wait_for_tool_request(client, "wait_forever")
    await h.interject("stop", images={"this[:]": ih})
    await _wait_for_assistant_call_prefix(client, "stop_")
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
            {"question": "What is the dominant color?", "images": {"this[:]": 42}},
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
    images = {
        "[0:5]": ImageHandle(
            manager=manager,
            image=Image(image_id=42, caption="blue square", data=b64_blue),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello",
        tools={"need_clar": need_clar},
        images=images,
        max_steps=20,
        timeout=240,
    )

    await _wait_for_assistant_call_prefix(client, "clarify_")
    await _wait_for_tool_message_prefix(client, "clarify_")
    final = await h.result()
    assert final.strip().lower().endswith("done")


@pytest.mark.asyncio
@_handle_project
async def test_notification_payload_appends_images() -> None:
    async def notify(*, notification_up_q: asyncio.Queue[dict]) -> dict:
        await notification_up_q.put({"message": "progress", "images": {"this[:]": 42}})
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
    images = {
        "[0:2]": ImageHandle(
            manager=manager,
            image=Image(image_id=42, caption="blue tile", data=b64_blue),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Go",
        tools={"notify": notify},
        images=images,
        max_steps=10,
        timeout=240,
    )

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
    images = {
        "[0:5]": ImageHandle(
            manager=manager,
            image=Image(image_id=42, caption="blue square", data=b64_blue),
        ),
    }

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
async def test_dynamic_sources_multi_append_overview() -> None:
    """
    Single session appends images from multiple dynamic sources and verifies
    the overview reflects each source: interjection, ask, clar_request,
    clar_answer, notification, stop.
    """

    # Base tool that emits notification and clarification with images, then waits for interjection and clarify-answer.
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

    # Real LLM client and image setup
    client = unify.AsyncUnify(
        "gpt-5@openai",
        reasoning_effort="high",
        service_tier="priority",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )
    client.set_system_message(
        "You are running inside an automated test. Perform the steps in order:\n"
        "1) Call `do_run`.\n"
        "2) Call `ask_image` once for the aligned image.\n"
        "3) When the user says 'keep going', call the `_interject_…` helper with that content.\n"
        "4) Answer the clarification using the `_clarify_…` helper with 'blue'.\n"
        "5) Call the `_stop_…` helper to stop the running call.\n"
        "6) Finally answer with 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    images = {
        "[0:5]": ImageHandle(
            manager=manager,
            image=Image(image_id=42, caption="blue square", data=b64_blue),
        ),
    }

    h = start_async_tool_loop(
        client=client,
        message="Hello world",
        tools={"do_run": do_run},
        images=images,
        max_steps=40,
        timeout=300,
    )

    await _wait_for_tool_request(client, "do_run")
    await _wait_for_assistant_call_prefix(client, "ask_image")
    await _wait_for_tool_result(client, tool_name="ask_image", min_results=1)
    await h.interject("keep going")
    await _wait_for_assistant_call_prefix(client, "interject_")
    await _wait_for_tool_message_prefix(client, "interject ")
    await _wait_for_assistant_call_prefix(client, "clarify_")
    await _wait_for_tool_message_prefix(client, "clarify_")
    await _wait_for_assistant_call_prefix(client, "stop_")
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

    # Use real ImageHandles with solid blue PNGs
    manager = ImageManager()
    b64_blue = make_solid_png_base64(2, 2, (0, 0, 255))
    john_img = Image(image_id=301, caption="john's blue paint", data=b64_blue)
    david_img = Image(image_id=302, caption="david's blue paint", data=b64_blue)
    jenny_img = Image(image_id=303, caption="jenny's blue paint", data=b64_blue)
    john_handle = ImageHandle(manager=manager, image=john_img)
    david_handle = ImageHandle(manager=manager, image=david_img)
    jenny_handle = ImageHandle(manager=manager, image=jenny_img)

    images = {
        f"[{pos_this}:{pos_this + len(seg_this)}]": john_handle,
        f"[{pos_that}:{pos_that + len(seg_that)}]": david_handle,
    }

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
        images={
            "this[:]": jenny_handle,
        },  # source-scoped mapping; registers the new live image
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
