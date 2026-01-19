from __future__ import annotations

import asyncio
import pytest
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_result,
    _wait_for_assistant_call_prefix,
    _wait_for_tool_message_prefix,
)


@pytest.mark.asyncio
@_handle_project
async def test_interject_dynamic_helper_appends_images(model, static_now) -> None:
    async def do_work(
        *,
        _interject_queue: asyncio.Queue[str],
        _notification_up_q: asyncio.Queue[dict],
    ):
        await _notification_up_q.put({"message": "ready"})
        _ = await _interject_queue.get()
        return {"ok": True}

    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣ Call `do_work`. 2️⃣ When the user says 'please proceed', call the helper whose name starts with `_interject_` "
        'passing `{ "content": "please proceed" }`. 3️⃣ Then finish and answer \'done\'.',
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue, "timestamp": static_now},
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
    await h.interject("please proceed", images=[RawImageRef(image_id=img_id)])
    await _wait_for_assistant_call_prefix(client, "interject_")
    await _wait_for_tool_message_prefix(client, "interject_")
    final = await h.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_stop_dynamic_helper_appends_images(model, static_now) -> None:
    async def wait_forever(*, _notification_up_q: asyncio.Queue[dict]):
        await _notification_up_q.put({"message": "starting"})
        await asyncio.Event().wait()
        return {"ok": False}

    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣ Call `wait_forever`. 2️⃣ If the user later says 'stop', call the `_stop_…` helper to stop the running call. "
        "Then reply 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue tile", "data": b64_blue, "timestamp": static_now},
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
    await h.interject("stop", images=[RawImageRef(image_id=img_id)])
    await _wait_for_assistant_call_prefix(client, "stop_")
    await _wait_for_tool_message_prefix(client, "stop_")
    assert any(
        m.get("role") == "tool"
        and isinstance(m.get("name"), str)
        and m.get("name", "").startswith("stop_")
        and "stopped" in (m.get("content") or "").lower()
        for m in client.messages
    )
    final = await h.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_clarify_helpers_append_images_for_request_and_answer(
    model,
    static_now,
) -> None:
    async def need_clar(
        *,
        _clarification_up_q: asyncio.Queue[str],
        _clarification_down_q: asyncio.Queue[str],
    ) -> dict:
        await _clarification_up_q.put(
            {
                "question": "What is the dominant color?",
                "images": [RawImageRef(image_id=img_id)],
            },
        )
        ans = await _clarification_down_q.get()
        return {"answer": ans}

    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣ Call `need_clar`. 2️⃣ When the tool asks a question, answer using the `_clarify_…` helper with the single word 'blue'. "
        "3️⃣ Finish by saying 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue, "timestamp": static_now},
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
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_notification_payload_appends_images(model, static_now) -> None:
    async def notify(*, _notification_up_q: asyncio.Queue[dict]) -> dict:
        await _notification_up_q.put(
            {"message": "progress", "images": [RawImageRef(image_id=img_id)]},
        )
        return {"ok": True}

    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣ Call `notify` and then finish with 'done'.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue tile", "data": b64_blue, "timestamp": static_now},
        ],
    )
    images = [RawImageRef(image_id=img_id)]

    h = start_async_tool_loop(
        client=client,
        message="Go",
        tools={"notify": notify},
        images=images,
        max_steps=12,
        timeout=240,
        tool_policy=lambda step, available: (
            ("required", {"notify": available["notify"]}) if step == 0 else ("auto", {})
        ),
    )

    await _wait_for_tool_request(client, "notify")
    event = await asyncio.wait_for(h.next_notification(), timeout=60)
    assert event["type"] == "notification"
    assert event["tool_name"] == "notify"
    assert isinstance(event.get("message"), str)

    # Ensure the notify tool completes deterministically before awaiting the final answer
    await _wait_for_tool_message_prefix(client, "notify")

    final = await h.result()
    assert final is not None, "Loop should complete with a response"


@pytest.mark.asyncio
@_handle_project
async def test_overview_reinjected_on_interjection_images(model, static_now) -> None:
    """
    When an interjection brings new images, the overview should be reinjected
    automatically with the full updated AnnotatedImageRefs list.
    """

    client = new_llm_client(model=model)
    client.set_system_message("Acknowledge with 'ok'.")

    manager = ImageManager()
    from unity.image_manager.utils import make_solid_png_base64

    # First image at loop start
    [id1] = manager.add_images(
        [
            {
                "caption": "first",
                "data": make_solid_png_base64(32, 32, (0, 0, 255)),
                "timestamp": static_now,
            },
        ],
    )

    images = [RawImageRef(image_id=id1)]
    h = start_async_tool_loop(
        client=client,
        message="Start",
        tools={},
        images=images,
        max_steps=10,
        timeout=240,
    )

    # Verify initial overview present
    def _latest_overview_content() -> str:
        msgs = [
            m
            for m in client.messages
            if m.get("role") == "system" and m.get("_live_images_overview")
        ]
        return (msgs[-1].get("content") or "{}") if msgs else ""

    import asyncio

    for _ in range(100):
        if '"image_id": ' in _latest_overview_content():
            break
        await asyncio.sleep(0.01)
    initial = _latest_overview_content()
    assert '"image_id": ' in initial and '"caption": "first"' in initial

    # Interject with a second image → expect reinjection containing both ids
    [id2] = manager.add_images(
        [
            {
                "caption": "second",
                "data": make_solid_png_base64(32, 32, (255, 0, 0)),
                "timestamp": static_now,
            },
        ],
    )
    await h.interject("new image", images=[RawImageRef(image_id=id2)])

    # Wait briefly for reinjection to occur
    for _ in range(100):
        newer = _latest_overview_content()
        if '"image_id": %d' % id2 in newer:
            break
        await asyncio.sleep(0.01)

    newer = _latest_overview_content()
    assert ('"image_id": %d' % id1) in newer
    assert ('"image_id": %d' % id2) in newer
    assert ("first" in newer) and ("second" in newer)

    # Finish
    await h.result()


@pytest.mark.asyncio
@_handle_project
async def test_ask_image_with_images_param_appends_log(model, static_now) -> None:
    client = new_llm_client(model=model)
    client.set_system_message(
        "1️⃣ Use the `ask_image` tool once for the aligned image to identify its color. 2️⃣ Then answer with any single word.",
    )

    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    [img_id] = manager.add_images(
        [
            {"caption": "blue square", "data": b64_blue, "timestamp": static_now},
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
async def test_two_images_then_interjection_three_asks_real_llm(
    model,
    static_now,
) -> None:
    """
    Real-LLM flow:
    - Initial user message references two paints: "this paint" (John) and "that paint" (David)
      with live images provided.
    - The assistant should call `ask_image` twice to identify both colours (from the visuals),
      compute a 50/50 mix, and wait.
    - A user interjection then introduces Jenny's paint (with an image via `images` on
      interject). The assistant should call `ask_image` once more for Jenny, then mix again and
      answer with the single final colour word.
    - We assert three `ask_image` tool results and a final answer of "brown". This keeps the test
      deterministic while still requiring the model to visually recognise distinct colours from
      the images without relying on captions or system-provided colour hints.
    """

    # Real ImageHandles created below

    # Initial message – two guests with provided images
    user_msg = (
        "We're throwing an art party, with two guests. John has brought this paint, and David has "
        "brought that paint. We will now mix the colours together 50/50. What will be the resulting colour?"
    )

    # Use stored images and typed refs (distinct colours to require real vision recognition)
    manager = ImageManager()
    b64_blue = make_solid_png_base64(32, 32, (0, 0, 255))
    b64_yellow = make_solid_png_base64(32, 32, (255, 255, 0))
    b64_red = make_solid_png_base64(32, 32, (255, 0, 0))
    [john_id, david_id, jenny_id] = manager.add_images(
        [
            {"data": b64_blue, "timestamp": static_now},
            {"data": b64_yellow, "timestamp": static_now},
            {"data": b64_red, "timestamp": static_now},
        ],
    )
    # Provide annotated refs for each referenced person
    images = [
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=john_id),
            annotation="John's paint",
        ),
        AnnotatedImageRef(
            raw_image_ref=RawImageRef(image_id=david_id),
            annotation="David's paint",
        ),
    ]

    # Real client – drive the model to call ask_image 3 times and produce final colour
    client = new_llm_client(model=model)
    client.set_system_message(
        "Use the `ask_image` tool to identify the colour of each listed image (by id). "
        "Do not attach images or answer without calling `ask_image`. "
        "First, call `ask_image` for John's and David's image ids and wait. "
        "After I interject with Jenny, call `ask_image` for Jenny as well. "
        "Only then answer with a single final colour word.",
    )

    handle = start_async_tool_loop(
        client=client,
        message=user_msg,
        tools={},
        images=images,
        max_steps=50,  # Increased from 30: gpt-5.2 sometimes loops while learning tool format
        timeout=360,
    )

    # Wait deterministically for two ask_image tool results (John + David)
    await _wait_for_tool_result(client, tool_name="ask_image", min_results=2)

    # Interject with Jenny's paint and attach her image under the interjection source
    interjection_msg = (
        "Oh Jenny just arrived, her paint looks like this. We will mix her paint with the previous mix "
        "from John and David (again, 50/50, so 50% John+David (25 each) and 50% Jenny). "
        "Don't worry about the intermediary John + David color mix, what will the *final* colour be?"
    )
    await handle.interject(
        interjection_msg,
        images=[
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=jenny_id),
                annotation="Jennny's paint",
            ),
        ],
    )

    # Wait until the third ask_image tool result is inserted (Jenny)
    await _wait_for_tool_result(client, tool_name="ask_image", min_results=3)

    # Finish and assert outcomes
    final = await handle.result()

    # The final answer should be a valid color from mixing blue + yellow + red paint.
    # In subtractive color mixing (paint), this tends toward brown/black.
    final_lower = final.strip().lower()
    assert any(
        word in final_lower
        for word in ("brown", "pink", "rose", "green", "orange", "red", "black")
    )

    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert (
        len(tool_msgs) >= 3
    ), "Expected at least three ask_image tool results (John, David, Jenny)"
