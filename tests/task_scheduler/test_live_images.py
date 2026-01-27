from __future__ import annotations

import base64
import os
import pytest

from tests.assertion_helpers import find_tool_calls_and_results
from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import ImageRefs, RawImageRef, AnnotatedImageRef


def _load_png_b64(filename: str) -> str:
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, filename)
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "first_head",
    [
        "Invitation Emails",  # scenario A – Invitation first
        "Organize Weekly Rota",  # scenario B – Organize first
    ],
)
async def test_ask_live_images_queue_order(first_head: str, static_now) -> None:
    ts = TaskScheduler()

    # Seed three tasks and materialize a single queue; switch head based on param
    created = ts._create_tasks(
        tasks=[
            {
                "name": "Organize Weekly Rota",
                "description": (
                    "Go through all hired admin assistants, look through the spreadsheet containing their weekly availability, "
                    "and work out the best weekly rota"
                ),
            },
            {
                "name": "Invitation Emails",
                "description": "Send out all of the invitation Emails",
            },
            {
                "name": "Image Edits",
                "description": "Crop all of the images such that they show headshots, cutting from the shoulders down",
            },
        ],
        queue_ordering=[
            {
                # Task ids will be 0..2 in the same order as above
                "order": (
                    [1, 0, 2] if first_head == "Invitation Emails" else [0, 1, 2]
                ),
                # Head has a start_at timestamp; followers are chained
                "queue_head": {"start_at": "2036-06-01T09:00:00+00:00"},
            },
        ],
    )

    assert created["details"]["task_ids"] == [0, 1, 2]

    # Persist two live images and provide typed refs with annotations mapping to the prompt's "this one" / "this other one"
    manager = ImageManager()
    b64_first = _load_png_b64("invitation_emails.png")
    b64_second = _load_png_b64("organize_weekly_rotar.png")
    generic_caption = "screenshots captured from the user sharing their screen with us during our live ongoing meet; a more detailed caption is pending..."
    [img_a] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_first, "timestamp": static_now},
        ],
    )
    [img_b] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_second, "timestamp": static_now},
        ],
    )

    # Align refs – the first annotation corresponds to the user's first "this one" mention
    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_a)),
                annotation=(
                    "the first tab opened, exactly when the user said the first 'this one'"
                ),
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_b)),
                annotation=(
                    "the second tab opened, afterwards the first, exactly when the user said 'this other one'"
                ),
            ),
        ],
    )

    question = (
        "I'm looking through some of the tasks scheduled for next week, and I've started to explore what is entailed for each. "
        "I've made a start on each of the tasks, performing the first few steps, and I've got the associated tabs opened right now. Can you remind me, which task comes first in the schedule for next week, "
        "this one or this other one. I can't remember their names, but you should be able to infer which ones I mean based on the steps I've taken in each of the tabs."
    )

    handle = await ts.ask(question, images=images, _return_reasoning_steps=True)
    answer, messages = await handle.result()

    # 1) Verify the model used ask_image or attach_image_raw to process the images
    ask_image_calls, _ = find_tool_calls_and_results(messages, "ask_image")
    attach_image_calls, _ = find_tool_calls_and_results(messages, "attach_image_raw")
    assert (
        ask_image_calls or attach_image_calls
    ), "Expected the model to use ask_image or attach_image_raw with live images"

    # 2) Verify a tasks lookup occurred (search_tasks or filter_tasks)
    search_calls, _ = find_tool_calls_and_results(messages, "search_tasks")
    filter_calls, _ = find_tool_calls_and_results(messages, "filter_tasks")
    assert (
        search_calls or filter_calls
    ), "Expected a tasks lookup (search_tasks or filter_tasks)"

    # The answer (or the tool outputs) should indicate which of the two comes first
    expected_first = first_head.lower()
    low_answer = (answer or "").lower()
    found = expected_first in low_answer
    if not found:
        for m in messages:
            if m.get("role") == "tool" and (
                m.get("name")
                in (
                    "search_tasks",
                    "filter_tasks",
                    "get_queue",
                    "get_queue_for_task",
                )
            ):
                content = str(m.get("content") or "").lower()
                if expected_first in content:
                    found = True
                    break
    assert found, (
        "Expected the earliest task to be correctly identified from images and lookups: "
        f"{first_head} should come first."
    )


@pytest.mark.asyncio
@_handle_project
async def test_update_live_images_reorder_three_tasks(static_now) -> None:
    ts = TaskScheduler()

    # Seed three tasks and materialize a single queue with a fixed initial order
    created = ts._create_tasks(
        tasks=[
            {
                "name": "Organize Weekly Rota",
                "description": (
                    "Go through all hired admin assistants, look through the spreadsheet containing their weekly availability, "
                    "and work out the best weekly rota"
                ),
            },
            {
                "name": "Invitation Emails",
                "description": "Send out all of the invitation Emails",
            },
            {
                "name": "Image Edits",
                "description": "Crop all of the images such that they show headshots, cutting from the shoulders down",
            },
        ],
        queue_ordering=[
            {
                # Task ids will be 0..2 in the same order as above
                "order": [0, 1, 2],
                # Head has a start_at timestamp; followers are chained
                "queue_head": {"start_at": "2036-06-01T09:00:00+00:00"},
            },
        ],
    )

    assert created["details"]["task_ids"] == [0, 1, 2]

    # Persist three live images with typed refs and minimal annotations that map
    # to "this", "then this", and "finally this" in the user's message
    manager = ImageManager()
    b64_invite = _load_png_b64("invitation_emails.png")
    b64_rota = _load_png_b64("organize_weekly_rotar.png")
    b64_photo = _load_png_b64("photo_editing.png")
    generic_caption = "screenshots captured from the user sharing their screen with us during our live ongoing meet; a more detailed caption is pending..."
    [img_invite] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_invite, "timestamp": static_now},
        ],
    )
    [img_rota] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_rota, "timestamp": static_now},
        ],
    )
    [img_photo] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_photo, "timestamp": static_now},
        ],
    )

    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_invite)),
                annotation="this",
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_rota)),
                annotation="then this",
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_photo)),
                annotation="finally this",
            ),
        ],
    )

    # Ask the scheduler to re-order the three tasks to match the visual order
    command = (
        "I'm looking through some of the tasks scheduled for next week, and I've started to explore what is entailed for each. "
        "I've made a start on each of the tasks, performing the first few steps, and I've got the associated tabs opened right now. "
        "Please reorder the next week's runnable schedule such that this comes first, then this, and finally this. "
        "I don't remember the task names, but the correct order should be the same as the visual order of those three tabs, "
        "where each tab showed my partial progress in *performing* each of the tasks."
    )

    handle = await ts.update(command, images=images, _return_reasoning_steps=True)
    _, messages = await handle.result()

    # 1) Verify the model used ask_image or attach_image_raw to process the images
    ask_image_calls, _ = find_tool_calls_and_results(messages, "ask_image")
    attach_image_calls, _ = find_tool_calls_and_results(messages, "attach_image_raw")
    assert (
        ask_image_calls or attach_image_calls
    ), "Expected the model to use ask_image or attach_image_raw with live images"

    # 2) Verify a tasks lookup occurred (search_tasks or filter_tasks)
    search_calls, _ = find_tool_calls_and_results(messages, "search_tasks")
    filter_calls, _ = find_tool_calls_and_results(messages, "filter_tasks")
    assert (
        search_calls or filter_calls
    ), "Expected a tasks lookup (search_tasks or filter_tasks)"

    # After update, verify the queue order is Invitation → Organize → Image Edits
    row0 = ts._filter_tasks(filter="task_id == 0")[0]
    qid = row0.queue_id
    chain = (
        ts._get_queue(queue_id=qid)
        if qid is not None
        else ts._get_queue_for_task(task_id=0)
    )
    queue = [t.task_id for t in chain]
    assert queue == [1, 0, 2]
