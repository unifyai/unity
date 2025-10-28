from __future__ import annotations

import base64
import os
import pytest

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
        "Deprioritize Tasks",  # scenario A – Deprioritize first
        "Organize Weekly Rota",  # scenario B – Organize first
    ],
)
async def test_taskscheduler_ask_live_images_queue_order(first_head: str) -> None:
    ts = TaskScheduler()

    # Seed four tasks and materialize a single queue; switch head based on param
    created = ts._create_tasks(
        tasks=[
            {
                "name": "Deprioritize Tasks",
                "description": 'take all tasks marked as "urgent" and reduce their urgency down to "high"',
            },
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
                # Task ids will be 0..3 in the same order as above
                "order": (
                    [0, 1, 2, 3] if first_head == "Deprioritize Tasks" else [1, 0, 2, 3]
                ),
                # Head has a start_at timestamp; followers are chained
                "queue_head": {"start_at": "2036-06-01T09:00:00+00:00"},
            },
        ],
    )

    assert created["details"]["task_ids"] == [0, 1, 2, 3]

    # Persist two live images and provide typed refs with annotations mapping to the prompt's "this one" / "this other one"
    manager = ImageManager()
    b64_first = _load_png_b64("deprioritize_tasks.png")
    b64_second = _load_png_b64("organize_weekly_rotar.png")
    generic_caption = "screenshots captured from the user sharing their screen with us during our live ongoing meet"
    [img_a] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_first},
        ],
    )
    [img_b] = manager.add_images(
        [
            {"caption": generic_caption, "data": b64_second},
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

    # Expect image-aware behaviour via ask_image or attach_image_raw
    image_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn in ("ask_image", "attach_image_raw"):
                image_calls.append(tc)
    assert (
        image_calls
    ), "Expected ask_image or attach_image_raw to be used with live images"

    # Verify a tasks lookup occurred (search_tasks or filter_tasks)
    lookup_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn in ("search_tasks", "filter_tasks"):
                lookup_calls.append(tc)
    assert lookup_calls, "Expected a tasks lookup (search_tasks or filter_tasks)"

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
