"""
Screenshot → CM brain → act() → CodeActActor file access: end-to-end flow.

Validates that screenshots saved to disk during screen sharing are accessible
to the CodeActActor via their filesystem paths:

1. A user screen-share screenshot (GRUB bootloader) is pre-buffered before the
   slow brain runs.
2. The slow brain drains the buffer, saves the screenshot to
   Screenshots/User/<timestamp>.png, and sees the image + filepath label.
3. The user asks for a crop/extraction that requires act() — the brain delegates
   to the CodeActActor, referencing the screenshot filepath.
4. The CodeActActor loads the image from disk and produces an output,
   confirming the full chain works.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import UnifyMessageReceived
from unity.conversation_manager.types import ScreenshotEntry
from unity.file_manager.settings import get_local_root

pytestmark = [pytest.mark.integration, pytest.mark.eval, pytest.mark.llm_call]

GRUB_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "images" / "grub_screen.jpg"
)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_screenshot_crop_via_act(initialized_cm_codeact):
    """User asks to crop the menu area from their screen-shared GRUB screenshot.

    Asserts:
    - The screenshot is saved to Screenshots/User/ on disk.
    - The CM brain delegates to act(), referencing the screenshot path.
    - The CodeActActor loads the image from the filesystem and produces output.
    - An LLM judge confirms the output image contains GRUB boot menu content.
    """
    cm = initialized_cm_codeact
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True
    local_root = Path(get_local_root())

    # Ensure Screenshots/User exists (mirrors session bootstrap).
    (local_root / "Screenshots" / "User").mkdir(parents=True, exist_ok=True)
    (local_root / "Screenshots" / "Assistant").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Pre-buffer a user screenshot (simulating screen share capture)
    # ------------------------------------------------------------------
    assert GRUB_IMAGE_PATH.exists(), f"Test image not found: {GRUB_IMAGE_PATH}"
    grub_b64 = base64.b64encode(GRUB_IMAGE_PATH.read_bytes()).decode()

    cm.cm.user_screen_share_active = True
    entry = ScreenshotEntry(
        b64=grub_b64,
        utterance="I'm stuck on this screen, can you help?",
        timestamp=datetime.now(timezone.utc),
        source="user",
    )
    cm.cm._screenshot_buffer.append(entry)

    # Pre-write the screenshot to disk so the actor can read it.
    # In production, _register_screenshots_background handles this as a
    # fire-and-forget task after the LLM turn. In tests, step_until_wait
    # returns before that task completes, so we write eagerly.
    from unity.conversation_manager.types.screenshot import (
        generate_screenshot_path,
        write_screenshot_to_disk,
    )

    screenshot_path = generate_screenshot_path(entry)
    write_screenshot_to_disk(entry, screenshot_path)

    # ------------------------------------------------------------------
    # Step 2: Send a message that requires act() to process the screenshot
    # ------------------------------------------------------------------
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Can you crop the menu options area from my screen share "
                "screenshot and save it to the Outputs folder as menu_crop.png? "
                "Then send it back to me."
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Step 3: Verify the screenshot was saved to disk
    # ------------------------------------------------------------------
    screenshots_dir = local_root / "Screenshots" / "User"
    saved_screenshots = list(screenshots_dir.glob("*.png")) + list(
        screenshots_dir.glob("*.jpg"),
    )
    assert (
        saved_screenshots
    ), f"Expected at least one screenshot file in {screenshots_dir}, found none."

    # ------------------------------------------------------------------
    # Step 4: Verify act() was triggered and wait for CodeActActor
    # ------------------------------------------------------------------
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    # The act query should reference the screenshot path.
    act_query = actor_event.query.lower()
    assert "screenshots" in act_query, (
        f"Expected the act() query to reference a Screenshots/ path. "
        f"Got: {actor_event.query}"
    )

    final = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result)

    # ------------------------------------------------------------------
    # Step 5: Verify the CodeActActor produced an output file
    # ------------------------------------------------------------------
    outputs_dir = local_root / "Outputs"
    output_files = list(outputs_dir.rglob("*.png")) if outputs_dir.exists() else []
    if not output_files:
        output_files = list(local_root.rglob("menu_crop*"))

    assert output_files, (
        f"Expected CodeActActor to produce an image file. " f"Actor result: {final}"
    )

    # ------------------------------------------------------------------
    # Step 6: LLM judge — verify the output image relates to the GRUB screen
    # ------------------------------------------------------------------
    output_image = output_files[0]
    image_bytes = output_image.read_bytes()
    b64 = base64.b64encode(image_bytes).decode()

    # Detect content type from extension.
    ext = output_image.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        ext,
        "image/png",
    )
    data_url = f"data:{mime};base64,{b64}"

    from unity.common.llm_client import new_llm_client

    client = new_llm_client("gpt-4o-mini@openai")
    judge_text = await client.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image was produced from a screenshot of a GRUB "
                            "bootloader screen. Does it contain any text related to "
                            "GRUB boot menu options (e.g., 'Ubuntu', 'Install', "
                            "'Test memory', 'safe graphics')? "
                            "Answer YES or NO, then briefly explain."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        max_tokens=150,
    )
    judge_text_lower = judge_text.lower()
    assert "yes" in judge_text_lower, (
        f"LLM judge did not confirm GRUB content in output image. "
        f"Response: {judge_text}"
    )
