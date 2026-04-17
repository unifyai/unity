"""
Outbound file attachment: CM → CodeActActor → generate image → send with attachment.

Validates the full production path:
1. User asks the actor to generate an image (red square)
2. Actor creates a .png file via execute_code
3. CM brain sends the result back with attachment_filepath
4. LLM judge confirms the image depicts a red square
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import UnifyMessageReceived, UnifyMessageSent
from unity.file_manager.settings import get_local_root

pytestmark = [pytest.mark.integration, pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_generate_image_and_send_as_attachment(initialized_cm_codeact):
    """Generate a red square image via the actor and send it back as an attachment.

    Asserts:
    - The actor creates a .png file inside the workspace.
    - The CM brain calls send_unify_message with a non-empty attachment_filepath.
    - An LLM judge confirms the image contains a red square.
    """
    cm = initialized_cm_codeact
    cm.cm.vm_ready = True
    cm.cm.file_sync_complete = True
    local_root = Path(get_local_root())

    # ------------------------------------------------------------------
    # Step 1: Ask the actor to generate a red square image
    # ------------------------------------------------------------------
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Please generate a simple PNG image of a solid red square on a white "
                "background (200x200 pixels) and save it to the Outputs folder. "
                "Send me the image file when done."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result)

    # ------------------------------------------------------------------
    # Step 2: Verify a .png file was created inside the workspace
    # ------------------------------------------------------------------
    outputs_dir = local_root / "Outputs"
    png_files = list(outputs_dir.rglob("*.png")) if outputs_dir.exists() else []

    # Also check the root in case the actor saved it there
    if not png_files:
        png_files = list(local_root.rglob("*.png"))

    assert png_files, (
        f"Expected at least one .png file in the workspace after actor completed. "
        f"Actor result: {final}"
    )
    generated_image_path = png_files[0]

    # ------------------------------------------------------------------
    # Step 3: Inject actor result and run CM brain, capturing
    #         send_unify_message calls to verify attachment_filepath
    # ------------------------------------------------------------------
    # Mock upload_unify_attachment to return a fake success result so the
    # attachment flow completes without hitting real infrastructure.
    fake_upload = AsyncMock(
        return_value={
            "id": "fake-attachment-id",
            "filename": generated_image_path.name,
            "gs_url": "gs://fake-bucket/fake-path",
            "content_type": "image/png",
            "size_bytes": generated_image_path.stat().st_size,
        },
    )

    with patch(
        "unity.comms.primitives.comms_utils.upload_unify_attachment",
        fake_upload,
    ):
        await inject_actor_result(
            cm,
            handle_id=handle_id,
            result=final,
            success=True,
        )
        followup_events = await run_cm_until_wait(cm, max_steps=6)

    # Check for UnifyMessageSent events with attachments
    msg_events = [e for e in followup_events if isinstance(e, UnifyMessageSent)]
    attachment_events = [e for e in msg_events if e.attachments]

    assert attachment_events, (
        f"Expected a UnifyMessageSent event with a non-empty attachment. "
        f"Got {len(msg_events)} UnifyMessageSent event(s), none with attachments. "
        f"Actor result: {final}"
    )

    # Verify upload was called (which means attachment_filepath was provided
    # and the file was read successfully)
    assert fake_upload.called, (
        "Expected upload_unify_attachment to be called, meaning "
        "send_unify_message received a valid attachment_filepath."
    )

    # ------------------------------------------------------------------
    # Step 4: LLM judge — ask a vision model what's in the image
    # ------------------------------------------------------------------
    import base64

    from unity.common.llm_client import new_llm_client

    image_bytes = generated_image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    client = new_llm_client("gpt-4o-mini@openai")
    judge_text = await client.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is shown in this image? Describe it in one sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        max_tokens=100,
    )
    judge_text = judge_text.lower()
    assert (
        "red" in judge_text
    ), f"LLM judge did not mention 'red'. Response: {judge_text}"
    assert (
        "square" in judge_text
    ), f"LLM judge did not mention 'square'. Response: {judge_text}"
