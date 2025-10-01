from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.unit
@_handle_project
def test_get_images_for_message_returns_metadata_only_tm():
    tm = TranscriptManager()
    im = ImageManager()

    # Seed a single blue image
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue screen capture",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # Log a message that references the image via images mapping
    msg = Message(
        medium="whatsapp_call",
        sender_id=101,
        receiver_ids=[202],
        timestamp=datetime.now(timezone.utc),
        content="Video conference: screen looks one colour",
        exchange_id=424242,
        images={"[0:1]": int(img_id)},
    )
    tm.log_messages(msg)
    tm.join_published()

    # Fetch message_id back then query image metadata via private tool
    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    assert stored and len(stored) == 1
    mid = int(stored[0].message_id)

    items = tm._get_images_for_message(message_id=mid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "blue screen capture"
    assert isinstance(entry.get("timestamp"), str)
    # Ensure no raw image/base64 field is present
    assert "image" not in entry


@pytest.mark.unit
@_handle_project
def test_attach_image_to_context_promotes_image_block_tm():
    tm = TranscriptManager()
    im = ImageManager()

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny blue pixel",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    payload = tm._attach_image_to_context(image_id=int(img_id), note="see screen")
    assert isinstance(payload, dict)
    assert "image" in payload and isinstance(payload["image"], str)
    # Sanity: looks like base64 (decoding should not raise)
    base64.b64decode(payload["image"])  # will raise if invalid


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_can_use_images_for_color_question_tm():
    """
    Contrived scenario:
    - Create an image (blue pixel) and log a message that references it, framed as a
      video-conference issue ("my screen is one colour").
    - Ask TranscriptManager.ask a question that naturally invites image reasoning.
    - Expect a non-empty textual answer mentioning a blue-ish colour.
    - Also verify the tool trajectory shows either an image attachment or an image
      question tool being used (or an image block in messages).
    """
    tm = TranscriptManager()
    im = ImageManager()

    # Seed image
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc) - timedelta(days=7),
                "caption": "zoe video-conference blue screen",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # Log message linked to the image – note: keep content suggestive of a VC context
    tm.log_messages(
        Message(
            medium="whatsapp_call",
            sender_id=301,  # assume 'Zoe' in external context; ID is fine for this test
            receiver_ids=[302],
            timestamp=datetime.now(timezone.utc) - timedelta(days=7),
            content=(
                "Zoe on video conference: my screen is one colour, what is happening?"
            ),
            exchange_id=777001,
            images={"[0:1]": int(img_id)},
        ),
    )
    tm.join_published()

    # Ask the higher-level question; request reasoning steps for inspection
    h = await tm.ask(
        "What colour did Zoe's screen turn on the video conference last week?",
        _return_reasoning_steps=True,
    )
    answer, steps = await h.result()

    assert isinstance(answer, str) and answer.strip(), "Expected a textual description"
    # Heuristic: The tiny asset is blue; allow synonyms or general color mention
    assert any(
        kw in answer.lower() for kw in ("blue", "azure", "navy", "cyan")
    ), f"Answer does not reference blue-ish color: {answer!r}"

    # Validate that the trajectory reflects image usage in at least one of these ways:
    # - an image block promoted in messages (image_url / data:image)
    # - or explicit tool selection of ask_image/attach_image_to_context/attach_message_images_to_context
    serialized = str(steps)
    assert (
        ("image_url" in serialized)
        or ("data:image" in serialized)
        or ("ask_image" in serialized)
        or ("attach_image_to_context" in serialized)
        or ("attach_message_images_to_context" in serialized)
    ), "Expected image-aware reasoning (image block or image tools) to appear in steps"

    # The textual answer itself should not include raw image data
    assert "data:image" not in answer and "image_url" not in answer


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_boot_option_and_fourth_item_tm():
    tm = TranscriptManager()
    im = ImageManager()

    # Load real screenshots for the walkthrough
    import os

    here = os.path.dirname(__file__)
    grub_path = os.path.join(here, "grub_screen.jpg")
    wizard_path = os.path.join(here, "wizard_screen.jpg")
    with open(grub_path, "rb") as f:
        grub_bytes = f.read()
    with open(wizard_path, "rb") as f:
        wizard_bytes = f.read()

    [grub_id, wizard_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "GRUB boot menu screenshot",
                "data": grub_bytes,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Ubuntu installer wizard screenshot",
                "data": wizard_bytes,
            },
        ],
    )

    user_message = (
        "Boot the PC from the Ubuntu USB stick and, when the GRUB screen appears, "
        "select “Try or Install Ubuntu” (or use “Ubuntu (safe graphics)” if needed). "
        "After the live system loads, the installation wizard opens: choose your language on the left "
        "and click “Install Ubuntu” (or “Try Ubuntu” if you just want to explore)."
    )

    # Log the walkthrough message with images mapped to spans
    tm.log_messages(
        Message(
            medium="unify_chat",
            sender_id=10,
            receiver_ids=[20],
            timestamp=datetime.now(timezone.utc),
            content=user_message,
            exchange_id=88001,
            images={
                "[52:147]": int(grub_id),
                "[182:314]": int(wizard_id),
            },
        ),
    )
    tm.join_published()

    question = (
        "According to the walkthrough, which boot option should you select to proceed, "
        "and what is the fourth item shown in the boot menu?"
    )

    handle = await tm.ask(question, _return_reasoning_steps=True)
    answer, steps = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Expected textual answer"
    low = answer.lower()
    assert "try or install ubuntu" in low, f"Missing boot option in: {answer!r}"
    assert "test memory" in low, f"Missing fourth menu item in: {answer!r}"

    # Ensure some vision signal appeared during reasoning
    serialized = str(steps)
    assert (
        ("image_url" in serialized)
        or ("data:image" in serialized)
        or ("attach_message_images_to_context" in serialized)
        or ("attach_image_to_context" in serialized)
        or ("ask_image" in serialized)
    ), "Expected image-aware reasoning to be used"


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_compare_two_screens_requires_raw_context_tm():
    tm = TranscriptManager()
    im = ImageManager()

    # Load real screenshots for the comparison
    import os

    here = os.path.dirname(__file__)
    grub_path = os.path.join(here, "grub_screen.jpg")
    wizard_path = os.path.join(here, "wizard_screen.jpg")
    with open(grub_path, "rb") as f:
        grub_bytes = f.read()
    with open(wizard_path, "rb") as f:
        wizard_bytes = f.read()

    [grub_id, wizard_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "GRUB boot menu screenshot",
                "data": grub_bytes,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Ubuntu installer wizard screenshot",
                "data": wizard_bytes,
            },
        ],
    )

    user_message = (
        "Boot the PC from the Ubuntu USB stick and, when the GRUB screen appears, "
        "select “Try or Install Ubuntu” (or use “Ubuntu (safe graphics)” if needed). "
        "After the live system loads, the installation wizard opens: choose your language on the left "
        "and click “Install Ubuntu” (or “Try Ubuntu” if you just want to explore)."
    )

    # Log the walkthrough message with images mapped to spans
    tm.log_messages(
        Message(
            medium="unify_chat",
            sender_id=10,
            receiver_ids=[20],
            timestamp=datetime.now(timezone.utc),
            content=user_message,
            exchange_id=99001,
            images={
                "[52:147]": int(grub_id),
                "[182:314]": int(wizard_id),
            },
        ),
    )
    tm.join_published()

    question = (
        "Which screen looks the most modern and sleek? Which has the most clickable elements? Which appears brighter?\n"
        "Please answer in exactly three lines with this format:\n"
        "Modern: <answer>\n"
        "Clickable elements: <answer>\n"
        "Brightness: <answer>"
    )

    handle = await tm.ask(question, _return_reasoning_steps=True)
    answer, steps = await handle.result()

    # Basic answer shape
    assert isinstance(answer, str) and answer.strip(), "Expected textual answer"

    # Validate that the loop chose to ATTACH images (raw) rather than ask one-off questions
    # Inspect executed tool messages directly instead of brittle substring scans over the entire trace.
    def _executed_tools(step_msgs):
        names = []
        msgs = []
        for m in step_msgs or []:
            if isinstance(m, dict) and m.get("role") == "tool":
                nm = m.get("name")
                if isinstance(nm, str):
                    names.append(nm)
                    msgs.append(m)
        return names, msgs

    tool_names, tool_msgs = _executed_tools(steps)

    # Expect either a batched attach via message id or two individual attaches
    num_single_attaches = sum(1 for n in tool_names if n == "attach_image_to_context")
    used_batched_attach = any(
        n == "attach_message_images_to_context" for n in tool_names
    )
    assert used_batched_attach or (
        num_single_attaches >= 2
    ), "Expected images to be attached into the loop context (raw)"

    # If batched attach was used, check that at least two images were attached via the tool payload
    if used_batched_attach:
        attach_msgs = [
            m for m in tool_msgs if m.get("name") == "attach_message_images_to_context"
        ]

        # Extract attached_count from the tool content, handling both plain strings and text blocks
        def _extract_attached_count(m: dict) -> int:
            content = m.get("content")
            text = None
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for blk in content:
                    if (
                        isinstance(blk, dict)
                        and blk.get("type") == "text"
                        and isinstance(blk.get("text"), str)
                    ):
                        parts.append(blk.get("text"))
                if parts:
                    text = "\n".join(parts)
            if not text:
                return -1
            import json as _json

            # Try strict JSON first
            try:
                payload = _json.loads(text)
                return int(payload.get("attached_count", -1))
            except Exception:
                pass
            # Fallback: regex for attached_count
            try:
                import re as _re

                mobj = _re.search(r"\"attached_count\"\s*:\s*(\d+)", text)
                if mobj:
                    return int(mobj.group(1))
            except Exception:
                pass
            return -1

        counts = [
            c for c in (_extract_attached_count(m) for m in attach_msgs) if c >= 0
        ]
        assert (
            counts and max(counts) >= 2
        ), "Expected at least two images attached in batched attach"

    # Ensure no executed per-image ask tool was used
    assert all(
        n != "ask_image" for n in tool_names
    ), "Should not use per-image ask for multi-image comparison"

    # Parse three labeled lines
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]

    # Find the specific labeled entries (case-insensitive startswith)
    def _find_line(prefix: str) -> str:
        pfx = prefix.lower()
        for ln in lines:
            if ln.lower().startswith(pfx):
                return ln
        return ""

    modern_line = _find_line("Modern:")
    clickable_line = _find_line("Clickable elements:")
    brightness_line = _find_line("Brightness:")

    assert (
        modern_line and clickable_line and brightness_line
    ), f"Answer must contain three labeled lines. Got: {answer!r}"

    # Heuristics: installer/wizard is more modern and brighter; GRUB has more buttons/menu items
    mod_low = modern_line.lower()
    clk_low = clickable_line.lower()
    bri_low = brightness_line.lower()

    assert any(
        k in mod_low for k in ("wizard", "installer")
    ), f"Modern selection should reference installer/wizard: {modern_line!r}"
    assert any(
        k in clk_low for k in ("wizard", "installer", "installer wizard")
    ), f"Clickable selection should reference installer/wizard: {clickable_line!r}"
    assert any(
        k in bri_low for k in ("wizard", "installer", "installer wizard")
    ), f"Brightness selection should reference installer/wizard: {brightness_line!r}"
