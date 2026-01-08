from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project
from unity.image_manager.types import AnnotatedImageRefs, RawImageRef, AnnotatedImageRef
from unity.contact_manager.types.contact import Contact


PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


@_handle_project
def test_get_images_returns_metadata_only():
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

    # Log a message that references the image via AnnotatedImageRefs
    exchange_id = 424242
    tm.log_messages(
        {
            "medium": "phone_call",
            "sender_id": Contact(first_name="Zoe"),
            "receiver_ids": [Contact(first_name="Alex")],
            "timestamp": datetime.now(timezone.utc),
            "content": "Video conference: screen looks one colour",
            "exchange_id": exchange_id,
            "images": AnnotatedImageRefs.model_validate(
                [
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(img_id)),
                        annotation="blue screen",
                    ),
                ],
            ),
        },
    )
    tm.join_published()

    # Fetch message_id back then query image metadata via private tool
    stored = tm._filter_messages(filter=f"exchange_id == {exchange_id}")["messages"]
    assert stored and len(stored) == 1
    mid = int(stored[0].message_id)

    items = tm._get_images_for_message(message_id=mid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "blue screen capture"
    assert isinstance(entry.get("timestamp"), str)
    # Stored metadata should include the freeform annotation
    assert entry.get("annotation") == "blue screen"
    # Ensure no raw image/base64 field is present
    assert "image" not in entry


@_handle_project
def test_attach_image_promotes_block():
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


@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_uses_images_for_color(static_now):
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
                "timestamp": static_now - timedelta(days=7),
                "caption": "zoe video-conference blue screen",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # Log message linked to the image – provide Contact objects so contacts are auto-created
    tm.log_messages(
        {
            "medium": "phone_call",
            "sender_id": Contact(first_name="Zoe"),
            "receiver_ids": [Contact(first_name="Sam")],
            "timestamp": static_now - timedelta(days=7),
            "content": (
                "Zoe on video conference: my screen is one colour, what is happening?"
            ),
            "exchange_id": 777001,
            "images": AnnotatedImageRefs.model_validate(
                [
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(img_id)),
                        annotation="blue screen",
                    ),
                ],
            ),
        },
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


@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_boot_option(static_now):
    tm = TranscriptManager()
    im = ImageManager()

    # Load real screenshots for the walkthrough
    import os

    here = os.path.dirname(__file__)
    images_dir = os.path.abspath(os.path.join(here, "..", "images"))
    grub_path = os.path.join(images_dir, "grub_screen.jpg")
    wizard_path = os.path.join(images_dir, "wizard_screen.jpg")
    with open(grub_path, "rb") as f:
        grub_bytes = f.read()
    with open(wizard_path, "rb") as f:
        wizard_bytes = f.read()

    [grub_id, wizard_id] = im.add_images(
        [
            {
                "timestamp": static_now,
                "caption": "GRUB boot menu screenshot",
                "data": grub_bytes,
            },
            {
                "timestamp": static_now,
                "caption": "Ubuntu installer wizard screenshot",
                "data": wizard_bytes,
            },
        ],
    )

    user_message = (
        "Boot the PC from the Ubuntu USB stick and, when the GRUB screen appears, "
        'select "Try or Install Ubuntu" (or use "Ubuntu (safe graphics)" if needed). '
        "After the live system loads, the installation wizard opens: choose your language on the left "
        'and click "Install Ubuntu" (or "Try Ubuntu" if you just want to explore).'
    )

    # Log the walkthrough message with annotated image references
    tm.log_messages(
        {
            "medium": "unify_message",
            "sender_id": Contact(first_name="Jamie"),
            "receiver_ids": [Contact(first_name="Taylor")],
            "timestamp": static_now,
            "content": user_message,
            "exchange_id": 88001,
            "images": AnnotatedImageRefs.model_validate(
                [
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(grub_id)),
                        annotation="GRUB boot menu screenshot for boot selection",
                    ),
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(wizard_id)),
                        annotation="Ubuntu installer wizard screenshot",
                    ),
                ],
            ),
        },
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


@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_compare_screens_raw_context(static_now):
    """
    Multi-image comparison test: verify the model can reason about two screenshots.

    The model may use either ask_image (one-off questions per image) or attach tools
    (persistent visual context). Both approaches are valid as long as some image
    tool is used and the answer correctly identifies the installer wizard as more
    modern, having more clickable elements, and being brighter than GRUB.
    """
    from tests.assertion_helpers import find_tool_calls_and_results

    tm = TranscriptManager()
    im = ImageManager()

    # Load real screenshots for the comparison
    import os

    here = os.path.dirname(__file__)
    images_dir = os.path.abspath(os.path.join(here, "..", "images"))
    grub_path = os.path.join(images_dir, "grub_screen.jpg")
    wizard_path = os.path.join(images_dir, "wizard_screen.jpg")
    with open(grub_path, "rb") as f:
        grub_bytes = f.read()
    with open(wizard_path, "rb") as f:
        wizard_bytes = f.read()

    [grub_id, wizard_id] = im.add_images(
        [
            {
                "timestamp": static_now,
                "caption": "GRUB boot menu screenshot",
                "data": grub_bytes,
            },
            {
                "timestamp": static_now,
                "caption": "Ubuntu installer wizard screenshot",
                "data": wizard_bytes,
            },
        ],
    )

    user_message = (
        "Boot the PC from the Ubuntu USB stick and, when the GRUB screen appears, "
        'select "Try or Install Ubuntu" (or use "Ubuntu (safe graphics)" if needed). '
        "After the live system loads, the installation wizard opens: choose your language on the left "
        'and click "Install Ubuntu" (or "Try Ubuntu" if you just want to explore).'
    )

    # Log the walkthrough message with annotated image references
    tm.log_messages(
        {
            "medium": "unify_message",
            "sender_id": Contact(first_name="Jamie"),
            "receiver_ids": [Contact(first_name="Taylor")],
            "timestamp": static_now,
            "content": user_message,
            "exchange_id": 99001,
            "images": AnnotatedImageRefs.model_validate(
                [
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(grub_id)),
                        annotation="GRUB boot menu screenshot",
                    ),
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=int(wizard_id)),
                        annotation="Ubuntu installer wizard screenshot",
                    ),
                ],
            ),
        },
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

    # 1) Verify the model used some image tool (ask_image or attach variants)
    ask_image_calls, _ = find_tool_calls_and_results(steps, "ask_image")
    attach_single_calls, _ = find_tool_calls_and_results(
        steps,
        "attach_image_to_context",
    )
    attach_batch_calls, _ = find_tool_calls_and_results(
        steps,
        "attach_message_images_to_context",
    )

    assert (
        ask_image_calls or attach_single_calls or attach_batch_calls
    ), "Expected the model to use ask_image or attach tools to process the images"

    # 2) Parse three labeled lines from answer
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]

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

    # 3) Verify the installer wizard (image_id 1 / "screen 2") is identified as the winner
    # Accept various phrasings: "wizard", "installer", "screen 2", "image 1", "image_id 1"
    installer_keywords = ("wizard", "installer", "screen 2", "image 1", "image_id 1")

    mod_low = modern_line.lower()
    clk_low = clickable_line.lower()
    bri_low = brightness_line.lower()

    assert any(
        k in mod_low for k in installer_keywords
    ), f"Modern selection should reference installer/wizard: {modern_line!r}"
    assert any(
        k in clk_low for k in installer_keywords
    ), f"Clickable selection should reference installer/wizard: {clickable_line!r}"
    assert any(
        k in bri_low for k in installer_keywords
    ), f"Brightness selection should reference installer/wizard: {brightness_line!r}"
