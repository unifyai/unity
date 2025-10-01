from __future__ import annotations

from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


# Slightly larger solid-blue PNG to satisfy provider parsing (e.g., 8x8)
PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_guidance_persistent_image_context_then_reason():
    """
    Flow:
    1) Create an image and a guidance row that references it.
    2) Call GuidanceManager.ask with a request that attaches the image into the loop
       using the new attach tool.
    3) Ask a follow-up that depends on seeing the image.
    4) Expect a non-empty textual answer (no base64 in answer), leveraging persistent context.
    """
    # Seed one image
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue pixel art icon",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # Create a guidance entry pointing to that image
    gm = GuidanceManager()
    out = gm._add_guidance(
        title="Pixel Icon",
        content="Review the icon layout and color.",
        images={"[0:1]": int(img_id)},
    )
    gid = out["details"]["guidance_id"]

    # Step 1: Attach the guidance-linked image persistently to the loop context
    h1 = await gm.ask(
        f"For guidance ID {gid}, attach the image so you can see it, then confirm once attached.",
    )
    ans1 = await h1.result()
    assert (
        isinstance(ans1, str) and ans1.strip()
    ), "Attachment confirmation should be text"
    assert "data:image" not in ans1 and "image_url" not in ans1

    # Step 2: Follow-up question that benefits from persistent image context
    h2 = await gm.ask("Now, describe the dominant color visible.")
    ans2 = await h2.result()
    assert isinstance(ans2, str) and ans2.strip(), "Expected a textual description"
    # Heuristic: The tiny asset is blue; allow synonyms or general color mention
    assert any(
        kw in ans2.lower() for kw in ("blue", "azure", "navy", "cyan")
    ), f"Answer does not reference blue-ish color: {ans2!r}"


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_guidance_boot_option_and_fourth_item_gm():
    gm = GuidanceManager()
    im = ImageManager()

    # Load real screenshots for the walkthrough (reuse TM assets)
    import os

    here = os.path.dirname(__file__)
    tm_dir = os.path.abspath(os.path.join(here, "..", "test_transcript_manager"))
    grub_path = os.path.join(tm_dir, "grub_screen.jpg")
    wizard_path = os.path.join(tm_dir, "wizard_screen.jpg")
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

    # Create a guidance entry that mirrors an installation step-by-step
    guidance_text = (
        "Ubuntu USB installation: when the GRUB screen appears during boot, select "
        "'Try or Install Ubuntu' to proceed (use 'Ubuntu (safe graphics)' if needed). "
        "After the live session starts, the installer wizard opens; choose your language on the left "
        "and click 'Install Ubuntu' to begin."
    )

    out = gm._add_guidance(
        title="Ubuntu install from USB (boot + wizard)",
        content=guidance_text,
        images={
            "[52:147]": int(grub_id),
            "[182:314]": int(wizard_id),
        },
    )
    gid = int(out["details"]["guidance_id"])

    question = (
        "At the boot menu, which option should I pick to start the installation, "
        "and what is the fourth item listed in that GRUB menu?"
    )

    handle = await gm.ask(question, _return_reasoning_steps=True)
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
        or ("attach_guidance_images_to_context" in serialized)
        or ("attach_image_to_context" in serialized)
        or ("ask_image" in serialized)
    ), "Expected image-aware reasoning to be used"

    # The textual answer itself should not include raw image data
    assert "data:image" not in answer and "image_url" not in answer


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_guidance_compare_two_screens_requires_raw_context_gm():
    gm = GuidanceManager()
    im = ImageManager()

    # Load real screenshots for the comparison (reuse TM assets)
    import os

    here = os.path.dirname(__file__)
    tm_dir = os.path.abspath(os.path.join(here, "..", "test_transcript_manager"))
    grub_path = os.path.join(tm_dir, "grub_screen.jpg")
    wizard_path = os.path.join(tm_dir, "wizard_screen.jpg")
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

    guidance_text = (
        "Guided install reference: on boot, the GRUB menu is shown before the desktop. "
        "After choosing the appropriate boot option, the Ubuntu installer wizard opens with language options and action buttons."
    )
    out = gm._add_guidance(
        title="Compare GRUB vs Installer screens",
        content=guidance_text,
        images={
            "[52:147]": int(grub_id),
            "[182:314]": int(wizard_id),
        },
    )
    gid = int(out["details"]["guidance_id"])

    question = (
        "Which screen looks the most modern and sleek? Which has the most clickable elements? Which appears brighter?\n"
        "Please answer in exactly three lines with this format:\n"
        "Modern: <answer>\n"
        "Clickable elements: <answer>\n"
        "Brightness: <answer>"
    )

    handle = await gm.ask(question, _return_reasoning_steps=True)
    answer, steps = await handle.result()

    # Basic answer shape
    assert isinstance(answer, str) and answer.strip(), "Expected textual answer"

    # Validate that the loop chose to ATTACH images (raw) rather than ask one-off questions
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

    # Expect either a batched attach via guidance id or two individual attaches
    num_single_attaches = sum(1 for n in tool_names if n == "attach_image_to_context")
    used_batched_attach = any(
        n == "attach_guidance_images_to_context" for n in tool_names
    )
    assert used_batched_attach or (
        num_single_attaches >= 2
    ), "Expected images to be attached into the loop context (raw)"

    # If batched attach was used, check that at least two images were attached via the tool payload
    if used_batched_attach:
        attach_msgs = [
            m for m in tool_msgs if m.get("name") == "attach_guidance_images_to_context"
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

    # Heuristics: installer/wizard is more modern and brighter; clickable elements often richer in the installer UI
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
