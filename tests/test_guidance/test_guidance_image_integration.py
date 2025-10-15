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
        images=[{"image_id": int(img_id), "annotation": "icon screenshot"}],
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
        images=[
            {"image_id": int(grub_id), "annotation": "GRUB boot menu"},
            {"image_id": int(wizard_id), "annotation": "Installer wizard"},
        ],
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
        "Guided install reference: when booting from USB you will first see the GRUB menu with several boot choices. "
        "After selecting the appropriate option, the Ubuntu installer wizard opens with language selection on the left and clear action buttons to continue installation."
    )
    out = gm._add_guidance(
        title="Compare GRUB vs Installer screens",
        content=guidance_text,
        images=[
            {"image_id": int(grub_id), "annotation": "GRUB boot menu"},
            {"image_id": int(wizard_id), "annotation": "Installer wizard"},
        ],
    )
    gid = int(out["details"]["guidance_id"])

    question = (
        "I'm following the install guidance. First, visually compare BOTH screenshots side-by-side (you will need to look at them together):\n"
        "- decide which screen is more modern/sleek,\n"
        "- count which screen appears to have MORE clickable controls, and\n"
        "- estimate which screen looks BRIGHTER overall.\n"
        "Base your judgments ONLY on what you SEE in the images (do not rely on the guidance text).\n"
        "Then, answer in exactly three lines using this format:\n"
        "Next step: <what should I do next according to the guidance?>\n"
        "Most actionable: <which screen has clearer buttons/controls to proceed?>\n"
        "Boot menu item 4: <what is the fourth entry shown in the GRUB menu?>"
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

    # Note: per-image ask may be used for a sub-question; attachments above are required regardless.

    # Parse three labeled lines
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]

    # Find the specific labeled entries (case-insensitive startswith)
    def _find_line(prefix: str) -> str:
        pfx = prefix.lower()
        for ln in lines:
            if ln.lower().startswith(pfx):
                return ln
        return ""

    next_step_line = _find_line("Next step:")
    actionable_line = _find_line("Most actionable:")
    item4_line = _find_line("Boot menu item 4:")

    assert (
        next_step_line and actionable_line and item4_line
    ), f"Answer must contain three labeled lines. Got: {answer!r}"

    # Heuristics for guidance-style outputs
    ns_low = next_step_line.lower()
    act_low = actionable_line.lower()
    it4_low = item4_line.lower()

    # Next step should reference proceeding with the installer/wizard
    assert any(
        k in ns_low
        for k in (
            "install ubuntu",
            "start the installer",
            "open the installer",
            "continue with the installer",
            "click install",
        )
    ), f"Next step should reference using the installer/wizard: {next_step_line!r}"

    # Most actionable should reference the installer/wizard screen (has buttons)
    assert any(
        k in act_low
        for k in ("wizard", "installer", "install screen", "installer wizard")
    ), f"Most actionable should reference installer/wizard: {actionable_line!r}"

    # Boot menu item 4 should be 'Test memory'
    assert (
        "test memory" in it4_low
    ), f"Boot menu item 4 should mention 'Test memory': {item4_line!r}"
