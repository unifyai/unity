"""Tests for ask_about_file with image files (JPEG/PNG vision path)."""

from __future__ import annotations

import pytest
from pathlib import Path

from PIL import Image, ImageDraw

from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter

from tests.helpers import _handle_project
from tests.file_manager.helpers import ask_judge


def _create_test_image(path: Path, *, text: str = "HELLO", fmt: str = "JPEG") -> None:
    """Create a small image with visible text so a vision model can describe it."""
    img = Image.new("RGB", (200, 100), color=(30, 60, 180))
    draw = ImageDraw.Draw(img)
    draw.text((40, 35), text, fill=(255, 255, 255))
    img.save(str(path), format=fmt)


@pytest.fixture()
def image_fm(tmp_path: Path) -> FileManager:
    """FileManager backed by a local adapter rooted at tmp_path (non-singleton)."""
    return FileManager(adapter=LocalFileSystemAdapter(str(tmp_path)))


# ────────────────────────────────────────────────────────────────────────────
# 1.  JPEG image — basic vision Q&A
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ask_about_jpeg(image_fm: FileManager, tmp_path: Path):
    img_path = tmp_path / "screenshot.jpg"
    _create_test_image(img_path, text="HELLO", fmt="JPEG")

    handle = await image_fm.ask_about_file(
        "screenshot.jpg",
        "What text is visible in the image?",
    )
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip()
    verdict = await ask_judge(
        "What text is visible in the image?",
        answer,
        file_content="A 200x100 blue image with white text reading 'HELLO'",
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed image ask incorrect: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 2.  PNG image — same path, different format
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_ask_about_png(image_fm: FileManager, tmp_path: Path):
    img_path = tmp_path / "chart.png"
    _create_test_image(img_path, text="DATA", fmt="PNG")

    handle = await image_fm.ask_about_file(
        "chart.png",
        "What text is visible in the image?",
    )
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip()
    verdict = await ask_judge(
        "What text is visible in the image?",
        answer,
        file_content="A 200x100 blue image with white text reading 'DATA'",
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed image ask incorrect: {verdict}"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Missing image raises FileNotFoundError
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_ask_about_missing_image_raises(image_fm: FileManager):
    with pytest.raises(FileNotFoundError):
        await image_fm.ask_about_file("does_not_exist.jpg", "describe it")


# ────────────────────────────────────────────────────────────────────────────
# 4.  Handle exposes done() correctly
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_image_handle_done_lifecycle(image_fm: FileManager, tmp_path: Path):
    img_path = tmp_path / "lifecycle.png"
    _create_test_image(img_path, text="OK", fmt="PNG")

    handle = await image_fm.ask_about_file(
        "lifecycle.png",
        "What color is the background?",
    )
    assert not handle.done()
    await handle.result()
    assert handle.done()


# ────────────────────────────────────────────────────────────────────────────
# 5.  _return_reasoning_steps wraps the result as a tuple
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_image_return_reasoning_steps(image_fm: FileManager, tmp_path: Path):
    img_path = tmp_path / "steps.jpg"
    _create_test_image(img_path, text="STEP", fmt="JPEG")

    handle = await image_fm.ask_about_file(
        "steps.jpg",
        "What text is shown?",
        _return_reasoning_steps=True,
    )
    result = await handle.result()
    assert isinstance(result, tuple) and len(result) == 2
    answer, steps = result
    assert isinstance(answer, str) and answer.strip()
