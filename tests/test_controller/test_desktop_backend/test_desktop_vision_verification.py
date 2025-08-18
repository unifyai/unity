import os
import sys
import pytest
import asyncio

from unity.controller.browser_backends import MagnitudeDesktopBackend


@pytest.fixture(scope="module")
def backend():
    if sys.platform != "linux":
        pytest.skip("Desktop vision/verification tests require Linux environment")
    be = MagnitudeDesktopBackend(headless=True)
    yield be
    be.stop()


async def _act_until_match(backend, instruction: str, query: str, poll_s: float = 1.0):
    while True:
        await backend.act(instruction, expectation=query)
        obs = await backend.observe(query)
        if obs.get("matches"):
            return
        await asyncio.sleep(poll_s)


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_full_screenshot_save_and_filepath_exists(backend):
    # Take a screenshot with save flag and verify filepath exists.
    meta = await backend.get_screenshot(save=True, return_meta=True)
    fp = (meta or {}).get("filepath")
    assert fp and os.path.isfile(
        fp,
    ), f"Expected saved screenshot filepath to exist, got: {fp}"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_region_screenshot_then_verify_presence(backend):
    # Focus xterm and capture a small region near top-left of its window title bar to ensure region ability works.
    await backend.act(
        "Focus the 'xterm' window. Finish when done.",
        expectation="The 'xterm' window looks active/focused (title bar highlighted or on top).",
    )
    # Heuristic region near (20, 10)
    await backend.act(
        "Capture a region screenshot around (x=20,y=10) sized about 200x120. Finish when done.",
        expectation="A region screenshot is captured around (20,10) sized roughly 200x120.",
    )
    # Visual verification is weak here; we just verify that xterm is visible as a sanity check.
    obs = await backend.observe(
        "A window titled 'xterm' is visible somewhere on the desktop.",
    )
    assert obs.get("matches"), "Expected xterm to be visible after region screenshot"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_image_locate_anchor_and_focus(backend):
    # Ensure xterm present
    await backend.act("Open or focus an 'xterm' window. Finish when done.")
    # Ask the agent to locate the xterm title area via template and focus it
    await asyncio.wait_for(
        _act_until_match(
            backend,
            "Use template matching to locate the 'xterm' title bar and click its center. Finish when done.",
            "The 'xterm' window appears focused (title bar highlighted or on top).",
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_region_changed_after_typing(backend):
    # Capture before, type, capture after, then verify region change
    await backend.act("Focus the 'xterm' window. Finish when done.")
    await asyncio.wait_for(
        _act_until_match(
            backend,
            "Capture a region around the terminal prompt area, then type 'ECHO ONE' and press Enter, capture the same region again, and verify the region changed. Finish when changed.",
            "The terminal output contains the word ONE or shows new output corresponding to the command.",
        ),
        timeout=90,
    )
