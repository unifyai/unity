import os
import sys
import pytest

from unity.controller.browser_backends import MagnitudeDesktopBackend


@pytest.fixture(scope="module")
def backend():
    if sys.platform != "linux":
        pytest.skip("Desktop vision/verification tests require Linux environment")
    be = MagnitudeDesktopBackend(headless=True)
    yield be
    be.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_full_screenshot_save_and_filepath_exists(backend):
    # Take a screenshot with save flag and verify filepath exists.
    meta = await backend.get_screenshot(save=True, return_meta=True)
    fp = (meta or {}).get("filepath")
    assert fp and os.path.isfile(
        fp,
    ), f"Expected saved screenshot filepath to exist, got: {fp}"


# @pytest.mark.asyncio
# @pytest.mark.timeout(90)
# async def test_region_screenshot_then_verify_presence(backend):
#     # Focus xterm and capture a small region near top-left of its window title bar to ensure region ability works.
#     await backend.act(
#         "Focus the 'xterm' window.",
#     )
#     # Heuristic region near (20, 10)
#     await backend.act(
#         "Capture a region screenshot around (x=20,y=10) sized about 200x120.",
#     )
#     # Visual verification through act's expectation
#     status = await backend.act(
#         "",
#         expectation="A window titled 'xterm' is visible somewhere on the desktop.",
#     )
#     assert status == "success", "Expected xterm to be visible after region screenshot"


# @pytest.mark.asyncio
# @pytest.mark.timeout(90)
# async def test_image_locate_anchor_and_focus(backend):
#     # Ensure xterm present
#     await backend.act("Open or focus an 'xterm' window.")
#     # Ask the agent to locate the xterm title area via template and focus it
#     await asyncio.wait_for(
#         backend.act(
#             "Use template matching to locate the 'xterm' title bar and click its center.",
#             expectation="The 'xterm' window appears focused (title bar highlighted or on top).",
#         ),
#         timeout=90,
#     )


# @pytest.mark.asyncio
# @pytest.mark.timeout(90)
# async def test_region_changed_after_typing(backend):
#     # Capture before, type, capture after, then verify region change
#     await backend.act("Focus the 'xterm' window.")
#     await asyncio.wait_for(
#         backend.act(
#             "Capture a region around the terminal prompt area, then type 'ECHO ONE' and press Enter, capture the same region again, and verify the region changed. Finish when changed.",
#             expectation="The terminal output contains the word ONE or shows new output corresponding to the command.",
#         ),
#         timeout=90,
#     )
