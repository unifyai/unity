import sys
import pytest
import asyncio

from unity.controller.browser_backends import MagnitudeDesktopBackend


@pytest.fixture(scope="module")
def backend():
    if sys.platform != "linux":
        pytest.skip("Desktop window management tests require Linux environment")
    be = MagnitudeDesktopBackend(headless=True)
    yield be
    be.stop()


async def _act_until_match(backend, instruction: str, query: str, poll_s: float = 1.0):
    while True:
        await backend.act(instruction)
        obs = await backend.observe(query)
        if obs.get("matches"):
            return
        await asyncio.sleep(poll_s)


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_focus_xterm(backend):
    instr = "Focus the window titled 'xterm'. Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr,
            "The 'xterm' window looks active/focused (title bar highlighted or on top).",
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_move_and_resize_xterm(backend):
    # Move to near top-left and resize to a medium size
    instr = (
        "Move the 'xterm' window to coordinates (50,50) and resize it to width 900 and height 700. "
        "Finish when done."
    )
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr,
            (
                "The 'xterm' window appears near the top-left area of the screen "
                "and is roughly around 900x700 pixels in size (±15%)."
            ),
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_maximize_and_restore_xterm(backend):
    # Maximize
    instr_max = "Maximize the window titled 'xterm'. Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_max,
            "The 'xterm' window appears maximized or filling most of the screen.",
        ),
        timeout=90,
    )

    # Restore (unmaximize)
    instr_restore = (
        "Restore the 'xterm' window to a normal size (not maximized). Finish when done."
    )
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_restore,
            "The 'xterm' window no longer fills most of the screen (not maximized).",
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_minimize_and_focus_again(backend):
    # Minimize the window
    instr_min = "Minimize the 'xterm' window. Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_min,
            "The 'xterm' window is not visible on the screen (appears minimized or hidden).",
        ),
        timeout=90,
    )

    # Bring it back to front
    instr_focus = "Focus the 'xterm' window again (restore/show it). Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_focus,
            "The 'xterm' window is visible on screen again and appears focused.",
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_close_and_reopen_xterm(backend):
    # Close xterm
    instr_close = "Close the window titled 'xterm'. Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_close,
            "There is no visible window titled 'xterm' on the screen.",
        ),
        timeout=90,
    )

    # Re-open xterm (the agent can use app_open or launcher)
    instr_open = "Open a new xterm terminal window. Finish when done."
    await asyncio.wait_for(
        _act_until_match(
            backend,
            instr_open,
            "A window titled 'xterm' is visible and appears focused on screen.",
        ),
        timeout=90,
    )
