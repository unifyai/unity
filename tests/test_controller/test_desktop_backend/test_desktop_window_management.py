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


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_focus_xterm(backend):
    instr = "Focus the window titled 'xterm'."
    await asyncio.wait_for(
        backend.act(
            instr,
            expectation="The 'xterm' window looks active/focused (title bar highlighted or on top).",
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
        backend.act(
            instr,
            expectation=(
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
    instr_max = "Maximize the window titled 'xterm'."
    await asyncio.wait_for(
        backend.act(
            instr_max,
        ),
        timeout=90,
    )

    # Restore (unmaximize)
    instr_restore = "Restore the 'xterm' window to a normal size (not maximized)."
    await asyncio.wait_for(
        backend.act(
            instr_restore,
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_minimize_and_focus_again(backend):
    # Minimize the window
    instr_min = "Minimize the 'xterm' window."
    await asyncio.wait_for(
        backend.act(
            instr_min,
        ),
        timeout=90,
    )

    # Bring it back to front
    instr_focus = "Focus the 'xterm' window again (restore/show it)."
    await asyncio.wait_for(
        backend.act(
            instr_focus,
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_close_and_reopen_xterm(backend):
    # Close xterm
    instr_close = "Close the window titled 'xterm'."
    await asyncio.wait_for(
        backend.act(
            instr_close,
        ),
        timeout=90,
    )

    # Re-open xterm (the agent can use app_open or launcher)
    instr_open = "Open a new xterm terminal window."
    await asyncio.wait_for(
        backend.act(
            instr_open,
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_app_install_and_use_cowsay(backend):
    # Ask the agent to install cowsay non-interactively and prove usage
    instr = (
        "If a terminal is available, use it; otherwise open one. "
        "Install 'cowsay' using 'apt-get update -y && apt-get install -y cowsay' and then run '/usr/games/cowsay READY'. "
        "Finish when the word READY from cowsay is visible on screen."
    )
    await asyncio.wait_for(
        backend.act(
            instr,
        ),
        timeout=90,
    )
