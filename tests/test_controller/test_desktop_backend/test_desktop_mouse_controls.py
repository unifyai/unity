import sys
import pytest
import asyncio

from unity.controller.browser_backends import MagnitudeDesktopBackend


@pytest.fixture(scope="module")
def backend():
    if sys.platform != "linux":
        pytest.skip("Desktop mouse control tests require Linux environment")
    backend = MagnitudeDesktopBackend(headless=True)
    yield backend
    backend.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_mouse_move_and_position(backend):
    instr = "Move the mouse to x=100 and y=200."
    await asyncio.wait_for(
        backend.act(
            instr,
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_highlight_text_with_drag(backend):
    # 1) Echo the test phrase
    await backend.act(
        "Focus the 'xterm' window and type 'echo THIS IS A TEST' and press Enter.",
    )
    # 2) Highlight the substring 'A TEST' via drag
    instr = "Focus the 'xterm' window, drag over the text 'A TEST' to highlight it."
    await asyncio.wait_for(
        backend.act(
            instr,
        ),
        timeout=90,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_scroll_until_ready(backend):
    # 1) Echo READY
    await backend.act(
        "Focus the 'xterm' window and type 'echo READY' and press Enter.",
    )
    # 2) Flood terminal with 70 lines
    await backend.act(
        "Focus the 'xterm' window and type 'for i in $(seq 1 70); do echo LINE $i; done' and press Enter.",
    )
    # 3) Scroll up until READY is visible
    instr = "Scroll up until the text 'READY' is visible on screen."
    await asyncio.wait_for(
        backend.act(
            instr,
        ),
        timeout=120,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_click_positions(backend):
    # Step 1: click at first position
    x1, y1 = 200, 300
    instr1 = f"Click at pixel ({x1},{y1})."
    await asyncio.wait_for(
        backend.act(
            instr1,
        ),
        timeout=90,
    )

    # Step 2: click at second position
    x2, y2 = 300, 400
    instr2 = f"Click at pixel ({x2},{y2})."
    await asyncio.wait_for(
        backend.act(
            instr2,
        ),
        timeout=90,
    )
