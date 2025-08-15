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
async def test_mouse_move_and_position(backend):
    instr = "Move the mouse to x=100 and y=200. Finish when done."
    for _ in range(90):
        await backend.act(instr)
        obs = await backend.observe(
            "Is the mouse at position (100,200)? Return matches True/False.",
        )
        if obs.get("matches"):
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("Timeout: pointer did not move to (100,200)")


@pytest.mark.asyncio
async def test_highlight_text_with_drag(backend):
    # 1) Echo the test phrase
    await backend.act(
        "Focus the 'xterm' window and type 'echo THIS IS A TEST' and press Enter. Finish when done.",
    )
    # 2) Highlight the substring 'A TEST' via drag
    instr = "Focus the 'xterm' window, drag over the text 'A TEST' to highlight it. Finish when done."
    for _ in range(90):
        await backend.act(instr)
        obs = await backend.observe(
            "Is the text 'A TEST' highlighted on screen? Return matches True/False.",
        )
        if obs.get("matches"):
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("Timeout: 'A TEST' was not highlighted by drag")


@pytest.mark.asyncio
async def test_scroll_until_ready(backend):
    # 1) Echo READY
    await backend.act(
        "Focus the 'xterm' window and type 'echo READY' and press Enter. Finish when done.",
    )
    # 2) Flood terminal with 70 lines
    await backend.act(
        "Focus the 'xterm' window and type 'for i in $(seq 1 70); do echo LINE $i; done' and press Enter. Finish when done.",
    )
    # 3) Scroll up until READY is visible
    instr = "Scroll up until the text 'READY' is visible on screen. Finish when done."
    for _ in range(90):
        await backend.act(instr)
        obs = await backend.observe(
            "The terminal shows the word READY visible on the screen.",
        )
        if obs.get("matches"):
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("Timeout: READY not visible after scrolling")


@pytest.mark.asyncio
async def test_click_positions(backend):
    # Step 1: click at first position
    x1, y1 = 200, 300
    instr1 = f"Click at pixel ({x1},{y1}). Finish when done."
    for _ in range(90):
        await backend.act(instr1)
        obs1 = await backend.observe(
            f"Is the mouse pointer at position ({x1},{y1})? Return matches True/False.",
        )
        if obs1.get("matches"):
            break
        await asyncio.sleep(1)
    else:
        pytest.fail(f"Timeout: pointer did not click to position ({x1},{y1})")

    # Step 2: click at second position
    x2, y2 = 300, 400
    instr2 = f"Click at pixel ({x2},{y2}). Finish when done."
    for _ in range(90):
        await backend.act(instr2)
        obs2 = await backend.observe(
            f"Is the mouse pointer at position ({x2},{y2})? Return matches True/False.",
        )
        if obs2.get("matches"):
            break
        await asyncio.sleep(1)
    else:
        pytest.fail(f"Timeout: pointer did not click to position ({x2},{y2})")
