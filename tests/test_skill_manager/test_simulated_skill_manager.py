from __future__ import annotations

import asyncio
import pytest

from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)

from unity.skill_manager.simulated import SimulatedSkillManager


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance
# ─────────────────────────────────────────────────────────────────────────────
def test_simulated_skill_docstrings_match_base():
    """
    Public methods in SimulatedSkillManager should copy the real
    BaseSkillManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.skill_manager.base import BaseSkillManager
    from unity.skill_manager.simulated import SimulatedSkillManager

    assert (
        BaseSkillManager.ask.__doc__.strip()
        in SimulatedSkillManager.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_skill_manager():
    sm = SimulatedSkillManager("Demo skills catalogue for unit-tests.")
    handle = await sm.ask("List your high-level skills.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Interject while running
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_skill_manager():
    sm = SimulatedSkillManager("Demo skills catalogue")
    handle = await sm.ask("Give me an overview of your skills.")
    await asyncio.sleep(0.05)
    # Async interject; no return value expected – should not raise
    await handle.interject("Also include any data-related skills, briefly.")
    # Still completes
    answer = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stop                                                                    #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_skill_manager():
    sm = SimulatedSkillManager()
    handle = await sm.ask("Produce an exhaustive list of capabilities.")
    await asyncio.sleep(0.05)
    handle.stop()
    result = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(result, str) and result.strip()
    assert handle.done(), "Handle should report done after stop()"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pause → Resume round-trip                                               #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_skill_manager():
    sm = SimulatedSkillManager()
    handle = await sm.ask("Summarize skills by domain.")

    # Pause before awaiting result; pause() returns None on this handle
    handle.pause()

    # Start result() – it should block while paused.
    res_task = asyncio.create_task(handle.result())
    await _assert_blocks_while_paused(res_task)

    # Resume and ensure execution proceeds.
    handle.resume()
    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Nested ask on handle                                                    #
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask_simulated_skill_manager():
    sm = SimulatedSkillManager()

    # Start an initial ask to obtain the live handle
    handle = await sm.ask("Summarize your most commonly used skills.")

    # Ask a nested question while running – returns a nested handle
    nested = await handle.ask("Which skills relate to spreadsheets?")
    nested_answer = await asyncio.wait_for(nested.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(nested_answer, str) and nested_answer.strip()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(handle_answer, str) and handle_answer.strip()
