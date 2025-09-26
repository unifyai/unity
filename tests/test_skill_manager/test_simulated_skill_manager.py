from __future__ import annotations

import pytest

from tests.helpers import _handle_project

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
