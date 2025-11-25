import asyncio
import pytest

from unity.skill_manager.skill_manager import SkillManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_skillmanager_ask():
    """
    Verify a flat, in‑flight SkillManager.ask loop reports a minimal structure.
    """
    sm = SkillManager()

    h = await sm.ask("What high-level skills do you have for spreadsheets and CSVs?")

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
            "tool": "SkillManager.ask",
            "children": [],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=60)  # type: ignore[attr-defined]
        except Exception:
            pass
