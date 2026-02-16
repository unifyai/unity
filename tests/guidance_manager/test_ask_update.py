from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval

from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_semantic_search_text_only():
    gm = GuidanceManager()
    # Seed some simple guidance entries
    gm.add_guidance(title="Onboarding", content="How to onboard a user step by step")
    gm.add_guidance(title="Billing", content="Explains invoices and payments")
    gm.add_guidance(title="Support", content="Handling support tickets efficiently")

    handle = await gm.ask("Find the guidance that explains onboarding")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip()
    assert "onboard" in answer.lower() or "onboarding" in answer.lower()


@pytest.mark.asyncio
@_handle_project
async def test_update_then_ask():
    gm = GuidanceManager()
    # Natural-language create
    handle = await gm.update(
        "Create guidance titled 'Checklists' with content 'Standard operating checklists for releases.'",
    )
    await handle.result()

    # Ask to verify it exists
    handle2 = await gm.ask("Show the guidance about checklists")
    ans = await handle2.result()
    assert isinstance(ans, str) and ans.strip()
    assert "checklist" in ans.lower()


@pytest.mark.asyncio
@_handle_project
async def test_update_modify_then_filter():
    gm = GuidanceManager()
    gid = gm.add_guidance(title="Ops Runbook", content="Legacy steps")["details"][
        "guidance_id"
    ]

    # Natural-language update to modify content
    h = await gm.update(
        f"Update guidance ID {gid}: set content to 'Modernised ops runbook'.",
    )
    await h.result()

    rows = gm.filter(filter=f"guidance_id == {gid}")
    assert rows and "modernised" in rows[0].content.lower()
