from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_verification_bypass_works(mock_verification):
    """Verify that verification is bypassed but plan generation works."""
    async with make_hierarchical_actor(impl="simulated") as actor:
        # Use a state-manager-only prompt to avoid any dependency on browser reasoning models.
        # Persist defaults to True on HierarchicalActor.act; in tests we want the handle to complete
        # immediately rather than pausing for interjections after the main plan finishes.
        handle = await actor.act(
            "Create a high-priority task called 'Draft Budget FY26'. Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is not None (routing test, not type test)
        assert result is not None

        # Verify plan was generated (not bypassed).
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code

        # Verify verification was bypassed (no verification failures in log).
        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
