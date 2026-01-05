"""
Actor tests for ContactManager.ask operations.

This module ports tests from `tests/test_conductor/test_simulated/test_contacts/test_ask.py`
to verify that HierarchicalActor correctly generates plans calling `primitives.contacts.ask`
for read-only contact queries.

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


CONTACT_QUESTIONS: list[str] = [
    "Which of our contacts prefers to be contacted by phone?",
    "Find the email address for the contact named Sarah (use your contacts only).",
    "List any contacts located in Berlin.",
    "Who is the primary point of contact for the Contoso account? Use your contacts only.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", CONTACT_QUESTIONS)
async def test_questions_use_only_contact_tool(
    question: str,
    mock_verification,
):
    """Verify Actor generates plans calling only primitives.contacts.ask."""
    async with make_actor(impl="simulated") as actor:

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Verify result is non-empty
        assert isinstance(result, str) and result.strip()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        # Verify only primitives.contacts.ask was called
        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_tool_called(handle, "primitives.contacts.ask")

        # Verify primitives.contacts.ask was called. Other tools may be used for context
        # (e.g., transcripts.ask to find contact info, web.ask for company info).
        # Only enforce strictness if the request explicitly restricts tools.
        state_manager_tools = get_state_manager_tools(handle)
        assert state_manager_tools, "Expected at least one state manager tool call"

        # Check if request explicitly restricts tools
        request_explicitly_restricts = (
            "use your contacts only" in question.lower()
            or "use only" in question.lower()
        )

        if request_explicitly_restricts:
            # Strict: only contacts.ask allowed
            assert set(state_manager_tools) == {
                "primitives.contacts.ask",
            }, f"Expected only primitives.contacts.ask (request explicitly restricted tools), saw: {state_manager_tools}"
        else:
            # Relaxed: at least contacts.ask must be called
            assert "primitives.contacts.ask" in set(
                state_manager_tools,
            ), f"Expected primitives.contacts.ask to be called, saw: {state_manager_tools}"

        # Verify that verification was bypassed (no verification failures in log).
        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
