"""Real TranscriptManager routing tests for CodeActActor.

Validates that CodeActActor uses ``execute_function`` for simple single-primitive
transcript operations, both with and without FunctionManager discovery tools.
"""

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)
from unity.contact_manager.types.contact import Contact
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes transcript question via execute_function."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        tm = ManagerRegistry.get_transcript_manager()

        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        handle = await actor.act(
            "Show the most recent message that mentions the budget.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_used_execute_function(handle)
        assert "primitives.transcripts.ask" in calls
        assert all(c.startswith("primitives.transcripts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_with_fm_tools():
    """CodeAct routes transcript query via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        tm = ManagerRegistry.get_transcript_manager()

        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        handle = await actor.act(
            "Show the most recent message that mentions the budget.",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_used_execute_function(handle)
        assert "primitives.transcripts.ask" in calls
        assert all(c.startswith("primitives.transcripts.") for c in calls)
