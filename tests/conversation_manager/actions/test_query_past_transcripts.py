"""
Transcript-focused ConversationManager brain routing tests.

Verify that the CM brain correctly routes to ``query_past_transcripts``
for scenarios where past conversation history is needed, and that the
query text includes the right details for the TranscriptManager to
service.

Uses SimulatedTranscriptManager — we only verify that the brain routes
correctly and passes a well-formed question, not the TM's actual answer.
The real TM is already thoroughly exercised in tests/transcript_manager/.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_efficient,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _assert_transcript_query_triggered(
    result,
    *,
    expected_substrings: list[str],
    cm=None,
) -> None:
    """Assert ``query_past_transcripts`` was called and each expected
    substring appears in the query text OR in the ``response_format`` keys.
    """
    events = filter_events_by_type(result.output_events, ActorHandleStarted)
    transcript_events = [
        e for e in events if e.action_name == "query_past_transcripts"
    ]
    assert transcript_events, (
        f"Expected query_past_transcripts to be triggered, "
        f"but got action(s): {[e.action_name for e in events] or 'none'}"
    )
    evt = transcript_events[0]
    query = evt.query.lower()
    rf_keys = " ".join((evt.response_format or {}).keys()).lower()
    searchable = f"{query} {rf_keys}"
    for substr in expected_substrings:
        assert substr.lower() in searchable, (
            f"Expected '{substr}' in query_past_transcripts query or response_format keys, "
            f"got query: {query}, response_format keys: {rf_keys}"
        )


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_reply_to_old_email_requests_email_id(initialized_cm):
    """Reply to an old email requires the email_id for threading.

    The brain must call ``query_past_transcripts`` and specifically request
    the ``email_id`` so that ``send_email`` can thread the reply correctly.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Reply to that email Alice sent me last month about the "
                "quarterly report. Just say thanks and that I've reviewed it. "
                "Make sure the reply is threaded on the original message."
            ),
        ),
    )

    _assert_transcript_query_triggered(
        result,
        expected_substrings=["email_id"],
        cm=cm,
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_past_sms_routes_to_transcripts(initialized_cm):
    """Asking about a past SMS should route to ``query_past_transcripts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What did Bob text me about yesterday?",
        ),
    )

    _assert_transcript_query_triggered(
        result,
        expected_substrings=["bob"],
        cm=cm,
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_past_call_summary_routes_to_transcripts(initialized_cm):
    """Asking for a call summary should route to ``query_past_transcripts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Can you summarise my last phone call with Charlie?",
        ),
    )

    _assert_transcript_query_triggered(
        result,
        expected_substrings=["charlie"],
        cm=cm,
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_past_email_search_by_topic(initialized_cm):
    """Searching for an email by topic should route to ``query_past_transcripts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find the email thread about the budget proposal from last week.",
        ),
    )

    _assert_transcript_query_triggered(
        result,
        expected_substrings=["budget"],
        cm=cm,
    )
    assert_efficient(result, 3)
