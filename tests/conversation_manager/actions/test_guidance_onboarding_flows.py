"""
tests/conversation_manager/actions/test_guidance_onboarding_flows.py
=====================================================================

Tests for nuanced onboarding, explanatory, and guidance flows where the
boundary between "conversational context" and "actionable content worth
storing" is blurred.

These scenarios test whether the CM correctly:

1. Recognises when the user is teaching it something worth persisting
   (even without an explicit "save this" directive).
2. Accumulates multi-message explanations and stores them in batch when
   the explanation is complete, rather than acting on each fragment.
3. Engages with screen-share tutorials via ``persist=True`` sessions
   that can absorb ongoing teaching and then transition to interactive
   execution when the user says "now you try".
4. References screenshot filepaths in act queries when visual
   demonstrations are involved.

These are eval tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    filter_events_by_type,
    has_steering_tool_call,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    UnifyMessageReceived,
    ActorHandleStarted,
)
from unity.conversation_manager.types import ScreenshotEntry

pytestmark = pytest.mark.eval

# Minimal valid 8x8 white JPEG used as a stand-in screenshot.
# Content is irrelevant — what matters is the LLM seeing screenshot entries
# with filepaths in the rendered state alongside the user's verbal descriptions.
# Must be actual JPEG since the multimodal API validates the content type.
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
    "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
    "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEA"
    "AAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    "E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RF"
    "RkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKj"
    "pKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP0"
    "9fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgEC"
    "BAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLR"
    "ChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0"
    "dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbH"
    "yMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iig"
    "gD//2Q=="
)


def _make_screenshot(utterance: str, filepath: str) -> ScreenshotEntry:
    """Create a screenshot entry with a pre-set filepath (skips disk I/O)."""
    return ScreenshotEntry(
        b64=_TINY_JPEG_B64,
        utterance=utterance,
        timestamp=datetime.now(timezone.utc),
        source="user",
        filepath=filepath,
    )


def _assert_no_act_triggered(result, description: str) -> None:
    """Assert that no act() call was made during this step."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Expected NO ActorHandleStarted events — {description}. "
        f"Got {len(actor_events)} event(s) with queries: "
        f"{[e.query for e in actor_events]}"
    )


def _assert_persist_true(cm, description: str) -> None:
    """Assert that the most recent act() call used persist=True."""
    actions = cm.cm.in_flight_actions
    assert actions, f"No in-flight actions found — {description}"
    last_action = list(actions.values())[-1]
    assert last_action.get("persist") is True, (
        f"Expected persist=True, got persist={last_action.get('persist')}. "
        f"{description}"
    )


def _get_act_query(result) -> str:
    """Extract the query string from the ActorHandleStarted event."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert actor_events, "No ActorHandleStarted event found"
    return actor_events[0].query


def _has_act(result) -> bool:
    """Check whether an act() call was made during this step."""
    return len(filter_events_by_type(result.output_events, ActorHandleStarted)) > 0


# ---------------------------------------------------------------------------
#  1. Explicit "remember this" with facts → one-shot act to store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_explicit_remember_facts_triggers_act(initialized_cm):
    """User explicitly asks the assistant to remember important facts.

    "Remember this, it's important: {facts}" should trigger act to store
    the information — even though the user didn't say "save to knowledge"
    or "store this guidance". The intent to persist is clear from context.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Remember this, it's important: our API rate limit is 100 "
                "requests per minute for the standard tier and 1000 for premium. "
                "The rate limit resets every 60 seconds."
            ),
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "'Remember this' should trigger act to store the facts",
        cm=cm,
    )
    query = _get_act_query(result)
    query_lower = query.lower()
    assert (
        "rate limit" in query_lower or "100" in query_lower
    ), f"Expected act query to reference the rate limit facts. Got: {query}"
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  2. Multi-message procedural explanation → act after completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_multi_message_explanation_stored_on_completion(initialized_cm):
    """User explains a procedure across several messages, then signals done.

    The assistant should accumulate the explanation and only trigger act
    to store it when the user signals completion, not on each intermediate
    message. The act query should capture the FULL procedure, not just
    the final fragment.
    """
    cm = initialized_cm

    # Message 1: Start of procedure, explicitly multi-step — don't act yet
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "So, to log into the onboarding platform, there are a few "
                "steps. First, you go to portal.example.com"
            ),
        ),
    )
    _assert_no_act_triggered(result1, "First fragment should not trigger act")
    assert_efficient(result1, 3)

    # Message 2: Continuation — still don't act
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Then enter company code ACME-2024, and use your employee ID "
                "as the username"
            ),
        ),
    )
    _assert_no_act_triggered(result2, "Second fragment should not trigger act")
    assert_efficient(result2, 3)

    # Message 3: Completion signal — NOW act to store the full procedure
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "And lastly, the password is your start date in YYYYMMDD "
                "format plus your badge number. That's the full login process."
            ),
        ),
    )
    assert_act_triggered(
        result3,
        ActorHandleStarted,
        "Completion signal should trigger act to store the procedure",
        cm=cm,
    )

    query = _get_act_query(result3)
    query_lower = query.lower()
    assert "portal" in query_lower or "onboarding" in query_lower, (
        f"Expected act query to reference the portal/onboarding context "
        f"from earlier messages, not just the password fragment. Got: {query}"
    )
    assert (
        "password" in query_lower or "badge" in query_lower
    ), f"Expected act query to include the password procedure. Got: {query}"
    assert_efficient(result3, 3)


# ---------------------------------------------------------------------------
#  3. Detailed single-message explanation → act to store is desirable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_detailed_procedural_explanation_handled(initialized_cm):
    """User gives a detailed procedural explanation in one message.

    Without an explicit "remember this", the LLM may either proactively
    store the procedure via act, or just acknowledge it. Both outcomes
    are acceptable. If it does call act, the query should capture the
    procedural content.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "So here's how the quarterly reporting works: first you pull "
                "the data from our analytics dashboard, then you cross-reference "
                "it with the finance team's numbers, and the tricky part is the "
                "currency conversion — you have to use the exchange rate from the "
                "first business day of the quarter, not the current rate."
            ),
        ),
    )

    if _has_act(result):
        query = _get_act_query(result)
        query_lower = query.lower()
        assert "quarterly" in query_lower or "reporting" in query_lower, (
            f"Act was triggered but query doesn't capture the procedure. "
            f"Got: {query}"
        )
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  4. Screen share demonstration → act to store visual guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_visual_demonstration_triggers_guidance_storage(initialized_cm):
    """User demonstrates a process on screen and explains it verbally.

    A detailed visual demonstration with verbal explanation should trigger
    act to store as guidance. The query should reference the screenshot
    filepaths so the Actor can process the visual content.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Context-setting — no act
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Let me show you how our admin panel works.",
        ),
    )
    _assert_no_act_triggered(result1, "Context-setting should not trigger act")

    # Phase 2: Detailed visual explanation with screenshot
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="See this section here? Whenever a customer calls about billing...",
            filepath="Screenshots/User/admin-panel-billing.jpg",
        ),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "See this section here? Whenever a customer calls about a "
                "billing issue, you always check this Override section first, "
                "then click the Transactions tab to see the full history."
            ),
        ),
    )

    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "Visual demonstration should trigger act to store guidance",
        cm=cm,
    )
    query = _get_act_query(result2)
    query_lower = query.lower()
    assert "billing" in query_lower or "override" in query_lower, (
        f"Expected act query to reference the billing/override procedure. "
        f"Got: {query}"
    )
    assert "screenshot" in query_lower or "screen" in query_lower, (
        f"Expected act query to reference the screenshot/screen context. "
        f"Got: {query}"
    )
    assert_efficient(result2, 3)


# ---------------------------------------------------------------------------
#  5. Visual demo → "now you try" → persist session by the interactive phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_visual_demo_then_you_try_produces_persist_session(initialized_cm):
    """User demonstrates a process, then says "now you try".

    The LLM may engage with the screen share demonstration at any point
    during the teaching phase — an early persist=True session that absorbs
    ongoing teaching is a valid strategy. What matters is that by the time
    the user says "now you try":

    1. The task details have been conveyed to an action (via act query or
       interject).
    2. That action uses persist=True — the task is a multi-step UI
       walkthrough on a live screen share, so a persist=False action
       could complete before the user's next instruction arrives.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Context — no act
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Let me show you how to submit expense reports in our system.",
        ),
    )
    _assert_no_act_triggered(result1, "Context-setting should not trigger act")

    # Phase 2: Demonstration — act may or may not fire here (both acceptable)
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="First you click New Expense, I'll walk you through each field",
            filepath="Screenshots/User/expense-new-form.jpg",
        ),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "So the first step is clicking New Expense here — I'll walk "
                "you through each field. You put the amount here, and the "
                "category here..."
            ),
        ),
    )
    acted_in_demo = _has_act(result2)

    # Phase 2b: Continuation of demonstration
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="Then you attach the receipt down here",
            filepath="Screenshots/User/expense-receipt-attach.jpg",
        ),
    )
    result2b = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Then you attach the receipt photo down here at the bottom, "
                "and hit Submit. That's the whole process."
            ),
        ),
    )

    # Phase 3: "Now you try" — interactive task begins
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="Okay now you try",
            filepath="Screenshots/User/expense-form-filled.jpg",
        ),
    )
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Okay, now you try — submit an expense report for yesterday's "
                "team lunch, $45.50 under the Meals category."
            ),
        ),
    )

    # The key assertion: by phase 3 the interactive task must be conveyed
    # to a persist=True action. The task is a multi-step UI walkthrough on
    # a live screen share — the user is watching and will give corrections
    # or further instructions, so a persist=False action could close before
    # the next incremental step arrives.
    #
    # Multiple strategies for getting here are valid:
    #
    #  (a) Single persist session from the demo absorbs the "now you try"
    #      via interject — persist session already running.
    #  (b) Demo persist session is stopped, new act(persist=True) for the
    #      interactive execution.
    #  (c) No act during demo, single act(persist=True) on "now you try".
    #
    # What matters: (1) task details reach an action, and (2) that action
    # is persist=True so the session stays alive for ongoing interaction.
    act_events_3 = filter_events_by_type(result3.output_events, ActorHandleStarted)
    if act_events_3:
        query_lower = act_events_3[0].query.lower()
        assert "expense" in query_lower or "45" in query_lower, (
            f"New act in phase 3 should reference the task. "
            f"Got: {act_events_3[0].query}"
        )
    else:
        assert has_steering_tool_call(cm, "interject_"), (
            "No new act in phase 3 — expected interject to relay the "
            f"interactive task to the existing session. "
            f"Tool calls: {cm.all_tool_calls}"
        )

    # The action handling the interactive execution must be persist=True.
    actions = cm.cm.in_flight_actions
    assert actions, (
        "Expected at least one in-flight action after 'now you try'. "
        f"Tool calls: {cm.all_tool_calls}"
    )
    last_action = list(actions.values())[-1]
    assert last_action.get("persist") is True, (
        f"The interactive screen-share execution must use persist=True — "
        f"a persist=False action could complete before the user's next "
        f"instruction arrives. Got persist={last_action.get('persist')}. "
        f"Tool calls: {cm.all_tool_calls}"
    )
    assert_efficient(result3, 3)


# ---------------------------------------------------------------------------
#  6. Screen share teaching with accumulated screenshots → act references paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_screen_share_teaching_references_screenshot_paths(initialized_cm):
    """Visual tutorial should produce an act query referencing screenshot filepaths.

    When the user teaches via screen sharing across multiple messages, the
    eventual act call should reference the accumulated screenshot filepaths
    so the Actor can process and store the visual content as guidance.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Context
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="I want to show you how our inventory system works.",
        ),
    )
    _assert_no_act_triggered(result1, "Context-setting should not trigger act")

    # Phase 2: Teaching with screenshot
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="This is the main inventory dashboard",
            filepath="Screenshots/User/inventory-dashboard.jpg",
        ),
    )
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "This is the main inventory dashboard. When stock drops below "
                "the red line, you need to trigger a reorder. Make sure you "
                "remember this process — it's critical."
            ),
        ),
    )

    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "Visual teaching with 'remember this' should trigger act",
        cm=cm,
    )
    query = _get_act_query(result2)
    query_lower = query.lower()
    assert (
        "inventory" in query_lower or "reorder" in query_lower
    ), f"Expected act query to capture inventory procedure. Got: {query}"
    assert "screenshot" in query_lower or "screen" in query_lower, (
        f"Expected act query to reference screenshot/screen visual context. "
        f"Got: {query}"
    )
    assert_efficient(result2, 3)


# ---------------------------------------------------------------------------
#  7. Act query must include the specific screenshot filepath
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_act_query_includes_screenshot_filepath(initialized_cm):
    """The act query must include the exact screenshot filepath.

    The CodeActActor has no other way to locate screenshot files — the CM
    must pass the filepaths explicitly in the freeform act query text.
    This test verifies the specific filepath (not just "screenshot") appears
    in the query so the Actor can load the image from disk.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    filepath = "Screenshots/User/crm-contact-list.jpg"

    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="This is the contact list page",
            filepath=filepath,
        ),
    )
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "This is the CRM contact list. Remember this: when a client "
                "calls, you pull up their record here and check the Notes column "
                "before transferring them."
            ),
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Visual instruction should trigger act",
        cm=cm,
    )
    query = _get_act_query(result)
    assert filepath in query, (
        f"Expected the exact screenshot filepath '{filepath}' in the act "
        f"query so the CodeActActor can locate the image. Got: {query}"
    )
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  8. Cross-turn voice demo → act query includes ALL historic filepaths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_cross_turn_voice_demo_act_includes_all_screenshot_filepaths(
    initialized_cm,
):
    """Multi-turn voice screen share demo: act query must include ALL filepaths.

    Uses ``InboundUnifyMeetUtterance`` (the production event type for voice
    meetings) so that ``_claim_pending_user_screenshot`` stamps each
    screenshot onto its corresponding message. In subsequent turns, earlier
    filepaths appear as ``[Screenshots: ...]`` annotations on rendered
    messages — the LLM must scan back and include them all in the act query.

    This is the real production scenario: a multi-turn guided demo over a
    Unify Meet screen share, where the CM needs to pass ALL accumulated
    visual context to the Actor.
    """
    from unity.conversation_manager.events import (
        UnifyMeetReceived,
        UnifyMeetStarted,
        UnifyMeetEnded,
        InboundUnifyMeetUtterance,
    )

    cm = initialized_cm

    # Set up Unify Meet session
    await cm.step(UnifyMeetReceived(contact=BOSS), run_llm=False)
    await cm.step(UnifyMeetStarted(contact=BOSS), run_llm=False)
    cm.cm.user_screen_share_active = True

    filepath_1 = "Screenshots/User/payroll-dashboard.jpg"
    filepath_2 = "Screenshots/User/payroll-employee-detail.jpg"

    # Turn 1: First screenshot + context-setting utterance — no act expected.
    # _claim_pending_user_screenshot stamps this screenshot onto the message.
    # After this turn, the message renders with [Screenshots: filepath_1].
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="This is the payroll dashboard",
            filepath=filepath_1,
        ),
    )
    result1 = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=BOSS,
            content=(
                "Okay so this is the payroll dashboard — there are a couple "
                "of things I need to show you here..."
            ),
        ),
    )
    _assert_no_act_triggered(result1, "First demo step should not trigger act")

    # Turn 2: Second screenshot + completion signal — act should fire.
    # The LLM now sees: turn 1 message with [Screenshots: filepath_1] in
    # the conversation text, plus the current turn's screenshot (filepath_2)
    # in the multimodal content. The act query must include BOTH filepaths.
    cm.cm._screenshot_buffer.append(
        _make_screenshot(
            utterance="And this is the employee detail page",
            filepath=filepath_2,
        ),
    )
    result2 = await cm.step_until_wait(
        InboundUnifyMeetUtterance(
            contact=BOSS,
            content=(
                "And this is the employee detail view. To process a pay "
                "adjustment, you select the employee here and click Adjust. "
                "Make sure you remember both of these screens."
            ),
        ),
    )

    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "Completed multi-turn visual demo should trigger act",
        cm=cm,
    )
    query = _get_act_query(result2)
    assert filepath_1 in query, (
        f"Expected the FIRST screenshot filepath '{filepath_1}' from the "
        f"earlier turn in the act query — the CodeActActor needs all visual "
        f"context, not just the most recent screenshot. Got: {query}"
    )
    assert filepath_2 in query, (
        f"Expected the SECOND screenshot filepath '{filepath_2}' in the act "
        f"query. Got: {query}"
    )

    # Cleanup meet session
    await cm.step(UnifyMeetEnded(contact=BOSS), run_llm=False)
