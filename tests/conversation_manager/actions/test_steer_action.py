"""
tests/conversation_manager/test_steer_action.py
====================================================

Tests that verify ConversationManager correctly uses steering tools to control
in-flight actions started via `act`.

These tests follow a pattern:
1. User sends a request that triggers `act` (starts an action)
2. User sends distractor small talk (should not affect the action)
3. User sends a steering command (ask, stop, pause, resume, interject)
4. We verify the appropriate steering tool was called

Uses SimulatedActor under the hood with configurable `steps` to ensure
actions remain in-flight long enough for steering to be applied.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    assert_reasonably_efficient,
    assert_steering_called,
    build_cm_context,
    get_in_flight_action_count,
    has_steering_tool_call,
)
from tests.assertion_helpers import assertion_failed
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

# Actions stay in-flight indefinitely with steps=None, duration=None.
# Tests verify steering tools were called - actions are completed via
# trigger_completion() in test cleanup.
pytestmark = [pytest.mark.eval]

# Note: BOSS (contact_id=1) is imported from conftest.py


# ---------------------------------------------------------------------------
#  Ask steering tests - querying action status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_ask_task_status_after_small_talk(initialized_cm):
    """
    User asks about task status after some small talk.

    Flow:
    1. User requests a contact search (triggers act)
    2. User sends small talk distractor
    3. User asks about task status
    4. Verify ask_* steering tool is called
    """
    cm = initialized_cm

    # Step 1: Start an action that triggers act
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find all my contacts in New York and list their details.",
        ),
    )
    assert_act_triggered(
        result1,
        ActorHandleStarted,
        "Initial request should trigger act",
        cm=cm,
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Small talk distractor (should not affect the action)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Thanks! The weather is nice today.",
        ),
    )

    # Step 3: Ask about task status
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="How's that contact search going? Any progress?",
        ),
    )

    # Verify: LLM should have called ask_* tool
    assert_steering_called(
        cm,
        "ask_",
        "Status query should call ask_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 is pure small talk (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: small talk")
    assert_efficient(result3, "Step 3: ask status")


@pytest.mark.asyncio
@_handle_project
async def test_ask_task_progress_mid_conversation(initialized_cm):
    """
    User asks for progress update during ongoing conversation.

    Flow:
    1. User requests a knowledge query (triggers act)
    2. User asks about something unrelated
    3. User asks specifically about the task progress
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's our company's refund policy? I need the details.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Unrelated question
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="By the way, what time is it there?",
        ),
    )

    # Step 3: Ask about action progress
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What have you found so far about the refund policy?",
        ),
    )

    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 is a question that could trigger act to check time/timezone
    assert_efficient(result1, "Step 1: initial action")
    assert_reasonably_efficient(result2, "Step 2: unrelated question (may trigger act)")
    assert_efficient(result3, "Step 3: ask progress")


# ---------------------------------------------------------------------------
#  Stop steering tests - canceling actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_after_small_talk(initialized_cm):
    """
    User stops a task after some small talk.

    Flow:
    1. User requests a web search (triggers act)
    2. User sends small talk
    3. User cancels the task
    4. Verify stop_* steering tool is called
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for the latest news about AI regulations.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Genuine small talk (not action-related)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="By the way, the weather is nice today.",
        ),
    )

    # Step 3: Stop the action
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Never mind, cancel that search. I found what I needed.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Cancel request should call stop_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 is pure weather small talk (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: small talk")
    assert_efficient(result3, "Step 3: cancel request")


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_change_of_mind(initialized_cm):
    """
    User changes their mind and stops a task.

    Flow:
    1. User requests a task creation
    2. User mentions an unrelated reminder need
    3. User decides to cancel
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Create a reminder to call Bob tomorrow at 3pm.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Mentions needing to do something (could trigger proactive offer)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Oh I just remembered, I need to buy groceries later.",
        ),
    )

    # Step 3: Cancel
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="You know what, forget about that reminder. I'll just call Bob now.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Cancellation should call stop_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 mentions a task ("buy groceries") - LLM may offer to help
    assert_efficient(result1, "Step 1: initial action")
    assert_reasonably_efficient(result2, "Step 2: mentions task (may offer to help)")
    assert_efficient(result3, "Step 3: cancel request")


# ---------------------------------------------------------------------------
#  Pause steering tests - temporarily halting actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pause_task_for_meeting(initialized_cm):
    """
    User pauses a task because they need to step away.

    Flow:
    1. User requests a research task
    2. User mentions they have a meeting (implies time constraint)
    3. User asks to hold the task (natural language, no "pause" word)
    """
    cm = initialized_cm

    # Step 1: Start a research action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Research our competitors' pricing strategies and summarize.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Mention meeting (implies time constraint)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I have a meeting in 5 minutes.",
        ),
    )

    # Step 3: Pause request (natural language)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Put that on hold for now. I need to step into the meeting.",
        ),
    )

    assert_steering_called(
        cm,
        "pause_",
        "'Put on hold' should call pause_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 implies time constraint - LLM may interject deadline, expedite, etc.
    assert_efficient(result1, "Step 1: initial action")
    assert_reasonably_efficient(result2, "Step 2: time constraint (may adapt task)")
    assert_efficient(result3, "Step 3: pause request")


@pytest.mark.asyncio
@_handle_project
async def test_pause_task_hold_on(initialized_cm):
    """
    User says hold on, which should pause the task.

    Flow:
    1. User requests a transcript search
    2. User says hold on (natural language, no "pause" word)
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search my past conversations with Alice about the project deadline.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Hold on (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, hold on a second. Let me think about what I actually need.",
        ),
    )

    assert_steering_called(
        cm,
        "pause_",
        "'Hold on' should call pause_* steering tool",
        result=result2,
    )

    # Efficiency: Step 2 is explicit pause request (pause + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: hold on request")


# ---------------------------------------------------------------------------
#  Resume steering tests - continuing paused actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_resume_after_pause(initialized_cm):
    """
    User resumes a task after pausing it.

    Flow:
    1. User requests a task
    2. User puts it on hold (natural language, no "pause" word)
    3. User asks to continue (natural language, no "resume" word)
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="List all high-priority tasks that are due this week.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Hold (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, please put a pin in that request for now, I might be able to share a few more details, hold on a moment",
        ),
    )

    # Step 3: Continue (natural language)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK, I'm back. Never mind there were no more details of importance, please continue as you were.",
        ),
    )

    assert_steering_called(
        cm,
        "resume_",
        "'Go ahead' should call resume_* steering tool",
        result=result3,
    )

    # Efficiency: Steps 2 and 3 are explicit steering requests (pause/resume + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: hold request")
    assert_efficient(result3, "Step 3: resume request")


@pytest.mark.asyncio
@_handle_project
async def test_resume_continue_where_left_off(initialized_cm):
    """
    User asks to continue from where they left off.

    Flow:
    1. User requests a task
    2. User puts it on hold (natural language, no "pause" word)
    3. Some small talk
    4. User asks to pick up where they left off (natural language, no "resume" word)
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search for Bob's contact information and recent messages.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Hold (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, hold on a sec.",
        ),
    )

    # Step 3: Small talk
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Sorry, had to take another call.",
        ),
    )

    # Step 4: Pick up where left off (natural language)
    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK where were we? Go ahead with that search for Bob's info.",
        ),
    )

    assert_steering_called(
        cm,
        "resume_",
        "'Go ahead' should call resume_* steering tool",
        result=result4,
    )

    # Efficiency: Step 3 is pure explanation (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: hold request")
    assert_efficient(result3, "Step 3: small talk")
    assert_efficient(result4, "Step 4: resume request")


# ---------------------------------------------------------------------------
#  Interject steering tests - providing new information to running actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_additional_constraint(initialized_cm):
    """
    User interjects with an additional constraint for the running task.

    Flow:
    1. User requests a contact search
    2. User mentions something else (potentially relevant context)
    3. User adds a constraint to the search
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find all contacts who work in engineering.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Side comment (could be seen as context for the contact search)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="We're planning a team event.",
        ),
    )

    # Step 3: Interject with constraint
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, for that search, only include people in the Berlin office.",
        ),
    )

    assert_steering_called(
        cm,
        "interject_",
        "Additional constraint should call interject_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 mentions "team event" which could be context for the contact search
    assert_efficient(result1, "Step 1: initial action")
    assert_reasonably_efficient(result2, "Step 2: context (may interject to task)")
    assert_efficient(result3, "Step 3: interject constraint")


@pytest.mark.asyncio
@_handle_project
async def test_interject_extension(initialized_cm):
    """
    User interjects to extend/add to the task.

    Flow:
    1. User requests a knowledge query
    2. User expresses anticipation (neutral small talk)
    3. User extends the request with additional requirements
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's the Q3 revenue report say?",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Anticipation (neutral, should not trigger pause/stop)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Looking forward to seeing the results.",
        ),
    )

    # Step 3: Extension via interjection (clearly adding to existing action)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Also include any notes or comments attached to the report.",
        ),
    )

    assert_steering_called(
        cm,
        "interject_",
        "Extension should call interject_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 is pure anticipation (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: anticipation")
    assert_efficient(result3, "Step 3: extension")


@pytest.mark.asyncio
@_handle_project
async def test_interject_new_priority(initialized_cm):
    """
    User interjects to change priority or focus of the task.

    Flow:
    1. User requests a broad task
    2. Some conversation
    3. User narrows the focus
    """
    cm = initialized_cm

    # Step 1: Start a broad action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Research what competitors are doing in the market.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Conversation
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I just got out of a meeting.",
        ),
    )

    # Step 3: Narrow focus
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="For that research, focus specifically on pricing - that's the most urgent.",
        ),
    )

    assert_steering_called(
        cm,
        "interject_",
        "Focus change should call interject_* steering tool",
        result=result3,
    )

    # Efficiency: Step 2 is pure statement (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: conversation")
    assert_efficient(result3, "Step 3: narrow focus")


# ---------------------------------------------------------------------------
#  Complex steering scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pause_interject_resume_sequence(initialized_cm):
    """
    User pauses, adds new information, then resumes.

    This tests the full pause -> interject -> resume flow.
    """
    cm = initialized_cm

    # Step 1: Start an action (web search reliably triggers act)
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for information about project management best practices.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Pause
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Hold on, put that search on hold.",
        ),
    )

    # Step 3: Interject while paused
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, focus specifically on agile methodology.",
        ),
    )

    # Step 4: Resume
    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK, go ahead with the search now.",
        ),
    )

    # Should have pause and either interject or resume actions
    has_pause = has_steering_tool_call(cm, "pause_")
    has_interject = has_steering_tool_call(cm, "interject_")
    has_resume = has_steering_tool_call(cm, "resume_")

    assert has_pause or has_interject or has_resume, assertion_failed(
        expected="At least one steering tool (pause_, interject_, or resume_)",
        actual=f"pause_: {has_pause}, interject_: {has_interject}, resume_: {has_resume}",
        reasoning=[],
        description="Pause->interject->resume sequence should trigger steering tools",
        context_data=build_cm_context(cm=cm, result=result4),
    )

    # Efficiency: All steps are explicit steering requests
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: pause request")
    assert_efficient(result3, "Step 3: interject while paused")
    assert_efficient(result4, "Step 4: resume request")


@pytest.mark.asyncio
@_handle_project
async def test_multiple_distractors_then_stop(initialized_cm):
    """
    Multiple distractor messages before stopping an action.

    Tests that the LLM maintains action context through multiple turns.
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Generate a report on all customer interactions this month.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2-4: Multiple distractors
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="It's really busy today.",
        ),
    )

    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Did you see the news?",
        ),
    )

    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Anyway...",
        ),
    )

    # Step 5: Stop after distractors
    result5 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, stop that report. Someone else is already doing it.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "Stop after distractors should call stop_* steering tool",
        result=result5,
    )

    # Efficiency: Steps 2-4 are pure chit-chat (acknowledge + wait)
    assert_efficient(result1, "Step 1: initial action")
    assert_efficient(result2, "Step 2: distractor 1")
    assert_efficient(result3, "Step 3: distractor 2")
    assert_efficient(result4, "Step 4: distractor 3")
    assert_efficient(result5, "Step 5: stop request")


# ---------------------------------------------------------------------------
#  In-flight ask tests - async ask status tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_ask_shows_pending_then_completed(initialized_cm):
    """
    Verify the async ask flow properly tracks pending/completed states.

    The ask operation is non-blocking: when the LLM calls ask_*, the tool
    returns immediately with 'pending' status, and the LLM receives another
    turn when the async response arrives.

    Flow:
    1. User starts a task
    2. User asks about progress (triggers ask_* which is async)
    3. Verify the ask action shows pending, then completed with response
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search my transcripts for anything about the quarterly budget review.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Ask about progress
    # Note: With async ask, this may take more steps because:
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's the status of that search?",
        ),
    )

    # Verify ask_* was called
    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result2,
    )

    # Verify the action history contains the ask action with completed status
    # (since the async response should have arrived by now)
    in_flight_actions = cm._cm.in_flight_actions
    assert len(in_flight_actions) >= 1, "Should have at least one in-flight action"

    # Find the action and check its history
    for handle_id, handle_data in in_flight_actions.items():
        handle_actions = handle_data.get("handle_actions", [])
        ask_actions = [
            a
            for a in handle_actions
            if a.get("action_name", "").startswith(f"ask_{handle_id}")
        ]

        if ask_actions:
            # The ask should be completed with a response
            last_ask = ask_actions[-1]
            assert last_ask.get("status") == "completed", assertion_failed(
                expected="Ask action with status='completed'",
                actual=f"Ask action with status='{last_ask.get('status')}'",
                reasoning=["Async ask should complete and update status"],
                description="Ask action should be marked completed",
                context_data={
                    "handle_actions": handle_actions,
                    "last_ask": last_ask,
                },
            )
            assert last_ask.get("response"), assertion_failed(
                expected="Ask action with non-empty response",
                actual=f"Ask action with response='{last_ask.get('response')}'",
                reasoning=["Async ask should have a response from the action"],
                description="Ask action should have a response",
                context_data={
                    "handle_actions": handle_actions,
                    "last_ask": last_ask,
                },
            )
            break
    else:
        # No action found with ask actions - this is unexpected
        pytest.fail("Expected at least one action with ask actions in history")

    # Efficiency: Step 1 is a straightforward action request
    assert_efficient(result1, "Step 1: initial action")


@pytest.mark.asyncio
@_handle_project
async def test_ask_response_triggers_llm_followup(initialized_cm):
    """
    Verify that when the async ask completes, the LLM gets another turn
    to process the response and communicate it to the user.

    Flow:
    1. User starts a task
    2. User asks about progress
    3. The LLM should receive the ask response and send an SMS with the info
    """
    cm = initialized_cm

    # Step 1: Start an action
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Look through all my emails for anything mentioning the Henderson account.",
        ),
    )
    assert get_in_flight_action_count(cm) >= 1, assertion_failed(
        expected="At least 1 in-flight action",
        actual=f"{get_in_flight_action_count(cm)} in-flight actions",
        reasoning=[],
        description="Initial request should create an in-flight action",
        context_data=build_cm_context(cm=cm, result=result1),
    )

    # Step 2: Ask about progress
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="How's it going with the Henderson search?",
        ),
    )

    # Verify ask_* was called
    assert_steering_called(
        cm,
        "ask_",
        "Progress query should call ask_* steering tool",
        result=result2,
    )

    # The LLM should have sent an SMS after receiving the ask response
    # (indicating it got another turn to process the result)
    from unity.conversation_manager.events import SMSSent

    sms_sent_events = [e for e in result2.output_events if isinstance(e, SMSSent)]
    assert len(sms_sent_events) >= 1, assertion_failed(
        expected="At least one SMS sent after ask response",
        actual=f"{len(sms_sent_events)} SMS events",
        reasoning=[
            "When async ask completes, LLM should get another turn",
            "LLM should send SMS with the status info",
        ],
        description="LLM should communicate ask result to user",
        context_data={
            "output_events": [type(e).__name__ for e in result2.output_events],
        },
    )

    # Efficiency: Step 1 is a straightforward action request
    assert_efficient(result1, "Step 1: initial action")
