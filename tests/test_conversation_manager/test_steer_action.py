"""
tests/test_conversation_manager/test_steer_action.py
====================================================

Tests that verify ConversationManager correctly uses steering tools to control
in-flight actions started via `act`.

These tests follow a pattern:
1. User sends a request that triggers `act` (starts a task)
2. User sends distractor small talk (should not affect the task)
3. User sends a steering command (ask, stop, pause, resume, interject)
4. We verify the appropriate steering tool was called

Uses SimulatedActor under the hood with configurable `steps` to ensure
tasks remain in-flight long enough for steering to be applied.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Convenience references to test contacts
BOSS = TEST_CONTACTS[1]  # contact_id 1 - the main user

# Maximum LLM steps for efficient tool calling
# - Ideal: 2 steps (action + acknowledge concurrent, then wait)
# - Acceptable: 3 steps (action, acknowledge, wait - or action + query + wait)
MAX_EFFICIENT_STEPS = 3


def _only(events, typ):
    """Filter events by type."""
    return [e for e in events if isinstance(e, typ)]


def _assert_efficient(result, context: str = ""):
    """Assert that the LLM completed efficiently (concurrent tool calls + wait)."""
    assert result.llm_step_count <= MAX_EFFICIENT_STEPS, (
        f"Expected efficient concurrent tool calling (<= {MAX_EFFICIENT_STEPS} steps), "
        f"but took {result.llm_step_count} steps. "
        f"LLM should call tools concurrently in one step, then wait. {context}"
    )


def _get_steering_action(result, operation_prefix):
    """
    Check if the LLM called a steering tool with the given operation prefix.

    Returns the action name if found, None otherwise.
    """
    # Check the LLM's last action for steering tool calls
    # The action name pattern is: {operation}_{short_name}__{handle_id}
    if hasattr(result, "last_action") and result.last_action:
        action = result.last_action
        if action.startswith(operation_prefix):
            return action
    return None


def _has_steering_in_handle_actions(cm, operation_prefix):
    """
    Check if any steering tool with the given operation prefix was called.

    Checks all tool calls made during the test (tracked by CMStepDriver),
    not just active tasks, since completed/stopped tasks are removed from
    active_tasks.

    Returns True if found, False otherwise.
    """
    # Check all tool calls tracked by the driver (survives task completion/stop)
    for tool_name in cm.all_tool_calls:
        if tool_name.startswith(operation_prefix):
            return True
    return False


def _get_active_task_count(cm):
    """Get the number of active tasks."""
    return len(cm.cm.active_tasks or {})


# ---------------------------------------------------------------------------
#  Ask steering tests - querying task status
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

    # Step 1: Start a task that triggers act
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find all my contacts in New York and list their details.",
        ),
    )
    actor_events = _only(result1.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, "Expected act to be called"
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Small talk distractor (should not affect the task)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Thanks! The weather is nice today.",
        ),
    )
    _assert_efficient(result2, "Step 2: small talk")

    # Step 3: Ask about task status
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="How's that contact search going? Any progress?",
        ),
    )
    _assert_efficient(result3, "Step 3: status query")

    # Verify: LLM should have called ask_* tool
    assert _has_steering_in_handle_actions(cm, "ask_"), (
        f"Expected ask_* steering tool to be called for status query. "
        f"Active tasks: {list(cm.cm.active_tasks.keys())}"
    )


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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's our company's refund policy? I need the details.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Unrelated question
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="By the way, what time is it there?",
        ),
    )
    _assert_efficient(result2, "Step 2: unrelated question")

    # Step 3: Ask about task progress
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What have you found so far about the refund policy?",
        ),
    )
    _assert_efficient(result3, "Step 3: progress query")

    assert _has_steering_in_handle_actions(
        cm,
        "ask_",
    ), "Expected ask_* steering tool for progress query"


# ---------------------------------------------------------------------------
#  Stop steering tests - canceling tasks
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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for the latest news about AI regulations.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Genuine small talk (not task-related)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="By the way, the weather is nice today.",
        ),
    )
    _assert_efficient(result2, "Step 2: small talk")

    # Step 3: Stop the task
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Never mind, cancel that search. I found what I needed.",
        ),
    )
    _assert_efficient(result3, "Step 3: cancel request")

    assert _has_steering_in_handle_actions(
        cm,
        "stop_",
    ), "Expected stop_* steering tool for cancel request"


@pytest.mark.asyncio
@_handle_project
async def test_stop_task_change_of_mind(initialized_cm):
    """
    User changes their mind and stops a task.

    Flow:
    1. User requests a task creation
    2. User asks a clarifying question
    3. User decides to cancel
    """
    cm = initialized_cm

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Create a reminder to call Bob tomorrow at 3pm.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Genuine small talk (not task-related)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Oh I just remembered, I need to buy groceries later.",
        ),
    )
    _assert_efficient(result2, "Step 2: small talk")

    # Step 3: Cancel
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="You know what, forget about that reminder. I'll just call Bob now.",
        ),
    )
    _assert_efficient(result3, "Step 3: cancel request")

    assert _has_steering_in_handle_actions(
        cm,
        "stop_",
    ), "Expected stop_* steering tool for cancellation"


# ---------------------------------------------------------------------------
#  Pause steering tests - temporarily halting tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_pause_task_for_meeting(initialized_cm):
    """
    User pauses a task because they need to step away.

    Flow:
    1. User requests a research task
    2. User mentions they have a meeting
    3. User asks to hold the task (natural language, no "pause" word)
    """
    cm = initialized_cm

    # Step 1: Start a research task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Research our competitors' pricing strategies and summarize.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Mention meeting
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I have a meeting in 5 minutes.",
        ),
    )
    _assert_efficient(result2, "Step 2: mention meeting")

    # Step 3: Pause request (natural language)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Put that on hold for now. I need to step into the meeting.",
        ),
    )
    _assert_efficient(result3, "Step 3: pause request")

    assert _has_steering_in_handle_actions(
        cm,
        "pause_",
    ), "Expected pause_* steering tool for 'put on hold' request"


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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search my past conversations with Alice about the project deadline.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Hold on (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, hold on a second. Let me think about what I actually need.",
        ),
    )
    _assert_efficient(result2, "Step 2: hold on request")

    assert _has_steering_in_handle_actions(
        cm,
        "pause_",
    ), "Expected pause_* steering tool for 'hold on' request"


# ---------------------------------------------------------------------------
#  Resume steering tests - continuing paused tasks
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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="List all high-priority tasks that are due this week.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Hold (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, hold on a moment.",
        ),
    )
    _assert_efficient(result2, "Step 2: hold request")

    # Step 3: Continue (natural language)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK, I'm back. Go ahead with that task list.",
        ),
    )
    _assert_efficient(result3, "Step 3: resume request")

    assert _has_steering_in_handle_actions(
        cm,
        "resume_",
    ), "Expected resume_* steering tool for 'go ahead' request"


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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search for Bob's contact information and recent messages.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Hold (natural language)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Wait, hold on a sec.",
        ),
    )
    _assert_efficient(result2, "Step 2: hold request")

    # Step 3: Small talk
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Sorry, had to take another call.",
        ),
    )
    _assert_efficient(result3, "Step 3: small talk")

    # Step 4: Pick up where left off (natural language)
    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK where were we? Go ahead with that search for Bob's info.",
        ),
    )
    _assert_efficient(result4, "Step 4: resume request")

    assert _has_steering_in_handle_actions(
        cm,
        "resume_",
    ), "Expected resume_* steering tool for 'go ahead' request"


# ---------------------------------------------------------------------------
#  Interject steering tests - providing new information to running tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_interject_additional_constraint(initialized_cm):
    """
    User interjects with an additional constraint for the running task.

    Flow:
    1. User requests a contact search
    2. User mentions something else
    3. User adds a constraint to the search
    """
    cm = initialized_cm

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find all contacts who work in engineering.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Side comment
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="We're planning a team event.",
        ),
    )
    _assert_efficient(result2, "Step 2: side comment")

    # Step 3: Interject with constraint
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, for that search, only include people in the Berlin office.",
        ),
    )
    _assert_efficient(result3, "Step 3: interject constraint")

    assert _has_steering_in_handle_actions(
        cm,
        "interject_",
    ), "Expected interject_* steering tool for additional constraint"


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

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's the Q3 revenue report say?",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Anticipation (neutral, should not trigger pause/stop)
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Looking forward to seeing the results.",
        ),
    )
    _assert_efficient(result2, "Step 2: anticipation")

    # Step 3: Extension via interjection (clearly adding to existing task)
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Also include any notes or comments attached to the report.",
        ),
    )
    _assert_efficient(result3, "Step 3: extension")

    assert _has_steering_in_handle_actions(
        cm,
        "interject_",
    ), "Expected interject_* steering tool for extension"


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

    # Step 1: Start a broad task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Research what competitors are doing in the market.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Conversation
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I just got out of a meeting.",
        ),
    )
    _assert_efficient(result2, "Step 2: conversation")

    # Step 3: Narrow focus
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="For that research, focus specifically on pricing - that's the most urgent.",
        ),
    )
    _assert_efficient(result3, "Step 3: narrow focus")

    assert _has_steering_in_handle_actions(
        cm,
        "interject_",
    ), "Expected interject_* steering tool for focus change"


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

    # Step 1: Start a task (web search reliably triggers act)
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Search the web for information about project management best practices.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2: Pause
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Hold on, put that search on hold.",
        ),
    )
    _assert_efficient(result2, "Step 2: pause request")

    # Step 3: Interject while paused
    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, focus specifically on agile methodology.",
        ),
    )
    _assert_efficient(result3, "Step 3: interject while paused")

    # Step 4: Resume
    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="OK, go ahead with the search now.",
        ),
    )
    _assert_efficient(result4, "Step 4: resume request")

    # Should have pause and either interject or resume actions
    has_pause = _has_steering_in_handle_actions(cm, "pause_")
    has_interject = _has_steering_in_handle_actions(cm, "interject_")
    has_resume = _has_steering_in_handle_actions(cm, "resume_")

    assert (
        has_pause or has_interject or has_resume
    ), "Expected at least one steering action in pause->interject->resume sequence"


@pytest.mark.asyncio
@_handle_project
async def test_multiple_distractors_then_stop(initialized_cm):
    """
    Multiple distractor messages before stopping a task.

    Tests that the LLM maintains task context through multiple turns.
    """
    cm = initialized_cm

    # Step 1: Start a task
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Generate a report on all customer interactions this month.",
        ),
    )
    assert _get_active_task_count(cm) >= 1, "Expected at least one active task"
    _assert_efficient(result1, "Step 1: initial task")

    # Step 2-4: Multiple distractors
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="It's really busy today.",
        ),
    )
    _assert_efficient(result2, "Step 2: distractor 1")

    result3 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Did you see the news?",
        ),
    )
    _assert_efficient(result3, "Step 3: distractor 2")

    result4 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Anyway...",
        ),
    )
    _assert_efficient(result4, "Step 4: distractor 3")

    # Step 5: Stop after distractors
    result5 = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Actually, stop that report. Someone else is already doing it.",
        ),
    )
    _assert_efficient(result5, "Step 5: stop request")

    assert _has_steering_in_handle_actions(
        cm,
        "stop_",
    ), "Expected stop_* steering tool after multiple distractors"
