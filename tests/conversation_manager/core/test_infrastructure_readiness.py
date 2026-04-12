"""
tests/conversation_manager/core/test_infrastructure_readiness.py
================================================================

Eval tests verifying the slow brain respects infrastructure readiness state
(VM booting, filesystem syncing) shown in the state snapshot.

Phase 1 tests: when vm_ready=False or file_sync_complete=False, the brain
should acknowledge the user's request and explain it needs to wait.

Phase 2 tests: once the readiness event fires (setting the flag and
triggering a brain turn), the brain should follow up on the deferred
request — calling act and informing the user it is proceeding.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    filter_events_by_type,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    ActorHandleStarted,
    FileSyncComplete,
    InitializationComplete,
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@_handle_project
async def test_brain_defers_computer_action_when_vm_not_ready(initialized_cm):
    """When vm_ready=False the brain should not call act for a computer task.

    Instead it should reply to the user acknowledging the request and
    explaining that the desktop is not yet available.
    """
    cm = initialized_cm
    assert cm.cm.vm_ready is False, "vm_ready should start False"

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Open Google Chrome on the desktop and go to google.com",
        ),
    )

    assert "act" not in cm.all_tool_calls, (
        f"Brain should NOT call act when VM is not ready, "
        f"but called: {cm.all_tool_calls}"
    )

    replies = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert (
        len(replies) >= 1
    ), "Brain should send a reply explaining the VM is not ready yet"


@pytest.mark.asyncio
@_handle_project
async def test_brain_defers_file_access_when_sync_not_complete(initialized_cm):
    """When file_sync_complete=False the brain should not call act to read files.

    Instead it should reply explaining that files from previous sessions
    are still being synced.
    """
    cm = initialized_cm
    cm.cm.vm_ready = True
    assert cm.cm.file_sync_complete is False, "file_sync_complete should start False"

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Open the quarterly_results.xlsx file from my Attachments "
                "folder and summarise the contents for me."
            ),
        ),
    )

    assert "act" not in cm.all_tool_calls, (
        f"Brain should NOT call act to read files when sync is incomplete, "
        f"but called: {cm.all_tool_calls}"
    )

    replies = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert (
        len(replies) >= 1
    ), "Brain should send a reply explaining files are still syncing"


# ---------------------------------------------------------------------------
#  Phase 2: readiness events arrive, brain follows up on deferred request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_brain_acts_on_computer_task_after_vm_ready(initialized_cm):
    """After deferring a computer task, the brain acts once VM + sync are ready.

    Phase 1: user asks for a desktop task, brain defers (vm not ready).
    Phase 2: VM becomes ready and sync completes (FileSyncComplete event),
    the brain gets a thinking turn and should now call act and inform the
    user it is proceeding.
    """
    cm = initialized_cm
    assert cm.cm.vm_ready is False

    # Phase 1: user asks, brain defers
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Open Google Chrome on the desktop and go to google.com",
        ),
    )
    assert "act" not in cm.all_tool_calls
    cm.all_tool_calls.clear()

    # Phase 2: VM ready + file sync complete (event triggers brain turn)
    cm.cm.vm_ready = True
    result = await cm.step_until_wait(FileSyncComplete())

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Brain should call act for the deferred computer task once VM is ready",
        cm=cm,
    )


@pytest.mark.asyncio
@_handle_project
async def test_brain_acts_on_file_request_after_sync_complete(initialized_cm):
    """After deferring a file request, the brain acts once sync completes.

    Phase 1: user asks about a historical attachment, brain defers (sync pending).
    Phase 2: FileSyncComplete event fires, the brain gets a thinking turn
    and should now call act to search for the file and inform the user.
    """
    cm = initialized_cm
    cm.cm.vm_ready = True
    assert cm.cm.file_sync_complete is False

    # Phase 1: user asks, brain defers
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Open the quarterly_results.xlsx file from my Attachments "
                "folder and summarise the contents for me."
            ),
        ),
    )
    assert "act" not in cm.all_tool_calls
    cm.all_tool_calls.clear()

    # Phase 2: file sync completes (event triggers brain turn)
    result = await cm.step_until_wait(FileSyncComplete())

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Brain should call act to open the file once sync is complete",
        cm=cm,
    )


# ---------------------------------------------------------------------------
#  Manager initialization readiness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_brain_defers_when_not_initialized(initialized_cm):
    """When initialized=False the brain should not call act.

    Instead it should reply to the user acknowledging the request and
    explaining that it is still setting up.
    """
    cm = initialized_cm
    cm.cm.initialized = False

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="What meetings do I have today?",
        ),
    )

    assert "act" not in cm.all_tool_calls, (
        f"Brain should NOT call act when not initialized, "
        f"but called: {cm.all_tool_calls}"
    )

    replies = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert (
        len(replies) >= 1
    ), "Brain should send a reply explaining it is still initializing"


@pytest.mark.asyncio
@_handle_project
async def test_brain_acts_after_initialization_complete(initialized_cm):
    """Post-init context reveals the user's message was misunderstood.

    Scenario: an hour ago, the user asked the assistant to compile a
    quarterly revenue report.  The assistant acknowledged and got to work.
    The container then restarted (cold start).

    Phase 1 (pre-init): the user sends "how's it going?" — without
    historical context, the brain treats it as casual chat and replies
    socially.

    Phase 2 (post-init): InitializationComplete fires.  Hydrated history
    now shows the pending report task.  The brain should realise "how's
    it going?" was about the report — and follow up with act to work on
    it (or at minimum, send a corrective message to the user).
    """
    from datetime import datetime, timezone, timedelta
    from unity.conversation_manager.cm_types import Medium

    cm = initialized_cm
    cm.cm.initialized = False

    # Phase 1: user sends an ambiguous message during init.
    # Brain has no history, interprets it as casual.
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="how's it going?",
        ),
    )
    assert "act" not in cm.all_tool_calls
    cm.all_tool_calls.clear()

    # Simulate hydration: historical messages appear in the global thread
    # as they would after init_conv_manager completes hydration.
    an_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    ci = cm.cm.contact_index
    ci.push_message(
        contact_id=1,
        sender_name="Default User",
        thread_name=Medium.UNIFY_MESSAGE,
        message_content=(
            "Can you compile the quarterly revenue report? Pull the numbers "
            "from the finance spreadsheet and summarise the key trends."
        ),
        role="user",
        timestamp=an_hour_ago,
    )
    ci.push_message(
        contact_id=1,
        sender_name="Assistant",
        thread_name=Medium.UNIFY_MESSAGE,
        message_content=(
            "On it — I'll pull the finance data and have the report ready "
            "for you shortly."
        ),
        role="assistant",
        timestamp=an_hour_ago + timedelta(minutes=1),
    )

    # Phase 2: initialization completes.  Brain now has full context and
    # should realise "how's it going?" referred to the report task.
    cm.cm.initialized = True
    result = await cm.step_until_wait(InitializationComplete())

    # The brain should take meaningful action — either call act to work
    # on the report, or at minimum send a follow-up message to the user
    # acknowledging the misunderstanding.
    acted = "act" in cm.all_tool_calls
    replies = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert acted or len(replies) >= 1, (
        f"Brain should either call act or send a follow-up message when "
        f"post-init context reveals the user's earlier message was about "
        f"a pending task.  tool_calls={cm.all_tool_calls}, "
        f"replies={len(replies)}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_brain_waits_after_init_when_nothing_deferred(initialized_cm):
    """When init completes after casual chat, the brain waits.

    Phase 1: the user sends a casual message during init.  The brain
    replies (no act needed — just a greeting).  Phase 2: initialization
    completes and the brain gets a post-init turn with full context.
    Since the conversation was already handled and nothing is pending,
    the brain should call wait without sending another message.
    """
    cm = initialized_cm
    cm.cm.initialized = False

    # Phase 1: casual exchange during init — brain responds, no act needed
    phase1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Hey! Good morning, how are you doing?",
        ),
    )
    assert "act" not in cm.all_tool_calls
    phase1_replies = filter_events_by_type(phase1.output_events, UnifyMessageSent)
    assert len(phase1_replies) >= 1, "Brain should greet the user during Phase 1"
    cm.all_tool_calls.clear()

    # Phase 2: initialization completes — brain has full context but
    # the conversation was already handled; nothing new to act on.
    cm.cm.initialized = True
    result = await cm.step_until_wait(InitializationComplete())

    assert "act" not in cm.all_tool_calls, (
        f"Brain should NOT call act when there is nothing to follow up on, "
        f"but called: {cm.all_tool_calls}"
    )

    assert "wait" in cm.all_tool_calls, (
        f"Brain should call wait when there is nothing to follow up on, "
        f"but called: {cm.all_tool_calls}"
    )

    replies = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert len(replies) == 0, (
        f"Brain should NOT send an unsolicited message when there is nothing "
        f"to follow up on, but sent {len(replies)} message(s)"
    )
