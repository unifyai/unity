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
