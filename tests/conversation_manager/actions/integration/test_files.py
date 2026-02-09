"""
File-focused ConversationManager → CodeActActor integration tests.

These validate the production path where a user provides a file path and the actor:
- reads/parses the file (PDF/CSV fixtures)
- extracts/summarizes content
- handles missing paths gracefully
- reads downloaded attachments from Downloads/
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    wait_for_actor_completion,
)
from unity.conversation_manager.events import SMSReceived
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_summarize_pdf_by_path(initialized_cm_codeact, test_files):
    """Summarize a PDF by file path (basic FileManager read + extraction path)."""
    cm = initialized_cm_codeact
    pdf_path = test_files["test_report.pdf"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=f"Please summarize the PDF at {pdf_path} in 2 bullet points.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert "test report" in final.lower() or "fixture" in final.lower()
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_read_csv_extracts_names(initialized_cm_codeact, test_files):
    """Read a CSV by file path and extract simple structured facts (rows + names)."""
    cm = initialized_cm_codeact
    csv_path = test_files["test_data.csv"]

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                f"Read the CSV at {csv_path} and tell me how many rows it has and the names listed."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    lower = final.lower()
    assert "alice" in lower and "bob" in lower
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_file_missing_path_returns_helpful_error(initialized_cm_codeact):
    """Missing file path is handled gracefully (no crash; returns a helpful error)."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Summarize the file at /definitely/does/not/exist.pdf in one sentence.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert (
        "not found" in final.lower()
        or "no such" in final.lower()
        or "does not exist" in final.lower()
    )
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_downloaded_attachment_readable_by_actor(initialized_cm_codeact):
    """Actor can read a file that was auto-downloaded to Downloads/.

    Simulates the production attachment flow: a file lands in Downloads/ via
    save_file_to_downloads (which ingests it), then the user asks the actor
    about its contents.  The test is agnostic to *how* the actor reads the
    file (open(), primitives.files.*, etc.) — it only checks the answer.
    """
    cm = initialized_cm_codeact

    # Simulate an attachment download: save a .txt file with known content.
    fm = ManagerRegistry.get_file_manager()
    fm.save_file_to_downloads(
        "meeting_notes.txt",
        b"Project Aurora kickoff meeting\n"
        b"Attendees: Sarah Chen, Marcus Webb, Priya Patel\n"
        b"Decision: launch date set for March 15th\n"
        b"Action item: Marcus to prepare the budget forecast by Friday\n",
    )

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "I just received a file called meeting_notes.txt in Downloads. "
                "Who attended the meeting and what was the launch date?"
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    lower = final.lower()
    assert "sarah" in lower or "chen" in lower, (
        f"Expected actor to find attendee 'Sarah Chen' in the file. Got: {final}"
    )
    assert "march" in lower or "15" in lower, (
        f"Expected actor to find launch date 'March 15th' in the file. Got: {final}"
    )
