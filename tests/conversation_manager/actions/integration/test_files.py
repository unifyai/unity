"""
File-focused ConversationManager → CodeActActor integration tests.

These validate the production path where a user provides a file path and the actor:
- reads/parses the file (PDF/CSV fixtures)
- extracts/summarizes content
- handles missing paths gracefully
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

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
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
    final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    assert "test report" in final.lower() or "fixture" in final.lower()
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
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
    final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    lower = final.lower()
    assert "alice" in lower and "bob" in lower
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
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
    final = await wait_for_actor_completion(cm, handle_id, timeout=90)

    assert (
        "not found" in final.lower()
        or "no such" in final.lower()
        or "does not exist" in final.lower()
    )
    assert_no_errors(result)
