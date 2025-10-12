from __future__ import annotations

import asyncio
import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_semantic_search_no_failures(capfd):
    """
    Intentionally trigger many concurrent `_search_contacts` calls on a fresh
    ContactManager to surface races in vector provisioning.

    This test is expected to FAIL at present (flaky errors/races), but serves
    as a guard once we add proper locking around vector ensure.
    """

    cm = ContactManager()

    # Seed a few contacts with bios to ensure semantic path is taken
    cm._create_contact(
        first_name="Alice",
        bio="Enjoys email threads and detailed reports",
    )
    cm._create_contact(
        first_name="Bob",
        bio="Prefers short text messages, hates long emails",
    )
    cm._create_contact(
        first_name="Carol",
        bio="Commutes by train; likes concise updates over email",
    )

    # Use the same reference to force contention on the same embedding column
    references = {"bio": "short emails and texts"}

    async def _one_call():
        # Run the potentially blocking call in a thread to allow true parallelism
        return await asyncio.to_thread(cm._search_contacts, references=references, k=3)

    # Clear any prior captured output before we start
    try:
        capfd.readouterr()
    except Exception:
        pass

    # Launch two concurrent calls to provoke the race
    tasks = [asyncio.create_task(_one_call()) for _ in range(2)]

    # If any task raises, gather will surface the exception → test fails
    results = await asyncio.gather(*tasks)

    # Basic sanity: we should have results from all tasks
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)

    # Assert that no vector-provisioning errors were logged during concurrency
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + "\n" + (captured.err or "")
    # These substrings are emitted when the embedding ensure races
    assert "Failed to create derived column" not in combined_output
    assert "UniqueViolation" not in combined_output
