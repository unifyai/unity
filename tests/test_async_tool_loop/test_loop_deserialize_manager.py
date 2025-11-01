from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.async_tool_loop import AsyncToolLoopHandle


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_contact_manager_resume():
    cm = ContactManager()

    # Start a loop and immediately snapshot (may still be in-flight)
    handle = await cm.ask("Find contact Alice")
    snap = handle.serialize()

    # Resume from snapshot and ensure we obtain a final answer
    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and len(answer) > 0


@pytest.mark.asyncio
@_handle_project
async def test_deserialize_transcript_manager_resume():
    tm = TranscriptManager()

    # Seed a minimal prompt that will exercise search tools
    handle = await tm.ask(
        "Do I have any transcripts? Reply briefly.",
    )
    snap = handle.serialize()

    resumed = AsyncToolLoopHandle.deserialize(snap)
    answer = await resumed.result()
    assert isinstance(answer, str) and len(answer) > 0
