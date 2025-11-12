import asyncio
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_transcriptmanager_ask():
    """
    Verify a flat, in‑flight TranscriptManager.ask loop reports a minimal structure.
    """
    tm = TranscriptManager()

    h = await tm.ask("Show me the most recent message mentioning budgeting or banking.")

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
            "tool": "TranscriptManager.ask",
            "children": [],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=60)  # type: ignore[attr-defined]
        except Exception:
            pass
