from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@_handle_project
async def test_guard_triggers_when_enabled(monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, TranscriptManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    tm = TranscriptManager()

    # Spy on guard stop() to verify classifier-triggered early stop
    stop_called: dict[str, str | None] = {"reason": None}

    orig_stop = ReadOnlyAskGuardHandle.stop

    def _wrapped_stop(self, reason: str | None = None) -> str:  # type: ignore[no-redef]
        stop_called["reason"] = reason
        return orig_stop(self, reason)

    monkeypatch.setattr(ReadOnlyAskGuardHandle, "stop", _wrapped_stop, raising=True)

    # Mutation-intent phrasing to trigger the real classifier
    handle = await tm.ask(
        "Please change Bob Johnson's email to foo@bar.com; make the update now.",
    )
    result = await handle.result()

    # Sanity-check: some answer returned
    assert isinstance(result, str) and result.strip() != ""

    # Verify early stop due to mutation intent
    assert stop_called["reason"] is not None
    assert "mutation intent detected" in str(stop_called["reason"]).lower()
