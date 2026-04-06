from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.secret_manager.secret_manager import SecretManager
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_guard_triggers_when_enabled(monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, SecretManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    sm = SecretManager()

    # Spy on guard stop() to verify classifier-triggered early stop
    stop_called: dict[str, str | None] = {"reason": None}

    orig_stop = ReadOnlyAskGuardHandle.stop

    def _wrapped_stop(self, reason: str | None = None) -> str:  # type: ignore[no-redef]
        stop_called["reason"] = reason
        return orig_stop(self, reason)

    monkeypatch.setattr(ReadOnlyAskGuardHandle, "stop", _wrapped_stop, raising=True)

    # Prevent the main loop's done callback from cancelling the classifier
    # task before it completes. Without this, the main loop finishes first
    # and the done callback kills the classifier, so stop() is never called.
    # With cancellation disabled, ReadOnlyAskGuardHandle.result() awaits
    # the classifier task as its synchronization point.
    monkeypatch.setattr(
        ReadOnlyAskGuardHandle,
        "_cancel_classifier",
        lambda self: None,
    )

    # Mutation-intent phrasing to trigger the guard classifier (real LLM classification)
    handle = await sm.ask(
        "Please rotate all secrets and update the values now.",
    )
    result = await handle.result()

    # Sanity-check: some answer returned
    assert isinstance(result, str) and result.strip() != ""

    # Verify early stop due to mutation intent
    assert stop_called["reason"] is not None
    assert "mutation intent detected" in str(stop_called["reason"]).lower()
