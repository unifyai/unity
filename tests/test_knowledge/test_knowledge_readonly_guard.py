from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_manager_ask_guard_triggers_when_enabled(monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, KnowledgeManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    # Avoid initializing heavy FileManager/spaCy by injecting a lightweight stub
    class _StubFileManager:  # minimal stub – not used by ask()
        pass

    km = KnowledgeManager(file_manager=_StubFileManager())

    # Monkeypatch the guard's stop() to detect classifier-triggered early stop
    stop_called: dict[str, str | None] = {"reason": None}

    orig_stop = ReadOnlyAskGuardHandle.stop

    def _wrapped_stop(self, reason: str | None = None) -> str:  # type: ignore[no-redef]
        stop_called["reason"] = reason
        return orig_stop(self, reason)

    monkeypatch.setattr(ReadOnlyAskGuardHandle, "stop", _wrapped_stop, raising=True)

    # Mutation-intent phrasing to trigger the guard classifier (real LLM classification)
    handle = await km.ask(
        "Please change the schema for the Content table; make the update now.",
    )
    result = await handle.result()
    # We only sanity-check that some answer was returned
    assert isinstance(result, str) and result.strip() != ""

    # Verify the guard actually detected mutation intent and stopped the loop early
    assert stop_called["reason"] is not None
    assert "mutation intent detected" in str(stop_called["reason"]).lower()
