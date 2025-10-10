from __future__ import annotations

import os
import tempfile

import pytest

from tests.helpers import _handle_project
from unity.file_manager.file_manager import FileManager
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@_handle_project
async def test_file_manager_ask_guard_triggers_when_enabled(monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, FileManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    fm = FileManager()

    # Create a temporary file and register it with the FileManager
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"Hello world")
        tmp_path = tf.name

    try:
        display_name = fm.import_file(tmp_path)

        # Spy on guard stop() to verify classifier-triggered early stop
        stop_called: dict[str, str | None] = {"reason": None}

        orig_stop = ReadOnlyAskGuardHandle.stop

        def _wrapped_stop(self, reason: str | None = None) -> str:  # type: ignore[no-redef]
            stop_called["reason"] = reason
            return orig_stop(self, reason)

        monkeypatch.setattr(ReadOnlyAskGuardHandle, "stop", _wrapped_stop, raising=True)

        # Mutation-intent phrasing to trigger the guard classifier (real LLM classification)
        handle = await fm.ask(
            display_name,
            "Please modify this file and save the changes now.",
        )
        result = await handle.result()

        # Sanity-check: some answer returned
        assert isinstance(result, str) and result.strip() != ""

        # Verify early stop due to mutation intent
        assert stop_called["reason"] is not None
        assert "mutation intent detected" in str(stop_called["reason"]).lower()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
