from __future__ import annotations

import os

import pytest

from tests.helpers import _handle_project
from unity.file_manager.managers.local import LocalFileManager as FileManager
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle


@pytest.mark.asyncio
@_handle_project
async def test_file_manager_ask_guard_triggers_when_enabled(fm_root, monkeypatch):
    """
    When UNITY_READONLY_ASK_GUARD is enabled, FileManager.ask should be guarded:
    mutation intent triggers an early stop and returns the early response.
    """

    # Ensure the env flag is on for this test only
    monkeypatch.setenv("UNITY_READONLY_ASK_GUARD", "true")

    fm = FileManager(root=fm_root)

    try:
        # Create a file under fm_root (no import needed)
        display_name = "guard_demo.txt"
        open(os.path.join(fm_root, display_name), "wb").write(b"Hello world")

        # Spy on guard stop() to verify classifier-triggered early stop
        stop_called: dict[str, str | None] = {"reason": None}

        orig_stop = ReadOnlyAskGuardHandle.stop

        def _wrapped_stop(self, reason: str | None = None) -> str:  # type: ignore[no-redef]
            stop_called["reason"] = reason
            return orig_stop(self, reason)

        monkeypatch.setattr(ReadOnlyAskGuardHandle, "stop", _wrapped_stop, raising=True)

        instruction = f"Please overwrite {display_name} with the following content: 'Hello city' and save the changes now."

        # Mutation-intent phrasing to trigger the guard classifier (real LLM classification)
        handle = await fm.ask(instruction)
        result = await handle.result()

        # Sanity-check: some answer returned
        assert isinstance(result, str) and result.strip() != ""

        # Verify early stop due to mutation intent
        assert stop_called["reason"] is not None
        assert "mutation intent detected" in str(stop_called["reason"]).lower()
    finally:
        try:
            os.unlink(os.path.join(fm_root, display_name))
        except Exception:
            pass
