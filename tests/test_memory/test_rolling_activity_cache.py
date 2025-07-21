from __future__ import annotations

"""Unit-test for the process-wide rolling-activity cache.

Verifies that:

1. The first call to ``get_broader_context`` lazy-initialises the cache by
   calling **exactly once** into ``MemoryManager.get_broader_context``.
2. Subsequent reads return the cached value without further backend calls.
3. ``set_broader_context`` atomically updates the cached snapshot and all
   future reads reflect the new value **without** invoking
   ``MemoryManager.get_broader_context`` again.
"""

import importlib

import pytest

from tests.helpers import _handle_project


@pytest.mark.asyncio  # marker kept for uniformity – test itself is sync
@_handle_project
async def test_global_rolling_activity_cache(monkeypatch):
    # ------------------------------------------------------------------
    # 0.  Fresh module instance so globals start from a clean slate
    # ------------------------------------------------------------------
    import unity.memory_manager.broader_context as ra

    importlib.reload(ra)  # resets internal _ROLLING_ACTIVITY to None

    # ------------------------------------------------------------------
    # 1.  Provide a lightweight stub for MemoryManager to avoid importing the
    #     full dependency tree (which pulls in unrelated modules and slows the
    #     test or fails on syntax errors elsewhere).  We inject the stub into
    #     sys.modules *before* the lazy import inside rolling_activity runs.
    # ------------------------------------------------------------------

    import types, sys

    call_counter = {"count": 0}

    class _StubMemoryManager:  # noqa: D401 – simple stub
        def get_broader_context(self):  # noqa: D401 – match real API
            call_counter["count"] += 1
            return "INITIAL"

    stub_mod = types.ModuleType("unity.memory_manager.memory_manager")
    stub_mod.MemoryManager = _StubMemoryManager  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "unity.memory_manager.memory_manager", stub_mod)

    # ------------------------------------------------------------------
    # 2.  First access → invokes patched getter exactly once
    # ------------------------------------------------------------------
    first = ra.get_broader_context()
    assert first == "INITIAL"
    assert call_counter["count"] == 1, "Initialisation should call MemoryManager once"

    # ------------------------------------------------------------------
    # 3.  Second access → returns cached value without extra calls
    # ------------------------------------------------------------------
    second = ra.get_broader_context()
    assert second == "INITIAL"
    assert call_counter["count"] == 1, "Subsequent reads must use cache"

    # ------------------------------------------------------------------
    # 4.  Update cache via set_broader_context → future reads reflect change
    # ------------------------------------------------------------------
    ra.set_broader_context("UPDATED")
    updated = ra.get_broader_context()
    assert updated == "UPDATED", "Cache should reflect latest written value"
    assert call_counter["count"] == 1, "Setter must not trigger extra backend calls"
