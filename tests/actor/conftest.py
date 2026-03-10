from __future__ import annotations

import functools

from typing import Any

import pytest

from unity.actor.execution.session import SessionExecutor


def _normalize_execute_function_duration(result: Any) -> Any:
    if result is None:
        return result
    if isinstance(result, dict):
        result["duration_ms"] = 0
    elif hasattr(result, "duration_ms"):
        result.duration_ms = 0
    return result


@pytest.fixture(autouse=True)
def stabilize_execute_function_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep execution result duration_ms deterministic so LLM cache keys are stable."""
    original_execute = SessionExecutor.execute

    @functools.wraps(original_execute)
    async def _patched_execute(self, *args, **kwargs):
        result = await original_execute(self, *args, **kwargs)
        return _normalize_execute_function_duration(result)

    monkeypatch.setattr(SessionExecutor, "execute", _patched_execute, raising=True)
