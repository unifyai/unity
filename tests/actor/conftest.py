from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import tempfile

from typing import Any

import pytest

from unity.actor.execution.capture import StreamLike
from unity.actor.execution.session import SessionExecutor

_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


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


@pytest.fixture(autouse=True)
def _sanitize_sandbox_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip non-deterministic ``at 0x…`` addresses from sandbox output so LLM cache keys stay stable."""
    original_write = StreamLike.write

    @functools.wraps(original_write)
    def _sanitized_write(self, obj: str) -> int:
        return original_write(self, _ADDR_RE.sub(" at 0x...", obj))

    monkeypatch.setattr(StreamLike, "write", _sanitized_write)


@pytest.fixture(autouse=True)
def _isolate_local_root(request, monkeypatch):
    """Give each test its own deterministic HOME directory.

    Prevents filesystem leakage between tests — even when tests run
    concurrently in separate processes (parallel_run.sh) or sequentially
    in the same process.

    ``get_local_root()`` defaults to ``~/Unity/Local``, so changing HOME
    is sufficient to isolate all filesystem paths that flow through it
    (prompts, LocalFileSystemAdapter, venv dirs, etc.).

    The path is derived from the test's node ID via a stable hash, so
    the same test always gets the same directory.  This keeps LLM cache
    keys stable across re-runs of the same test.
    """
    suffix = hashlib.md5(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    test_home = os.path.join(tempfile.gettempdir(), f"unity_test_home_{suffix}")
    os.makedirs(test_home, exist_ok=True)

    monkeypatch.setenv("HOME", test_home)

    yield test_home

    shutil.rmtree(test_home, ignore_errors=True)
