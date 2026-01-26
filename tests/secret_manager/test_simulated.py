from __future__ import annotations

import asyncio
import re
import functools
import pytest

from unity.secret_manager.simulated import (
    SimulatedSecretManager,
    _SimulatedSecretHandle,
)

from tests.helpers import (
    _handle_project,
    _ack_ok,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance
# ─────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """Public methods in SimulatedSecretManager should copy BaseSecretManager doc-strings."""
    from unity.secret_manager.base import BaseSecretManager
    from unity.secret_manager.simulated import SimulatedSecretManager

    assert (
        BaseSecretManager.ask.__doc__.strip()
        in SimulatedSecretManager.ask.__doc__.strip()
    )
    assert (
        BaseSecretManager.update.__doc__.strip()
        in SimulatedSecretManager.update.__doc__.strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask():
    sm = SimulatedSecretManager("Demo Secret Manager for unit-tests.")
    h = await sm.ask("List all secret keys.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
    sm = SimulatedSecretManager()

    h1 = await sm.ask(
        "Please propose a safe placeholder, output only the placeholder name like ${token_name}.",
    )
    placeholder = (await h1.result()).strip()
    assert placeholder, "Placeholder should not be empty"
    # Extract a single ${key} token from the first answer
    m = re.search(r"\$\{[^}]+\}", placeholder)
    assert m, "Response should contain a ${name} placeholder token"
    token = m.group(0).lower()

    h2 = await sm.ask("What placeholder did you just propose?")
    answer2 = (await h2.result()).lower()
    assert token in answer2, "LLM should recall the placeholder token it generated"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_update_then_ask():
    sm = SimulatedSecretManager()

    upd = await sm.update(
        "Create a secret named api_key with a value (do not reveal it).",
    )
    await upd.result()

    hq = await sm.ask("Confirm that ${api_key} exists and is stored.")
    ans = (await hq.result()).lower()
    assert (
        "${api_key}" in ans
    ), "Secret created via update should be referenced by placeholder"


# ─────────────────────────────────────────────────────────────────────────────
# Steerable handle tests
# ─────────────────────────────────────────────────────────────────────────────


# 5.  Interject
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedSecretHandle.interject

    @functools.wraps(orig)
    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(_SimulatedSecretHandle, "interject", wrapped, raising=True)

    sm = SimulatedSecretManager()
    h = await sm.ask("Show me recent secret activity.")
    await asyncio.sleep(0.05)
    reply = h.interject("Also mention any new keys created today.")
    assert _ack_ok(reply)
    await h.result()
    assert calls["interject"] == 1


# 6.  Stop
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop():
    sm = SimulatedSecretManager()
    h = await sm.ask("Generate a summary of configured secrets.")
    await asyncio.sleep(0.05)
    h.stop()
    with pytest.raises(asyncio.CancelledError):
        await h.result()
    assert h.done(), "Handle should report done after stop()"


# 7.  Clarification handshake
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    sm = SimulatedSecretManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await sm.ask(
        "Show me the placeholder for the database password. If ambiguous, request clarification.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()
    await down_q.put("I mean the one named db_password.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "${db_password}" in answer


# 8.  Pause → Resume round-trip
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    call_counts = {"pause": 0, "resume": 0}

    original_pause = _SimulatedSecretHandle.pause

    @functools.wraps(original_pause)
    def _patched_pause(self):  # type: ignore[override]
        call_counts["pause"] += 1
        return original_pause(self)

    monkeypatch.setattr(_SimulatedSecretHandle, "pause", _patched_pause, raising=True)

    original_resume = _SimulatedSecretHandle.resume

    @functools.wraps(original_resume)
    def _patched_resume(self):  # type: ignore[override]
        call_counts["resume"] += 1
        return original_resume(self)

    monkeypatch.setattr(_SimulatedSecretHandle, "resume", _patched_resume, raising=True)

    sm = SimulatedSecretManager()
    handle = await sm.ask("Summarize current secret inventory.")

    # pause available before pausing
    tools_initial = handle.valid_tools
    assert "pause" in tools_initial and "resume" not in tools_initial

    pause_msg = await handle.pause()
    assert "pause" in pause_msg.lower()

    tools_after_pause = handle.valid_tools
    assert "resume" in tools_after_pause and "pause" not in tools_after_pause

    res_task = asyncio.create_task(handle.result())
    await _assert_blocks_while_paused(res_task)

    resume_msg = await handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    tools_after_resume = handle.valid_tools
    assert "pause" in tools_after_resume and "resume" not in tools_after_resume

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()

    assert call_counts == {"pause": 1, "resume": 1}


# 9.  Nested ask on handle
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask_nested():
    """
    The internal handle returned by SimulatedSecretManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    sm = SimulatedSecretManager()

    # Start an initial ask to obtain the live handle
    handle = await sm.ask("Summarize current secret placeholders used in the system.")

    # Add extra context to ensure nested prompt includes it
    handle.interject("Focus on API and database related placeholders.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("Which placeholders are most critical to rotate?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip()

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip()


# 10.  Clear – reset and remain usable
@pytest.mark.asyncio
@_handle_project
async def test_clear():
    """
    SimulatedSecretManager.clear should reset the manager and remain usable afterwards.
    """
    sm = SimulatedSecretManager()
    # Do an update/ask to create some prior state in the stateful LLM
    h_upd = await sm.update("Create a temporary secret named temp_token.")
    await h_upd.result()

    # Clear should not raise and should be quick (no LLM roundtrip requirement)
    sm.clear()

    # Post-clear, an ask should still work
    h_q = await sm.ask("List any secret placeholders referenced in the system.")
    answer = await h_q.result()
    assert isinstance(answer, str) and answer.strip()


# 11.  Placeholder conversion helpers
@pytest.mark.asyncio
@_handle_project
async def test_from_and_to_placeholder_roundtrip():
    """
    Verify the public placeholder conversion helpers round-trip as expected.
    """
    sm = SimulatedSecretManager()
    original = "Use ${api_key} for requests and ${db_password} for DB."

    # Convert to opaque value tokens
    to_values = await sm.from_placeholder(original)
    assert "<value:api_key>" in to_values
    assert "<value:db_password>" in to_values

    # Convert back to placeholders
    back_to_placeholders = await sm.to_placeholder(to_values)
    assert "${api_key}" in back_to_placeholders
    assert "${db_password}" in back_to_placeholders


# 12.  Stop while paused should finish immediately
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    sm = SimulatedSecretManager()
    h = await sm.ask("Produce a long secrets audit.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    h.stop("cancelled by user")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert h.done()


# 13.  Stop while waiting for clarification should finish immediately
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    sm = SimulatedSecretManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await sm.ask(
        "Confirm placeholder names.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    h.stop("no longer needed")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert h.done()
