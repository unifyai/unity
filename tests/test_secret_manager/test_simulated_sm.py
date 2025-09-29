from __future__ import annotations

import asyncio
import functools
import pytest

from unity.secret_manager.simulated import (
    SimulatedSecretManager,
    _SimulatedSecretHandle,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance
# ─────────────────────────────────────────────────────────────────────────────
def test_simulated_sm_docstrings_match_base():
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
async def test_start_and_ask_simulated_sm():
    sm = SimulatedSecretManager("Demo Secret Manager for unit-tests.")
    h = await sm.ask("List all secret keys.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sm_stateful_memory_serial_asks():
    sm = SimulatedSecretManager()

    h1 = await sm.ask(
        "Please propose a safe placeholder, reply with only the placeholder like ${token_name}.",
    )
    placeholder = (await h1.result()).strip()
    assert placeholder, "Placeholder should not be empty"

    h2 = await sm.ask("What placeholder did you just propose?")
    answer2 = (await h2.result()).lower()
    assert (
        placeholder.lower() in answer2
    ), "LLM should recall the placeholder it generated"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sm_stateful_update_then_ask():
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
async def test_handle_interject_sm(monkeypatch):
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
    assert "ack" in reply.lower() or "noted" in reply.lower()
    await h.result()
    assert calls["interject"] == 1


# 6.  Stop
@pytest.mark.asyncio
async def test_handle_stop_sm():
    sm = SimulatedSecretManager()
    h = await sm.ask("Generate a summary of configured secrets.")
    await asyncio.sleep(0.05)
    h.stop()
    with pytest.raises(asyncio.CancelledError):
        await h.result()
    assert h.done(), "Handle should report done after stop()"


# 7.  Clarification handshake
@pytest.mark.asyncio
async def test_handle_requests_clarification_sm():
    sm = SimulatedSecretManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await sm.ask(
        "Show me the placeholder for the database password. If ambiguous, request clarification.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        _requests_clarification=True,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=60)
    assert "clarify" in question.lower()
    await down_q.put("I mean the one named db_password.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "${db_password}" in answer


# 8.  Pause → Resume round-trip
@pytest.mark.asyncio
async def test_handle_pause_and_resume_sm(monkeypatch):
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

    pause_msg = handle.pause()
    assert "pause" in pause_msg.lower()

    tools_after_pause = handle.valid_tools
    assert "resume" in tools_after_pause and "pause" not in tools_after_pause

    res_task = asyncio.create_task(handle.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()

    resume_msg = handle.resume()
    assert "resume" in resume_msg.lower() or "running" in resume_msg.lower()

    tools_after_resume = handle.valid_tools
    assert "pause" in tools_after_resume and "resume" not in tools_after_resume

    answer = await asyncio.wait_for(res_task, timeout=60)
    assert isinstance(answer, str) and answer.strip()

    assert call_counts == {"pause": 1, "resume": 1}
