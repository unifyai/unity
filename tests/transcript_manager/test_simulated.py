from __future__ import annotations

import asyncio
import pytest
import functools

from unity.transcript_manager.simulated import (
    SimulatedTranscriptManager,
    _SimulatedTranscriptHandle,
)
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import Medium
from datetime import datetime, timezone

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
    _unique_token,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedTranscriptManager should copy the real
    BaseTranscriptManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.transcript_manager.base import BaseTranscriptManager
    from unity.transcript_manager.simulated import SimulatedTranscriptManager

    assert (
        BaseTranscriptManager.ask.__doc__.strip()
        in SimulatedTranscriptManager.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask():
    tm = SimulatedTranscriptManager("Demo transcript DB.")
    handle = await tm.ask("Show me my unread emails.")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    because the manager's LLM is stateful.

    To reduce brittleness from formatting/phrasing, we seed the first turn with
    a unique token and then require that the second turn recalls that exact
    token somewhere in its response.
    """
    tm = SimulatedTranscriptManager()

    # 1) Seed a unique token inside a realistic transcript request
    token = _unique_token("TICKET")
    handle1 = await tm.ask(
        "Please produce exactly one realistic transcript message. "
        f"Ensure the message content includes this exact ticket number verbatim: {token}",
    )
    first_answer = (await handle1.result()).strip()
    assert first_answer, "First answer should not be empty"

    # 2) Ask the LLM to recall the previously mentioned token
    handle2 = await tm.ask(
        "What ticket number did you mention earlier? Quote it verbatim in your answer.",
    )
    answer2 = await handle2.result()
    assert (
        isinstance(answer2, str) and answer2.strip()
    ), "Second answer should be non-empty"

    # The second answer should mention the exact token (substring check is robust to formatting)
    assert token in answer2, "LLM should recall the previously mentioned token"


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                     #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 4.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_interject(monkeypatch):
    counts = {"interject": 0}
    original_interject = _SimulatedTranscriptHandle.interject

    @functools.wraps(original_interject)
    async def wrapped(self, message: str, **kwargs) -> str:  # type: ignore[override]
        counts["interject"] += 1
        return await original_interject(self, message, **kwargs)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "interject",
        wrapped,
        raising=True,
    )

    tm = SimulatedTranscriptManager()
    handle = await tm.ask("Summarise yesterday's Slack exchange with Bob.")
    # interject while running
    await asyncio.sleep(0.05)
    await handle.interject("Also include any emojis Bob used.")

    await handle.result()
    assert counts["interject"] == 1, ".interject should be called exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_stop():
    tm = SimulatedTranscriptManager()
    handle = await tm.ask("Produce a full export of all messages.")
    await asyncio.sleep(0.05)
    await handle.stop()
    await handle.result()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_requests_clarification():
    tm = SimulatedTranscriptManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await tm.ask(
        "Find important messages.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )

    # Must ask for clarification first
    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    # Provide clarification
    await down_q.put("Focus on project Alpha deadlines.")
    answer = await handle.result()

    assert isinstance(answer, str) and answer.strip(), "Answer should not be empty"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    """
    Ensure a `_SimulatedTranscriptHandle` can be paused and resumed.
    """
    counts = {"pause": 0, "resume": 0}

    # --- patch pause -------------------------------------------------------
    orig_pause = _SimulatedTranscriptHandle.pause

    @functools.wraps(orig_pause)
    def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return orig_pause(self)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    # --- patch resume ------------------------------------------------------
    orig_resume = _SimulatedTranscriptHandle.resume

    @functools.wraps(orig_resume)
    def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(
        _SimulatedTranscriptHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    tm = SimulatedTranscriptManager()
    handle = await tm.ask("List unread DMs.")

    # Pause the handle.
    pause_reply = await handle.pause()
    assert "pause" in pause_reply.lower()

    # Start result() – it should block while paused.
    res_task = await _assert_blocks_while_paused(handle.result())

    # Resume and ensure execution proceeds.
    resume_reply = await handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    answer = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()

    # Each steering method should have been called exactly once.
    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 8.  Nested ask on handle                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_handle_ask():
    """
    The internal handle returned by SimulatedTranscriptManager.ask exposes a
    dynamic ask() method that should produce a nested handle whose result can
    be awaited independently of the parent.
    """
    tm = SimulatedTranscriptManager()

    # Start an initial ask to obtain the live handle
    handle = await tm.ask("Summarize all unread messages this week.")

    # Add extra context to ensure nested prompt includes it
    await handle.interject("Focus on European enterprise accounts.")

    # Invoke the dynamic ask on the running handle
    nested = await handle.ask("What is the key point to emphasize?")

    nested_answer = await nested.result()
    assert isinstance(nested_answer, str) and nested_answer.strip(), (
        "Nested ask() should yield a non-empty string answer",
    )
    assert any(substr in nested_answer.lower() for substr in ("europe", "eu"))

    # The original handle should still be awaitable and produce an answer
    handle_answer = await handle.result()
    assert isinstance(handle_answer, str) and handle_answer.strip(), (
        "Handle should still yield a non-empty answer after nested ask",
    )


# ────────────────────────────────────────────────────────────────────────────
# 10.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stop_while_paused():
    tm = SimulatedTranscriptManager()
    h = await tm.ask("Generate a long transcript report.")
    await h.pause()
    res_task = asyncio.create_task(h.result())
    await asyncio.sleep(0.1)
    assert not res_task.done()
    await h.stop("cancelled by user")
    out = await asyncio.wait_for(res_task, timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 11.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stop_while_waiting_clarification():
    tm = SimulatedTranscriptManager()
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    h = await tm.ask(
        "Please summarise inbox.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
        _requests_clarification=True,
    )
    q = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in q.lower()
    await h.stop("no longer needed")
    out = await asyncio.wait_for(h.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(out, str)
    assert h.done()


# ────────────────────────────────────────────────────────────────────────────
# 9.  Simulated programmatic helpers (sync)                                   #
# ────────────────────────────────────────────────────────────────────────────
@_handle_project
def test_filter_exchanges_sync():
    """
    SimulatedTranscriptManager.filter_exchanges should return a capped list of exchanges.
    """
    tm = SimulatedTranscriptManager()
    # Seed two exchanges
    m = {
        "medium": Medium.EMAIL,
        "sender_id": 1,
        "receiver_ids": [2],
        "timestamp": datetime.now(timezone.utc),
        "content": "Hello A",
    }
    tm.log_first_message_in_new_exchange(m)
    m2 = {
        "medium": Medium.SMS_MESSAGE,
        "sender_id": 3,
        "receiver_ids": [4],
        "timestamp": datetime.now(timezone.utc),
        "content": "Hello B",
    }
    tm.log_first_message_in_new_exchange(m2)

    out = tm.filter_exchanges(filter="True", limit=1)
    assert isinstance(out, dict) and "exchanges" in out
    exchanges = out["exchanges"]
    assert isinstance(exchanges, list)
    assert len(exchanges) <= 1
    if exchanges:
        ex = exchanges[0]
        assert hasattr(ex, "exchange_id"), "Exchange should expose exchange_id"


@_handle_project
def test_update_contact_id_sync():
    """
    SimulatedTranscriptManager.update_contact_id should report an updated count.
    """
    tm = SimulatedTranscriptManager()
    # Create an exchange with a message referencing contact id 10
    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": Medium.EMAIL,
            "sender_id": 10,
            "receiver_ids": [20, 30],
            "timestamp": datetime.now(timezone.utc),
            "content": "Initial",
        },
        exchange_initial_metadata={"topic": "alpha"},
    )
    # Add another message in the same exchange also referencing id 10
    created = tm.log_messages(
        {
            "medium": Medium.EMAIL,
            "sender_id": 5,
            "receiver_ids": [10],
            "timestamp": datetime.now(timezone.utc),
            "content": "Follow-up",
            "exchange_id": exid,
        },
    )
    assert isinstance(created, list) and isinstance(created[0], Message)

    out = tm.update_contact_id(original_contact_id=10, new_contact_id=11)
    assert isinstance(out, dict) and "details" in out
    assert out["details"]["old_contact_id"] == 10
    assert out["details"]["new_contact_id"] == 11
    assert out["details"]["updated_messages"] >= 1


@_handle_project
def test_clear_sync():
    """
    SimulatedTranscriptManager.clear should reset the manager and remain usable.
    """
    tm = SimulatedTranscriptManager()
    # Create prior state
    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": Medium.EMAIL,
            "sender_id": 1,
            "receiver_ids": [2],
            "timestamp": datetime.now(timezone.utc),
            "content": "Hello",
        },
    )
    tm.log_messages(
        {
            "medium": Medium.EMAIL,
            "sender_id": 2,
            "receiver_ids": [1],
            "timestamp": datetime.now(timezone.utc),
            "content": "Reply",
            "exchange_id": exid,
        },
    )
    # Clear should be quick and not raise
    tm.clear()
    # Post-clear, synchronous helper still works
    exid2, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": Medium.SMS_MESSAGE,
            "sender_id": 3,
            "receiver_ids": [4],
            "timestamp": datetime.now(timezone.utc),
            "content": "New thread",
        },
    )
    assert isinstance(exid2, int) and exid2 >= 1


@_handle_project
def test_build_plain_transcript():
    """
    build_plain_transcript should render 'Sender: content' given simple dict inputs.
    """
    lines = SimulatedTranscriptManager.build_plain_transcript(
        [
            {"sender": "Alice Example", "content": "Hi Bob"},
            {"sender": "Bob Example", "content": "Hi Alice"},
        ],
    )
    assert (
        isinstance(lines, str)
        and "Alice Example: Hi Bob" in lines
        and "Bob Example: Hi Alice" in lines
    )


@_handle_project
def test_log_sync_and_metadata():
    """
    Validate log_messages returns Message objects and exchange metadata APIs round-trip.
    """
    tm = SimulatedTranscriptManager()
    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": Medium.EMAIL,
            "sender_id": 100,
            "receiver_ids": [200],
            "timestamp": datetime.now(timezone.utc),
            "content": "Kickoff",
        },
        exchange_initial_metadata={"project": "Phoenix"},
    )
    # Log another message into the same exchange
    created = tm.log_messages(
        {
            "medium": Medium.EMAIL,
            "sender_id": 200,
            "receiver_ids": [100],
            "timestamp": datetime.now(timezone.utc),
            "content": "Acknowledged",
            "exchange_id": exid,
        },
    )
    assert isinstance(created, list) and len(created) == 1
    msg = created[0]
    assert isinstance(msg, Message)
    assert msg.exchange_id == exid

    # Metadata get/update roundtrip
    meta = tm.get_exchange_metadata(exid)
    assert hasattr(meta, "exchange_id") and meta.exchange_id == exid

    updated = tm.update_exchange_metadata(exid, {"stage": "review"})
    assert hasattr(updated, "metadata") and updated.metadata.get("stage") == "review"

    # Should not raise
    tm.join_published()


@_handle_project
def test_simulated_transcript_manager_reduce_shapes():
    tm = SimulatedTranscriptManager()

    scalar = tm.reduce(metric="sum", keys="message_id")
    assert isinstance(scalar, (int, float))

    multi = tm.reduce(metric="max", keys=["message_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"message_id"}

    grouped = tm.reduce(metric="sum", keys="message_id", group_by="medium")
    assert isinstance(grouped, dict)
