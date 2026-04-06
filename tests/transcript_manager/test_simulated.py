from __future__ import annotations

import pytest

from unity.transcript_manager.simulated import (
    SimulatedTranscriptManager,
)
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import Medium
from datetime import datetime, timezone

# Helper identical to the one used elsewhere in the test-suite
from tests.helpers import (
    _handle_project,
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
