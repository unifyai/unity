# Extended integration tests for TranscriptManager, KnowledgeManager, TaskListManager.
# These tests exercise realistic, end‑to‑end user flows – both the Ivy demo cases
# and additional hypothetical scenarios covering conflicting preferences, dynamic
# schema evolution, recurring tasks, and compound queries that span multiple
# memory managers.

from __future__ import annotations

import pytest
import random
from datetime import datetime, timezone, timedelta

from tests.helpers import _handle_project
from unity.communication.types.message import Message, Medium
from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.task_list_manager.task_list_manager import TaskListManager


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #

_NOW = datetime.now(timezone.utc)


def _stamp(delta_mins: int) -> str:
    """ISO timestamp *delta_mins* minutes from *now* (negative = past)."""
    return (_NOW + timedelta(minutes=delta_mins)).isoformat()


def _msg(
    delta_mins: int,
    content: str,
    *,
    medium: Medium = Medium.WHATSAPP_MSG,
    sender: int = 1,
    receiver: int = 0,
    ex_id: int = 0,
) -> Message:
    """Convenience factory that always matches the Message schema."""
    return Message(
        medium=medium,
        sender_id=sender,
        receiver_id=receiver,
        timestamp=_stamp(delta_mins),
        content=content,
        exchange_id=ex_id,
    )


# --------------------------------------------------------------------------- #
#  Transcript-centric scenarios                                               #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.eval
def test_transcript_preference_recall() -> None:
    tm = TranscriptManager()
    tm.start()

    # Exchange 0: the user sets a preference for WhatsApp
    tm.log_messages(
        [
            _msg(-2880, "Hi, I prefer WhatsApp over email.", ex_id=0),
            _msg(
                -2879,
                "Acknowledged – will use WhatsApp.",
                sender=0,
                receiver=1,
                medium=Medium.WHATSAPP_MSG,
                ex_id=0,
            ),
        ],
    )
    tm.summarize(exchange_ids=0)  # generate & persist summary

    answer = tm.ask("What channel should I use to contact the user?")
    assert "whatsapp" in answer.lower()


@_handle_project
@pytest.mark.eval
def test_transcript_preference_override() -> None:
    tm = TranscriptManager()
    tm.start()

    tm.log_messages([_msg(-1440, "Please message me on WhatsApp.", ex_id=1)])
    tm.summarize(exchange_ids=1)

    # 24 h later the user changes their mind
    tm.log_messages(
        [
            _msg(
                -5,
                "Actually, email is better for me now.",
                medium=Medium.EMAIL,
                ex_id=2,
            )
        ]
    )

    answer = tm.ask("How should I contact the user today?")
    assert "email" in answer.lower()


@_handle_project
@pytest.mark.eval
def test_cross_exchange_semantic_search() -> None:
    tm = TranscriptManager()
    tm.start()

    # Three different exchanges, media and contacts
    tm.log_messages(
        [
            _msg(
                -60,
                "Billie, the SC-123 shipment has left the warehouse.",
                medium=Medium.EMAIL,
                sender=2,
                receiver=0,
                ex_id=10,
            ),
            _msg(
                -50,
                "Great, thanks!",
                medium=Medium.EMAIL,
                sender=0,
                receiver=2,
                ex_id=10,
            ),
            _msg(
                -30,
                "Heads-up: tracking number is ZX-98765.",
                medium=Medium.SMS_MESSAGE,
                sender=2,
                receiver=0,
                ex_id=11,
            ),
            _msg(
                -5,
                "Remember, ZX-98765 arrives tomorrow.",
                medium=Medium.WHATSAPP_MSG,
                sender=2,
                receiver=0,
                ex_id=12,
            ),
        ],
    )

    answer = tm.ask("What tracking number did the supplier give me?")
    assert "zx-98765" in answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeManager – simple natural-language round-trip
# ─────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.eval
def test_knowledge_roundtrip():
    """
    Store a fact in plain English and make sure we can
    query it back via natural-language retrieval.
    """
    km = KnowledgeManager()
    km.start()
    km.store(
        "My flight BA117 from Karachi (KHI) to New-York (JFK) departs on 20-May-2025.",
    )

    answer = km.retrieve(
        "What's the reference number for my KHI → JFK flight on 20 May 2025?",
    )
    assert "ba117" in answer.lower()
