from __future__ import annotations

from datetime import UTC, datetime
import time

import pytest
import unify

from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project
from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _space_ids() -> tuple[int, int]:
    base = int(time.time_ns() % 1_000_000_000)
    return base, base + 1


def _message(content: str, *, exchange_id: int) -> Message:
    return Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content=content,
        exchange_id=exchange_id,
    )


def _delete_context_tree(root: str) -> None:
    try:
        children = list(unify.get_contexts(prefix=f"{root}/").keys())
    except Exception:
        children = []
    for context in sorted(children, key=len, reverse=True):
        try:
            unify.delete_context(context)
        except Exception:
            pass
    try:
        unify.delete_context(root)
    except Exception:
        pass


@_handle_project
@pytest.mark.asyncio
async def test_ask_reads_the_relevant_accessible_space_transcript() -> None:
    patch_space_id, research_space_id = _space_ids()
    target_token = "amber-coupler-5197"
    personal_decoy = "private-lotus-1189"
    research_decoy = "market-cedar-2048"

    try:
        SESSION_DETAILS.space_ids = [patch_space_id, research_space_id]
        SESSION_DETAILS.space_summaries = [
            {
                "space_id": patch_space_id,
                "name": "Patch Reliability",
                "description": (
                    "Shared workspace for field dispatch, compressor incidents, "
                    "maintenance handoffs, and team-visible repair coordination."
                ),
            },
            {
                "space_id": research_space_id,
                "name": "Market Research",
                "description": (
                    "Shared workspace for competitive research, pricing studies, "
                    "customer interviews, and analyst notes."
                ),
            },
        ]

        manager = TranscriptManager()
        manager.log_messages(
            _message(
                f"My private browser notes use personal lookup token {personal_decoy}.",
                exchange_id=51001,
            ),
            synchronous=True,
        )
        manager.log_messages(
            _message(
                (
                    "Competitive interview archive: the analyst pack uses "
                    f"research lookup token {research_decoy}."
                ),
                exchange_id=51002,
            ),
            synchronous=True,
            destination=f"space:{research_space_id}",
        )
        manager.log_messages(
            _message(
                (
                    "Compressor callback handoff: the repair rota told field "
                    f"dispatch coordinators to use bundle access token {target_token}."
                ),
                exchange_id=51003,
            ),
            synchronous=True,
            destination=f"space:{patch_space_id}",
        )

        handle = await manager.ask(
            (
                "The field reliability coordinators are reviewing the compressor "
                "callback handoff. Which access token should they use for the "
                "bundle?"
            ),
            _return_reasoning_steps=True,
        )
        answer, steps = await handle.result()
        normalized = answer.lower()

        assert target_token in normalized, assertion_failed(
            target_token,
            answer,
            steps,
            "TranscriptManager.ask should answer from the relevant shared transcript",
        )
        assert personal_decoy not in normalized
        assert research_decoy not in normalized
    finally:
        _delete_context_tree(f"Spaces/{patch_space_id}")
        _delete_context_tree(f"Spaces/{research_space_id}")
        SESSION_DETAILS.reset()
