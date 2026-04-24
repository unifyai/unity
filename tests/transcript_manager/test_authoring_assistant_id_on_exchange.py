"""Opener stamp on ``Exchange.authoring_assistant_id`` is immutable.

The body that first opens an exchange stamps
``Exchange.authoring_assistant_id`` with its own agent id. When a
Hive-mate body later writes into the same shared exchange the stamp
must stay at the opener — it records who started the conversation,
not who wrote the last message. The per-message stamp on
``Transcripts.authoring_assistant_id`` is what tracks each writer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import unify

from tests.helpers import (
    bind_body,
    bootstrap_hive_body,
    cleanup_hive_context,
    fresh_hive_id,
    pin_hive_body_base,
)
from unity.contact_manager.contact_manager import ContactManager
from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.transcript_manager import TranscriptManager


@pytest.mark.requires_real_unify
def test_exchange_authoring_id_is_opener_and_preserved_across_bodies():
    hive_id = fresh_hive_id()
    exchanges_ctx = f"Hives/{hive_id}/Exchanges"
    transcripts_ctx = f"Hives/{hive_id}/Transcripts"
    opener_agent_id = 4001
    replier_agent_id = 4002

    try:
        _, opener_cid = bootstrap_hive_body(hive_id, agent_id=opener_agent_id)
        _, replier_cid = bootstrap_hive_body(hive_id, agent_id=replier_agent_id)

        bind_body(hive_id=hive_id, agent_id=opener_agent_id)
        pin_hive_body_base(hive_id, opener_agent_id)
        cm_opener = ContactManager()
        tm_opener = TranscriptManager(contact_manager=cm_opener)

        exid, _ = tm_opener.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": opener_cid,
                "receiver_ids": [replier_cid],
                "timestamp": datetime.now(UTC),
                "content": "Opener starts the thread.",
            },
        )
        tm_opener.join_published()

        bind_body(hive_id=hive_id, agent_id=replier_agent_id)
        pin_hive_body_base(hive_id, replier_agent_id)
        cm_replier = ContactManager()
        tm_replier = TranscriptManager(contact_manager=cm_replier)

        exid_reply, _ = tm_replier.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": replier_cid,
                "receiver_ids": [opener_cid],
                "timestamp": datetime.now(UTC),
                "content": "Replier joins the same shared row.",
            },
        )
        tm_replier.join_published()

        assert exid == exid_reply, (
            f"replier must join the opener's exchange (opener={exid}, "
            f"reply={exid_reply})"
        )

        exchange_rows = unify.get_logs(
            context=exchanges_ctx,
            filter=f"exchange_id == {exid}",
            limit=1,
            from_fields=["exchange_id", "authoring_assistant_id"],
        )
        assert exchange_rows, f"no Exchange row found for exchange_id={exid}"
        stamped = exchange_rows[0].entries.get("authoring_assistant_id")
        assert stamped == opener_agent_id, (
            f"Exchange.authoring_assistant_id must remain the opener's id "
            f"({opener_agent_id}) after a subsequent Hive-mate write; "
            f"saw {stamped}"
        )

        transcripts = unify.get_logs(
            context=transcripts_ctx,
            filter=f"exchange_id == {exid}",
            limit=10,
            from_fields=["message_id", "authoring_assistant_id", "content"],
        )
        per_message_authors = {
            row.entries.get("authoring_assistant_id") for row in transcripts
        }
        assert per_message_authors == {opener_agent_id, replier_agent_id}, (
            "Transcripts.authoring_assistant_id must record the actual writer "
            f"per message; saw {per_message_authors}"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)
