"""Dedup picks the most recent exchange when duplicates already exist.

The dedup contract tolerates rare races that produce multiple
``Exchanges`` rows on the same ``(medium, counterparty_contact_id)``
(for example two Hive-mate bodies opening simultaneously before
either row is visible to the other). When the next opener looks
up an exchange to join, it must pick the **most recent** matching
row — resolved by descending ``exchange_id`` (the auto-counting
column is monotonic inside a Hive context) — rather than picking
an older row or creating a third.
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
def test_dedup_reuses_most_recent_when_duplicates_exist():
    hive_id = fresh_hive_id()
    exchanges_ctx = f"Hives/{hive_id}/Exchanges"
    opener_agent_id = 5001
    replier_agent_id = 5002

    try:
        _, opener_cid = bootstrap_hive_body(hive_id, agent_id=opener_agent_id)
        _, replier_cid = bootstrap_hive_body(hive_id, agent_id=replier_agent_id)

        bind_body(hive_id=hive_id, agent_id=opener_agent_id)
        pin_hive_body_base(hive_id, opener_agent_id)
        cm = ContactManager()
        tm = TranscriptManager(contact_manager=cm)

        older_exid = tm._open_exchange_row(
            medium="email",
            counterparty_contact_id=replier_cid,
            initial_metadata={"label": "older-race-winner"},
        )
        newer_exid = tm._open_exchange_row(
            medium="email",
            counterparty_contact_id=replier_cid,
            initial_metadata={"label": "newer-race-winner"},
        )

        assert newer_exid > older_exid, (
            "exchange_id is auto-counting; the second open must be higher "
            f"than the first (older={older_exid}, newer={newer_exid})"
        )

        resolved_exid, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": opener_cid,
                "receiver_ids": [replier_cid],
                "timestamp": datetime.now(UTC),
                "content": "Dedup should join the most recent race winner.",
            },
        )
        tm.join_published()

        assert resolved_exid == newer_exid, (
            "dedup must pick the most recent matching row "
            f"(expected newer={newer_exid}, got {resolved_exid})"
        )

        rows = unify.get_logs(
            context=exchanges_ctx,
            filter=(f"medium == 'email' and counterparty_contact_id == {replier_cid}"),
            limit=10,
            from_fields=["exchange_id"],
        )
        assert len(rows) == 2, (
            "no third row must be opened when duplicates already exist; "
            f"saw {len(rows)}: {[r.entries for r in rows]}"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)
