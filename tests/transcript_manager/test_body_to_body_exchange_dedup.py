"""Two Hive-mate bodies converging on one shared ``Exchange`` row.

:meth:`TranscriptManager.log_first_message_in_new_exchange` recognises a
Hive-mate counterparty (``Contact.assistant_id`` is populated) and
reuses the most recent exchange on the same medium whose
``counterparty_contact_id`` matches either participant's self-contact
id, so body A writing to body B and body B writing back to body A
converge on a single row. External contacts and solo bodies always
open a fresh exchange — the dedup fires only for body-to-body
conversation inside a Hive.
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
def test_two_bodies_converge_on_one_exchange():
    hive_id = fresh_hive_id()
    exchanges_ctx = f"Hives/{hive_id}/Exchanges"
    transcripts_ctx = f"Hives/{hive_id}/Transcripts"

    try:
        _, a_cid = bootstrap_hive_body(hive_id, agent_id=1001)
        _, b_cid = bootstrap_hive_body(hive_id, agent_id=2002)

        bind_body(hive_id=hive_id, agent_id=1001)
        pin_hive_body_base(hive_id, 1001)
        cm_a = ContactManager()
        tm_a = TranscriptManager(contact_manager=cm_a)

        exid_a, _ = tm_a.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": a_cid,
                "receiver_ids": [b_cid],
                "timestamp": datetime.now(UTC),
                "content": "Hi B — this is A opening the thread.",
            },
        )
        tm_a.join_published()

        bind_body(hive_id=hive_id, agent_id=2002)
        pin_hive_body_base(hive_id, 2002)
        cm_b = ContactManager()
        tm_b = TranscriptManager(contact_manager=cm_b)

        exid_b, _ = tm_b.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": b_cid,
                "receiver_ids": [a_cid],
                "timestamp": datetime.now(UTC),
                "content": "Hey A — replying on the same shared row.",
            },
        )
        tm_b.join_published()

        assert exid_a == exid_b, (
            f"body B should have reused body A's exchange " f"(a={exid_a}, b={exid_b})"
        )

        exchange_rows = unify.get_logs(
            context=exchanges_ctx,
            filter=f"medium == 'email' and counterparty_contact_id in [{a_cid}, {b_cid}]",
            limit=10,
        )
        assert len(exchange_rows) == 1, (
            f"expected one shared exchange row, saw {len(exchange_rows)}: "
            f"{[r.entries for r in exchange_rows]}"
        )

        transcripts = unify.get_logs(
            context=transcripts_ctx,
            filter=f"exchange_id == {exid_a}",
            limit=10,
        )
        authors = {row.entries.get("authoring_assistant_id") for row in transcripts}
        assert authors == {1001, 2002}, (
            f"both bodies should have authored messages on the shared exchange, "
            f"saw authors={authors}"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)


@pytest.mark.requires_real_unify
def test_external_contact_always_opens_new_exchange():
    """Two opens for the same external contact must not collapse onto one row.

    External contacts have ``assistant_id`` unset, so
    ``_counterparty_is_hive_mate`` rejects them and the dedup path
    returns ``None``. Each call opens a fresh exchange.
    """
    hive_id = fresh_hive_id()
    try:
        tm, self_cid = bootstrap_hive_body(hive_id, agent_id=3003)
        cm = ContactManager()
        external = cm._create_contact(
            first_name="External",
            email_address=f"ext-{hive_id}@example.com",
        )
        external_cid = int(external["details"]["contact_id"])

        exid_one, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": self_cid,
                "receiver_ids": [external_cid],
                "timestamp": datetime.now(UTC),
                "content": "first",
            },
        )
        exid_two, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": self_cid,
                "receiver_ids": [external_cid],
                "timestamp": datetime.now(UTC),
                "content": "second — should open a fresh exchange",
            },
        )
        tm.join_published()

        assert exid_one != exid_two, (
            "external-contact opens must never dedup onto the prior row "
            f"(got {exid_one} == {exid_two})"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)


@pytest.mark.requires_real_unify
def test_solo_body_never_dedups():
    """A solo body (no ``hive_id``) always opens fresh exchanges.

    ``_find_hive_mate_exchange_id`` short-circuits when
    ``SESSION_DETAILS.hive_id`` is ``None`` — otherwise a solo body
    would start collapsing conversations with the same contact onto
    one row, contradicting the async-media contract.
    """
    import random

    user_id = f"solo-tm-{random.randint(10_000_000, 99_999_999)}"
    agent_id = random.randint(10_000_000, 99_999_999)
    solo_root = f"{user_id}/{agent_id}"

    try:
        bind_body(
            hive_id=None,
            agent_id=agent_id,
            user_id=user_id,
            solo_base=solo_root,
        )
        cm = ContactManager()
        tm = TranscriptManager(contact_manager=cm)

        peer = cm._create_contact(
            first_name="Peer",
            email_address=f"peer-{agent_id}@example.com",
        )
        peer_cid = int(peer["details"]["contact_id"])

        exid_one, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": 0,
                "receiver_ids": [peer_cid],
                "timestamp": datetime.now(UTC),
                "content": "first",
            },
        )
        exid_two, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": 0,
                "receiver_ids": [peer_cid],
                "timestamp": datetime.now(UTC),
                "content": "second",
            },
        )
        tm.join_published()

        assert exid_one != exid_two, (
            "solo bodies must never dedup — every open is a fresh exchange "
            f"(got {exid_one} == {exid_two})"
        )
    finally:
        SESSION_DETAILS.reset()
        try:
            unify.delete_context(context=solo_root, include_children=True)
        except Exception:
            pass
