"""Async-media inbounds always open a fresh exchange.

SMS, email, and WhatsApp text are async media: each inbound message
produces its own exchange row unless the calling layer explicitly
supplies an ``exchange_id``. The body-to-body dedup applies only
when the counterparty is a Hive-mate body — an external contact's
``Contact.assistant_id`` is unset, so
``_counterparty_is_hive_mate`` rejects the lookup and every open
falls through to the "create new" path regardless of medium.

The regression guards against accidentally widening the dedup
surface to external async-media inbounds, which would collapse
unrelated inbound SMS threads onto one shared row.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import unify

from tests.helpers import bootstrap_hive_body, cleanup_hive_context, fresh_hive_id
from unity.contact_manager.contact_manager import ContactManager
from unity.conversation_manager.cm_types.medium import Medium
from unity.session_details import SESSION_DETAILS


@pytest.mark.requires_real_unify
@pytest.mark.parametrize(
    "medium",
    [Medium.SMS_MESSAGE, Medium.EMAIL, Medium.WHATSAPP_MESSAGE],
)
def test_async_media_inbound_from_external_never_dedups(medium: Medium):
    hive_id = fresh_hive_id()
    exchanges_ctx = f"Hives/{hive_id}/Exchanges"

    try:
        tm, self_cid = bootstrap_hive_body(hive_id, agent_id=6000)
        cm = ContactManager()
        external = cm._create_contact(
            first_name="External",
            email_address=f"ext-{hive_id}-{medium}@example.com",
            number=f"+{hive_id}{abs(hash(str(medium))) % 9_999}",
        )
        external_cid = int(external["details"]["contact_id"])

        first_exid, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": medium,
                "sender_id": external_cid,
                "receiver_ids": [self_cid],
                "timestamp": datetime.now(UTC),
                "content": "first inbound",
            },
        )
        second_exid, _ = tm.log_first_message_in_new_exchange(
            {
                "medium": medium,
                "sender_id": external_cid,
                "receiver_ids": [self_cid],
                "timestamp": datetime.now(UTC),
                "content": "second inbound on same medium/counterparty",
            },
        )
        tm.join_published()

        assert first_exid != second_exid, (
            f"async-media inbounds from an external contact must never dedup "
            f"(medium={medium}, got {first_exid} == {second_exid})"
        )

        rows = unify.get_logs(
            context=exchanges_ctx,
            filter=(
                f"medium == '{medium}' and counterparty_contact_id == {external_cid}"
            ),
            limit=10,
            from_fields=["exchange_id"],
        )
        assert len(rows) == 2, (
            f"expected two distinct exchange rows for async-media "
            f"(medium={medium}); saw {len(rows)}: {[r.entries for r in rows]}"
        )
    finally:
        SESSION_DETAILS.reset()
        cleanup_hive_context(hive_id)
