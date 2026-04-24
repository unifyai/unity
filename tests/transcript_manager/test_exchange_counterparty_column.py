"""Exchange round-trip for ``counterparty_contact_id`` and authoring stamp.

Opening a new exchange via :meth:`log_first_message_in_new_exchange`
must stamp the participant-derived ``counterparty_contact_id`` and the
opener's ``authoring_assistant_id`` on the new row; the model read-back
must surface them intact. The dedup path in
``log_first_message_in_new_exchange`` keys on these two columns so a
regression here silently breaks every body-to-body conversation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import unify

from tests.helpers import _handle_project
from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.exchange import Exchange


@_handle_project
def test_new_exchange_stamps_counterparty_and_authoring():
    SESSION_DETAILS.populate(agent_id=4242, user_id="exchange-author")
    tm = TranscriptManager()

    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": 101,
            "receiver_ids": [202],
            "timestamp": datetime.now(UTC),
            "content": "hello",
        },
    )
    tm.join_published()

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows, "new exchange row must exist"
    entries = rows[0].entries
    # No ``"self"`` overlay is provisioned here, so ``_self_contact_id``
    # returns ``None`` and the sender is picked as the non-self
    # counterparty — see ``_counterparty_contact_id``.
    assert entries.get("counterparty_contact_id") == 101
    assert entries.get("authoring_assistant_id") == 4242


@_handle_project
def test_exchange_model_round_trips_counterparty_and_authoring():
    SESSION_DETAILS.populate(agent_id=7777, user_id="exchange-author")
    tm = TranscriptManager()

    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": 500,
            "receiver_ids": [600],
            "timestamp": datetime.now(UTC),
            "content": "seed",
        },
    )
    tm.join_published()

    ex = tm.get_exchange_metadata(exid)
    assert isinstance(ex, Exchange)
    assert ex.exchange_id == exid
    assert ex.counterparty_contact_id == 500
    assert ex.authoring_assistant_id == 7777
