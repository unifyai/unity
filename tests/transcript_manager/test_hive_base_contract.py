"""TranscriptManager Hive-base contract.

Both tables TranscriptManager owns — ``Transcripts`` and ``Exchanges`` —
are Hive-shared. When the active body belongs to a Hive, both tables
resolve under ``Hives/{hive_id}/...`` so every member of the Hive reads
and writes the same rows; a solo body keeps its per-body
``{user}/{assistant}/...`` root.

The model layer must also carry ``authoring_assistant_id`` on both
``Message`` and ``Exchange`` (stamped once at write time) and an
``Exchange.counterparty_contact_id`` column that pins the non-body
participant at open time so body-to-body dedup can find existing rows.
"""

from __future__ import annotations

import pytest

from unity.common.context_registry import ContextRegistry
from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.types.exchange import Exchange
from unity.transcript_manager.types.message import Message

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_transcripts_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Transcripts") == "Hives/42"


def test_hive_member_resolves_exchanges_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("Exchanges") == "Hives/42"


def test_solo_body_resolves_transcripts_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Transcripts") == pinned_hive_body


def test_solo_body_resolves_exchanges_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("Exchanges") == pinned_hive_body


def test_message_model_declares_authoring_assistant_id():
    """``Message`` carries an optional ``authoring_assistant_id`` stamp."""
    fields = set(Message.model_fields.keys())
    assert "authoring_assistant_id" in fields
    assert Message.SHORTHAND_MAP.get("authoring_assistant_id") == "auth_aid"


def test_exchange_model_declares_authoring_and_counterparty():
    """``Exchange`` declares both Hive-scoping columns required for dedup."""
    fields = set(Exchange.model_fields.keys())
    assert "authoring_assistant_id" in fields
    assert "counterparty_contact_id" in fields
