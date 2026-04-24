"""BlackListManager Hive-base contract.

``"BlackList"`` is in ``_HIVE_SCOPED_TABLES`` so a body inside a Hive reads
and writes it under ``Hives/{hive_id}/BlackList``; a solo body keeps its
per-body ``{user}/{assistant}/BlackList`` root. Every row carries
``authoring_assistant_id`` so Hive-shared rows can be attributed back to
the authoring body.
"""

from __future__ import annotations

import pytest

from unity.blacklist_manager.types.blacklist import BlackList
from unity.common.context_registry import ContextRegistry
from unity.conversation_manager.cm_types import Medium
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")


def test_hive_member_resolves_blacklist_to_hive_root():
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for("BlackList") == "Hives/42"


def test_solo_body_resolves_blacklist_to_per_body_root(pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for("BlackList") == pinned_hive_body


def test_blacklist_model_declares_authoring_assistant_id():
    field = BlackList.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None


def test_blacklist_model_accepts_authoring_assistant_id_stamp():
    entry = BlackList(
        medium=Medium.EMAIL,
        contact_detail="spam@example.com",
        reason="spam",
        authoring_assistant_id=7,
    )
    assert entry.authoring_assistant_id == 7
