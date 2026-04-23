"""ContactManager resolves ``Contacts`` through the Hive or per-body root.

``"Contacts"`` is in ``_HIVE_SCOPED_TABLES``, so Hive members share a
single ``Hives/{hive_id}/Contacts`` table while solo bodies read and
write the per-body ``{user}/{assistant}/Contacts`` path. These tests
exercise the resolver contract directly so they don't need a live
Unify backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry
from unity.contact_manager.contact_manager import ContactManager
from unity.session_details import SESSION_DETAILS

_PER_BODY_BASE = "u7/7"


@pytest.fixture(autouse=True)
def _reset_session_details():
    """Each test gets a clean ``SESSION_DETAILS`` and registry state."""
    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(agent_id=7, user_id="u7")
    previous_base = ContextRegistry._base_context
    ContextRegistry._base_context = _PER_BODY_BASE
    yield
    SESSION_DETAILS.reset()
    ContextRegistry._base_context = previous_base


def _resolve() -> dict:
    """Return ``_get_contexts_for_manager`` output for ContactManager.

    Bypasses ``ContactManager.__init__`` entirely — it requires a live
    Unify backend — and inspects the resolver's output to assert the
    storage roots each required context lands on.
    """
    return ContextRegistry._get_contexts_for_manager(ContactManager)


def test_hive_member_resolves_contacts_to_hive_root():
    """A Hive member reads and writes ``Contacts`` under ``Hives/{hive_id}``."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert resolved["Contacts"]["resolved_name"] == "Hives/42/Contacts"


def test_solo_body_resolves_contacts_to_per_body_root():
    """A solo body stores ``Contacts`` under its per-body root."""
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    assert resolved["Contacts"]["resolved_name"] == f"{_PER_BODY_BASE}/Contacts"


def test_hive_member_keeps_contact_membership_on_per_body_root():
    """``ContactMembership`` is the per-body overlay and never Hive-shared."""
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    assert (
        resolved["ContactMembership"]["resolved_name"]
        == f"{_PER_BODY_BASE}/ContactMembership"
    )


def test_contact_membership_fk_resolves_to_hive_contacts_for_hive_member():
    """The overlay's FK points at the shared ``Hives/{hive_id}/Contacts`` row.

    The FK is declared as the relative ``Contacts.contact_id``; the
    registry rewrites it through the *target* table's base so it lands
    on the Hive root automatically.
    """
    SESSION_DETAILS.hive_id = 42

    resolved = _resolve()

    fks = resolved["ContactMembership"]["resolved_foreign_keys"]
    assert fks and len(fks) == 1
    assert fks[0]["references"] == "Hives/42/Contacts.contact_id"


def test_contact_membership_fk_resolves_to_per_body_contacts_for_solo():
    """Solo bodies target ``{user}/{assistant}/Contacts.contact_id`` instead."""
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    fks = resolved["ContactMembership"]["resolved_foreign_keys"]
    assert fks and len(fks) == 1
    assert fks[0]["references"] == f"{_PER_BODY_BASE}/Contacts.contact_id"


def test_solo_body_resolves_contacts_and_overlay_on_same_base():
    """Solo bodies keep ``Contacts`` and its overlay FK on one per-body base.

    The per-body path stores the shared table and the overlay under the
    same root, and the FK resolves against that same root. This pins the
    solo contract so cross-table routing cannot silently split them.
    """
    assert SESSION_DETAILS.hive_id is None

    resolved = _resolve()

    assert resolved["Contacts"]["resolved_name"] == f"{_PER_BODY_BASE}/Contacts"
    assert (
        resolved["ContactMembership"]["resolved_name"]
        == f"{_PER_BODY_BASE}/ContactMembership"
    )
    fks = resolved["ContactMembership"]["resolved_foreign_keys"]
    assert fks[0]["references"] == f"{_PER_BODY_BASE}/Contacts.contact_id"


def test_resolver_provisions_both_contexts_once():
    """Both required contexts run through the same provisioning hook."""
    SESSION_DETAILS.hive_id = 42

    with patch.object(
        ContextRegistry,
        "_create_context_wrapper",
        side_effect=lambda name, entry: entry["resolved_name"],
    ) as provision:
        resolved = ContextRegistry._get_contexts_for_manager(ContactManager)
        for entry in resolved.values():
            ContextRegistry._create_context_wrapper(
                ContactManager.__name__,
                entry,
            )

    resolved_names = {
        call.args[1]["resolved_name"] for call in provision.call_args_list
    }
    assert "Hives/42/Contacts" in resolved_names
    assert f"{_PER_BODY_BASE}/ContactMembership" in resolved_names
