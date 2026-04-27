"""System-contact provisioning hooks for Hive and solo bodies.

Three system-contact helpers cover ``self`` and ``boss`` provisioning:

- ``provision_self_contact`` — materializes this body's ``"self"``
  overlay, creating the shared row (with ``assistant_id`` stamped) if
  it is missing.
- ``provision_hive_boss_contact`` — Hive-creation pipeline helper that
  writes a shared boss row under ``Hives/{hive_id}/Contacts``.
- ``provision_boss_overlay`` — materializes this body's ``"boss"``
  overlay against whichever shared row represents the boss.

``ContactManager.__init__`` calls ``_provision_system_overlays``, which
fans out to these helpers and reconciles the resolved ids into the
runtime session.
"""

from __future__ import annotations

import os

from unity.contact_manager import system_contacts
from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@_handle_project
def test_provision_system_overlays_is_idempotent(monkeypatch):
    """A second call adds no duplicate ``"self"`` / ``"boss"`` overlays."""
    import unify

    cm = ContactManager()

    system_contacts.provision_system_overlays(cm)
    system_contacts.provision_system_overlays(cm)
    system_contacts.provision_system_overlays(cm)

    rows = unify.get_logs(
        context=cm._membership_ctx,
        filter="relationship == 'self' or relationship == 'boss'",
        limit=10,
    )
    relationships = [r.entries.get("relationship") for r in rows]
    assert relationships.count("self") == 1
    assert relationships.count("boss") == 1


@_handle_project
def test_self_overlay_points_at_assistant_id_row(monkeypatch):
    """The ``"self"`` overlay references the shared row stamped with this body."""
    from unity.session_details import SESSION_DETAILS

    SESSION_DETAILS.populate(agent_id=4242, user_id="u-self")

    cm = ContactManager()

    import unify

    self_overlay = unify.get_logs(
        context=cm._membership_ctx,
        filter="relationship == 'self'",
        limit=1,
    )
    assert self_overlay, "setup must eagerly mint a self overlay"

    cid = int(self_overlay[0].entries.get("contact_id"))
    shared_rows = unify.get_logs(
        context=cm._ctx,
        filter=f"contact_id == {cid}",
        limit=1,
    )
    assert shared_rows
    # The shared row for a body representing itself carries
    # ``assistant_id`` populated with the session's agent id.
    assert shared_rows[0].entries.get("assistant_id") == 4242


@_handle_project
def test_boss_overlay_points_at_is_system_row():
    """``"boss"`` overlay lands on an ``is_system`` shared row."""
    import unify

    cm = ContactManager()

    boss_overlay = unify.get_logs(
        context=cm._membership_ctx,
        filter="relationship == 'boss'",
        limit=1,
    )
    assert boss_overlay, "setup must eagerly mint a boss overlay"
    cid = int(boss_overlay[0].entries.get("contact_id"))

    shared = unify.get_logs(
        context=cm._ctx,
        filter=f"contact_id == {cid}",
        limit=1,
    )
    assert shared
    assert shared[0].entries.get("is_system") is True


@_handle_project
def test_system_overlay_provisioning_reconciles_session_contact_ids(monkeypatch):
    """First-boot overlay provisioning refreshes stale session anchors."""
    import unify
    from unity.session_details import SESSION_DETAILS

    monkeypatch.delenv("SELF_CONTACT_ID", raising=False)
    monkeypatch.delenv("BOSS_CONTACT_ID", raising=False)

    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(
        agent_id=5151,
        user_id="u-reconcile",
        assistant_first_name="Megan",
        assistant_surname="Richardson",
        user_first_name="Yusha",
        user_surname="Arif",
        user_email="boss-reconcile@example.com",
    )
    assert SESSION_DETAILS.assistant.contact_id is None
    assert SESSION_DETAILS.user.contact_id is None

    cm = ContactManager()

    overlays = unify.get_logs(
        context=cm._membership_ctx,
        filter="relationship == 'self' or relationship == 'boss'",
        limit=10,
    )
    ids_by_relationship = {
        row.entries["relationship"]: int(row.entries["contact_id"]) for row in overlays
    }

    assert SESSION_DETAILS.assistant.contact_id == ids_by_relationship["self"]
    assert SESSION_DETAILS.user.contact_id == ids_by_relationship["boss"]
    assert os.environ["SELF_CONTACT_ID"] == str(ids_by_relationship["self"])
    assert os.environ["BOSS_CONTACT_ID"] == str(ids_by_relationship["boss"])


def _unique_hive_id() -> int:
    """A per-call hive_id so hive-rooted integration tests don't collide."""
    import random

    return random.randint(10_000_000, 99_999_999)


def _setup_hive_contacts_context(hive_id: int) -> None:
    """Register the shared Contacts context under a synthetic hive root.

    In production the Orchestra Hive-creation pipeline (1.4) provisions
    ``Hives/{hive_id}/Contacts`` with the usual ``contact_id`` primary
    key + auto-counting; tests for the unity-side helper mirror that
    setup so the inserted row actually receives a ``contact_id``.
    """
    import unify

    try:
        unify.create_context(
            f"Hives/{int(hive_id)}/Contacts",
            unique_keys={"contact_id": "int"},
            auto_counting={"contact_id": None},
        )
    except Exception:
        pass


def _cleanup_hive_context(hive_id: int) -> None:
    """Delete the ``Hives/{hive_id}`` context on the backend, best-effort."""
    import unify

    try:
        unify.delete_context(
            context=f"Hives/{int(hive_id)}",
            include_children=True,
        )
    except Exception:
        pass


def test_provision_hive_boss_contact_is_idempotent_by_email():
    """Two calls with the same email return the same ``contact_id``.

    The helper is Orchestra-pipeline-owned and keyed on email so the
    Hive-creation flow can retry safely — we assert against the live
    backend rather than a mocked ``unify_log``.
    """
    hive_id = _unique_hive_id()
    _setup_hive_contacts_context(hive_id)
    try:
        first = system_contacts.provision_hive_boss_contact(
            hive_id=hive_id,
            email="boss@hive.example",
            first_name="The",
            surname="Boss",
            timezone="UTC",
        )
        second = system_contacts.provision_hive_boss_contact(
            hive_id=hive_id,
            email="boss@hive.example",
            first_name="The",
            surname="Boss",
            timezone="UTC",
        )

        assert first is not None
        assert first == second

        import unify

        rows = unify.get_logs(
            context=f"Hives/{hive_id}/Contacts",
            filter="email_address == 'boss@hive.example'",
            limit=5,
        )
        assert len(rows) == 1
    finally:
        _cleanup_hive_context(hive_id)


def test_provision_hive_boss_contact_writes_under_hive_root():
    """The shared boss row lands under ``Hives/{hive_id}/Contacts``.

    ``assistant_id`` is intentionally not stamped — a boss row
    represents a human, not a Hive-member body.
    """
    hive_id = _unique_hive_id()
    _setup_hive_contacts_context(hive_id)
    try:
        cid = system_contacts.provision_hive_boss_contact(
            hive_id=hive_id,
            email="b@h.com",
            first_name="B",
        )
        assert cid is not None

        import unify

        rows = unify.get_logs(
            context=f"Hives/{hive_id}/Contacts",
            filter=f"contact_id == {cid}",
            limit=1,
        )
        assert rows
        entries = rows[0].entries
        assert entries.get("email_address") == "b@h.com"
        assert entries.get("is_system") is True
        assert entries.get("assistant_id") is None
    finally:
        _cleanup_hive_context(hive_id)


def test_provision_hive_boss_contact_returns_none_without_hive_id():
    """Without a ``hive_id`` the helper is a safe no-op."""
    assert (
        system_contacts.provision_hive_boss_contact(hive_id=0, email="x@y.com") is None
    )
