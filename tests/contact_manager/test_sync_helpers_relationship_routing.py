"""Sync helpers route via ``ContactMembership.relationship``.

The per-body overlay is the primary routing signal for the
``_maybe_sync_*`` helpers: ``"self"`` → assistant-side backend,
``"coworker"`` → no-op, ``"boss"`` or the ``is_system + email``
fallback → user-side sync.

These tests exercise a real :class:`ContactManager` against the
live backend and stub only the outbound HTTP sync wrappers (which
would otherwise hit the Unify Orchestra user/assistant profile
endpoints). The routing logic itself — overlay lookup, shared-row
fetch, ``is_system`` fallback — runs end-to-end.
"""

from __future__ import annotations

from unity.contact_manager import ops
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.settings import (
    RELATIONSHIP_BOSS,
    RELATIONSHIP_COWORKER,
    RELATIONSHIP_SELF,
)
from tests.helpers import _handle_project


def _install_sync_stubs(monkeypatch):
    """Replace outbound sync wrappers with recording stubs.

    Returns a dict that the test can inspect; keys are populated when
    the corresponding sync fires.
    """
    calls: dict = {
        "assistant_tz": 0,
        "user_tz_email": None,
        "assistant_about": 0,
        "user_bio_email": None,
        "assistant_job_title": None,
    }

    def fake_assistant_tz(aid, tz):
        calls["assistant_tz"] += 1
        return True

    def fake_user_tz(aid, email, tz):
        calls["user_tz_email"] = email
        return True

    def fake_assistant_about(aid, about):
        calls["assistant_about"] += 1
        return True

    def fake_user_bio(aid, email, about):
        calls["user_bio_email"] = email
        return True

    def fake_assistant_job_title(aid, jt):
        calls["assistant_job_title"] = jt
        return True

    monkeypatch.setattr(
        "unity.contact_manager.backend_sync.sync_assistant_timezone",
        fake_assistant_tz,
    )
    monkeypatch.setattr(
        "unity.contact_manager.backend_sync.sync_user_timezone",
        fake_user_tz,
    )
    monkeypatch.setattr(
        "unity.contact_manager.backend_sync.sync_assistant_about",
        fake_assistant_about,
    )
    monkeypatch.setattr(
        "unity.contact_manager.backend_sync.sync_user_bio",
        fake_user_bio,
    )
    monkeypatch.setattr(
        "unity.contact_manager.backend_sync.sync_assistant_job_title",
        fake_assistant_job_title,
    )
    monkeypatch.setattr(ops, "_get_assistant_id", lambda: 7)
    return calls


def _mint_contact_with_relationship(
    cm: ContactManager,
    *,
    email: str,
    relationship: str,
    assistant_id: int | None = None,
    is_system: bool = False,
) -> int:
    """Create a fresh shared Contact + overlay with the given relationship."""
    outcome = cm._create_contact(
        first_name="Target",
        email_address=email,
        _relationship=relationship,
        _is_system=is_system,
        _assistant_id=assistant_id,
    )
    return int(outcome["details"]["contact_id"])


@_handle_project
def test_self_overlay_routes_to_assistant_backend(monkeypatch):
    """``relationship == "self"`` fires the assistant-side timezone sync."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="me@example.com",
        relationship=RELATIONSHIP_SELF,
    )
    ops._maybe_sync_timezone_to_backend(cm, cid, "UTC")

    assert calls["assistant_tz"] == 1
    assert calls["user_tz_email"] is None


@_handle_project
def test_coworker_overlay_is_noop(monkeypatch):
    """``"coworker"`` short-circuits — another body's backend is not ours."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="mate@hive.com",
        relationship=RELATIONSHIP_COWORKER,
        assistant_id=999,
    )
    ops._maybe_sync_timezone_to_backend(cm, cid, "UTC")

    assert calls["assistant_tz"] == 0
    assert calls["user_tz_email"] is None


@_handle_project
def test_boss_overlay_routes_to_user_backend(monkeypatch):
    """``"boss"`` with an email on the shared row fires the user-side sync."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="boss@example.com",
        relationship=RELATIONSHIP_BOSS,
    )
    ops._maybe_sync_timezone_to_backend(cm, cid, "UTC")

    assert calls["user_tz_email"] == "boss@example.com"
    assert calls["assistant_tz"] == 0


@_handle_project
def test_missing_overlay_falls_back_to_is_system_email(monkeypatch):
    """No overlay + ``is_system`` + ``email_address`` → user-side sync fires.

    A body that has not yet materialized its ``"boss"`` overlay still
    needs the boss contact's timezone to reach the user record; the
    shared ``is_system`` + ``email_address`` signals drive that
    fallback.
    """
    import unify

    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="legacy@example.com",
        relationship=RELATIONSHIP_BOSS,
        is_system=True,
    )
    # Drop the overlay so the helper falls back to is_system+email.
    ids = unify.get_logs(
        context=cm._membership_ctx,
        filter=f"contact_id == {cid}",
        limit=1,
        return_ids_only=True,
    )
    if ids:
        unify.delete_logs(logs=ids)

    ops._maybe_sync_timezone_to_backend(cm, cid, "UTC")

    assert calls["user_tz_email"] == "legacy@example.com"


@_handle_project
def test_bio_helper_mirrors_timezone_routing(monkeypatch):
    """``_maybe_sync_bio_to_backend`` routes through the same overlay signal."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="me@example.com",
        relationship=RELATIONSHIP_SELF,
    )
    ops._maybe_sync_bio_to_backend(cm, cid, "Hello.")

    assert calls["assistant_about"] == 1
    assert calls["user_bio_email"] is None


@_handle_project
def test_job_title_only_syncs_self_relationship(monkeypatch):
    """``"boss"`` never flows ``job_title`` to the backend — it's a self-only field."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="boss@example.com",
        relationship=RELATIONSHIP_BOSS,
    )
    ops._maybe_sync_job_title_to_backend(cm, cid, "Director")

    assert calls["assistant_job_title"] is None


@_handle_project
def test_job_title_syncs_when_self(monkeypatch):
    """``"self"`` mirrors the assistant's ``job_title`` to the backend."""
    calls = _install_sync_stubs(monkeypatch)
    cm = ContactManager()

    cid = _mint_contact_with_relationship(
        cm,
        email="me@example.com",
        relationship=RELATIONSHIP_SELF,
    )
    ops._maybe_sync_job_title_to_backend(cm, cid, "Director")

    assert calls["assistant_job_title"] == "Director"
