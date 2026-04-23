"""Merge contacts rewrites every Hive body's ``ContactMembership`` overlay.

``ContactMembership.contact_id`` is a foreign key onto the shared
``Contacts.contact_id``. Its cascade rule is ``ON DELETE SET NULL``,
which means a plain ``delete`` of the losing row would silently null
out every body's overlay pointing at it — losing the per-body
``relationship`` / ``response_policy`` decisions for the merged
contact. :func:`merge_contacts` therefore fans out to every Hive
member's overlay context and rewrites its ``contact_id`` *before*
deleting the losing shared row.

These tests exercise the fan-out end-to-end against a live Unify
backend, stubbing only the Orchestra Hive-membership lookup
(``list_hive_assistants``) so the test can stand up synthetic peer
bodies without provisioning a real Hive.
"""

from __future__ import annotations

import uuid

import unify

from unity.common.model_to_fields import model_to_fields
from unity.contact_manager import ops
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import ContactMembership
from unity.session_details import SESSION_DETAILS
from tests.helpers import _handle_project


def _peer_overlay_context() -> str:
    """A stand-in per-body overlay context for a synthetic Hive peer.

    Uses a project-root path outside the test tree so the primary
    body's own ``_membership_ctx`` is never accidentally aliased to
    the peer — the whole point of the fan-out test is to watch both
    contexts independently.
    """
    return f"cm_merge_peer_{uuid.uuid4().hex[:10]}/99/ContactMembership"


def _create_peer_overlay_context(context: str) -> None:
    """Provision a peer body's overlay context with the real schema."""
    unify.create_context(
        context,
        unique_keys={"contact_id": "int"},
        auto_counting=None,
    )
    # A full ``foreign_keys`` declaration would bind the peer overlay
    # to ``Hives/{hive_id}/Contacts``, which this helper does not
    # provision. The rewrite path only exercises ``get_logs`` /
    # ``update_logs`` / ``delete_logs`` on ``contact_id``, so the FK
    # is not load-bearing here and omitting it keeps the synthetic
    # peer context lightweight.
    fields = model_to_fields(ContactMembership)
    if fields:
        try:
            from unify import create_fields

            create_fields(fields, context=context)
        except Exception:
            pass


def _write_peer_overlay(
    context: str,
    *,
    contact_id: int,
    relationship: str = "other",
    response_policy: str | None = "keep me",
) -> int:
    """Write one ``ContactMembership`` row to a peer's overlay context."""
    log = unify.log(
        context=context,
        contact_id=int(contact_id),
        relationship=relationship,
        should_respond=True,
        response_policy=response_policy,
        can_edit=True,
    )
    return int(log.id)


def _cleanup_peer_overlay_context(context: str) -> None:
    """Best-effort delete of a peer overlay context."""
    try:
        root = context.split("/")[0]
        unify.delete_context(context=root, include_children=True)
    except Exception:
        pass


def _overlay_rows(context: str, *, contact_id: int) -> list:
    """Read every overlay row on *context* referencing ``contact_id``."""
    return (
        unify.get_logs(
            context=context,
            filter=f"contact_id == {int(contact_id)}",
            limit=50,
        )
        or []
    )


@_handle_project
def test_solo_merge_rewrites_this_body_overlay():
    """A solo body's merge rewrites its own overlay from delete_id to keep_id."""
    cm = ContactManager()

    keep_id = cm._create_contact(
        first_name="Solo",
        email_address=f"solo.keep.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]
    delete_id = cm._create_contact(
        first_name="Dupe",
        email_address=f"solo.dupe.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    cm._merge_contacts(
        contact_id_1=keep_id,
        contact_id_2=delete_id,
        overrides={"contact_id": 1},
    )

    assert _overlay_rows(cm._membership_ctx, contact_id=delete_id) == []
    survivors = _overlay_rows(cm._membership_ctx, contact_id=keep_id)
    assert len(survivors) == 1, (
        "Surviving overlay missing after solo merge: " f"{survivors!r}"
    )


@_handle_project
def test_merge_preserves_authoring_assistant_id(monkeypatch):
    """The surviving shared row keeps its own ``authoring_assistant_id``.

    The merger stamps itself as author on create, but must never
    overwrite a row authored by a different body during a merge —
    the survivor retains the original author stamp as a write-once
    audit field.
    """
    cm = ContactManager()

    monkeypatch.setattr(ops, "_get_assistant_id", lambda: 101)
    keep_id = cm._create_contact(
        first_name="Original",
        email_address=f"orig.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    monkeypatch.setattr(ops, "_get_assistant_id", lambda: 202)
    delete_id = cm._create_contact(
        first_name="Other",
        email_address=f"other.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    # Merge runs under a third body to prove the survivor's stamp is
    # anchored to the authoring body, not to whoever invokes the merge.
    monkeypatch.setattr(ops, "_get_assistant_id", lambda: 303)
    cm._merge_contacts(
        contact_id_1=keep_id,
        contact_id_2=delete_id,
        overrides={"contact_id": 1, "first_name": 2},
    )

    rows = unify.get_logs(
        context=cm._ctx,
        filter=f"contact_id == {int(keep_id)}",
        limit=1,
    )
    assert rows, "Surviving shared Contact row missing after merge"
    assert rows[0].entries.get("authoring_assistant_id") == 101, (
        "Merger overwrote the surviving row's authoring_assistant_id: "
        f"{rows[0].entries!r}"
    )
    assert (
        rows[0].entries.get("first_name") == "Other"
    ), "Override did not apply to the non-audit field"


@_handle_project
def test_hive_merge_fan_out_rewrites_peer_body_overlay(monkeypatch):
    """Every Hive member's overlay is rewritten, not just the merger's.

    Stubs ``list_hive_assistants`` with a synthetic peer body, pre-
    creates that peer's overlay context on the backend, and asserts
    the merge-driven fan-out visits it.
    """
    cm = ContactManager()

    keep_id = cm._create_contact(
        first_name="HiveKeep",
        email_address=f"hive.keep.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]
    delete_id = cm._create_contact(
        first_name="HiveDupe",
        email_address=f"hive.dupe.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    peer_ctx = _peer_overlay_context()
    peer_user, peer_assistant, _suffix = peer_ctx.split("/", 2)
    _create_peer_overlay_context(peer_ctx)

    try:
        _write_peer_overlay(peer_ctx, contact_id=delete_id)

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)
        # ``list_hive_assistants`` is imported lazily inside the
        # rewrite helper from ``backend_sync``; patch the source module
        # so the stub is picked up at call time.
        monkeypatch.setattr(
            "unity.contact_manager.backend_sync.list_hive_assistants",
            lambda _hid: [(peer_user, int(peer_assistant))],
        )

        ops._rewrite_membership_overlays_for_merge(
            cm,
            delete_id=delete_id,
            keep_id=keep_id,
        )

        assert _overlay_rows(peer_ctx, contact_id=delete_id) == []
        survivors = _overlay_rows(peer_ctx, contact_id=keep_id)
        assert (
            len(survivors) == 1
        ), f"Peer overlay was not rewritten to survivor: {survivors!r}"
        assert survivors[0].entries.get("response_policy") == "keep me", (
            "Peer overlay fields were lost during rewrite: " f"{survivors[0].entries!r}"
        )
    finally:
        _cleanup_peer_overlay_context(peer_ctx)


@_handle_project
def test_hive_merge_drops_peer_losing_overlay_when_surviving_exists(monkeypatch):
    """A peer body with overlays for both ids drops the losing one.

    Two overlays on the same shared contact would collide on the
    unique-key invariant, so the rewrite path deletes the losing row
    outright when the surviving overlay already exists.
    """
    cm = ContactManager()

    keep_id = cm._create_contact(
        first_name="DualKeep",
        email_address=f"dual.keep.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]
    delete_id = cm._create_contact(
        first_name="DualDupe",
        email_address=f"dual.dupe.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    peer_ctx = _peer_overlay_context()
    peer_user, peer_assistant, _suffix = peer_ctx.split("/", 2)
    _create_peer_overlay_context(peer_ctx)

    try:
        _write_peer_overlay(
            peer_ctx,
            contact_id=keep_id,
            response_policy="surviving policy",
        )
        _write_peer_overlay(
            peer_ctx,
            contact_id=delete_id,
            response_policy="losing policy",
        )

        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)
        monkeypatch.setattr(
            "unity.contact_manager.backend_sync.list_hive_assistants",
            lambda _hid: [(peer_user, int(peer_assistant))],
        )

        ops._rewrite_membership_overlays_for_merge(
            cm,
            delete_id=delete_id,
            keep_id=keep_id,
        )

        assert _overlay_rows(peer_ctx, contact_id=delete_id) == [], (
            "Losing overlay should have been deleted when the surviving "
            "overlay already existed on the peer body"
        )
        survivors = _overlay_rows(peer_ctx, contact_id=keep_id)
        assert (
            len(survivors) == 1
        ), f"Surviving overlay lost during collision resolution: {survivors!r}"
        assert survivors[0].entries.get("response_policy") == "surviving policy"
    finally:
        _cleanup_peer_overlay_context(peer_ctx)


@_handle_project
def test_hive_merge_skips_peer_without_matching_overlay(monkeypatch):
    """Peers with no overlay on either id are left untouched.

    Bodies that never interacted with either contact have nothing to
    rewrite; the rewrite helper must not mint a fresh overlay on
    their behalf. Lazy materialization happens at interaction
    boundaries, not at merge time.
    """
    cm = ContactManager()

    keep_id = cm._create_contact(
        first_name="QuietKeep",
        email_address=f"quiet.keep.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]
    delete_id = cm._create_contact(
        first_name="QuietDupe",
        email_address=f"quiet.dupe.{uuid.uuid4().hex[:8]}@example.com",
    )["details"]["contact_id"]

    peer_ctx = _peer_overlay_context()
    peer_user, peer_assistant, _suffix = peer_ctx.split("/", 2)
    _create_peer_overlay_context(peer_ctx)

    try:
        monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)
        monkeypatch.setattr(
            "unity.contact_manager.backend_sync.list_hive_assistants",
            lambda _hid: [(peer_user, int(peer_assistant))],
        )

        ops._rewrite_membership_overlays_for_merge(
            cm,
            delete_id=delete_id,
            keep_id=keep_id,
        )

        rows_keep = _overlay_rows(peer_ctx, contact_id=keep_id)
        rows_delete = _overlay_rows(peer_ctx, contact_id=delete_id)
        assert rows_keep == [], "Rewrite helper minted a fresh overlay"
        assert rows_delete == []
    finally:
        _cleanup_peer_overlay_context(peer_ctx)
