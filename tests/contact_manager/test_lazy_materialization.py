"""Overlay rows materialize only at interaction boundaries.

One invariant: ``_hydrate`` and every other read helper must **never**
mint an overlay. Materialization happens only at explicit interaction
boundaries — inbound-message routing and LLM tool calls that set
relationship/response fields — via the
``ContactManager.materialize_membership_if_missing`` helper.

``_create_contact`` also mints an overlay as part of the write flow
so the authoring body owns a relationship from the first moment.
These tests therefore exercise the lazy path by creating a shared
contact, deleting its overlay (simulating a contact this body has
never interacted with), and asserting the helper then mints, is
idempotent, and never fires from a pure read.
"""

from __future__ import annotations

import unify

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


def _drop_overlay(cm: ContactManager, contact_id: int) -> None:
    """Delete every ContactMembership row for *contact_id* on this body."""
    existing = (
        unify.get_logs(
            context=cm._membership_ctx,
            filter=f"contact_id == {contact_id}",
            limit=10,
            return_ids_only=True,
        )
        or []
    )
    for log_id in existing:
        unify.delete_logs(context=cm._membership_ctx, logs=log_id)


@_handle_project
def test_materialize_mints_and_is_idempotent_on_second_call():
    """First call creates the overlay; a second returns ``False``."""
    cm = ContactManager()
    result = cm._create_contact(
        first_name="Twice",
        email_address="twice@example.com",
    )
    cid = result["details"]["contact_id"]

    _drop_overlay(cm, cid)

    created_first = cm.materialize_membership_if_missing(
        cid,
        relationship="coworker",
    )
    created_second = cm.materialize_membership_if_missing(
        cid,
        relationship="coworker",
    )

    assert created_first is True
    assert created_second is False

    rows = unify.get_logs(
        context=cm._membership_ctx,
        filter=f"contact_id == {cid}",
        limit=5,
    )
    assert len(rows) == 1


@_handle_project
def test_materialize_respects_explicit_relationship():
    """Callers that pass ``relationship`` get that value on the overlay."""
    cm = ContactManager()
    result = cm._create_contact(
        first_name="Mover",
        email_address="mover@example.com",
    )
    cid = result["details"]["contact_id"]

    _drop_overlay(cm, cid)

    cm.materialize_membership_if_missing(
        cid,
        relationship="other",
        should_respond=False,
        response_policy="quarantined",
    )

    rows = unify.get_logs(
        context=cm._membership_ctx,
        filter=f"contact_id == {cid}",
        limit=1,
    )
    assert len(rows) == 1
    entries = rows[0].entries
    assert entries.get("relationship") == "other"
    assert entries.get("should_respond") is False
    assert entries.get("response_policy") == "quarantined"


@_handle_project
def test_read_paths_do_not_mint_overlay():
    """``get_contact_info`` and ``_hydrate`` leave a dropped overlay absent.

    The default read path is hot and must not mint overlays — doing so
    would silently couple every read to an overlay write and turn the
    overlay into a de-facto cache of the shared row.
    """
    cm = ContactManager()
    result = cm._create_contact(
        first_name="ColdRead",
        email_address="coldread@example.com",
    )
    cid = result["details"]["contact_id"]

    _drop_overlay(cm, cid)

    cm.get_contact_info(contact_id=cid)
    cm._hydrate(cid)

    after = (
        unify.get_logs(
            context=cm._membership_ctx,
            filter=f"contact_id == {cid}",
            limit=5,
            return_ids_only=True,
        )
        or []
    )
    assert not after, "read path must not materialize an overlay"


@_handle_project
def test_materialize_preserves_existing_row():
    """Pre-existing overlay rows are preserved as-is by a retry.

    The helper is deliberately not an upsert — callers that want to
    change fields on an existing overlay write through
    ``_write_membership_overlay`` or the ``update_contact`` overlay path.
    """
    cm = ContactManager()
    result = cm._create_contact(
        first_name="Preserve",
        email_address="preserve@example.com",
    )
    cid = result["details"]["contact_id"]

    _drop_overlay(cm, cid)

    cm.materialize_membership_if_missing(
        cid,
        relationship="coworker",
        should_respond=False,
        response_policy="existing-policy",
    )

    created_again = cm.materialize_membership_if_missing(
        cid,
        relationship="other",
        should_respond=True,
        response_policy="different",
    )

    assert created_again is False
    rows = unify.get_logs(
        context=cm._membership_ctx,
        filter=f"contact_id == {cid}",
        limit=5,
    )
    assert len(rows) == 1
    entries = rows[0].entries
    assert entries.get("relationship") == "coworker"
    assert entries.get("should_respond") is False
    assert entries.get("response_policy") == "existing-policy"
