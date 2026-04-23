"""``ContactManager._hydrate`` and ``include_membership`` semantics.

The hydration helper composes the existing ``ContactIndex`` /
``DataStore`` cache for the shared row with a single-row
``ContactMembership`` log-context lookup for the overlay. It does
**not** materialize overlay rows — miss returns a ``HydratedContact``
with ``membership=None`` and no write fires. Materialization is
reserved for the interaction-boundary sites covered in
``test_lazy_materialization.py``.
"""

from __future__ import annotations

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import (
    Contact,
    ContactMembership,
    HydratedContact,
)
from tests.helpers import _handle_project


def test_hydrated_contact_is_a_frozen_pair():
    """``HydratedContact`` carries the shared + overlay pair and is immutable."""
    shared = Contact(contact_id=17, first_name="Iris")
    overlay = ContactMembership(contact_id=17, relationship="coworker")

    merged = HydratedContact(shared=shared, membership=overlay)

    assert merged.shared.contact_id == 17
    assert merged.membership is not None
    assert merged.membership.relationship == "coworker"


@_handle_project
def test_hydrate_returns_membership_when_overlay_exists():
    """A materialized overlay is returned alongside the shared row.

    ``_create_contact`` mints an overlay with ``relationship="other"``
    for a freshly-created external contact (no ``assistant_id`` on the
    shared row, no explicit relationship passed). ``_hydrate`` must
    compose that overlay into the :class:`HydratedContact`.
    """
    cm = ContactManager()
    result = cm._create_contact(
        first_name="Quinn",
        email_address="quinn@example.com",
    )
    cid = result["details"]["contact_id"]

    hydrated = cm._hydrate(cid)

    assert hydrated is not None
    assert hydrated.shared.contact_id == cid
    assert hydrated.membership is not None
    assert hydrated.membership.relationship == "other"


@_handle_project
def test_hydrate_does_not_mint_overlay_on_miss():
    """Miss returns ``membership=None`` and leaves the overlay absent.

    ``_hydrate`` is a read helper — if the overlay is missing the
    caller must explicitly materialize it via the lazy-materialization
    path. Silently writing on read would violate the locked design
    decision that read helpers never materialize overlays.
    """
    import unify

    cm = ContactManager()
    result = cm._create_contact(
        first_name="Orphan",
        email_address="orphan@example.com",
    )
    cid = result["details"]["contact_id"]

    rows_before = (
        unify.get_logs(
            context=cm._membership_ctx,
            filter=f"contact_id == {cid}",
            limit=1,
            return_ids_only=True,
        )
        or []
    )
    for log_id in rows_before:
        unify.delete_logs(context=cm._membership_ctx, logs=log_id)

    hydrated = cm._hydrate(cid)

    assert hydrated is not None
    assert hydrated.shared.contact_id == cid

    rows_after = (
        unify.get_logs(
            context=cm._membership_ctx,
            filter=f"contact_id == {cid}",
            limit=1,
            return_ids_only=True,
        )
        or []
    )
    assert not rows_after, "`_hydrate` must not materialize an overlay on miss"


@_handle_project
def test_get_contact_info_without_membership_skips_overlay_lookup(monkeypatch):
    """``include_membership=False`` (default) never touches the overlay ctx.

    Count every ``unify.get_logs`` call and assert none of them target
    the ``ContactMembership`` context unless ``include_membership=True``.
    """
    import unify

    cm = ContactManager()
    result = cm._create_contact(
        first_name="NoOverlay",
        email_address="nooverlay@example.com",
    )
    cid = result["details"]["contact_id"]

    membership_ctx = cm._membership_ctx
    overlay_calls = {"count": 0}
    original_get_logs = unify.get_logs

    def counting_get_logs(*args, **kwargs):
        if kwargs.get("context") == membership_ctx:
            overlay_calls["count"] += 1
        return original_get_logs(*args, **kwargs)

    monkeypatch.setattr(unify, "get_logs", counting_get_logs)

    cm.get_contact_info(contact_id=cid)
    assert overlay_calls["count"] == 0

    cm.get_contact_info(contact_id=cid, include_membership=True)
    assert overlay_calls["count"] >= 1


@_handle_project
def test_get_contact_info_with_membership_returns_overlay_fields():
    """``include_membership=True`` yields the merged view.

    The overlay fields arrive on the returned row because
    ``_create_contact`` seeded a ``ContactMembership`` with
    ``should_respond`` and ``response_policy`` forwarded from the
    create call; ``_hydrate`` merges them back into the shared-row dict.
    """
    cm = ContactManager()
    result = cm._create_contact(
        first_name="Merged",
        email_address="merged@example.com",
        should_respond=False,
        response_policy="quarantined",
    )
    cid = result["details"]["contact_id"]

    info = cm.get_contact_info(contact_id=cid, include_membership=True)

    assert cid in info
    row = info[cid]
    assert row.get("relationship") == "other"
    assert row.get("should_respond") is False
    assert row.get("response_policy") == "quarantined"
