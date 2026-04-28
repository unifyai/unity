"""``ContactManager.should_respond_to`` policy contract.

The outbound-comms gate and the prompt renderer both consult
``should_respond_to`` to decide whether the assistant is permitted to
proactively message a contact. The contract is:

- absent any per-body policy override, the answer is ``True`` (match
  the ``ContactMembership.should_respond`` model default);
- an explicit ``False`` on the per-body overlay blocks the send;
- an explicit ``True`` on the overlay allows it.

These tests exercise the policy decision directly so a regression in
either subclass is caught without booting the full conversation
pipeline.
"""

from __future__ import annotations

from unittest.mock import patch

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.contact_manager.types.contact import (
    Contact,
    ContactMembership,
    HydratedContact,
)


def _hydrated(contact_id: int, membership: ContactMembership | None) -> HydratedContact:
    return HydratedContact(
        shared=Contact(contact_id=contact_id, first_name="Test"),
        membership=membership,
    )


def _real_cm(hydrate_returns: HydratedContact | None) -> ContactManager:
    cm = ContactManager.__new__(ContactManager)
    patcher = patch.object(cm, "_hydrate", return_value=hydrate_returns)
    patcher.start()
    return cm


def _simulated_cm(contacts: dict[int, dict]) -> SimulatedContactManager:
    cm = SimulatedContactManager.__new__(SimulatedContactManager)
    cm._contacts = contacts
    return cm


# ---------------------------------------------------------------------------
# Real ContactManager — policy reads via _hydrate
# ---------------------------------------------------------------------------


def test_real_cm_allows_when_overlay_missing() -> None:
    """No overlay row materialized yet → default-allow."""
    cm = _real_cm(_hydrated(7, None))
    assert cm.should_respond_to(7) is True


def test_real_cm_allows_when_overlay_should_respond_true() -> None:
    cm = _real_cm(_hydrated(7, ContactMembership(contact_id=7, should_respond=True)))
    assert cm.should_respond_to(7) is True


def test_real_cm_blocks_when_overlay_should_respond_false() -> None:
    """Explicit ``should_respond=False`` on the overlay blocks the send."""
    cm = _real_cm(_hydrated(7, ContactMembership(contact_id=7, should_respond=False)))
    assert cm.should_respond_to(7) is False


def test_real_cm_allows_when_contact_unknown() -> None:
    """Unknown ``contact_id`` (no shared row) defaults to allow.

    The gate's "Contact not found" branch covers the unknown-contact
    error path; the policy method itself stays permissive so callers
    can compose freely.
    """
    cm = _real_cm(None)
    assert cm.should_respond_to(7) is True


# ---------------------------------------------------------------------------
# Simulated ContactManager — policy reads from in-memory store
# ---------------------------------------------------------------------------


def test_simulated_cm_allows_when_should_respond_true() -> None:
    cm = _simulated_cm({7: {"contact_id": 7, "should_respond": True}})
    assert cm.should_respond_to(7) is True


def test_simulated_cm_blocks_when_should_respond_false() -> None:
    cm = _simulated_cm({7: {"contact_id": 7, "should_respond": False}})
    assert cm.should_respond_to(7) is False


def test_simulated_cm_allows_when_field_absent() -> None:
    cm = _simulated_cm({7: {"contact_id": 7, "first_name": "Carol"}})
    assert cm.should_respond_to(7) is True


def test_simulated_cm_allows_when_contact_unknown() -> None:
    cm = _simulated_cm({})
    assert cm.should_respond_to(7) is True
