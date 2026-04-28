"""Regression tests for the outbound-comms ``should_respond`` gate.

``CommsPrimitives._check_outbound_allowed`` is the single chokepoint that
decides whether an assistant-owned send (SMS / email / WhatsApp / Unify
message / call / Discord / Teams) is permitted for a given contact. The
policy field is ``ContactMembership.should_respond``, which lives on the
per-body overlay and defaults to ``True`` on the model.

Because many call sites resolve the contact through a shared-row read
that does not compose the overlay, the dict the gate sees often does
*not* carry ``should_respond`` at all. Treat that absence as "no
per-body policy override" and allow the send; only an explicit
``False`` blocks.
"""

from __future__ import annotations

from unity.comms.primitives import CommsPrimitives


def _make_gate() -> CommsPrimitives:
    return CommsPrimitives()


def test_allows_when_should_respond_key_absent() -> None:
    """Shared-row reads omit ``should_respond``; the gate must default-allow.

    Reproduces the production regression where every outbound was
    blocked because the gate defaulted a missing key to ``False``.
    """
    gate = _make_gate()
    contact = {"contact_id": 7, "first_name": "Yusha", "surname": "Arif"}

    assert gate._check_outbound_allowed(contact) is None


def test_allows_when_should_respond_is_true() -> None:
    gate = _make_gate()
    contact = {
        "contact_id": 7,
        "first_name": "Yusha",
        "surname": "Arif",
        "should_respond": True,
    }

    assert gate._check_outbound_allowed(contact) is None


def test_blocks_when_should_respond_is_false() -> None:
    """Explicit ``should_respond=False`` is honored when the dict carries it."""
    gate = _make_gate()
    contact = {
        "contact_id": 7,
        "first_name": "Yusha",
        "surname": "Arif",
        "should_respond": False,
    }

    reason = gate._check_outbound_allowed(contact)

    assert reason is not None
    assert "Yusha Arif" in reason
    assert "should_respond is False" in reason


def test_rejects_unknown_contact() -> None:
    gate = _make_gate()

    assert gate._check_outbound_allowed(None) == "Contact not found"
    assert gate._check_outbound_allowed({}) == "Contact not found"
