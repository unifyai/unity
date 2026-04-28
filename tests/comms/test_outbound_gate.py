"""Outbound-comms response-policy gate.

``CommsPrimitives._check_outbound_allowed`` is the single chokepoint
through which every assistant-owned outbound communication flows. The
gate must:

- block sends to contacts the assistant has been told not to respond
  to (``ContactMembership.should_respond == False``);
- allow sends when ``ContactManager.should_respond_to`` returns
  ``True``;
- reject sends to unknown contacts with a "Contact not found" reason.

Policy is fetched from
:meth:`ContactManager.should_respond_to`, so these tests stub that
method to express each contract clearly.
"""

from __future__ import annotations

from typing import Any

import pytest

from unity.comms.primitives import CommsPrimitives


class _StubContactManager:
    """Minimal ContactManager stand-in that returns a fixed verdict."""

    def __init__(self, verdict: bool) -> None:
        self._verdict = verdict
        self.calls: list[int] = []

    def should_respond_to(self, contact_id: int) -> bool:
        self.calls.append(contact_id)
        return self._verdict


def _gate_with(verdict: bool) -> tuple[CommsPrimitives, _StubContactManager]:
    gate = CommsPrimitives.__new__(CommsPrimitives)
    cm = _StubContactManager(verdict)
    gate._contact_manager = lambda: cm  # type: ignore[method-assign]
    return gate, cm


def _make_contact(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"first_name": "Alex", "surname": "Doe"}
    base.update(overrides)
    return base


def test_allows_when_policy_says_yes() -> None:
    gate, cm = _gate_with(verdict=True)
    assert gate._check_outbound_allowed(_make_contact(), contact_id=7) is None
    assert cm.calls == [7]


def test_blocks_when_policy_says_no() -> None:
    """Explicit ``should_respond=False`` on the overlay must block."""
    gate, cm = _gate_with(verdict=False)
    reason = gate._check_outbound_allowed(_make_contact(), contact_id=7)
    assert reason is not None
    assert "Alex Doe" in reason
    assert "should_respond is False" in reason
    assert cm.calls == [7]


def test_ignores_should_respond_in_shared_row_dict() -> None:
    """The gate ignores any ``should_respond`` on the shared-row dict;
    only the ContactManager policy decides.
    """
    gate, cm = _gate_with(verdict=False)
    contact = _make_contact(should_respond=True)
    assert gate._check_outbound_allowed(contact, contact_id=7) is not None
    assert cm.calls == [7]


@pytest.mark.parametrize("contact", [None, {}])
def test_rejects_unknown_contact(contact: dict | None) -> None:
    """Unknown contact short-circuits before the policy lookup."""
    gate, cm = _gate_with(verdict=True)
    assert gate._check_outbound_allowed(contact, contact_id=7) == "Contact not found"
    assert cm.calls == []
