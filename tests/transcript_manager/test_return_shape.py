"""
Tests for TranscriptManager return shape consistency.

NOTE: These tests use the tm_manager_scenario fixture which provides
pre-seeded data. They should NOT use @_handle_project as that would
conflict with the scenario's context management.
"""

from __future__ import annotations

from unity.contact_manager.types.contact import Contact
from unity.transcript_manager.types.message import Message

# Pull the shorthand legends straight from the models so the assertions stay
# locked to the source of truth. Hard-coding them here only invites drift
# every time a new message or contact field lands.
_EXPECTED_FWD = Message.shorthand_map()
_EXPECTED_INV = Message.shorthand_inverse_map()
_CONTACT_FWD = Contact.shorthand_map()
_CONTACT_INV = Contact.shorthand_inverse_map()

_EXPECTED_KEYS_ORDER = [
    "contact_keys_to_shorthand",
    "contacts",
    "shorthand_to_contact_keys",
    "message_keys_to_shorthand",
    "messages",
    "shorthand_to_message_keys",
]


def test_filter_return_shape(tm_manager_scenario):
    tm, _ = tm_manager_scenario

    out = tm._filter_messages(limit=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings (contacts + messages)
    assert out["contact_keys_to_shorthand"] == _CONTACT_FWD
    assert out["shorthand_to_contact_keys"] == _CONTACT_INV
    assert out["message_keys_to_shorthand"] == _EXPECTED_FWD
    assert out["shorthand_to_message_keys"] == _EXPECTED_INV

    # Types
    assert isinstance(out["contacts"], list)
    assert isinstance(out["messages"], list)


def test_search_return_shape(tm_manager_scenario):
    tm, _ = tm_manager_scenario

    # references=None path returns latest messages directly
    out = tm._search_messages(references=None, k=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings (contacts + messages)
    assert out["contact_keys_to_shorthand"] == _CONTACT_FWD
    assert out["shorthand_to_contact_keys"] == _CONTACT_INV
    assert out["message_keys_to_shorthand"] == _EXPECTED_FWD
    assert out["shorthand_to_message_keys"] == _EXPECTED_INV

    # Types
    assert isinstance(out["contacts"], list)
    assert isinstance(out["messages"], list)
