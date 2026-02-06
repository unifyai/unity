"""
Tests for TranscriptManager return shape consistency.

NOTE: These tests use the tm_manager_scenario fixture which provides
pre-seeded data. They should NOT use @_handle_project as that would
conflict with the scenario's context management.
"""

from __future__ import annotations

from unity.contact_manager.types.contact import Contact

_EXPECTED_FWD = {
    "message_id": "mid",
    "medium": "med",
    "sender_id": "sid",
    "receiver_ids": "rids",
    "timestamp": "ts",
    "content": "c",
    "exchange_id": "xid",
    "images": "imgs",
    "attachments": "atts",
}

_EXPECTED_INV = {v: k for k, v in _EXPECTED_FWD.items()}

# Contact shorthand legends (derived from Contact.SHORTHAND_MAP)
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
