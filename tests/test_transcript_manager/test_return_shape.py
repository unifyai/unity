from __future__ import annotations

import pytest

from tests.helpers import _handle_project


_EXPECTED_FWD = {
    "message_id": "mid",
    "medium": "med",
    "sender_id": "sid",
    "receiver_ids": "rids",
    "timestamp": "ts",
    "content": "c",
    "exchange_id": "xid",
    "images": "imgs",
}

_EXPECTED_INV = {v: k for k, v in _EXPECTED_FWD.items()}

_EXPECTED_KEYS_ORDER = [
    "contacts",
    "message_keys_to_shorthand",
    "messages",
    "shorthand_to_message_keys",
]


@pytest.mark.unit
@_handle_project
def test_filter_messages_return_shape(tm_manager_scenario):
    tm, _ = tm_manager_scenario

    out = tm._filter_messages(limit=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings
    assert out["message_keys_to_shorthand"] == _EXPECTED_FWD
    assert out["shorthand_to_message_keys"] == _EXPECTED_INV

    # Types
    assert isinstance(out["contacts"], list)
    assert isinstance(out["messages"], list)


@pytest.mark.unit
@_handle_project
def test_search_messages_return_shape(tm_manager_scenario):
    tm, _ = tm_manager_scenario

    # references=None path returns latest messages directly
    out = tm._search_messages(references=None, k=1)

    # Key order
    assert list(out.keys()) == _EXPECTED_KEYS_ORDER

    # Legend mappings
    assert out["message_keys_to_shorthand"] == _EXPECTED_FWD
    assert out["shorthand_to_message_keys"] == _EXPECTED_INV

    # Types
    assert isinstance(out["contacts"], list)
    assert isinstance(out["messages"], list)
