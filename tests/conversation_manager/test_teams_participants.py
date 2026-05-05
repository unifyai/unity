"""Unit tests for Teams participants resolution at ingress."""

from __future__ import annotations

import pytest

from unity.conversation_manager.comms_manager import _resolve_teams_participants
from unity.conversation_manager.events import (
    TeamsChannelMessageReceived,
    TeamsChannelMessageSent,
    TeamsMessageReceived,
    TeamsMessageSent,
)

TEST_SELF_CONTACT_ID = 337


def _fake_resolver_factory(mapping: dict[str, int]):
    """Build a resolver that returns a contact dict for mapped emails, None otherwise.

    Also records the exact calls made so tests can assert no redundant work.
    """
    calls: list[tuple[str, str]] = []

    def _resolver(medium: str, email: str):
        calls.append((medium, email))
        cid = mapping.get(email)
        if cid is None:
            return None
        return {"contact_id": cid, "email_address": email}

    _resolver.calls = calls  # type: ignore[attr-defined]
    return _resolver


def test_participants_all_pre_resolved() -> None:
    resolver = _fake_resolver_factory({})
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": TEST_SELF_CONTACT_ID, "email": "assistant@acme.com"},
            {"contact_id": 7, "email": "alice@acme.com"},
            {"contact_id": 12, "email": "carol@acme.com"},
        ],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == sorted([TEST_SELF_CONTACT_ID, 7, 12])
    assert resolver.calls == []


def test_unresolved_entry_routed_through_resolver() -> None:
    resolver = _fake_resolver_factory({"bob@ext.com": 42})
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": TEST_SELF_CONTACT_ID, "email": "assistant@acme.com"},
            {"contact_id": 7, "email": "alice@acme.com"},
            {"contact_id": None, "email": "bob@ext.com"},
        ],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_channel_message",
        unknown_contact_resolver=resolver,
    )
    assert result == sorted([TEST_SELF_CONTACT_ID, 7, 42])
    assert resolver.calls == [("teams_channel_message", "bob@ext.com")]


def test_synthetic_teams_email_is_dropped() -> None:
    resolver = _fake_resolver_factory({})
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": TEST_SELF_CONTACT_ID, "email": "assistant@acme.com"},
            {"contact_id": None, "email": "abc-123@teams"},
            {"contact_id": None, "email": None},
        ],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == sorted([TEST_SELF_CONTACT_ID, 7])
    assert resolver.calls == []


def test_sender_entry_skipped_no_redundant_resolver_call() -> None:
    resolver = _fake_resolver_factory({"alice@acme.com": 7})
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": None, "email": "alice@acme.com"},
            {"contact_id": 12, "email": "carol@acme.com"},
        ],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == [7, 12]
    assert resolver.calls == []


def test_resolver_returning_none_entry_dropped() -> None:
    resolver = _fake_resolver_factory({})  # always returns None
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": TEST_SELF_CONTACT_ID, "email": "assistant@acme.com"},
            {"contact_id": None, "email": "stranger@ext.com"},
        ],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == sorted([TEST_SELF_CONTACT_ID, 7])
    assert resolver.calls == [("teams_message", "stranger@ext.com")]


def test_sender_with_no_contact_id_still_resolves_others() -> None:
    resolver = _fake_resolver_factory({})
    result = _resolve_teams_participants(
        raw_participants=[
            {"contact_id": TEST_SELF_CONTACT_ID, "email": "assistant@acme.com"},
            {"contact_id": 12, "email": "carol@acme.com"},
        ],
        sender_email="",
        sender_contact_id=None,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == sorted([TEST_SELF_CONTACT_ID, 12])


def test_empty_participants_returns_only_sender() -> None:
    resolver = _fake_resolver_factory({})
    result = _resolve_teams_participants(
        raw_participants=[],
        sender_email="alice@acme.com",
        sender_contact_id=7,
        medium="teams_message",
        unknown_contact_resolver=resolver,
    )
    assert result == [7]


# --- event dataclass tests ---------------------------------------------- #


@pytest.mark.parametrize(
    "cls,extra",
    [
        (TeamsMessageReceived, {"content": "hi"}),
        (TeamsChannelMessageReceived, {"content": "hi"}),
        (TeamsMessageSent, {"content": "hi"}),
        (TeamsChannelMessageSent, {"content": "hi"}),
    ],
)
def test_teams_event_participants_field_defaults_empty(cls, extra) -> None:
    event = cls(contact={"contact_id": 7}, **extra)
    assert event.participants == []


@pytest.mark.parametrize(
    "cls,extra",
    [
        (TeamsMessageReceived, {"content": "hi"}),
        (TeamsChannelMessageReceived, {"content": "hi"}),
        (TeamsMessageSent, {"content": "hi"}),
        (TeamsChannelMessageSent, {"content": "hi"}),
    ],
)
def test_teams_event_participants_roundtrips_through_to_dict(cls, extra) -> None:
    event = cls(
        contact={"contact_id": 7},
        participants=[TEST_SELF_CONTACT_ID, 7, 12],
        **extra,
    )
    payload = event.to_dict()["payload"]
    assert payload["participants"] == [TEST_SELF_CONTACT_ID, 7, 12]
    restored = cls.from_dict({"event_name": cls.__name__, "payload": payload})
    assert restored.participants == [TEST_SELF_CONTACT_ID, 7, 12]


def test_teams_event_deserializes_legacy_payload_without_participants() -> None:
    """Old persisted events predate the `participants` field; default to []."""
    payload = {
        "contact": {"contact_id": 7},
        "content": "hi",
        "chat_id": "abc",
        "message_id": "m1",
        "chat_type": None,
        "chat_topic": None,
        "attachments": [],
        "timestamp": "2026-04-20T12:00:00",
    }
    restored = TeamsMessageReceived.from_dict(
        {"event_name": "TeamsMessageReceived", "payload": payload},
    )
    assert restored.participants == []
