"""Symbolic tests for Teams chat/channel creation in CommsPrimitives.

These tests stub out the HTTP transport (`comms_utils.*`) and verify only the
primitive-side dispatch and validation logic:

- `send_teams_message` correctly routes to find-or-create when chat_id is
  omitted, producing `oneOnOne` for a single recipient and `group` for
  multiple, and rejecting invalid combinations.
- `create_teams_channel` calls the correct transport helper and validates
  `membership_type` + `owner_contact_ids`.

No LLM calls are involved — these tests lock in the symbolic contract.
"""

from unittest.mock import AsyncMock

import pytest

from unity.comms import offline_support
from unity.comms.primitives import CommsPrimitives
from unity.conversation_manager.domains import comms_utils


def _make_comms_with_teams(monkeypatch) -> CommsPrimitives:
    """Build a CommsPrimitives instance wired up with Teams enabled."""
    monkeypatch.setattr(
        "unity.comms.primitives.SESSION_DETAILS.assistant.contact_id",
        0,
    )
    comms = CommsPrimitives()
    monkeypatch.setattr(comms, "_assistant_has_teams", lambda: True)
    monkeypatch.setattr(
        "unity.comms.primitives.reserve_outbound_operation",
        lambda **kwargs: offline_support.OfflineOutboundDecision(
            reservation=None,
            response=None,
        ),
    )
    comms._event_broker.publish = AsyncMock()
    return comms


def _contact_fixture() -> dict:
    return {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "email_address": "alice@example.com",
        "should_respond": True,
    }


@pytest.mark.anyio
async def test_send_teams_message_find_or_create_dm(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    comms._get_contact = lambda **kwargs: dict(_contact_fixture())

    create_calls: list[dict] = []
    send_calls: list[dict] = []

    async def _fake_create_chat(*, chat_type, member_emails, topic):
        create_calls.append(
            {"chat_type": chat_type, "members": member_emails, "topic": topic},
        )
        return {"success": True, "chat_id": "19:chat-dm", "chat_type": chat_type}

    async def _fake_send(*, chat_id, team_id, channel_id, body, attachments):
        send_calls.append(
            {
                "chat_id": chat_id,
                "team_id": team_id,
                "channel_id": channel_id,
                "body": body,
            },
        )
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr(comms_utils, "create_teams_chat", _fake_create_chat)
    monkeypatch.setattr(comms_utils, "send_teams_message", _fake_send)

    result = await comms.send_teams_message(contact_id=5, content="Hi there")

    assert result == {"status": "ok"}
    assert create_calls == [
        {"chat_type": "oneOnOne", "members": ["alice@example.com"], "topic": None},
    ]
    assert send_calls[0]["chat_id"] == "19:chat-dm"
    assert send_calls[0]["body"] == "Hi there"


@pytest.mark.anyio
async def test_send_teams_message_find_or_create_group_with_topic(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    roster = {
        5: {
            "contact_id": 5,
            "first_name": "Alice",
            "surname": "Owner",
            "email_address": "alice@example.com",
            "should_respond": True,
        },
        6: {
            "contact_id": 6,
            "first_name": "Bob",
            "surname": "Builder",
            "email_address": "bob@example.com",
            "should_respond": True,
        },
    }
    comms._get_contact = lambda **kwargs: (
        dict(roster[kwargs["contact_id"]])
        if kwargs.get("contact_id") in roster
        else None
    )

    create_calls: list[dict] = []

    async def _fake_create_chat(*, chat_type, member_emails, topic):
        create_calls.append(
            {"chat_type": chat_type, "members": member_emails, "topic": topic},
        )
        return {"success": True, "chat_id": "19:chat-group", "chat_type": chat_type}

    async def _fake_send(**kwargs):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr(comms_utils, "create_teams_chat", _fake_create_chat)
    monkeypatch.setattr(comms_utils, "send_teams_message", _fake_send)

    result = await comms.send_teams_message(
        contact_id=[5, 6],
        content="Hi team",
        chat_topic="Planning",
    )

    assert result == {"status": "ok"}
    assert create_calls == [
        {
            "chat_type": "group",
            "members": ["alice@example.com", "bob@example.com"],
            "topic": "Planning",
        },
    ]


@pytest.mark.anyio
async def test_send_teams_message_chat_reply_rejects_multiple_contacts(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    comms._get_contact = lambda **kwargs: dict(_contact_fixture())

    create_mock = AsyncMock()
    send_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_chat", create_mock)
    monkeypatch.setattr(comms_utils, "send_teams_message", send_mock)

    result = await comms.send_teams_message(
        contact_id=[5, 6],
        content="Hi",
        chat_id="19:existing",
    )

    assert result["status"] == "error"
    assert "Multiple contact_ids" in result["error"]
    create_mock.assert_not_awaited()
    send_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_teams_message_rejects_topic_for_chat_reply(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    comms._get_contact = lambda **kwargs: dict(_contact_fixture())

    create_mock = AsyncMock()
    send_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_chat", create_mock)
    monkeypatch.setattr(comms_utils, "send_teams_message", send_mock)

    result = await comms.send_teams_message(
        contact_id=5,
        content="Hi",
        chat_id="19:existing",
        chat_topic="Nope",
    )

    assert result["status"] == "error"
    assert "chat_topic" in result["error"]
    create_mock.assert_not_awaited()
    send_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_teams_message_existing_chat_reply_skips_create(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    comms._get_contact = lambda **kwargs: dict(_contact_fixture())

    create_mock = AsyncMock()
    send_calls: list[dict] = []

    async def _fake_send(*, chat_id, team_id, channel_id, body, attachments):
        send_calls.append({"chat_id": chat_id, "team_id": team_id})
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr(comms_utils, "create_teams_chat", create_mock)
    monkeypatch.setattr(comms_utils, "send_teams_message", _fake_send)

    result = await comms.send_teams_message(
        contact_id=5,
        content="Hi",
        chat_id="19:existing",
    )

    assert result == {"status": "ok"}
    create_mock.assert_not_awaited()
    assert send_calls[0]["chat_id"] == "19:existing"


@pytest.mark.anyio
async def test_create_teams_channel_standard_success(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    monkeypatch.setattr(
        comms,
        "_assistant_anchor_contact",
        lambda: {"contact_id": 0, "first_name": "A", "surname": "B"},
    )

    create_calls: list[dict] = []

    async def _fake_create_channel(
        *,
        team_id,
        display_name,
        description,
        membership_type,
        owner_emails,
    ):
        create_calls.append(
            {
                "team_id": team_id,
                "display_name": display_name,
                "description": description,
                "membership_type": membership_type,
                "owner_emails": owner_emails,
            },
        )
        return {"success": True, "channel_id": "19:channel-1", "team_id": team_id}

    monkeypatch.setattr(comms_utils, "create_teams_channel", _fake_create_channel)

    result = await comms.create_teams_channel(
        team_id="team-42",
        display_name="New Channel",
        description="Launch planning",
    )

    assert result == {
        "status": "ok",
        "team_id": "team-42",
        "channel_id": "19:channel-1",
    }
    assert create_calls == [
        {
            "team_id": "team-42",
            "display_name": "New Channel",
            "description": "Launch planning",
            "membership_type": "standard",
            "owner_emails": None,
        },
    ]


@pytest.mark.anyio
async def test_create_teams_channel_private_requires_owners(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    monkeypatch.setattr(
        comms,
        "_assistant_anchor_contact",
        lambda: {"contact_id": 0, "first_name": "A", "surname": "B"},
    )

    create_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_channel", create_mock)

    result = await comms.create_teams_channel(
        team_id="team-42",
        display_name="Secret Channel",
        membership_type="private",
    )

    assert result["status"] == "error"
    assert "private" in result["error"] and "owner" in result["error"]
    create_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_channel_private_with_owner_contact_id(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    monkeypatch.setattr(
        comms,
        "_assistant_anchor_contact",
        lambda: {"contact_id": 0, "first_name": "A", "surname": "B"},
    )
    comms._get_contact = lambda **kwargs: dict(_contact_fixture())

    async def _fake_create_channel(*, owner_emails, **kwargs):
        return {
            "success": True,
            "channel_id": "19:priv-1",
            "team_id": kwargs["team_id"],
        }

    captured: list[dict] = []

    async def _wrapper(**kwargs):
        captured.append(dict(kwargs))
        return await _fake_create_channel(**kwargs)

    monkeypatch.setattr(comms_utils, "create_teams_channel", _wrapper)

    result = await comms.create_teams_channel(
        team_id="team-42",
        display_name="Secret",
        membership_type="private",
        owner_contact_ids=[5],
    )

    assert result["status"] == "ok"
    assert captured[0]["owner_emails"] == ["alice@example.com"]
    assert captured[0]["membership_type"] == "private"


@pytest.mark.anyio
async def test_create_teams_channel_rejects_invalid_membership_type(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    monkeypatch.setattr(
        comms,
        "_assistant_anchor_contact",
        lambda: {"contact_id": 0, "first_name": "A", "surname": "B"},
    )

    create_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_channel", create_mock)

    result = await comms.create_teams_channel(
        team_id="team-42",
        display_name="Broken",
        membership_type="publicish",
    )

    assert result["status"] == "error"
    assert "membership_type" in result["error"]
    create_mock.assert_not_awaited()
