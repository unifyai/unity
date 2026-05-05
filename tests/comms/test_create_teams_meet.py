"""Symbolic tests for `CommsPrimitives.create_teams_meet`.

These tests stub out the HTTP transport (`comms_utils.create_teams_meet`) and
verify only the primitive-side dispatch and validation logic:

- Routing of ``mode="instant"`` and ``mode="scheduled"`` payloads.
- Default ``start``/``end`` computation for scheduled mode.
- Attendee resolution from contact ids → email addresses with deduping.
- Validation of ``mode`` value, missing ``subject``, malformed ``start``,
  attendees missing email addresses.
- Capability gating when Teams is disabled for the assistant.
- Surface of the underlying transport's success / error envelope.
- Publication of the ``app:comms:teams_meet_created`` event on success.

No LLM calls are involved.
"""

from datetime import datetime, timedelta, timezone
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
        comms,
        "_assistant_anchor_contact",
        lambda: {"contact_id": 0, "first_name": "A", "surname": "B"},
    )
    monkeypatch.setattr(
        "unity.comms.primitives.reserve_outbound_operation",
        lambda **kwargs: offline_support.OfflineOutboundDecision(
            reservation=None,
            response=None,
        ),
    )
    comms._event_broker.publish = AsyncMock()
    return comms


def _alice() -> dict:
    return {
        "contact_id": 5,
        "first_name": "Alice",
        "surname": "Owner",
        "email_address": "alice@example.com",
        "should_respond": True,
    }


def _bob() -> dict:
    return {
        "contact_id": 6,
        "first_name": "Bob",
        "surname": "Builder",
        "email_address": "bob@example.com",
        "should_respond": True,
    }


@pytest.mark.anyio
async def test_create_teams_meet_instant_success(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/AAA",
            "meeting_id": "meet-instant-1",
            "event_id": "",
            "subject": "",
            "start": "",
            "end": "",
            "web_link": "",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    result = await comms.create_teams_meet(mode="instant")

    assert result["status"] == "ok"
    assert result["mode"] == "instant"
    assert result["join_web_url"] == "https://teams.microsoft.com/l/meetup-join/AAA"
    assert result["meeting_id"] == "meet-instant-1"
    assert result["event_id"] == ""
    assert captured[0]["mode"] == "instant"
    assert captured[0]["subject"] is None
    assert captured[0]["start"] is None
    assert captured[0]["end"] is None
    assert captured[0]["attendees"] is None
    assert captured[0]["body_html"] is None

    comms._event_broker.publish.assert_awaited_once()
    topic, payload = comms._event_broker.publish.await_args.args
    assert topic == "app:comms:teams_meet_created"
    assert '"mode": "instant"' in payload
    assert '"meeting_id": "meet-instant-1"' in payload


@pytest.mark.anyio
async def test_create_teams_meet_instant_ignores_attendees(monkeypatch):
    """Instant mode should not resolve attendees (Graph ignores them)."""
    comms = _make_comms_with_teams(monkeypatch)

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/B",
            "meeting_id": "m-2",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    get_calls: list[dict] = []

    def _get(**kwargs):
        get_calls.append(kwargs)
        return _alice()

    comms._get_contact = _get

    result = await comms.create_teams_meet(
        mode="instant",
        attendee_contact_ids=[5, 6],
    )

    assert result["status"] == "ok"
    assert captured[0]["attendees"] is None
    assert get_calls == []


@pytest.mark.anyio
async def test_create_teams_meet_defaults_to_scheduled(monkeypatch):
    """A bare call with only a subject defaults to scheduled mode."""
    comms = _make_comms_with_teams(monkeypatch)

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/DEF",
            "meeting_id": "",
            "event_id": "evt-default",
            "subject": kwargs["subject"],
            "start": kwargs["start"],
            "end": kwargs["end"],
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    before = datetime.now(timezone.utc)
    result = await comms.create_teams_meet(subject="Standup")
    after = datetime.now(timezone.utc)

    assert result["status"] == "ok"
    assert result["mode"] == "scheduled"
    assert result["subject"] == "Standup"
    assert result["event_id"] == "evt-default"
    assert captured[0]["mode"] == "scheduled"
    sent_start = datetime.fromisoformat(captured[0]["start"])
    sent_end = datetime.fromisoformat(captured[0]["end"])
    assert before + timedelta(minutes=4) <= sent_start <= after + timedelta(minutes=6)
    assert sent_end - sent_start == timedelta(minutes=30)


@pytest.mark.anyio
async def test_create_teams_meet_no_args_requires_subject(monkeypatch):
    """With scheduled as the default, a no-arg call surfaces the missing-subject error."""
    comms = _make_comms_with_teams(monkeypatch)
    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet()

    assert result["status"] == "error"
    assert "subject" in result["error"]
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_success_with_attendees(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    roster = {5: _alice(), 6: _bob()}
    comms._get_contact = lambda **kwargs: (
        dict(roster[kwargs["contact_id"]])
        if kwargs.get("contact_id") in roster
        else None
    )

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/SCHED",
            "meeting_id": "",
            "event_id": "evt-99",
            "subject": kwargs["subject"],
            "start": kwargs["start"],
            "end": kwargs["end"],
            "web_link": "https://outlook.office.com/calendar/item/evt-99",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    start = "2026-05-01T15:00:00+00:00"
    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Q3 planning",
        start=start,
        duration_minutes=45,
        attendee_contact_ids=[5, 6],
        body_html="<p>Agenda</p>",
        location="HQ",
    )

    assert result["status"] == "ok"
    assert result["mode"] == "scheduled"
    assert result["event_id"] == "evt-99"
    assert result["meeting_id"] == ""
    assert result["start"] == start
    expected_end = (datetime.fromisoformat(start) + timedelta(minutes=45)).isoformat()
    assert result["end"] == expected_end
    assert result["web_link"].endswith("/evt-99")

    sent = captured[0]
    assert sent["mode"] == "scheduled"
    assert sent["subject"] == "Q3 planning"
    assert sent["start"] == start
    assert sent["end"] == expected_end
    assert sent["attendees"] == ["alice@example.com", "bob@example.com"]
    assert sent["body_html"] == "<p>Agenda</p>"
    assert sent["location"] == "HQ"

    comms._event_broker.publish.assert_awaited_once()


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_defaults_start_to_future(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/X",
            "event_id": "evt-1",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    before = datetime.now(timezone.utc)
    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Standup",
    )
    after = datetime.now(timezone.utc)

    assert result["status"] == "ok"
    sent_start = datetime.fromisoformat(captured[0]["start"])
    sent_end = datetime.fromisoformat(captured[0]["end"])
    assert before + timedelta(minutes=4) <= sent_start <= after + timedelta(minutes=6)
    assert sent_end - sent_start == timedelta(minutes=30)


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_dedupes_attendee_emails(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    roster = {5: _alice(), 7: dict(_alice(), contact_id=7)}
    comms._get_contact = lambda **kwargs: (
        dict(roster[kwargs["contact_id"]])
        if kwargs.get("contact_id") in roster
        else None
    )

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/D",
            "event_id": "evt-d",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Sync",
        start="2026-05-01T15:00:00+00:00",
        attendee_contact_ids=[5, 7],
    )

    assert result["status"] == "ok"
    assert captured[0]["attendees"] == ["alice@example.com"]


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_attendee_inline_email(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    contact_no_email = {
        "contact_id": 9,
        "first_name": "Carol",
        "surname": "Newby",
        "email_address": None,
        "should_respond": True,
    }
    # Mutable roster so a refetch after `update_contact` observes the inline
    # email attachment (mirrors how the real contact manager behaves).
    roster: dict[int, dict] = {9: dict(contact_no_email)}
    comms._get_contact = lambda **kwargs: (
        dict(roster[kwargs["contact_id"]])
        if kwargs.get("contact_id") in roster
        else None
    )
    comms._find_conflicting_contact = lambda **kwargs: None

    class _CM:
        def update_contact(self, **kw):
            cid = kw.pop("contact_id")
            roster[cid].update({k: v for k, v in kw.items() if v is not None})

    comms._contact_manager = lambda: _CM()

    captured: list[dict] = []

    async def _fake_transport(**kwargs):
        captured.append(dict(kwargs))
        return {
            "success": True,
            "join_web_url": "https://teams.microsoft.com/l/meetup-join/E",
            "event_id": "evt-e",
        }

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Onboarding",
        start="2026-05-01T15:00:00+00:00",
        attendee_contact_ids=[
            {"contact_id": 9, "email_address": "carol@example.com"},
        ],
    )

    assert result["status"] == "ok"
    assert captured[0]["attendees"] == ["carol@example.com"]


@pytest.mark.anyio
async def test_create_teams_meet_rejects_invalid_mode(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet(mode="recurring")

    assert result["status"] == "error"
    assert "mode" in result["error"]
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_requires_subject(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet(mode="scheduled")

    assert result["status"] == "error"
    assert "subject" in result["error"]
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_rejects_bad_start(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Sync",
        start="not-a-date",
    )

    assert result["status"] == "error"
    assert "ISO-8601" in result["error"]
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_scheduled_attendee_missing_email(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    contact_no_email = {
        "contact_id": 9,
        "first_name": "Carol",
        "surname": "Newby",
        "email_address": None,
        "should_respond": True,
    }
    comms._get_contact = lambda **kwargs: dict(contact_no_email)

    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Onboarding",
        start="2026-05-01T15:00:00+00:00",
        attendee_contact_ids=[9],
    )

    assert result["status"] == "error"
    assert "email" in result["error"].lower()
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_no_teams_returns_error(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)
    monkeypatch.setattr(comms, "_assistant_has_teams", lambda: False)

    transport_mock = AsyncMock()
    monkeypatch.setattr(comms_utils, "create_teams_meet", transport_mock)

    result = await comms.create_teams_meet()

    assert result["status"] == "error"
    assert "Teams" in result["error"]
    transport_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_create_teams_meet_surfaces_transport_error(monkeypatch):
    comms = _make_comms_with_teams(monkeypatch)

    async def _fake_transport(**kwargs):
        return {"success": False, "error": "HTTP 409: scope missing"}

    monkeypatch.setattr(comms_utils, "create_teams_meet", _fake_transport)

    result = await comms.create_teams_meet(
        mode="scheduled",
        subject="Sync",
        start="2026-05-01T15:00:00+00:00",
    )

    assert result["status"] == "error"
    assert "scope missing" in result["error"]
    comms._event_broker.publish.assert_awaited()
    topic_arg, _payload = comms._event_broker.publish.await_args.args
    assert topic_arg == "app:comms:teams_meet_created"
