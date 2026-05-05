"""Symbolic tests for browser-meet channel routing in `LivekitCallManager`.

These tests stub out every external dependency of ``_start_meet`` (LiveKit
room creation, agent-service HTTP join, the IPC socket server, and the call
subprocess) and verify only the channel-dispatch logic:

- The agent-service POST URL is derived from ``_MEET_PATHS[channel]["path"]``
  (``googlemeet`` for Google Meet, ``teamsmeet`` for Microsoft Teams).
- The LiveKit room name is derived from ``_MEET_PATHS[channel]["room"]``
  via ``make_room_name`` (``unity_<id>_gmeet`` vs ``unity_<id>_teams``).
- The active-channel state (``_call_channel``, ``has_active_google_meet``,
  ``has_active_teams_meet``) is set correctly and exclusively per channel.
- The session id returned by agent-service is captured into
  ``_meet_session_id`` and the joining flag is cleared on success.

No LLM or LiveKit calls are involved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from unity.conversation_manager.domains import call_manager as call_manager_module
from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
    make_room_name,
)

_ASSISTANT_ID = "42"
_MEET_URL = "https://example.test/meeting/abc"
_CONTACT = {"contact_id": 2, "first_name": "Alice", "is_system": False}
_BOSS = {"contact_id": 1, "first_name": "Boss", "is_system": True}


def _build_call_manager() -> LivekitCallManager:
    """Build a `LivekitCallManager` with empty config and no event broker."""
    cfg = CallConfig(
        assistant_id=_ASSISTANT_ID,
        user_id="user-1",
        assistant_bio="bio",
        assistant_number="+15555550000",
        voice_provider="elevenlabs",
        voice_id="voice-1",
        assistant_name="Assistant",
        job_name="job-1",
    )
    return LivekitCallManager(cfg, event_broker=None)


def _patch_meet_dependencies(
    monkeypatch,
    cm: LivekitCallManager,
    *,
    join_status: int = 200,
    session_id: str = "session-xyz",
):
    """Patch all external dependencies of ``_start_meet``.

    Returns a dict of capture buckets:
      * ``room_creates``: list of ``CreateRoomRequest`` payloads handed to LiveKit.
      * ``http_posts``: list of ``(url, json_body, headers)`` tuples for the
        agent-service POST.
      * ``subprocess_calls``: list of ``(room_name, channel, contact, boss,
        outbound, extra_env)`` tuples for the legacy subprocess path.
    """
    room_creates: list = []
    http_posts: list = []
    subprocess_calls: list = []

    fake_lk = MagicMock()
    fake_lk.aclose = AsyncMock()

    async def _fake_create_room(req):
        room_creates.append(req)

    fake_lk.room.create_room = _fake_create_room

    def _lk_factory(**_kwargs):
        return fake_lk

    monkeypatch.setattr(call_manager_module, "LiveKitAPI", _lk_factory)

    fake_resp = MagicMock()
    fake_resp.status = join_status
    fake_resp.json = AsyncMock(return_value={"sessionId": session_id})

    async def _fake_post(url, *, json=None, headers=None, timeout=None):
        http_posts.append((url, json, headers))
        return fake_resp

    fake_session = MagicMock()
    fake_session.post = _fake_post
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    def _session_factory(*_args, **_kwargs):
        return fake_session

    monkeypatch.setattr(call_manager_module.aiohttp, "ClientSession", _session_factory)

    async def _noop_ensure_socket():
        return None

    monkeypatch.setattr(cm, "_ensure_socket_server", _noop_ensure_socket)

    async def _capture_subprocess(
        room_name,
        channel,
        contact,
        boss,
        outbound,
        *,
        extra_env=None,
    ):
        subprocess_calls.append(
            (room_name, channel, contact, boss, outbound, extra_env),
        )

    monkeypatch.setattr(cm, "_start_call_subprocess", _capture_subprocess)

    cm._worker_proc = None
    cm._socket_server = None

    return {
        "room_creates": room_creates,
        "http_posts": http_posts,
        "subprocess_calls": subprocess_calls,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "expected_path", "expected_room_suffix"),
    [
        ("google_meet", "googlemeet", "gmeet"),
        ("teams_meet", "teamsmeet", "teams"),
    ],
)
async def test_start_meet_routes_per_channel(
    monkeypatch,
    channel: str,
    expected_path: str,
    expected_room_suffix: str,
):
    """`_start_meet(channel, ...)` must use the channel-specific URL path
    and LiveKit room suffix from `_MEET_PATHS`, capture the returned
    sessionId, and flip only the matching `has_active_*` property."""
    cm = _build_call_manager()
    captured = _patch_meet_dependencies(monkeypatch, cm)

    ok = await cm._start_meet(channel, _MEET_URL, _CONTACT, _BOSS)

    assert ok is True
    assert cm._call_channel == channel
    assert cm._meet_session_id == "session-xyz"
    assert cm._meet_joining is False
    assert cm._disconnect_contact == _CONTACT

    expected_room_name = make_room_name(_ASSISTANT_ID, expected_room_suffix)
    assert cm.room_name == expected_room_name

    assert len(captured["room_creates"]) == 1
    create_req = captured["room_creates"][0]
    assert create_req.name == expected_room_name
    assert create_req.empty_timeout >= 3600
    assert create_req.departure_timeout >= 3600

    assert len(captured["http_posts"]) == 1
    url, body, _headers = captured["http_posts"][0]
    assert url == f"http://localhost:3000/{expected_path}/join"
    assert body == {"meetUrl": _MEET_URL, "displayName": "Assistant"}

    assert len(captured["subprocess_calls"]) == 1
    sp_room, sp_channel, sp_contact, sp_boss, sp_outbound, sp_extra = captured[
        "subprocess_calls"
    ][0]
    assert sp_room == expected_room_name
    assert sp_channel == channel
    assert sp_contact == _CONTACT
    assert sp_boss == _BOSS
    assert sp_outbound is False
    assert sp_extra is not None
    assert sp_extra["meet_url"] == _MEET_URL
    assert sp_extra["meet_display_name"] == "Assistant"

    if channel == "teams_meet":
        assert cm.has_active_teams_meet is True
        assert cm.has_active_google_meet is False
    else:
        assert cm.has_active_google_meet is True
        assert cm.has_active_teams_meet is False
    assert cm.has_active_meet() is True


@pytest.mark.asyncio
async def test_start_teams_meet_wrapper_delegates_to_start_meet(monkeypatch):
    """The public `start_teams_meet` wrapper must forward to `_start_meet`
    with the ``teams_meet`` channel argument unchanged."""
    cm = _build_call_manager()

    seen: dict = {}

    async def _capture(channel, meet_url, contact, boss, display_name=""):
        seen["channel"] = channel
        seen["meet_url"] = meet_url
        seen["contact"] = contact
        seen["boss"] = boss
        seen["display_name"] = display_name
        return True

    monkeypatch.setattr(cm, "_start_meet", _capture)

    ok = await cm.start_teams_meet(
        _MEET_URL,
        _CONTACT,
        _BOSS,
        display_name="Custom Name",
    )

    assert ok is True
    assert seen == {
        "channel": "teams_meet",
        "meet_url": _MEET_URL,
        "contact": _CONTACT,
        "boss": _BOSS,
        "display_name": "Custom Name",
    }


@pytest.mark.asyncio
async def test_start_meet_join_failure_clears_state(monkeypatch):
    """When agent-service returns a non-200 status, ``_start_meet`` must
    clear the joining flag, leave no captured session id, and run cleanup
    so the per-channel ``has_active_*`` property goes back to False."""
    cm = _build_call_manager()
    captured = _patch_meet_dependencies(monkeypatch, cm, join_status=500)

    cleanup_calls: list = []

    async def _capture_cleanup(channel):
        cleanup_calls.append(channel)

    monkeypatch.setattr(cm, "_cleanup_meet", _capture_cleanup)

    ok = await cm._start_meet("teams_meet", _MEET_URL, _CONTACT, _BOSS)

    assert ok is False
    assert cm._meet_session_id is None
    assert cm._meet_joining is False
    assert cleanup_calls == ["teams_meet"]
    assert len(captured["http_posts"]) == 1
    assert captured["http_posts"][0][0].endswith("/teamsmeet/join")
