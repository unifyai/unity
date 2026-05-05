"""Unit tests for Teams message_id ingress dedup."""

from __future__ import annotations

import time

import pytest

from unity.conversation_manager import comms_manager as cm
from unity.conversation_manager.comms_manager import _already_seen_teams


@pytest.fixture(autouse=True)
def _clear_teams_dedup_cache():
    cm._seen_teams_ids.clear()
    yield
    cm._seen_teams_ids.clear()


def test_first_observation_is_fresh() -> None:
    assert _already_seen_teams("msg-1") is False


def test_duplicate_within_ttl_is_deduped() -> None:
    assert _already_seen_teams("msg-2") is False
    assert _already_seen_teams("msg-2") is True
    assert _already_seen_teams("msg-2") is True


def test_distinct_ids_are_independent() -> None:
    assert _already_seen_teams("a") is False
    assert _already_seen_teams("b") is False
    assert _already_seen_teams("a") is True
    assert _already_seen_teams("b") is True


def test_entries_expire_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    base = time.time()

    monkeypatch.setattr(cm.time, "time", lambda: base)
    assert _already_seen_teams("old") is False
    assert _already_seen_teams("old") is True

    monkeypatch.setattr(cm.time, "time", lambda: base + cm._TEAMS_DEDUP_TTL + 1)
    assert _already_seen_teams("old") is False
