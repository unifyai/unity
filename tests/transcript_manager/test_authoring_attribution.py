from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
import unify

from tests.helpers import _handle_project
from unity.common.colleague_cache import ColleagueNameCache
from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.transcript_manager import TranscriptManager


def _space_id() -> int:
    return int(time.time_ns() % 1_000_000_000)


def _message_payload(content: str, *, exchange_id: int) -> dict:
    return {
        "medium": "email",
        "sender_id": 0,
        "receiver_ids": [1],
        "timestamp": datetime.now(UTC),
        "content": content,
        "exchange_id": exchange_id,
    }


def _delete_context_tree(root: str) -> None:
    try:
        children = list(unify.get_contexts(prefix=f"{root}/").keys())
    except Exception:
        children = []
    for context in sorted(children, key=len, reverse=True):
        try:
            unify.delete_context(context)
        except Exception:
            pass
    try:
        unify.delete_context(root)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_session_details():
    SESSION_DETAILS.reset()
    yield
    SESSION_DETAILS.reset()


@_handle_project
def test_shared_authoring_attribution_enriches_messages_and_reuses_cache(monkeypatch):
    space_id = _space_id()
    SESSION_DETAILS.assistant.agent_id = 684
    SESSION_DETAILS.assistant.first_name = "Avery"
    SESSION_DETAILS.assistant.surname = "Ops"
    SESSION_DETAILS.user.id = "boss-user"
    SESSION_DETAILS.org_id = 77
    SESSION_DETAILS.unify_key = "owner-key"
    SESSION_DETAILS.space_ids = [space_id]
    manager = TranscriptManager()

    try:
        [own_message] = manager.log_messages(
            _message_payload(
                "self-authored shared transcript marker",
                exchange_id=1101,
            ),
            synchronous=True,
            destination=f"space:{space_id}",
        )

        SESSION_DETAILS.assistant.agent_id = 90210
        [colleague_message] = manager.log_messages(
            _message_payload(
                "colleague-authored shared transcript marker",
                exchange_id=1102,
            ),
            synchronous=True,
            destination=f"space:{space_id}",
        )
        SESSION_DETAILS.assistant.agent_id = 684

        list_calls: list[dict] = []

        def fake_list_assistants(**kwargs):
            list_calls.append(kwargs)
            return [
                {
                    "agent_id": 90210,
                    "first_name": "Sarah",
                    "surname": "Chen",
                },
            ]

        monkeypatch.setattr(
            "unity.common.colleague_cache.unify.list_assistants",
            fake_list_assistants,
        )

        first = manager._filter_messages(limit=20)
        second = manager._filter_messages(limit=20)
        semantic = manager._search_messages(
            references={"content": "colleague-authored shared transcript marker"},
            k=2,
        )

        assert "message_authoring_attribution" in first
        assert "message_authoring_attribution" in semantic
        by_message_id = {
            row["message_id"]: row for row in first["message_authoring_attribution"]
        }
        own_attribution = by_message_id[own_message.message_id]
        colleague_attribution = by_message_id[colleague_message.message_id]

        assert own_attribution["source_label"] == "Your conversation"
        assert own_attribution["is_current_assistant"] is True
        assert colleague_attribution["source_label"] == "From: Sarah Chen"
        assert colleague_attribution["authoring_assistant_name"] == "Sarah Chen"
        assert colleague_attribution["is_current_assistant"] is False

        assert list_calls == [
            {
                "agent_id": 90210,
                "list_all_org": True,
                "api_key": "owner-key",
            },
        ]
        assert (
            second["message_authoring_attribution"]
            == first["message_authoring_attribution"]
        )
    finally:
        _delete_context_tree(f"Spaces/{space_id}")


def test_colleague_name_cache_invalidates_when_org_scope_changes(monkeypatch):
    SESSION_DETAILS.assistant.agent_id = 123
    SESSION_DETAILS.org_id = 1
    SESSION_DETAILS.unify_key = "scope-one"
    cache = ColleagueNameCache()
    calls: list[dict] = []

    def fake_list_assistants(**kwargs):
        calls.append(kwargs)
        if kwargs["api_key"] == "scope-one":
            return [{"agent_id": 999, "first_name": "Mina", "surname": "Ops"}]
        return [{"agent_id": 999, "first_name": "Rafi", "surname": "Ops"}]

    monkeypatch.setattr(
        "unity.common.colleague_cache.unify.list_assistants",
        fake_list_assistants,
    )

    assert cache.resolve(999) == "Mina Ops"
    assert cache.resolve(999) == "Mina Ops"

    SESSION_DETAILS.unify_key = "scope-two"
    assert cache.resolve(999) == "Rafi Ops"
    assert len(calls) == 2


def test_colleague_name_cache_caches_error_fallback(monkeypatch):
    SESSION_DETAILS.assistant.agent_id = 321
    SESSION_DETAILS.unify_key = "owner-key"
    cache = ColleagueNameCache()
    calls: list[dict] = []

    def failing_list_assistants(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("temporary transport error")

    monkeypatch.setattr(
        "unity.common.colleague_cache.unify.list_assistants",
        failing_list_assistants,
    )

    assert cache.resolve(654) == "a colleague"
    assert cache.resolve(654) == "a colleague"
    assert len(calls) == 1


@_handle_project
def test_shared_authoring_attribution_uses_former_colleague_fallback(monkeypatch):
    space_id = _space_id()
    SESSION_DETAILS.assistant.agent_id = 321
    SESSION_DETAILS.assistant.first_name = "Mina"
    SESSION_DETAILS.assistant.surname = "Support"
    SESSION_DETAILS.user.id = "boss-user"
    SESSION_DETAILS.space_ids = [space_id]
    manager = TranscriptManager()

    try:
        SESSION_DETAILS.assistant.agent_id = 55555
        [colleague_message] = manager.log_messages(
            _message_payload("missing-assistant marker", exchange_id=2201),
            synchronous=True,
            destination=f"space:{space_id}",
        )
        SESSION_DETAILS.assistant.agent_id = 321

        monkeypatch.setattr(
            "unity.common.colleague_cache.unify.list_assistants",
            lambda **_: [],
        )

        result = manager._filter_messages(limit=10)
        by_message_id = {
            row["message_id"]: row for row in result["message_authoring_attribution"]
        }
        attribution = by_message_id[colleague_message.message_id]

        assert attribution["authoring_assistant_name"] == "a former colleague"
        assert attribution["source_label"] == "From: a former colleague"
        assert attribution["is_current_assistant"] is False
    finally:
        _delete_context_tree(f"Spaces/{space_id}")
