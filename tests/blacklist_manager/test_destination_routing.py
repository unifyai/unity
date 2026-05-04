from __future__ import annotations

import uuid

import pytest
import unify

from tests.helpers import _handle_project
from unity.blacklist_manager.blacklist_manager import BlackListManager
from unity.common.context_registry import ContextRegistry
from unity.conversation_manager.cm_types import Medium
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS


def _configure_spaces() -> tuple[int, int]:
    base_space_id = 30_000_000 + uuid.uuid4().int % 1_000_000_000
    space_ids = (base_space_id, base_space_id + 1)
    SESSION_DETAILS.space_ids = list(space_ids)
    SESSION_DETAILS.space_summaries = [
        {
            "space_id": space_ids[0],
            "name": "Comms Safety",
            "description": "Shared workspace for communication safety rules.",
        },
        {
            "space_id": space_ids[1],
            "name": "Vendors",
            "description": "Shared vendor coordination workspace.",
        },
    ]
    ContextRegistry.clear()
    ManagerRegistry.clear()
    return space_ids


def _reset_spaces(space_ids: tuple[int, int]) -> None:
    for space_id in space_ids:
        try:
            unify.delete_context(f"Spaces/{space_id}/BlackList")
        except Exception:
            pass
    SESSION_DETAILS.space_ids = []
    SESSION_DETAILS.space_summaries = []
    ContextRegistry.clear()
    ManagerRegistry.clear()


@_handle_project
def test_blacklist_writes_route_to_destination_and_reads_merge_roots():
    space_ids = _configure_spaces()
    manager = BlackListManager()
    personal_detail = f"personal.{uuid.uuid4().hex}@example.com"
    shared_detail = f"shared.{uuid.uuid4().hex}@example.com"

    try:
        manager.create_blacklist_entry(
            medium=Medium.EMAIL,
            contact_detail=personal_detail,
            reason="personal block",
        )
        manager.create_blacklist_entry(
            medium=Medium.EMAIL,
            contact_detail=shared_detail,
            reason="shared block",
            destination=f"space:{space_ids[0]}",
        )

        merged = manager.filter_blacklist()["entries"]
        assert {entry.contact_detail for entry in merged} >= {
            personal_detail,
            shared_detail,
        }
        destinations_by_detail = {
            entry.contact_detail: entry.destination for entry in merged
        }
        assert destinations_by_detail[personal_detail] == "personal"
        assert destinations_by_detail[shared_detail] == f"space:{space_ids[0]}"

        shared_rows = unify.get_logs(
            context=f"Spaces/{space_ids[0]}/BlackList",
            filter=f"contact_detail == '{shared_detail}'",
        )
        assert len(shared_rows) == 1
    finally:
        _reset_spaces(space_ids)


@_handle_project
def test_blacklist_any_visible_root_blocks_contact_detail(monkeypatch):
    from unity.conversation_manager import comms_manager
    from unity.settings import SETTINGS

    space_ids = _configure_spaces()
    detail = f"blocked.{uuid.uuid4().hex}@example.com"
    manager = BlackListManager()

    try:
        manager.create_blacklist_entry(
            medium=Medium.EMAIL,
            contact_detail=detail,
            reason="shared block",
            destination=f"space:{space_ids[0]}",
        )
        monkeypatch.setattr(
            SETTINGS.conversation,
            "BLACKLIST_CHECKS_ENABLED",
            True,
        )

        assert comms_manager._is_blacklisted(Medium.EMAIL.value, detail) is True
    finally:
        _reset_spaces(space_ids)


@_handle_project
def test_blacklist_invalid_destination_returns_tool_error():
    space_ids = _configure_spaces()
    manager = BlackListManager()

    try:
        outcome = manager.create_blacklist_entry(
            medium=Medium.EMAIL,
            contact_detail=f"bad.{uuid.uuid4().hex}@example.com",
            reason="bad destination",
            destination="space:99999999",
        )
    finally:
        _reset_spaces(space_ids)

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@_handle_project
def test_blacklist_clear_invalid_destination_returns_tool_error():
    space_ids = _configure_spaces()
    manager = BlackListManager()

    try:
        outcome = manager.clear(destination="space:99999999")
    finally:
        _reset_spaces(space_ids)

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"


@pytest.mark.parametrize(
    "call",
    [
        lambda manager: manager.create_blacklist_entry(
            medium=Medium.EMAIL,
            contact_detail=f"bad.{uuid.uuid4().hex}@example.com",
            reason="bad destination",
            destination="space:99999999",
        ),
        lambda manager: manager.update_blacklist_entry(
            blacklist_id=1,
            reason="updated",
            destination="space:99999999",
        ),
        lambda manager: manager.delete_blacklist_entry(
            blacklist_id=1,
            destination="space:99999999",
        ),
        lambda manager: manager.clear(destination="space:99999999"),
    ],
)
@_handle_project
def test_blacklist_write_tools_return_tool_error_for_invalid_destination(call):
    space_ids = _configure_spaces()
    manager = BlackListManager()

    try:
        outcome = call(manager)
    finally:
        _reset_spaces(space_ids)

    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:99999999"
