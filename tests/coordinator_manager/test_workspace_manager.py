from __future__ import annotations

import inspect

import pytest

from unity.coordinator_manager.workspace_manager import CoordinatorWorkspaceManager
from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS


@pytest.fixture(autouse=True)
def _reset_state():
    SESSION_DETAILS.reset()
    ManagerRegistry.clear()
    yield
    SESSION_DETAILS.reset()
    ManagerRegistry.clear()


def test_workspace_manager_blocks_non_coordinator_sessions():
    SESSION_DETAILS.is_coordinator = False

    manager = CoordinatorWorkspaceManager()
    result = manager.list_assistants()

    assert result["error_kind"] == "permission_denied"
    assert result["details"]["is_coordinator"] is False


def test_workspace_manager_delegates_reads_for_coordinator(monkeypatch):
    SESSION_DETAILS.is_coordinator = True

    monkeypatch.setattr(
        CoordinatorTools,
        "list_assistants",
        lambda self, **_: [{"agent_id": 42, "first_name": "Ops"}],
    )

    manager = CoordinatorWorkspaceManager()
    result = manager.list_assistants(agent_id=42)

    assert result == [{"agent_id": 42, "first_name": "Ops"}]


def test_workspace_manager_blocks_mutations_when_role_is_not_coordinator(monkeypatch):
    SESSION_DETAILS.is_coordinator = False
    called = {"create_space": False}

    def _fake_create_space(self, **_kwargs):
        called["create_space"] = True
        return {"space_id": 99}

    monkeypatch.setattr(CoordinatorTools, "create_space", _fake_create_space)

    manager = CoordinatorWorkspaceManager()
    result = manager.create_space(name="Ops", description="Operations")

    assert result["error_kind"] == "permission_denied"
    assert called["create_space"] is False


def test_workspace_manager_inherits_rich_delegate_docstrings():
    """Actor-facing coordinator primitives keep delegate 'when to use' guidance."""

    methods = (
        "create_assistant",
        "pre_seed_colleague",
        "create_space",
        "add_space_member",
        "commission_colleague_into_workspace",
        "add_setup_checklist_item",
        "update_setup_checklist_item",
    )
    for method_name in methods:
        workspace_doc = inspect.getdoc(
            getattr(CoordinatorWorkspaceManager, method_name),
        )
        delegate_doc = inspect.getdoc(getattr(CoordinatorTools, method_name))
        assert workspace_doc is not None
        assert delegate_doc is not None
        workspace_paragraphs = [p for p in workspace_doc.split("\n\n") if p.strip()]
        delegate_paragraphs = [p for p in delegate_doc.split("\n\n") if p.strip()]
        assert len(workspace_paragraphs) >= 2
        assert len(delegate_paragraphs) >= 2
        assert workspace_paragraphs[1] == delegate_paragraphs[1]
        assert "Parameters" in workspace_doc
