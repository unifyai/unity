"""Shared helpers for DashboardManager tests."""

from __future__ import annotations

import json

import unify

from unity.dashboard_manager.dashboard_manager import DashboardManager
from unity.manager_registry import ManagerRegistry


def fresh_dashboard_manager() -> DashboardManager:
    """Create a DashboardManager after clearing singleton manager state."""
    ManagerRegistry.clear()
    return DashboardManager()


def context_rows(context: str) -> list[dict]:
    """Return raw row entries for a Unify context."""
    return [log.entries for log in unify.get_logs(context=context)]


def context_titles(context: str) -> set[str]:
    """Return all row titles in a Unify context."""
    return {row["title"] for row in context_rows(context)}


def create_context_if_missing(context: str) -> None:
    """Create a Unify context, ignoring the already-exists case."""
    unify.create_context(context)


def active_read_root() -> str:
    """Return the current project's active read root."""
    return unify.get_active_context()["read"]


def serialized_binding_context(tile) -> str:
    """Return the single serialized binding context for a tile."""
    assert tile.data_bindings_json is not None
    bindings = json.loads(tile.data_bindings_json)
    assert len(bindings) == 1
    return bindings[0]["context"]
