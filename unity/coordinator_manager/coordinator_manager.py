"""State manager for the Coordinator's onboarding checklist."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import unify
from pydantic import BaseModel, Field
from unify.utils.http import RequestError

from unity.common.context_registry import ContextRegistry, TableContext
from unity.common.log_utils import log as unity_log
from unity.common.model_to_fields import model_to_fields
from unity.common.state_managers import BaseStateManager
from unity.common.tool_outcome import ToolError, ToolOutcome
from unity.manager_registry import SingletonABCMeta
from unity.session_details import SESSION_DETAILS

COORDINATOR_STATE_CONTEXT = "Coordinator/State"
COORDINATOR_CHECKLIST_CONTEXT = "Coordinator/Checklist"
COORDINATOR_STATE_MODES = {"active", "ready_to_go"}
COORDINATOR_CHECKLIST_STATUSES = {"pending", "done", "skipped"}


class CoordinatorState(BaseModel):
    """Single-row state for the Coordinator's onboarding mode."""

    mode: Literal["active", "ready_to_go"] = "active"
    started_at: datetime
    ready_at: datetime | None = None


class CoordinatorChecklistItem(BaseModel):
    """One onboarding checklist item owned by the Coordinator."""

    item_id: int = Field(default=0)
    title: str
    description: str | None = None
    kind: str | None = None
    status: Literal["pending", "done", "skipped"] = "pending"
    created_at: datetime
    updated_at: datetime


class CoordinatorOnboardingManager(BaseStateManager, metaclass=SingletonABCMeta):
    """Manage the Coordinator's private setup state and checklist."""

    class Config:
        required_contexts = [
            TableContext(
                name=COORDINATOR_STATE_CONTEXT,
                description="Single-row Coordinator onboarding state.",
                fields=model_to_fields(CoordinatorState),
            ),
            TableContext(
                name=COORDINATOR_CHECKLIST_CONTEXT,
                description="Coordinator-owned setup checklist items.",
                fields=model_to_fields(CoordinatorChecklistItem),
                unique_keys={"item_id": "int"},
                auto_counting={"item_id": None},
            ),
        ]

    def __init__(self) -> None:
        """Resolve assistant-scope contexts and initialize local caches."""

        super().__init__()
        self._state_context = ContextRegistry.get_context(
            self,
            COORDINATOR_STATE_CONTEXT,
        )
        self._checklist_context = ContextRegistry.get_context(
            self,
            COORDINATOR_CHECKLIST_CONTEXT,
        )
        self._org_members_cache_key: tuple[int | None, str] | object = _CACHE_EMPTY
        self._org_members_cache: list[dict[str, Any]] | object = _CACHE_EMPTY

    def get_state(self) -> dict[str, Any] | None:
        """Return the current Coordinator state row, if one exists."""

        rows = unify.get_logs(context=self._state_context, limit=1)
        if not rows:
            return None

        return dict(rows[0].entries or {})

    def get_checklist(self) -> list[dict[str, Any]]:
        """Return Coordinator checklist rows ordered by item id."""

        rows = unify.get_logs(
            context=self._checklist_context,
            sorting={"item_id": "ascending"},
            limit=200,
        )
        return [dict(row.entries or {}) for row in rows]

    def get_org_members(self) -> list[dict[str, Any]]:
        """Return authorized humans in the Coordinator's organization."""

        cache_key = (SESSION_DETAILS.org_id, SESSION_DETAILS.unify_key)
        if (
            self._org_members_cache_key == cache_key
            and self._org_members_cache is not _CACHE_EMPTY
        ):
            return self._org_members_cache  # type: ignore[return-value]

        if SESSION_DETAILS.org_id is None:
            self._org_members_cache_key = cache_key
            self._org_members_cache = []
            return []

        try:
            members = unify.list_org_members(
                SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError:
            return []
        self._org_members_cache_key = cache_key
        self._org_members_cache = members
        return members

    def set_state(
        self,
        *,
        mode: Literal["active", "ready_to_go"],
        ready_at: datetime | None = None,
    ) -> ToolOutcome | ToolError:
        """Create or update the Coordinator onboarding state row."""

        if mode not in COORDINATOR_STATE_MODES:
            return _tool_error(
                "invalid_argument",
                "Coordinator state mode must be 'active' or 'ready_to_go'.",
                {"mode": mode},
            )

        now = _utc_now()
        current = self.get_state()
        next_ready_at = ready_at
        if mode == "ready_to_go" and next_ready_at is None:
            next_ready_at = now

        if current is None:
            unity_log(
                context=self._state_context,
                mode=mode,
                started_at=now,
                ready_at=next_ready_at,
                new=True,
                mutable=True,
                add_to_all_context=False,
            )
        else:
            ids = unify.get_logs(
                context=self._state_context,
                limit=1,
                return_ids_only=True,
            )
            if not ids:
                return _tool_error(
                    "not_found",
                    "Coordinator state row disappeared before it could be updated.",
                    {},
                )
            updates: dict[str, Any] = {"mode": mode, "ready_at": next_ready_at}
            unify.update_logs(
                logs=[ids[0]],
                context=self._state_context,
                entries=updates,
                overwrite=True,
            )

        return {"outcome": "coordinator state updated", "details": {"mode": mode}}

    def add_checklist_item(
        self,
        *,
        title: str,
        description: str | None = None,
        kind: str | None = None,
    ) -> ToolOutcome | ToolError:
        """Add a pending item to the Coordinator setup checklist."""

        if not title.strip():
            return _tool_error(
                "invalid_argument",
                "Checklist item title is required.",
                {"title": title},
            )

        now = _utc_now()
        row = unity_log(
            context=self._checklist_context,
            title=title,
            description=description,
            kind=kind,
            status="pending",
            created_at=now,
            updated_at=now,
            new=True,
            mutable=True,
            add_to_all_context=False,
        )
        return {
            "outcome": "checklist item added",
            "details": {"item_id": row.entries["item_id"]},
        }

    def update_checklist_item(
        self,
        *,
        item_id: int,
        status: str | None = None,
        title: str | None = None,
        description: str | None = None,
        kind: str | None = None,
    ) -> ToolOutcome | ToolError:
        """Update mutable fields on one Coordinator checklist item."""

        if status is not None and status not in COORDINATOR_CHECKLIST_STATUSES:
            return _tool_error(
                "invalid_argument",
                "Checklist status must be 'pending', 'done', or 'skipped'.",
                {"status": status},
            )

        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status
        if title is not None:
            if not title.strip():
                return _tool_error(
                    "invalid_argument",
                    "Checklist item title cannot be empty.",
                    {"title": title},
                )
            updates["title"] = title
        if description is not None:
            updates["description"] = description
        if kind is not None:
            updates["kind"] = kind
        if not updates:
            return _tool_error(
                "invalid_argument",
                "At least one checklist field must be provided.",
                {"item_id": item_id},
            )
        updates["updated_at"] = _utc_now()

        ids = self._checklist_log_ids(item_id)
        if isinstance(ids, dict):
            return ids

        unify.update_logs(
            logs=ids,
            context=self._checklist_context,
            entries=updates,
            overwrite=True,
        )
        return {"outcome": "checklist item updated", "details": {"item_id": item_id}}

    def delete_checklist_item(self, *, item_id: int) -> ToolOutcome | ToolError:
        """Delete one item from the Coordinator setup checklist."""

        ids = self._checklist_log_ids(item_id)
        if isinstance(ids, dict):
            return ids

        unify.delete_logs(context=self._checklist_context, logs=ids[0])
        return {"outcome": "checklist item deleted", "details": {"item_id": item_id}}

    def _checklist_log_ids(self, item_id: int) -> list[int] | ToolError:
        ids = unify.get_logs(
            context=self._checklist_context,
            filter=f"item_id == {int(item_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            return _tool_error(
                "not_found",
                f"No checklist item found with item_id {item_id}.",
                {"item_id": item_id},
            )
        if len(ids) > 1:
            return _tool_error(
                "conflict",
                f"Multiple checklist items found with item_id {item_id}.",
                {"item_id": item_id},
            )
        return ids


_CACHE_EMPTY = object()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _tool_error(error_kind: str, message: str, details: Any) -> ToolError:
    return {"error_kind": error_kind, "message": message, "details": details}
