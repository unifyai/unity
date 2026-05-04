"""Privileged lifecycle tools exposed only to Coordinator sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import unify
from unify.utils.http import RequestError

from unity.common.tool_outcome import ToolError
from unity.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


class CoordinatorTools:
    """Per-turn tool collaborator for Coordinator-owned lifecycle actions."""

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._assistant_cache: list[dict[str, Any]] | None = None

    def create_assistant(
        self,
        *,
        first_name: str,
        surname: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a colleague in the Coordinator owner's workspace."""

        try:
            result = unify.create_assistant(
                first_name=first_name,
                surname=surname,
                config=config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._clear_assistant_cache()
        return result

    def delete_assistant(
        self,
        *,
        agent_id: int,
    ) -> dict[str, Any] | str | ToolError:
        """Delete a reachable colleague by assistant id."""

        reachable = self._assistant_is_reachable(agent_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return {
                "error_kind": "not_found",
                "message": f"Assistant {agent_id} is not reachable by this Coordinator.",
                "details": {"agent_id": agent_id},
            }
        try:
            result = unify.delete_assistant(
                agent_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._clear_assistant_cache()
        return result

    def update_assistant_config(
        self,
        *,
        agent_id: int,
        config: dict[str, Any],
    ) -> dict[str, Any] | ToolError:
        """Update configuration for a reachable colleague."""

        reachable = self._assistant_is_reachable(agent_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return {
                "error_kind": "not_found",
                "message": f"Assistant {agent_id} is not reachable by this Coordinator.",
                "details": {"agent_id": agent_id},
            }
        try:
            result = unify.update_assistant_config(
                agent_id,
                config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._clear_assistant_cache()
        return result

    def list_assistants(
        self,
        *,
        phone: str | None = None,
        email: str | None = None,
        agent_id: int | None = None,
    ) -> list[dict[str, Any]] | ToolError:
        """List assistants visible to the Coordinator owner."""

        try:
            assistants = unify.list_assistants(
                phone=phone,
                email=email,
                agent_id=agent_id,
                list_all_org=SESSION_DETAILS.org_id is not None,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        if phone is None and email is None and agent_id is None:
            self._assistant_cache = assistants
        return assistants

    def list_org_members(self) -> list[dict[str, Any]] | ToolError:
        """List authorized humans in the Coordinator's organization."""

        if SESSION_DETAILS.org_id is None:
            return []
        try:
            return unify.list_org_members(
                SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the Coordinator-only tools for the slow-brain loop."""

        return {
            "create_assistant": self.create_assistant,
            "delete_assistant": self.delete_assistant,
            "update_assistant_config": self.update_assistant_config,
            "list_assistants": self.list_assistants,
            "list_org_members": self.list_org_members,
        }

    def _assistant_is_reachable(self, agent_id: int) -> bool | ToolError:
        assistants = self._assistant_cache
        if assistants is None:
            listed = self.list_assistants(agent_id=agent_id)
            if isinstance(listed, dict) and "error_kind" in listed:
                return listed
            assistants = listed
        reachable_ids = {str(row.get("agent_id")) for row in assistants}
        return str(agent_id) in reachable_ids

    def _clear_assistant_cache(self) -> None:
        self._assistant_cache = None


def _request_error_to_tool_error(exc: RequestError) -> ToolError:
    """Translate an SDK request failure into the shared tool-error envelope."""

    response = exc.response
    status_code = getattr(response, "status_code", 500)
    if status_code in (400, 422):
        error_kind = "invalid_argument"
    elif status_code in (401, 403):
        error_kind = "permission_denied"
    elif status_code == 404:
        error_kind = "not_found"
    elif status_code == 409:
        error_kind = "conflict"
    else:
        error_kind = "internal"

    response_text = getattr(response, "text", "") or str(exc)
    return {
        "error_kind": error_kind,
        "message": response_text,
        "details": {"status_code": status_code},
    }
