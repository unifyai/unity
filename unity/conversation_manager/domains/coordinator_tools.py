"""Privileged workspace tools exposed only to Coordinator sessions."""

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
    """Per-turn tool collaborator for Coordinator-owned workspace actions."""

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._assistant_cache: list[dict[str, Any]] | None = None
        self._known_assistant_ids: set[str] = set()
        self._space_cache: list[dict[str, Any]] | None = None
        self._known_space_ids: set[str] = set()

    def create_assistant(
        self,
        *,
        first_name: str,
        surname: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a confirmed colleague after exact setup scope is agreed."""

        try:
            result = unify.create_assistant(
                first_name=first_name,
                surname=surname,
                config=config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._remember_assistant(result)
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
            return _assistant_not_found(agent_id)
        try:
            result = unify.delete_assistant(
                agent_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._known_assistant_ids.discard(str(agent_id))
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
            return _assistant_not_found(agent_id)
        try:
            result = unify.update_assistant_config(
                agent_id,
                config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._remember_assistant(result)
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
        self._remember_assistants(assistants)
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

    def create_space(
        self,
        *,
        name: str,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a confirmed team space after exact setup scope is agreed."""

        del organization_id, owner_user_id
        try:
            result = unify.create_space(
                name=name,
                organization_id=SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._remember_space(result)
        self._clear_space_cache()
        return result

    def delete_space(self, *, space_id: int) -> dict[str, Any] | ToolError:
        """Delete a reachable team space."""

        reachable = self._space_is_reachable(space_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return _space_not_found(space_id)
        try:
            result = unify.delete_space(
                space_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._known_space_ids.discard(str(space_id))
        self._clear_space_cache()
        return result

    def update_space(
        self,
        *,
        space_id: int,
        patch: dict[str, Any],
    ) -> dict[str, Any] | ToolError:
        """Update a reachable team space after the intended change is agreed."""

        reachable = self._space_is_reachable(space_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return _space_not_found(space_id)
        try:
            result = unify.update_space(
                space_id,
                patch,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._remember_space(result)
        self._clear_space_cache()
        return result

    def add_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Add a reachable assistant to a reachable space after membership is agreed."""

        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            return invalid

        try:
            return unify.add_space_member(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def remove_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Remove a reachable assistant from a reachable space."""

        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            return invalid

        try:
            return unify.remove_space_member(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def list_spaces(
        self,
        *,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]] | ToolError:
        """List spaces visible to the Coordinator owner."""

        del organization_id, owner_user_id
        try:
            spaces = unify.list_spaces(
                organization_id=SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._space_cache = spaces
        self._remember_spaces(spaces)
        return spaces

    def list_space_members(
        self,
        *,
        space_id: int,
    ) -> list[dict[str, Any]] | ToolError:
        """List live assistant members for a reachable space."""

        reachable = self._space_is_reachable(space_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return _space_not_found(space_id)
        try:
            return unify.list_space_members(
                space_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def list_spaces_for_assistant(
        self,
        *,
        assistant_id: int,
    ) -> list[dict[str, Any]] | ToolError:
        """List spaces for a reachable assistant."""

        reachable = self._assistant_is_reachable(assistant_id)
        if isinstance(reachable, dict):
            return reachable
        if not reachable:
            return _assistant_not_found(assistant_id)
        try:
            return unify.list_spaces_for_assistant(
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def invite_assistant_to_space(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Invite a reachable assistant's owner to a space after membership is agreed."""

        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            return invalid

        try:
            return unify.invite_assistant_to_space(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def cancel_space_invitation(self, *, invite_id: int) -> dict[str, Any] | ToolError:
        """Cancel a pending space invitation created by the Coordinator owner.

        Space invitations are keyed by invite id; Orchestra enforces that only
        the inviter can cancel an invitation.
        """

        try:
            return unify.cancel_space_invitation(
                invite_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def list_pending_invitations(self) -> list[dict[str, Any]] | ToolError:
        """List pending space invitations for the Coordinator owner."""

        try:
            return unify.list_pending_invitations(api_key=SESSION_DETAILS.unify_key)
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
            "create_space": self.create_space,
            "delete_space": self.delete_space,
            "update_space": self.update_space,
            "add_space_member": self.add_space_member,
            "remove_space_member": self.remove_space_member,
            "list_spaces": self.list_spaces,
            "list_space_members": self.list_space_members,
            "list_spaces_for_assistant": self.list_spaces_for_assistant,
            "invite_assistant_to_space": self.invite_assistant_to_space,
            "cancel_space_invitation": self.cancel_space_invitation,
            "list_pending_invitations": self.list_pending_invitations,
        }

    def _assistant_is_reachable(self, agent_id: int) -> bool | ToolError:
        if str(agent_id) in self._known_assistant_ids:
            return True
        assistants = self._assistant_cache
        if assistants is None:
            listed = self.list_assistants(agent_id=agent_id)
            if isinstance(listed, dict) and "error_kind" in listed:
                return listed
            assistants = listed
        reachable_ids = {str(row.get("agent_id")) for row in assistants}
        self._known_assistant_ids.update(reachable_ids)
        return str(agent_id) in reachable_ids

    def _space_is_reachable(self, space_id: int) -> bool | ToolError:
        if str(space_id) in self._known_space_ids:
            return True
        spaces = self._space_cache
        if spaces is None:
            listed = self.list_spaces()
            if isinstance(listed, dict) and "error_kind" in listed:
                return listed
            spaces = listed
        reachable_ids = {str(row.get("space_id")) for row in spaces}
        self._known_space_ids.update(reachable_ids)
        return str(space_id) in reachable_ids

    def _validate_space_and_assistant(
        self,
        space_id: int,
        assistant_id: int,
    ) -> ToolError | None:
        reachable_space = self._space_is_reachable(space_id)
        if isinstance(reachable_space, dict):
            return reachable_space
        if not reachable_space:
            return _space_not_found(space_id)

        reachable_assistant = self._assistant_is_reachable(assistant_id)
        if isinstance(reachable_assistant, dict):
            return reachable_assistant
        if not reachable_assistant:
            return _assistant_not_found(assistant_id)
        return None

    def _clear_assistant_cache(self) -> None:
        self._assistant_cache = None

    def _clear_space_cache(self) -> None:
        self._space_cache = None

    def _remember_assistants(self, assistants: list[dict[str, Any]]) -> None:
        self._known_assistant_ids.update(
            str(row["agent_id"])
            for row in assistants
            if row.get("agent_id") is not None
        )

    def _remember_assistant(self, assistant: dict[str, Any]) -> None:
        agent_id = assistant.get("agent_id")
        if agent_id is not None:
            self._known_assistant_ids.add(str(agent_id))

    def _remember_space(self, space: dict[str, Any]) -> None:
        space_id = space.get("space_id")
        if space_id is not None:
            self._known_space_ids.add(str(space_id))

    def _remember_spaces(self, spaces: list[dict[str, Any]]) -> None:
        for space in spaces:
            self._remember_space(space)


def _assistant_not_found(agent_id: int) -> ToolError:
    """Build a tool error for assistant ids outside the reachable set."""

    return {
        "error_kind": "not_found",
        "message": f"Assistant {agent_id} is not reachable by this Coordinator.",
        "details": {"agent_id": agent_id},
    }


def _space_not_found(space_id: int) -> ToolError:
    """Build a tool error for space ids outside the reachable set."""

    return {
        "error_kind": "not_found",
        "message": f"Space {space_id} is not reachable by this Coordinator.",
        "details": {"space_id": space_id},
    }


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
