"""Privileged workspace tools exposed only to Coordinator sessions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import unify
from pydantic import BaseModel, ConfigDict, Field
from unify.utils.http import RequestError

from unity.common.tool_outcome import ToolError
from unity.coordinator_manager.activity import (
    activity_entity,
    coordinator_activity_id,
    publish_coordinator_activity,
    safe_activity_text,
)
from unity.coordinator_manager.coordinator_manager import CoordinatorOnboardingManager
from unity.events.types.coordinator_activity import (
    CoordinatorActivityEntity,
    CoordinatorActivityPhase,
    CoordinatorActivityStage,
    CoordinatorActivitySurface,
)
from unity.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


class CoordinatorPreseedWrite(BaseModel):
    """One colleague-owned context batch that the Coordinator can pre-seed."""

    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        ...,
        min_length=1,
        description=(
            "Relative colleague-owned context name, such as Tasks, Knowledge, "
            "Guidance, Functions/..., Data, or Dashboards/...."
        ),
    )
    entries: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Rows to write into that context.",
    )


def _preseed_write_payload(
    writes: Sequence[CoordinatorPreseedWrite | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return SDK-ready preseed writes from model or mapping inputs."""
    payload: list[dict[str, Any]] = []
    for write in writes:
        if isinstance(write, CoordinatorPreseedWrite):
            payload.append(write.model_dump())
        else:
            payload.append(dict(write))
    return payload


class CoordinatorTools:
    """Per-turn tool collaborator for Coordinator-owned workspace actions."""

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._assistant_cache: list[dict[str, Any]] | None = None
        self._known_assistant_ids: set[str] = set()
        self._space_cache: list[dict[str, Any]] | None = None
        self._known_space_ids: set[str] = set()
        self._activity_metadata: dict[
            str,
            tuple[
                list[CoordinatorActivitySurface],
                list[CoordinatorActivityEntity | dict[str, Any]],
            ],
        ] = {}

    @staticmethod
    def _derived_colleague_job_title(
        *,
        first_name: str,
        surname: str | None,
    ) -> str | None:
        """Return a role-style job title inferred from commissioning inputs."""
        candidate = (surname or "").strip()
        if not candidate:
            return None
        if " " in first_name.strip():
            return candidate
        return None

    def _assistant_defaults_from_coordinator(
        self,
        *,
        first_name: str,
        surname: str | None,
        config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Build coordinator-inherited defaults for colleague creation."""
        merged = dict(config or {})
        coordinator_timezone = (SESSION_DETAILS.assistant.timezone or "").strip()
        coordinator_nationality = (SESSION_DETAILS.assistant.nationality or "").strip()
        if not merged.get("timezone") and coordinator_timezone:
            merged["timezone"] = coordinator_timezone
        if not merged.get("nationality") and coordinator_nationality:
            merged["nationality"] = coordinator_nationality
        if not merged.get("job_title"):
            derived_job_title = self._derived_colleague_job_title(
                first_name=first_name,
                surname=surname,
            )
            if derived_job_title:
                merged["job_title"] = derived_job_title
        return merged or None

    def create_assistant(
        self,
        *,
        first_name: str,
        surname: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a confirmed colleague after exact setup scope is agreed."""
        suppression = self._suppress_duplicate_commissioning_tool(
            tool_name="create_assistant",
            tool_args={
                "first_name": first_name,
                "surname": surname,
                "config": config,
            },
        )
        if suppression is not None:
            return suppression

        assistant_config = self._assistant_defaults_from_coordinator(
            first_name=first_name,
            surname=surname,
            config=config,
        )
        colleague_name = _display_name(first_name=first_name, surname=surname)
        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title=f"Creating {colleague_name} colleague",
            surfaces=["colleagues"],
            related_entities=[
                activity_entity("colleague", name=colleague_name),
            ],
        )
        try:
            result = unify.create_assistant(
                first_name=first_name,
                surname=surname,
                config=assistant_config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title=f"Could not create {colleague_name} colleague",
                error=error,
            )
            return error
        self._remember_assistant(result)
        self._clear_assistant_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title=f"Created {safe_activity_text(_assistant_display_name(result), fallback=colleague_name)} colleague",
            surfaces=["colleagues"],
            related_entities=[_assistant_entity(result, fallback=colleague_name)],
            activity_id=activity_id,
        )
        return result

    def delete_assistant(
        self,
        *,
        agent_id: int,
    ) -> dict[str, Any] | str | ToolError:
        """Delete a reachable colleague by assistant id."""

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Removing colleague",
            surfaces=["colleagues"],
            related_entities=[
                activity_entity("colleague", name="Colleague", entity_id=agent_id),
            ],
        )
        reachable = self._assistant_is_reachable(agent_id)
        if isinstance(reachable, dict):
            self._publish_failure(
                activity_id,
                title="Could not remove colleague",
                error=reachable,
            )
            return reachable
        if not reachable:
            error = _assistant_not_found(agent_id)
            self._publish_failure(
                activity_id,
                title="Could not remove colleague",
                error=error,
            )
            return error
        try:
            result = unify.delete_assistant(
                agent_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not remove colleague",
                error=error,
            )
            return error
        self._known_assistant_ids.discard(str(agent_id))
        self._clear_assistant_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Removed colleague",
            surfaces=["colleagues"],
            related_entities=[
                activity_entity("colleague", name="Colleague", entity_id=agent_id),
            ],
            activity_id=activity_id,
        )
        return result

    def update_assistant_config(
        self,
        *,
        agent_id: int,
        config: dict[str, Any],
    ) -> dict[str, Any] | ToolError:
        """Update configuration for a reachable colleague."""

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Updating colleague profile",
            surfaces=["colleagues"],
            related_entities=[
                activity_entity("colleague", name="Colleague", entity_id=agent_id),
            ],
        )
        reachable = self._assistant_is_reachable(agent_id)
        if isinstance(reachable, dict):
            self._publish_failure(
                activity_id,
                title="Could not update colleague profile",
                error=reachable,
            )
            return reachable
        if not reachable:
            error = _assistant_not_found(agent_id)
            self._publish_failure(
                activity_id,
                title="Could not update colleague profile",
                error=error,
            )
            return error
        try:
            result = unify.update_assistant_config(
                agent_id,
                config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not update colleague profile",
                error=error,
            )
            return error
        self._remember_assistant(result)
        self._clear_assistant_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Updated colleague profile",
            surfaces=["colleagues"],
            related_entities=[_assistant_entity(result, fallback="Colleague")],
            activity_id=activity_id,
        )
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

    def pre_seed_colleague(
        self,
        *,
        target_assistant_id: int,
        writes: list[CoordinatorPreseedWrite],
    ) -> dict[str, Any] | ToolError:
        """Seed confirmed rows into a reachable colleague's own contexts.

        Args:
            target_assistant_id: The colleague assistant that should own the rows.
            writes: Required context batches shaped as
                ``{"context": "Tasks", "entries": [{...}]}``. Use relative context
                names only, such as ``Tasks``, ``Knowledge``, or ``Guidance``. Do
                not use ``Spaces/...`` paths or shared-space destinations here.
        """

        surfaces = _surfaces_for_preseed(writes)
        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Preparing colleague setup rows",
            surfaces=surfaces,
            related_entities=[
                activity_entity(
                    "colleague",
                    name="Colleague",
                    entity_id=target_assistant_id,
                ),
            ],
        )
        reachable = self._assistant_is_reachable(target_assistant_id)
        if isinstance(reachable, dict):
            self._publish_failure(
                activity_id,
                title="Could not prepare colleague setup rows",
                error=reachable,
            )
            return reachable
        if not reachable:
            error = _assistant_not_found(target_assistant_id)
            self._publish_failure(
                activity_id,
                title="Could not prepare colleague setup rows",
                error=error,
            )
            return error
        try:
            result = unify.pre_seed_colleague(
                target_assistant_id,
                _preseed_write_payload(writes),
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not prepare colleague setup rows",
                error=error,
            )
            return error
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Prepared colleague setup rows",
            surfaces=surfaces,
            related_entities=[
                activity_entity(
                    "colleague",
                    name="Colleague",
                    entity_id=target_assistant_id,
                ),
            ],
            activity_id=activity_id,
        )
        return result

    def create_space(
        self,
        *,
        name: str,
        description: str,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a confirmed team space after exact setup scope is agreed."""
        suppression = self._suppress_duplicate_commissioning_tool(
            tool_name="create_space",
            tool_args={
                "name": name,
                "description": description,
                "organization_id": organization_id,
                "owner_user_id": owner_user_id,
            },
        )
        if suppression is not None:
            return suppression

        del organization_id, owner_user_id
        workspace_name = safe_activity_text(name, fallback="Workspace")
        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title=f"Creating {workspace_name} workspace",
            surfaces=["workspaces"],
            related_entities=[
                activity_entity("workspace", name=workspace_name),
            ],
        )
        try:
            result = unify.create_space(
                name=name,
                description=description,
                organization_id=SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title=f"Could not create {workspace_name} workspace",
                error=error,
            )
            return error
        self._remember_space(result)
        self._clear_space_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title=f"Created {safe_activity_text(_space_display_name(result), fallback=workspace_name)} workspace",
            surfaces=["workspaces"],
            related_entities=[_space_entity(result, fallback=workspace_name)],
            activity_id=activity_id,
        )
        return result

    def delete_space(self, *, space_id: int) -> dict[str, Any] | ToolError:
        """Delete a reachable team space."""

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Removing workspace",
            surfaces=["workspaces"],
            related_entities=[
                activity_entity("workspace", name="Workspace", entity_id=space_id),
            ],
        )
        reachable = self._space_is_reachable(space_id)
        if isinstance(reachable, dict):
            self._publish_failure(
                activity_id,
                title="Could not remove workspace",
                error=reachable,
            )
            return reachable
        if not reachable:
            error = _space_not_found(space_id)
            self._publish_failure(
                activity_id,
                title="Could not remove workspace",
                error=error,
            )
            return error
        try:
            result = unify.delete_space(
                space_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not remove workspace",
                error=error,
            )
            return error
        self._known_space_ids.discard(str(space_id))
        self._clear_space_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Removed workspace",
            surfaces=["workspaces"],
            related_entities=[
                activity_entity("workspace", name="Workspace", entity_id=space_id),
            ],
            activity_id=activity_id,
        )
        return result

    def update_space(
        self,
        *,
        space_id: int,
        patch: dict[str, Any],
    ) -> dict[str, Any] | ToolError:
        """Update a reachable team space after the intended change is agreed."""

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Updating workspace",
            surfaces=["workspaces"],
            related_entities=[
                activity_entity("workspace", name="Workspace", entity_id=space_id),
            ],
        )
        reachable = self._space_is_reachable(space_id)
        if isinstance(reachable, dict):
            self._publish_failure(
                activity_id,
                title="Could not update workspace",
                error=reachable,
            )
            return reachable
        if not reachable:
            error = _space_not_found(space_id)
            self._publish_failure(
                activity_id,
                title="Could not update workspace",
                error=error,
            )
            return error
        try:
            result = unify.update_space(
                space_id,
                patch,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not update workspace",
                error=error,
            )
            return error
        self._remember_space(result)
        self._clear_space_cache()
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Updated workspace",
            surfaces=["workspaces"],
            related_entities=[_space_entity(result, fallback="Workspace")],
            activity_id=activity_id,
        )
        return result

    def add_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Add a reachable assistant to a reachable space after membership is agreed."""
        suppression = self._suppress_duplicate_commissioning_tool(
            tool_name="add_space_member",
            tool_args={"space_id": space_id, "assistant_id": assistant_id},
        )
        if suppression is not None:
            return suppression

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Adding colleague to workspace",
            surfaces=["membership"],
            related_entities=_membership_entities(space_id, assistant_id),
        )
        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            self._publish_failure(
                activity_id,
                title="Could not add colleague to workspace",
                error=invalid,
            )
            return invalid

        try:
            result = unify.add_space_member(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not add colleague to workspace",
                error=error,
            )
            return error
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Added colleague to workspace",
            surfaces=["membership"],
            related_entities=_membership_entities(space_id, assistant_id),
            activity_id=activity_id,
        )
        return result

    def commission_colleague_into_workspace(
        self,
        *,
        assistant_first_name: str,
        assistant_surname: str | None = None,
        space_name: str,
        space_description: str,
        assistant_config: dict[str, Any] | None = None,
        assistant_id: int | None = None,
        space_id: int | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create/reuse a colleague and workspace, then ensure membership."""
        suppression = self._suppress_duplicate_commissioning_tool(
            tool_name="commission_colleague_into_workspace",
            tool_args={
                "assistant_first_name": assistant_first_name,
                "assistant_surname": assistant_surname,
                "space_name": space_name,
                "space_description": space_description,
                "assistant_config": assistant_config,
                "assistant_id": assistant_id,
                "space_id": space_id,
            },
        )
        if suppression is not None:
            return suppression

        colleague_name = _display_name(
            first_name=assistant_first_name,
            surname=assistant_surname,
        )
        workspace_name = safe_activity_text(space_name, fallback="Workspace")
        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title=f"Commissioning {colleague_name} into {workspace_name}",
            surfaces=["colleagues", "workspaces", "membership"],
            related_entities=[
                activity_entity(
                    "colleague",
                    name=colleague_name,
                    entity_id=assistant_id,
                ),
                activity_entity("workspace", name=workspace_name, entity_id=space_id),
            ],
        )
        assistant_step = self._resolve_or_create_commission_assistant(
            assistant_first_name=assistant_first_name,
            assistant_surname=assistant_surname,
            assistant_config=assistant_config,
            assistant_id=assistant_id,
        )
        if _is_tool_error(assistant_step):
            self._publish_failure(
                activity_id,
                title=f"Could not commission {colleague_name} into {workspace_name}",
                error=assistant_step,
            )
            return assistant_step

        assistant_row = assistant_step["assistant"]
        resolved_assistant_id = int(assistant_row["agent_id"])
        space_step = self._resolve_or_create_commission_space(
            space_name=space_name,
            space_description=space_description,
            space_id=space_id,
        )
        if _is_tool_error(space_step):
            self._publish_failure(
                activity_id,
                title=f"Could not commission {colleague_name} into {workspace_name}",
                error=space_step,
            )
            return space_step

        space_row = space_step["space"]
        resolved_space_id = int(space_row["space_id"])
        membership_step = self._ensure_commission_membership(
            space_id=resolved_space_id,
            assistant_id=resolved_assistant_id,
        )
        if _is_tool_error(membership_step):
            self._publish_failure(
                activity_id,
                title=f"Could not commission {colleague_name} into {workspace_name}",
                error=membership_step,
            )
            return membership_step

        result = {
            "assistant": {
                "status": assistant_step["status"],
                "assistant_id": resolved_assistant_id,
                "assistant": assistant_row,
            },
            "space": {
                "status": space_step["status"],
                "space_id": resolved_space_id,
                "space": space_row,
            },
            "membership": {
                "status": membership_step["status"],
                "space_id": resolved_space_id,
                "assistant_id": resolved_assistant_id,
            },
        }
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title=f"Commissioned {safe_activity_text(_assistant_display_name(assistant_row), fallback=colleague_name)} into {safe_activity_text(_space_display_name(space_row), fallback=workspace_name)}",
            surfaces=["colleagues", "workspaces", "membership"],
            related_entities=[
                _assistant_entity(assistant_row, fallback=colleague_name),
                _space_entity(space_row, fallback=workspace_name),
                *_membership_entities(resolved_space_id, resolved_assistant_id),
            ],
            activity_id=activity_id,
        )
        return result

    def remove_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Remove a reachable assistant from a reachable space."""

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Removing colleague from workspace",
            surfaces=["membership"],
            related_entities=_membership_entities(space_id, assistant_id),
        )
        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            self._publish_failure(
                activity_id,
                title="Could not remove colleague from workspace",
                error=invalid,
            )
            return invalid

        try:
            result = unify.remove_space_member(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not remove colleague from workspace",
                error=error,
            )
            return error
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Removed colleague from workspace",
            surfaces=["membership"],
            related_entities=_membership_entities(space_id, assistant_id),
            activity_id=activity_id,
        )
        return result

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

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Sending workspace invitation",
            surfaces=["invitations", "membership"],
            related_entities=_membership_entities(space_id, assistant_id),
        )
        invalid = self._validate_space_and_assistant(space_id, assistant_id)
        if invalid is not None:
            self._publish_failure(
                activity_id,
                title="Could not send workspace invitation",
                error=invalid,
            )
            return invalid

        try:
            result = unify.invite_assistant_to_space(
                space_id,
                assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not send workspace invitation",
                error=error,
            )
            return error
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Sent workspace invitation",
            surfaces=["invitations", "membership"],
            related_entities=[
                *_membership_entities(space_id, assistant_id),
                _invitation_entity(result),
            ],
            activity_id=activity_id,
        )
        return result

    def cancel_space_invitation(self, *, invite_id: int) -> dict[str, Any] | ToolError:
        """Cancel a pending space invitation created by the Coordinator owner.

        Space invitations are keyed by invite id; Orchestra enforces that only
        the inviter can cancel an invitation.
        """

        activity_id = self._publish_activity(
            phase="started",
            stage="implementation",
            title="Cancelling workspace invitation",
            surfaces=["invitations"],
            related_entities=[
                activity_entity("invitation", name="Invitation", entity_id=invite_id),
            ],
        )
        try:
            result = unify.cancel_space_invitation(
                invite_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            self._publish_failure(
                activity_id,
                title="Could not cancel workspace invitation",
                error=error,
            )
            return error
        self._publish_activity(
            phase="completed",
            stage="implementation",
            title="Cancelled workspace invitation",
            surfaces=["invitations"],
            related_entities=[
                activity_entity("invitation", name="Invitation", entity_id=invite_id),
            ],
            activity_id=activity_id,
        )
        return result

    def list_pending_invitations(self) -> list[dict[str, Any]] | ToolError:
        """List pending space invitations for the Coordinator owner."""

        try:
            return unify.list_pending_invitations(api_key=SESSION_DETAILS.unify_key)
        except RequestError as exc:
            return _request_error_to_tool_error(exc)

    def set_setup_state(
        self,
        *,
        mode: Literal["active", "ready_to_go"],
        ready_at: datetime | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Update setup mode and optionally attach a handoff suggested reply."""
        try:
            return CoordinatorOnboardingManager().set_state(
                mode=mode,
                ready_at=ready_at,
                chat_prompt=chat_prompt,
                chat_prompt_label=chat_prompt_label,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        except Exception as exc:
            return {
                "error_kind": "internal",
                "message": "Failed to update coordinator setup state.",
                "details": {"error": str(exc)},
            }

    def add_setup_checklist_item(
        self,
        *,
        title: str,
        description: str | None = None,
        kind: str | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Add one user-facing setup step and optional suggested reply CTA."""
        try:
            return CoordinatorOnboardingManager().add_checklist_item(
                title=title,
                description=description,
                kind=kind,
                chat_prompt=chat_prompt,
                chat_prompt_label=chat_prompt_label,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        except Exception as exc:
            return {
                "error_kind": "internal",
                "message": "Failed to add setup checklist item.",
                "details": {"error": str(exc)},
            }

    def update_setup_checklist_item(
        self,
        *,
        item_id: int,
        status: str | None = None,
        title: str | None = None,
        description: str | None = None,
        kind: str | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Update one user-facing setup step and optional suggested reply CTA."""
        try:
            return CoordinatorOnboardingManager().update_checklist_item(
                item_id=item_id,
                status=status,
                title=title,
                description=description,
                kind=kind,
                chat_prompt=chat_prompt,
                chat_prompt_label=chat_prompt_label,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        except Exception as exc:
            return {
                "error_kind": "internal",
                "message": "Failed to update setup checklist item.",
                "details": {"error": str(exc)},
            }

    def _suppress_duplicate_commissioning_tool(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolError | None:
        suppressor = getattr(self._cm, "suppress_duplicate_commissioning_tool", None)
        if not callable(suppressor):
            return None
        suppression = suppressor(tool_name=tool_name, tool_args=tool_args)
        if _is_tool_error(suppression):
            return suppression
        return None

    def _resolve_or_create_commission_assistant(
        self,
        *,
        assistant_first_name: str,
        assistant_surname: str | None,
        assistant_config: dict[str, Any] | None,
        assistant_id: int | None,
    ) -> dict[str, Any] | ToolError:
        if assistant_id is not None:
            listed = self.list_assistants(agent_id=assistant_id)
            if _is_tool_error(listed):
                return listed
            if not listed:
                return _assistant_not_found(assistant_id)
            if len(listed) > 1:
                return _tool_conflict(
                    message=(
                        "Multiple assistants matched the provided assistant_id while "
                        "commissioning."
                    ),
                    details={"assistant_id": assistant_id, "matches": len(listed)},
                )
            assistant = listed[0]
            self._remember_assistant(assistant)
            return {"status": "reused", "assistant": assistant}

        listed = self.list_assistants()
        if _is_tool_error(listed):
            return listed
        matches = [
            assistant
            for assistant in listed
            if _assistant_name_matches(
                assistant,
                first_name=assistant_first_name,
                surname=assistant_surname,
            )
        ]
        if len(matches) > 1:
            return _tool_conflict(
                message=(
                    "Multiple existing assistants matched the requested name. "
                    "Pass assistant_id to disambiguate."
                ),
                details={
                    "assistant_first_name": assistant_first_name,
                    "assistant_surname": assistant_surname,
                    "matches": [row.get("agent_id") for row in matches],
                },
            )
        if len(matches) == 1:
            assistant = matches[0]
            self._remember_assistant(assistant)
            return {"status": "reused", "assistant": assistant}

        resolved_assistant_config = self._assistant_defaults_from_coordinator(
            first_name=assistant_first_name,
            surname=assistant_surname,
            config=assistant_config,
        )
        try:
            created = unify.create_assistant(
                first_name=assistant_first_name,
                surname=assistant_surname,
                config=resolved_assistant_config,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._clear_assistant_cache()
        self._remember_assistant(created)
        return {"status": "created", "assistant": created}

    def _resolve_or_create_commission_space(
        self,
        *,
        space_name: str,
        space_description: str,
        space_id: int | None,
    ) -> dict[str, Any] | ToolError:
        if space_id is not None:
            listed = self.list_spaces()
            if _is_tool_error(listed):
                return listed
            matches = [
                space for space in listed if str(space.get("space_id")) == str(space_id)
            ]
            if not matches:
                return _space_not_found(space_id)
            if len(matches) > 1:
                return _tool_conflict(
                    message=(
                        "Multiple spaces matched the provided space_id while "
                        "commissioning."
                    ),
                    details={"space_id": space_id, "matches": len(matches)},
                )
            space = matches[0]
            self._remember_space(space)
            return {"status": "reused", "space": space}

        listed = self.list_spaces()
        if _is_tool_error(listed):
            return listed
        matches = [
            space
            for space in listed
            if _space_name_matches(space, space_name=space_name)
        ]
        if len(matches) > 1:
            return _tool_conflict(
                message=(
                    "Multiple existing spaces matched the requested name. "
                    "Pass space_id to disambiguate."
                ),
                details={
                    "space_name": space_name,
                    "matches": [row.get("space_id") for row in matches],
                },
            )
        if len(matches) == 1:
            space = matches[0]
            self._remember_space(space)
            return {"status": "reused", "space": space}

        try:
            created = unify.create_space(
                name=space_name,
                description=space_description,
                organization_id=SESSION_DETAILS.org_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            return _request_error_to_tool_error(exc)
        self._clear_space_cache()
        self._remember_space(created)
        return {"status": "created", "space": created}

    def _ensure_commission_membership(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        listed = self.list_space_members(space_id=space_id)
        if _is_tool_error(listed):
            return listed
        if any(
            _member_matches_assistant(member, assistant_id=assistant_id)
            for member in listed
        ):
            return {"status": "already_member"}
        try:
            unify.add_space_member(
                space_id=space_id,
                assistant_id=assistant_id,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError as exc:
            error = _request_error_to_tool_error(exc)
            if (
                error.get("error_kind") == "conflict"
                and "already"
                in _tool_error_message(
                    error,
                ).lower()
            ):
                return {"status": "already_member"}
            return error
        return {"status": "added"}

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the Coordinator-only tools for the slow-brain loop."""

        return {
            "create_assistant": self.create_assistant,
            "delete_assistant": self.delete_assistant,
            "update_assistant_config": self.update_assistant_config,
            "list_assistants": self.list_assistants,
            "list_org_members": self.list_org_members,
            "pre_seed_colleague": self.pre_seed_colleague,
            "create_space": self.create_space,
            "delete_space": self.delete_space,
            "update_space": self.update_space,
            "add_space_member": self.add_space_member,
            "remove_space_member": self.remove_space_member,
            "list_spaces": self.list_spaces,
            "list_space_members": self.list_space_members,
            "list_spaces_for_assistant": self.list_spaces_for_assistant,
            "commission_colleague_into_workspace": (
                self.commission_colleague_into_workspace
            ),
            "invite_assistant_to_space": self.invite_assistant_to_space,
            "cancel_space_invitation": self.cancel_space_invitation,
            "list_pending_invitations": self.list_pending_invitations,
            "set_setup_state": self.set_setup_state,
            "add_setup_checklist_item": self.add_setup_checklist_item,
            "update_setup_checklist_item": self.update_setup_checklist_item,
        }

    def _assistant_is_reachable(self, agent_id: int) -> bool | ToolError:
        if str(agent_id) in self._known_assistant_ids:
            return True
        assistants = self._assistant_cache
        if assistants is None:
            listed = self.list_assistants(agent_id=agent_id)
            if _is_tool_error(listed):
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
            if _is_tool_error(listed):
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

    def _publish_activity(
        self,
        *,
        phase: CoordinatorActivityPhase,
        stage: CoordinatorActivityStage,
        title: str,
        surfaces: Sequence[CoordinatorActivitySurface],
        related_entities: Sequence[CoordinatorActivityEntity | dict[str, Any]] = (),
        activity_id: str | None = None,
    ) -> str:
        resolved_activity_id = activity_id or coordinator_activity_id("setup")
        publish_coordinator_activity(
            phase=phase,
            stage=stage,
            title=title,
            surfaces=surfaces,
            related_entities=related_entities,
            activity_id=resolved_activity_id,
            correlation_id=resolved_activity_id,
        )
        if phase in {"started", "progress", "needs_input", "blocked"}:
            self._activity_metadata[resolved_activity_id] = (
                list(dict.fromkeys(surfaces)),
                list(related_entities),
            )
        elif phase in {"completed", "failed"}:
            self._activity_metadata.pop(resolved_activity_id, None)
        return resolved_activity_id

    def _publish_failure(
        self,
        activity_id: str,
        *,
        title: str,
        error: ToolError,
    ) -> None:
        surfaces, related_entities = self._activity_metadata.pop(
            activity_id,
            ([], []),
        )
        publish_coordinator_activity(
            phase="failed",
            stage="implementation",
            title=title,
            surfaces=surfaces,
            related_entities=related_entities,
            activity_id=activity_id,
            correlation_id=activity_id,
            status="error",
            error=_tool_error_message(error),
        )


def _tool_conflict(*, message: str, details: dict[str, Any]) -> ToolError:
    return {
        "error_kind": "conflict",
        "message": message,
        "details": details,
    }


def _is_tool_error(value: Any) -> bool:
    return isinstance(value, dict) and "error_kind" in value


def _normalize_lookup_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _assistant_name_matches(
    assistant: dict[str, Any],
    *,
    first_name: str,
    surname: str | None,
) -> bool:
    candidate = _normalize_lookup_text(
        f"{assistant.get('first_name') or assistant.get('firstName') or ''} "
        f"{assistant.get('surname') or assistant.get('last_name') or assistant.get('lastName') or ''}",
    )
    target = _normalize_lookup_text(f"{first_name} {surname or ''}")
    return bool(target) and candidate == target


def _space_name_matches(space: dict[str, Any], *, space_name: str) -> bool:
    return _normalize_lookup_text(space.get("name")) == _normalize_lookup_text(
        space_name,
    )


def _member_matches_assistant(member: dict[str, Any], *, assistant_id: int) -> bool:
    member_assistant_id = member.get("agent_id")
    if member_assistant_id is None:
        member_assistant_id = member.get("assistant_id")
    return str(member_assistant_id) == str(assistant_id)


def _assistant_not_found(agent_id: int) -> ToolError:
    """Build a tool error for assistant ids outside the reachable set."""

    return {
        "error_kind": "not_found",
        "message": f"Assistant {agent_id} is not reachable by this Coordinator.",
        "details": {"agent_id": agent_id},
    }


def _display_name(*, first_name: object, surname: object | None = None) -> str:
    name = " ".join(str(part or "").strip() for part in (first_name, surname)).strip()
    return safe_activity_text(name, fallback="New colleague")


def _assistant_display_name(assistant: dict[str, Any]) -> str:
    first_name = assistant.get("first_name") or assistant.get("firstName")
    surname = (
        assistant.get("surname")
        or assistant.get("last_name")
        or assistant.get("lastName")
    )
    return _display_name(first_name=first_name, surname=surname)


def _assistant_entity(
    assistant: dict[str, Any],
    *,
    fallback: str,
):
    return activity_entity(
        "colleague",
        name=_assistant_display_name(assistant) or fallback,
        entity_id=assistant.get("agent_id")
        or assistant.get("agentId")
        or assistant.get("id"),
    )


def _space_display_name(space: dict[str, Any]) -> str:
    return safe_activity_text(space.get("name"), fallback="Workspace")


def _space_entity(
    space: dict[str, Any],
    *,
    fallback: str,
):
    return activity_entity(
        "workspace",
        name=_space_display_name(space) or fallback,
        entity_id=space.get("space_id") or space.get("spaceId") or space.get("id"),
    )


def _membership_entities(space_id: int, assistant_id: int):
    return [
        activity_entity("workspace", name="Workspace", entity_id=space_id),
        activity_entity("colleague", name="Colleague", entity_id=assistant_id),
    ]


def _invitation_entity(invitation: dict[str, Any]):
    return activity_entity(
        "invitation",
        name="Invitation",
        entity_id=(
            invitation.get("invite_id")
            or invitation.get("inviteId")
            or invitation.get("invitation_id")
            or invitation.get("id")
        ),
    )


def _surfaces_for_preseed(
    writes: Sequence[CoordinatorPreseedWrite | dict[str, Any]],
) -> list[CoordinatorActivitySurface]:
    surfaces: list[CoordinatorActivitySurface] = []
    for write in writes:
        context = (
            write.context
            if isinstance(write, CoordinatorPreseedWrite)
            else write.get("context")
        )
        context_name = str(context or "").lower()
        if context_name.startswith("tasks"):
            surfaces.append("tasks")
        elif context_name.startswith("knowledge"):
            surfaces.append("memory")
        elif context_name.startswith("guidance"):
            surfaces.append("guidance")
        elif context_name.startswith("dashboards"):
            surfaces.append("dashboards")
        elif context_name.startswith("functions"):
            surfaces.append("functions")
        elif context_name.startswith("data"):
            surfaces.append("data")
    return list(dict.fromkeys(surfaces or ["memory"]))


def _tool_error_message(error: ToolError) -> str:
    return safe_activity_text(
        error.get("message"),
        fallback="The setup step could not finish.",
    )


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
    message = response_text
    details: dict[str, Any] = {"status_code": status_code}

    detail_payload: Any = None
    if response is not None:
        try:
            response_payload = response.json()
        except Exception:
            response_payload = None
        if isinstance(response_payload, dict):
            detail_payload = response_payload.get("detail")

    if isinstance(detail_payload, dict):
        message = safe_activity_text(
            detail_payload.get("message"),
            fallback=safe_activity_text(
                detail_payload.get("error"),
                fallback=response_text,
            ),
        )
        existing_id = detail_payload.get("existing_id")
        if isinstance(existing_id, int):
            details["existing_id"] = existing_id
        error_code = detail_payload.get("error")
        if isinstance(error_code, str) and error_code.strip():
            details["error"] = error_code
        nested_details = detail_payload.get("details")
        if isinstance(nested_details, dict):
            details.update(nested_details)
    elif isinstance(detail_payload, str) and detail_payload.strip():
        message = detail_payload

    return {
        "error_kind": error_kind,
        "message": message,
        "details": details,
    }
