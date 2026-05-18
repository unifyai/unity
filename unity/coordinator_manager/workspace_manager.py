"""Coordinator workspace primitives exposed under ``primitives.coordinator``.

The coordinator slow-brain should delegate setup and workspace mutation work to
the actor loop. The actor accesses these methods through the scoped primitives
runtime, which guarantees consistent reachability checks and activity tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from unity.common.tool_outcome import ToolError
from unity.conversation_manager.domains.coordinator_tools import (
    AssistantCreateConfig,
    AssistantConfigPatch,
    ChecklistItemStatus,
    CoordinatorPreseedWriteValue,
    COORDINATOR_TOOL_METHOD_NAMES,
    CoordinatorTools,
    InviteOrgRoleName,
    SpacePatch,
)
from unity.manager_registry import SingletonABCMeta
from unity.session_details import SESSION_DETAILS


def _coordinator_role_required_error() -> ToolError:
    return {
        "error_kind": "permission_denied",
        "message": "Coordinator workspace primitives are only available in Coordinator sessions.",
        "details": {"is_coordinator": bool(SESSION_DETAILS.is_coordinator)},
    }


class CoordinatorWorkspaceManager(metaclass=SingletonABCMeta):
    """Coordinator-only workspace lifecycle primitive owner."""

    _PRIMITIVE_METHODS = COORDINATOR_TOOL_METHOD_NAMES

    @property
    def _delegate(self) -> CoordinatorTools:
        """Return a short-lived delegate to avoid stale reachability caches."""
        return CoordinatorTools(cm=None)

    @staticmethod
    def _require_coordinator_role() -> ToolError | None:
        if SESSION_DETAILS.is_coordinator:
            return None
        return _coordinator_role_required_error()

    def create_assistant(
        self,
        *,
        first_name: str,
        surname: str | None = None,
        about: str,
        job_title: str | None = None,
        timezone: str | None = None,
        nationality: str | None = None,
        config: AssistantCreateConfig | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a colleague assistant in the active coordinator workspace.

        Use this when the user has confirmed the colleague profile and asked the
        Coordinator to actually provision the assistant now. The tool merges the
        provided profile fields with coordinator-derived defaults (timezone,
        nationality, and inferred role metadata), then creates the assistant in
        the current organization scope.

        Parameters
        ----------
        first_name : str
            Colleague first name to provision.
        surname : str | None, optional
            Optional colleague surname.
        about : str
            Required profile summary describing the colleague role.
        job_title : str | None, optional
            Optional explicit role title for the colleague profile.
        timezone : str | None, optional
            Optional IANA timezone override for this colleague.
        nationality : str | None, optional
            Optional nationality metadata for profile defaults.
        config : AssistantCreateConfig | None, optional
            Structured assistant-create overrides merged with coordinator
            defaults (for example ``create_infra`` and ``is_local``).
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.create_assistant(
            first_name=first_name,
            surname=surname,
            about=about,
            job_title=job_title,
            timezone=timezone,
            nationality=nationality,
            config=config,
        )

    def delete_assistant(
        self,
        *,
        agent_id: int,
    ) -> dict[str, Any] | str | ToolError:
        """Delete a reachable colleague assistant from the coordinator workspace.

        Use this only for explicitly confirmed destructive actions. The tool
        first verifies that the target assistant is reachable from the current
        coordinator scope, then performs deletion and returns structured errors
        when the target is missing or inaccessible.

        Parameters
        ----------
        agent_id : int
            Assistant identifier to delete after explicit confirmation.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.delete_assistant(agent_id=agent_id)

    def update_assistant_config(
        self,
        *,
        agent_id: int,
        config: AssistantConfigPatch,
    ) -> dict[str, Any] | ToolError:
        """Update config fields for a reachable colleague assistant.

        Use this after the user has confirmed concrete profile or behavior
        edits (for example bio, role, timezone, or other config-backed fields).
        This mutates an existing assistant only; use ``create_assistant`` when
        the colleague does not yet exist.

        Parameters
        ----------
        agent_id : int
            Target colleague assistant identifier.
        config : AssistantConfigPatch
            Structured profile patch payload. Preferred keys include
            ``first_name``, ``surname``, ``about``, ``job_title``,
            ``timezone``, ``nationality``, ``weekly_limit``,
            ``max_parallel``, ``desktop_mode``, and voice fields
            (``voice_id`` with ``voice_provider``).
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.update_assistant_config(agent_id=agent_id, config=config)

    def list_assistants(
        self,
        *,
        phone: str | None = None,
        email: str | None = None,
        agent_id: int | None = None,
    ) -> list[dict[str, Any]] | ToolError:
        """List assistants visible inside the coordinator's reachable scope.

        Use this before assistant/workspace mutations to resolve ids, confirm
        whether a colleague already exists, and disambiguate by phone, email, or
        ``agent_id``. Unfiltered calls also refresh the local reachability cache
        used by other coordinator tools in the same turn.

        Parameters
        ----------
        phone : str | None, optional
            Optional phone filter for narrow assistant lookup.
        email : str | None, optional
            Optional email filter for narrow assistant lookup.
        agent_id : int | None, optional
            Optional assistant id filter for exact match lookup.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.list_assistants(
            phone=phone,
            email=email,
            agent_id=agent_id,
        )

    def list_org_members(self) -> list[dict[str, Any]] | ToolError:
        """List authorized human organization members for membership targeting.

        Use this when membership actions target ``member_user_id`` (human org
        users) rather than assistant ids. This helps validate that a referenced
        org member is authorized and reachable before attempting workspace
        membership mutations.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.list_org_members()

    def invite_org_member(
        self,
        *,
        email: str,
        role_name: InviteOrgRoleName | None = None,
    ) -> dict[str, Any] | ToolError:
        """Invite a human member into the active organization by email.

        Use this when onboarding or team setup requires inviting someone who is
        not yet a member of the organization.

        Parameters
        ----------
        email : str
            Email address to invite into the organization.
        role_name : InviteOrgRoleName | None, optional
            Optional role to assign on acceptance. Accepted values are
            ``"Admin"``, ``"Member"``, and ``"Viewer"``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.invite_org_member(
            email=email,
            role_name=role_name,
        )

    def pre_seed_colleague(
        self,
        *,
        target_assistant_id: int,
        writes: list[CoordinatorPreseedWriteValue],
    ) -> dict[str, Any] | ToolError:
        """Seed colleague-owned setup rows into assistant-private contexts.

        Use this when setup should belong to a specific colleague's private
        contexts (for example ``Tasks``, ``Knowledge``, ``Guidance``, or other
        assistant-owned surfaces). This is the right tool for "this colleague
        should own this workflow" decisions after confirmation.

        Parameters
        ----------
        target_assistant_id : int
            Assistant id that should own the seeded rows.
        writes : list[CoordinatorPreseedWriteValue]
            Relative context + entries batches to persist for this colleague.
            Each write follows ``{"context": "...", "entries": [...]}``.
            Allowed contexts are:
            ``Tasks``, ``Knowledge``, ``Guidance``, ``Data``,
            ``Functions/<name>``, and ``Dashboards/<name>``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.pre_seed_colleague(
            target_assistant_id=target_assistant_id,
            writes=writes,
        )

    def create_space(
        self,
        *,
        name: str,
        description: str,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Create a shared workspace in the coordinator organization.

        Use this when the user has agreed on a concrete workspace name/purpose
        and wants the space created now. This tool creates the workspace object;
        membership is a separate step via ``add_space_member`` unless a single
        composite provisioning step is better.

        Parameters
        ----------
        name : str
            Workspace name to create.
        description : str
            Human-readable summary of workspace intent.
        organization_id : int | None, optional
            Reserved compatibility field; runtime scope controls organization.
        owner_user_id : str | None, optional
            Reserved compatibility field; ownership derives from coordinator scope.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.create_space(
            name=name,
            description=description,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
        )

    def delete_space(self, *, space_id: int) -> dict[str, Any] | ToolError:
        """Delete a reachable shared workspace after explicit confirmation.

        Use only after explicit destructive confirmation for the exact target
        workspace. The tool validates reachability first and returns structured
        errors when the workspace is missing or outside coordinator scope.

        Parameters
        ----------
        space_id : int
            Workspace identifier to delete.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.delete_space(space_id=space_id)

    def update_space(
        self,
        *,
        space_id: int,
        patch: SpacePatch,
    ) -> dict[str, Any] | ToolError:
        """Apply metadata updates to a reachable shared workspace.

        Use this for confirmed workspace field edits (for example name or
        description changes). This tool mutates workspace properties only; use
        membership tools for adding/removing colleagues from the workspace.

        Parameters
        ----------
        space_id : int
            Workspace identifier to update.
        patch : SpacePatch
            Partial update payload for editable workspace fields.
            Accepted keys are ``name`` and ``description``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.update_space(space_id=space_id, patch=patch)

    def add_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int | None = None,
        member_user_id: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Add one assistant or one organization member to a workspace.

        Use this when the user has confirmed a workspace membership change.
        Provide exactly one selector:
        - ``assistant_id`` to add an assistant colleague
        - ``member_user_id`` to add an authorized human org member

        Parameters
        ----------
        space_id : int
            Workspace receiving the membership grant.
        assistant_id : int | None, optional
            Assistant id to add as a member.
        member_user_id : str | None, optional
            Organization user id to add as a human member.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.add_space_member(
            space_id=space_id,
            assistant_id=assistant_id,
            member_user_id=member_user_id,
        )

    def remove_space_member(
        self,
        *,
        space_id: int,
        assistant_id: int,
    ) -> dict[str, Any] | ToolError:
        """Remove a colleague assistant from a shared workspace.

        Use this only after the user has explicitly confirmed membership removal.
        The tool validates both workspace and assistant reachability before
        mutation so failures return clear, actionable error envelopes.

        Parameters
        ----------
        space_id : int
            Workspace from which membership should be removed.
        assistant_id : int
            Assistant identifier to remove from membership.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.remove_space_member(
            space_id=space_id,
            assistant_id=assistant_id,
        )

    def list_spaces(
        self,
        *,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]] | ToolError:
        """List shared spaces visible to the active coordinator.

        Use this to resolve ``space_id`` values, verify workspace existence, and
        avoid duplicate space creation before mutating workspace metadata or
        membership.

        Parameters
        ----------
        organization_id : int | None, optional
            Reserved compatibility field; visibility comes from coordinator scope.
        owner_user_id : str | None, optional
            Reserved compatibility field for caller parity.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.list_spaces(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
        )

    def list_space_members(
        self,
        *,
        space_id: int,
    ) -> list[dict[str, Any]] | ToolError:
        """List members currently attached to a reachable workspace.

        Use this before add/remove membership mutations to verify current
        membership, prevent duplicate operations, and confirm who already has
        workspace access.

        Parameters
        ----------
        space_id : int
            Workspace identifier whose membership should be listed.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.list_space_members(space_id=space_id)

    def list_spaces_for_assistant(
        self,
        *,
        assistant_id: int,
    ) -> list[dict[str, Any]] | ToolError:
        """List shared spaces currently attached to a colleague assistant.

        Use this when auditing a colleague's workspace footprint before changing
        access, performing cleanup, or explaining current ownership boundaries.

        Parameters
        ----------
        assistant_id : int
            Assistant identifier whose workspace memberships should be listed.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.list_spaces_for_assistant(assistant_id=assistant_id)

    def commission_colleague_into_workspace(
        self,
        *,
        assistant_first_name: str,
        assistant_surname: str | None = None,
        space_name: str,
        space_description: str,
        assistant_about: str | None = None,
        assistant_job_title: str | None = None,
        assistant_timezone: str | None = None,
        assistant_nationality: str | None = None,
        assistant_config: AssistantCreateConfig | None = None,
        assistant_id: int | None = None,
        space_id: int | None = None,
    ) -> dict[str, Any] | ToolError:
        """Commission a colleague into a workspace with idempotent step reporting.

        Use this when the user has confirmed "set up this colleague in this
        workspace" and the slow-brain should avoid partial primitive sequencing.
        This tool resolves or creates the colleague, resolves or creates the
        workspace, then ensures the colleague is a member.

        Parameters
        ----------
        assistant_first_name : str
            Colleague first name for lookup or creation.
        assistant_surname : str | None, optional
            Colleague surname for lookup or creation.
        space_name : str
            Workspace name for lookup or creation.
        space_description : str
            Workspace description when creation is required.
        assistant_about : str | None, optional
            Required profile summary when a new assistant must be created.
        assistant_job_title : str | None, optional
            Optional explicit colleague job title.
        assistant_timezone : str | None, optional
            Optional explicit colleague timezone.
        assistant_nationality : str | None, optional
            Optional explicit colleague nationality.
        assistant_config : AssistantCreateConfig | None, optional
            Additional structured assistant-create overrides merged when
            creation is needed.
        assistant_id : int | None, optional
            Optional explicit assistant id to reuse instead of name lookup.
        space_id : int | None, optional
            Optional explicit workspace id to reuse instead of name lookup.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.commission_colleague_into_workspace(
            assistant_first_name=assistant_first_name,
            assistant_surname=assistant_surname,
            space_name=space_name,
            space_description=space_description,
            assistant_about=assistant_about,
            assistant_job_title=assistant_job_title,
            assistant_timezone=assistant_timezone,
            assistant_nationality=assistant_nationality,
            assistant_config=assistant_config,
            assistant_id=assistant_id,
            space_id=space_id,
        )

    def set_setup_state(
        self,
        *,
        mode: Literal["active", "ready_to_go"],
        ready_at: datetime | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Update the coordinator setup lifecycle state for the workspace.

        Use this when onboarding progress changes between active setup work and
        ready-to-go handoff. Optional chat prompt fields attach a suggested next
        user action that can appear as a CTA in the setup experience.

        Parameters
        ----------
        mode : Literal["active", "ready_to_go"]
            Coordinator setup lifecycle mode to persist.
        ready_at : datetime | None, optional
            Optional explicit ready timestamp for handoff transitions.
        chat_prompt : str | None, optional
            Optional CTA prompt text for user follow-up guidance.
        chat_prompt_label : str | None, optional
            Optional CTA button label paired with ``chat_prompt``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.set_setup_state(
            mode=mode,
            ready_at=ready_at,
            chat_prompt=chat_prompt,
            chat_prompt_label=chat_prompt_label,
        )

    def add_setup_checklist_item(
        self,
        *,
        title: str,
        status: ChecklistItemStatus | None = None,
        description: str | None = None,
        kind: str | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Add a setup checklist row to the coordinator onboarding plan.

        Use this when a newly discovered onboarding requirement should be tracked
        explicitly in the Coordinator checklist. Optionally set initial status,
        description, step kind, and a suggested chat CTA for the user.

        Parameters
        ----------
        title : str
            User-visible checklist item title.
        status : ChecklistItemStatus | None, optional
            Optional initial checklist status (``pending``, ``done``,
            or ``skipped``).
        description : str | None, optional
            Optional detail text for the checklist row.
        kind : str | None, optional
            Optional checklist row category value.
        chat_prompt : str | None, optional
            Optional CTA prompt associated with the checklist row.
        chat_prompt_label : str | None, optional
            Optional CTA label paired with ``chat_prompt``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.add_setup_checklist_item(
            title=title,
            status=status,
            description=description,
            kind=kind,
            chat_prompt=chat_prompt,
            chat_prompt_label=chat_prompt_label,
        )

    def update_setup_checklist_item(
        self,
        *,
        item_id: int,
        status: ChecklistItemStatus | None = None,
        title: str | None = None,
        description: str | None = None,
        kind: str | None = None,
        chat_prompt: str | None = None,
        chat_prompt_label: str | None = None,
    ) -> dict[str, Any] | ToolError:
        """Update an existing setup checklist row in coordinator state.

        Use this when progress changes for a known checklist row (status changes,
        title/description edits, or CTA updates). This keeps setup bookkeeping
        aligned with completed validation and remaining onboarding work.

        Parameters
        ----------
        item_id : int
            Checklist row identifier to update.
        status : ChecklistItemStatus | None, optional
            Optional status update (``pending``, ``done``, or ``skipped``).
        title : str | None, optional
            Optional replacement title text.
        description : str | None, optional
            Optional replacement description text.
        kind : str | None, optional
            Optional replacement row category.
        chat_prompt : str | None, optional
            Optional CTA prompt update for this row.
        chat_prompt_label : str | None, optional
            Optional CTA label update paired with ``chat_prompt``.
        """
        permission_error = self._require_coordinator_role()
        if permission_error is not None:
            return permission_error
        return self._delegate.update_setup_checklist_item(
            item_id=item_id,
            status=status,
            title=title,
            description=description,
            kind=kind,
            chat_prompt=chat_prompt,
            chat_prompt_label=chat_prompt_label,
        )
