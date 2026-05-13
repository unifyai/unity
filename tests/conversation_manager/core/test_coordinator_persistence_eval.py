from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
import requests
import unify
from unify.utils import http
from unify.utils.helpers import _create_request_header
from unify.utils.http import RequestError

from tests.conversation_manager.core.test_coordinator_product_literacy_eval import (
    _BOSS_CONTACT,
    _PRIMARY_LLM_CONFIG,
    _RecordingTools,
    CoordinatorScenario,
    DialogueTurn,
    _format_failure,
    _run_target_decision,
    _tool_payloads,
)
from tests.destination_routing_helpers import (
    assert_tool_destination,
    run_direct_routing_loop,
)
from unity.common.context_registry import ContextRegistry
from unity.common.llm_helpers import methods_to_tool_dict
from unity.conversation_manager.domains.coordinator_tools import (
    CoordinatorPreseedWrite,
    CoordinatorTools,
)
from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.session_details import SESSION_DETAILS, AssistantDetails, SpaceSummary

pytestmark = [pytest.mark.eval, pytest.mark.integration, pytest.mark.llm_call]

_ASSISTANTS_PROJECT_NAME = "Assistants"
_MEMORY_CONTEXTS = {"Guidance", "Knowledge"}


@dataclass(frozen=True)
class LiveOrganization:
    """Disposable organization in the configured Orchestra environment."""

    organization_id: int
    api_key: str
    coordinator: dict[str, Any]


@pytest.fixture(autouse=True)
def _reset_runtime_context() -> Iterator[None]:
    SESSION_DETAILS.reset()
    ContextRegistry.clear()
    try:
        unify.unset_context()
    except Exception:
        pass
    try:
        yield
    finally:
        SESSION_DETAILS.reset()
        ContextRegistry.clear()
        try:
            unify.unset_context()
        except Exception:
            pass


class _AssistantAwareRecordingTools(_RecordingTools):
    """Side-effect-free Coordinator tools backed by configured assistant ids."""

    def __init__(self, assistants: list[dict[str, Any]]) -> None:
        self._assistants = assistants

    def list_assistants(
        self,
        *,
        phone: str | None = None,
        email: str | None = None,
        agent_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List assistants visible to the Coordinator owner."""

        del phone, email
        if agent_id is None:
            return list(self._assistants)
        return [
            assistant
            for assistant in self._assistants
            if int(assistant["agent_id"]) == int(agent_id)
        ]


class _WorkspaceAwareRecordingTools(_AssistantAwareRecordingTools):
    """Side-effect-free Coordinator tools backed by the live test ids."""

    def __init__(
        self,
        *,
        assistants: list[dict[str, Any]],
        spaces: list[dict[str, Any]],
        memberships: dict[int, list[dict[str, Any]]],
    ) -> None:
        super().__init__(assistants)
        self._spaces = spaces
        self._memberships = memberships

    def list_spaces(
        self,
        *,
        organization_id: int | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List spaces visible to the Coordinator owner."""

        del organization_id, owner_user_id
        return list(self._spaces)

    def list_space_members(self, *, space_id: int) -> list[dict[str, Any]]:
        """List live assistant members for a reachable space."""

        return list(self._memberships.get(int(space_id), []))


def _headers(api_key: str) -> dict[str, str]:
    return _create_request_header(api_key)


def _require_live_orchestra_url() -> str:
    orchestra_url = os.environ.get("ORCHESTRA_URL")
    if not orchestra_url:
        pytest.skip("Coordinator persistence evals require ORCHESTRA_URL")
    return orchestra_url.rstrip("/")


def _require_user_key() -> str:
    user_key = os.environ.get("UNIFY_KEY")
    if not user_key:
        pytest.skip("UNIFY_KEY is required")
    return user_key


def _require_admin_key() -> str:
    admin_key = os.environ.get("COORDINATOR_TEST_ADMIN_KEY") or os.environ.get(
        "ORCHESTRA_ADMIN_KEY",
    )
    if not admin_key:
        pytest.skip("COORDINATOR_TEST_ADMIN_KEY or ORCHESTRA_ADMIN_KEY is required")
    return admin_key


def _delete_test_organization(
    base_url: str,
    organization_id: int,
    api_key: str,
) -> None:
    response = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = http.delete(
                f"{base_url}/organizations/{organization_id}",
                headers=_headers(api_key),
                raise_for_status=False,
                timeout=30,
            )
        except (RequestError, requests.exceptions.RequestException) as exc:
            last_error = exc
            response = None
        else:
            if response.status_code in {204, 404}:
                return
        if attempt < 2:
            time.sleep(1)
    assert response is not None, last_error
    assert response.status_code in {204, 404}, response.text


def _credit_test_organization(
    *,
    base_url: str,
    organization_id: int,
    admin_key: str,
) -> None:
    response = http.post(
        f"{base_url}/admin/create_recharge",
        headers=_headers(admin_key),
        json={
            "organization_id": organization_id,
            "quantity": 10,
            "type": "promo",
        },
        timeout=30,
    )
    assert response.status_code < 300, response.text


@contextmanager
def _managed_test_organization() -> Iterator[LiveOrganization]:
    base_url = _require_live_orchestra_url()
    admin_key = _require_admin_key()
    user_key = _require_user_key()
    try:
        owner = unify.get_user_basic_info(api_key=user_key)
    except (RequestError, requests.exceptions.RequestException) as exc:
        pytest.skip(f"Coordinator persistence eval needs a valid user key: {exc}")
    org_name = f"Coordinator Eval {uuid.uuid4().hex[:12]}"

    response = http.post(
        f"{base_url}/organizations",
        headers=_headers(user_key),
        json={
            "name": org_name,
            "timezone": owner.get("timezone") or "UTC",
        },
        timeout=30,
    )
    assert response.status_code == 201, response.text
    organization = response.json()
    organization_id = int(organization["id"])
    _credit_test_organization(
        base_url=base_url,
        organization_id=organization_id,
        admin_key=admin_key,
    )

    try:
        coordinator = {
            "agent_id": int(organization["coordinator_id"]),
            "first_name": "Avery",
            "surname": "Coordinator",
            "is_coordinator": True,
            "user_id": owner["user_id"],
        }
        yield LiveOrganization(
            organization_id=organization_id,
            api_key=organization["api_key"],
            coordinator=coordinator,
        )
    finally:
        _delete_test_organization(base_url, organization_id, user_key)


@contextmanager
def _organization_api_key(api_key: str) -> Iterator[None]:
    previous = os.environ.get("UNIFY_KEY")
    os.environ["UNIFY_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNIFY_KEY", None)
        else:
            os.environ["UNIFY_KEY"] = previous


def _create_test_assistant(
    *,
    first_name: str,
    organization: LiveOrganization,
) -> dict[str, Any]:
    return unify.create_assistant(
        first_name=first_name,
        surname="Ops",
        config={
            "create_infra": False,
            "is_local": True,
            "timezone": "UTC",
        },
        api_key=organization.api_key,
    )


def _configure_session(
    *,
    organization: LiveOrganization,
    coordinator: dict[str, Any],
    spaces: list[SpaceSummary] | None = None,
) -> None:
    SESSION_DETAILS.unify_key = organization.api_key
    SESSION_DETAILS.org_id = organization.organization_id
    SESSION_DETAILS.assistant = AssistantDetails(
        agent_id=int(coordinator["agent_id"]),
        first_name=coordinator.get("first_name") or "Avery",
        surname=coordinator.get("surname") or "Coordinator",
        is_coordinator=True,
    )
    SESSION_DETAILS.user.id = coordinator["user_id"]
    SESSION_DETAILS.user.first_name = _BOSS_CONTACT["first_name"]
    SESSION_DETAILS.user.surname = _BOSS_CONTACT["surname"]
    SESSION_DETAILS.space_ids = [space.space_id for space in spaces or []]
    SESSION_DETAILS.space_summaries = spaces or []


def _activate_assistant_context(
    *,
    organization: LiveOrganization,
    assistant: dict[str, Any],
) -> None:
    unify.activate(
        _ASSISTANTS_PROJECT_NAME,
        overwrite=False,
        api_key=organization.api_key,
    )
    unify.set_context(f"{assistant['user_id']}/{assistant['agent_id']}", relative=False)


def _normalize_selected_writes(writes: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for write in writes:
        if isinstance(write, CoordinatorPreseedWrite):
            normalized.append(write.model_dump())
        else:
            normalized.append(dict(write))
    return normalized


def _row_entries_text(rows: list[Any]) -> str:
    return json.dumps([row.entries for row in rows], sort_keys=True).lower()


def _single_tool_call(result, tool_name: str):
    calls = [tool for tool in result.tools if tool.name == tool_name]
    assert len(calls) == 1, json.dumps(_tool_payloads(result), indent=2)
    return calls[0]


def _logs(context: str, organization: LiveOrganization) -> list[Any]:
    return unify.get_logs(
        project=_ASSISTANTS_PROJECT_NAME,
        context=context,
        api_key=organization.api_key,
        limit=100,
    )


@pytest.mark.asyncio
async def test_coordinator_persists_confirmed_colleague_setup_rows():
    """Natural Coordinator text can select and persist colleague-owned rows."""

    with _managed_test_organization() as organization:
        assistant = _create_test_assistant(
            first_name=f"Revenue{uuid.uuid4().hex[:10]}",
            organization=organization,
        )
        coordinator = organization.coordinator
        assistant_id = int(assistant["agent_id"])
        assistant_user_id = assistant["user_id"]

        scenario = CoordinatorScenario(
            scenario_id="live-colleague-setup-persistence",
            title="Confirmed colleague setup persists",
            business_context=(
                "A B2B SaaS founder has confirmed that a Revenue Ops colleague "
                "should own renewal-risk monitoring and blocked-account guidance."
            ),
            turns=(
                DialogueTurn(
                    "user",
                    f"Revenue Ops is assistant {assistant_id}. We already agreed "
                    "it should own the weekday renewal-risk summary and remember "
                    "how to check blocked enterprise accounts. Please set that "
                    "up on Revenue Ops now.",
                    new=True,
                ),
            ),
            masked_components=(
                "The target colleague id is explicitly supplied.",
                "No shared team space has been requested.",
            ),
            rubric=(
                "The response should use `pre_seed_colleague` for the supplied "
                "Revenue Ops assistant, with a `Tasks` entry for the weekday "
                "renewal-risk summary and a `Guidance` or `Knowledge` entry for "
                "blocked enterprise account checks."
            ),
            required_tools=frozenset({"pre_seed_colleague"}),
            forbidden_tools=frozenset({"act", "create_space", "add_space_member"}),
        )

        selection = await _run_target_decision(
            scenario=scenario,
            llm_config=dict(_PRIMARY_LLM_CONFIG),
            tool_source=_AssistantAwareRecordingTools(
                assistants=[
                    {
                        "agent_id": assistant_id,
                        "first_name": assistant["first_name"],
                        "surname": assistant.get("surname"),
                    },
                ],
            ),
        )
        called_tools = {tool.name for tool in selection.tools}
        assert "pre_seed_colleague" in called_tools, _format_failure(
            scenario,
            selection,
        )
        assert not (called_tools & scenario.forbidden_tools), _format_failure(
            scenario,
            selection,
        )

        selected_call = _single_tool_call(selection, "pre_seed_colleague")
        assert int(selected_call.args["target_assistant_id"]) == assistant_id
        selected_writes = _normalize_selected_writes(selected_call.args["writes"])
        selected_contexts = {write["context"] for write in selected_writes}
        assert "Tasks" in selected_contexts, json.dumps(selected_writes, indent=2)
        assert selected_contexts & _MEMORY_CONTEXTS, json.dumps(
            selected_writes,
            indent=2,
        )

        _configure_session(organization=organization, coordinator=coordinator)
        persisted = CoordinatorTools(cm=object()).pre_seed_colleague(
            target_assistant_id=assistant_id,
            writes=selected_writes,
        )
        assert "error_kind" not in persisted, persisted
        coordinator_id = int(persisted["coordinator_id"])

        persisted_contexts = [write["context"] for write in persisted["writes"]]
        assert all(
            context.startswith(f"{assistant_user_id}/{assistant_id}/")
            for context in persisted_contexts
        )

        rows_by_context = {
            context: _logs(context, organization) for context in persisted_contexts
        }
        assert all(rows for rows in rows_by_context.values())

        task_rows = rows_by_context[f"{assistant_user_id}/{assistant_id}/Tasks"]
        assert all(
            row.entries["_assistant_id"] == str(assistant_id) for row in task_rows
        )
        assert all(
            row.entries["authoring_assistant_id"] == coordinator_id
            for rows in rows_by_context.values()
            for row in rows
        )

        persisted_text = _row_entries_text(
            [row for rows in rows_by_context.values() for row in rows],
        )
        assert "renewal" in persisted_text
        assert "blocked" in persisted_text
        assert "enterprise" in persisted_text


@pytest.mark.asyncio
async def test_coordinator_persists_confirmed_shared_space_guidance():
    """Natural Coordinator text can select shared setup and persist space rows."""

    with _managed_test_organization() as organization:
        suffix = uuid.uuid4().hex[:8]
        revenue = _create_test_assistant(
            first_name=f"Revenue{suffix}",
            organization=organization,
        )
        support = _create_test_assistant(
            first_name=f"Support{suffix}",
            organization=organization,
        )
        coordinator = organization.coordinator
        space_description = (
            "Shared launch coordination memory for revenue operations, "
            "support handoffs, launch SOPs, and escalation rules."
        )
        space = unify.create_space(
            name=f"Launch War Room {suffix}",
            description=space_description,
            api_key=organization.api_key,
        )
        space_id = int(space["space_id"])
        for assistant in (revenue, support):
            unify.add_space_member(
                space_id,
                int(assistant["agent_id"]),
                api_key=organization.api_key,
            )

        sentinel = f"LAUNCH-HANDOFF-{uuid.uuid4().hex[:10]}"
        space_summary = SpaceSummary(
            space_id=space_id,
            name=space["name"],
            description=space_description,
        )
        scenario = CoordinatorScenario(
            scenario_id="live-shared-space-setup-persistence",
            title="Confirmed shared-space setup persists",
            business_context=(
                "A product launch team wants two colleagues to share one launch "
                "handoff SOP from an existing team space."
            ),
            turns=(
                DialogueTurn(
                    "user",
                    f"{space['name']} space {space_id} already has Revenue Ops "
                    f"assistant {revenue['agent_id']} and Support Ops assistant "
                    f"{support['agent_id']} as members. Put the {sentinel} launch "
                    "handoff SOP in that shared space so both colleagues use the "
                    "same source.",
                    new=True,
                ),
            ),
            masked_components=(
                "A reachable shared space id and assistant ids are supplied.",
                "The user says membership is already settled.",
                "The user explicitly wants one shared source across colleagues.",
            ),
            rubric=(
                "The response should treat this as shared-space setup, not "
                "colleague-owned setup. It should use `act` with instructions to "
                "write the handoff SOP into the supplied shared space, and must "
                "not call `pre_seed_colleague`."
            ),
            required_tools=frozenset({"act"}),
            forbidden_tools=frozenset(
                {
                    "pre_seed_colleague",
                    "create_space",
                    "add_space_member",
                },
            ),
            space_summaries=(space_summary,),
        )
        selection_tools = _WorkspaceAwareRecordingTools(
            assistants=[
                {
                    "agent_id": int(revenue["agent_id"]),
                    "first_name": revenue["first_name"],
                    "surname": revenue.get("surname"),
                },
                {
                    "agent_id": int(support["agent_id"]),
                    "first_name": support["first_name"],
                    "surname": support.get("surname"),
                },
            ],
            spaces=[{"space_id": space_id, "name": space["name"]}],
            memberships={
                space_id: [
                    {
                        "space_id": space_id,
                        "assistant_id": int(revenue["agent_id"]),
                        "name": "Revenue Ops",
                    },
                    {
                        "space_id": space_id,
                        "assistant_id": int(support["agent_id"]),
                        "name": "Support Ops",
                    },
                ],
            },
        )

        selection = await _run_target_decision(
            scenario=scenario,
            llm_config=dict(_PRIMARY_LLM_CONFIG),
            tool_source=selection_tools,
        )
        called_tools = {tool.name for tool in selection.tools}
        assert "act" in called_tools, _format_failure(scenario, selection)
        assert not (called_tools & scenario.forbidden_tools), _format_failure(
            scenario,
            selection,
        )

        act_call = _single_tool_call(selection, "act")
        act_query = act_call.args["query"]
        assert sentinel in act_query, _format_failure(scenario, selection)
        assert str(space_id) in act_query or f"space:{space_id}" in act_query

        _configure_session(
            organization=organization,
            coordinator=coordinator,
            spaces=[space_summary],
        )
        with _organization_api_key(organization.api_key):
            _activate_assistant_context(
                organization=organization,
                assistant=coordinator,
            )
            manager = GuidanceManager()
            messages = await run_direct_routing_loop(
                llm_config=dict(_PRIMARY_LLM_CONFIG),
                tools=methods_to_tool_dict(
                    manager.add_guidance,
                    include_class_name=True,
                ),
                accessible_spaces=[space_summary],
                message=act_query,
                loop_id="coordinator-shared-space-persistence",
            )

        assert_tool_destination(messages, "add_guidance", f"space:{space_id}")
        shared_rows = _logs(f"Spaces/{space_id}/Guidance", organization)
        matching_shared = [
            row for row in shared_rows if sentinel in json.dumps(row.entries)
        ]
        assert matching_shared
        assert all(
            row.entries["authoring_assistant_id"] == int(coordinator["agent_id"])
            for row in matching_shared
        )

        personal_rows = _logs(
            f"{coordinator['user_id']}/{coordinator['agent_id']}/Guidance",
            organization,
        )
        assert sentinel not in _row_entries_text(personal_rows)
