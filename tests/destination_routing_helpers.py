from __future__ import annotations

import asyncio
import inspect
import json
import textwrap
import uuid
from collections.abc import Iterable
from typing import Any

import pytest
import unify
from pydantic import BaseModel, Field
from unify.utils.http import RequestError

from tests.async_tool_loop.conftest import LLM_CONFIGS
from unity.blacklist_manager.base import BaseBlackListManager
from unity.common.accessible_spaces_block import build_accessible_spaces_block
from unity.common.async_tool_loop import start_async_tool_loop
from unity.common.context_registry import ContextRegistry
from unity.common.llm_client import new_llm_client
from unity.data_manager.base import BaseDataManager
from unity.file_manager.managers.base import BaseFileManager
from unity.session_details import SESSION_DETAILS, SpaceSummary

PATCH_SPACE_DESTINATION = "space:41001"
FAMILY_SPACE_DESTINATION = "space:41002"
PERSONAL_DESTINATIONS = {None, "", "personal"}

EVAL_SPACE_SUMMARIES = [
    SpaceSummary(
        space_id=41001,
        name="South-East repairs patch",
        description=(
            "Daily operations for Patch-1 supervisors and operatives. Carries "
            "operational data like open work orders, KPIs, and lateness logs; "
            "shared reference files like SOPs and customer briefs; and "
            "team-level blacklist entries the patch has agreed to refuse."
        ),
    ),
    SpaceSummary(
        space_id=41002,
        name="Family logistics",
        description=(
            "Household scheduling, school logistics, family reference files, "
            "and a household-level blacklist of robocallers and scammers that "
            "every adult's phone should refuse."
        ),
    ),
]

ROUTING_TOOL_METHODS = (
    ("FileManager", BaseFileManager.ingest_files),
    ("DataManager", BaseDataManager.insert_rows),
    ("BlackListManager", BaseBlackListManager.create_blacklist_entry),
)


@pytest.fixture(scope="function")
def manager_routing_context(request):
    """Provide an isolated personal root plus one isolated shared space."""

    context = f"tests/destination_routing/{request.node.name}/{uuid.uuid4().hex}"
    space_id = 20_000_000 + uuid.uuid4().int % 1_000_000_000
    ContextRegistry.clear()
    SESSION_DETAILS.reset()
    unify.set_context(context, relative=False)
    SESSION_DETAILS.space_ids = [space_id]
    SESSION_DETAILS.space_summaries = [
        {
            "space_id": space_id,
            "name": "Ops Team",
            "description": "Shared operations workspace for team-visible memory.",
        },
    ]
    yield context, space_id
    for root in (context, f"Spaces/{space_id}"):
        delete_context_tree(root)
    unify.unset_context()
    SESSION_DETAILS.reset()
    ContextRegistry.clear()


@pytest.fixture(params=LLM_CONFIGS)
def llm_config(request) -> dict[str, str]:
    """Return each production-like LLM config used by routing evals."""

    return request.param


class RoutingScenario:
    """Isolated memory roots and shared-space metadata for routing evals."""

    def __init__(self, name: str) -> None:
        unique = uuid.uuid4().hex
        base_id = 40_000_000 + uuid.uuid4().int % 1_000_000_000
        self.context = f"tests/destination_routing_eval/{name}/{unique}"
        self.patch_space_id = base_id
        self.research_space_id = base_id + 1
        self.space_summaries = [
            SpaceSummary(
                space_id=self.patch_space_id,
                name="Patch Reliability",
                description=(
                    "Shared workspace for field dispatch, compressor incidents, "
                    "maintenance runbooks, customer outage triage, and team-visible "
                    "repair automation used by the patch reliability coordinators."
                ),
            ),
            SpaceSummary(
                space_id=self.research_space_id,
                name="Market Research",
                description=(
                    "Shared workspace for competitive research, analyst notes, "
                    "pricing studies, market sizing, and customer interview synthesis."
                ),
            ),
        ]

    def setup(self) -> None:
        ContextRegistry.clear()
        SESSION_DETAILS.reset()
        unify.set_context(self.context, relative=False)
        SESSION_DETAILS.space_ids = [self.patch_space_id, self.research_space_id]
        SESSION_DETAILS.space_summaries = self.space_summaries

    def teardown(self) -> None:
        for root in (
            self.context,
            f"Spaces/{self.patch_space_id}",
            f"Spaces/{self.research_space_id}",
        ):
            delete_context_tree(root)
        unify.unset_context()
        SESSION_DETAILS.reset()
        ContextRegistry.clear()

    @property
    def patch_destination(self) -> str:
        return f"space:{self.patch_space_id}"

    @property
    def research_destination(self) -> str:
        return f"space:{self.research_space_id}"


class DestinationRoutingDecision(BaseModel):
    """LLM-selected manager, tool, and destination for a state write."""

    manager: str = Field(description="The manager surface to use.")
    tool: str = Field(description="The write tool to call, or request_clarification.")
    destination: str | None = Field(
        default=None,
        description='The chosen destination argument, such as "personal" or "space:<id>".',
    )
    clarification_requested: bool = Field(
        default=False,
        description="Whether the model would ask the user to clarify before writing.",
    )
    rationale: str = Field(description="Brief reason for the routing choice.")


def routing_decision_prompt(user_request: str) -> str:
    """Build the shared prompt context for destination-routing evals."""

    tool_descriptions = "\n\n".join(
        f"{manager_name}.{method.__name__}\n"
        f"{inspect.signature(method)}\n"
        f"{inspect.cleandoc(method.__doc__ or '')}"
        for manager_name, method in ROUTING_TOOL_METHODS
    )

    return (
        f"{build_accessible_spaces_block(EVAL_SPACE_SUMMARIES)}\n\n"
        f"Available write tools from the live manager docstrings:\n{tool_descriptions}\n\n"
        "request_clarification(question): ask before a write that would go to a "
        "wider audience when the user intent is ambiguous.\n\n"
        f"User request: {user_request}\n\n"
        "Return the manager, tool, destination argument, whether clarification is "
        "needed, and a short rationale."
    )


def assert_personal_or_clarification(decision: DestinationRoutingDecision) -> None:
    """Assert that an ambiguous write did not widen to a shared destination."""

    assert (
        decision.clarification_requested
        or decision.destination in PERSONAL_DESTINATIONS
    )


def tool_name(decision: DestinationRoutingDecision) -> str:
    """Return the unqualified tool name chosen by the model."""

    return decision.tool.rsplit(".", 1)[-1]


def delete_context_tree(root: str) -> None:
    """Delete a context and all currently listed descendants."""

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


def rows_containing(context: str, sentinel: str) -> list[dict[str, Any]]:
    """Return rows containing a sentinel, treating missing contexts as empty."""

    try:
        logs = unify.get_logs(context=context, limit=100)
    except RequestError as exc:
        if "context" in str(exc).lower() and "not found" in str(exc).lower():
            return []
        raise
    return [row.entries for row in logs if sentinel in json.dumps(row.entries)]


def decode_tool_arguments(raw: Any) -> dict[str, Any]:
    """Decode tool-call arguments across provider response shapes."""

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def iter_tool_calls(
    messages: Iterable[dict[str, Any]],
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield tool-call names and decoded arguments from an LLM transcript."""

    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or call.get("name")
            args = decode_tool_arguments(
                function.get("arguments", call.get("arguments")),
            )
            if name:
                yield name, args

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") not in {"tool_use", "tool_call"}:
                    continue
                name = item.get("name") or item.get("tool_name")
                args = decode_tool_arguments(
                    item.get("input", item.get("arguments")),
                )
                if name:
                    yield name, args


def matching_tool_calls(
    messages: Iterable[dict[str, Any]],
    tool_suffix: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Return calls whose tool name ends with the expected suffix."""

    return [
        (name, args)
        for name, args in iter_tool_calls(messages)
        if name.endswith(tool_suffix)
    ]


def assert_tool_destination(
    messages: Iterable[dict[str, Any]],
    tool_suffix: str,
    expected_destination: str,
) -> None:
    """Assert a matching tool call used the expected shared-space destination."""

    matching_calls = matching_tool_calls(messages, tool_suffix)
    assert matching_calls, f"Expected a tool call ending with {tool_suffix!r}"
    assert any(
        args.get("destination") == expected_destination for _, args in matching_calls
    ), matching_calls


def assert_personal_tool_destination(
    messages: Iterable[dict[str, Any]],
    tool_suffix: str,
) -> None:
    """Assert every matching tool call stayed in personal memory."""

    matching_calls = matching_tool_calls(messages, tool_suffix)
    assert matching_calls, f"Expected a tool call ending with {tool_suffix!r}"
    assert all(
        args.get("destination") in {None, "personal"} for _, args in matching_calls
    )


async def run_direct_routing_loop(
    *,
    llm_config: dict[str, str],
    tools: dict[str, Any],
    accessible_spaces: list[SpaceSummary],
    message: str,
    loop_id: str,
) -> list[dict[str, Any]]:
    """Run a real LLM write loop with shared-space routing context."""

    client = new_llm_client(**llm_config)
    client.set_system_message(
        textwrap.dedent(
            f"""
            You write durable assistant memory by calling the available tool.
            Read the tool schema carefully, especially the destination parameter.
            Choose the destination from the user's meaning and the space descriptions,
            not from literal keywords alone. Use personal memory for private user
            preferences or ownership that does not clearly belong to a shared space.
            Do not answer without performing the requested write.

            {build_accessible_spaces_block(accessible_spaces)}
            """,
        ).strip(),
    )
    handle = start_async_tool_loop(
        client,
        message=message,
        tools=tools,
        loop_id=loop_id,
        max_parallel_tool_calls=1,
    )
    try:
        await asyncio.wait_for(handle.result(), timeout=180)
        return handle.get_history()
    finally:
        if not handle.done():
            await handle.stop("test cleanup")
