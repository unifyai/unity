"""CodeActActor integration checks for ``primitives.coordinator``."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from tests.actor.state_managers.utils import (
    instrument_basic_primitives_calls,
    wait_for_recorded_primitives_call,
)
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.common.context_registry import ContextRegistry
from unity.events import manager_event_logging
from unity.function_manager.primitives import PrimitiveScope, Primitives
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS, AssistantDetails

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]

EVAL_TIMEOUT_SECONDS = 300.0


class _NoopEventBus:
    async def publish(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


@pytest.fixture(autouse=True)
def _reset_runtime_context() -> None:
    previous_impl = os.environ.get("UNITY_FUNCTION_IMPL")
    previous_base_context = getattr(ContextRegistry, "_base_context", None)
    previous_event_bus = manager_event_logging.EVENT_BUS
    os.environ["UNITY_FUNCTION_IMPL"] = "simulated"
    SESSION_DETAILS.reset()
    ContextRegistry.clear()
    ContextRegistry.set_base_context("UnityTests/CoordinatorCodeAct")
    ManagerRegistry.clear()
    manager_event_logging.EVENT_BUS = _NoopEventBus()
    yield
    if previous_impl is None:
        os.environ.pop("UNITY_FUNCTION_IMPL", None)
    else:
        os.environ["UNITY_FUNCTION_IMPL"] = previous_impl
    SESSION_DETAILS.reset()
    ContextRegistry.clear()
    if previous_base_context:
        ContextRegistry.set_base_context(previous_base_context)
    manager_event_logging.EVENT_BUS = previous_event_bus
    ManagerRegistry.clear()


def _configure_session(*, is_coordinator: bool) -> None:
    SESSION_DETAILS.assistant = AssistantDetails(
        agent_id=7001,
        first_name="Avery",
        surname="Coordinator",
        is_coordinator=is_coordinator,
    )
    SESSION_DETAILS.org_id = 90210
    SESSION_DETAILS.unify_key = "test-api-key"


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_code_act_routes_to_coordinator_list_assistants(monkeypatch) -> None:
    """Coordinator sessions can invoke coordinator primitives through CodeAct."""

    _configure_session(is_coordinator=True)
    expected_assistants = [
        {"agent_id": 7101, "first_name": "Revenue", "surname": "Ops"},
    ]

    def _fake_list_assistants(
        *,
        phone: str | None = None,
        email: str | None = None,
        agent_id: int | None = None,
        list_all_org: bool = False,
        api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        del phone, email, agent_id
        assert list_all_org is True
        assert api_key == "test-api-key"
        return expected_assistants

    monkeypatch.setattr(
        "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
        _fake_list_assistants,
    )
    primitives = Primitives(primitive_scope=PrimitiveScope.single("coordinator"))
    calls = instrument_basic_primitives_calls(primitives)
    actor = CodeActActor(
        environments=[StateManagerEnvironment(primitives)],
        function_manager=None,
    )
    try:
        handle = await actor.act(
            "Use execute_code to run primitives.coordinator.list_assistants() and report the first colleague name.",
            clarification_enabled=False,
            can_store=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=EVAL_TIMEOUT_SECONDS)
        assert result is not None
        await wait_for_recorded_primitives_call(
            calls,
            "primitives.coordinator.list_assistants",
        )
    finally:
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(EVAL_TIMEOUT_SECONDS)
async def test_code_act_enforces_coordinator_permission_gate(monkeypatch) -> None:
    """Non-coordinator sessions receive permission-denied from coordinator primitives."""

    _configure_session(is_coordinator=False)
    upstream_called = False

    def _should_not_call_upstream(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal upstream_called
        upstream_called = True
        raise AssertionError(f"Unexpected upstream call: {kwargs}")

    monkeypatch.setattr(
        "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
        _should_not_call_upstream,
    )
    primitives = Primitives(primitive_scope=PrimitiveScope.single("coordinator"))
    calls = instrument_basic_primitives_calls(primitives)
    actor = CodeActActor(
        environments=[StateManagerEnvironment(primitives)],
        function_manager=None,
    )
    try:
        handle = await actor.act(
            "Use execute_code to run primitives.coordinator.list_assistants() and report what happens.",
            clarification_enabled=False,
            can_store=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=EVAL_TIMEOUT_SECONDS)
        await wait_for_recorded_primitives_call(
            calls,
            "primitives.coordinator.list_assistants",
        )
        assert "permission_denied" in str(result).lower(), result
        assert not upstream_called
    finally:
        await actor.close()
