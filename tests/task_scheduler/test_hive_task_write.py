"""TaskScheduler honours Hive membership when persisting tasks.

A body born into a Hive shares a single ``Hives/{id}/Tasks`` context so every
member reads the same task definitions. A solo body keeps the per-body path
``{user_id}/{assistant_id}/Tasks`` resolved by :class:`ContextRegistry`. Either
way, every Tasks row must carry ``_assistant_id`` so Orchestra's task-machine
projector can route activations to the specific body that owns execution.

These tests exercise the resolver and private-field injection without booting
a full TaskScheduler, which requires a live Unify backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.common.context_registry import ContextRegistry
from unity.common.log_utils import _inject_private_fields
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.task_scheduler import TaskScheduler


@pytest.fixture(autouse=True)
def _reset_session_details():
    """Each test gets a clean SESSION_DETAILS + ContextRegistry cache."""
    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(agent_id=7, user_id="u7")
    ContextRegistry._registry.pop((TaskScheduler.__name__, "Tasks"), None)
    yield
    SESSION_DETAILS.reset()
    ContextRegistry._registry.pop((TaskScheduler.__name__, "Tasks"), None)


def _call_resolver() -> str:
    """Invoke ``TaskScheduler._resolve_tasks_base`` without running ``__init__``.

    ``TaskScheduler.__init__`` provisions live Unify contexts, spins up manager
    registries, and loads tool definitions. None of that is needed to verify
    the resolver's contract; bypassing ``__init__`` keeps the test fast and
    side-effect-free.
    """
    scheduler = TaskScheduler.__new__(TaskScheduler)
    return TaskScheduler._resolve_tasks_base(scheduler)


def test_resolver_delegates_to_context_registry_for_solo_body():
    """Solo bodies keep the per-body path resolved by ContextRegistry."""
    with patch.object(
        ContextRegistry,
        "get_context",
        return_value="u7/7/Tasks",
    ) as get_context:
        resolved = _call_resolver()

    assert resolved == "u7/7/Tasks"
    get_context.assert_called_once()
    args, _ = get_context.call_args
    assert args[1] == "Tasks"


def test_resolver_returns_hive_scoped_path_for_hive_member():
    """Hive members route task writes to ``Hives/{id}/Tasks``."""
    SESSION_DETAILS.hive_id = 42

    with patch.object(ContextRegistry, "_create_context_wrapper") as provision:
        resolved = _call_resolver()

    assert resolved == "Hives/42/Tasks"
    provision.assert_called_once()


def test_resolver_caches_hive_provisioning_across_repeated_calls():
    """The Hive-scoped Tasks context is provisioned at most once per body."""
    SESSION_DETAILS.hive_id = 42
    ContextRegistry._registry[(TaskScheduler.__name__, "Tasks")] = "Hives/42/Tasks"

    with patch.object(ContextRegistry, "_create_context_wrapper") as provision:
        resolved = _call_resolver()

    assert resolved == "Hives/42/Tasks"
    provision.assert_not_called()


@pytest.mark.parametrize("hive_id", [None, 42])
def test_private_field_injection_stamps_assistant_id(hive_id):
    """Every write through ``unity_log`` carries ``_assistant_id`` for the body.

    The Hive path shares storage across bodies, so ``_assistant_id`` is the only
    signal Orchestra's task-machine projector can use to route each activation
    back to the body that owns execution. The solo path sets it redundantly,
    which is harmless.
    """
    SESSION_DETAILS.hive_id = hive_id

    injected = _inject_private_fields({"name": "demo"})

    assert injected["_assistant_id"] == "7"
    assert injected["_user_id"] == "u7"
