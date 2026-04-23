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
    previous_base = ContextRegistry._base_context
    ContextRegistry._base_context = "u7/7"
    ContextRegistry._registry.pop((TaskScheduler.__name__, "Tasks"), None)
    yield
    SESSION_DETAILS.reset()
    ContextRegistry._base_context = previous_base
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


def test_resolver_returns_per_body_tasks_path_for_solo_body():
    """Solo bodies keep the per-body ``{user}/{assistant}/Tasks`` path."""
    with patch.object(
        ContextRegistry,
        "_create_context_wrapper",
        side_effect=lambda name, entry: entry["resolved_name"],
    ) as provision:
        resolved = _call_resolver()

    assert resolved == "u7/7/Tasks"
    provision.assert_called_once()
    entry = provision.call_args.args[1]
    assert entry["resolved_name"] == "u7/7/Tasks"


def test_resolver_returns_hive_scoped_path_for_hive_member():
    """Hive members route task writes to ``Hives/{id}/Tasks``.

    The resolver delegates to :meth:`ContextRegistry.get_context`, which
    uses :meth:`base_for` to pick the Hive root and provisions the
    context through the same path every other manager uses. The
    provisioning hook fires with the Tasks ``table_context`` and the
    fully-qualified Hive path.
    """
    SESSION_DETAILS.hive_id = 42

    with patch.object(
        ContextRegistry,
        "_create_context_wrapper",
        side_effect=lambda name, entry: entry["resolved_name"],
    ) as provision:
        resolved = _call_resolver()

    assert resolved == "Hives/42/Tasks"
    provision.assert_called_once()
    entry = provision.call_args.args[1]
    assert entry["table_context"].name == "Tasks"
    assert entry["resolved_name"] == "Hives/42/Tasks"


def test_resolver_reuses_cached_registry_entry_for_hive_member():
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
