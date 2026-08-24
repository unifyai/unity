"""Task Scheduler: create, search, update, and execute tasks."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

import unisdk
import unillm
from pydantic import BaseModel

from ..actor.base import BaseActor
from ..common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from ..common.context_registry import (
    ContextRegistry,
    PERSONAL_DESTINATION,
    TEAM_CONTEXT_PREFIX,
    TEAM_DESTINATION_PREFIX,
    TableContext,
)
from ..common.custom_sync import (
    CUSTOM_RELEASED_FIELD,
    MANAGED_BY_DEPLOYMENT,
    CustomSyncAdapter,
    managed_rows_filter,
    released_rows_filter,
    require_consumed,
    run_custom_sync,
    stored_hash_field,
)
from ..common.embed_utils import ensure_vector_column, list_private_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.sync_lease import exclusive_sync_lease
from ..common.log_utils import assigned_row_id, create_logs as unity_create_logs
from ..common.llm_client import new_llm_client
from ..common.llm_helpers import methods_to_tool_dict
from ..common.metrics_utils import reduce_logs
from ..common.model_to_fields import model_to_fields
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..common.search_utils import table_search_top_k
from ..common.sentinels import _UnsetSentinel
from ..common.task_execution_context import (
    current_task_execution_ancestors,
    current_task_execution_delegate,
)
from ..common.tool_outcome import ToolOutcome, ToolErrorException
from ..common.tool_spec import ToolSpec, read_only
from ..events.manager_event_logging import log_manager_call
from ..manager_registry import ManagerRegistry
from ..session_details import SESSION_DETAILS
from ..settings import SETTINGS
from .active_task import ActiveTask
from .base import BaseTaskScheduler
from .custom_tasks import (
    compute_custom_tasks_hash,
)
from .machine_state import (
    TaskRunProvenance,
    TaskRunReference,
    build_task_run_key,
    consume_live_task_run_provenance,
    find_running_execution_for_task,
    find_terminal_execution_for_task,
    list_task_run_history,
    peek_live_task_run_provenance,
    remember_live_task_run_provenance,
    update_task_run_record,
)
from .prompt_builders import (
    build_ask_prompt,
    build_provider_event_run_guidelines,
    build_provider_event_task_request,
    build_task_execution_request,
    build_task_run_guidelines,
    build_update_prompt,
)
from .provider_trigger_actor import (
    annotate_provider_trigger_catalog,
    annotate_provider_trigger_connections,
    describe_provider_trigger,
    list_eligible_provider_trigger_connections,
    task_revision_conflict_outcome,
)
from .provider_trigger_resources import list_provider_trigger_resources
from .provider_trigger_health import (
    compose_provider_trigger_state,
    sanitize_event_context_for_actor,
)
from .resource_requirements import resolve_task_resource_requirements
from .storage import TasksStore
from . import typed_tasks_client
from .typed_tasks_client import TaskRevisionConflictError
from .types.activated_by import ActivatedBy
from .types.meta import TaskMeta
from .types.priority import Priority
from .types.repetition import (
    Frequency,
    RepeatPattern,
    Weekday,
    normalize_repeat_patterns,
)
from .types.schedule import Schedule
from .types.task import Task, TaskBase
from .types.task_row_field import split_provider_event_task_update
from .types.run_source import RunSource
from .types.execution import Delivery, ExecutionState, Wake
from .types.trigger import ProviderEventTrigger, TaskTrigger, parse_task_trigger

ScheduleLike = Optional[Union[Schedule, Dict[str, Any]]]
TriggerLike = Optional[Union[TaskTrigger, Dict[str, Any]]]
RepeatLike = Optional[List[Union[RepeatPattern, Dict[str, Any]]]]
ToolsDict = Dict[str, Callable[..., Any]]

TASKS_META_TABLE = "Tasks/Meta"
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


_UNSET = _UnsetSentinel()


class StaleActivationSuperseded(Exception):
    """A scheduled activation no longer matches the task definition's slot.

    Raised when the definition's ``schedule.start_at`` moved after the
    activation was projected (re-arm on a concurrent run start, manual
    re-arm, reconcile jitter). The activation is superseded — the run is a
    benign no-op and must never terminalize the definition row.
    """


# Columns definitions carried before run state moved to Tasks/Executions, and
# the self-attested offline certification keys that verification replaced.
# Dropped on read so rows written before those changes still load.
_LEGACY_DEFINITION_FIELDS = frozenset(
    {
        "status",
        "activated_by",
        "instance_id",
        "certification_status",
        "certification_metadata",
        "certification_result",
    },
)


class TaskSelfInvocationError(RuntimeError):
    """A task attempted to execute itself while already active in its own run.

    Raised when ``TaskScheduler.execute(task_id=N)`` is called (directly, or
    via the ``primitives.tasks.execute`` tool) while ``N`` is already an
    ancestor of the current execution -- i.e. the same task invoking itself,
    rather than a genuinely separate concurrent instance or a distinct child
    task. Do the work directly instead of re-invoking this task, or target a
    different ``task_id`` if a distinct child task is intended.
    """


class TaskScheduler(BaseTaskScheduler):
    """Concrete scheduler backed by the Tasks context."""

    class Config:
        required_contexts = [
            TableContext(
                name="Tasks",
                description=(
                    "List of all tasks with their name, description, "
                    "schedule, deadline, repeat pattern, and priority."
                ),
                fields=model_to_fields(Task),
                unique_keys={"task_id": "int"},
                auto_counting={
                    "task_id": None,
                },
                foreign_keys=[
                    {
                        "name": "entrypoint",
                        "references": "Functions/Compositional.function_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name=TASKS_META_TABLE,
                description="Metadata for source-defined custom task sync state.",
                fields=model_to_fields(TaskMeta),
                unique_keys={"meta_id": "int"},
            ),
        ]

    def __init__(
        self,
        *,
        actor: Optional[BaseActor] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """Create a scheduler for durable tasks in the current context."""

        super().__init__()

        # Get ContactManager via registry so its bound methods can act as tools
        self._contact_manager = ManagerRegistry.get_contact_manager()

        ask_tools = {
            **methods_to_tool_dict(
                ToolSpec(fn=self._filter_tasks, display_label="Filtering tasks"),
                ToolSpec(fn=self._search_tasks, display_label="Searching tasks"),
                ToolSpec(fn=self._reduce, display_label="Summarising tasks"),
                ToolSpec(
                    fn=self._list_task_runs,
                    display_label="Listing task runs",
                ),
                ToolSpec(
                    fn=self._list_provider_trigger_catalog,
                    display_label="Listing provider trigger catalog",
                ),
                ToolSpec(
                    fn=self._list_provider_trigger_connections,
                    display_label="Listing provider trigger connections",
                ),
                ToolSpec(
                    fn=self._describe_provider_trigger,
                    display_label="Describing provider trigger config",
                ),
                ToolSpec(
                    fn=self._list_provider_trigger_resources,
                    display_label="Listing provider trigger resources",
                ),
                ToolSpec(
                    fn=self._get_provider_trigger_health,
                    display_label="Inspecting provider trigger health",
                ),
                ToolSpec(
                    fn=self._get_provider_event_context,
                    display_label="Inspecting provider event context",
                ),
                ToolSpec(
                    fn=self.get_run_event_children,
                    display_label="Listing task-run event children",
                ),
                ToolSpec(
                    fn=self.get_run_event,
                    display_label="Loading one task-run event node",
                ),
                include_class_name=False,
            ),
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contact details",
                ),
                include_class_name=True,
            ),
        }
        self._ask_tools = dict(ask_tools)
        self.add_tools("ask", ask_tools)

        update_tools = {
            **methods_to_tool_dict(
                ToolSpec(fn=self.ask, display_label="Querying tasks"),
                ToolSpec(fn=self._filter_tasks, display_label="Filtering tasks"),
                ToolSpec(fn=self._search_tasks, display_label="Searching tasks"),
                ToolSpec(
                    fn=self._create_tasks,
                    display_label="Creating multiple tasks",
                ),
                ToolSpec(fn=self._create_task, display_label="Creating a new task"),
                ToolSpec(fn=self._delete_task, display_label="Deleting a task"),
                ToolSpec(fn=self._cancel_tasks, display_label="Cancelling tasks"),
                ToolSpec(fn=self._update_task, display_label="Updating a task"),
                ToolSpec(
                    fn=self._pause_provider_trigger,
                    display_label="Pausing provider trigger automation",
                ),
                ToolSpec(
                    fn=self._resume_provider_trigger,
                    display_label="Resuming provider trigger automation",
                ),
                ToolSpec(
                    fn=self._retry_provider_trigger,
                    display_label="Retrying provider trigger provisioning",
                ),
                # Read-only catalog inspection: the update prompt's authoring
                # order (list catalog -> list connections -> describe schema ->
                # resolve resources) must be executable from this loop, not
                # only from ``ask`` -- otherwise trigger feasibility gets
                # decided blind and reported as "no supported trigger".
                ToolSpec(
                    fn=self._list_provider_trigger_catalog,
                    display_label="Listing provider trigger catalog",
                ),
                ToolSpec(
                    fn=self._list_provider_trigger_connections,
                    display_label="Listing provider trigger connections",
                ),
                ToolSpec(
                    fn=self._describe_provider_trigger,
                    display_label="Describing provider trigger config",
                ),
                ToolSpec(
                    fn=self._list_provider_trigger_resources,
                    display_label="Listing provider trigger resources",
                ),
                ToolSpec(
                    fn=self._export_provider_event_context,
                    display_label="Exporting provider event context",
                ),
                ToolSpec(
                    fn=self._delete_provider_event_context,
                    display_label="Deleting provider event context",
                ),
                include_class_name=False,
            ),
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contact details",
                ),
                include_class_name=True,
            ),
        }
        self._update_tools = dict(update_tools)
        self.add_tools("update", update_tools)

        self.__actor = actor
        self._ctx = ContextRegistry.get_context(self, "Tasks")
        self._personal_tasks_context = self._ctx
        self._meta_ctx = ContextRegistry.get_context(self, TASKS_META_TABLE)
        self._root_stores: Dict[str, TasksStore] = {}
        self._active_task_root_context: Optional[str] = None
        self._custom_tasks_synced_sources: set[tuple[str, str]] = set()
        self._destination_context_lock = threading.RLock()
        self._provision_storage()

        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._num_tasks_cached: Optional[int] = None

    def _actor_for_task_run(self) -> BaseActor | None:
        """Return the fallback actor only when task execution is not delegated."""

        if current_task_execution_delegate.get() is not None:
            return None
        return self.__actor

    def _build_task_entrypoint_review(
        self,
        *,
        task: Task,
        reason: ActivatedBy,
    ) -> dict[str, Any] | None:
        """Return post-run entrypoint review context for description-driven tasks."""

        if task.entrypoint is not None:
            return None
        if task.repeat is None and task.trigger is None:
            return None

        metadata: dict[str, Any] = {
            "task_id": task.task_id,
            "task_name": task.name,
            "task_description": task.description,
            "activation_reason": reason.value,
            "response_policy": task.response_policy,
            "schedule": (
                task.schedule.model_dump(mode="json")
                if task.schedule is not None
                else None
            ),
            "trigger": (
                task.trigger.model_dump(mode="json")
                if task.trigger is not None
                else None
            ),
            "repeat": (
                [pattern.model_dump(mode="json") for pattern in task.repeat]
                if task.repeat is not None
                else None
            ),
        }

        def _attach_entrypoint(
            *,
            function_id: int,
            rationale: str,
        ) -> dict[str, Any]:
            return self._attach_entrypoint_to_definition(
                task_id=task.task_id,
                function_id=function_id,
                rationale=rationale,
            )

        def _promote_offline() -> dict[str, Any]:
            return self.promote_task_offline(task_id=task.task_id)

        return {
            "metadata": metadata,
            "attach_entrypoint": _attach_entrypoint,
            "promote_task_offline": _promote_offline,
        }

    def _build_task_run_context(
        self,
        *,
        task: Task,
        wake: Wake | RunSource | str,
        task_run_provenance: TaskRunProvenance | None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Return deterministic execution facts supplied by the scheduler."""

        scheduled_for = None
        revision = None
        run_key = None
        delivery = task.delivery_mode.value
        if task_run_provenance is not None:
            scheduled_for = task_run_provenance.scheduled_for
            revision = task_run_provenance.revision
            run_key = build_task_run_key(task_run_provenance)
            delivery = task_run_provenance.delivery.value
        if scheduled_for is None and task.schedule_start_at is not None:
            scheduled_for = task.schedule_start_at.isoformat()
        normalized_wake = Wake.normalize(
            wake.value if isinstance(wake, RunSource) else wake,
        )
        return {
            "task_id": task.task_id,
            "wake": normalized_wake.value,
            "delivery": delivery,
            "revision": revision,
            "scheduled_for": scheduled_for,
            "state": state,
            "run_key": run_key,
        }

    def _build_entrypoint_kwargs(
        self,
        *,
        task: Task,
        wake: Wake | RunSource | str,
        task_run_provenance: TaskRunProvenance | None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Return explicit kwargs available to symbolic task entrypoints."""

        context = self._build_task_run_context(
            task=task,
            wake=wake,
            task_run_provenance=task_run_provenance,
            state=state,
        )
        return {
            "task_id": task.task_id,
            **context,
            "task_execution_context": context,
        }

    def detach_entrypoint_from_definition(
        self,
        *,
        task_id: int,
        reason: str,
    ) -> dict[str, Any]:
        """Drop a task's symbolic entrypoint so future runs plan from scratch.

        An entrypoint is an optimisation: the task ran through the full actor
        loop before one existed, and dropping it costs speed and determinism
        but never correctness. That asymmetry is what makes detaching the
        right answer to an entrypoint that cannot be made to work -- one that
        dangles, that was derived against a description since rewritten, or
        that keeps being refused before it can run.

        Without this, such a task delivers nothing on every occurrence
        forever: the run holds, the definition still names the entrypoint,
        and the next wake repeats it. Detaching converts a permanent outage
        into a slower success, and the next successful run may distil a
        fresh entrypoint of its own.
        """

        task = self._get_task_or_raise(task_id)
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {task_id}",
                limit=1,
                return_ids_only=False,
            )
            if not log_objs:
                return {"outcome": "definition_missing", "task_id": task_id}
            previous = log_objs[0].entries.get("entrypoint")
            if previous is None:
                return {"outcome": "no_entrypoint", "task_id": task_id}
            self._write_log_entries(
                logs=log_objs[0].id,
                entries={"entrypoint": None},
            )
            logger.warning(
                "Detached entrypoint %s from task %s: %s",
                previous,
                task_id,
                reason,
            )
            return {
                "outcome": "entrypoint_detached",
                "task_id": task_id,
                "function_id": int(previous),
                "reason": reason,
            }

    def _attach_entrypoint_to_definition(
        self,
        *,
        task_id: int,
        function_id: int,
        rationale: str,
    ) -> dict[str, Any]:
        """Record a symbolic executor on the task definition row."""

        if function_id < 0:
            raise ValueError("function_id must be a non-negative integer.")

        task = self._get_task_or_raise(task_id)
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {task_id}",
                limit=1,
                return_ids_only=False,
            )
            if not log_objs:
                return {
                    "outcome": "definition_missing",
                    "task_id": task_id,
                    "function_id": function_id,
                    "rationale": rationale,
                }
            if log_objs[0].entries.get("entrypoint") is not None:
                return {
                    "outcome": "already_has_entrypoint",
                    "task_id": task_id,
                    "function_id": function_id,
                    "rationale": rationale,
                }
            self._write_log_entries(
                logs=log_objs[0].id,
                entries={"entrypoint": int(function_id)},
            )
            return {
                "outcome": "entrypoint_recorded",
                "task_id": task_id,
                "function_id": int(function_id),
                "rationale": rationale,
                "next": (
                    "Future runs execute this function under verification; it "
                    "earns trust from independent verdicts, and the task is "
                    "promoted to offline delivery once every function it calls "
                    "is trusted."
                ),
            }

    def _function_manager_for_trust(self) -> Any:
        from unify.function_manager.function_manager import FunctionManager

        fm = ManagerRegistry.get_function_manager()
        return fm if isinstance(fm, FunctionManager) else None

    def offline_eligible(self, task: Task) -> tuple[bool, list[str]]:
        """Whether ``task`` may run offline: a bound entrypoint whose whole closure is trusted.

        Returns ``(eligible, reasons)``; each reason names an offending
        function_id — untrusted, or ``unsafe_effectful`` with a class that
        was only inferred from third-party imports and never confirmed.
        """
        if task.entrypoint is None:
            return False, ["no_entrypoint"]
        fm = self._function_manager_for_trust()
        if fm is None:
            return False, ["no_function_manager"]
        from unify.actor.verification_runtime import closure_rows, rederive_trust

        rows = fm.filter_functions(
            filter=f"function_id == {int(task.entrypoint)}",
            destination=task.destination,
            include_implementations=True,
        )
        if not rows:
            return False, [f"entrypoint_missing:{int(task.entrypoint)}"]
        closure = closure_rows(fm, rows[0])
        rederive_trust(fm, closure, settings=fm.verification_settings)
        reasons: list[str] = []
        for row in closure.values():
            fid = int(row["function_id"])
            if row.get("verify", True):
                reasons.append(f"untrusted:{fid}")
            if (
                row.get("side_effect_class") == "unsafe_effectful"
                and row.get("class_source") == "inferred_third_party"
            ):
                reasons.append(f"unconfirmed_unsafe_class:{fid}")
        return (not reasons), reasons

    def promote_task_offline(self, *, task_id: int) -> dict[str, Any]:
        """Promote a task to offline delivery when its entrypoint closure is trusted.

        Eligibility is derived from the verification ledger, never attested.
        Loss of trust later never demotes: offline is a delivery choice and
        the headless lane runs the same verification.
        """
        task = self._get_task_or_raise(task_id)
        eligible, reasons = self.offline_eligible(task)
        if not eligible:
            return {
                "outcome": "not_eligible",
                "task_id": task_id,
                "function_id": task.entrypoint,
                "reasons": reasons,
            }
        return self._promote_definition_to_offline(task_id=task_id)

    def _promote_definition_to_offline(self, *, task_id: int) -> dict[str, Any]:
        """Write ``offline=True`` on the definition; callers check eligibility first."""

        task = self._get_task_or_raise(task_id)
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {task_id}",
                limit=1,
                return_ids_only=False,
            )
            if not log_objs:
                return {"outcome": "definition_missing", "task_id": task_id}
            row = log_objs[0]
            if bool(row.entries.get("offline")):
                return {
                    "outcome": "already_offline",
                    "task_id": task_id,
                    "function_id": row.entries.get("entrypoint"),
                }
            self._write_log_entries(
                logs=row.id,
                entries={"offline": True},
            )
            return {
                "outcome": "offline_promoted",
                "task_id": task_id,
                "function_id": row.entries.get("entrypoint"),
            }

    def warm_embeddings(self) -> None:
        """Ensure vector columns used by semantic search exist."""

        for col in ("name", "description"):
            try:
                ensure_vector_column(
                    self._ctx,
                    embed_column=f"_{col}_emb",
                    source_column=col,
                )
            except Exception:
                pass

    def _provision_storage(self) -> None:
        """Install the storage adapter for the current Tasks context."""

        self._store = TasksStore(self._ctx)
        self._root_stores[self._ctx] = self._store

    def _task_context_from_root(self, root_context: str) -> str:
        """Return the concrete Tasks context under one registry root."""

        return f"{root_context.strip('/')}/Tasks"

    def _destination_from_task_context(self, context_name: str) -> str | None:
        """Return the public destination represented by a concrete Tasks context."""

        if context_name.startswith(TEAM_CONTEXT_PREFIX):
            raw_team_id = context_name[len(TEAM_CONTEXT_PREFIX) :].split("/", 1)[0]
            return f"{TEAM_DESTINATION_PREFIX}{raw_team_id}"
        return None

    def _store_for_task_context(self, context_name: str) -> TasksStore:
        """Return a per-root store for a concrete Tasks context."""

        if context_name in self._root_stores:
            return self._root_stores[context_name]
        store = TasksStore(context_name)
        self._root_stores[context_name] = store
        return store

    def _task_context_for_destination(self, destination: str | None) -> str:
        """Resolve a write destination into a concrete Tasks context.

        For team-owned assistants, ``personal`` / ``None`` is the owning-team
        home (``Teams/{owner}/Tasks``), matching ContextRegistry shared-table
        routing. Never short-circuit through a cached personal path — that
        silently re-seeded ``{user}/{agent}/Tasks`` when session identity was
        wrong at init time.
        """

        destination = destination or os.environ.get("TASK_DESTINATION") or None
        root_context = ContextRegistry.write_root(
            self,
            "Tasks",
            destination=destination,
        )
        return self._task_context_from_root(root_context)

    def _read_task_contexts(self) -> list[str]:
        """Return ordered concrete Tasks contexts visible to this assistant."""

        if self._active_task_root_context is not None:
            return [self._active_task_root_context]
        root_contexts = ContextRegistry.read_roots(self, "Tasks")
        contexts = [self._task_context_from_root(root) for root in root_contexts]
        return list(dict.fromkeys(contexts))

    @contextmanager
    def _use_task_destination(self, destination: str | None):
        """Temporarily scope scheduler storage to one task destination."""

        context_name = self._task_context_for_destination(destination)
        previous_context = self._ctx
        previous_store = self._store
        previous_active_root = self._active_task_root_context
        self._ctx = context_name
        self._store = self._store_for_task_context(context_name)
        self._active_task_root_context = context_name
        try:
            yield context_name
        finally:
            self._ctx = previous_context
            self._store = previous_store
            self._active_task_root_context = previous_active_root

    @functools.wraps(BaseTaskScheduler.clear, updated=())
    def clear(self) -> None:
        """Delete the current Tasks context and recreate local state."""

        unisdk.delete_context(self._ctx)
        self._num_tasks_cached = None
        self._active_task_root_context = None

        ContextRegistry.forget(self, "Tasks")
        ContextRegistry.forget(self, TASKS_META_TABLE)
        self._ctx = ContextRegistry.get_context(self, "Tasks")
        self._personal_tasks_context = self._ctx
        self._meta_ctx = ContextRegistry.get_context(self, TASKS_META_TABLE)
        self._root_stores.clear()
        self._custom_tasks_synced_sources.clear()
        self._provision_storage()

    def _task_id_to_log_id_map(self, task_ids: List[int]) -> Dict[int, int]:
        """Resolve a mapping of task_id to log_id in one call."""

        try:
            log_objs = self._get_logs_by_task_ids(
                task_ids=task_ids,
                return_ids_only=False,
            )
        except Exception:
            log_objs = []

        id_map: Dict[int, int] = {}
        for lg in log_objs:
            task_id = lg.entries.get("task_id")
            if task_id is not None:
                id_map[int(task_id)] = int(lg.id)
        return id_map

    def _get_task_log(self, *, task_id: int) -> unisdk.Log:
        """Return the physical Tasks definition row for one task_id."""

        task = self._get_task_or_raise(task_id)
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {task_id}",
                limit=2,
                return_ids_only=False,
            )
        if not log_objs:
            raise ValueError(f"No task row found for task_id={task_id}.")
        if len(log_objs) != 1:
            raise ValueError(f"Ambiguous task rows for task_id={task_id}.")
        return log_objs[0]

    def _get_task_for_source_log_id(
        self,
        *,
        source_task_log_id: int,
        expected_task_id: int,
    ) -> Task:
        """Return the task instance addressed by an activation source log id."""

        for context_name in self._read_task_contexts():
            store = self._store_for_task_context(context_name)
            log_objs = store.get_rows_by_log_ids(log_ids=[source_task_log_id])
            if not log_objs:
                continue
            if len(log_objs) != 1:
                raise ValueError(
                    f"Activation source task log {source_task_log_id} is ambiguous.",
                )
            entries = dict(log_objs[0].entries or {})
            row_task_id = entries.get("task_id")
            if row_task_id != expected_task_id:
                raise ValueError(
                    "Activation source task log does not match requested task: "
                    f"expected task_id={expected_task_id}, got task_id={row_task_id}.",
                )
            entries.setdefault(
                "destination",
                self._destination_from_task_context(context_name),
            )
            entries.setdefault("assistant_id", SESSION_DETAILS.assistant_context)
            sanitized = self._sanitize_activation(entries)
            return Task(**sanitized)
        raise ValueError(
            f"Activation source task log {source_task_log_id} was not found.",
        )

    @staticmethod
    def _normalize_activation_datetime(value: Any) -> str | None:
        """Normalize scheduler timestamps into comparable UTC ISO strings."""

        if value is None:
            return None
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _validate_task_matches_provenance(
        self,
        *,
        task: Task,
        provenance: TaskRunProvenance | None,
    ) -> None:
        """Reject stale scheduled provenance before mutating a task instance."""

        if provenance is None or provenance.wake != Wake.scheduled:
            return
        if provenance.scheduled_for is None:
            return
        task_scheduled_for = self._normalize_activation_datetime(task.schedule_start_at)
        provenance_scheduled_for = self._normalize_activation_datetime(
            provenance.scheduled_for,
        )
        if task_scheduled_for is None or provenance_scheduled_for is None:
            return
        # `schedule.start_at` is the series anchor and is never rewritten, so
        # only the first occurrence can equal it — every projected successor is
        # later. An occurrence is superseded exactly when it falls *behind* the
        # anchor: the author re-armed the series past it (the same `not_before`
        # rule the backend projection applies). Requiring equality here skipped
        # every successor as "stale", each skip projected no successor of its
        # own, and a healthy ten-minute series silently ended one occurrence
        # after each re-arm.
        try:
            is_stale = datetime.fromisoformat(
                provenance_scheduled_for,
            ) < datetime.fromisoformat(task_scheduled_for)
        except ValueError:
            # An unparseable timestamp survives normalization verbatim; the
            # only safe comparison left is identity.
            is_stale = provenance_scheduled_for != task_scheduled_for
        if is_stale:
            raise StaleActivationSuperseded(
                "Scheduled activation superseded by re-armed task definition: "
                f"task_id={task.task_id}, task_start_at={task_scheduled_for}, "
                f"activation_scheduled_for={provenance_scheduled_for}.",
            )

    def _same_instance_already_active(
        self,
        *,
        task_id: int,
        provenance: TaskRunProvenance | None,
    ) -> bool:
        """Return whether this Execution attempt is already running.

        Definition-only Tasks share one row per ``task_id``, so concurrency is
        gated by Execution ``run_key`` (via create-or-adopt), not by the
        definition row's ``active`` status.
        """

        _ = task_id
        _ = provenance
        return False

    @functools.wraps(BaseTaskScheduler.ask, updated=())
    @log_manager_call(
        "TaskScheduler",
        "ask",
        payload_key="question",
        display_label="Checking tasks",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        """Answer read-only questions about existing tasks.

        Quoted task names/descriptions/reference tokens are matched
        exactly — copy them verbatim into ``text``, including punctuation
        and suffixes; verification and later lookups depend on the
        literal string.
        """

        client = new_llm_client()
        tools = dict(self.get_tools("ask"))

        _clar_queues = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_ask_prompt(
                tools,
                num_tasks=self._num_tasks(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ).to_list(),
        )

        effective_tool_policy = (
            self._default_ask_tool_policy if tool_policy == "default" else tool_policy
        )

        handle = self._start_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=_parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNIFY_READONLY_ASK_GUARD else None
            ),
            response_format=response_format,
            clarification_queues=_clar_queues,
        )

        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)
        return handle

    @functools.wraps(BaseTaskScheduler.update, updated=())
    @log_manager_call(
        "TaskScheduler",
        "update",
        payload_key="request",
        display_label="Updating tasks",
    )
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
    ) -> SteerableToolHandle:
        """Apply a mutation request expressed in plain English.

        Quoted task names/descriptions/reference tokens are matched
        exactly — copy them verbatim into ``text``, including punctuation
        and suffixes; verification and later lookups depend on the
        literal string.
        """

        client = new_llm_client()
        tools = dict(self.get_tools("update"))

        _clar_queues = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_update_prompt(
                tools,
                num_tasks=self._num_tasks(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ).to_list(),
        )

        effective_tool_policy = (
            self._default_update_tool_policy
            if tool_policy == "default"
            else tool_policy
        )

        handle = self._start_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=_parent_chat_context,
            log_steps=_log_tool_steps,
            tool_policy=effective_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
        )

        if _return_reasoning_steps:
            handle = self._wrap_result_with_messages(handle, client)
        return handle

    @functools.wraps(BaseTaskScheduler.execute, updated=())
    @log_manager_call(
        "TaskScheduler",
        "execute",
        payload_key="request",
        display_label="Working on task",
    )
    async def execute(
        self,
        task_id: int,
        *,
        trigger_attempt_token: str | None = None,
        response_format: Optional[Type[BaseModel]] = None,
        _activated_by: ActivatedBy | None = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        """Start one runnable task instance and return its live handle."""

        all_task_instances = self._filter_tasks(
            filter=f"task_id == {task_id}",
        )
        if not all_task_instances:
            raise ValueError(f"No task found with id={task_id}")

        if len(all_task_instances) != 1:
            raise ValueError(
                f"Task definition must be unique for task_id={task_id}; "
                f"found {len(all_task_instances)} rows.",
            )
        task = all_task_instances[0]
        finished = self._one_shot_already_ran(task)
        if finished is not None:
            raise ValueError(
                f"No runnable task found with id={task_id}: one-shot already ran "
                f"(run_key={finished.run_key}).",
            )

        # Reject reentrancy: a task invoking its own task_id again (directly,
        # or transitively via a nested primitives.tasks.execute call) while it
        # is already active in this same execution chain. This is narrower
        # than "any instance of task_id is active anywhere" -- see the
        # concurrent-instance note below, which remains intentionally allowed.
        if task_id in current_task_execution_ancestors.get():
            raise TaskSelfInvocationError(
                f"Task {task_id} cannot invoke itself via primitives.tasks.execute "
                f"while it is already active in its own execution chain. Perform "
                f"the work directly instead of re-invoking this task, or target a "
                f"different task_id if a distinct child task is intended.",
            )

        # Concurrent instances of the same task_id are allowed. Only block when
        # execution provenance targets a row that is already active.
        task_run_wake = (
            Wake.triggered
            if trigger_attempt_token
            else Wake.normalize(
                RunSource.from_activation_reason(
                    _activated_by or ActivatedBy.explicit,
                ).value,
            )
        )
        pending_provenance = peek_live_task_run_provenance(
            assistant_id=SESSION_DETAILS.assistant.agent_id,
            task_id=task_id,
            wake=task_run_wake,
            trigger_attempt_token=trigger_attempt_token,
        )
        if self._same_instance_already_active(
            task_id=task_id,
            provenance=pending_provenance,
        ):
            raise RuntimeError(
                f"Task {task_id} instance source_task_log_id="
                f"{pending_provenance.source_task_log_id} is already active.",
            )

        if _activated_by is not None:
            reason = _activated_by
        else:
            if task.trigger is not None:
                reason = ActivatedBy.trigger
            elif task.schedule_start_at is not None:
                reason = ActivatedBy.schedule
            else:
                reason = ActivatedBy.explicit

        fallback_actor = self._actor_for_task_run()
        if fallback_actor is None and current_task_execution_delegate.get() is None:
            raise RuntimeError(
                "TaskScheduler.execute requires a run-scoped actor delegate or an explicit actor. "
                "Description-driven tasks should be executed from Actor.act via primitives.tasks.execute(...).",
            )

        task_run_wake = (
            Wake.triggered
            if trigger_attempt_token
            else Wake.normalize(RunSource.from_activation_reason(reason).value)
        )
        task_run_provenance = consume_live_task_run_provenance(
            assistant_id=SESSION_DETAILS.assistant.agent_id,
            task_id=task_id,
            wake=task_run_wake,
            destination=task.destination,
            trigger_attempt_token=trigger_attempt_token,
            offline=task.offline,
        )
        explicit_assistant_id = str(
            SESSION_DETAILS.assistant.agent_id
            or SESSION_DETAILS.assistant_context
            or "",
        ).strip()
        if task_run_provenance is None and explicit_assistant_id:
            # An explicit run has no dispatcher to remember provenance for it.
            # Synthesize it so the run still materializes a Tasks/Executions
            # row: run state lives only there now, so a run without one is
            # invisible to every guard, listing and audit that reads it.
            # Executions are assistant-owned in Orchestra, so a session with no
            # assistant (unit tests) simply has no run ledger to write to.
            task_run_provenance = TaskRunProvenance(
                assistant_id=explicit_assistant_id,
                task_id=int(task_id),
                wake=task_run_wake,
                delivery=Delivery.offline if task.offline else Delivery.live,
                source_task_log_id=self._source_task_log_id(task_id),
                revision=(
                    str(task.task_revision) if task.task_revision is not None else None
                ),
                destination=task.destination,
                task_name=task.name,
                attempt_token=trigger_attempt_token,
            )
        if task_run_provenance and task_run_provenance.source_task_log_id is not None:
            task = self._get_task_for_source_log_id(
                source_task_log_id=task_run_provenance.source_task_log_id,
                expected_task_id=task_id,
            )
            # Definitions carry intent only, so concurrent Executions against
            # one row are expected. A run in flight never blocks the next wake;
            # only a finished one-shot is unrunnable.
            if self._one_shot_already_ran(task) is not None:
                raise ValueError(
                    "Task definition is not runnable: "
                    f"task_id={task.task_id} is a one-shot that already ran.",
                )
            if _activated_by is None:
                if task.trigger is not None:
                    reason = ActivatedBy.trigger
                elif task.schedule_start_at is not None:
                    reason = ActivatedBy.schedule
                else:
                    reason = ActivatedBy.explicit
                task_run_wake = (
                    Wake.triggered
                    if trigger_attempt_token
                    else Wake.normalize(
                        RunSource.from_activation_reason(reason).value,
                    )
                )
        self._validate_task_matches_provenance(
            task=task,
            provenance=task_run_provenance,
        )
        if not task.enabled:
            raise ValueError(
                f"Task {task_id} is disabled and cannot be executed. "
                "Re-enable it before executing.",
            )

        entrypoint_kwargs = None
        if task.entrypoint is not None:
            entrypoint_kwargs = self._build_entrypoint_kwargs(
                task=task,
                wake=task_run_wake,
                task_run_provenance=task_run_provenance,
                state=ExecutionState.running.value,
            )

        workflow_slug, installation_settings = self._workflow_run_settings(task)
        task_request = build_task_execution_request(
            task,
            installation_settings=installation_settings,
            workflow_slug=workflow_slug,
        )

        # The successor is projected by Orchestra when this run is marked
        # running: recurrence is a ledger invariant, not something each
        # dispatcher must remember to do.

        # Extend the ancestor chain for the duration of starting the child run,
        # so a spawned run that calls back into TaskScheduler.execute with this
        # same task_id (directly or via nested primitives.tasks.execute) is
        # caught by the reentrancy check above. The spawned run's own asyncio
        # task captures a copy of this context at creation time, so resetting
        # it here afterward does not affect the already-running child.
        ancestor_token = current_task_execution_ancestors.set(
            current_task_execution_ancestors.get() | {task_id},
        )
        try:
            handle = await ActiveTask.create(
                fallback_actor,
                task_description=task_request,
                _parent_chat_context=_parent_chat_context,
                _clarification_up_q=_clarification_up_q,
                _clarification_down_q=_clarification_down_q,
                task_id=task_id,
                scheduler=self,
                entrypoint=task.entrypoint,
                entrypoint_kwargs=entrypoint_kwargs,
                entrypoint_repair_context=(
                    {
                        "task_name": task.name,
                        "task_run_context": entrypoint_kwargs.get(
                            "task_execution_context",
                            {},
                        ),
                        "task_request": task_request,
                    }
                    if entrypoint_kwargs is not None
                    else None
                ),
                destination=task.destination,
                task_run_provenance=task_run_provenance,
                task_entrypoint_review=self._build_task_entrypoint_review(
                    task=task,
                    reason=reason,
                ),
                task_guidelines=build_task_run_guidelines(task, reason),
            )
        finally:
            current_task_execution_ancestors.reset(ancestor_token)

        return handle

    async def start_provider_event_instance(
        self,
        *,
        request: "ProviderEventDispatchRequest",
        captured_task_revision: int,
        provider_event_context: dict[str, Any],
    ) -> SteerableToolHandle:
        """Start one provider-event execution against the authored definition.

        Materializes/adopts the Orchestra-precreated ``Tasks/Executions`` row by
        ``run_key`` and leaves the definition row untouched (no Task-row clone).
        Validates the accepted receipt authorization on ``request`` rather than
        current trigger state. Event content must arrive as structured untrusted
        data.
        """

        from unify.task_scheduler.provider_event_dispatch import (
            ProviderEventDispatchRequest,
            ProviderEventDispatchValidationError,
        )

        if not isinstance(request, ProviderEventDispatchRequest):
            raise TypeError("request must be a ProviderEventDispatchRequest")
        if request.delivery not in {"live", "offline"}:
            raise ProviderEventDispatchValidationError("invalid_delivery")
        if str(request.wake) != Wake.provider_event.value:
            raise ProviderEventDispatchValidationError("run_wake_mismatch")

        definition = self._get_provider_event_definition(task_id=request.task_id)
        if not definition.enabled:
            raise ProviderEventDispatchValidationError("task_disabled")
        if not self._task_has_provider_event_trigger(definition):
            raise ProviderEventDispatchValidationError("task_trigger_mismatch")

        source_task_log_id = self._get_task_log(task_id=definition.task_id).id

        provenance = TaskRunProvenance(
            assistant_id=str(request.assistant_id),
            task_id=request.task_id,
            wake=Wake.provider_event,
            delivery=(
                Delivery.offline if request.delivery == "offline" else Delivery.live
            ),
            source_task_log_id=int(source_task_log_id),
            revision=request.accepted_revision,
            destination=definition.destination,
            source_ref=request.receipt_id,
            attempt_token=request.operation_id,
            task_name=definition.name,
        )
        remember_live_task_run_provenance(provenance)

        # Adopt the Orchestra-precreated run by its exact run_key. Do not let
        # ActiveTask rebuild a different key from provenance and create a second
        # run — provider-event dispatch is adopt-only.
        task_run_reference = TaskRunReference(
            assistant_id=str(request.assistant_id),
            run_key=request.run_key,
            source_task_log_id=int(source_task_log_id),
        )
        update_task_run_record(
            task_run_reference,
            {
                "state": "running",
                "source_task_log_id": int(source_task_log_id),
                "revision": request.accepted_revision,
                "captured_task_revision": captured_task_revision,
                "started_at": _now_iso(),
            },
        )

        fallback_actor = self._actor_for_task_run()
        if fallback_actor is None and current_task_execution_delegate.get() is None:
            raise RuntimeError(
                "Provider-event dispatch requires a run-scoped actor "
                "delegate or an explicit actor.",
            )

        entrypoint_kwargs = self._build_entrypoint_kwargs(
            task=definition,
            wake=Wake.provider_event,
            task_run_provenance=provenance,
            state=ExecutionState.running.value,
        )
        entrypoint_kwargs["provider_event_context"] = provider_event_context
        entrypoint_kwargs["operation_id"] = request.operation_id
        entrypoint_kwargs["receipt_id"] = request.receipt_id
        entrypoint_kwargs["binding_id"] = request.binding_id
        entrypoint_kwargs["run_id"] = request.run_id
        entrypoint_kwargs["run_key"] = request.run_key
        entrypoint_kwargs["accepted_revision"] = request.accepted_revision
        entrypoint_kwargs["captured_task_revision"] = captured_task_revision
        if isinstance(entrypoint_kwargs.get("task_execution_context"), dict):
            entrypoint_kwargs["task_execution_context"]["run_key"] = request.run_key
            entrypoint_kwargs["task_execution_context"][
                "captured_task_revision"
            ] = captured_task_revision

        if definition.entrypoint is None:
            task_request = build_provider_event_task_request(
                definition,
                provider_event_context,
            )
        else:
            slug, settings = self._workflow_run_settings(definition)
            task_request = build_task_execution_request(
                definition,
                installation_settings=settings,
                workflow_slug=slug,
            )
        return await ActiveTask.create(
            fallback_actor,
            task_description=task_request,
            task_id=definition.task_id,
            scheduler=self,
            entrypoint=definition.entrypoint,
            entrypoint_kwargs=entrypoint_kwargs,
            entrypoint_repair_context=(
                {
                    "task_name": definition.name,
                    "task_run_context": entrypoint_kwargs.get(
                        "task_execution_context",
                        {},
                    ),
                    "task_request": task_request,
                }
                if definition.entrypoint is not None
                else None
            ),
            destination=definition.destination,
            task_run_reference=task_run_reference,
            task_run_provenance=provenance,
            task_guidelines=build_provider_event_run_guidelines(definition),
            preserve_definition_status=True,
        )

    def _get_provider_event_definition(self, *, task_id: int) -> Task:
        """Return the authored definition row for one provider-event task.

        Prefer a non-captured Tasks-store definition when present. Fall back to
        the typed Tasks API for definitions authored only through that path.
        Captured execution instances are never returned here.
        """

        from unify.task_scheduler.provider_event_dispatch import (
            ProviderEventDispatchValidationError,
        )

        rows = self._filter_tasks(filter=f"task_id == {task_id}", limit=1000)
        definitions = [
            row for row in rows if self._task_has_provider_event_trigger(row)
        ]
        if definitions:
            return definitions[0]
        try:
            return self._get_provider_event_task_or_raise(task_id)
        except ValueError as exc:
            if not rows:
                raise ValueError(f"No task found with id={task_id}") from exc
            raise ProviderEventDispatchValidationError("task_trigger_mismatch") from exc

    def create_task(
        self,
        *,
        name: str,
        description: str,
        destination: str | None = None,
    ) -> ToolOutcome:
        """Create a task with just the required descriptive fields."""

        return self._create_task(
            name=name,
            description=description,
            destination=destination,
        )

    def _one_shot_already_ran(self, task: Task) -> Any | None:
        """The terminal execution of a finished one-shot, if it has run.

        Derived rather than stored: "has this already run?" is a fact about the
        run ledger, and keeping it there leaves the definition as pure authored
        intent. Repeating and triggerable definitions are never "done" — their
        next occurrence is always ahead of them.
        """

        if task.repeat is not None or task.trigger is not None:
            return None
        return find_terminal_execution_for_task(
            task_id=int(task.task_id),
            destination=task.destination,
        )

    def _mark_one_shot_completed(self, task_id: int) -> None:
        """Disarm a finished one-shot.

        No-op for repeating and triggerable definitions: they stay armed for
        their next occurrence, and the outcome of the run that just finished
        belongs on its ``Tasks/Executions`` row. This is the only path that
        disarms a definition as a result of a run, and it can never apply to a
        standing schedule.
        """

        task = self._get_task_or_raise(task_id)
        if task.repeat is not None or task.trigger is not None:
            return
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {task_id}",
                return_ids_only=False,
            )
            if not log_objs:
                raise ValueError(f"No task definition ({task_id}) found.")
            if len(log_objs) != 1:
                log_ids = [getattr(row, "id", None) for row in log_objs]
                raise ValueError(
                    "Tasks definition must be unique for "
                    f"task_id={task_id}; found {len(log_objs)} rows with log ids {log_ids}.",
                )
            self._write_log_entries(
                logs=log_objs[0].id,
                entries={"enabled": False},
            )

    def _validate_scheduled_invariants(
        self,
        *,
        schedule: ScheduleLike,
        trigger: TriggerLike = None,
        err_prefix: str = "Invalid task state:",
    ) -> None:
        """Validate the remaining scheduler invariants for task state."""

        start_at = None
        if isinstance(schedule, Schedule):
            start_at = schedule.start_at
        elif isinstance(schedule, dict):
            start_at = schedule.get("start_at")

        if schedule is not None and trigger is None and start_at is None:
            raise ValueError(
                f"{err_prefix} a scheduled task must have a start_at timestamp.",
            )

    def _source_task_log_id(self, task_id: int) -> int | None:
        """Orchestra log id of one definition row, for execution provenance."""

        task = self._get_task_or_raise(task_id)
        with self._use_task_destination(task.destination):
            log_objs = self._store.get_rows(
                filter=f"task_id == {int(task_id)}",
                limit=1,
                return_ids_only=False,
            )
        return int(log_objs[0].id) if log_objs else None

    def _running_execution(
        self,
        task_id: int,
        *,
        states: tuple[ExecutionState, ...] | None = None,
    ) -> Any | None:
        """The execution for one definition matching ``states``, if any.

        Defaults to the full open-state set (scheduled/triggerable/running).
        """

        # Typed provider-event definitions have no Tasks log mirror, so the
        # store-only lookup cannot resolve them; use the same typed fallback
        # as authored mutations.
        task = self._resolve_task_for_mutation(task_id)
        kwargs: Dict[str, Any] = {
            "task_id": int(task_id),
            "destination": task.destination,
        }
        if states is not None:
            kwargs["states"] = states
        return find_running_execution_for_task(**kwargs)

    def _cancel_open_executions(self, task_id: int) -> None:
        """Terminalize any run still in flight for one definition."""

        execution = self._running_execution(task_id)
        if execution is None:
            return
        # The row this is cancelling is the one already in hand, so reference
        # it directly. Looking it up again called
        # latest_task_run_reference_for_source without its required
        # source_task_log_id, which raised TypeError the moment anything
        # reached here.
        update_task_run_record(
            TaskRunReference(
                assistant_id=execution.assistant_id
                or str(SESSION_DETAILS.assistant_context),
                run_key=execution.run_key,
                source_task_log_id=execution.source_task_log_id,
            ),
            {
                "state": ExecutionState.cancelled.value,
                # completed_at, not ended_at: nothing reads ended_at, so
                # cancelled runs showed no finish time anywhere.
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _ensure_not_active_task(self, task_ids: Union[int, List[int]]) -> None:
        """Guard against mutating a definition with a genuinely running execution.

        Reads ``Tasks/Executions`` rather than the definition: whether a run is
        happening is a fact about the run. Scoped to ``running`` only —
        ``scheduled``/``triggerable`` rows (e.g. an armed provider-event task)
        are not mutation hazards and must not block this guard.
        """

        ids = [task_ids] if isinstance(task_ids, int) else list(task_ids)
        ids = [int(task_id) for task_id in ids]
        for task_id in ids:
            if (
                self._running_execution(task_id, states=(ExecutionState.running,))
                is not None
            ):
                raise RuntimeError(
                    f"Operation not permitted on the active task (task_id={task_id})",
                )

    @overload
    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: Literal[True] = True,
    ) -> List[int]: ...

    @overload
    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: Literal[False],
    ) -> List[unisdk.Log]: ...

    def _get_logs_by_task_ids(
        self,
        *,
        task_ids: Union[int, List[int]],
        return_ids_only: bool = True,
    ):
        """Fetch log objects or ids for one or many logical task ids."""

        task_id_list = task_ids if isinstance(task_ids, list) else [task_ids]
        matches: list[unisdk.Log] = []
        for context_name in self._read_task_contexts():
            store = self._store_for_task_context(context_name)
            rows = store.get_logs_by_task_ids(
                task_ids=task_id_list,
                return_ids_only=False,
            )
            destination = self._destination_from_task_context(context_name)
            for row in rows:
                row.entries.setdefault("destination", destination)
                row.entries.setdefault(
                    "assistant_id",
                    SESSION_DETAILS.assistant_context,
                )
                matches.append(row)

        if isinstance(task_ids, int):
            root_destinations = {
                row.entries.get("destination") or PERSONAL_DESTINATION
                for row in matches
            }
            if len(root_destinations) > 1:
                raise ValueError(
                    f"Task id {task_ids} exists in multiple task roots; provide destination.",
                )

        if return_ids_only:
            return [int(row.id) for row in matches]
        return matches

    def _create_task(
        self,
        *,
        name: str,
        description: str,
        schedule: ScheduleLike = None,
        trigger: TriggerLike = None,
        deadline: Optional[Union[str, datetime]] = None,
        max_runtime_seconds: Optional[int] = None,
        repeat: RepeatLike = None,
        priority: Priority = Priority.normal,
        response_policy: Optional[str] = None,
        entrypoint: Optional[int] = None,
        offline: bool = False,
        requires_filesystem: bool = False,
        requires_computer: bool = False,
        enabled: bool = True,
        destination: str | None = None,
        _root_applied: bool = False,
        _sync_identity: Optional[Dict[str, Any]] = None,
    ) -> ToolOutcome:
        """Create a single task with the given name and description.

        Supports optional scheduling (start time, deadline, recurrence),
        event-based triggers, execution mode (agentic vs symbolic via
        ``entrypoint``), background offline execution, an optional per-attempt
        runtime bound (``max_runtime_seconds``), and an enabled flag.
        Returns a ``ToolOutcome`` containing the newly assigned ``task_id``.

        A successful create is the arming: an enabled task with a schedule
        or recurrence fires without any further step. Run rows are
        materialized separately by the deployment's scheduler and may not
        appear until the first run starts, so an empty run ledger right
        after creation is normal and is not evidence the task is disarmed.
        Never recreate or disable a task because no run row is visible yet;
        to confirm arming, read the definition back — ``enabled`` plus its
        schedule and recurrence.

        ``_sync_identity`` carries the custom-sync row identity
        (``custom_key``/``custom_hash`` and sync-owned extras) so
        deployment-owned rows are born with their identity in the same
        write as the rest of the row.
        """

        if not _root_applied:
            effective_destination = (
                destination or os.environ.get("TASK_DESTINATION") or None
            )
            with self._use_task_destination(effective_destination):
                return self._create_task(
                    name=name,
                    description=description,
                    schedule=schedule,
                    trigger=trigger,
                    deadline=deadline,
                    max_runtime_seconds=max_runtime_seconds,
                    repeat=repeat,
                    priority=priority,
                    response_policy=response_policy,
                    entrypoint=entrypoint,
                    offline=offline,
                    requires_filesystem=requires_filesystem,
                    requires_computer=requires_computer,
                    enabled=enabled,
                    destination=effective_destination,
                    _root_applied=True,
                    _sync_identity=_sync_identity,
                )

        if not name or not description:
            raise ValueError("Both 'name' and 'description' are required")

        duplicate_rows = self._find_name_desc_collisions(
            name=name,
            description=description,
            limit=2,
        )
        if duplicate_rows:
            for row in duplicate_rows:
                if row.get("name") == name:
                    raise ValueError(
                        f"A task with {'name'!r} = {name!r} already exists",
                    )
                if row.get("description") == description:
                    raise ValueError(
                        f"A task with {'description'!r} = {description!r} already exists",
                    )

        if schedule is not None and isinstance(schedule, dict):
            schedule = Schedule(**schedule)
        if trigger is not None and isinstance(trigger, dict):
            trigger = parse_task_trigger(trigger)
        if repeat is not None:
            repeat = [
                RepeatPattern(**item) if isinstance(item, dict) else item
                for item in repeat
            ]
            repeat = normalize_repeat_patterns(repeat)

        if schedule is not None and trigger is not None:
            raise ValueError("`schedule` and `trigger` are mutually exclusive.")

        self._validate_scheduled_invariants(
            schedule=schedule,
            trigger=trigger,
            err_prefix="While creating a task:",
        )

        task_details = TaskBase(
            assistant_id=SESSION_DETAILS.assistant_context,
            destination=(
                destination if destination not in (None, PERSONAL_DESTINATION) else None
            ),
            name=name,
            description=description,
            schedule=schedule,
            trigger=trigger,
            deadline=deadline,
            max_runtime_seconds=max_runtime_seconds,
            repeat=repeat,
            priority=priority,
            response_policy=response_policy,
            entrypoint=entrypoint,
            offline=offline,
            requires_filesystem=requires_filesystem,
            requires_computer=requires_computer,
            enabled=enabled,
        ).to_post_json()

        if trigger is not None and isinstance(trigger, ProviderEventTrigger):
            created = typed_tasks_client.create_task(payload=task_details)
            task_id = int(created["task_id"])
            if _sync_identity:
                # The sealed provider-event façade owns the create payload
                # shape, so identity is stamped immediately after creation
                # rather than in-band.
                log_ids = self._store.get_rows(
                    filter=f"task_id == {task_id}",
                    return_ids_only=True,
                )
                self._write_log_entries(logs=log_ids, entries=dict(_sync_identity))
        else:
            entries = (
                {**task_details, **_sync_identity} if _sync_identity else task_details
            )
            log = self._store.log(entries=entries, new=True)
            task_id = assigned_row_id(log, "task_id", context=self._store.context)
            if self._num_tasks_cached is not None:
                self._num_tasks_cached += 1

        return {
            "outcome": "task created successfully",
            "details": {"task_id": task_id},
        }

    def _create_tasks(
        self,
        *,
        tasks: List[Dict[str, Any]],
        destination: str | None = None,
        _root_applied: bool = False,
    ) -> ToolOutcome:
        """Create multiple tasks in the given order and return their IDs.

        Accepts a list of task definitions; each entry follows the same schema
        as ``_create_task``.  Tasks are written in list order and their
        assigned ``task_id`` values are returned.  Destination routing is
        applied uniformly to all created tasks.
        """

        if not _root_applied:
            effective_destination = (
                destination or os.environ.get("TASK_DESTINATION") or None
            )
            with self._use_task_destination(effective_destination):
                return self._create_tasks(
                    tasks=tasks,
                    destination=effective_destination,
                    _root_applied=True,
                )

        if not tasks:
            return {"outcome": "tasks created", "details": {"task_ids": []}}

        seen_names: set[str] = set()
        seen_descs: set[str] = set()
        created_ids: List[int] = []
        for index, spec in enumerate(tasks):
            name = spec.get("name")
            description = spec.get("description")
            if not name or not description:
                raise ValueError(
                    f"Each task spec must include non-empty 'name' and 'description' (index={index}).",
                )
            if name in seen_names:
                raise ValueError(
                    f"Duplicate task name in batch: {name!r} (index={index})",
                )
            if description in seen_descs:
                raise ValueError(
                    "Duplicate task description in batch – descriptions must be unique: "
                    f"{description!r} (index={index})",
                )
            seen_names.add(str(name))
            seen_descs.add(str(description))

            payload: Dict[str, Any] = {}
            for key in (
                "name",
                "description",
                "schedule",
                "trigger",
                "deadline",
                "repeat",
                "priority",
                "response_policy",
                "entrypoint",
                "offline",
                "requires_filesystem",
                "requires_computer",
                "enabled",
            ):
                if key in spec:
                    payload[key] = spec[key]

            out = self._create_task(
                **payload,
                destination=destination,
                _root_applied=True,
            )
            created_ids.append(int(out["details"]["task_id"]))

        return {
            "outcome": "tasks created",
            "details": {"task_ids": created_ids},
        }

    def _delete_task(
        self,
        *,
        task_id: int,
        destination: str | None = None,
        _root_applied: bool = False,
    ) -> ToolOutcome:
        """Permanently delete all rows for the given task id.

        Removes every instance row (all recurrence clones included) stored
        under the provided ``task_id``.  Raises if the task is currently
        active.  This action is irreversible.
        """

        if not _root_applied:
            resolved_destination = (
                destination or os.environ.get("TASK_DESTINATION") or None
            )
            if resolved_destination is None:
                resolved_destination = (
                    self._resolve_task_for_mutation(task_id).destination
                    or PERSONAL_DESTINATION
                )
            with self._use_task_destination(resolved_destination):
                return self._delete_task(
                    task_id=task_id,
                    destination=resolved_destination,
                    _root_applied=True,
                )

        self._ensure_not_active_task(task_id)
        task = self._resolve_task_for_mutation(task_id)
        log_ids = self._store.get_rows(
            filter=f"task_id == {task_id}",
            return_ids_only=True,
        )
        if self._task_has_provider_event_trigger(task):
            if task.task_revision is None:
                raise ValueError(
                    f"Task {task_id} is missing task_revision; re-read before deleting.",
                )
            try:
                typed_tasks_client.delete_task(
                    task_id=task_id,
                    expected_task_revision=int(task.task_revision),
                )
            except TaskRevisionConflictError as exc:
                return task_revision_conflict_outcome(exc)
        else:
            self._store.delete(logs=log_ids)
        removed_count = len(log_ids)
        if self._num_tasks_cached is not None and removed_count:
            self._num_tasks_cached = max(
                0,
                int(self._num_tasks_cached) - int(removed_count),
            )
        return {
            "outcome": "task deleted",
            "details": {"task_id": task_id},
        }

    def _cancel_tasks(self, task_ids: List[int]) -> ToolOutcome:
        """Cancel one or more tasks by id, disarming them permanently.

        Cancelling disarms the definition and terminalizes any open Executions.
        A run already in flight is cancelled with it; cancellation is the one
        operation that reaches into live runs, because the operator's intent is
        to stop the work, not merely to stop the next wake.
        """

        requested_task_ids = list(dict.fromkeys(int(task_id) for task_id in task_ids))

        missing: list[int] = []
        for task_id in requested_task_ids:
            task = self._get_task_or_raise(task_id)
            if self._one_shot_already_ran(task) is not None:
                raise ValueError(f"Cannot cancel completed task (id={task_id}).")
            with self._use_task_destination(task.destination):
                logs = self._store.get_rows(
                    filter=f"task_id == {task_id}",
                    return_ids_only=False,
                )
                if not logs:
                    missing.append(task_id)
                    continue
                self._write_log_entries(
                    logs=[int(log.id) for log in logs],
                    entries={"enabled": False},
                )
            self._cancel_open_executions(task_id)

        if missing:
            raise ValueError(f"No matching task_ids resolved: {missing}")
        return {
            "outcome": "tasks cancelled",
            "details": {"task_ids": requested_task_ids},
        }

    def _set_tasks_enabled(
        self,
        *,
        task_ids: Union[int, List[int]],
        enabled: bool,
    ) -> Dict[str, str]:
        """Arm or disarm one or many definitions.

        Disarming stops the next wake and leaves runs in flight alone. Use
        :meth:`_cancel_tasks` when the intent is to stop current work too.
        """

        ids = [task_ids] if isinstance(task_ids, int) else list(task_ids)
        ids = [int(task_id) for task_id in ids]
        if not ids:
            return {"detail": "No updates"}

        last_result: Dict[str, str] = {"detail": "No updates"}
        for task_id in ids:
            task = self._get_task_or_raise(task_id)
            if enabled and self._one_shot_already_ran(task) is not None:
                raise ValueError(
                    f"Task {task_id} is a one-shot that already ran and cannot "
                    "be re-armed; create a new task instead of re-running it.",
                )
            with self._use_task_destination(task.destination):
                log_ids = self._store.get_rows(
                    filter=f"task_id == {task_id}",
                    return_ids_only=True,
                )
                last_result = self._write_log_entries(
                    logs=log_ids,
                    entries={"enabled": enabled},
                )
            if not enabled:
                self._withdraw_open_occurrence(task)
        return last_result

    def _withdraw_open_occurrence(self, task: Task) -> None:
        """Retire the occurrence a disarmed definition would still have fired.

        Disarming the definition is not enough on its own. Projection has
        already minted the next occurrence and it sits in the ledger as
        ``scheduled``; the dispatcher works from that row, so the wake still
        arrives, the run is refused with "task is disabled", and the refusal
        is recorded as a failed start -- which now also tells the owner their
        task failed. Pausing something is supposed to be quiet.

        Withdrawing it closes that: no wake, no pod, no failure to explain.
        Re-arming projects a fresh occurrence, so nothing is lost but the one
        the user asked not to happen.

        A run already under way is left alone, which is this method's caller's
        stated contract: disarming stops the next wake, and stopping current
        work is a cancel. "Open" spans ``running`` too, so the state has to be
        checked rather than assumed from openness.

        Best-effort: the definition is already disarmed by the time this runs,
        so the worst case is the old behaviour rather than a broken pause.
        """

        from .machine_state import get_open_task_execution, update_task_run_record

        try:
            open_run = get_open_task_execution(
                assistant_id=SESSION_DETAILS.assistant.agent_id,
                task_id=int(task.task_id),
                destination=task.destination,
            )
            if open_run is None or not open_run.run_key:
                return
            if open_run.state == ExecutionState.running.value:
                return
            update_task_run_record(
                TaskRunReference(
                    assistant_id=open_run.assistant_id or "",
                    run_key=open_run.run_key,
                    source_task_log_id=open_run.source_task_log_id,
                ),
                {
                    "state": ExecutionState.cancelled.value,
                    "completed_at": _now_iso(),
                    "result_summary": (
                        "Withdrawn: the task was paused before this occurrence "
                        "was due, so it was never dispatched."
                    ),
                },
            )
        except Exception:
            logger.exception(
                "Could not withdraw the open occurrence for task %s",
                task.task_id,
            )

    def list_custom_tasks(
        self,
        *,
        managed_by: str,
        destination: str | None = None,
    ) -> List[Dict[str, Any]]:
        """The task definitions one source planted, with their ids.

        The counterpart to :meth:`set_custom_tasks_enabled`: the installer
        that plants and arms a source's tasks also has to be able to name
        them, so a caller can start one on demand. Resolved here rather
        than by the caller, because the context a destination maps to is
        this manager's to decide.
        """
        tasks_context, _meta_context, _is_personal = self._sync_destination_contexts(
            destination,
        )
        store = self._store_for_task_context(tasks_context)
        rows = store.get_rows(filter=managed_rows_filter(managed_by), limit=1000)
        planted: List[Dict[str, Any]] = []
        for row in rows:
            entries = dict(row.entries or {})
            task_id = entries.get("task_id")
            if task_id is None:
                continue
            entrypoint = entries.get("entrypoint")
            planted.append(
                {
                    "task_id": int(task_id),
                    "name": entries.get("name", ""),
                    # False while the source holds them on a missing
                    # connection: the definition exists and nothing will
                    # start it, an explicit run included.
                    "enabled": entries.get("enabled") is not False,
                    # The function this definition runs, when it has one.
                    # Reported because it is the only record of which
                    # functions a source's tasks reference -- an uninstall
                    # reads it to find what its own runs distilled, which
                    # no bundle source lists.
                    "entrypoint": None if entrypoint is None else int(entrypoint),
                },
            )
        return sorted(planted, key=lambda task: task["task_id"])

    def set_custom_tasks_enabled(
        self,
        *,
        managed_by: str,
        enabled: bool,
        destination: str | None = None,
    ) -> List[int]:
        """Arm or disarm every task definition one source planted.

        Custom-synced tasks are born disarmed (``_insert_custom_task``
        writes ``enabled=False``), so the installer that planted them must
        arm them once their requirements are met — and hold them, still
        planted and visible, while a required connection is missing.

        When arming, a one-shot whose run already happened is skipped
        rather than raised on: re-arming the rest of a source's tasks must
        not fail because its provisioning task already did its job.

        Returns the ids of the definitions whose flag actually changed. A
        definition already in the requested state is left unwritten and
        unreported, so callers reacting to the return value — a connect
        event arming whatever a missing app held — see real transitions,
        not every reconcile pass restating the status quo.
        """
        tasks_context, _meta_context, _is_personal = self._sync_destination_contexts(
            destination,
        )
        store = self._store_for_task_context(tasks_context)
        rows = store.get_rows(filter=managed_rows_filter(managed_by), limit=1000)
        touched: List[int] = []
        for row in rows:
            entries = dict(row.entries or {})
            task_id = entries.get("task_id")
            if task_id is None:
                continue
            if (entries.get("enabled") is not False) == enabled:
                continue
            if enabled:
                task = self._get_task_or_raise(int(task_id))
                if self._one_shot_already_ran(task) is not None:
                    continue
            self._set_tasks_enabled(task_ids=int(task_id), enabled=enabled)
            touched.append(int(task_id))
        return touched

    def _workflow_run_settings(
        self,
        task: Task,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """The installed settings of the workflow that planted *task*, if any.

        Resolved once at run start and carried on the run request, so the
        run's configuration is a deterministic input rather than something
        the actor has to discover mid-run. Best-effort by design: a task
        whose settings cannot be read still runs, with the request saying
        nothing about settings rather than something wrong about them.
        """
        if not task.custom_hash:
            return None, None
        try:
            managed_by, released = self._task_reconcile_owner(int(task.task_id))
            if released or not managed_by or managed_by == MANAGED_BY_DEPLOYMENT:
                return None, None
            from unify.manager_registry import ManagerRegistry

            manager = ManagerRegistry.get_workflow_manager()
            if manager is None:
                return None, None
            settings = manager.get_installation_params(
                slug=managed_by,
                destination=task.destination,
            )
        except Exception:
            logger.warning(
                "Could not resolve installation settings for task %s; the "
                "run proceeds without them",
                task.task_id,
                exc_info=True,
            )
            return None, None
        if not isinstance(settings, dict):
            return None, None
        return managed_by, settings

    def _task_reconcile_owner(self, task_id: int) -> tuple[Optional[str], bool]:
        """Who reconciles this row, and whether the user already owns it.

        Both live on the row rather than the typed model: ``managed_by`` is
        reconcile provenance, not a task attribute. The pair is read together
        because a null ``managed_by`` is ambiguous on its own — a released
        row and a row written before ``managed_by`` existed both have none,
        and only the first is the user's.
        """
        rows = self._store.get_rows(filter=f"task_id == {task_id}", limit=1)
        if not rows:
            return None, False
        entries = dict(rows[0].entries or {})
        managed_by = entries.get("managed_by")
        return (
            str(managed_by) if managed_by else None,
            bool(entries.get(CUSTOM_RELEASED_FIELD)),
        )

    def _update_task(
        self,
        *,
        task_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        start_at: Any = _UNSET,
        deadline: Any = _UNSET,
        repeat: Any = _UNSET,
        priority: Optional[Union[Priority, str]] = None,
        trigger: Any = _UNSET,
        entrypoint: Any = _UNSET,
        offline: Any = _UNSET,
        requires_filesystem: Any = _UNSET,
        requires_computer: Any = _UNSET,
        enabled: Any = _UNSET,
        destination: str | None = None,
        _root_applied: bool = False,
    ) -> Dict[str, Any]:
        """Update mutable fields on an existing task.

        Accepts any subset of a task's mutable attributes (name, description,
        schedule, deadline, repeat, priority, trigger, entrypoint, offline
        flag, enabled flag).  Only the fields that are explicitly provided are
        changed; omitted fields keep their current values.

        ``start_at``, ``deadline``, ``repeat`` and ``trigger`` distinguish
        *omitted* from *explicitly null*: passing ``None`` clears the field.
        ``start_at=None`` removes the schedule (and sweeps ``repeat`` with it
        unless a new ``repeat`` is set in the same call — a cadence has
        nothing to anchor to without a schedule). Converting a scheduled task
        to a triggered one is a single call: ``trigger=..., start_at=None``.

        Schedule, recurrence and ``enabled`` edits re-arm the series by
        rewriting the definition; the deployment's scheduler picks them up
        when it next materializes a run. As with creation, no ``scheduled``
        run row may be visible in the meantime — that is not evidence the
        edit failed to take.
        """

        if not _root_applied:
            resolved_destination = (
                destination or os.environ.get("TASK_DESTINATION") or None
            )
            if resolved_destination is None:
                resolved_destination = (
                    self._resolve_task_for_mutation(task_id).destination
                    or PERSONAL_DESTINATION
                )
            with self._use_task_destination(resolved_destination):
                return self._update_task(
                    task_id=task_id,
                    name=name,
                    description=description,
                    start_at=start_at,
                    deadline=deadline,
                    repeat=repeat,
                    priority=priority,
                    trigger=trigger,
                    entrypoint=entrypoint,
                    offline=offline,
                    requires_filesystem=requires_filesystem,
                    requires_computer=requires_computer,
                    enabled=enabled,
                    destination=resolved_destination,
                    _root_applied=True,
                )

        self._ensure_not_active_task(task_id)

        trigger_provided = trigger is not _UNSET
        offline_provided = offline is not _UNSET
        requires_filesystem_provided = requires_filesystem is not _UNSET
        requires_computer_provided = requires_computer is not _UNSET
        enabled_provided = enabled is not _UNSET
        start_at_provided = start_at is not _UNSET
        deadline_provided = deadline is not _UNSET
        repeat_provided = repeat is not _UNSET
        schedule_cleared = start_at_provided and start_at is None
        task = self._resolve_task_for_mutation(task_id)

        # A managed task (custom_hash set) gets its authored fields from a
        # source, and who that source is decides what an edit means.
        #
        # The deployment's own tasks are infrastructure: a runtime mutation
        # silently diverges from the source until a resync overwrites it, and
        # because the sync short-circuits on its aggregate hash, a mutation
        # that damages a derived reference (e.g. nulling the entrypoint) is
        # never healed. Those refuse the edit; runtime state (enabled) stays
        # mutable and authored fields change in the source.
        #
        # A task a workflow planted is the user's to shape. The first authored
        # edit hands the row over — provenance is cleared, identity is kept —
        # so the workflow stops reconciling it and later updates leave the
        # edit standing instead of overwriting it.
        release_ownership = False
        if task.custom_hash:
            authored_touched = sorted(
                field
                for field, provided in (
                    ("name", name is not None),
                    ("description", description is not None),
                    ("start_at", start_at_provided),
                    ("deadline", deadline_provided),
                    ("repeat", repeat_provided),
                    ("priority", priority is not None),
                    ("trigger", trigger_provided),
                    ("entrypoint", entrypoint is not _UNSET),
                    ("offline", offline_provided),
                    ("requires_filesystem", requires_filesystem_provided),
                    ("requires_computer", requires_computer_provided),
                )
                if provided
            )
            if authored_touched:
                managed_by, already_released = self._task_reconcile_owner(task_id)
                if already_released:
                    # Handed over on an earlier edit; nobody reconciles it now,
                    # so there is nothing left to refuse or to release again.
                    pass
                elif not managed_by or managed_by == MANAGED_BY_DEPLOYMENT:
                    raise ValueError(
                        f"Task {task_id} is deployment-owned (custom_hash set); "
                        "refusing runtime update of authored field(s) "
                        f"{', '.join(authored_touched)}. Edit the bundle source "
                        "and re-sync via deployment reconcile.",
                    )
                else:
                    release_ownership = True

        if (
            name is None
            and description is None
            and not start_at_provided
            and not deadline_provided
            and not repeat_provided
            and priority is None
            and not trigger_provided
            and entrypoint is _UNSET
            and not offline_provided
            and not requires_filesystem_provided
            and not requires_computer_provided
            and not enabled_provided
        ):
            raise ValueError("At least one field must be provided for an update.")

        if (
            trigger_provided
            and trigger is not None
            and task.schedule is not None
            and not schedule_cleared
        ):
            raise ValueError(
                "Cannot add a trigger while a schedule exists. Clear it in the "
                "same call (start_at=None) or remove the schedule first.",
            )

        if isinstance(start_at, datetime):
            start_at = start_at.isoformat()
        if isinstance(deadline, datetime):
            deadline = deadline.isoformat()

        schedule_payload: Optional[Dict[str, Any]] = None
        if start_at_provided and start_at is not None:
            if task.trigger is not None and not (trigger_provided and trigger is None):
                raise ValueError(
                    "Cannot add or update start_at while the task is trigger-based.",
                )
            schedule_payload = {"start_at": start_at}

        prospective_trigger: TriggerLike
        if not trigger_provided:
            prospective_trigger = task.trigger
        elif trigger is None:
            prospective_trigger = None
        elif isinstance(trigger, dict):
            prospective_trigger = parse_task_trigger(trigger)
        else:
            prospective_trigger = trigger

        prospective_schedule: ScheduleLike
        if schedule_cleared:
            prospective_schedule = None
        elif schedule_payload is not None:
            prospective_schedule = schedule_payload
        else:
            prospective_schedule = task.schedule
        if prospective_schedule is not None and prospective_trigger is not None:
            raise ValueError("A task cannot have both a schedule and a trigger.")

        if start_at_provided or trigger_provided:
            self._validate_scheduled_invariants(
                schedule=prospective_schedule,
                trigger=prospective_trigger,
                err_prefix=f"While updating task {task_id}:",
            )

        entries: Dict[str, Any] = {}
        if name is not None:
            entries["name"] = name
        if description is not None:
            entries["description"] = description
        if deadline_provided:
            entries["deadline"] = deadline  # explicit None clears the deadline
        if repeat_provided:
            if repeat is None:
                entries["repeat"] = None
            else:
                normalized_repeat = normalize_repeat_patterns(
                    [
                        RepeatPattern(**item) if isinstance(item, dict) else item
                        for item in repeat
                    ],
                )
                entries["repeat"] = [
                    (
                        item.model_dump(mode="json")
                        if isinstance(item, RepeatPattern)
                        else item
                    )
                    for item in normalized_repeat or []
                ]
        if priority is not None:
            if isinstance(priority, Priority):
                entries["priority"] = priority
            else:
                try:
                    entries["priority"] = Priority(str(priority))
                except Exception as exc:
                    raise ValueError(f"Invalid priority {priority!r}.") from exc
        if trigger_provided:
            if prospective_trigger is None:
                entries["trigger"] = None
            elif isinstance(prospective_trigger, BaseModel):
                entries["trigger"] = prospective_trigger.model_dump(mode="json")
            else:
                entries["trigger"] = prospective_trigger
        if schedule_payload is not None:
            entries["schedule"] = schedule_payload
        elif schedule_cleared:
            entries["schedule"] = None
            # A repeat pattern has nothing to anchor to without a schedule;
            # clearing the schedule sweeps the cadence too unless the caller
            # replaced it explicitly in the same call.
            if task.repeat is not None and not repeat_provided:
                entries["repeat"] = None
        if entrypoint is not _UNSET:
            if entrypoint is None:
                entries["entrypoint"] = None
            else:
                try:
                    entries["entrypoint"] = int(entrypoint)
                except Exception as exc:
                    raise ValueError("entrypoint must be an integer or None") from exc
        if offline_provided:
            if isinstance(offline, str):
                normalized_offline = offline.strip().lower()
                if normalized_offline in {"true", "1"}:
                    offline = True
                elif normalized_offline in {"false", "0"}:
                    offline = False
                else:
                    raise ValueError("offline must be a boolean value")
            else:
                offline = bool(offline)
            entries["offline"] = offline
        if requires_filesystem_provided:
            if isinstance(requires_filesystem, str):
                normalized_requires_filesystem = requires_filesystem.strip().lower()
                if normalized_requires_filesystem in {"true", "1"}:
                    requires_filesystem = True
                elif normalized_requires_filesystem in {"false", "0"}:
                    requires_filesystem = False
                else:
                    raise ValueError("requires_filesystem must be a boolean value")
            else:
                requires_filesystem = bool(requires_filesystem)
            entries["requires_filesystem"] = requires_filesystem
        if requires_computer_provided:
            if isinstance(requires_computer, str):
                normalized_requires_computer = requires_computer.strip().lower()
                if normalized_requires_computer in {"true", "1"}:
                    requires_computer = True
                elif normalized_requires_computer in {"false", "0"}:
                    requires_computer = False
                else:
                    raise ValueError("requires_computer must be a boolean value")
            else:
                requires_computer = bool(requires_computer)
            entries["requires_computer"] = requires_computer
        if enabled_provided:
            if isinstance(enabled, str):
                normalized_enabled = enabled.strip().lower()
                if normalized_enabled in {"true", "1"}:
                    enabled = True
                elif normalized_enabled in {"false", "0"}:
                    enabled = False
                else:
                    raise ValueError("enabled must be a boolean value")
            else:
                enabled = bool(enabled)
            entries["enabled"] = enabled

        log_ids = self._store.get_rows(
            filter=f"task_id == {task_id}",
            return_ids_only=True,
        )

        if release_ownership:
            # Identity stays — ``custom_key`` still names the entry this row
            # grew from, which is how the workflow keeps counting it as
            # planted — and so does the content hash. Provenance goes, and
            # the flag says the absence is deliberate rather than a row that
            # predates ``managed_by``.
            #
            # A separate write, not part of the patch: a provider-event
            # update routes its fields through a typed classifier that knows
            # only authored and runtime task fields, and provenance is
            # neither. Stamped *before* the edit so a failure leaves the row
            # managed and unedited — the user retries and it works — rather
            # than edited and still claimed by the workflow, which the next
            # reconcile would silently overwrite.
            self._write_log_entries(
                logs=log_ids,
                entries={"managed_by": None, CUSTOM_RELEASED_FIELD: True},
            )

        if self._task_has_provider_event_trigger(task):
            return self._write_provider_event_task_update(
                task_id=task_id,
                task=task,
                entries=entries,
            )

        return self._write_log_entries(
            logs=log_ids,
            entries=entries,
        )

    def _write_provider_event_task_update(
        self,
        *,
        task_id: int,
        task: Task,
        entries: Dict[str, Any],
    ) -> Dict[str, str]:
        """Route one provider-event patch across typed API and runtime logs."""

        log_ids = self._store.get_rows(
            filter=f"task_id == {task_id}",
            return_ids_only=True,
        )
        authored_entries, runtime_entries = split_provider_event_task_update(
            entries,
        )
        if authored_entries and runtime_entries:
            raise ValueError(
                "Cannot update authored and runtime provider-event fields in one call.",
            )
        if authored_entries:
            if task.task_revision is None:
                raise ValueError(
                    f"Task {task_id} is missing task_revision; re-read before updating.",
                )
            try:
                typed_tasks_client.patch_task(
                    task_id=task_id,
                    expected_task_revision=int(task.task_revision),
                    updates=authored_entries,
                )
            except TaskRevisionConflictError as exc:
                return task_revision_conflict_outcome(exc)
            return {"detail": "Provider-event authored update applied."}
        if runtime_entries:
            return self._write_log_entries(
                logs=log_ids,
                entries=runtime_entries,
            )
        return {"detail": "No-op provider-event task update."}

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_tasks on the first step when configured."""

        if (
            SETTINGS.FIRST_ASK_TOOL_IS_SEARCH
            and step_index < 1
            and "search_tasks" in current_tools
        ):
            return ("required", {"search_tasks": current_tools["search_tasks"]})
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step when configured."""

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    def _write_log_entries(
        self,
        *,
        logs: Union[int, unisdk.Log, List[Union[int, unisdk.Log]]],
        entries: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, str]:
        """Centralize task-row writes through the current store."""

        return self._store.update(
            logs=logs,
            entries=entries,
        )

    def _list_provider_trigger_catalog(
        self,
        *,
        canonical_app_slug: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> ToolOutcome:
        """List staged provider triggers visible for this assistant's connected apps.

        Returns catalog metadata plus trigger slugs/config schemas for apps the
        assistant already has an active integration connection for. An empty
        trigger list usually means no matching connection yet, not that the
        provider lacks the trigger globally. Prefer connecting the app first,
        then re-list the catalog before enabling a provider-event task.

        The unfiltered catalog can be large. Once
        ``list_provider_trigger_connections`` shows which app/backend is
        connected, pass that app's ``canonical_app_slug`` here to narrow the
        response, and use ``limit``/``offset`` to page through the rest.
        """

        catalog = typed_tasks_client.get_trigger_catalog(
            canonical_app_slug=canonical_app_slug,
            limit=limit,
            offset=offset,
        )
        return {
            "outcome": "provider trigger catalog listed",
            "details": annotate_provider_trigger_catalog(
                catalog if isinstance(catalog, dict) else {},
            ),
        }

    def _list_provider_trigger_connections(
        self,
        *,
        canonical_app_slug: str | None = None,
        backend_id: str | None = None,
    ) -> ToolOutcome:
        """List assistant-owned connections usable for provider triggers.

        Returns only active assistant-scoped integration connections that can
        back provider-event task triggers. Filter by ``canonical_app_slug`` and
        ``backend_id`` when the actor already knows which app/backend it needs.
        """

        connections = list_eligible_provider_trigger_connections(
            canonical_app_slug=canonical_app_slug,
            backend_id=backend_id,
        )
        return {
            "outcome": "provider trigger connections listed",
            "details": annotate_provider_trigger_connections(connections),
        }

    def _describe_provider_trigger(
        self,
        *,
        provider_trigger_slug: str,
        backend_id: str,
    ) -> ToolOutcome:
        """Return config schema for one staged provider trigger.

        Use the catalog listing first to discover valid
        ``provider_trigger_slug`` / ``backend_id`` pairs, then call this tool
        to inspect the trigger's config schema before authoring a task trigger.
        """

        catalog = typed_tasks_client.get_trigger_catalog()
        trigger = describe_provider_trigger(
            provider_trigger_slug=provider_trigger_slug,
            backend_id=backend_id,
            catalog_triggers=catalog.get("triggers"),
        )
        return {
            "outcome": "provider trigger described",
            "details": trigger,
        }

    def _list_provider_trigger_resources(
        self,
        *,
        target_resource_family: str,
        query: str | None = None,
        drive_id: str | None = None,
        parent_item_id: str | None = None,
    ) -> ToolOutcome:
        """List workspace resources for native provider-event trigger_config.

        Pass ``target_resource_family`` from ``describe_provider_trigger``. For
        Drive, omit parents to list roots, pass ``drive_id`` + ``parent_item_id``
        to browse children, or pass ``query`` to search by name.
        """

        details = list_provider_trigger_resources(
            target_resource_family=target_resource_family,
            query=query,
            drive_id=drive_id,
            parent_item_id=parent_item_id,
        )
        return {
            "outcome": "provider trigger resources listed",
            "details": details,
        }

    def _get_provider_trigger_health(self, *, task_id: int) -> ToolOutcome:
        """Inspect composed provider-trigger health, coverage, and remediation.

        Returns a provider-neutral composed lifecycle state plus runtime
        health, coverage windows, and remediation guidance for actor responses.
        """

        self._get_provider_event_task_or_raise(task_id)
        health = typed_tasks_client.get_trigger_health(task_id=task_id)
        return {
            "outcome": "provider trigger health inspected",
            "details": compose_provider_trigger_state(health),
        }

    def _get_provider_event_context(
        self,
        *,
        task_id: int,
        run_id: int,
        include_source_body: bool = False,
    ) -> ToolOutcome:
        """Inspect authorized provider-event context for one run.

        Returns the curated projection and envelope by default. Include raw
        source_body only when the user explicitly requests advanced inspection.
        """

        self._get_provider_event_task_or_raise(task_id)
        context = typed_tasks_client.get_event_context(
            task_id=task_id,
            run_id=run_id,
        )
        return {
            "outcome": "provider event context inspected",
            "details": sanitize_event_context_for_actor(
                context,
                include_source_body=include_source_body,
            ),
        }

    def get_run_event_children(
        self,
        *,
        run_key: str,
        parent: str | None = None,
        limit: int = 50,
        events_base_context: str | None = None,
    ) -> dict[str, Any]:
        """Return **immediate** EventBus children for one ``Tasks/Executions`` run.

        Depth-1 only. Pass a child's ``node_id`` as ``parent`` to drill one level
        deeper. Assign in ``execute_code`` and inspect selectively — do not dump
        the full forest into the observation.
        """

        from unify.task_scheduler.task_run_events import (
            fetch_task_run_events,
            project_immediate_children,
        )

        key = str(run_key or "").strip()
        if not key:
            raise ValueError("run_key must be a non-empty string")

        events_root = self._resolve_events_base_context(events_base_context)
        hierarchy_prefix = str(parent).strip() if parent else None
        # Root listing needs the Task.run segment; fetch by run_key only.
        # Drills may narrow with hierarchy_label.startswith(parent).
        tree = fetch_task_run_events(
            key,
            events_base_context=events_root,
            hierarchy_prefix=hierarchy_prefix,
            limit_per_type=1000,
        )
        children = project_immediate_children(
            tree.rows,
            run_key=key,
            parent_prefix=hierarchy_prefix,
            limit=int(limit),
        )
        return {
            "run_key": key,
            "parent": hierarchy_prefix,
            "events_base_context": tree.events_base_context,
            "children": children,
        }

    def get_run_event(
        self,
        *,
        run_key: str,
        node_id: str,
        event_id: str | None = None,
        events_base_context: str | None = None,
    ) -> dict[str, Any]:
        """Return near-raw EventBus row(s) for **one** hierarchy node only.

        Use after ``get_run_event_children`` when a specific node needs payload
        detail — a failed ``execute_code``, say. Descendants are not included;
        drill with ``get_run_event_children`` instead. Pass ``event_id`` from
        the child stub's ``event_ids`` when several rows share the node (a
        ManagerMethod's incoming and outgoing rows, for example). Assign in
        ``execute_code`` and inspect selectively — payloads are unsummarized
        and do not belong in the observation.
        """

        from unify.task_scheduler.task_run_events import (
            fetch_task_run_events,
            find_rows_at_node,
        )

        key = str(run_key or "").strip()
        if not key:
            raise ValueError("run_key must be a non-empty string")
        nid = str(node_id or "").strip()
        if not nid:
            raise ValueError("node_id must be a non-empty hierarchy prefix")

        events_root = self._resolve_events_base_context(events_base_context)
        tree = fetch_task_run_events(
            key,
            events_base_context=events_root,
            hierarchy_prefix=nid,
            limit_per_type=1000,
        )
        events = find_rows_at_node(
            tree.rows,
            node_id=nid,
            event_id=event_id,
        )
        return {
            "run_key": key,
            "node_id": nid,
            "events_base_context": tree.events_base_context,
            "events": events,
        }

    @staticmethod
    def _resolve_events_base_context(events_base_context: str | None) -> str:
        """Resolve the assistant Events root for task-run diagnostics."""

        if events_base_context and str(events_base_context).strip():
            return str(events_base_context).strip()
        from unify.common.log_utils import _get_assistant_id, _get_user_id

        user_id = _get_user_id()
        assistant_id = _get_assistant_id()
        if not user_id or not assistant_id:
            raise RuntimeError(
                "Cannot resolve Events context: SESSION_DETAILS user/assistant "
                "ids are required when events_base_context is omitted.",
            )
        return f"{user_id}/{assistant_id}/Events"

    def _pause_provider_trigger(
        self,
        *,
        task_id: int,
        task_revision: int,
    ) -> ToolOutcome:
        """Pause provider-event automation while keeping manual run available.

        Closes the acceptance fence for new provider deliveries without
        disabling the global task.enabled gate or blocking manual execution.
        """

        self._get_provider_event_task_or_raise(task_id)
        try:
            updated = typed_tasks_client.pause_trigger(
                task_id=task_id,
                expected_task_revision=int(task_revision),
            )
        except TaskRevisionConflictError as exc:
            return task_revision_conflict_outcome(exc)
        return {
            "outcome": "provider trigger paused",
            "details": {
                "task_id": task_id,
                "task_revision": updated.get("task_revision"),
            },
        }

    def _resume_provider_trigger(
        self,
        *,
        task_id: int,
        task_revision: int,
    ) -> ToolOutcome:
        """Resume provider-event automation for one task.

        Reopens provider automation under the current authored revision and
        schedules reconciliation for the active subscription generation.
        """

        self._get_provider_event_task_or_raise(task_id)
        try:
            updated = typed_tasks_client.resume_trigger(
                task_id=task_id,
                expected_task_revision=int(task_revision),
            )
        except TaskRevisionConflictError as exc:
            return task_revision_conflict_outcome(exc)
        return {
            "outcome": "provider trigger resumed",
            "details": {
                "task_id": task_id,
                "task_revision": updated.get("task_revision"),
            },
        }

    def _retry_provider_trigger(self, *, task_id: int) -> ToolOutcome:
        """Request immediate provider-trigger reconciliation.

        Schedules binding and subscription reconciliation without changing the
        authored task revision or mutating trigger intent directly.
        """

        self._get_provider_event_task_or_raise(task_id)
        result = typed_tasks_client.retry_trigger(task_id=task_id)
        return {
            "outcome": "provider trigger reconciliation requested",
            "details": result,
        }

    def _export_provider_event_context(
        self,
        *,
        task_id: int,
        run_id: int,
    ) -> ToolOutcome:
        """Export authorized provider-event context with audit logging.

        Returns the full authorized context for user-requested export while
        keeping credentials and backend details out of the actor response.
        """

        self._get_provider_event_task_or_raise(task_id)
        context = typed_tasks_client.export_event_context(
            task_id=task_id,
            run_id=run_id,
        )
        return {
            "outcome": "provider event context exported",
            "details": sanitize_event_context_for_actor(
                context,
                include_source_body=True,
            ),
        }

    def _delete_provider_event_context(
        self,
        *,
        task_id: int,
        run_id: int,
        task_revision: int,
    ) -> ToolOutcome:
        """Delete provider-event context for one run.

        Makes the event context unavailable immediately and records the
        deletion under the current authored task revision when supplied.
        """

        task = self._get_provider_event_task_or_raise(task_id)
        if task.task_revision is None:
            raise ValueError(
                f"Task {task_id} is missing task_revision; re-read before deleting context.",
            )
        if int(task.task_revision) != int(task_revision):
            return task_revision_conflict_outcome(
                TaskRevisionConflictError(
                    latest_task_revision=int(task.task_revision),
                ),
            )
        typed_tasks_client.delete_event_context(task_id=task_id, run_id=run_id)
        return {
            "outcome": "provider event context deleted",
            "details": {"task_id": task_id, "run_id": run_id},
        }

    @staticmethod
    def _task_has_provider_event_trigger(task: Task) -> bool:
        trigger = parse_task_trigger(task.trigger)
        return isinstance(trigger, ProviderEventTrigger)

    def _start_loop(
        self,
        client: unillm.AsyncUnify,
        text: str,
        tools: ToolsDict,
        *,
        loop_id: str,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        log_steps: bool = True,
        tool_policy: Optional[
            Union[
                Literal["default"],
                Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            ]
        ] = None,
        handle_cls: Optional[type[SteerableToolHandle]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        clarification_queues: Optional[Tuple[asyncio.Queue, asyncio.Queue]] = None,
    ) -> SteerableToolHandle:
        """Centralized wrapper around start_async_tool_loop."""

        return start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=loop_id,
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            log_steps=log_steps,
            tool_policy=tool_policy,
            handle_cls=handle_cls,
            response_format=response_format,
            clarification_queues=clarification_queues,
        )

    def _wrap_result_with_messages(
        self,
        handle: SteerableToolHandle,
        client: unillm.AsyncUnify,
    ) -> SteerableToolHandle:
        """Wrap handle.result so it also returns client messages."""

        original_result = handle.result

        async def wrapped_result():
            answer = await original_result()
            return answer, client.messages

        handle.result = wrapped_result  # type: ignore[assignment]
        return handle

    def _task_from_typed_response(self, typed_response: dict[str, Any]) -> Task:
        """Build one Task from a typed Tasks API row."""

        entries = {"task_id": int(typed_response["task_id"])}
        for key in (
            "task_revision",
            "provider_event_binding_id",
            "name",
            "description",
            "trigger",
            "schedule",
            "enabled",
            "offline",
            "priority",
            "entrypoint",
            "requires_filesystem",
            "requires_computer",
        ):
            if key in typed_response and typed_response[key] is not None:
                entries[key] = typed_response[key]
        entries.setdefault("assistant_id", SESSION_DETAILS.assistant_context)
        return Task(**self._sanitize_activation(entries))

    def _get_provider_event_task_or_raise(self, task_id: int) -> Task:
        """Return one provider-event task from the typed Tasks API.

        Authored provider-event rows are owned by the typed Tasks API. Tool
        paths that mutate or inspect that contract must not depend on a Tasks
        log mirror in the current session root.
        """

        try:
            typed_response = typed_tasks_client.get_task(task_id=task_id)
        except ValueError as exc:
            if str(exc) == "Task not found.":
                raise ValueError(f"No task found with id={task_id}") from exc
            raise
        task = self._task_from_typed_response(typed_response)
        if not self._task_has_provider_event_trigger(task):
            raise ValueError(
                f"Task {task_id} does not have a provider-event trigger.",
            )
        return task

    def _resolve_task_for_mutation(self, task_id: int) -> Task:
        """Resolve one task for authored update/delete, preferring typed CAS."""

        try:
            task = self._get_task_or_raise(task_id)
        except ValueError as exc:
            if "multiple task roots" in str(exc):
                raise
            return self._get_provider_event_task_or_raise(task_id)
        if self._task_has_provider_event_trigger(task):
            return self._get_provider_event_task_or_raise(task_id)
        return task

    def _get_task_or_raise(self, task_id: int) -> Task:
        """Fetch exactly one task id or raise when it is missing or ambiguous."""

        tasks = self._filter_tasks(filter=f"task_id == {task_id}", limit=1000)
        if not tasks:
            raise ValueError(f"No task found with id={task_id}")
        destinations = {task.destination or PERSONAL_DESTINATION for task in tasks}
        if len(destinations) > 1:
            raise ValueError(
                f"Task id {task_id} exists in multiple task roots; provide destination.",
            )
        return tasks[0]

    def _search_tasks(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Task]:
        """Run semantic search across all tasks and return the closest matches.

        Uses vector similarity to find tasks whose name or description is
        semantically close to the provided references.  Optionally limits
        results to at most ``k`` rows.  Returns an empty list when no tasks
        are stored or when the query produces no meaningful matches.
        """

        allowed_fields: List[str] = [
            "task_id",
            "name",
            "description",
            "priority",
            "schedule",
            "deadline",
        ]
        filled = table_search_top_k(
            self._ctx,
            references,
            k=k,
            allowed_fields=allowed_fields,
            row_filter=None,
            unique_id_field="task_id",
        )
        return [Task(**self._sanitize_activation(dict(lg))) for lg in filled]

    def _list_task_runs(
        self,
        *,
        task_id: int,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return one task's runs, newest first: when each ran and what happened.

        Use this for any question about a task actually running, as opposed to
        how it is set up: did it run today, when did it last run, did it fail,
        what did it produce, when is it due next. The task row itself carries
        only authored intent — its schedule and whether it is armed — and can
        never answer these.

        Each entry carries ``state`` (``scheduled`` for an occurrence still
        ahead, then ``running`` and one of ``completed`` / ``failed`` /
        ``cancelled``), ``scheduled_for`` (the moment it belongs to),
        ``started_at`` / ``completed_at``, ``result_summary`` for what the run
        produced, and ``error`` when it failed. A single ``scheduled`` entry
        dated ahead is the next run. Its absence does not mean the task is
        disarmed: run rows are materialized by the deployment's scheduler,
        not by task creation, and for a task that has never run the first
        one may appear only when that run starts. Whether a task is armed
        is answered by its definition (``enabled`` plus its schedule or
        trigger); this listing answers what actually happened, and — once a
        ``scheduled`` row exists — exactly when the next run is due.

        ``run_key`` identifies one run, and is what
        ``get_run_event_children`` needs to walk that run's internals when a
        failure needs diagnosing beyond its ``error``.

        Parameters
        ----------
        task_id : int
            The task whose runs to list.
        limit : int, default ``20``
            Maximum runs returned.
        """

        task = self._resolve_task_for_mutation(task_id)
        return list_task_run_history(
            task_id=int(task_id),
            destination=task.destination,
            limit=limit,
        )

    def _filter_tasks(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Task]:
        """Filter tasks using a boolean expression over task fields.

        Returns all task rows that match the given filter expression.
        The expression uses field names from the task schema and Python
        literals (e.g. ``task_id == 42``, ``priority == 'high'``).  Boolean
        fields such as ``enabled`` compare against ``True``/``False``
        capitalized — ``enabled == true`` is not a boolean literal here and
        matches no rows rather than raising.  Returns an empty list when no
        rows match.
        """

        normalized_filter = normalize_filter_expr(filter)
        include_fields = list(Task.model_fields.keys())

        rows: list[dict[str, Any]] = []
        for context_name in self._read_task_contexts():
            store = self._store_for_task_context(context_name)
            destination = self._destination_from_task_context(context_name)
            root_logs = store.get_rows(
                filter=normalized_filter,
                offset=0,
                limit=max(limit + offset, 1000 if limit >= 1000 else limit),
                return_ids_only=False,
                include_fields=include_fields,
            )
            # Order by log id so callers that disambiguate duplicate rows for one
            # task_id (a pre-migration shape) always resolve the oldest row.
            for log in sorted(root_logs, key=lambda obj: int(obj.id)):
                row = dict(log.entries or {})
                row.setdefault("assistant_id", SESSION_DETAILS.assistant_context)
                row["destination"] = destination
                rows.append(row)
        rows = rows[offset : offset + limit]

        def _rehydrate_repeat(item: dict) -> dict:
            if not isinstance(item, dict):
                return item
            out = dict(item)
            freq = out.get("frequency")
            if isinstance(freq, str):
                token = freq.split(".")[-1] if "." in freq else freq
                try:
                    out["frequency"] = Frequency[token]
                except Exception:
                    try:
                        out["frequency"] = Frequency(token)
                    except Exception:
                        pass

            weekdays = out.get("weekdays")
            if isinstance(weekdays, list):
                new_weekdays = []
                for weekday in weekdays:
                    if isinstance(weekday, str):
                        token = weekday.split(".")[-1] if "." in weekday else weekday
                        try:
                            new_weekdays.append(Weekday[token])
                        except Exception:
                            try:
                                new_weekdays.append(Weekday(token))
                            except Exception:
                                new_weekdays.append(weekday)
                    else:
                        new_weekdays.append(weekday)
                out["weekdays"] = new_weekdays

            for optional_key in ("count", "until", "time_of_day"):
                if optional_key not in out:
                    out[optional_key] = None
            return out

        hydrated: list[Task] = []
        for row in rows:
            repeat = row.get("repeat")
            if isinstance(repeat, list):
                row["repeat"] = [_rehydrate_repeat(item) for item in repeat]
            sanitized = self._sanitize_activation(row)
            hydrated.append(Task(**sanitized))
        return hydrated

    def _get_columns(self) -> Dict[str, str]:
        """Return the tasks-table schema for the current context."""

        return self._store.fields

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, str] | list[str]:
        """Return available task columns, optionally with types."""

        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _num_tasks(self) -> int:
        """Return the total number of rows in the current Tasks context."""

        if self._num_tasks_cached is None:
            try:
                self._num_tasks_cached = int(
                    self._store.get_metric_count(key="task_id"),
                )
            except Exception:
                self._num_tasks_cached = 0
        return int(self._num_tasks_cached)

    @read_only
    def _reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """Compute aggregate metrics over the current task list.

        Supports count, sum, mean, min, max, and other standard reductions
        grouped by one or more task fields (e.g. ``priority``, ``enabled``).
        Any ``filter`` follows the same expression rules as the task filter
        tool, including capitalized ``True``/``False`` for boolean fields.
        Returns a dictionary of group keys to metric values.
        """

        return reduce_logs(
            context=self._ctx,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    def _find_name_desc_collisions(
        self,
        *,
        name: str,
        description: str,
        limit: int = 2,
    ) -> List[Dict[str, Any]]:
        """Return existing rows that collide on name or description."""

        try:
            logs = self._store.get_rows(
                filter=f"name == {name!r} or description == {description!r}",
                limit=limit,
                return_ids_only=False,
            )
        except Exception:
            return []
        return [dict(log.entries or {}) for log in logs]

    @overload
    def _sanitize_activation(self, task: Dict[str, Any]) -> Dict[str, Any]: ...

    @overload
    def _sanitize_activation(self, task: Task) -> Task: ...

    def _sanitize_activation(
        self,
        task: Union[Dict[str, Any], Task],
    ) -> Union[Dict[str, Any], Task]:
        """Prepare a task row for ``Task`` construction after an Orchestra read.

        Drops ``status`` and ``activated_by``, which definitions no longer
        carry: run state lives on ``Tasks/Executions``. Rows written before the
        migration still have those columns, and a read must not fail on one.

        Also drops keys whose value is ``None``: Orchestra ``get_logs`` /
        ``include_fields`` returns explicit null for unset columns, and Pydantic
        v2 does not apply ``Field(default=…)`` when the key is present with
        ``None`` (e.g. ``offline`` / ``enabled``).
        """
        if isinstance(task, Task):
            return task
        return {
            key: value
            for key, value in task.items()
            if value is not None and key not in _LEGACY_DEFINITION_FIELDS
        }

    def _meta_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Tasks/Meta context."""
        root_context = ContextRegistry.write_root(
            self,
            TASKS_META_TABLE,
            destination=destination,
        )
        return f"{root_context.strip('/')}/{TASKS_META_TABLE}"

    @contextmanager
    def _temporary_tasks_meta_context(self, context: str):
        """Temporarily bind task meta reads/writes to a resolved context."""
        with self._destination_context_lock:
            original = self._meta_ctx
            self._meta_ctx = context
            try:
                yield
            finally:
                self._meta_ctx = original

    def _sync_destination_contexts(
        self,
        destination: str | None,
    ) -> tuple[str, str, bool]:
        """Return destination-scoped tasks context, meta context, and personal flag."""
        data_context = self._task_context_for_destination(destination)
        meta_context = self._meta_context_for_destination(destination)
        return data_context, meta_context, destination in (None, "personal")

    def _get_stored_custom_tasks_hash(
        self,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> str:
        field = stored_hash_field("custom_tasks_hash", managed_by)
        try:
            logs = unisdk.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get(field, "") or ""
        except Exception as exc:
            logger.warning("Failed to read custom tasks hash: %s", exc)
        return ""

    def _store_custom_tasks_hash(
        self,
        hash_value: str,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        field = stored_hash_field("custom_tasks_hash", managed_by)
        try:
            logs = unisdk.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unisdk.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={field: hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, field: hash_value}],
                    stamp_authoring=True,
                )
        except Exception as exc:
            logger.warning("Failed to store custom tasks hash: %s", exc)

    def _delete_custom_task_by_key(
        self,
        custom_key: str,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        logs = unisdk.get_logs(
            context=self._ctx,
            filter=(
                f"custom_key == '{custom_key}' and "
                f"{managed_rows_filter(managed_by)}"
            ),
            limit=1,
        )
        if not logs:
            return False
        task_id = int(logs[0].entries["task_id"])
        try:
            self._delete_task(task_id=task_id, _root_applied=True)
        except RuntimeError:
            logger.warning(
                "Skipping delete for active custom task key=%s task_id=%s",
                custom_key,
                task_id,
            )
            return False
        return True

    def _insert_custom_task(
        self,
        data: Dict[str, Any],
        *,
        function_name_to_id: Dict[str, int],
    ) -> int:
        payload = dict(data)
        custom_key = payload.pop("custom_key")
        custom_hash = payload.pop("custom_hash")
        managed_by = payload.pop("managed_by", MANAGED_BY_DEPLOYMENT)
        # Destination routing is owned by sync_custom_tasks / sync_custom: the
        # caller has already scoped self._ctx to the target root. Per-row
        # destination is grouping metadata only (same as guidance/contacts).
        payload.pop("destination", None)
        entrypoint_function = payload.pop("entrypoint_function", None)
        schedule = payload.pop("schedule", None)
        trigger = payload.pop("trigger", None)
        deadline = payload.pop("deadline", None)
        max_runtime_seconds = payload.pop("max_runtime_seconds", None)
        repeat = payload.pop("repeat", None)
        priority = payload.pop("priority", Priority.normal)
        tags = payload.pop("tags", None)
        response_policy = payload.pop("response_policy", None)
        offline = bool(payload.pop("offline", False))
        requires_filesystem, requires_computer = resolve_task_resource_requirements(
            {
                "requires_filesystem": payload.pop("requires_filesystem", False),
                "requires_computer": payload.pop("requires_computer", False),
            },
        )
        name = payload.pop("name")
        description = payload.pop("description")
        require_consumed(payload, kind="tasks", custom_key=custom_key)

        entrypoint = None
        if entrypoint_function:
            entrypoint = function_name_to_id.get(entrypoint_function)
            if entrypoint is None:
                logger.warning(
                    "Could not resolve entrypoint_function=%s for task key=%s",
                    entrypoint_function,
                    custom_key,
                )

        current_destination = self._destination_from_task_context(self._ctx)
        destination_arg = (
            None
            if current_destination in (None, PERSONAL_DESTINATION)
            else current_destination
        )
        sync_identity: Dict[str, Any] = {
            "custom_key": custom_key,
            "custom_hash": custom_hash,
            "managed_by": managed_by,
        }
        if tags is not None:
            sync_identity["tags"] = tags
        result = self._create_task(
            name=name,
            description=description,
            schedule=schedule,
            trigger=trigger,
            deadline=deadline,
            max_runtime_seconds=max_runtime_seconds,
            repeat=repeat,
            priority=priority,
            response_policy=response_policy,
            entrypoint=entrypoint,
            offline=offline,
            requires_filesystem=requires_filesystem,
            requires_computer=requires_computer,
            enabled=False,
            destination=destination_arg,
            _root_applied=True,
            _sync_identity=sync_identity,
        )
        return int(result["details"]["task_id"])

    def _update_custom_task(
        self,
        *,
        task_id: int,
        data: Dict[str, Any],
        function_name_to_id: Dict[str, int],
        detach_entrypoint: bool = False,
    ) -> None:
        payload = dict(data)
        custom_key = payload.pop("custom_key")
        custom_hash = payload.pop("custom_hash")
        managed_by = payload.pop("managed_by", MANAGED_BY_DEPLOYMENT)
        payload.pop("destination", None)
        entrypoint_function = payload.pop("entrypoint_function", None)
        schedule = payload.pop("schedule", None)
        trigger = payload.pop("trigger", None)
        deadline = payload.pop("deadline", None)
        max_runtime_seconds = payload.pop("max_runtime_seconds", None)
        repeat = payload.pop("repeat", None)
        priority = payload.pop("priority", None)
        tags = payload.pop("tags", None)
        response_policy = payload.pop("response_policy", None)
        offline = payload.pop("offline", None)
        requires_filesystem, requires_computer = resolve_task_resource_requirements(
            {
                "requires_filesystem": payload.pop("requires_filesystem", False),
                "requires_computer": payload.pop("requires_computer", False),
            },
        )
        name = payload.pop("name", None)
        description = payload.pop("description", None)
        require_consumed(payload, kind="tasks", custom_key=custom_key)

        # ``_UNSET`` leaves whatever the row holds, which is what keeps a
        # runtime-attached entrypoint alive across an ordinary bundle update.
        # ``None`` is the deliberate opposite: the caller has decided this
        # row's entrypoint is no longer valid and the task must fall back to
        # the full loop.
        entrypoint: Any = None if detach_entrypoint else _UNSET
        if entrypoint_function is not None:
            if entrypoint_function == "":
                entrypoint = None
            else:
                resolved = function_name_to_id.get(entrypoint_function)
                if resolved is None:
                    logger.warning(
                        "Could not resolve entrypoint_function=%s for task key=%s",
                        entrypoint_function,
                        custom_key,
                    )
                else:
                    entrypoint = resolved

        self._ensure_not_active_task(task_id)

        if schedule is not None and trigger is not None:
            raise ValueError("A task cannot have both a schedule and a trigger.")

        self._validate_scheduled_invariants(
            schedule=schedule,
            trigger=trigger,
            err_prefix=f"While updating custom task {task_id}:",
        )

        entries: Dict[str, Any] = {
            "custom_key": custom_key,
            "custom_hash": custom_hash,
            "managed_by": managed_by,
            "name": name,
            "description": description,
            "schedule": schedule,
            "trigger": trigger,
            "deadline": deadline,
            "max_runtime_seconds": max_runtime_seconds,
            # None clears: removing every tag from the source must untag the row.
            "tags": tags,
            "response_policy": response_policy,
            "offline": bool(offline) if offline is not None else None,
            "requires_filesystem": requires_filesystem,
            "requires_computer": requires_computer,
        }
        if repeat is not None:
            normalized_repeat = normalize_repeat_patterns(
                [
                    RepeatPattern(**item) if isinstance(item, dict) else item
                    for item in repeat
                ],
            )
            entries["repeat"] = [
                (
                    item.model_dump(mode="json")
                    if isinstance(item, RepeatPattern)
                    else item
                )
                for item in normalized_repeat or []
            ]
        if priority is not None:
            entries["priority"] = (
                priority if isinstance(priority, Priority) else Priority(str(priority))
            )
        if entrypoint is not _UNSET:
            entries["entrypoint"] = entrypoint

        entries = {key: value for key, value in entries.items() if value is not _UNSET}

        sync_meta = {
            key: entries[key]
            for key in ("custom_key", "custom_hash", "managed_by")
            if key in entries
        }
        provider_entries = {
            key: value for key, value in entries.items() if key not in sync_meta
        }

        task = self._get_task_or_raise(task_id)
        log_ids = self._store.get_rows(
            filter=f"task_id == {task_id}",
            return_ids_only=True,
        )
        if self._task_has_provider_event_trigger(task) and provider_entries:
            self._write_provider_event_task_update(
                task_id=task_id,
                task=task,
                entries=provider_entries,
            )
        elif provider_entries:
            self._write_log_entries(logs=log_ids, entries=provider_entries)
        if sync_meta:
            self._write_log_entries(logs=log_ids, entries=sync_meta)

    def _custom_task_sync_workers(self) -> int:
        raw = (os.environ.get("UNIFY_DEPLOY_TASK_SYNC_WORKERS") or "8").strip()
        try:
            workers = int(raw)
        except ValueError:
            workers = 8
        return max(1, min(workers, 32))

    def sync_custom_tasks(
        self,
        *,
        source_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
        function_name_to_id: Optional[Dict[str, int]] = None,
        destination: str | None = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Ensure custom task rows match source ``tasks.jsonl`` definitions.

        Reconciles only the rows *managed_by* owns; rows planted in the same
        context by other sources are neither read nor pruned.
        """
        try:
            tasks_context, meta_context, _is_personal = self._sync_destination_contexts(
                destination,
            )
        except ToolErrorException as exc:
            logger.warning(
                "Skipping custom tasks sync for destination %r: %s",
                destination,
                exc.payload,
            )
            return False

        env_owner = (os.environ.get("OWNER_TEAM_ID") or "").strip()
        if SESSION_DETAILS.team_owned:
            owner_team_id = SESSION_DETAILS.owner_team_id
            expected_prefix = f"Teams/{owner_team_id}/"
            if not str(tasks_context).startswith(expected_prefix):
                raise RuntimeError(
                    "Refusing custom-tasks sync onto "
                    f"{tasks_context!r} for team-owned assistant; expected under "
                    f"Teams/{owner_team_id}/Tasks "
                    "(destination 'personal' means the owning-team home).",
                )
        elif env_owner:
            raise RuntimeError(
                "OWNER_TEAM_ID is set but SESSION_DETAILS.owner_team_id is missing; "
                "refusing custom-tasks sync until team ownership is bound.",
            )

        previous_context = self._ctx
        previous_store = self._store
        previous_active_root = self._active_task_root_context
        self._ctx = tasks_context
        self._store = self._store_for_task_context(tasks_context)
        self._active_task_root_context = tasks_context
        try:
            with (
                exclusive_sync_lease(f"{meta_context}:custom_sync"),
                self._temporary_tasks_meta_context(meta_context),
            ):
                source_tasks = source_tasks or {}
                synced_key = (tasks_context, managed_by)

                name_to_id = function_name_to_id or {}
                return run_custom_sync(
                    adapter=_TaskSyncAdapter(
                        self,
                        function_name_to_id=name_to_id,
                        managed_by=managed_by,
                    ),
                    source=source_tasks,
                    expected_hash=compute_custom_tasks_hash(
                        source_tasks=source_tasks,
                        entrypoint_resolution={
                            name: name_to_id.get(name)
                            for entry in source_tasks.values()
                            if (name := entry.get("entrypoint_function"))
                        },
                    ),
                    stored_hash=self._get_stored_custom_tasks_hash(managed_by),
                    already_synced=synced_key in self._custom_tasks_synced_sources,
                    mark_synced=lambda: self._custom_tasks_synced_sources.add(
                        synced_key,
                    ),
                    store_hash=lambda value: self._store_custom_tasks_hash(
                        value,
                        managed_by=managed_by,
                    ),
                )
        finally:
            self._ctx = previous_context
            self._store = previous_store
            self._active_task_root_context = previous_active_root

    def sync_custom(
        self,
        *,
        source_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
        function_name_to_id: Optional[Dict[str, int]] = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Sync custom tasks from pre-collected sources across destinations."""
        if source_tasks is None:
            source_tasks = {}

        by_destination: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for custom_key, source_data in source_tasks.items():
            destination = source_data.get("destination") or "personal"
            by_destination.setdefault(destination, {})[custom_key] = source_data

        changed = False
        for destination, group in by_destination.items():
            destination_arg = None if destination == "personal" else destination
            changed |= self.sync_custom_tasks(
                source_tasks=group,
                function_name_to_id=function_name_to_id,
                destination=destination_arg,
                managed_by=managed_by,
            )
        return changed


class _TaskSyncAdapter(CustomSyncAdapter):
    """Storage mechanics for the custom tasks reconcile.

    Updates run in parallel across independent keys; the engine
    serializes collision probes and inserts under one lock. An update is
    vetoed (and retried next reconcile, because the aggregate hash stays
    unstored) while the task has an execution genuinely running.
    """

    kind = "tasks"

    def __init__(
        self,
        scheduler: TaskScheduler,
        *,
        function_name_to_id: Dict[str, int],
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        self._scheduler = scheduler
        self._function_name_to_id = function_name_to_id
        self.managed_by = managed_by
        self.max_workers = scheduler._custom_task_sync_workers()
        self._resolved: Dict[int, bool] = {}

    def _entrypoint_resolves(self, function_id: int) -> bool:
        """Whether a stored entrypoint id still points at a function.

        Memoized for the pass: several tasks can share one distilled
        entrypoint, and the answer cannot change while a reconcile holds the
        sync lease.
        """

        from ..function_manager.function_manager import function_id_resolves

        if function_id not in self._resolved:
            self._resolved[function_id] = function_id_resolves(function_id)
        return self._resolved[function_id]

    def _detach_runtime_entrypoint(
        self,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> bool:
        """Whether this row's entrypoint was attached at runtime and must go.

        A task can acquire an ``entrypoint`` its source never declared: the
        post-run review distils a successful trajectory into a function and
        attaches it, which is how a recurring task gets cheaper and more
        deterministic every time it runs. That is worth having, and it is
        deliberately not promoted into the bundle -- a bundle is universal to
        everyone who installs it, while a distillation is the product of one
        installation's own trajectory and data.

        So the bundle stays the base and the distillation is a local overlay,
        and reconcile has to keep both. It does that by detaching in exactly
        two cases, each of which makes the overlay wrong rather than merely
        old:

        * **It dangles.** The function was deleted or renumbered, so the task
          now points at nothing and every future run fails on it.
        * **The base moved underneath it.** A distillation encodes the task
          description it was derived from. When an update rewrites that
          description, keeping the overlay runs yesterday's logic under
          today's instructions -- a confident wrong answer, which is worse
          than the slow path.

        Detaching is always safe, and that asymmetry is what makes this
        self-healing rather than merely defensive: the task ran correctly
        through the full actor loop before any entrypoint existed, so dropping
        one degrades to correct-and-slower, never to broken. A later run
        re-derives against the new description if it is still worth doing.
        """

        if fields.get("entrypoint_function"):
            return False
        stored = live_row.get("entrypoint")
        if stored is None:
            return False
        if not self._entrypoint_resolves(int(stored)):
            return True
        return self._distillation_base_moved(live_row, fields)

    @staticmethod
    def _distillation_base_moved(
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> bool:
        """Whether the source changed in a way a distillation depends on.

        Not the whole content hash. That covers every synced field -- name,
        priority, tags, schedule, repeat -- so a cosmetic edit read as "the
        base moved" and threw away a working distillation, forcing a full
        re-derivation on the next run. Stamping a timezone onto a schedule
        did exactly that once.

        A distillation is derived from what the task asked for, so only the
        fields that state the request can invalidate it: the description it
        was distilled against, and the response policy that governs what the
        result has to be. Everything else changes when the task runs, not
        what it does.
        """

        for field in ("description", "response_policy"):
            if str(live_row.get(field) or "") != str(fields.get(field) or ""):
                return True
        return False

    def live_rows(self) -> List[Dict[str, Any]]:
        logs = unisdk.get_logs(
            context=self._scheduler._ctx,
            filter=managed_rows_filter(self.managed_by),
            limit=1000,
            exclude_fields=list_private_fields(self._scheduler._ctx),
        )
        return [dict(lg.entries or {}) for lg in logs]

    def should_update(
        self,
        key: str,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> bool:
        task_id = int(live_row["task_id"])
        return (
            self._scheduler._running_execution(
                task_id,
                states=(ExecutionState.running,),
            )
            is None
        )

    def derived_stale(
        self,
        key: str,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> bool:
        """The stored ``entrypoint`` is a function id resolved from
        ``entrypoint_function`` at write time. When the functions store is
        re-registered under new ids, the row dangles while its content
        hash — which covers the name, never the id — still matches.

        An entrypoint attached at runtime dangles the same way and used to be
        unrepairable by construction: this returned ``False`` on its first
        line whenever the *source* named no ``entrypoint_function``, which is
        precisely the case the runtime creates. The check the method exists to
        make was skipped for the one drift no other detector can see.
        """
        entrypoint_function = fields.get("entrypoint_function")
        if not entrypoint_function:
            return self._detach_runtime_entrypoint(live_row, fields)
        resolved = self._function_name_to_id.get(entrypoint_function)
        if resolved is None:
            return False
        stored = live_row.get("entrypoint")
        return stored is None or int(stored) != int(resolved)

    def insert(self, key: str, fields: Dict[str, Any]) -> None:
        self._scheduler._insert_custom_task(
            fields,
            function_name_to_id=self._function_name_to_id,
        )

    def update(
        self,
        key: str,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> None:
        self._scheduler._update_custom_task(
            task_id=int(live_row["task_id"]),
            data=fields,
            function_name_to_id=self._function_name_to_id,
            detach_entrypoint=self._detach_runtime_entrypoint(live_row, fields),
        )

    def delete(self, key: str, live_row: Dict[str, Any]) -> None:
        self._scheduler._delete_custom_task_by_key(key, managed_by=self.managed_by)

    def find_released(
        self,
        key: str,
        fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """The row this key planted, once the user has taken it over.

        A planted task is the user's to edit — the first authored edit
        clears ``managed_by`` — and after that the source must leave it
        alone. Without this probe the key reads as missing, because
        ``live_rows`` filters on ``managed_by``, and the pass plants a
        second copy beside the edited one.
        """
        existing = unisdk.get_logs(
            context=self._scheduler._ctx,
            filter=released_rows_filter(key),
            limit=1,
        )
        if not existing:
            return None
        return dict(existing[0].entries or {})

    def find_collision(
        self,
        key: str,
        fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        existing = unisdk.get_logs(
            context=self._scheduler._ctx,
            filter=f"custom_key == '{key}'",
            limit=1,
        )
        if not existing:
            return None
        return dict(existing[0].entries or {})

    def remove_collision(self, key: str, live_row: Dict[str, Any]) -> None:
        self._scheduler._delete_task(
            task_id=int(live_row["task_id"]),
            _root_applied=True,
        )
