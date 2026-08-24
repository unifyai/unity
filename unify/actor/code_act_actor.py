import asyncio
import contextvars
import copy
import functools
import inspect
import json
import re
import sys
import traceback
import uuid
import weakref
from secrets import token_hex as _token_hex
import logging
from typing import (
    Annotated,
    Any,
    Callable,
    Awaitable,
    Dict,
    NamedTuple,
    Optional,
    Type,
    Union,
    TYPE_CHECKING,
)
from pydantic import BaseModel

from unify.actor.base import BaseCodeActActor
from unify.common.context_dump import make_messages_safe_for_context_dump
from unify.actor.execution import (
    ExecutionResult,
    PackageOverlay,
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    _CURRENT_ENVIRONMENTS,
    _CURRENT_PACKAGE_OVERLAY,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _validate_execution_params,
)
from unify.common.async_tool_loop import (
    AsyncToolLoopHandle,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unify.common.task_execution_context import (
    PostRunReviewContext,
    current_post_run_review_context,
    TaskExecutionDelegate,
    current_task_execution_delegate,
)
from unify.events.event_bus import EVENT_BUS, Event
from unify.common.llm_client import new_llm_client
from unify.common.llm_meter import RunMeter, current_run_meter, new_run_meter
from unify.common.act_llm_profiles import (
    CURRENT_ACT_LLM_PROFILE,
    resolve_act_llm_profile,
)
from unify.common.llm_helpers import methods_to_tool_dict
from unify.common.tool_spec import ToolSpec, llm_soft_required
from unify.function_manager.base import BaseFunctionManager
from unify.function_manager.function_manager import strip_ledger_internals
from unify.function_manager.primitives import ComputerPrimitives
from unify.actor.prompt_builders import build_code_act_prompt
from unify.actor.verification_runtime import (
    EntrypointOutcome,
    Frame,
    HeldOutcome,
    RepairRefused,
    RewindRequested,
    VerifierPasses,
    closure_rows,
    install_wrappers,
    rederive_trust,
    run_probe,
    run_verified_entrypoint,
)
from unify.events.manager_event_logging import log_manager_call
from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX
from unify.common.hierarchical_logger import log_boundary_event
from unify.events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
)
from unify.events.active_work import ACTIVE_WORK, ActiveWorkHandle
from unify.integrations.approval import build_pending_approval_payload
from unify.integrations.function_metadata import is_provider_backed_function

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager
    from unify.knowledge_manager.knowledge_manager import KnowledgeManager
    from unify.workflow_manager.workflow_manager import WorkflowManager


# ---------------------------------------------------------------------------
# Tool-policy type alias and sentinel
# ---------------------------------------------------------------------------

ToolPolicyFn = Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]]
"""Signature for a tool-policy callback.

Receives ``(step_index, tools_dict)`` and returns ``(tool_choice_mode,
filtered_tools_dict)`` where *tool_choice_mode* is ``"auto"`` or
``"required"``.  An optional third dict ``{"eager": True}`` may be returned
to request immediate follow-up LLM turns while the policy remains eager
(see the async tool loop ``tool_policy`` docs).
"""

_USE_DEFAULT: object = object()
"""Sentinel indicating 'use the built-in discovery-first tool policy'."""

# Tools visible while discovery-first gates are still open. Write/mutate tools
# stay hidden until every present library family has been touched once.
_DISCOVERY_GATE_TOOLS: frozenset[str] = frozenset(
    {
        "FunctionManager_search_functions",
        "FunctionManager_filter_functions",
        "FunctionManager_list_functions",
        "GuidanceManager_search",
        "GuidanceManager_filter",
        "GuidanceManager_get_guidance",
        "KnowledgeManager_search",
        "KnowledgeManager_filter",
        "KnowledgeManager_get_knowledge",
    },
)

# Prefer one semantic-search discovery tool per family while the gate is open
# so hard tool_choice=required + eager follow-up turns map onto a small set.
_DISCOVERY_PREFERRED_TOOLS: dict[str, str] = {
    "FunctionManager_": "FunctionManager_search_functions",
    "GuidanceManager_": "GuidanceManager_search",
    "KnowledgeManager_": "KnowledgeManager_search",
}

_UNSET: object = object()
"""Sentinel indicating 'parameter was not explicitly provided'."""


class _ActiveWorkNotificationQueue:
    def __init__(
        self,
        target: asyncio.Queue[dict],
        active_work: ActiveWorkHandle,
    ) -> None:
        self._target = target
        self._active_work = active_work

    async def put(self, item: dict) -> None:
        self._active_work.record_user_notification()
        await self._target.put(item)

    def put_nowait(self, item: dict) -> None:
        self._active_work.record_user_notification()
        self._target.put_nowait(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _resolve_param(explicit: object, code_value: object, default: object) -> object:
    """Three-tier resolution: explicit constructor arg > code config > hardcoded default."""
    if explicit is not _UNSET:
        return explicit
    if code_value is not None:
        return code_value
    return default


def _discovery_tools_for_prefix(
    filtered: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """Return the preferred discovery tool for *prefix*, with family fallback."""
    family = {
        k: v
        for k, v in filtered.items()
        if k in _DISCOVERY_GATE_TOOLS and k.startswith(prefix)
    }
    preferred = _DISCOVERY_PREFERRED_TOOLS.get(prefix)
    if preferred is not None and preferred in family:
        return {preferred: family[preferred]}
    return family


_DISCOVERY_PREFERRED_ARGS: dict[str, dict[str, Any]] = {
    "FunctionManager_search_functions": {"query": "relevant functions", "n": 5},
    "GuidanceManager_search": {"query": "relevant guidance", "n": 5},
    "KnowledgeManager_search": {"query": "relevant knowledge", "n": 5},
}


def _tool_names_from_openai_tools(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _is_discovery_gate_schema(tool_names: list[str]) -> bool:
    """True when the visible schema is only discovery-read tools (+ loop extras)."""
    if not tool_names:
        return False
    names = set(tool_names)
    extras = {"compress_context"}
    core = {n for n in names if n not in extras and not n.startswith("check_status_")}
    if not core or not core.issubset(_DISCOVERY_GATE_TOOLS):
        return False
    families = sum(
        1
        for prefix in _DISCOVERY_PREFERRED_TOOLS
        if any(n.startswith(prefix) for n in core)
    )
    return families >= 2


def _discovery_preferred_for_schema(tool_names: list[str]) -> list[tuple[str, dict]]:
    """Return [(tool_name, args), ...] for each family present in *tool_names*."""
    preferred_calls: list[tuple[str, dict]] = []
    for prefix, preferred in _DISCOVERY_PREFERRED_TOOLS.items():
        family = [n for n in tool_names if n.startswith(prefix)]
        if not family:
            continue
        tool_name = preferred if preferred in family else family[0]
        args = dict(_DISCOVERY_PREFERRED_ARGS.get(tool_name, {}))
        preferred_calls.append((tool_name, args))
    return preferred_calls


def _build_discovery_parallel_mutator() -> Any:
    """Complete partial discovery-gate turns with missing family tool calls.

    Hard OpenRouter hosts still sometimes serialize discovery families under
    ``tool_choice="required"`` even with ``parallel_tool_calls=True``. This
    Unify-local mutator appends the missing preferred discovery calls so the
    first tool-calling turn covers every present family in parallel.
    """
    from unillm.clients.completion_mutator import CompletionMutatorContext

    def _mutator(completion: Any, context: CompletionMutatorContext) -> Any:
        if context.original_tool_choice != "required":
            return completion
        tool_names = _tool_names_from_openai_tools(context.request_kw.get("tools"))
        if not _is_discovery_gate_schema(tool_names):
            return completion

        msg = completion.choices[0].message
        existing = list(msg.tool_calls or [])
        if not existing:
            return completion

        called_names: list[str] = []
        for tc in existing:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
            if isinstance(name, str) and name:
                called_names.append(name)

        missing: list[tuple[str, dict]] = []
        for tool_name, args in _discovery_preferred_for_schema(tool_names):
            prefix = next(
                (p for p in _DISCOVERY_PREFERRED_TOOLS if tool_name.startswith(p)),
                None,
            )
            if prefix is None:
                continue
            if any(n.startswith(prefix) for n in called_names):
                continue
            missing.append((tool_name, args))
        if not missing:
            return completion

        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        for index, (tool_name, args) in enumerate(missing):
            existing.append(
                ChatCompletionMessageToolCall(
                    id=f"call_discovery_{index}",
                    type="function",
                    function=Function(
                        name=tool_name,
                        arguments=json.dumps(args),
                    ),
                ).model_dump(warnings=False),
            )
        msg.tool_calls = existing
        msg.content = None
        completion.choices[0].finish_reason = "tool_calls"
        return completion

    return _mutator


def _default_tool_policy(
    has_fm_tools: bool,
    has_gm_tools: bool,
    filter_tools: Callable[[Dict[str, Any]], Dict[str, Any]],
    has_km_tools: bool = False,
) -> ToolPolicyFn:
    """Build the default *discovery-first* tool policy.

    Until each present gate among ``FunctionManager_*``, ``GuidanceManager_*``,
    and ``KnowledgeManager_*`` has been called at least once, the LLM is
    restricted to only those families' discovery/read tools (with
    ``tool_choice="required"``). Write tools and non-library tools such as
    ``execute_code`` stay hidden. Once all present gates are satisfied the
    full (statically-filtered) tool set is returned with ``"auto"`` mode.

    While gates remain open the policy also sets ``eager=True``, so the async
    tool loop grants another LLM turn immediately after each partial discovery
    call is scheduled (without waiting for that call's result).  That way a
    model that only fires one of the required discovery tools on the first
    turn is prompted for the missing family right away, overlapping the
    in-flight search.

    When only a subset of the manager tool families is present, those families
    act as the gates.  When none are present the policy is a no-op pass-through.

    Parameters
    ----------
    has_fm_tools:
        Whether the base tool set contains any ``FunctionManager_*`` tools.
    has_gm_tools:
        Whether the base tool set contains any ``GuidanceManager_*`` tools.
    filter_tools:
        The static-filter callable (``_filter_tools``) that enforces
        ``can_compose`` / ``can_store`` / ``can_spawn_sub_agents``.
    has_km_tools:
        Whether the base tool set contains any ``KnowledgeManager_*`` tools.
    """

    def _policy(
        step: int,
        tools: Dict[str, Any],
        called_tools: list[str],
    ) -> tuple[str, Dict[str, Any]] | tuple[str, Dict[str, Any], dict]:
        filtered = filter_tools(tools)

        fm_satisfied = (not has_fm_tools) or any(
            t.startswith("FunctionManager_") for t in called_tools
        )
        gm_satisfied = (not has_gm_tools) or any(
            t.startswith("GuidanceManager_") for t in called_tools
        )
        km_satisfied = (not has_km_tools) or any(
            t.startswith("KnowledgeManager_") for t in called_tools
        )

        if fm_satisfied and gm_satisfied and km_satisfied:
            return "auto", filtered

        # Expose one preferred discovery tool per unsatisfied gate family.
        gated: Dict[str, Any] = {}
        if not fm_satisfied:
            gated.update(_discovery_tools_for_prefix(filtered, "FunctionManager_"))
        if not gm_satisfied:
            gated.update(_discovery_tools_for_prefix(filtered, "GuidanceManager_"))
        if not km_satisfied:
            gated.update(_discovery_tools_for_prefix(filtered, "KnowledgeManager_"))

        if gated:
            return "required", gated, {"eager": True}
        return "auto", filtered

    return _policy


# ---------------------------------------------------------------------------
# Resolved session tuple returned by _resolve_session
# ---------------------------------------------------------------------------


class _ResolvedSession(NamedTuple):
    language: str
    venv_id: Optional[int]
    session_id: Optional[int]


# ---------------------------------------------------------------------------
# Agent context for tracking execution depth and providing handle access
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field as dataclass_field


@dataclass
class AgentContext:
    """Runtime context for agent execution, accessible via get_current_agent_context().

    Attributes:
        depth: Nesting level (0 = root agent, 1 = first subagent, etc.)
        agent_id: Unique identifier for this agent run
        handle: Reference to the AsyncToolLoopHandle (for accessing history, etc.)
    """

    depth: int = 0
    agent_id: str = dataclass_field(default_factory=lambda: str(uuid.uuid4()))
    handle: "AsyncToolLoopHandle | None" = None
    proactive_storage_summaries: list[str] = dataclass_field(default_factory=list)


_CURRENT_AGENT_CONTEXT: contextvars.ContextVar[AgentContext] = contextvars.ContextVar(
    "code_act_agent_context",
    default=AgentContext(),
)


def get_current_agent_context() -> AgentContext:
    """Get the current agent execution context.

    Use this inside service methods to:
    - Check agent depth and prevent infinite recursion
    - Access the current agent's handle for message history, etc.

    Returns:
        AgentContext with depth, agent_id, and handle

    Example:
        ctx = get_current_agent_context()
        if ctx.depth >= 2:
            raise RuntimeError("Max depth exceeded")
        if ctx.handle:
            history = ctx.handle.get_history()
    """
    return _CURRENT_AGENT_CONTEXT.get()


logger = logging.getLogger(__name__)


class _CodeActEntrypointHandle(SteerableToolHandle):  # type: ignore[abstract-method]
    """Execute a FunctionManager entrypoint function without invoking the CodeAct LLM loop.

    TaskScheduler delegates task execution to an actor via:
    `primitives.actor.act(task_description, entrypoint=<function_id>, persist=False)`.

    When an `entrypoint` is provided, CodeActActor resolves the function by id,
    injects it into the sandbox namespace, and executes it in an asyncio task.
    """

    def __init__(
        self,
        *,
        entrypoint_id: int,
        execution_task: asyncio.Task[Any],
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
        meter: Optional[RunMeter] = None,
    ) -> None:
        self._entrypoint_id = int(entrypoint_id)
        self._execution_task = execution_task
        self._meter = meter
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._error: Optional[BaseException] = None
        self._stopped = False
        self._on_finally = on_finally
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        self._follow_up: Optional[asyncio.Task[Any]] = None
        # Set when the run finished without performing an effect because a
        # verdict it depended on failed, timed out or could not be settled.
        self.held_outcome: Optional[HeldOutcome] = None
        # Verification accounting for the execution row.
        self.run_stats: dict[str, Any] = {}

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self) -> None:
        try:
            out = await self._execution_task
            if isinstance(out, EntrypointOutcome):
                self.held_outcome = out.held
                self.run_stats = {
                    "verdicts": dict(out.verdict_counts),
                    "rewinds": int(out.rewinds),
                    "verifier_tasks": int(out.verifier_tasks),
                    "held_reason": (
                        f"{out.held.code}: {out.held.reason}" if out.held else None
                    ),
                    "tokens": (
                        self._meter.snapshot()["tokens"] if self._meter else None
                    ),
                }
                self._follow_up = out.follow_up
                if not self._stopped:
                    if out.held is not None:
                        self._result_str = out.held.message
                    else:
                        self._result_str = (
                            str(out.result) if out.result is not None else ""
                        )
            elif not self._stopped:
                self._result_str = str(out) if out is not None else ""
        except asyncio.CancelledError:
            self._stopped = True
            if self._result_str is None:
                self._result_str = f"Entrypoint {self._entrypoint_id} was cancelled."
        except Exception as e:
            self._error = e
            self._result_str = f"Error: {e}"
        finally:
            self._completion_event.set()
            if self._follow_up is not None:
                try:
                    await self._follow_up
                except Exception:
                    pass
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception:
                    pass
            self._notification_q.put_nowait({})

    async def push_notification(self, message: str) -> None:
        """Deliver an owner-facing follow-up (a correction after early delivery)."""
        await self._notification_q.put(
            {"type": "notification", "message": message, "completed": True},
        )

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        status = "completed" if self.done() else "still running"
        client = new_llm_client(purpose="planning", origin="EntrypointHandle.ask")
        client.set_system_message(
            "You are an AI assistant answering a status question about an in-flight entrypoint execution. "
            "Be brief and factual.",
        )
        msg = (
            f"Entrypoint {self._entrypoint_id} status: {status}.\n\n"
            f"User question: {question}"
        )
        return start_async_tool_loop(
            client=client,
            message=msg,
            tools={},
            loop_id=f"EntrypointQuestion({self._entrypoint_id})",
            max_consecutive_failures=1,
        )

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        # No-op for non-LLM entrypoint execution.
        pass

    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        if self._completion_event.is_set():
            return
        self._stopped = True
        self._result_str = (
            f"Entrypoint {self._entrypoint_id} stopped."
            if not reason
            else f"Entrypoint {self._entrypoint_id} stopped: {reason}"
        )
        self._execution_task.cancel()
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    async def pause(self) -> Optional[str]:
        return None

    async def resume(self) -> Optional[str]:
        return None

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._completion_event.wait()
        if self._error is not None:
            raise self._error
        return self._result_str or ""

    async def next_clarification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class _CodeActTaskExecutionDelegate:
    """Route durable task execution through the CodeActActor run that requested it."""

    def __init__(self, actor: "CodeActActor") -> None:
        self._actor = actor

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None,
        parent_chat_context: list[dict] | None,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
        images: Any | None = None,
        **kwargs: Any,
    ) -> SteerableToolHandle:
        """Start one task run using this actor's CodeAct execution machinery."""

        _ = images
        task_guidelines = kwargs.pop("guidelines", None)
        entrypoint_kwargs = kwargs.pop("entrypoint_kwargs", None)
        entrypoint_repair_context = kwargs.pop("entrypoint_repair_context", None)
        destination = kwargs.pop("destination", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                "TaskExecutionDelegate.start_task_run got unexpected "
                f"keyword arguments: {unexpected}",
            )
        return await self._actor.act(
            task_description,
            guidelines=task_guidelines,
            _parent_chat_context=parent_chat_context,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
            entrypoint=entrypoint,
            entrypoint_kwargs=entrypoint_kwargs,
            entrypoint_repair_context=entrypoint_repair_context,
            destination=destination,
            persist=False,
            _reuse_actor_slot=entrypoint is not None,
        )


# ---------------------------------------------------------------------------
# Verification tools shared by the storage and repair loops
# ---------------------------------------------------------------------------


def _verification_librarian_tools(fm: Any) -> Dict[str, Callable]:
    """``confirm_side_effect_class`` and ``set_verification_policy`` bound to ``fm``.

    Both shape how much verification a stored function needs; neither can
    grant trust, and the ledger is written only by the runtime.
    """
    if fm is None:
        return {}

    async def confirm_side_effect_class(
        function_id: int,
        side_effect_class: str,
        rationale: str,
    ) -> str:
        return str(
            fm.confirm_side_effect_class(
                function_id=int(function_id),
                side_effect_class=str(side_effect_class),
                rationale=str(rationale),
            ),
        )

    async def set_verification_policy(
        function_id: int,
        always_verify: bool | None = None,
        required_passes: int | None = None,
        min_distinct_inputs: int | None = None,
        fixture_only: bool | None = None,
        spot_check_rate: float | None = None,
    ) -> str:
        return str(
            fm.set_verification_policy(
                function_id=int(function_id),
                always_verify=always_verify,
                required_passes=required_passes,
                min_distinct_inputs=min_distinct_inputs,
                fixture_only=fixture_only,
                spot_check_rate=spot_check_rate,
            ),
        )

    confirm_side_effect_class.__doc__ = fm.confirm_side_effect_class.__doc__
    set_verification_policy.__doc__ = fm.set_verification_policy.__doc__
    return {
        "confirm_side_effect_class": confirm_side_effect_class,
        "set_verification_policy": set_verification_policy,
    }


# ---------------------------------------------------------------------------
# Shared storage-review prompt sections
# ---------------------------------------------------------------------------

_DEFAULT_STORAGE_REVIEW_LABEL = "Storing reusable skills"
_DEFAULT_STORAGE_REVIEW_INSTRUCTIONS = (
    "Review the trajectory and store any reusable functions, "
    "compositional guidance, and durable knowledge claims."
)


def _signature_compatible_kwargs(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return only the scheduler-supplied kwargs accepted by a callable."""

    signature = inspect.signature(fn)
    parameters = signature.parameters
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return dict(kwargs)
    accepted = {
        name
        for name, param in parameters.items()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


_STORAGE_WHAT_CAN_BE_STORED = (
    "## What Can Be Stored\n\n"
    "Any code that executed successfully in `execute_code` during "
    "this trajectory can be stored as a function. Environment-provided "
    "namespaces (`primitives`, `primitives.computer`, `primitives.actor`) and "
    "other stored functions referenced in the code are automatically "
    "detected from the source and injected at runtime — you do not "
    "need to add imports or worry about whether these names will be "
    "available when the function runs later. Focus on whether a "
    "pattern is *worth* reusing, not whether it is *technically "
    "executable* in isolation.\n\n"
    "### Bake configuration into reusable callables\n\n"
    "Stored functions should NOT be verbatim copies of code blocks "
    "from the trajectory. During execution, the agent discovered the "
    "right combination of parameters, tool selections, and strategies "
    "through reasoning — that configuration knowledge is the valuable "
    "part. A stored function should **bake in** the hard-won "
    "configuration as fixed values and **expose** only the parts "
    "that genuinely vary between uses (typically the task-specific "
    "input). This produces a function that future callers can use "
    "without rediscovering the right setup.\n\n"
    "The same applies to trajectories that unrolled as "
    "`execute_code` -> observe -> agent reasoning -> `execute_code` loops. "
    "Do not assume the future procedure needs a full CodeActActor. First "
    "ask whether the agent's intermediate reasoning was open-ended "
    "planning or bounded semantic judgment inside an otherwise stable "
    "control flow. If it was bounded, distill the trajectory into one "
    "function: deterministic Python for control flow, managed primitives "
    "for side effects, and focused `query_llm(...)` calls with "
    "structured outputs, low temperature, and an explicit model for "
    "classification, summarization, drafting, ranking, or source "
    "selection. Leave it live-agent or guidance-driven when the reasoning "
    "involved changing tool discovery, unknown-state debugging, user "
    "clarification, or broad strategy selection.\n\n"
    "Semantic downgrades are bugs. When a live trajectory interpreted or "
    "produced unstructured data — classification, extraction, routing, "
    "summarization, drafting, rewriting, personalization, or other "
    "human-facing synthesis — the stored function should preserve that "
    "fuzzy step as `query_llm(...)` with a stable contract. Do not "
    "replace it with keyword ladders, regex classifiers, label-specific "
    "canned prose, or templates inferred from observed examples unless "
    "the user explicitly requested fixed deterministic rules/templates. "
    "Generalize by preserving the LLM call, not by memorizing the sample "
    "cases.\n\n"
    "### The distillation dial\n\n"
    "Skill maturation is a dial, not a switch. Every procedure "
    "decomposes into a skeleton — control flow that is the same on "
    "every run — and joints — the substeps where meaning must be "
    "judged. Each substep sits independently at one notch: plain "
    "deterministic Python, a focused stateless `query_llm(...)` call, "
    "a sub-agent via `primitives.actor.act(...)`, or left to the live "
    "agent. Distilling a trajectory means choosing a notch per "
    "substep, not one mode for the whole task: freeze the skeleton, "
    "and keep every genuine judgment fluid at the cheapest notch that "
    "preserves it.\n\n"
    "Know what each direction costs. Distilling too little is loud "
    "and cheap to fix: the ledger shows planning cost paid again on "
    "every run, forever. Distilling too much is quiet and expensive: "
    "everything stays cheap and correct until the first input the "
    "frozen structure was never validated on — and a frozen function "
    "processes an anomaly as if it were normal, because the fluid "
    "intelligence that would have noticed was distilled away. The "
    "same asymmetry governs placement: freezing a semantic joint into "
    "keyword ladders or templates fails silently on the first "
    "paraphrase, while leaving exact work inside an LLM call merely "
    "wastes tokens. When in doubt, a joint stays semantic — an "
    "unnecessary `query_llm(...)` call costs cents; an unnecessary "
    "regex costs correctness.\n\n"
    "Freeze only observed invariance. The evidence for a notch is the "
    "trajectory itself: structure you watched hold across the run(s) "
    "may be frozen; structure inferred from a single example may not. "
    "A branch the agent reasoned about once is still a judgment, not "
    "yet control flow.\n\n"
    "Preserve the surprise signal. A live agent notices when an input "
    "is strange; a stored function must be given that noticing back "
    "explicitly. State the envelope the structure was validated "
    "inside — type hints, checkable postconditions, precondition "
    "guards on the ranges, shapes, and assumptions the trajectory "
    "actually exhibited — and make the function raise or return early "
    "on inputs outside it, so the calling actor (or a `query_llm(...)` "
    "judgment) handles the anomaly instead of the frozen path "
    "swallowing it.\n\n"
    "Distillation is reversible. Stored functions are inspectable and "
    "revisable, and any change re-enters verification; re-opening one "
    "joint later is routine maintenance, not a failure. Do not "
    "under-distill out of caution — freeze the skeleton the evidence "
    "supports, guard it, and let the ledger and future corrections "
    "move the dial.\n\n"
    "### Model choice is part of distillation\n\n"
    "Every `query_llm(...)` call you bake into a stored function is a "
    "standing model choice. Choose `model=` deliberately per the "
    '"Choosing A Model" section of `help(query_llm)` — bounded, '
    "repeated classification/extraction rarely needs the default "
    "high-reasoning model. When candidates look close, trial them "
    "against the trajectory itself: it already contains the concrete "
    "inputs and known-good outputs for each semantic substep, so "
    "replay those cases through each candidate with the same prompt "
    "and `response_format`, keep the cheapest model that passes, and "
    "record the rationale in the function's docstring. Store the "
    "trial cases as fixtures where the function is pure, so later "
    "model or prompt changes replay them automatically.\n\n"
    "### Preserving user-facing communication points\n\n"
    "When wrapping a procedure into a stored function, pay attention to "
    "points where the original code depended on the user being "
    "informed — especially states that block until the user takes an "
    "external action. Stored functions cannot emit user-facing "
    "notifications mid-execution, so do not bake an indefinite wait on "
    "external human action (approving an auth prompt, granting a "
    "permission, confirming a destructive operation) into the function "
    "body. Instead, have the function return early or raise with a "
    "clear message describing what the user must do, so the calling "
    "actor can inform the user (via the `send_notification` tool "
    "between `execute_code` blocks) and resume afterwards. A silent "
    "in-function wait is a deadlock: the function blocks on a "
    "condition the user does not know about.\n\n"
    "### Durable task executor candidates\n\n"
    "A function intended to become a future TaskScheduler executor must "
    "preserve the observed live execution chain, not merely produce a "
    "plausible answer for the same example. Map each live trajectory step "
    "to the candidate code path that replaces or preserves it. Keep managed "
    "primitives, helper calls, validation gates, side-effect ordering, "
    "retries, cleanup, result shape, and failure semantics unless the "
    "candidate declares and validates an equivalent replacement. A live "
    "thinking step may become `query_llm(...)` only when it has a stable "
    "input/output contract and validation; if it required agentic "
    "exploration, preserve that substep or leave the task "
    "description-driven.\n\n"
    "Executor candidates must not hardcode observations from live tool "
    "results, remove validation gates, reorder dependent side effects, "
    "discard recovery branches, replace managed tools with weaker ad hoc "
    "mechanisms, or replace semantic LLM work with brittle symbolic "
    "approximations.\n\n"
    "### Expressive logging in stored functions\n\n"
    "Soft failures (empty results, skipped branches, degraded fallbacks, "
    "status dicts that report problems without raising) are the common "
    "bug shape. Stored functions must leave a reconstructable trail with "
    "the stdlib `logging` module — not via user-facing notifications. Use "
    "markers `PHASE`, `SKIP`, and `SOFT_FAIL` in the message text so "
    "Job/EventBus captures stay greppable, e.g. "
    "`logging.getLogger(__name__).info('PHASE load_rows count=%s', n)` "
    "or `.warning('SKIP empty_result')` / `.error('SOFT_FAIL partial …')`. "
    "Log every meaningful stage boundary, every intentional skip, and "
    "every soft failure. Do **not** strip PHASE/SKIP/SOFT_FAIL trails when "
    "distilling a live trajectory into a stored function. You may remove "
    "dead exploratory `print`s, duplicated setup, or formatting noise, "
    "but never remove validation gates, recovery branches, or diagnostic "
    "logging that explains why a path returned early or returned empty. "
    "Store non-executor helpers and guidance freely; a stored executor earns "
    "trust from independent verification of its runs, and offline promotion "
    "follows from that trust.\n\n"
    "### Async / event-loop safety in stored functions\n\n"
    "Offline TaskScheduler Jobs already own an event loop via "
    "`asyncio.run`. Nested `asyncio.run(...)` inside a sync helper then "
    "raises `RuntimeError: asyncio.run() cannot be called from a running "
    "event loop`. Prefer `async def` entrypoints / helpers and `await` "
    "end-to-end (including `await query_llm(...)`). When a sync façade is "
    "required, call the injected `run_coro_sync(factory)` helper (also "
    "`from unify.common.asyncio_compat import run_coro_sync`) instead of "
    "nesting `asyncio.run`.\n\n"
    "### Verifiable functions\n\n"
    "A stored function is not trusted when it is stored. Every call of it "
    "runs under independent verification — a static review of its source, "
    "an argument review, a precondition probe before any effect, and a "
    "post-execution review — until enough independent verdicts have "
    "accumulated for its effect class; only then does it run bare. You "
    "cannot grant that trust and you never write the verification ledger. "
    "What you can do is store functions that are cheap to verify and hard "
    "to get wrong:\n\n"
    "- **Thin effects.** A function that performs an irreversible effect "
    "(send, post, delete, pay, drive a desktop) must do only that. Compute "
    "in one function, perform the effect in another, and let the root "
    "compose them. This is what makes a failed verdict cheap to repair and "
    "blame precise: the computation can be re-run and corrected without "
    "the effect ever having happened. Example — instead of one "
    "`send_weekly_summary(week)` that fetches, totals and posts, store "
    "`compute_weekly_summary(week) -> dict` (read-only, verifiable "
    "against its inputs) and `post_summary(channel: str, text: str) -> "
    "dict` (the one effect), with a root that calls the first, then the "
    "second.\n"
    "- **Type hints on every parameter and the return.** The input and "
    "output contracts a call is checked against are derived from them; an "
    "unhinted function has no deterministic contract.\n"
    "- **A docstring whose first sentence is a checkable postcondition** "
    '("Return the sum of `amount` over the rows, in minor units, as an '
    'int"), not a paraphrase of the name. Where the postcondition is '
    "expressible as an expression over `result` and `kwargs`, author it "
    "via `FunctionManager_add_functions(contracts={name: "
    "{'postconditions': [...]}})` so it is checked on every call.\n"
    "- **Fixtures for pure functions.** When the trajectory contains "
    "concrete inputs and the exact output a pure (`safe_noop`) function "
    "reproduces, store them via `FunctionManager_add_functions(fixtures="
    "{name: [{'args': {...}, 'result': ...}]})`; they are replayed "
    "whenever the function changes and reject silent regressions.\n"
    "- **Confirm the effect class.** Detection from the source is a lower "
    "bound (safe_noop < read_only < idempotent_effectful < "
    "unsafe_effectful). When you know a function's real class — an effect "
    "the source does not reveal, or a third-party import that only reads "
    "— call `confirm_side_effect_class(function_id, side_effect_class, "
    "rationale)`; raising is always allowed, lowering stops at the "
    "detected bound. Use `set_verification_policy(function_id, ...)` to "
    "demand more verification for unusually consequential functions; it "
    "can only raise the bar.\n\n"
    "### Third-party package dependencies\n\n"
    "If the trajectory used `install_python_packages` and the function "
    "you want to store imports any of those packages (anything beyond "
    "the Python standard library and the environment-provided "
    "namespaces `primitives` and `pydantic`), the function **requires "
    "a virtual environment**. `FunctionManager_add_functions` will "
    "reject the function if third-party imports are detected without "
    "a `venv_id`.\n\n"
    "Steps:\n"
    "1. Check existing venvs with `FunctionManager_list_venvs` — if "
    "one already declares the needed packages, reuse it.\n"
    "2. If no suitable venv exists, create one with "
    "`FunctionManager_add_venv`. Pass a minimal `pyproject.toml` "
    "string declaring only the packages the function actually "
    "imports. Example:\n\n"
    "```\n"
    "[project]\n"
    'name = "google-cloud-tools"\n'
    'version = "0.1.0"\n'
    'requires-python = ">=3.11"\n'
    "dependencies = [\n"
    '    "google-cloud-storage>=2.0.0",\n'
    "]\n"
    "```\n\n"
    "3. Pass the returned `venv_id` to "
    "`FunctionManager_add_functions(venv_id=<id>)`.\n\n"
    "Multiple functions that share the same dependency set should "
    "share a single venv. Do not create a separate venv per function "
    "when the dependency overlap is high — update an existing venv "
    "with `FunctionManager_update_venv` to add extra packages "
    "instead.\n\n"
)

_STORAGE_THREE_STORES = (
    "## Three Stores\n\n"
    "### Function Store — the *what*\n\n"
    "The FunctionManager stores concrete reusable callables. Add a "
    "genuinely new function with `FunctionManager_add_functions` "
    "(`venv_id` required for third-party imports; venvs are managed via "
    "`FunctionManager_add_venv` / `list_venvs` / `update_venv` / "
    "`delete_venv` / `set_function_venv`). Revise an existing function in "
    "place with `overwrite=True`. When a new function subsumes narrower "
    "variants, delete the superseded entries "
    "(`FunctionManager_delete_function`). Shape verification with "
    "`confirm_side_effect_class` and `set_verification_policy` — neither "
    "grants trust; verdicts from independent verification do. Do NOT "
    "store trivial one-liners, test scaffolding, or functions too "
    "task-specific to be reusable.\n\n"
    "### Guidance Store — the *how*\n\n"
    "The GuidanceManager stores procedural recipes: multi-step "
    "compositions, SOPs, and decision points — prose that references "
    "functions, not executable code (`GuidanceManager_add_guidance` / "
    "`GuidanceManager_update_guidance` / `GuidanceManager_delete_guidance`, "
    "cross-referencing concrete functions via `function_ids`).\n\n"
    "Guidance earns its entry only when a composition strategy is "
    "non-obvious and would be hard to rediscover — or when a simple "
    "*domain* operation required a non-obvious correction (an "
    "error-recovery loop against an external API, a silent data failure "
    "mode, a precondition discovered by trial and error): there the "
    "domain insight is the value. Do NOT store agent-runtime or tooling "
    "meta-tips (tool-loop plumbing, namespace-injection quirks, "
    "clarification-tool usage) — those are session mechanics, not domain "
    "playbooks. Do NOT duplicate what a function docstring already "
    "explains: when the only reusable artifact is one standalone function "
    "whose docstring fully covers its use, store the function and finish "
    "the review — no wrapper procedure restating its contract.\n\n"
    "**Shared rules and policies are the other first-class use of "
    "guidance.** A durable rule that could equally govern other "
    "procedures (thresholds, routing or escalation criteria, formatting "
    "or tone conventions, approval rules) belongs in ONE canonical "
    "guidance entry, linked via `function_ids` to every stored function "
    "that applies it — even when the procedure itself is simple. Search "
    "guidance for an existing statement of the rule first and link the "
    "new function into it rather than writing a second copy. Functions "
    "may bake the rule's current parameters into their implementation; "
    "name the linked entry in the function's docstring. When the rule "
    "changes later, the entry's `function_ids` enumerate exactly which "
    "functions must be revised — complete links at storage time are what "
    "make that maintenance reliable.\n\n"
    "### Knowledge Store — the *is*\n\n"
    "The KnowledgeManager stores durable sourced claims: facts, "
    "policies, definitions, decisions, constraints, insights, and "
    "preferences, carrying provenance (`source_refs`) when known. "
    "Search/filter before writing (`KnowledgeManager_search` / "
    "`KnowledgeManager_filter`); add with "
    "`KnowledgeManager_add_knowledge`; revise in place with "
    "`KnowledgeManager_update_knowledge`; retire with "
    "`KnowledgeManager_invalidate_knowledge` / "
    "`KnowledgeManager_supersede_knowledge`, or "
    "`KnowledgeManager_delete_knowledge` when hard removal is "
    "appropriate. The bar is high: only durable non-person, "
    "non-procedure, non-secret claims future sessions would otherwise "
    "rediscover — contact attributes belong in ContactManager, "
    "procedures in GuidanceManager, credentials in SecretManager. A "
    "no-op is fine; most trajectories yield no new claims.\n\n"
    "### Composing the stores\n\n"
    "Function = executable *what*; guidance = natural-language *how* "
    "referencing functions; knowledge = sourced *is*. When a trajectory "
    "reveals both a useful function and a non-trivial procedure using "
    "it, store the function first, then a guidance entry referencing it "
    "via `function_ids`. Store claims only when durable domain facts "
    "matter independently of how to act on them.\n\n"
)

_STORAGE_SUB_AGENT_PATTERNS = (
    "## Sub-Agent Delegation Patterns\n\n"
    'First apply the distillation dial (see "What Can Be Stored"): a '
    "`primitives.actor.act(...)` call is worth storing *as an agent* only "
    "when the sub-task genuinely needed its plan discovered at runtime. "
    "When the sub-agent's work was actually bounded judgment inside "
    "stable control flow, distill it down the dial — a function with "
    "`query_llm(...)` at the joints — instead of preserving the agent "
    "wrapper. Note also that `primitives.actor.act` is classified at the "
    "most consequential effect class (`unsafe_effectful`), so a stored "
    "function that spawns an agent carries the heaviest verification "
    "burden; thin, bounded functions earn trust far faster.\n\n"
    "Calls that survive this test are especially "
    "high-value storage candidates because they represent **pre-configured "
    "specialist agents**. Each `primitives.actor.act` invocation encodes a curated "
    "combination of `prompt_functions` (which tools the sub-agent sees), "
    "`guidelines` (how it should reason and compose those tools), "
    "`discovery_scope` (what it can find via search), and permission "
    "flags — together these define a specialist that can handle a "
    "particular *class* of tasks, not just the single task it was "
    "originally invoked for.\n\n"
    "### When to store\n\n"
    "Not every `primitives.actor.act` call is worth storing. Use this spectrum:\n\n"
    "- **Low value** — broad, unscoped delegation: all state managers "
    "in `prompt_functions`, generic or no `guidelines`, no "
    "`discovery_scope`, trivial `request`. This is just a passthrough "
    "that any future agent could reconstruct trivially.\n"
    "- **High value** — curated specialist: a carefully selected set of "
    "`prompt_functions`, detailed `guidelines` explaining how to compose "
    "those specific tools, a narrowed `discovery_scope`, and a non-trivial "
    "task that the sub-agent solved successfully. The configuration "
    "required real reasoning to discover and would be hard to "
    "rediscover from scratch.\n\n"
    "The more curation and domain knowledge went into the `primitives.actor.act` "
    "parameters, the more valuable it is to store.\n\n"
    "### What to bake in vs expose\n\n"
    "The parameters split naturally into two categories:\n\n"
    "- **Bake in** (agent specification): `guidelines`, "
    "`prompt_functions`, `discovery_scope`, `can_compose`, `can_store`, "
    "`can_spawn_sub_agents`, `timeout` — these define *what kind of "
    "specialist* this is and should be fixed in the stored function.\n"
    "- **Expose** (task specification): `request` — this defines *what "
    "to ask the specialist to do* and should be a parameter of the "
    "stored function.\n\n"
    "The result is a function that future callers can invoke with just "
    "a `request` string, without needing to know anything about the "
    "right tool selection, scoping, or behavioral guidelines.\n\n"
)

_STORAGE_RECURRING_DELIVERABLE = (
    "## Recurring Deliverables Without A Task\n\n"
    "A deliverable can be recurring with no scheduled task: the requester "
    'hands the job over once ("every week, ...") and simply asks again '
    "each time. Converge exactly as for a recurring task, with the "
    "conversation as the trigger. The first successful production is the "
    "evidence for a stored function named after the deliverable: skeleton "
    "in deterministic code, each judging substep at its own notch on the "
    "dial. Stated-but-dormant requirements belong in the function — a rule "
    'the requester stated ("if X ever happens, do Y") is evidence even '
    "unexercised — but never freeze structure neither stated nor observed; "
    "outside that envelope the function raises or returns early rather "
    "than guessing. Later instances refine the same function in place "
    "(`FunctionManager_add_functions` with `overwrite=True`), never "
    "near-duplicates. Then say so in your summary — name, numeric "
    "`function_id`, and calling convention — so the live session executes "
    "the stored function next time instead of re-deriving the procedure.\n\n"
)

_STORAGE_BASE_INSTRUCTIONS = (
    "## Instructions\n\n"
    "1. Review the trajectory for reusable patterns — including **pitfall "
    "patterns**, where an obvious approach failed in a non-obvious way (a "
    "silent data loss, a precondition an API doesn't enforce, an "
    "error-recovery loop after a misleading tool contract). Corrected "
    "pitfalls have high reuse value even when the fix is simple, because "
    "every future actor will attempt the obvious approach first; a brief "
    "guidance entry — what fails, why, the correct approach — saves them "
    "the discovery cycle.\n"
    "2. Search the existing stores to understand what already exists "
    "(use each store's search/filter tools).\n"
    "3. Decide what would improve the library. Prefer a clean, "
    "non-redundant library over a large one — most trajectories warrant "
    "function changes at most. Add guidance when a composition is "
    "genuinely non-obvious, and factor any durable shared rule into a "
    "single linked guidance entry per the Shared rules section.\n"
    "4. **Delete superseded functions when you add a generalization** "
    "(`FunctionManager_delete_function` on the now-redundant "
    "`function_id`s) — the same for outright duplicates and narrow "
    "special cases the new function handles.\n"
    "5. When done (or if there is nothing worth changing), respond with "
    "a brief summary of what you did (or that nothing was needed)."
)

# ---------------------------------------------------------------------------
# Shared tool docstrings
# ---------------------------------------------------------------------------

# One contract for both package-install tools (the act() sandbox overlay and
# the entrypoint-execution overlay bind differently, but the LLM-facing
# behavior is identical).
_INSTALL_PYTHON_PACKAGES_DOC = """Install Python packages into the current execution environment.

**You MUST use this tool whenever you need a Python package that is not
already available.** Never install via ``execute_code`` (``!pip install``,
``subprocess.run(["pip", ...])``, ``uv pip install``, or any other
shell-based method) — direct installs bypass the managed overlay and leave
the environment in an inconsistent state.

Installed packages are immediately importable in subsequent ``execute_code``
Python calls, and are removed automatically when the current task completes —
they never persist to later trajectories. If a requested package conflicts
with a system dependency, the pre-installed version takes precedence.

Parameters
----------
packages : list[str]
    pip/uv specifiers, e.g. ``"pandas"``, ``"pandas==2.1.0"``,
    ``"pandas>=2.0,<3.0"``, ``"pandas[sql]"``,
    ``"git+https://github.com/user/repo.git"``, ``"./path/to/wheel.whl"``.

Returns
-------
dict
    ``success`` (bool), ``stdout`` / ``stderr`` (installer output — on
    failure inspect ``stderr`` and adjust the specifiers), and ``packages``
    (the requested specifiers).
"""

# ---------------------------------------------------------------------------
# Trajectory compaction for storage review prompts
# ---------------------------------------------------------------------------

# Read-only store operations whose results the librarian can (and should)
# re-derive live with its own tools. Write operations (add/update/delete/…)
# stay verbatim — their results are small and record what changed.
_STORE_READ_TOOL_RE = re.compile(
    r"^(?:FunctionManager|GuidanceManager|KnowledgeManager)"
    r"_(?:search|filter|list|get)",
)
_TRAJ_SYSTEM_STUB_THRESHOLD = 2_000
_TRAJ_STORE_READ_STUB_THRESHOLD = 300
_TRAJ_TOOL_RESULT_HEAD = 4_000
_TRAJ_TOOL_RESULT_TAIL = 1_000


def _prepare_trajectory_for_storage_review(
    messages: list[dict] | None,
) -> list[dict]:
    """Compact a trajectory snapshot for a skill-librarian prompt.

    The librarian judges what the actor *did*, not what it was told it
    could do, and it queries the stores live with its own tools. Three
    rewrites keep the review prompt bounded without hiding decision
    signal:

    * Large system messages collapse to a one-line stub — the review
      prompt's own storage doctrine is authoritative. Small system
      messages (e.g. parent-chat context) stay verbatim.
    * Large results of store *reads* (FunctionManager / GuidanceManager /
      KnowledgeManager search/filter/list/get, including results
      delivered through ``check_status_*`` placeholders) collapse to an
      entry count. The call and its arguments stay visible: "searched
      the store, found nothing, built it by hand" is exactly the signal
      that something is worth storing, and empty/short results stay
      verbatim for that reason.
    * Any other oversized tool result keeps its head and tail around an
      elision marker.

    Provider reasoning payloads (encrypted blobs, reasoning summaries) are
    dropped outright: the librarian judges visible actions and results,
    and an encrypted chain of thought is unreadable bulk in its prompt.
    """
    from unify.common._async_tool.messages import strip_reasoning_payloads

    prepared = make_messages_safe_for_context_dump(messages)
    for _msg in prepared:
        if isinstance(_msg, dict):
            strip_reasoning_payloads(_msg)

    name_by_call_id: dict[str, str] = {}
    for msg in prepared:
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn_name = (tc.get("function") or {}).get("name")
            if tc_id and fn_name:
                name_by_call_id[str(tc_id)] = str(fn_name)

    def _origin_tool_name(msg: dict) -> str | None:
        name = msg.get("name") or name_by_call_id.get(
            str(msg.get("tool_call_id") or ""),
        )
        if not name:
            return None
        name = str(name)
        # ``check_status_<call_id>`` delivers an async tool's real result;
        # resolve back to the originating tool for classification.
        if name.startswith("check_status_"):
            return name_by_call_id.get(name[len("check_status_") :], name)
        return name

    def _entry_count(content: str) -> str | None:
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, list):
            return f"{len(parsed)} entries"
        return None

    def _content_text(content: object) -> str | None:
        """Flatten message content to plain text for measurement/rewrites.

        Tool results arrive either as a plain string or as an OpenAI
        content-parts list (``[{"type": "text", "text": ...}, ...]``) —
        execute_code results use the latter. Rewrites always store back a
        plain string; that is fine for a serialized trajectory dump.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    parts.append(
                        (
                            text
                            if isinstance(text, str)
                            else f"[{part.get('type') or 'non-text'} part]"
                        ),
                    )
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return None

    for msg in prepared:
        content = _content_text(msg.get("content"))
        if content is None:
            continue
        role = msg.get("role")
        if role == "system":
            if len(content) > _TRAJ_SYSTEM_STUB_THRESHOLD:
                msg["content"] = (
                    f"[System prompt omitted ({len(content):,} chars). The "
                    "storage doctrine in this review prompt is "
                    "authoritative.]"
                )
            continue
        if role != "tool":
            continue
        origin = _origin_tool_name(msg)
        if (
            origin
            and _STORE_READ_TOOL_RE.match(origin)
            and len(content) > _TRAJ_STORE_READ_STUB_THRESHOLD
        ):
            count = _entry_count(content)
            size = f"{len(content):,} chars"
            detail = f"{count}, {size}" if count else size
            msg["content"] = (
                f"[{origin} result omitted ({detail}). Query the stores "
                "directly with your tools; they are the source of truth.]"
            )
        elif len(content) > _TRAJ_TOOL_RESULT_HEAD + _TRAJ_TOOL_RESULT_TAIL:
            omitted = len(content) - _TRAJ_TOOL_RESULT_HEAD - _TRAJ_TOOL_RESULT_TAIL
            msg["content"] = (
                content[:_TRAJ_TOOL_RESULT_HEAD]
                + f"\n… [{omitted:,} chars omitted] …\n"
                + content[-_TRAJ_TOOL_RESULT_TAIL:]
            )
    return prepared


# ---------------------------------------------------------------------------
# Shared storage tool construction
# ---------------------------------------------------------------------------


def _build_storage_tools(
    *,
    actor: "CodeActActor",
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    task_entrypoint_review: dict[str, Any] | None = None,
) -> tuple[Dict[str, Callable], list[str], list[str]]:
    """Build the tool dict shared by both post-processing and proactive storage loops.

    Returns ``(tools, storage_active_lines, dormant_lines)`` so callers can
    reference which inner tools are still actively reviewing skills and
    which completed tools can be queried. Tool docstrings deliberately stay
    static — per-run listings belong in the (volatile tail of the) system
    prompt so the serialized tool schemas are byte-identical across loops
    and stay prompt-cache-friendly.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    km = actor.knowledge_manager

    storage_methods: list[Any] = [
        fm.search_functions,
        fm.filter_functions,
        fm.list_functions,
        fm.add_functions,
        fm.delete_function,
        fm.reconcile_dependencies,
        fm.add_venv,
        fm.list_venvs,
        fm.get_venv,
        fm.update_venv,
        fm.delete_venv,
        fm.set_function_venv,
        fm.get_function_venv,
        gm.search,
        gm.filter,
        gm.get_guidance,
        gm.add_guidance,
        gm.update_guidance,
        gm.delete_guidance,
        gm.reconcile_dependencies,
    ]
    if km is not None:
        storage_methods.extend(
            [
                km.search,
                km.filter,
                km.get_knowledge,
                km.add_knowledge,
                km.update_knowledge,
                km.delete_knowledge,
                km.invalidate_knowledge,
                km.supersede_knowledge,
            ],
        )

    tools: Dict[str, Callable] = {
        **methods_to_tool_dict(
            *storage_methods,
            include_class_name=True,
        ),
        **_verification_librarian_tools(fm),
    }

    # ── Wire ask_about_completed_tool from snapshot ───────────────────

    _meta = completed_tool_metadata or {}
    storage_active_lines: list[str] = []
    storage_active_handles: Dict[str, Any] = {}
    dormant_lines: list[str] = []
    for name, fn in ask_tools.items():
        entry = None
        for _cid, _m in _meta.items():
            if _m.get("ask_fn") is fn:
                entry = _m
                break
        handle = entry.get("handle") if entry else None
        if handle is not None and hasattr(handle, "done") and not handle.done():
            storage_active_lines.append(f"- `{name}` [storage-active]")
            storage_active_handles[name] = handle
        else:
            dormant_lines.append(f"- `{name}`")

    if ask_tools:

        async def ask_about_completed_tool(
            tool_name: str,
            question: str,
        ) -> str:
            """Ask a follow-up question about a completed tool from the trajectory.

            Use this to inspect a completed tool's internal reasoning or
            results. The available tool names are listed in the Completed
            Tools section of this prompt.
            """
            fn = ask_tools.get(tool_name)
            if fn is None:
                return (
                    f"Tool '{tool_name}' not found. Available: {list(ask_tools.keys())}"
                )
            handle = await fn(question=question)
            if hasattr(handle, "result"):
                result = handle.result
                if callable(result):
                    result = result()
                if inspect.isawaitable(result):
                    result = await result
                return str(result)
            return str(handle)

        tools["ask_about_completed_tool"] = ask_about_completed_tool

    # ── Wire steering tools for storage-active inner handles ──────────

    if storage_active_handles:
        _sa_handles = storage_active_handles

        def _resolve_handle(tool_name: str) -> tuple[Any | None, str | None]:
            h = _sa_handles.get(tool_name)
            if h is None:
                avail = list(_sa_handles.keys())
                return None, (
                    f"Tool '{tool_name}' not found or no longer storage-active. "
                    f"Available: {avail}"
                )
            if hasattr(h, "done") and h.done():
                return None, (
                    f"Tool '{tool_name}' has already finished its storage review."
                )
            return h, None

        async def stop_inner_storage(tool_name: str, reason: str) -> str:
            """Stop an inner storage loop, preventing it from storing anything further.

            Use this when you have determined that the inner agent's storage
            would be redundant (e.g. you are storing a comprehensive function
            that already covers the inner agent's scope).
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.stop(reason=reason)
            return f"Stopped inner storage for '{tool_name}': {reason}"

        async def interject_inner_storage(tool_name: str, message: str) -> str:
            """Inject a directive into an inner storage loop's conversation.

            Use this to provide context that should influence the inner
            loop's storage decisions (e.g. "The parent is storing a
            comprehensive function — only store yours if it is genuinely
            independent and reusable in isolation").
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.interject(message)
            return f"Interjected into inner storage for '{tool_name}'."

        async def pause_inner_storage(tool_name: str) -> str:
            """Temporarily pause an inner storage loop.

            Use this to halt an inner loop while you make decisions,
            then resume it with ``resume_inner_storage``. The inner loop
            will not proceed until resumed (or until its timeout expires).
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.pause()
            return f"Paused inner storage for '{tool_name}'."

        async def resume_inner_storage(tool_name: str) -> str:
            """Resume a previously paused inner storage loop.

            Call this after ``pause_inner_storage`` to let the inner
            loop continue its skill review.
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.resume()
            return f"Resumed inner storage for '{tool_name}'."

        tools["stop_inner_storage"] = stop_inner_storage
        tools["interject_inner_storage"] = interject_inner_storage
        tools["pause_inner_storage"] = pause_inner_storage
        tools["resume_inner_storage"] = resume_inner_storage

    if task_entrypoint_review:
        attach_entrypoint = task_entrypoint_review.get("attach_entrypoint")
        promote_task_offline_hook = task_entrypoint_review.get("promote_task_offline")
        metadata = dict(task_entrypoint_review.get("metadata") or {})
        task_id = metadata.get("task_id")
        task_name = metadata.get("task_name") or metadata.get("name") or "the task"

        async def attach_entrypoint_to_recurring_task(
            function_id: int,
            rationale: str,
        ) -> str:
            """Bind a stored function as the executor of future runs of this task.

            Use this only after you have reviewed the completed trajectory and
            decided that the stored function captures a stable reusable procedure
            that preserves the observed operational contract. Binding does not
            grant trust: future runs execute the function under independent
            verification, trust is earned from those verdicts, and offline
            delivery follows once every function it calls is trusted. Leaving
            the task description-driven is valid when future runs still need
            broad planning or tool discovery.
            """

            if not callable(attach_entrypoint):
                return "No task entrypoint attachment hook is available."
            # A recorded entrypoint id that does not resolve at execution time
            # fails every future wake of the task, so verify the id actually
            # persisted before recording it on the definition.
            if fm is None:
                return (
                    "Refusing to attach an entrypoint: no FunctionManager is "
                    "available to verify the function id."
                )
            try:
                resolution = fm.filter_functions(
                    filter=f"function_id == {int(function_id)}",
                    include_implementations=False,
                )
            except Exception as exc:
                return (
                    f"Refusing to attach function_id {int(function_id)}: "
                    f"resolvability check failed ({type(exc).__name__}: {exc})."
                )
            if not resolution:
                return (
                    f"Refusing to attach function_id {int(function_id)}: no "
                    "stored function resolves to that id. Store the function "
                    "first (or re-check the id from the add_functions result) "
                    "and attach the id that actually persisted."
                )
            return str(
                attach_entrypoint(
                    function_id=int(function_id),
                    rationale=str(rationale),
                ),
            )

        attach_entrypoint_to_recurring_task.__doc__ += (
            f"\n\nCurrent task: {task_name} (task_id={task_id}). "
            "The tool only patches future runs; it never rewrites the "
            "completed run, grants trust, or flips delivery to offline."
        )
        tools["attach_entrypoint_to_recurring_task"] = (
            attach_entrypoint_to_recurring_task
        )

        async def promote_task_offline() -> str:
            """Move this task to offline (headless) delivery if its executor is trusted.

            Eligibility is read from the verification ledger: the task must
            have a bound entrypoint and every function in that entrypoint's
            transitive closure must have earned trust (verify=False). This
            tool never grants trust and takes no evidence; when the closure is
            not yet trusted it reports which function ids still stand in the
            way. Promotion also happens automatically at the end of the run in
            which the last member of the closure earns trust, so calling this
            is only needed to promote a task whose closure was already trusted.
            """

            if not callable(promote_task_offline_hook):
                return "No task offline-promotion hook is available."
            return str(promote_task_offline_hook())

        promote_task_offline.__doc__ += (
            f"\n\nCurrent task: {task_name} (task_id={task_id})."
        )
        tools["promote_task_offline"] = promote_task_offline

    return tools, storage_active_lines, dormant_lines


# ---------------------------------------------------------------------------
# Storage check: start a review loop and return its handle
# ---------------------------------------------------------------------------


def _start_storage_check_loop(
    *,
    trajectory: list[dict],
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    actor: "CodeActActor",
    original_result: str,
    parent_lineage: list[str] | None = None,
    stop_reason: str | None = None,
    proactive_summaries: list[str] | None = None,
    post_run_review_context: PostRunReviewContext | None = None,
    live_session: bool = False,
) -> "AsyncToolLoopHandle | None":
    """Start a loop that reviews a completed trajectory for reusable knowledge.

    With ``live_session=True`` the trajectory belongs to a persistent
    session that is still running: the review covers the turns completed
    so far, ``original_result`` is the latest turn's response, and the
    librarian's summary is delivered back into the live session as a
    background note.

    The loop maintains three complementary stores:

    * **FunctionManager** — stores the *what*: concrete, reusable function
      implementations (the building blocks).
    * **GuidanceManager** — stores the *how*: high-level guidance on
      composing multiple functions together to accomplish broader tasks
      (the recipes / playbooks).
    * **KnowledgeManager** — stores the *is*: durable sourced claims
      (optional; included when present on the actor).

    FunctionManager and GuidanceManager are required. Returns ``None``
    when either is missing. KnowledgeManager tools are included when
    present; absence of KnowledgeManager does not block the loop.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None
    task_entrypoint_review = (
        post_run_review_context.extensions.get("task_entrypoint_review")
        if post_run_review_context is not None
        else None
    )

    tools, storage_active_lines, dormant_lines = _build_storage_tools(
        actor=actor,
        ask_tools=ask_tools,
        completed_tool_metadata=completed_tool_metadata,
        task_entrypoint_review=task_entrypoint_review,
    )

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(
        _prepare_trajectory_for_storage_review(trajectory),
        default=str,
    )

    completed_tools_section = ""
    if storage_active_lines or dormant_lines:
        listing = "\n".join([*storage_active_lines, *dormant_lines])
        completed_tools_section = (
            "## Completed Tools\n\n"
            "These completed tools from the trajectory can be queried via "
            "`ask_about_completed_tool` (entries marked [storage-active] "
            "are still running their own background skill review):\n\n"
            f"{listing}\n\n"
        )

    # Build optional section about inner storage loops.
    inner_storage_section = ""
    if storage_active_lines:
        inner_storage_section = (
            "## Inner Storage Loops\n\n"
            "Some inner tools from this trajectory are currently running "
            "their own background skill-review loops:\n\n"
            + "\n".join(storage_active_lines)
            + "\n\n"
            "These inner loops may be storing functions independently at a "
            "finer granularity. You can:\n"
            "- Query them via `ask_about_completed_tool`\n"
            "- Inject directives via `interject_inner_storage`\n"
            "- Stop them via `stop_inner_storage`\n"
            "- Pause/resume them via `pause_inner_storage` / "
            "`resume_inner_storage`\n\n"
            "Use these to coordinate storage decisions (e.g. stop an inner "
            "loop that would store something redundant, or interject context "
            "about what you plan to store at the higher level).\n\n"
        )

    # ── Proactive storage awareness ───────────────────────────────────
    proactive_storage_section = ""
    if proactive_summaries:
        summaries_text = "\n\n".join(
            f"**Proactive pass {i + 1}:**\n{s}"
            for i, s in enumerate(proactive_summaries)
        )
        proactive_storage_section = (
            "## Proactive Storage Already Performed\n\n"
            "The executing agent proactively triggered skill storage during "
            "this run via the `store_skills` tool. Below are the summaries "
            "from each proactive storage pass:\n\n"
            f"{summaries_text}\n\n"
            "Check the function, guidance, and knowledge stores to confirm "
            "what was already added. Do not duplicate existing entries. "
            "Focus on any additional reusable patterns — especially from "
            "sections of the trajectory *after* the last `store_skills` "
            "call — that the proactive passes may have missed.\n\n"
        )

    instructions = _STORAGE_BASE_INSTRUCTIONS
    if proactive_summaries:
        instructions = (
            "## Instructions\n\n"
            "1. Skill storage was proactively triggered during this run. "
            "Start by reviewing the proactive storage summaries below and "
            "checking the function, guidance, and knowledge stores to see "
            "what was already added.\n"
            "2. Search the existing stores to confirm exactly what was stored "
            "(use the search/filter tools for each store).\n"
            "3. Review the full trajectory — especially sections after the "
            "last `store_skills` call — for any additional reusable patterns "
            "the proactive passes may have missed.\n"
            "4. Do not duplicate entries that already exist. Only add, update, "
            "or merge if there is genuinely new value.\n"
            "5. When done (or if there is nothing more to add), respond "
            "with a brief summary of what you did (or that nothing additional "
            "was needed)."
        )

    stop_context_section = ""
    if stop_reason:
        stop_context_section = (
            "## Session Termination Context\n\n"
            "This session was explicitly stopped by the user. The stop reason "
            "provides important signal about whether the user intended the "
            "work to be saved:\n\n"
            f"> {stop_reason}\n\n"
            "Weigh this context when deciding what to store. If the reason "
            "indicates the user wanted the procedure remembered or saved, that "
            "is a strong positive signal — look for reusable patterns in the "
            "trajectory. If the reason indicates cancellation or abandonment, "
            "the trajectory is less likely to contain patterns worth "
            "persisting, though genuinely reusable sub-patterns may still "
            "be worth storing.\n\n"
        )

    live_session_section = ""
    if live_session:
        live_session_section = (
            "## Live Session Turn Review\n\n"
            "The trajectory below is a persistent interactive session that "
            "is still running; the agent has just completed a request turn "
            "and is waiting for the next instruction. You are reviewing "
            "mid-session — earlier turns are included as context and may "
            "already have been reviewed (prior passes appear under "
            "'Proactive Storage Already Performed').\n\n"
            "- Focus on what the latest turn(s) added since the last "
            "review pass.\n"
            "- Steady state is cheap: when the latest turn(s) only "
            "re-executed already-stored procedures and the requester added "
            "no new requirement, amendment or correction, there is nothing "
            "to do — say so in one sentence and finish immediately, "
            "without searching the stores first.\n"
            "- Prefer updating an existing stored entry over adding a "
            "near-duplicate: when this session already stored the "
            "procedure and a later turn refined its spec, apply the "
            "refinement with `FunctionManager_add_functions` "
            "(`overwrite=True`).\n"
            "- Your final summary is delivered to the live session as a "
            "background note. Make it actionable: name what changed "
            "(function names, numeric ids, calling conventions) so the "
            "session can execute stored functions on the next request "
            "rather than re-deriving procedures.\n\n"
        )

    task_entrypoint_section = ""
    if task_entrypoint_review:
        metadata = dict(task_entrypoint_review.get("metadata") or {})
        metadata_json = json.dumps(metadata, indent=2, default=str)
        task_entrypoint_section = (
            "## Recurring Task Entrypoint Review\n\n"
            "This trajectory completed a scheduled or triggered task that had "
            "no stored entrypoint when it ran. Explicitly consider whether "
            "the successful run revealed a stable reusable procedure worth "
            "attaching to future instances. No-op is valid: keep the task "
            "description-driven if future runs need broad planning, changing "
            "tool discovery, or open-ended judgment. A stabilized procedure "
            "may still use focused `query_llm(...)` calls for bounded "
            "semantic substeps, choosing `model=` deliberately — see "
            '"Model choice is part of distillation" above.\n\n'
            "If you store a FunctionManager function that is a stable "
            "candidate for future runs, call "
            "`attach_entrypoint_to_recurring_task(function_id=..., rationale=...)` "
            "— only after the function is persisted and you have its numeric "
            "function_id. The candidate must preserve the observed live "
            'execution chain per "Durable task executor candidates" above; '
            "if it materially changes primitives, inputs, ordering, or "
            "failure behavior, store it as a helper/guidance only and do not "
            "bind it.\n\n"
            "Binding records the executor; it does not grant trust or "
            "promote the task to offline delivery. Each call then runs under "
            "independent verification, verdicts accumulate on the function's "
            "ledger, and once every function the entrypoint calls is trusted "
            "the task is promoted to offline delivery automatically — "
            "`promote_task_offline()` only re-checks that eligibility. There "
            "is no evidence to submit.\n\n"
            "Task metadata:\n"
            f"```json\n{metadata_json}\n```\n\n"
        )

    role_line = (
        (
            "You are a skill librarian. A CodeActActor is running a "
            "persistent interactive session and has just completed a "
            "request turn. Your job is to review the session trajectory so "
            "far and decide whether anything is worth persisting for future "
            "reuse. Often nothing is — that is perfectly fine.\n\n"
        )
        if live_session
        else (
            "You are a skill librarian. A CodeActActor has just completed a task. "
            "Your job is to review the execution trajectory and decide whether "
            "anything is worth persisting for future reuse. Often nothing is — "
            "that is perfectly fine.\n\n"
        )
    )
    # The recurring-deliverable doctrine covers conversational convergence;
    # a task-bound run has its own entrypoint-review section instead.
    recurring_deliverable_section = (
        "" if task_entrypoint_review else _STORAGE_RECURRING_DELIVERABLE
    )
    trajectory_header = (
        "## Session Trajectory So Far\n\n"
        if live_session
        else "## Completed Trajectory\n\n"
    )
    result_header = (
        "## Latest Turn Response\n\n" if live_session else "## Final Result\n\n"
    )

    # Static doctrine first, volatile trajectory last: every storage loop
    # shares the same byte-identical prefix (role + doctrine + instructions),
    # so provider prompt caching only pays cold tokens for the per-run tail.
    system_prompt = (
        f"{role_line}"
        f"{_STORAGE_WHAT_CAN_BE_STORED}"
        f"{_STORAGE_THREE_STORES}"
        f"{_STORAGE_SUB_AGENT_PATTERNS}"
        f"{recurring_deliverable_section}"
        f"{instructions}"
        "\n\n"
        f"{stop_context_section}"
        f"{live_session_section}"
        f"{task_entrypoint_section}"
        f"{inner_storage_section}"
        f"{completed_tools_section}"
        f"{proactive_storage_section}"
        f"{trajectory_header}"
        f"{trajectory_json}\n\n"
        f"{result_header}"
        f"{original_result}"
    )

    client = new_llm_client(actor._model, purpose="planning", origin="StorageCheck")
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message=(
            "Review the trajectory and store any reusable functions, "
            "compositional guidance, and durable knowledge claims."
        ),
        tools=tools,
        loop_id="StorageCheck(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


# ---------------------------------------------------------------------------
# Proactive storage: on-demand storage loop triggered from the doing loop
# ---------------------------------------------------------------------------


def _start_proactive_storage_loop(
    *,
    trajectory: list[dict],
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    actor: "CodeActActor",
    request: str,
    parent_lineage: list[str] | None = None,
) -> "AsyncToolLoopHandle | None":
    """Start an on-demand storage review loop triggered mid-flight by the doing loop.

    Shares the same tool set and core prompt sections as the post-processing
    ``_start_storage_check_loop``, but uses a distinct prompt framing:
    the trajectory is partial (task still in progress), there is no final
    result, and the ``request`` parameter focuses the reviewer on specific
    skills worth storing.

    Returns ``None`` when either FunctionManager or GuidanceManager is
    missing.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None

    tools, storage_active_lines, dormant_lines = _build_storage_tools(
        actor=actor,
        ask_tools=ask_tools,
        completed_tool_metadata=completed_tool_metadata,
    )

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(
        _prepare_trajectory_for_storage_review(trajectory),
        default=str,
    )

    completed_tools_section = ""
    if storage_active_lines or dormant_lines:
        listing = "\n".join([*storage_active_lines, *dormant_lines])
        completed_tools_section = (
            "## Completed Tools\n\n"
            "These completed tools from the trajectory can be queried via "
            "`ask_about_completed_tool` (entries marked [storage-active] "
            "are still running their own background skill review):\n\n"
            f"{listing}\n\n"
        )

    inner_storage_section = ""
    if storage_active_lines:
        inner_storage_section = (
            "## Inner Storage Loops\n\n"
            "Some inner tools from this trajectory are currently running "
            "their own background skill-review loops:\n\n"
            + "\n".join(storage_active_lines)
            + "\n\n"
            "These inner loops may be storing functions independently at a "
            "finer granularity. You can:\n"
            "- Query them via `ask_about_completed_tool`\n"
            "- Inject directives via `interject_inner_storage`\n"
            "- Stop them via `stop_inner_storage`\n"
            "- Pause/resume them via `pause_inner_storage` / "
            "`resume_inner_storage`\n\n"
            "Use these to coordinate storage decisions (e.g. stop an inner "
            "loop that would store something redundant, or interject context "
            "about what you plan to store at the higher level).\n\n"
        )

    instructions = (
        "## Instructions\n\n"
        "1. Review the trajectory so far, focusing on the storage request.\n"
        "2. Search the existing stores to understand what already exists "
        "(use the search/filter tools for each store).\n"
        "3. Decide what actions (if any) would improve the library based on "
        "the requested skill(s). Prefer a clean, non-redundant library over "
        "a large one.\n"
        "4. When done (or if there is nothing worth storing), respond "
        "with a brief, concrete summary of what you stored (function names, "
        "guidance titles, knowledge claim titles) or that nothing was needed. "
        "This summary will be visible to both the executing agent and a "
        "follow-up storage review, so be specific."
    )

    # Static doctrine first, volatile trajectory last — same prompt-cache
    # prefix as the post-run storage check.
    system_prompt = (
        "You are a skill librarian. A CodeActActor is currently executing "
        "a task and has proactively requested skill storage. Your job is "
        "to review the execution trajectory so far and store the "
        "requested skill(s) for future reuse. Often nothing is worth "
        "storing — that is perfectly fine.\n\n"
        f"{_STORAGE_WHAT_CAN_BE_STORED}"
        f"{_STORAGE_THREE_STORES}"
        f"{_STORAGE_SUB_AGENT_PATTERNS}"
        f"{instructions}"
        "\n\n"
        f"{inner_storage_section}"
        f"{completed_tools_section}"
        "## Storage Request\n\n"
        f"{request}\n\n"
        "## Trajectory So Far\n\n"
        f"{trajectory_json}"
    )

    client = new_llm_client(actor._model, purpose="planning", origin="ProactiveStorage")
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message=(
            f"The executing agent has proactively requested skill storage: "
            f"{request!r}. Review the trajectory so far and store the "
            f"relevant functions, guidance, and knowledge claims."
        ),
        tools=tools,
        loop_id="ProactiveStorage(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


class _StorageCheckHandle(SteerableToolHandle):
    """Wraps an inner handle and runs a storage check after task completion.

    Lifecycle phases:

    * **task** -- the inner tool loop is running.  All steering methods
      forward to the inner handle.  Notifications from the inner handle
      are relayed to consumers.
    * **storage** -- the task has completed.  ``result()`` has already
      resolved with the original task result.  A second loop reviews the
      trajectory for reusable skills.  The handle remains live: steering
      methods (ask, interject, stop, pause, resume) operate on the
      storage loop, and ``done()`` returns ``False``.
    * **done** -- both phases have completed (or were stopped/skipped).
      ``done()`` returns ``True``.

    ``result()`` resolves at the end of Phase 1 — callers get the task
    result without waiting for storage.  ``done()`` reflects full
    lifecycle completion (including storage).  This means nested actor
    loops propagate results immediately while storage runs concurrently
    in the background.
    """

    def __init__(
        self,
        *,
        inner: "AsyncToolLoopHandle",
        actor: "CodeActActor",
        post_run_review_context: PostRunReviewContext | None = None,
        meter: Optional[RunMeter] = None,
        turn_reviews_enabled: bool = False,
    ) -> None:
        self._inner = inner
        self._actor = actor
        self._post_run_review_context = post_run_review_context
        self._meter = meter
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        self._task_done_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._original_result: Optional[str] = None
        self._task_failure: Optional[BaseException] = None
        self._storage_handle: Optional["AsyncToolLoopHandle"] = None
        self._phase: str = "task"  # "task" | "storage" | "done"
        self._stopped: bool = False
        self._stop_reason: Optional[str] = None
        self._active_relay: Optional[asyncio.Task] = None

        # Turn-boundary reviews for persistent sessions. A persist=True loop
        # never self-completes, so Phase 2 alone would defer distillation to
        # whenever the session is finally stopped — a session that performs
        # the same deliverable every turn would never converge. Instead, each
        # completed turn that ran tools gets a mid-session review of the
        # trajectory so far; its summary is recorded for the final review and
        # delivered back into the live loop as a transcript note.
        self._turn_reviews_enabled = bool(turn_reviews_enabled)
        self._turn_review_task: Optional[asyncio.Task] = None
        self._turn_review_handle: Optional["AsyncToolLoopHandle"] = None
        self._turn_review_rerun: bool = False
        self._latest_turn_response: str = ""
        self._reviewed_tool_msg_count: int = 0

        # Start the two-phase lifecycle manager.
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())

    @property
    def run_stats(self) -> dict[str, Any]:
        """Token accounting for the execution row (planning tokens for an agentic run)."""
        if self._meter is None:
            return {}
        return {"tokens": self._meter.snapshot()["tokens"]}

    # ── Internal helpers ──────────────────────────────────────────────

    @property
    def _pause_event(self):
        """Delegate to the active inner handle so get_handle_paused_state works."""
        handle = self._active_handle
        if handle is not None:
            return getattr(handle, "_pause_event", None)
        return None

    @property
    def _active_handle(self) -> Optional["SteerableToolHandle"]:
        """The currently active inner handle for steering delegation."""
        if self._phase == "task":
            return self._inner
        if self._phase == "storage":
            return self._storage_handle
        return None

    async def _relay_notifications_from(
        self,
        source: "SteerableToolHandle",
    ) -> None:
        """Forward notifications from *source* into our queue until cancelled.

        ``type="response"`` notifications are the persist-mode turn
        boundary — the loop has finished a request and re-entered its wait
        state — so they are also the trigger for mid-session storage
        reviews when those are enabled.
        """
        try:
            while True:
                notif = await source.next_notification()
                await self._notification_q.put(notif)
                if (
                    self._turn_reviews_enabled
                    and isinstance(notif, dict)
                    and notif.get("type") == "response"
                ):
                    self._note_turn_boundary(str(notif.get("content") or ""))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def _note_turn_boundary(self, latest_response: str) -> None:
        """Schedule a mid-session storage review for a completed turn.

        At most one review runs at a time; a boundary that arrives while
        one is in flight coalesces into a single re-run against the
        then-current trajectory.
        """
        if self._phase != "task":
            return
        self._latest_turn_response = latest_response
        if self._turn_review_task is not None and not self._turn_review_task.done():
            self._turn_review_rerun = True
            return
        self._turn_review_task = asyncio.create_task(self._run_turn_reviews())

    @staticmethod
    def _tool_activity_count(messages: list) -> int:
        """Completed tool results in the transcript — the 'work happened' signal."""
        return sum(
            1 for m in messages if isinstance(m, dict) and m.get("role") == "tool"
        )

    def _snapshot_inner_trajectory(self) -> list[dict]:
        try:
            client = getattr(self._inner, "_client", None)
            if client is not None:
                return make_messages_safe_for_context_dump(
                    list(getattr(client, "messages", []) or []),
                )
        except Exception:
            pass
        return []

    async def _run_turn_reviews(self) -> None:
        """Run mid-session storage reviews until no boundary is pending.

        A turn with no new completed tool activity (pure conversation) is
        skipped — there is nothing new to distill. Compaction can shrink
        the transcript; the watermark follows it down so counting stays
        monotone against the live message list.
        """
        while True:
            self._turn_review_rerun = False
            trajectory = self._snapshot_inner_trajectory()
            tool_count = self._tool_activity_count(trajectory)
            if tool_count < self._reviewed_tool_msg_count:
                self._reviewed_tool_msg_count = tool_count
            if tool_count > self._reviewed_tool_msg_count:
                await self._run_one_turn_review(
                    trajectory,
                    tool_count,
                    reviewed_messages=len(trajectory),
                )
            if not self._turn_review_rerun:
                return

    async def _run_one_turn_review(
        self,
        trajectory: list[dict],
        tool_count: int,
        *,
        reviewed_messages: int,
    ) -> None:
        ask_tools: dict = {}
        try:
            ask_tools = getattr(self._inner._task, "get_ask_tools", lambda: {})()
        except Exception:
            pass
        completed_tool_metadata: dict = {}
        try:
            completed_tool_metadata = getattr(
                self._inner._task,
                "get_completed_tool_metadata",
                lambda: {},
            )()
        except Exception:
            pass

        proactive_summaries: list[str] = []
        _ctx = _CURRENT_AGENT_CONTEXT.get(None)
        if _ctx is not None:
            proactive_summaries = list(_ctx.proactive_storage_summaries)

        _tr_suffix = _token_hex(2)
        _tr_call_id = new_call_id()
        _tr_parent = TOOL_LOOP_LINEAGE.get([])
        _tr_parent_lineage = list(_tr_parent) if isinstance(_tr_parent, list) else []
        _tr_hierarchy = [
            *_tr_parent_lineage,
            f"StorageCheck(CodeActActor.act)({_tr_suffix})",
        ]
        _tr_lineage_token = TOOL_LOOP_LINEAGE.set(_tr_hierarchy)
        _tr_suffix_token = _PENDING_LOOP_SUFFIX.set(_tr_suffix)
        try:
            await publish_manager_method_event(
                _tr_call_id,
                "CodeActActor",
                "StorageCheck",
                phase="incoming",
                display_label=_DEFAULT_STORAGE_REVIEW_LABEL,
                hierarchy=_tr_hierarchy,
                instructions=_DEFAULT_STORAGE_REVIEW_INSTRUCTIONS,
            )
            storage_handle = _start_storage_check_loop(
                trajectory=trajectory,
                ask_tools=ask_tools,
                completed_tool_metadata=completed_tool_metadata,
                actor=self._actor,
                original_result=self._latest_turn_response,
                parent_lineage=_tr_parent_lineage,
                proactive_summaries=proactive_summaries or None,
                live_session=True,
            )
            if storage_handle is None:
                return
            self._turn_review_handle = storage_handle
            try:
                summary = await storage_handle.result()
            except Exception as exc:
                logger.warning(
                    f"Turn StorageCheck failed: {type(exc).__name__}: {exc}",
                )
                await self._notification_q.put(
                    {
                        "type": "turn_storage_review_complete",
                        "message": (
                            f"StorageCheck failed: {type(exc).__name__}: {exc}"
                        ),
                        "success": False,
                    },
                )
                return
            finally:
                self._turn_review_handle = None

            self._reviewed_tool_msg_count = tool_count
            if _ctx is not None:
                _ctx.proactive_storage_summaries.append(summary)
            await self._notification_q.put(
                {
                    "type": "turn_storage_review_complete",
                    "message": summary,
                    "success": True,
                },
            )
            # Leave the librarian's summary in the live session's transcript
            # so the next request can execute what was stored instead of
            # re-deriving the procedure, and let the reviewed turns shed
            # their raw tool payloads — the review is the checkpoint that
            # makes them safe to compact. Both are transcript-only: no LLM
            # turn fires.
            queue = getattr(self._inner, "_queue", None)
            if queue is not None:
                queue.put_nowait(
                    {
                        "_transcript_note": {
                            "text": (
                                "[background skill consolidation — automated "
                                "note, not a user message]\n"
                                f"{summary}"
                            ),
                        },
                    },
                )
                queue.put_nowait(
                    {
                        "_compact_transcript": {
                            "reviewed_messages": reviewed_messages,
                        },
                    },
                )
        finally:
            await publish_manager_method_event(
                _tr_call_id,
                "CodeActActor",
                "StorageCheck",
                phase="outgoing",
                display_label=_DEFAULT_STORAGE_REVIEW_LABEL,
                hierarchy=_tr_hierarchy,
            )
            TOOL_LOOP_LINEAGE.reset(_tr_lineage_token)
            _PENDING_LOOP_SUFFIX.reset(_tr_suffix_token)

    async def abandon_storage_review(self, *, reason: str) -> None:
        """End the storage phase now, without waiting for the review to finish.

        Called when the actor the review depends on is closing. A review needs
        that actor's venv pool, shell pool and sandboxes to do anything useful,
        so once they are torn down the review cannot succeed -- it can only
        keep retrying against them. Five offline task pods on staging stayed
        busy for eight days that way, still issuing inference for runs recorded
        as finished the week before, because nothing connected the two
        lifetimes: the actor closed its pools and walked away from the review.

        The bound this gives a review is its actor's lifetime, not a clock. A
        review that is genuinely working is never interrupted -- the process
        that owns the run owns the actor, and only ends it when the run is
        done with it.

        ``stop`` is cooperative and a loop wedged in a retry against something
        already gone never notices, so the lifecycle task is cancelled after
        it. That is what actually ends the inference.
        """

        if self._completion_event.is_set():
            return
        turn_handle = self._turn_review_handle
        if turn_handle is not None:
            try:
                await turn_handle.stop(reason=reason)
            except Exception:
                pass
        turn_task = self._turn_review_task
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
        handle = self._storage_handle
        if handle is not None:
            try:
                await handle.stop(reason=reason)
            except Exception:
                pass
        lifecycle = self._lifecycle_task
        if lifecycle is not None and not lifecycle.done():
            lifecycle.cancel()
            try:
                await lifecycle
            except (asyncio.CancelledError, Exception):
                pass
        # The lifecycle task owns these; setting them here covers the case
        # where it was cancelled before reaching its own ``finally``.
        self._phase = "done"
        self._task_done_event.set()
        self._completion_event.set()

    async def _cancel_relay(self) -> None:
        """Cancel the active notification relay task, if any."""
        relay = self._active_relay
        if relay is not None and not relay.done():
            relay.cancel()
            try:
                await relay
            except (asyncio.CancelledError, Exception):
                pass
        self._active_relay = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _run_lifecycle(self) -> None:
        """Manage the two-phase lifecycle: task -> storage check -> done."""
        if self._meter is not None:
            # The librarian's calls are planning tokens of this run.
            current_run_meter.set(self._meter)
        try:
            # ── Phase 1: task execution ───────────────────────────────
            self._active_relay = asyncio.create_task(
                self._relay_notifications_from(self._inner),
            )

            try:
                self._original_result = await self._inner.result()
                task_succeeded = not self._stopped
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                task_succeeded = False
                # Kept so ``result()`` can re-raise. Flattening the failure into
                # a string here is what let a crashed run reach the scheduler
                # looking like a normal return, and be recorded as completed.
                self._task_failure = exc
                self._original_result = (
                    f"Error: inner task failed: {type(exc).__name__}: {exc}"
                )
                logger.error(
                    f"_StorageCheckHandle: inner result raised "
                    f"{type(exc).__name__}: {exc}",
                )
            await self._cancel_relay()
            self._task_done_event.set()

            # Snapshot trajectory and ask tools (client/messages are still
            # valid after result() returns -- cleanup only resets context
            # vars and releases the semaphore).
            trajectory: list[dict] = []
            ask_tools: dict = {}
            try:
                client = getattr(self._inner, "_client", None)
                if client is not None:
                    trajectory = make_messages_safe_for_context_dump(
                        list(getattr(client, "messages", []) or []),
                    )
            except Exception:
                pass
            try:
                _get_ask = getattr(
                    self._inner._task,
                    "get_ask_tools",
                    lambda: {},
                )
                ask_tools = _get_ask()
            except Exception:
                pass
            completed_tool_metadata: dict = {}
            try:
                _get_meta = getattr(
                    self._inner._task,
                    "get_completed_tool_metadata",
                    lambda: {},
                )
                completed_tool_metadata = _get_meta()
            except Exception:
                pass

            # ── Phase 2: storage check ────────────────────────────────
            # A crashed trajectory is not a source of reusable knowledge — the
            # librarian would derive functions, guidance, and claims from work
            # that never completed. A deliberate stop still reviews, because the
            # work up to that point was real.
            if self._task_failure is not None:
                return

            self._phase = "storage"

            # A mid-session turn review still in flight finishes first: its
            # summary joins ``proactive_storage_summaries``, so the final
            # review below builds on it instead of running concurrently
            # against the same trajectory.
            turn_task = self._turn_review_task
            if turn_task is not None and not turn_task.done():
                await asyncio.gather(turn_task, return_exceptions=True)

            _sc_suffix = _token_hex(2)
            _sc_call_id = new_call_id()
            _sc_parent = TOOL_LOOP_LINEAGE.get([])
            _sc_parent_lineage = (
                list(_sc_parent) if isinstance(_sc_parent, list) else []
            )
            _sc_hierarchy = [
                *_sc_parent_lineage,
                f"StorageCheck(CodeActActor.act)({_sc_suffix})",
            ]
            _sc_lineage_token = TOOL_LOOP_LINEAGE.set(_sc_hierarchy)
            _sc_suffix_token = _PENDING_LOOP_SUFFIX.set(_sc_suffix)

            try:
                active_review_context = (
                    self._post_run_review_context if task_succeeded else None
                )
                review_display_label = (
                    active_review_context.display_label
                    if active_review_context is not None
                    else _DEFAULT_STORAGE_REVIEW_LABEL
                )
                review_instructions = (
                    active_review_context.instructions
                    if active_review_context is not None
                    else _DEFAULT_STORAGE_REVIEW_INSTRUCTIONS
                )
                await publish_manager_method_event(
                    _sc_call_id,
                    "CodeActActor",
                    "StorageCheck",
                    phase="incoming",
                    display_label=review_display_label,
                    hierarchy=_sc_hierarchy,
                    instructions=review_instructions,
                )

                proactive_summaries: list[str] = []
                try:
                    _ctx = _CURRENT_AGENT_CONTEXT.get(None)
                    if _ctx is not None:
                        proactive_summaries = list(
                            _ctx.proactive_storage_summaries,
                        )
                except Exception:
                    pass

                storage_handle = _start_storage_check_loop(
                    trajectory=trajectory,
                    ask_tools=ask_tools,
                    completed_tool_metadata=completed_tool_metadata,
                    actor=self._actor,
                    original_result=str(self._original_result),
                    parent_lineage=_sc_parent_lineage,
                    stop_reason=self._stop_reason,
                    proactive_summaries=proactive_summaries or None,
                    post_run_review_context=active_review_context,
                )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )
                else:
                    self._storage_handle = storage_handle
                    storage_success = True
                    try:
                        storage_summary = await self._storage_handle.result()
                    except Exception as exc:
                        storage_success = False
                        storage_summary = (
                            f"StorageCheck failed: {type(exc).__name__}: {exc}"
                        )
                        logger.warning(
                            f"StorageCheck failed: {type(exc).__name__}: {exc}",
                        )

                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )

                    # ponytail: single-consumer signal — see event_handlers.py
                    # ActorNotification handler for the wake gate. Upgrade to
                    # a dedicated event class if a second consumer appears.
                    await self._notification_q.put(
                        {
                            "type": "storage_review_complete",
                            "message": storage_summary,
                            "success": storage_success,
                        },
                    )
            finally:
                _PENDING_LOOP_SUFFIX.reset(_sc_suffix_token)
                TOOL_LOOP_LINEAGE.reset(_sc_lineage_token)

        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._phase = "done"
            self._task_done_event.set()
            self._completion_event.set()

    # ── Steering: phase-aware forwarding ──────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        **kwargs,
    ) -> "SteerableToolHandle":
        # Task and done phases: ask about the completed/running task.
        if self._phase != "storage":
            return await self._inner.ask(
                question,
                _parent_chat_context=_parent_chat_context,
                **kwargs,
            )

        # ── Storage phase: thin routing loop ──────────────────────────
        inner_ref = self._inner
        storage_ref = self._storage_handle
        pcc = _parent_chat_context

        async def ask_about_task(question: str) -> str:
            """Ask a question about the **completed task** itself.

            Use this for anything related to:
            - What the task was and what the agent did to accomplish it
            - The reasoning, tool calls, or intermediate steps taken
            - The final result or output of the task
            - Errors or issues encountered during execution

            This queries the full execution trajectory of the finished
            task, NOT the skill-storage process that is running now.
            """
            h = await inner_ref.ask(question, _parent_chat_context=pcc)
            return await h.result()

        async def ask_about_skill_storage(question: str) -> str:
            """Ask a question about the **ongoing skill storage** process.

            Use this for anything related to:
            - Which functions are being considered for storage
            - What the skill librarian has stored, merged, or deleted so far
            - Progress or status of the skill consolidation review
            - Decisions about whether a function is worth keeping

            This queries the live storage-check loop that is reviewing
            the completed trajectory for reusable patterns, NOT the
            original task itself.
            """
            if storage_ref is not None:
                h = await storage_ref.ask(question, _parent_chat_context=pcc)
                return await h.result()
            return "Skill storage has not started yet."

        routing_tools: Dict[str, Callable] = {
            "ask_about_task": ask_about_task,
            "ask_about_skill_storage": ask_about_skill_storage,
        }

        routing_client = new_llm_client(purpose="planning", origin="StorageCheck.ask")
        routing_client.set_system_message(
            "You are answering a question about an agent that has completed "
            "its primary task and is now reviewing its execution trajectory "
            "to store reusable skills.\n\n"
            "You have two tools:\n"
            "- ask_about_task: for questions about the completed task, its "
            "approach, reasoning, or result\n"
            "- ask_about_skill_storage: for questions about the ongoing "
            "skill consolidation process\n\n"
            "Route the question to the appropriate tool. If the question "
            "spans both topics, call both tools and synthesize the answers.",
        )

        return start_async_tool_loop(
            client=routing_client,
            message=question,
            tools=routing_tools,
            loop_id="Question(StorageCheck.routing)",
        )

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        **kwargs,
    ) -> None:
        handle = self._active_handle
        if handle is not None:
            return await handle.interject(
                message,
                _parent_chat_context_cont=_parent_chat_context_cont,
                **kwargs,
            )

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        self._stopped = True
        self._stop_reason = reason
        handle = self._active_handle
        if handle is not None:
            await handle.stop(reason=reason, **kwargs)

    async def pause(self, **kwargs) -> Optional[str]:
        handle = self._active_handle
        if handle is not None:
            return await handle.pause(**kwargs)
        return None

    async def resume(self, **kwargs) -> Optional[str]:
        handle = self._active_handle
        if handle is not None:
            return await handle.resume(**kwargs)
        return None

    # ── Completion ────────────────────────────────────────────────────

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def wait_until_done(self) -> None:
        """Block until both phases have finished.

        ``result()`` deliberately resolves at the end of phase 1 so callers are
        not made to wait on a review they did not ask for. A caller that owns
        the actor's lifetime needs the other guarantee -- that the review has
        finished before the resources it runs on are torn down -- and this is
        how it waits for it.
        """

        await self._completion_event.wait()

    async def result(self) -> str:
        await self._task_done_event.wait()
        # An explicit stop is an outcome the caller asked for, so it reports as
        # one even if the actor then raised on its way down.
        if self._stopped and self._stop_reason:
            return (
                f"Task stopped as requested. Reason: {self._stop_reason}\n"
                f"Background skill storage is reviewing the completed work."
            )
        # Re-raise rather than returning the flattened error string: callers
        # decide success from whether this raises, so swallowing it here makes a
        # crashed run indistinguishable from one that returned normally.
        if self._task_failure is not None:
            raise self._task_failure
        return self._original_result or ""

    # ── Events ────────────────────────────────────────────────────────

    async def next_clarification(self) -> dict:
        handle = self._active_handle
        if handle is not None:
            return await handle.next_clarification()
        # Done: block forever (no more clarifications expected).
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        handle = self._active_handle
        if handle is not None:
            return await handle.answer_clarification(call_id, answer)

    def get_history(self) -> list[dict]:
        return self._inner.get_history()


# ---------------------------------------------------------------------------
# Code synthesis helpers for execute_function
# ---------------------------------------------------------------------------


def _synthesize_python_call(
    *,
    function_name: str,
    call_kwargs: Dict[str, Any],
    function_manager: Optional["FunctionManager"] = None,
) -> str:
    """Build a Python code snippet that calls *function_name* with *call_kwargs*.

    This function only synthesises the **code string** — it does not handle
    dependency injection.  Transitive dependencies (both bare compositional
    functions and dotted environment namespaces like ``actor`` or
    ``primitives``) are injected into the sandbox namespace *before* this
    code runs, through a separate path:

    * When the LLM discovers a function via ``FunctionManager_search_functions``
      (or filter/list), the FM's ``_inject_callables_for_functions`` calls
      ``_inject_dependencies``, which resolves every entry in ``depends_on``
      and places the result into the sandbox's ``global_state``.
    * Environment namespaces (``actor``, ``primitives``, etc.) are also
      already present in the sandbox if the CodeActActor was constructed
      with the corresponding environments.

    So by the time this synthesised code executes, all names the function
    references — whether bare helpers or dotted environment calls — are
    already available in scope.

    Resolution order for the *function itself*:
    1. Emit a plain call expression.  The sandbox namespace already contains
       environment-injected callables and previously-discovered FM functions,
       so this is the common-case fast path.
    2. If the FunctionManager has a stored implementation, prepend it as a
       preamble (defining the function) so the call works even in a fresh
       stateless session where discovery hasn't run yet.

    The call expression is always the **last expression** so that
    ``PythonExecutionSession``'s REPL semantics return its value (including
    steerable handles from primitives).
    """
    kwargs_repr = repr(call_kwargs) if call_kwargs else "{}"
    # Determine whether the function is async by inspecting the stored impl.
    # Default to ``await`` — environment-injected callables (primitives,
    # manager methods) are async, and ``await`` on a sync return value
    # produces a clear ``TypeError`` rather than silently discarding a
    # coroutine.
    is_async = True
    preamble = ""

    if function_manager is not None:
        func_data = function_manager._get_function_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            func_data = function_manager._get_primitive_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            get_stored_primitive = getattr(
                function_manager,
                "_get_stored_primitive_data_by_name",
                None,
            )
            if callable(get_stored_primitive):
                func_data = get_stored_primitive(name=function_name)

        if func_data is not None:
            impl = func_data.get("implementation")
            if impl and isinstance(impl, str) and impl.strip():
                is_async = "async def" in impl
                # Strip @custom_function decorators (not available in sandbox).
                from unify.function_manager.function_manager import (
                    _strip_custom_function_decorators,
                )

                preamble = _strip_custom_function_decorators(impl) + "\n\n"
            elif func_data.get("is_primitive"):
                is_async = True

    call_expr = f"{'await ' if is_async else ''}{function_name}(**{kwargs_repr})"
    return f"{preamble}{call_expr}"


def _synthesize_shell_call(
    *,
    function_name: str,
    call_kwargs: Dict[str, Any],
    function_manager: Optional["FunctionManager"] = None,
) -> str:
    """Build a shell script that runs the stored function with *call_kwargs*.

    Shell functions must have a stored implementation in the FunctionManager.
    ``call_kwargs`` are exported as environment variables before sourcing the
    implementation.
    """
    impl: str | None = None
    if function_manager is not None:
        func_data = function_manager._get_function_data_by_name(name=function_name)
        if func_data is not None:
            impl = func_data.get("implementation")

    if not impl or not isinstance(impl, str) or not impl.strip():
        raise ValueError(
            f"Shell function '{function_name}' has no stored implementation.",
        )

    # Export kwargs as environment variables.
    exports: list[str] = []
    for k, v in (call_kwargs or {}).items():
        escaped = str(v).replace("'", "'\\''")
        exports.append(f"export {k}='{escaped}'")

    parts = exports + [impl]
    return "\n".join(parts)


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
        knowledge_manager: Optional["KnowledgeManager"] = None,
        workflow_manager: Optional["WorkflowManager"] = None,
        can_compose: object = _UNSET,
        can_store: object = _UNSET,
        timeout: object = _UNSET,
        model: object = _UNSET,
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
        prompt_caching: object = _UNSET,
        guidelines: object = _UNSET,
        tool_policy: Union[ToolPolicyFn, None, object] = _USE_DEFAULT,
    ):
        """
        Initializes the CodeActActor.

        Args:
            environments: List of execution environments to install. Each environment
                injects a namespace into the sandbox (e.g. ``primitives``,
                ``primitives.computer``, ``primitives.actor``). Pass ``None`` or ``[]``
                for a bare actor with no environments.
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
            guidance_manager: Manages high-level guidance entries that describe *how* to
                compose functions together for tasks. Exposes read/write tools in the
                post-completion storage check loop alongside FunctionManager tools.
            knowledge_manager: Manages durable sourced knowledge claims (the *is*).
                Exposes JSON CRUD/lifecycle tools on the main loop and in the
                post-completion storage check loop when present.
            workflow_manager: Catalogue of installable workflow bundles. Exposes
                JSON install/uninstall/list/get tools on the main loop when the
                curated catalogue is configured; absent otherwise, so the tool
                schema is unchanged for deployments without the shelf.
            can_compose: Whether the LLM can write and execute arbitrary code via
                ``execute_code``. Set to False for function-execution-only mode.
            can_store: Whether a post-completion review loop should run to
                identify and store reusable functions and guidance from the
                trajectory. Storage is always deferred to a dedicated second
                loop after the main task completes — the main loop never
                exposes storage tools.
            timeout: Maximum seconds for individual code execution in sessions.
            model: Optional LLM model identifier. If None, uses the assistant's
                default model when set, otherwise SETTINGS.UNIFY_MODEL.
            preprocess_msgs: Optional callback to modify messages before each LLM call.
                Receives a list of message dicts and returns a modified list.
                Useful for pruning old messages, adding context, or transforming content.
            prompt_caching: Optional list of cache targets (e.g. ["system", "messages"]).
                Enables Anthropic prompt caching for the specified components to reduce
                costs and latency. Valid values: "tools", "system", "messages".
            guidelines: Persistent behavioral guidelines applied to every ``act()``
                invocation.  Per-invocation ``guidelines`` passed to ``act()`` are
                appended after these, so the constructor value acts as a baseline
                and ``act()`` adds task-specific refinements on top.
            tool_policy: Controls per-turn dynamic tool filtering and tool-choice mode.
                - ``_USE_DEFAULT`` (default): uses the built-in "discovery-first"
                  policy that requires both a FunctionManager and a GuidanceManager
                  discovery call before unlocking the full tool set.
                - A custom ``ToolPolicyFn`` callable: receives ``(step, tools)`` and
                  returns ``(mode, filtered_tools)``.  Static filters (``can_compose``,
                  ``can_store``, etc.) are always applied before the custom policy sees
                  the tools.
                - ``None``: no dynamic policy; only the static ``can_compose`` /
                  ``can_store`` filters apply.
        """
        super().__init__(
            environments=environments or [],
            function_manager=function_manager,
            guidance_manager=guidance_manager,
            knowledge_manager=knowledge_manager,
            workflow_manager=workflow_manager,
        )

        can_compose = can_compose if can_compose is not _UNSET else True
        can_store = can_store if can_store is not _UNSET else True
        timeout = timeout if timeout is not _UNSET else 3600.0
        model = model if model is not _UNSET else None
        prompt_caching = (
            prompt_caching
            if prompt_caching is not _UNSET
            else ("system", "tools", "messages")
        )
        guidelines = guidelines if guidelines is not _UNSET else None
        self._base_guidelines = guidelines

        # Collect function_ids from all environments, split by context, and set
        # them on the FunctionManager via setters. This prevents overlap between
        # prompt-injected environment tools and FunctionManager-discoverable
        # functions. We update in-place rather than replacing the FM instance so
        # that callers who pass a custom FM (e.g., SimulatedFunctionManager) keep
        # their instance intact.
        if self.function_manager is not None:
            _excl_primitive: set[int] = set()
            _excl_compositional: set[int] = set()
            for env in self.environments.values():
                # The exclusion deduplicates search results against what the
                # prompt documents. When an environment declares
                # `prompt_documented_names`, only that subset is excluded —
                # undocumented primitives must stay searchable. State manager
                # primitives declare an empty set (their method docs are not
                # inlined), so core methods like `ask`/`update` are
                # searchable; computer-control tools remain excluded because
                # their name index stays in the prompt.
                _documented = getattr(env, "prompt_documented_names", None)
                for tool_name, tool_meta in env.get_tools().items():
                    if tool_meta.function_id is not None:
                        if _documented is not None and tool_name not in _documented:
                            continue
                        if tool_meta.function_context == "primitive":
                            _excl_primitive.add(tool_meta.function_id)
                        elif tool_meta.function_context == "compositional":
                            _excl_compositional.add(tool_meta.function_id)

            if _excl_primitive:
                self.function_manager.exclude_primitive_ids = frozenset(
                    _excl_primitive,
                )
            if _excl_compositional:
                self.function_manager.exclude_compositional_ids = frozenset(
                    _excl_compositional,
                )
            try:
                from unify.integration_status import build_function_filter_scope

                function_scope = build_function_filter_scope()
                if function_scope:
                    current = getattr(self.function_manager, "filter_scope", None)
                    self.function_manager.filter_scope = (
                        f"({current}) and ({function_scope})"
                        if current
                        else function_scope
                    )
            except Exception:
                pass

        if self.guidance_manager is not None:
            try:
                from unify.integration_status import build_guidance_filter_scope

                guidance_scope = build_guidance_filter_scope()
                if guidance_scope:
                    current = getattr(self.guidance_manager, "filter_scope", None)
                    self.guidance_manager.filter_scope = (
                        f"({current}) and ({guidance_scope})"
                        if current
                        else guidance_scope
                    )
            except Exception:
                pass

        # Create persistent pools that survive across act() calls
        from unify.function_manager.function_manager import VenvPool
        from unify.function_manager.shell_pool import ShellPool

        self._venv_pool = VenvPool()
        self._shell_pool = ShellPool()
        self._session_executor = SessionExecutor(
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
            environments=self.environments,
            computer_primitives=self._computer_primitives,
            function_manager=self.function_manager,
            timeout=timeout,
        )

        # Session name registry: name -> (language, venv_id, session_id)
        self._session_names: Dict[str, SessionKey] = {}
        # Reverse map: (language, venv_id, session_id) -> set(names)
        self._session_names_rev: Dict[SessionKey, set[str]] = {}
        # Actor-level session cap (global across languages for this actor instance).
        self._max_sessions_total: int = 20
        self._next_session_id: dict[tuple[str, Optional[int]], int] = {}
        # Storage reviews started by this actor and not yet finished, so
        # ``close()`` can end them rather than leave them running against
        # pools it is about to tear down.
        self._live_storage_handles: "weakref.WeakSet[_StorageCheckHandle]" = (
            weakref.WeakSet()
        )

        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self.tool_policy: Union[ToolPolicyFn, None, object] = tool_policy
        self._model = model
        self._preprocess_msgs = preprocess_msgs
        self._prompt_caching = prompt_caching
        self._computer_tools = (
            self._get_computer_tools()
        )  # Register stable tools once; per-call sandboxes are bound via _CURRENT_SANDBOX.
        self.add_tools("act", self._build_tools())

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Concurrency guard: limit active sandboxes per actor instance.
        self._act_semaphore = asyncio.Semaphore(20)
        # Timeout used when acquiring the semaphore (prevents unbounded waits).
        self._act_semaphore_timeout_s: float = 30.0
        self._active_work_heartbeat_interval_s: float = 60.0
        self._active_work_fallback_initial_delay_s: float = 120.0
        self._active_work_fallback_repeat_interval_s: float = 300.0

    # ───────────────────────── Session name registry ─────────────────────── #

    def _register_session_name(
        self,
        *,
        name: str,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> None:
        key: SessionKey = (language, venv_id, int(session_id))
        existing = self._session_names.get(name)
        if existing is not None and existing != key:
            raise ValueError(
                f"Session name {name!r} is already bound to {existing}, cannot rebind to {key}.",
            )
        self._session_names[name] = key
        self._session_names_rev.setdefault(key, set()).add(name)

    def _resolve_session_name(self, name: str) -> SessionKey | None:
        return self._session_names.get(name)

    def _get_session_name(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> str | None:
        key: SessionKey = (language, venv_id, int(session_id))
        names = self._session_names_rev.get(key)
        if not names:
            return None
        # Prefer stable ordering for determinism.
        return sorted(names)[0]

    def _unregister_session_name(self, name: str) -> None:
        key = self._session_names.pop(name, None)
        if key is None:
            return
        names = self._session_names_rev.get(key)
        if names is not None:
            names.discard(name)
            if not names:
                self._session_names_rev.pop(key, None)

    def _unregister_all_names_for_session(self, *, key: SessionKey) -> None:
        names = self._session_names_rev.pop(key, None)
        if not names:
            return
        for n in list(names):
            self._session_names.pop(n, None)

    def _count_active_sessions_total(self) -> int:
        # Count unique in-process python sessions + persistent pool sessions.
        n = 0
        try:
            n += len(
                self._session_executor._python_sessions,
            )  # pylint: disable=protected-access
        except Exception:
            pass
        try:
            n += len(self._shell_pool.get_active_sessions())
        except Exception:
            pass
        try:
            n += len(self._venv_pool.list_active_sessions())
        except Exception:
            pass
        return n

    def _session_exists(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> bool:
        if language == "python":
            if venv_id is None:
                return self._session_executor.has_python_session(
                    session_id=int(session_id),
                    venv_id=None,
                )
            # venv-backed python session exists if pool has it active
            try:
                return (int(venv_id), int(session_id)) in set(
                    self._venv_pool.list_active_sessions(),
                )
            except Exception:
                return False
        # shell
        try:
            return self._shell_pool.has_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
        except Exception:
            return False

    def _validate_execution_params(
        self,
        *,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
        language: str,
        venv_id: int | None = None,
    ) -> dict | None:
        return _validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
            resolve_session_name=self._resolve_session_name,
            get_session_name_for_id=lambda l, v, s: self._get_session_name(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            session_exists=lambda l, v, s: self._session_exists(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            venv_exists=self._venv_exists,
            max_sessions_total=self._max_sessions_total,
            active_session_count=self._count_active_sessions_total(),
        )

    def _venv_exists(self, venv_id: int) -> bool:
        """Whether *venv_id* names a venv this actor can execute in.

        Venvs are owned by the FunctionManager, so without one there is nothing
        to check against — the venv-backed path rejects the call on its own.
        """
        if self.function_manager is None:
            return True
        return self.function_manager.get_venv(venv_id=venv_id) is not None

    def _resolve_session(
        self,
        *,
        state_mode: str,
        language: str,
        session_id: int | None,
        session_name: str | None,
        venv_id: int | None,
    ) -> _ResolvedSession:
        """Resolve/allocate a session and validate execution params.

        Handles the full session resolution flow used by both ``execute_code``
        and ``execute_function``:

        1. For stateful mode: resolve an existing session name, allocate a new
           session id, or default to session 0.
        2. Register session name aliases when both name and id are provided.
        3. Validate the resulting execution parameters.

        Returns a ``_ResolvedSession`` named tuple.  If ``error`` is not
        ``None``, the caller should return it as the tool result immediately.
        """
        # Resolve / allocate sessions for stateful.
        if state_mode == "stateful":
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is not None:
                    language, venv_id, session_id = resolved
                elif session_id is None:
                    key = (
                        str(language),
                        int(venv_id) if venv_id is not None else None,
                    )
                    next_id = self._next_session_id.get(key, 1)
                    session_id = next_id
                    self._next_session_id[key] = next_id + 1
                    self._register_session_name(
                        name=session_name,
                        language=str(language),
                        venv_id=venv_id,
                        session_id=int(session_id),
                    )
            elif session_id is None:
                session_id = 0

        # If name + id are both set but not registered yet, register alias.
        if state_mode == "stateful" and session_name and session_id is not None:
            if self._resolve_session_name(session_name) is None:
                self._register_session_name(
                    name=session_name,
                    language=str(language),
                    venv_id=venv_id,
                    session_id=int(session_id),
                )

        # Refuses by raising if the parameters cannot be executed as given.
        self._validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=str(language),
            venv_id=venv_id,
        )

        return _ResolvedSession(
            language=str(language),
            venv_id=venv_id,
            session_id=session_id,
        )

    async def _execute_on_surface(
        self,
        *,
        surface_name: str,
        code: str,
        language: str,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
        venv_id: int | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        """Run code on a non-local surface (assistant desktop or user desktop).

        Remote surfaces are stateless one-shots: sessions and venvs are
        local-only concepts, so a session/venv request is rejected with a
        structured error the model can self-correct against, rather than being
        silently ignored.
        """
        import time as _surface_time

        from unify.actor.execution.surface import ExecutionSurface
        from unify.actor.execution.targets import (
            TargetUnavailableError,
            get_target,
        )

        def _err(message: str, suggestion: str) -> dict[str, Any]:
            return {
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": message,
                "suggestion": suggestion,
                "language": language,
                "state_mode": state_mode,
                "session_id": None,
                "session_name": None,
                "venv_id": None,
                "session_created": False,
                "duration_ms": 0,
                "surface": surface_name,
            }

        try:
            surface = ExecutionSurface(surface_name)
        except ValueError:
            return _err(
                f"Unknown surface: {surface_name!r}",
                "Use one of: 'local', 'assistant_desktop', 'user_desktop'.",
            )

        if (
            state_mode != "stateless"
            or session_id is not None
            or session_name is not None
            or venv_id is not None
        ):
            return _err(
                f"Surface {surface_name!r} supports only stateless execution.",
                "Remove state_mode/session_id/session_name/venv_id (remote "
                "surfaces are stateless), or use surface='local' for sessions "
                "and venvs.",
            )

        t0 = _surface_time.perf_counter()
        try:
            target = get_target(
                surface,
                user_id=user_id,
                session_executor=self._session_executor,
                function_manager=self.function_manager,
            )
            await target.ensure_ready()
            if language == "python":
                res = await target.run_python(code)
            else:
                res = await target.run_shell(code)
        except TargetUnavailableError as e:
            return _err(
                str(e),
                "Check that the desktop is linked, reachable, and (for the "
                "user desktop) that the user has granted access.",
            )
        except ValueError as e:
            return _err(
                str(e),
                "Adjust the request to match the surface's capabilities.",
            )

        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "result": res.result if res.result is not None else res.returncode,
            "error": res.error,
            "returncode": res.returncode,
            "language": language,
            "state_mode": "stateless",
            "session_id": None,
            "session_name": None,
            "venv_id": None,
            "session_created": False,
            "duration_ms": int((_surface_time.perf_counter() - t0) * 1000),
            "surface": surface_name,
        }

    def _get_computer_tools(self) -> Dict[str, Callable]:
        """Extracts computer-related methods from the desktop namespace."""
        if not self._computer_primitives:
            return {}
        desktop = self._computer_primitives.desktop
        return {
            "navigate": desktop.navigate,
            "act": desktop.act,
            "observe": desktop.observe,
        }

    def _get_extra_ask_tools(self) -> Dict[str, Callable] | None:
        """Build domain-specific ask tools for handle.ask() inspection loops."""
        if self._computer_primitives is None:
            return None

        # Only offer the computer-progress probe when the managed desktop is
        # actually in use. Constructing ComputerPrimitives is unconditional in
        # deployments (container_url=None, no VM), and without a live session
        # ``desktop.query`` blocks on ``_vm_ready.wait(300)`` for five minutes
        # before failing — an inspection loop whose only progress tool is a
        # guaranteed 300s dead-end. ``has_live_desktop_session`` is missing on
        # older/mocked primitives; treat absence as live to preserve behaviour.
        _is_live = getattr(
            self._computer_primitives,
            "has_live_desktop_session",
            None,
        )
        if callable(_is_live) and not _is_live():
            return None

        computer_query = self._computer_primitives.desktop.query

        async def ask_computer_progress(
            question: str,
            *,
            _parent_chat_context: list[dict] | None = None,
        ) -> str:
            """Inspect ONLY the magnitude browser/computer agent's trajectory.

            Scope: this reads the browser-agent's own memory of in-flight
            ``session.act(...)`` / desktop work. It knows nothing about code
            execution, shell commands, API calls, file syncs, or LLM steps —
            all of those run outside the computer agent, so do NOT call this
            for them. Use it only when the inspected transcript shows an
            ongoing browser/desktop action whose detail is missing (for
            example, placeholders or terse summaries). This is memory/history
            introspection, not a fresh page read and not a way to trigger new
            actions.
            """
            _ = _parent_chat_context
            # Even with the live-session gate above, the backend can become
            # unreachable mid-flight. A read-only progress probe failing must
            # not look like an error: degrade to a plain answer so it cannot
            # burn the inspection loop's failure budget or be mistaken for the
            # inspected task failing.
            try:
                return await computer_query(question)
            except Exception as exc:
                return (
                    "No computer-agent progress is available to inspect "
                    f"({type(exc).__name__}). There may be no active browser/"
                    "computer session for this work (for example, a file sync "
                    "or shell command runs outside the computer agent). This is "
                    "not a failure of the underlying task."
                )

        return {"ask_computer_progress": ask_computer_progress}

    async def _run_active_work_heartbeat(
        self,
        active_work: ActiveWorkHandle,
        notification_q: asyncio.Queue[dict] | None,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(self._active_work_heartbeat_interval_s)
                active_work.heartbeat()
                if (
                    notification_q is not None
                    and active_work.fallback_notification_due(
                        initial_delay_s=self._active_work_fallback_initial_delay_s,
                        repeat_interval_s=self._active_work_fallback_repeat_interval_s,
                    )
                ):
                    await notification_q.put(
                        {
                            "type": "notification",
                            "message": "Still working on the code step...",
                            "source": "active_work",
                            "completed": False,
                            "active_work_id": active_work.work_id,
                        },
                    )
                    active_work.record_fallback_notification()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _sandbox_call_binding(
        *,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        interject_q: asyncio.Queue | None = None,
        notification_q: asyncio.Queue | None = None,
        pause_event: asyncio.Event | None = None,
    ):
        """Bind one tool call's channels onto the live sandbox.

        Two channels, both per-call and both restored on exit:

        * clarification queues, so nested manager clarifications write into the
          outer tool's ``clar_up_queue`` (mailbox A) watched by the async tool
          loop
        * a :class:`SteeringChannel`, so checkpoints inside the running block
          can observe interjections aimed at this call and suspend for a
          decision

        Yields the steering channel so the caller can report progress once
        execution ends, however it ended.
        """
        from contextlib import contextmanager

        from unify.actor.environments.base import (
            bind_sandbox_clarification_queues,
            restore_sandbox_clarification_queues,
        )
        from unify.function_manager.steering import SteeringSession, use_session
        from unify.function_manager.steering_patcher import build_patch_author

        @contextmanager
        def _binding():
            try:
                sb = _CURRENT_SANDBOX.get()
            except Exception:
                yield None
                return

            clar_token = None
            if clarification_up_q is not None and clarification_down_q is not None:
                clar_token = bind_sandbox_clarification_queues(
                    sb.global_state,
                    clarification_up_q,
                    clarification_down_q,
                )

            # The patch author only matters when there is a channel to be
            # corrected through; without one nothing can interrupt, so the
            # LLM client is never built.
            steering = SteeringSession(
                interject_q=interject_q,
                notification_q=notification_q,
                patch_author=build_patch_author() if interject_q is not None else None,
                pause_event=pause_event,
            )
            # Carried by context rather than installed on this sandbox:
            # stateless cells build a fresh sandbox per call that would
            # never see anything installed here.
            try:
                with use_session(steering):
                    yield steering
            finally:
                if clar_token is not None:
                    restore_sandbox_clarification_queues(sb.global_state, clar_token)

        return _binding()

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        @llm_soft_required(thought="")
        async def execute_code(
            thought: Annotated[
                str,
                "A brief, first-person, one-sentence explanation of what this "
                'code does and why you are running it right now (e.g. "Loading '
                'the data and computing the summary the user asked for."). Shown '
                "to the user as the rationale for this step; always provide it.",
            ],
            code: Optional[str] = None,
            *,
            language: str,
            state_mode: str | None = None,
            session_id: int | None = None,
            session_name: str | None = None,
            venv_id: int | None = None,
            surface: str = "local",
            user_id: str | None = None,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _clarification_up_q: asyncio.Queue[str] | None = None,
            _clarification_down_q: asyncio.Queue[str] | None = None,
            _interject_queue: asyncio.Queue | None = None,
            _pause_event: asyncio.Event | None = None,
            _parent_chat_context: list[dict] | None = None,
        ) -> Any:
            """
            Execute arbitrary code in a specified language and state mode.

            **IMPORTANT — single-call rule**: If the task requires only a
            single function or primitive call with no surrounding logic,
            use ``execute_function`` instead. ``execute_code`` is for
            **multi-step composition** — conditional logic, loops, or
            combining multiple primitives/functions where intermediate
            results are needed within the same code block.

            Key concepts
            -----------
            - **language**: "python" | "bash" | "zsh" | "sh" | "powershell"
            - **surface**: "local" (default; the only surface with stateful
              sessions and venvs), "assistant_desktop" (managed VM),
              "user_desktop" (the user's own linked machine; pass
              ``user_id`` when more than one is linked). Remote surfaces are
              **stateless one-shots**: ``state_mode`` must be "stateless"
              and ``session_id`` / ``session_name`` / ``venv_id`` omitted.
            - **state_mode**: omit it and a local venv-less Python cell
              runs **stateful in session 0** — the current per-call
              Python sandbox, so variables persist across cells — while
              shell, venv, and remote cells run stateless. Pass
              "stateless" for an isolated fresh run (environment globals
              and FunctionManager-discovered functions still available),
              "read_only" to read an existing session without
              persisting, or "stateful" with a session selector to
              target a named or shell/venv session.
            - **session_id/session_name**: stateful/read_only only.
              Stateful defaults to **session_id=0** — inside a running
              act() loop, the current per-call Python sandbox. Create an
              additional session with a fresh ``session_name`` (recommended)
              or an explicit ``session_id`` > 0; choose via
              ``list_sessions()`` / ``inspect_state()``.

            Output
            ------
            A dict or ExecutionResult with: ``stdout`` / ``stderr`` (rich
            List[TextPart | ImagePart] for in-process Python; plain string
            for shell/venv), ``result`` (last expression's value — a
            steerable handle as the last expression is automatically adopted
            by the outer loop for mid-flight steering), ``error``,
            ``language``, ``state_mode``, ``session_id``, ``session_name``,
            ``venv_id``, ``session_created``, ``duration_ms``.

            Runtime credential helpers
            --------------------------
            Python globals include ``get_oauth_access_token(provider)`` for
            connected-account OAuth: a local capability handle (not a raw
            token) used with the workspace proxy base URLs — see
            ``help(get_oauth_access_token)``. Static API keys stay in
            ``os.environ``.

            Steering while the block runs
            -----------------------------
            Python blocks are steerable in flight: checkpoints sit between
            top-level statements, at the top of every loop body, and before
            every ``primitives.*`` call. On a correction the block suspends
            and you get a turn with a progress report:
            ``stop_execute_code_<call_id>`` abandons the block (choose when
            the correction changes the remaining work); interjecting again
            resumes it as written. Generated code may read
            ``steering.messages`` to adapt without being abandoned. A
            checkpoint only runs when the block yields — synchronous
            blocking calls hold execution, so prefer async calls in work
            that may need correcting partway through.
            """
            _ = thought  # Thought is logged by the LLM; not used programmatically.
            if state_mode is None:
                # An omitted state_mode resolves per cell type: only local
                # venv-less Python cells get the persistent per-call sandbox.
                state_mode = (
                    "stateful"
                    if surface == "local" and language == "python" and venv_id is None
                    else "stateless"
                )
            if code is None or code.strip() == "":
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": None,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": 0,
                    "surface": surface,
                }

            # ──────────────────────────────────────────────────────────────
            # Boundary wrapper: execute_code (lineage + events + terminal log)
            # ──────────────────────────────────────────────────────────────

            _suffix = _token_hex(2)
            _call_id = new_call_id()
            _parent = TOOL_LOOP_LINEAGE.get([])
            _parent_lineage = list(_parent) if isinstance(_parent, list) else []
            _hierarchy = [*_parent_lineage, f"execute_code({_suffix})"]
            # Establish a boundary lineage frame so nested calls (e.g., FunctionManager-injected
            # functions calling state managers) keep a consistent parent->child chain.
            _lineage_token = TOOL_LOOP_LINEAGE.set(_hierarchy)

            async def _pub_safe(**payload: Any) -> None:
                try:
                    await publish_manager_method_event(
                        _call_id,
                        "CodeActActor",
                        "execute_code",
                        hierarchy=_hierarchy,
                        display_label="Running code",
                        **payload,
                    )
                except Exception as e:
                    log_boundary_event(
                        "->".join(_hierarchy),
                        f"Warning: failed to publish event: {type(e).__name__}: {e}",
                        icon="⚠️",
                        level="warning",
                    )

            try:
                await _pub_safe(phase="incoming")
            except Exception:
                pass
            log_boundary_event("->".join(_hierarchy), "Executing code...", icon="🛠️")

            out: dict[str, Any] | None = None
            tb_str: str | None = None
            exec_exc: Exception | None = None

            active_work = ACTIVE_WORK.begin(
                label="execute_code",
                metadata={
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "venv_id": venv_id,
                    "thought": thought[:500],
                },
            )
            heartbeat_task: asyncio.Task[None] | None = None
            try:
                heartbeat_task = asyncio.create_task(
                    self._run_active_work_heartbeat(active_work, _notification_up_q),
                )
                notification_q = (
                    _ActiveWorkNotificationQueue(_notification_up_q, active_work)
                    if _notification_up_q is not None
                    else None
                )
                try:
                    from unify.manager_registry import ManagerRegistry

                    # Keep generated code's normal environment-based credential
                    # path fresh at the execution boundary.  The SecretManager
                    # gate is debounced, so repeated execute_code calls only pay
                    # a cheap timestamp check within the TTL window.
                    ManagerRegistry.get_secret_manager().sync_assistant_secrets_if_stale(
                        ttl_seconds=60.0,
                        reason="execute_code",
                    )
                except Exception:
                    logger.warning(
                        "execute_code assistant secret sync failed",
                        exc_info=True,
                    )

                # Route non-local surfaces (assistant/user desktop) through the
                # execution targets. Remote surfaces are stateless, so they skip
                # the local session-resolution and pool machinery entirely.
                if surface != "local":
                    out = await self._execute_on_surface(
                        surface_name=surface,
                        code=code,
                        language=str(language),
                        state_mode=state_mode,
                        session_id=session_id,
                        session_name=session_name,
                        venv_id=venv_id,
                        user_id=user_id,
                    )
                    return out

                _rs = self._resolve_session(
                    state_mode=state_mode,
                    language=str(language),
                    session_id=session_id,
                    session_name=session_name,
                    venv_id=venv_id,
                )
                language, venv_id, session_id = (
                    _rs.language,
                    _rs.venv_id,
                    _rs.session_id,
                )
                # Execute via SessionExecutor. Route primitives if available in current sandbox.
                primitives = None
                computer_primitives = self._computer_primitives
                try:
                    sb = _CURRENT_SANDBOX.get()
                    primitives = sb.global_state.get("primitives")
                except Exception:
                    pass

                _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                _steering = None
                try:
                    with self._sandbox_call_binding(
                        clarification_up_q=_clarification_up_q,
                        clarification_down_q=_clarification_down_q,
                        interject_q=_interject_queue,
                        notification_q=notification_q,
                        pause_event=_pause_event,
                    ) as _steering:
                        try:
                            out = await self._session_executor.execute(
                                code=code,
                                language=str(language),  # type: ignore[arg-type]
                                state_mode=state_mode,  # type: ignore[arg-type]
                                session_id=session_id,
                                venv_id=venv_id,
                                primitives=primitives,
                                computer_primitives=computer_primitives,
                            )
                        except Exception as e:
                            exec_exc = e
                            tb = traceback.format_exc()
                            tb_str = tb
                            out = {
                                "stdout": "",
                                "stderr": "",
                                "result": None,
                                "error": tb,
                                "language": language,
                                "state_mode": state_mode,
                                "session_id": session_id,
                                "session_name": session_name,
                                "venv_id": venv_id,
                                "session_created": False,
                                "duration_ms": 0,
                            }
                finally:
                    _PARENT_CHAT_CONTEXT.reset(_pcc_token)

                # Only when something actually steered this block, or when it
                # failed — an uninterrupted success reports nothing, so the
                # common case costs no transcript weight.
                #
                # On failure the report earns its place: instrumentation shifts
                # the line numbers in a "<string>" traceback, which carries no
                # source text to cross-reference, whereas the checkpoint's
                # ``last_line_reached`` is in the coordinates of the code as
                # written.
                if _steering is not None and (_steering.messages or out.get("error")):
                    out["steering"] = _steering.progress()

                # Enrich with session name.
                if out.get("session_id") is not None:
                    out["session_name"] = self._get_session_name(
                        language=str(out.get("language")),
                        venv_id=out.get("venv_id"),
                        session_id=int(out["session_id"]),
                    )
                else:
                    out["session_name"] = None

                # Wrap in-process Python results in ExecutionResult for proper LLM
                # image formatting. In-process Python has stdout as List[OutputPart];
                # venv/shell have strings.
                if out.get("language") == "python" and isinstance(
                    out.get("stdout"),
                    list,
                ):
                    out = ExecutionResult(**out)

                return out
            finally:
                active_work.end()
                if heartbeat_task is not None and not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    _out_err = (
                        (
                            out.get("error")
                            if isinstance(out, dict)
                            else getattr(out, "error", None)
                        )
                        if out is not None
                        else None
                    )
                    if _out_err:
                        await _pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(_out_err),
                            error_type=(
                                type(exec_exc).__name__
                                if exec_exc is not None
                                else "Error"
                            ),
                            traceback=(tb_str or "")[:2000],
                        )
                    else:
                        await _pub_safe(phase="outgoing", status="ok")
                except Exception:
                    pass
                try:
                    TOOL_LOOP_LINEAGE.reset(_lineage_token)
                except Exception:
                    pass

        # ───────────────────────── Package installation tool ────────────────── #

        async def install_python_packages(
            packages: list[str],
        ) -> dict:
            try:
                sb = _CURRENT_SANDBOX.get()
                overlay: PackageOverlay | None = sb.global_state.get(
                    "__package_overlay__",
                )
            except Exception:
                overlay = None

            if overlay is None:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": (
                        "Package installation is not available outside of an "
                        "active act() session."
                    ),
                    "packages": list(packages),
                }

            return overlay.install(packages)

        install_python_packages.__doc__ = _INSTALL_PYTHON_PACKAGES_DOC

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_code": ToolSpec(fn=execute_code),
            "install_python_packages": ToolSpec(
                fn=install_python_packages,
                display_label="Installing Python packages",
            ),
        }

        # FunctionManager read tools: thin wrappers that inject callables
        # into the sandbox and return only metadata to the LLM. Docstrings
        # are inherited from the base class (the single source of truth).
        if self.function_manager:

            async def FunctionManager_search_functions(
                query: str = "",
                n: int = 5,
                include_implementations: bool = True,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.search_functions(
                    query=query,
                    n=n,
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return result["metadata"]

            FunctionManager_search_functions.__doc__ = (
                BaseFunctionManager.search_functions.__doc__
            )

            async def FunctionManager_filter_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
                include_implementations: bool = True,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.filter_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return strip_ledger_internals(result["metadata"])

            FunctionManager_filter_functions.__doc__ = (
                BaseFunctionManager.filter_functions.__doc__
            )

            async def FunctionManager_list_functions(
                include_implementations: bool = False,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.list_functions(
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return result["metadata"]

            FunctionManager_list_functions.__doc__ = (
                BaseFunctionManager.list_functions.__doc__
            )

            tools["FunctionManager_search_functions"] = ToolSpec(
                fn=FunctionManager_search_functions,
                display_label="Searching for relevant skills",
            )
            tools["FunctionManager_filter_functions"] = ToolSpec(
                fn=FunctionManager_filter_functions,
                display_label="Filtering saved skills",
            )
            tools["FunctionManager_list_functions"] = ToolSpec(
                fn=FunctionManager_list_functions,
                display_label="Listing existing skills",
            )

            fm = self.function_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=fm.add_functions,
                        display_label="Adding functions to the library",
                    ),
                    ToolSpec(
                        fn=fm.delete_function,
                        display_label="Deleting functions from the library",
                    ),
                    ToolSpec(
                        fn=fm.reconcile_dependencies,
                        display_label="Checking function dependencies",
                    ),
                    include_class_name=True,
                ),
            )

        # FunctionManager read tools (search/filter/list) use custom wrappers
        # that inject callables into the sandbox. All other FM/GM tools below
        # are plain CRUD with no sandbox side-effects.
        if self.guidance_manager:
            gm = self.guidance_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=gm.search,
                        display_label="Searching for relevant guidance",
                    ),
                    ToolSpec(fn=gm.filter, display_label="Filtering saved guidance"),
                    ToolSpec(
                        fn=gm.get_guidance,
                        display_label="Reading a full guidance entry",
                    ),
                    ToolSpec(fn=gm.add_guidance, display_label="Saving new guidance"),
                    ToolSpec(
                        fn=gm.update_guidance,
                        display_label="Updating saved guidance",
                    ),
                    ToolSpec(
                        fn=gm.delete_guidance,
                        display_label="Deleting saved guidance",
                    ),
                    ToolSpec(
                        fn=gm.reconcile_dependencies,
                        display_label="Checking guidance dependencies",
                    ),
                    include_class_name=True,
                ),
            )

        if self.knowledge_manager:
            km = self.knowledge_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=km.search,
                        display_label="Searching for relevant knowledge claims",
                    ),
                    ToolSpec(
                        fn=km.filter,
                        display_label="Filtering saved knowledge claims",
                    ),
                    ToolSpec(
                        fn=km.get_knowledge,
                        display_label="Reading a full knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.add_knowledge,
                        display_label="Saving a new knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.update_knowledge,
                        display_label="Updating a knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.delete_knowledge,
                        display_label="Deleting a knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.invalidate_knowledge,
                        display_label="Invalidating a knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.supersede_knowledge,
                        display_label="Superseding a knowledge claim",
                    ),
                    ToolSpec(
                        fn=km.reconcile_sources,
                        display_label="Reconciling knowledge provenance",
                    ),
                    include_class_name=True,
                ),
            )

        if self.workflow_manager:
            wm = self.workflow_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=wm.list_workflows,
                        display_label="Listing installable workflows",
                    ),
                    ToolSpec(
                        fn=wm.get_workflow,
                        display_label="Reading a workflow's record",
                    ),
                    ToolSpec(
                        fn=wm.install_workflow,
                        display_label="Installing a workflow",
                    ),
                    ToolSpec(
                        fn=wm.uninstall_workflow,
                        display_label="Uninstalling a workflow",
                    ),
                    ToolSpec(
                        fn=wm.get_installation_params,
                        display_label="Reading a workflow's settings",
                    ),
                    include_class_name=True,
                ),
            )

        # ── Proactive skill storage tool ──────────────────────────────
        if self.function_manager and self.guidance_manager:
            _actor_ref = self

            async def store_skills(request: str) -> Any:
                """Proactively store reusable skills from the current execution trajectory.

                Triggers a skill-storage review of the trajectory so far. A dedicated
                reviewer will examine the execution history and store any reusable
                functions, compositional guidance, and durable knowledge claims
                based on your request.

                Use this when you have just completed a complex subtask and recognize
                a reusable pattern worth preserving — for example, a non-obvious
                configuration of primitives.actor.act, a multi-step procedure, a
                function that bakes in hard-won configuration, or a durable sourced
                fact discovered during the run.

                Parameters
                ----------
                request : str
                    Describe the skill(s) you want stored. Be specific about which
                    part of the trajectory contains the reusable pattern and what
                    makes it valuable. For example: "Store the email lookup function
                    that uses primitives.contacts.ask with the scoped discovery_scope"
                    or "Store the multi-step data pipeline that combines file parsing
                    with a durable knowledge claim."

                Returns
                -------
                str
                    A summary of what was stored (functions, guidance, and/or
                    knowledge claims), or a note that nothing was worth storing.
                """
                ctx = get_current_agent_context()
                handle = ctx.handle
                if handle is None:
                    return "No active execution context to snapshot."

                _client = getattr(handle, "_client", None)
                _trajectory = (
                    make_messages_safe_for_context_dump(
                        list(getattr(_client, "messages", []) or []),
                    )
                    if _client
                    else []
                )

                _task = getattr(handle, "_task", None)
                _ask_tools = (
                    _task.get_ask_tools()
                    if _task and hasattr(_task, "get_ask_tools")
                    else {}
                )
                _completed_meta = (
                    _task.get_completed_tool_metadata()
                    if _task and hasattr(_task, "get_completed_tool_metadata")
                    else {}
                )

                _ps_call_id = new_call_id()
                _ps_parent = TOOL_LOOP_LINEAGE.get([])
                _ps_parent_lineage = (
                    list(_ps_parent) if isinstance(_ps_parent, list) else []
                )
                _ps_suffix = _token_hex(2)
                _ps_hierarchy = [
                    *_ps_parent_lineage,
                    f"ProactiveStorage(CodeActActor.act)({_ps_suffix})",
                ]

                await publish_manager_method_event(
                    _ps_call_id,
                    "CodeActActor",
                    "ProactiveStorage",
                    phase="incoming",
                    display_label="Proactive skill storage",
                    hierarchy=_ps_hierarchy,
                    instructions=request,
                )

                storage_handle = _start_proactive_storage_loop(
                    trajectory=_trajectory,
                    ask_tools=_ask_tools,
                    completed_tool_metadata=_completed_meta,
                    actor=_actor_ref,
                    request=request,
                    parent_lineage=_ps_parent_lineage,
                )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _ps_call_id,
                        "CodeActActor",
                        "ProactiveStorage",
                        phase="outgoing",
                        display_label="Proactive skill storage",
                        hierarchy=_ps_hierarchy,
                    )
                    return (
                        "Skill storage unavailable "
                        "(FunctionManager or GuidanceManager missing)."
                    )

                _orig_result_fn = storage_handle.result

                async def _tracking_result():
                    try:
                        result = await _orig_result_fn()
                        ctx.proactive_storage_summaries.append(result)
                        return result
                    finally:
                        await publish_manager_method_event(
                            _ps_call_id,
                            "CodeActActor",
                            "ProactiveStorage",
                            phase="outgoing",
                            display_label="Proactive skill storage",
                            hierarchy=_ps_hierarchy,
                        )

                storage_handle.result = _tracking_result  # type: ignore[assignment]

                return storage_handle

            tools["store_skills"] = store_skills

        if self.function_manager:

            @llm_soft_required(thought="")
            async def execute_function(
                thought: Annotated[
                    str,
                    "A brief, first-person, one-sentence explanation of what "
                    "this call does and why you are making it right now (e.g. "
                    '"Reading the five most recent inbox messages so I can '
                    'summarise them."). Shown to the user as the rationale for '
                    "this step; always provide it.",
                ],
                function_name: str,
                call_kwargs: Optional[Dict[str, Any]] = None,
                *,
                language: str = "python",
                state_mode: str = "stateless",
                session_id: int | None = None,
                session_name: str | None = None,
                _notification_up_q: asyncio.Queue[dict] | None = None,
                _clarification_up_q: asyncio.Queue[str] | None = None,
                _clarification_down_q: asyncio.Queue[str] | None = None,
                _interject_queue: asyncio.Queue | None = None,
                _pause_event: asyncio.Event | None = None,
                _parent_chat_context: list[dict] | None = None,
            ) -> Any:
                """
                Execute a single function or primitive by name.

                **This is the preferred tool for any task that maps to a single
                function or primitive call** — a primitive
                (``primitives.contacts.ask``, ``primitives.tasks.update``, …)
                or a stored function discovered via FunctionManager. It
                **structurally guarantees** the returned handle is exposed to
                the outer loop for steering (ask, stop, pause, resume,
                interject); inside ``execute_code`` a handle is only adopted
                if it happens to be the last expression. Use ``execute_code``
                only for genuine multi-step composition (conditional logic,
                loops, combining intermediate results).

                Resolution order: the current sandbox namespace first, then
                the FunctionManager store by exact name; otherwise a
                ``NameError`` is raised. ``language`` / ``state_mode`` /
                ``session_id`` / ``session_name`` keep ``execute_code``
                semantics, except ``state_mode`` here defaults to
                ``"stateless"``.

                Parameters
                ----------
                thought : str
                    One-sentence, first-person rationale for this call, shown
                    to the user. Always provide it.
                function_name : str
                    Exact name of the function or primitive to execute
                    (dotted path for primitives, e.g.
                    ``"primitives.contacts.ask"``).
                call_kwargs : dict, optional
                    Keyword arguments to pass. Values keep the callee's own
                    types — a plain keyword-argument mapping, not a string
                    map: numbers, booleans, lists, and objects unquoted,
                    exactly as the target signature declares them
                    (``{"max_results": 5}``, not ``{"max_results": "5"}``,
                    which fails type validation at the callee).

                Steering while the function runs
                -------------------------------
                Steerable in flight, like ``execute_code`` — checkpoints are
                placed inside a stored implementation's own body, so a long
                loop can be corrected partway through. A correction suspends
                the call and gives you a turn:
                ``stop_execute_function_<call_id>`` abandons it, interjecting
                again resumes it.

                Returns
                -------
                dict | ExecutionResult
                    Same shape as ``execute_code`` output.
                """
                _ = thought  # Thought is logged by the LLM; not used programmatically.
                call_kwargs = call_kwargs or {}
                resolved_venv_id: int | None = None
                function_data: dict[str, Any] | None = None
                get_function_data = getattr(
                    self.function_manager,
                    "_get_function_data_by_name",
                    None,
                )
                if callable(get_function_data):
                    function_data = get_function_data(name=function_name)
                if function_data is None:
                    get_stored_primitive = getattr(
                        self.function_manager,
                        "_get_stored_primitive_data_by_name",
                        None,
                    )
                    if callable(get_stored_primitive):
                        function_data = get_stored_primitive(name=function_name)
                stored_venv_id = (
                    function_data.get("venv_id")
                    if isinstance(function_data, dict)
                    and not function_data.get("is_primitive")
                    else None
                )
                if stored_venv_id is not None:
                    resolved_venv_id = int(stored_venv_id)

                # The synthesized-call path prepends the raw implementation
                # and runs it in the sandbox, shadowing any boundary-wrapped
                # callable — so the usage trace is fed here, where the row
                # is in hand, or this invocation would go unremembered.
                if isinstance(function_data, dict):
                    note_use = getattr(
                        self.function_manager,
                        "_note_function_use",
                        None,
                    )
                    if callable(note_use):
                        try:
                            note_use(function_data)
                        except Exception:  # noqa: BLE001 - never break a call
                            pass

                import time as _ef_time
                import logging as _ef_logging

                _ef_t0 = _ef_time.perf_counter()
                _ef_log = _ef_logging.getLogger("unify")
                _ef_fn_log = _ef_logging.getLogger(
                    f"unify.execute_function.{function_name}",
                )

                def _ef_ms():
                    return f"{(_ef_time.perf_counter() - _ef_t0) * 1000:.0f}ms"

                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] entered: {function_name}",
                )
                _ef_fn_log.info(
                    "START execute_function.%s params=%s",
                    function_name,
                    {
                        k: (
                            v
                            if isinstance(v, (int, float, bool, type(None)))
                            else repr(v)[:80]
                        )
                        for k, v in (call_kwargs or {}).items()
                        if not str(k).startswith("_")
                    },
                )
                _ef_fn_t0 = _ef_time.monotonic()

                # ── Synthesize the code string ────────────────────────────
                code: str | None = None

                if str(language) == "python":
                    code = _synthesize_python_call(
                        function_name=function_name,
                        call_kwargs=call_kwargs,
                        function_manager=self.function_manager,
                    )
                else:
                    # Shell: look up the stored implementation and append it
                    # with kwargs serialised as environment variables.
                    code = _synthesize_shell_call(
                        function_name=function_name,
                        call_kwargs=call_kwargs,
                        function_manager=self.function_manager,
                    )
                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] code synthesized",
                )

                # ── Lineage boundary ─────────────────────────────────────
                _ef_suffix = _token_hex(2)
                _ef_call_id = new_call_id()
                _ef_parent = TOOL_LOOP_LINEAGE.get([])
                _ef_parent_lineage = (
                    list(_ef_parent) if isinstance(_ef_parent, list) else []
                )
                _ef_hierarchy = [
                    *_ef_parent_lineage,
                    f"execute_function({function_name})({_ef_suffix})",
                ]
                _ef_lineage_token = TOOL_LOOP_LINEAGE.set(_ef_hierarchy)

                async def _ef_pub_safe(**payload: Any) -> None:
                    try:
                        await publish_manager_method_event(
                            _ef_call_id,
                            "CodeActActor",
                            "execute_function",
                            hierarchy=_ef_hierarchy,
                            display_label=f"Running: {function_name}",
                            **payload,
                        )
                    except Exception as e:
                        log_boundary_event(
                            "->".join(_ef_hierarchy),
                            f"Warning: failed to publish event: {type(e).__name__}: {e}",
                            icon="⚠️",
                            level="warning",
                        )

                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (incoming) start",
                )
                try:
                    await _ef_pub_safe(phase="incoming")
                except Exception:
                    pass
                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (incoming) done",
                )
                log_boundary_event(
                    "->".join(_ef_hierarchy),
                    f"Executing function {function_name}...",
                    icon="🛠️",
                )

                # ── Session resolution + execution (shared with execute_code) ──
                out: dict[str, Any] | None = None
                tb_str: str | None = None
                exec_exc: Exception | None = None

                active_work = ACTIVE_WORK.begin(
                    label="execute_function",
                    metadata={
                        "function_name": function_name,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "session_name": session_name,
                        "venv_id": resolved_venv_id,
                    },
                )
                heartbeat_task: asyncio.Task[None] | None = None
                try:
                    heartbeat_task = asyncio.create_task(
                        self._run_active_work_heartbeat(
                            active_work,
                            _notification_up_q,
                        ),
                    )
                    notification_q = (
                        _ActiveWorkNotificationQueue(_notification_up_q, active_work)
                        if _notification_up_q is not None
                        else None
                    )
                    _rs = self._resolve_session(
                        state_mode=state_mode,
                        language=str(language),
                        session_id=session_id,
                        session_name=session_name,
                        venv_id=resolved_venv_id,
                    )
                    language, resolved_venv_id, session_id = (
                        _rs.language,
                        _rs.venv_id,
                        _rs.session_id,
                    )
                    # Resolve primitives from current sandbox.
                    primitives = None
                    computer_primitives = self._computer_primitives
                    try:
                        sb = _CURRENT_SANDBOX.get()
                        primitives = sb.global_state.get("primitives")
                    except Exception:
                        pass

                    _ef_steering = None
                    with self._sandbox_call_binding(
                        clarification_up_q=_clarification_up_q,
                        clarification_down_q=_clarification_down_q,
                        interject_q=_interject_queue,
                        notification_q=notification_q,
                        pause_event=_pause_event,
                    ) as _ef_steering:
                        if (
                            isinstance(function_data, dict)
                            and function_data.get("is_primitive")
                            and is_provider_backed_function(function_data)
                        ):
                            _ef_log.debug(
                                f"⏱️ [execute_function +{_ef_ms()}] "
                                "provider primitive direct execute start",
                            )
                            try:
                                direct_result = (
                                    await self.function_manager.execute_function(
                                        function_name=function_name,
                                        call_kwargs=call_kwargs,
                                        target_venv_id=None,
                                        state_mode=state_mode,  # type: ignore[arg-type]
                                        session_id=session_id or 0,
                                        extra_namespaces=(
                                            {"primitives": primitives}
                                            if primitives is not None
                                            else None
                                        ),
                                        _parent_chat_context=_parent_chat_context,
                                    )
                                )
                                if (
                                    isinstance(direct_result, dict)
                                    and direct_result.get("status")
                                    == "confirmation_required"
                                ):
                                    direct_result = build_pending_approval_payload(
                                        function_name=function_name,
                                        function_data=function_data,
                                        call_kwargs=call_kwargs,
                                        provider_envelope=direct_result,
                                    )
                                    if notification_q is not None:
                                        await notification_q.put(direct_result)
                                out = {
                                    "stdout": [],
                                    "stderr": [],
                                    "result": direct_result,
                                    "error": None,
                                    "language": "python",
                                    "state_mode": state_mode,
                                    "session_id": session_id,
                                    "session_name": session_name,
                                    "venv_id": resolved_venv_id,
                                    "session_created": False,
                                    "duration_ms": int(
                                        (_ef_time.perf_counter() - _ef_t0) * 1000,
                                    ),
                                }
                                _ef_log.debug(
                                    f"⏱️ [execute_function +{_ef_ms()}] "
                                    "provider primitive direct execute done",
                                )
                            except Exception:
                                exec_exc = sys.exc_info()[1]
                                tb = traceback.format_exc()
                                tb_str = tb
                                out = {
                                    "stdout": [],
                                    "stderr": [],
                                    "result": None,
                                    "error": tb,
                                    "language": "python",
                                    "state_mode": state_mode,
                                    "session_id": session_id,
                                    "session_name": session_name,
                                    "venv_id": resolved_venv_id,
                                    "session_created": False,
                                    "duration_ms": int(
                                        (_ef_time.perf_counter() - _ef_t0) * 1000,
                                    ),
                                }
                        else:
                            _ef_log.debug(
                                f"⏱️ [execute_function +{_ef_ms()}] sandbox.execute start",
                            )
                            _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                            try:
                                try:
                                    out = await self._session_executor.execute(
                                        code=code,
                                        language=str(language),  # type: ignore[arg-type]
                                        state_mode=state_mode,  # type: ignore[arg-type]
                                        session_id=session_id,
                                        venv_id=resolved_venv_id,
                                        primitives=primitives,
                                        computer_primitives=computer_primitives,
                                    )
                                    _ef_log.debug(
                                        f"⏱️ [execute_function +{_ef_ms()}] sandbox.execute done",
                                    )
                                except Exception as e:
                                    exec_exc = e
                                    tb = traceback.format_exc()
                                    tb_str = tb
                                    out = {
                                        "stdout": "",
                                        "stderr": "",
                                        "result": None,
                                        "error": tb,
                                        "language": language,
                                        "state_mode": state_mode,
                                        "session_id": session_id,
                                        "session_name": session_name,
                                        "venv_id": resolved_venv_id,
                                        "session_created": False,
                                        "duration_ms": 0,
                                    }
                            finally:
                                _PARENT_CHAT_CONTEXT.reset(_pcc_token)

                    # Enrich with session name.
                    if out.get("session_id") is not None:
                        out["session_name"] = self._get_session_name(
                            language=str(out.get("language")),
                            venv_id=out.get("venv_id"),
                            session_id=int(out["session_id"]),
                        )
                    else:
                        out["session_name"] = None

                    # Wrap in-process Python results in ExecutionResult.
                    if out.get("language") == "python" and isinstance(
                        out.get("stdout"),
                        list,
                    ):
                        out = ExecutionResult(**out)

                    _ef_result_for_log = (
                        out.get("result")
                        if isinstance(out, dict)
                        else getattr(out, "result", None)
                    )
                    _ef_error_for_log = (
                        out.get("error")
                        if isinstance(out, dict)
                        else getattr(out, "error", None)
                    )
                    if _ef_error_for_log:
                        _ef_fn_log.error(
                            "FAIL execute_function.%s after %.1fs error=%s",
                            function_name,
                            _ef_time.monotonic() - _ef_fn_t0,
                            _ef_error_for_log,
                        )
                    else:
                        _ef_result_repr = repr(_ef_result_for_log)
                        if len(_ef_result_repr) > 240:
                            _ef_result_repr = _ef_result_repr[:237] + "..."
                        _ef_fn_log.info(
                            "END execute_function.%s after %.1fs result=%s",
                            function_name,
                            _ef_time.monotonic() - _ef_fn_t0,
                            _ef_result_repr,
                        )
                        if isinstance(_ef_result_for_log, dict):
                            _ef_status = str(
                                _ef_result_for_log.get("status") or "",
                            ).lower()
                            if (
                                "fail" in _ef_status
                                or _ef_result_for_log.get("error")
                                or _ef_result_for_log.get("partial_error")
                            ):
                                _ef_fn_log.error(
                                    "SOFT_FAIL execute_function.%s status=%r "
                                    "error=%r partial_error=%r",
                                    function_name,
                                    _ef_result_for_log.get("status"),
                                    _ef_result_for_log.get("error"),
                                    _ef_result_for_log.get("partial_error"),
                                )

                    # When the execution produced a bare SteerableToolHandle
                    # with no meaningful side output, return the handle directly
                    # so the core loop adopts it via the bare-handle path
                    # (no intermediate LLM turn required).
                    _ef_result_val = _ef_result_for_log
                    if isinstance(_ef_result_val, SteerableToolHandle):
                        _ef_stdout = (
                            out.get("stdout")
                            if isinstance(out, dict)
                            else getattr(out, "stdout", None)
                        )
                        _ef_stderr = (
                            out.get("stderr")
                            if isinstance(out, dict)
                            else getattr(out, "stderr", None)
                        )
                        _ef_error = (
                            out.get("error")
                            if isinstance(out, dict)
                            else getattr(out, "error", None)
                        )
                        _has_side_output = bool(
                            (
                                _ef_stdout
                                and (
                                    isinstance(_ef_stdout, str)
                                    and _ef_stdout.strip()
                                    or isinstance(_ef_stdout, list)
                                    and _ef_stdout
                                )
                            )
                            or (
                                _ef_stderr
                                and (
                                    isinstance(_ef_stderr, str)
                                    and _ef_stderr.strip()
                                    or isinstance(_ef_stderr, list)
                                    and _ef_stderr
                                )
                            )
                            or _ef_error,
                        )
                        if not _has_side_output:
                            _ef_log.debug(
                                f"⏱️ [execute_function +{_ef_ms()}] "
                                f"returning bare handle (no side output)",
                            )
                            return _ef_result_val

                    # Same contract as execute_code: report only when something
                    # actually steered this call. The bare-handle return above
                    # is exempt — that value is a handle the loop adopts, and
                    # steering continues through it rather than ending here.
                    if _ef_steering is not None and _ef_steering.messages:
                        if isinstance(out, dict):
                            out["steering"] = _ef_steering.progress()
                        else:
                            out.steering = _ef_steering.progress()

                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] returning result",
                    )
                    return out
                finally:
                    active_work.end()
                    if heartbeat_task is not None and not heartbeat_task.done():
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (outgoing) start",
                    )
                    try:
                        _out_err = (
                            (
                                out.get("error")
                                if isinstance(out, dict)
                                else getattr(out, "error", None)
                            )
                            if out is not None
                            else None
                        )
                        if _out_err:
                            await _ef_pub_safe(
                                phase="outgoing",
                                status="error",
                                error=str(_out_err),
                                error_type=(
                                    type(exec_exc).__name__
                                    if exec_exc is not None
                                    else "Error"
                                ),
                                traceback=(tb_str or "")[:2000],
                            )
                        else:
                            await _ef_pub_safe(phase="outgoing", status="ok")
                    except Exception:
                        pass
                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (outgoing) done",
                    )
                    try:
                        TOOL_LOOP_LINEAGE.reset(_ef_lineage_token)
                    except Exception:
                        pass

            def _ef_display_label(tc: dict) -> str:
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    return args.get("function_name", "execute_function")
                except Exception:
                    return "execute_function"

            tools["execute_function"] = ToolSpec(
                fn=execute_function,
                display_label=_ef_display_label,
            )

        # ───────────────────────── Session management tools ────────────────── #

        async def list_sessions(detail: str = "summary") -> Dict[str, Any]:
            """
            List all active sessions across all languages (Python + shell).

            Use this to choose which session a subsequent
            `execute_code(..., state_mode="stateful"/"read_only")` call
            should target.

            Parameters
            ----------
            detail:
                "summary" (default): metadata + a short `state_summary`;
                "full": best-effort enrichment via cheap inspection.

            Returns
            -------
            dict:
                {"sessions": [...]}; each entry carries language,
                session_id, venv_id (Python only), session_name,
                created_at / last_used, and state_summary. Session IDs are
                **scoped per (language, venv_id)**; the default per-call
                Python sandbox appears as `python` session_id=0
                (venv_id=None) when bound.
            """
            detail = (detail or "summary").strip()

            sessions: list[dict[str, Any]] = []

            # Default sandbox (current act sandbox) as python session 0 (venv_id=None).
            try:
                sb = _CURRENT_SANDBOX.get()
                sessions.append(
                    {
                        "language": "python",
                        "session_id": 0,
                        "venv_id": None,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "created_at": None,
                        "last_used": None,
                        "state_summary": f"{len(sb.global_state)} globals",
                    },
                )
            except Exception:
                pass

            # In-process python sessions created via SessionExecutor.
            for s in self._session_executor.list_in_process_python_sessions():
                s = dict(s)
                s["session_name"] = self._get_session_name(
                    language="python",
                    venv_id=s.get("venv_id"),
                    session_id=int(s["session_id"]),
                )
                sessions.append(s)

            # Venv sessions.
            try:
                for s in self._venv_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language="python",
                        venv_id=s.get("venv_id"),
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            # Shell sessions.
            try:
                for s in self._shell_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language=str(s.get("language")),
                        venv_id=None,
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            if detail == "full":
                # Best-effort enrich state_summary with inspection where cheap.
                for s in sessions:
                    try:
                        if (
                            s.get("language") == "python"
                            and s.get("venv_id") is not None
                        ):
                            st = await self._venv_pool.get_session_state(
                                venv_id=int(s["venv_id"]),
                                session_id=int(s["session_id"]),
                                function_manager=self.function_manager,
                                detail="summary",
                            )
                            if isinstance(st, dict) and "count" in st:
                                s["state_summary"] = f'{st["count"]} names'
                        elif s.get("language") in ("bash", "zsh", "sh", "powershell"):
                            st = await self._shell_pool.get_session_state(
                                language=s["language"],
                                session_id=int(s["session_id"]),
                                detail="summary",
                            )
                            if isinstance(st, dict) and "summary" in st:
                                s["state_summary"] = st["summary"]
                    except Exception:
                        continue

            return {"sessions": sessions}

        async def inspect_state(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
            detail: str = "summary",
        ) -> Dict[str, Any]:
            """
            Inspect the state of a specific session (Python or shell).

            Use it to decide whether to continue in a session, start fresh,
            run stateless, or do a read_only what-if.

            Parameters
            ----------
            session_name:
                Human-friendly alias (preferred when available).
            session_id + language (+ optional venv_id):
                Direct identity; `session_id` is scoped per (language,
                venv_id).
            detail:
                "summary" (quick context) | "names" (variable names only) |
                "full" (sparingly; values truncated/redacted best-effort).

            With no selector, inspects the **current per-call Python
            sandbox** (python session_id=0, venv_id=None) when bound.

            Returns
            -------
            dict with `session` ({language, session_id, session_name,
            venv_id}) and `state` (Python vars; shell
            cwd/env/functions/aliases).
            """
            detail = (detail or "summary").strip()

            # Resolve session.
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "error": f"Session {session_name!r} not found",
                        "error_type": "validation",
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))

            # Default: current sandbox.
            if resolved is None:
                try:
                    sb = _CURRENT_SANDBOX.get()
                except Exception as e:
                    return {
                        "error": f"No sandbox bound: {type(e).__name__}",
                        "error_type": "internal",
                    }

                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s

                names = sorted(names)
                state_obj: dict[str, Any]
                if detail == "full":
                    state_obj = {"variables": full_map, "functions": []}
                else:
                    state_obj = {"variables": names, "functions": []}

                return {
                    "session": {
                        "language": "python",
                        "session_id": 0,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            lang, resolved_venv_id, sid = resolved

            # Python venv-backed
            if lang == "python" and resolved_venv_id is not None:
                st = await self._venv_pool.get_session_state(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                    function_manager=self.function_manager,
                    detail=detail,
                )
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=int(resolved_venv_id),
                            session_id=int(sid),
                        ),
                        "venv_id": int(resolved_venv_id),
                    },
                    "state": st,
                }

            # Python in-process session (SessionExecutor)
            if lang == "python" and resolved_venv_id is None:
                key = (None, int(sid))
                sb = self._session_executor._python_sessions.get(
                    key,
                )  # pylint: disable=protected-access
                if sb is None:
                    return {
                        "error": f"Python session {sid} not found",
                        "error_type": "validation",
                    }
                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s
                names = sorted(names)
                state_obj = {
                    "variables": full_map if detail == "full" else names,
                    "functions": [],
                }
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=int(sid),
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            # Shell
            st = await self._shell_pool.get_session_state(
                language=lang,  # type: ignore[arg-type]
                session_id=int(sid),
                detail=detail,
            )
            return {
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": self._get_session_name(
                        language=str(lang),
                        venv_id=None,
                        session_id=int(sid),
                    ),
                    "venv_id": None,
                },
                "state": st,
            }

        async def close_session(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
        ) -> Dict[str, Any]:
            """
            Close a specific session and free resources.

            **Idempotent**: closing an already-closed/non-existent session
            returns `closed=False, reason="not_found"` rather than raising.

            Parameters
            ----------
            session_name:
                Preferred: close by human-friendly alias.
            session_id + language (+ optional venv_id):
                Close by canonical identity.

            Returns
            -------
            dict:
                closed (bool), reason ("success" | "not_found" | "error"),
                session ({language, session_id, session_name}).
            """
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "closed": False,
                        "reason": "not_found",
                        "session": {
                            "language": language,
                            "session_id": session_id,
                            "session_name": session_name,
                        },
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))
            else:
                return {
                    "closed": False,
                    "reason": "error",
                    "error": "Must provide session_name or (language + session_id).",
                }

            lang, resolved_venv_id, sid = resolved
            closed = False

            if lang == "python" and resolved_venv_id is not None:
                closed = await self._venv_pool.close_session(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                )
            elif lang == "python" and resolved_venv_id is None:
                closed = await self._session_executor.close_in_process_python_session(
                    session_id=int(sid),
                    venv_id=None,
                )
            else:
                closed = await self._shell_pool.close_session(language=lang, session_id=int(sid))  # type: ignore[arg-type]

            # Unregister all aliases for this session.
            self._unregister_all_names_for_session(
                key=(str(lang), resolved_venv_id, int(sid)),
            )

            return {
                "closed": bool(closed),
                "reason": "success" if closed else "not_found",
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": session_name
                    or self._get_session_name(
                        language=str(lang),
                        venv_id=resolved_venv_id,
                        session_id=int(sid),
                    ),
                },
            }

        async def close_all_sessions() -> Dict[str, Any]:
            """
            Close all active sessions across all languages.

            Blunt cleanup — prefer `close_session(...)` to discard one
            specific polluted/unused session.

            Returns
            -------
            dict:
                closed_count (int), languages (list[str]), details
                (per-language counts).
            """
            closed_counts: dict[str, int] = {
                "python": 0,
                "bash": 0,
                "zsh": 0,
                "sh": 0,
                "powershell": 0,
            }

            # Close in-process python sessions.
            for s in list(self._session_executor.list_in_process_python_sessions()):
                sid = int(s.get("session_id", 0))
                if await self._session_executor.close_in_process_python_session(
                    session_id=sid,
                    venv_id=None,
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(key=("python", None, sid))

            # Close venv python sessions.
            for vid, sid in list(self._venv_pool.list_active_sessions()):
                if await self._venv_pool.close_session(
                    venv_id=int(vid),
                    session_id=int(sid),
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(
                        key=("python", int(vid), int(sid)),
                    )

            # Close shell sessions.
            for lang, sid in list(self._shell_pool.get_active_sessions()):
                if await self._shell_pool.close_session(
                    language=lang,
                    session_id=int(sid),
                ):
                    closed_counts[str(lang)] = closed_counts.get(str(lang), 0) + 1
                    self._unregister_all_names_for_session(
                        key=(str(lang), None, int(sid)),
                    )

            # Clear any remaining aliases.
            self._session_names.clear()
            self._session_names_rev.clear()

            closed_total = sum(closed_counts.values())
            langs = [k for k, v in closed_counts.items() if v > 0]
            return {
                "closed_count": closed_total,
                "languages": langs,
                "details": closed_counts,
            }

        tools["list_sessions"] = ToolSpec(
            fn=list_sessions,
            display_label="Listing active sessions",
        )
        tools["inspect_state"] = ToolSpec(
            fn=inspect_state,
            display_label="Inspecting session state",
        )
        tools["close_session"] = ToolSpec(
            fn=close_session,
            display_label="Closing a session",
        )
        tools["close_all_sessions"] = ToolSpec(
            fn=close_all_sessions,
            display_label="Closing all sessions",
        )

        # ───────────────────── Package installation tool ───────────────── #

        async def install_python_packages(
            packages: list[str],
        ) -> Dict[str, Any]:
            overlay = _CURRENT_PACKAGE_OVERLAY.get()
            if overlay is None:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "No package overlay is bound for this trajectory.",
                    "packages": packages,
                }
            return overlay.install(packages)

        install_python_packages.__doc__ = _INSTALL_PYTHON_PACKAGES_DOC

        tools["install_python_packages"] = ToolSpec(
            fn=install_python_packages,
            display_label="Installing Python packages",
        )

        return tools

    async def _repair_function(
        self,
        *,
        function_id: int,
        request: str | dict | list[str | dict],
        entrypoint_kwargs: dict[str, Any],
        failure: BaseException | None,
        verdict: Any = None,
        frames: tuple[Frame, ...] = (),
        repair_context: dict[str, Any] | None,
        destination: str | None = None,
    ) -> str:
        """Run a bounded review loop that repairs one failing stored function in place.

        The target is the leaf a verdict blamed (or the innermost stored
        function in a traceback). Deployment-owned functions (``custom_hash``
        set) are never rewritten here: their bodies are re-synced from the
        bundle, so an in-place rewrite would silently diverge from it and
        mask the failure. ``RepairRefused`` is raised instead.
        """

        fm = self.function_manager
        if fm is None:
            raise RepairRefused(
                "Cannot repair a stored function without a FunctionManager.",
            )

        snapshot_namespace: dict[str, Any] = {}
        snapshot_result = fm.filter_functions(
            filter=f"function_id == {int(function_id)}",
            destination=destination,
            _return_callable=True,
            _namespace=snapshot_namespace,
            _also_return_metadata=True,
        )
        function_snapshot = (
            snapshot_result.get("metadata", [])
            if isinstance(snapshot_result, dict)
            else snapshot_result
        )
        for row in function_snapshot or []:
            if isinstance(row, dict) and row.get("custom_hash"):
                raise RepairRefused(
                    f"Function {row.get('name')!r} (id {function_id}) is "
                    "deployment-owned (custom_hash set); refusing repair. Fix the "
                    "bundle source and re-sync via deployment reconcile.",
                )
        # The repairer keeps the verdict history: it is being asked to answer
        # a verdict, and the prior ones on the same function are what tell it
        # whether this objection is new or whether its own last attempt
        # caused it.
        snapshot_for_prompt = strip_ledger_internals(
            [row for row in (function_snapshot or []) if isinstance(row, dict)],
            keep_verdict_history=True,
        )
        tools = methods_to_tool_dict(
            fm.search_functions,
            fm.filter_functions,
            fm.list_functions,
            fm.add_functions,
            fm.delete_function,
            fm.add_venv,
            fm.list_venvs,
            fm.get_venv,
            fm.update_venv,
            fm.delete_venv,
            fm.set_function_venv,
            fm.get_function_venv,
            include_class_name=True,
        )
        tools["run_diagnostic_probe"] = run_probe
        tools.update(self._verification_librarian_tools())
        client = new_llm_client(self._model, purpose="repair")
        client.set_system_message(
            "You are repairing a stored function that runs as part of a "
            "recurring task. The contract you must preserve is the task's "
            "OUTCOME: what it computes, the semantics and exactness of those "
            "values, which side effects it performs, where it delivers them, in "
            "what order, and how it fails when the outcome is truly "
            "unachievable. How the function READS its external inputs is not "
            "contract: external interfaces evolve after a function is stored "
            "(fields get renamed or nested, endpoints get versioned), and "
            "adapting ingestion to the environment's current shape while "
            "keeping the outcome exactly equivalent is precisely what repair is "
            "for. Diagnose before you rewrite: when the failure implicates an "
            "external input surface, first use run_diagnostic_probe to observe "
            "what that interface actually returns right now (its shape, keys, "
            "and a sample record) and base the repair on that observation. "
            "Probes are strictly read-only diagnosis — never perform the "
            "function's side effects through them. The task description "
            "records the environment as it looked when the task was created; "
            "when observed reality contradicts it, trust the observation over "
            "the description's input details. Bear in mind that the function's "
            "own validation messages describe its assumptions, not what the "
            "environment actually returned — a missing expected field usually "
            "means the interface changed shape, not that the data is corrupt, "
            "so prefer ingestion that reads the observed current shape over "
            "rejecting the input. Never weaken the outcome to make the error "
            "disappear: do not fabricate values, skip required side effects, "
            "or coerce genuinely invalid data. Update the existing function in "
            "place with overwrite=True so its function_id stays stable; never "
            "delete and re-add it, because references such as task entrypoints "
            "hold the id. Do not replace managed primitives with ad hoc weaker "
            "implementations. When an independent verifier failed the function, "
            "its verdict and the chain of calls that led to it are below: the "
            "verdict names what was wrong and whether the fault sits in this "
            "function (leaf) or in how its caller used it. A repaired pure "
            "function is replayed against its recorded fixtures before it is "
            "accepted; keep every recorded input/output pair reproducing. Trust "
            "in the repaired function is earned again by independent "
            "verification — you cannot grant it.",
        )
        failure_line = (
            f"Failure: {type(failure).__name__}: {failure}"
            if failure is not None
            else "Failure: verifier verdict (below)."
        )
        verdict_block = ""
        if verdict is not None:
            verdict_block = (
                "Verifier verdict:\n"
                f"```json\n{json.dumps(getattr(verdict, 'model_dump', lambda **_: verdict)(mode='json') if hasattr(verdict, 'model_dump') else verdict, indent=2, default=str)}\n```\n\n"
            )
        chain_block = ""
        if frames:
            chain_lines = []
            for index, frame in enumerate(frames, start=1):
                chain_lines.append(
                    f"{index}. {frame.name} [{frame.effect_class}] — "
                    f"{(frame.docstring or '').strip().splitlines()[0] if (frame.docstring or '').strip() else '(no docstring)'}",
                )
                if frame.call_site_line:
                    chain_lines.append(f"   called as: {frame.call_site_line.strip()}")
            chain_block = (
                "Call chain (root → failing call):\n" + "\n".join(chain_lines) + "\n\n"
            )
        message = (
            "A stored function failed during a symbolic task run.\n\n"
            f"Task request:\n{request}\n\n"
            "Deterministic entrypoint kwargs:\n"
            f"```json\n{json.dumps(entrypoint_kwargs, indent=2, default=str)}\n```\n\n"
            "Function snapshot:\n"
            f"```json\n{json.dumps(snapshot_for_prompt, indent=2, default=str)}\n```\n\n"
            "Repair context:\n"
            f"```json\n{json.dumps(repair_context or {}, indent=2, default=str)}\n```\n\n"
            f"{verdict_block}{chain_block}"
            f"{failure_line}\n\n"
            "Diagnose the failure — observing the current behavior of any "
            "implicated external input surface via run_diagnostic_probe "
            "(read-only) before deciding — then repair the stored function in "
            "place (overwrite=True) when an outcome-equivalent fix exists, "
            "including adapting input handling to an evolved external "
            "interface. Briefly summarize the observed evidence, the "
            "equivalence rationale, and the change made. Only if no fix can "
            "preserve the task's outcome semantics, say so without modifying "
            "the function."
        )
        handle = start_async_tool_loop(
            client=client,
            message=message,
            tools=tools,
            loop_id=f"FunctionRepair({function_id})",
            max_consecutive_failures=2,
        )
        result = await handle.result()
        return str(result)

    def _verification_librarian_tools(self) -> Dict[str, Callable]:
        """Tools that let a librarian or repair loop shape verification policy (never trust)."""
        return _verification_librarian_tools(self.function_manager)

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call(
        "CodeActActor",
        "act",
        payload_key="request",
        display_label=lambda kw: "Session" if kw.get("persist") else "Taking action",
        forward_kwargs=("persist",),
    )
    async def act(
        self,
        request: str | dict | list[str | dict],
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        _reuse_actor_slot: bool = False,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        entrypoint_repair_context: Optional[dict[str, Any]] = None,
        destination: Optional[str] = None,
        persist: Optional[bool] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        llm_profile: Optional[str] = None,
    ) -> SteerableToolHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        import time as _act_time

        _act_t0 = _act_time.perf_counter()

        def _act_ms() -> str:
            return f"{(_act_time.perf_counter() - _act_t0) * 1000:.0f}ms"

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] entered")

        from unify.runtime.drain_gate import refuse_if_draining

        refuse_if_draining()

        effective_can_compose = (
            self.can_compose if can_compose is None else bool(can_compose)
        )
        effective_can_store = self.can_store if can_store is None else bool(can_store)
        act_llm_profile = resolve_act_llm_profile(llm_profile)

        # can_compose=False requires a FunctionManager so the LLM has execute_function
        # and the discovery tools available. Without it there are no usable tools.
        if not effective_can_compose and self.function_manager is None:
            raise RuntimeError(
                "CodeActActor cannot run with can_compose=False: "
                "function_manager is required so execute_function and "
                "FunctionManager discovery tools are available.",
            )

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        # Clarification queues for sandbox env injection (managers called from
        # execute_code). Separate from the tool-loop clarification_queues below:
        # auto-created env queues are unread on the CM→act path, so the loop
        # gets (None, None) unless the caller supplied queues explicitly.
        caller_supplied_clarification_queues = (
            clarification_enabled
            and _clarification_up_q is not None
            and _clarification_down_q is not None
        )
        env_clarification_up_q: Optional[asyncio.Queue[str]]
        env_clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            env_clarification_up_q = _clarification_up_q or asyncio.Queue()
            env_clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            env_clarification_up_q = None
            env_clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] copying environments")
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unify.actor.environments.base import (
                _CompositeEnvironment as _CompositeEnv,
            )
            from unify.actor.environments import (
                ComputerEnvironment as _ComputerEnvironment,
                StateManagerEnvironment as _StateManagerEnvironment,
            )
        except Exception:
            _CompositeEnv = None  # type: ignore
            _ComputerEnvironment = None  # type: ignore
            _StateManagerEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _CompositeEnv is not None and isinstance(env, _CompositeEnv):
                    sandbox_envs[ns] = _CompositeEnv(
                        env.sub_environments,
                        clarification_up_q=env_clarification_up_q,
                        clarification_down_q=env_clarification_down_q,
                    )
                    continue
                if _ComputerEnvironment is not None and isinstance(
                    env,
                    _ComputerEnvironment,
                ):
                    sandbox_envs[ns] = _ComputerEnvironment(
                        env._computer_primitives,
                        clarification_up_q=env_clarification_up_q,
                        clarification_down_q=env_clarification_down_q,
                    )
                    continue
                if _StateManagerEnvironment is not None and isinstance(
                    env,
                    _StateManagerEnvironment,
                ):
                    sandbox_envs[ns] = _StateManagerEnvironment(
                        env.get_instance(),
                        clarification_up_q=env_clarification_up_q,
                        clarification_down_q=env_clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", env_clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", env_clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard for externally started actor runs.
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] envs copied, preparing actor slot",
        )
        acquired_actor_slot = False
        if not _reuse_actor_slot:
            try:
                await asyncio.wait_for(
                    self._act_semaphore.acquire(),
                    timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
                )
                acquired_actor_slot = True
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "CodeActActor is at capacity (too many concurrent sessions). "
                    "Try again later or reduce concurrency.",
                )
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] actor slot ready, creating sandbox",
        )
        sandbox = PythonExecutionSession(
            computer_primitives=self._computer_primitives,
            environments=sandbox_envs,
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
        )
        token = _CURRENT_SANDBOX.set(sandbox)
        env_token = _CURRENT_ENVIRONMENTS.set(sandbox_envs)
        llm_profile_token = CURRENT_ACT_LLM_PROFILE.set(act_llm_profile)

        # Set agent context for depth tracking and handle access
        parent_ctx = _CURRENT_AGENT_CONTEXT.get()
        new_ctx = AgentContext(
            depth=parent_ctx.depth + 1,
            agent_id=str(uuid.uuid4()),
            handle=None,  # Will be set after handle is created
        )
        ctx_token = _CURRENT_AGENT_CONTEXT.set(new_ctx)

        # Per-trajectory package overlay: lazily installs packages into a
        # temporary directory on sys.path and cleans them up when act() ends.
        # Created after AgentContext so it can use agent_id for directory naming,
        # and after _CURRENT_PACKAGE_OVERLAY is readable so child overlays
        # discover their parent's directory for hierarchical nesting.
        pkg_overlay = PackageOverlay(agent_id=new_ctx.agent_id)
        pkg_overlay_token = _CURRENT_PACKAGE_OVERLAY.set(pkg_overlay)

        # Mutable ref populated after handle creation so _cleanup can deregister.
        _registered_queue: list[asyncio.Queue | None] = [None]

        async def _cleanup() -> None:
            if (
                _registered_queue[0] is not None
                and self._computer_primitives is not None
            ):
                self._computer_primitives.deregister_interject_queue(
                    _registered_queue[0],
                )
            try:
                pkg_overlay.cleanup()
            except Exception:
                pass
            try:
                _CURRENT_PACKAGE_OVERLAY.reset(pkg_overlay_token)
            except Exception:
                pass
            try:
                # Best-effort cleanup
                if hasattr(sandbox, "close") and callable(getattr(sandbox, "close")):
                    await sandbox.close()  # type: ignore[misc]
            except Exception:
                pass
            try:
                _CURRENT_SANDBOX.reset(token)
            except Exception:
                pass
            try:
                _CURRENT_ENVIRONMENTS.reset(env_token)
            except Exception:
                pass
            try:
                CURRENT_ACT_LLM_PROFILE.reset(llm_profile_token)
            except Exception:
                pass
            try:
                _CURRENT_AGENT_CONTEXT.reset(ctx_token)
            except Exception:
                pass
            if acquired_actor_slot:
                try:
                    self._act_semaphore.release()
                except Exception:
                    pass

        task_execution_delegate: TaskExecutionDelegate = _CodeActTaskExecutionDelegate(
            self,
        )

        # If an explicit FunctionManager entrypoint is provided (e.g., TaskScheduler task execution),
        # bypass the CodeAct LLM loop and run the function directly.
        if entrypoint is not None:
            entrypoint_id = int(entrypoint)
            args = list(entrypoint_args or [])
            kwargs_for_entrypoint = dict(entrypoint_kwargs or {})
            fm = self.function_manager
            if fm is None:
                raise RuntimeError(
                    "CodeActActor cannot execute entrypoint: function_manager is None",
                )
            verification_settings = fm.verification_settings
            repair_context = (
                entrypoint_repair_context
                if isinstance(entrypoint_repair_context, dict)
                else None
            )
            task_name = str(
                (repair_context or {}).get("task_name")
                or (repair_context or {}).get("task_run_context", {}).get("task_name")
                or f"task {kwargs_for_entrypoint.get('task_id', '')}".strip()
                or "the task",
            )
            run_key = kwargs_for_entrypoint.get("run_key")
            task_id_value = kwargs_for_entrypoint.get("task_id")
            goal_text = (
                request
                if isinstance(request, str)
                else json.dumps(request, default=str)
            )

            def _resolve_closure() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
                rows = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    destination=destination,
                    include_implementations=True,
                )
                if not rows:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                root_row = dict(rows[0])
                fn_name = root_row.get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} has no valid function name.",
                    )
                closure = closure_rows(fm, root_row)
                # The stored flag can lag a change elsewhere in the closure;
                # the run sees the derived value.
                rederive_trust(fm, closure, settings=verification_settings)
                root_row["verify"] = closure[str(fn_name)]["verify"]
                last_closure_ids[0] = [
                    int(row["function_id"]) for row in closure.values()
                ]
                return root_row, closure

            async def _invoke_root(
                rows_by_name: dict[str, dict[str, Any]],
                supervisor: Any,
            ) -> Any:
                out = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    destination=destination,
                    _return_callable=True,
                    _namespace=sandbox.global_state,
                    _also_return_metadata=True,
                )
                metadata = (
                    list(out.get("metadata") or []) if isinstance(out, dict) else []
                )
                if not metadata:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                fn_name = str(metadata[0].get("name"))
                if supervisor is not None:
                    install_wrappers(
                        sandbox.global_state,
                        rows_by_name=rows_by_name,
                        supervisor=supervisor,
                    )
                fn = sandbox.global_state.get(fn_name)
                if fn is None:
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} ({fn_name}) was not injected into the sandbox namespace.",
                    )
                compatible_kwargs = _signature_compatible_kwargs(
                    getattr(fn, "__wrapped__", fn),
                    kwargs_for_entrypoint,
                )
                # Async entrypoints stay on this loop. Sync entrypoints run
                # on a worker thread so nested asyncio.run inside helpers is
                # safe (offline Jobs already own a loop via asyncio.run) and
                # long sync work does not starve the runner loop.
                if inspect.iscoroutinefunction(fn):
                    res = await fn(*args, **compatible_kwargs)
                else:
                    res = await asyncio.to_thread(fn, *args, **compatible_kwargs)
                    if inspect.isawaitable(res):
                        res = await res
                return res

            def _make_passes() -> VerifierPasses:
                return VerifierPasses(
                    function_manager=fm,
                    guidance_manager=self.guidance_manager,
                    goal=goal_text,
                    run_key=run_key,
                    task_id=int(task_id_value) if task_id_value is not None else None,
                    model=verification_settings.model,
                )

            async def _repair(rewind: RewindRequested) -> None:
                if rewind.target_function_id is None:
                    raise RepairRefused(
                        "No stored function could be identified to repair.",
                    )
                await self._repair_function(
                    function_id=int(rewind.target_function_id),
                    request=request,
                    entrypoint_kwargs=kwargs_for_entrypoint,
                    failure=rewind.exception,
                    verdict=rewind.verdict,
                    frames=rewind.frames,
                    repair_context=repair_context,
                    destination=destination,
                )

            entry_handle_ref: list[Any] = []
            last_closure_ids: list[list[int]] = [[]]

            async def _notify_owner(message: str) -> None:
                if entry_handle_ref:
                    await entry_handle_ref[0].push_notification(message)

            def _settle_ledger_and_promote() -> None:
                """Fold this run's verdicts synchronously; promote the task if its closure is trusted."""
                closure_ids = list(last_closure_ids[0])
                all_trusted = bool(closure_ids)
                for function_id in closure_ids:
                    if fm.refresh_trust(function_id) is not False:
                        all_trusted = False
                if (
                    not verification_settings.auto_promote_offline
                    or task_id_value is None
                    or not all_trusted
                ):
                    return
                from unify.manager_registry import ManagerRegistry

                scheduler = ManagerRegistry.get_task_scheduler()
                try:
                    scheduler.promote_task_offline(task_id=int(task_id_value))
                except ValueError:
                    # The run's task_id does not name a definition this
                    # scheduler holds (ad hoc entrypoint runs); nothing to promote.
                    return

            async def _run_entrypoint() -> EntrypointOutcome:
                outcome = await run_verified_entrypoint(
                    settings=verification_settings,
                    task_name=task_name,
                    resolve=_resolve_closure,
                    invoke=_invoke_root,
                    make_passes=_make_passes,
                    repair=_repair,
                    notify=_notify_owner,
                )
                if outcome.held is None and outcome.follow_up is None:
                    await asyncio.to_thread(_settle_ledger_and_promote)
                return outcome

            run_meter = new_run_meter()
            delegate_token = current_task_execution_delegate.set(
                task_execution_delegate,
            )
            meter_token = current_run_meter.set(run_meter)
            try:
                entry_task = asyncio.create_task(_run_entrypoint())
                entry_handle = _CodeActEntrypointHandle(
                    entrypoint_id=entrypoint_id,
                    execution_task=entry_task,
                    on_finally=_cleanup,
                    meter=run_meter,
                )
                entry_handle_ref.append(entry_handle)
            finally:
                current_run_meter.reset(meter_token)
                current_task_execution_delegate.reset(delegate_token)
            return entry_handle

        # Build the tool set for this call. When can_compose=False the LLM
        # can_compose=False: specialist may only discover and execute stored
        # functions — no arbitrary code, no function persistence.
        # can_store=False: function/guidance library is read-only.
        # Session tools are kept because execute_function supports the same
        # session/state_mode semantics.
        _compose_only_tools = {
            "execute_code",
            "install_python_packages",
        }
        _store_only_tools = {
            "store_skills",
            "FunctionManager_add_functions",
            "FunctionManager_delete_function",
            "FunctionManager_reconcile_dependencies",
            "GuidanceManager_reconcile_dependencies",
        }

        def _filter_tools(tool_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Apply static per-call filters (can_compose, can_store)."""
            out = dict(tool_dict)
            if not effective_can_compose:
                for name in _compose_only_tools:
                    out.pop(name, None)
                for name in _store_only_tools:
                    out.pop(name, None)
            if not effective_can_store:
                for name in _store_only_tools:
                    out.pop(name, None)
            return out

        base_tools = _filter_tools(self.get_tools("act"))

        # When execute_code is masked (can_compose=False), strip any
        # execute_code references from execute_function's docstring so the
        # LLM has no awareness that a code sandbox exists.
        if "execute_function" in base_tools and "execute_code" not in base_tools:
            _ef = base_tools["execute_function"]
            (_ef.fn if isinstance(_ef, ToolSpec) else _ef).__doc__ = (
                "Execute a known function by name and return its result.\n"
                "\n"
                "The function is resolved from the sandbox namespace or looked up\n"
                "in the FunctionManager by exact name. Functions discovered via the\n"
                "FunctionManager discovery tools are automatically available.\n"
                "\n"
                "Steps\n"
                "-----\n"
                "1. Discover stored functions via ``FunctionManager_search_functions``,\n"
                "   ``FunctionManager_filter_functions``, or\n"
                "   ``FunctionManager_list_functions``.\n"
                "2. Call ``execute_function`` with a stored match or a\n"
                "   prompt-documented callable by exact name (primitives are\n"
                "   excluded from discovery).\n"
                "\n"
                "Key concepts\n"
                "------------\n"
                '- **language**: ``"python"`` | ``"bash"`` | ``"zsh"`` | '
                '``"sh"`` | ``"powershell"``\n'
                "- **state_mode**:\n"
                '  - ``"stateless"``: no session; clean execution; no persistence\n'
                '  - ``"stateful"``: persistent session; state accumulates\n'
                '  - ``"read_only"``: reads from an existing session but does not\n'
                "    persist changes\n"
                "- **session_id / session_name**: only meaningful for\n"
                "  stateful / read_only\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "function_name : str\n"
                "    Exact name of the function to execute.\n"
                "call_kwargs : dict, optional\n"
                "    Keyword arguments to pass to the function. Values keep\n"
                "    the callee's declared types — numbers/booleans unquoted\n"
                '    (``{"max_results": 5}``, not ``{"max_results": "5"}``).\n'
                'language : str, default ``"python"``\n'
                "    Language of the function.\n"
                'state_mode : str, default ``"stateless"``\n'
                "    Execution state mode.\n"
                "session_id : int | None\n"
                "    Session ID for stateful/read_only modes.\n"
                "session_name : str | None\n"
                "    Human-friendly session alias.\n"
                "venv_id : int | None\n"
                "    Virtual environment ID (Python only).\n"
                "\n"
                "Returns\n"
                "-------\n"
                "dict | ExecutionResult\n"
                "    Same shape as code execution output (stdout, stderr, result,\n"
                "    error, language, state_mode, session_id, session_name, venv_id,\n"
                "    session_created, duration_ms).\n"
            )

        integration_summary = ""
        has_integration_packages = False
        try:
            from unify.integration_status import enabled_summary_for_prompt
            from unify.integration_status.discovery import (
                discover_available_packages,
            )

            has_integration_packages = bool(discover_available_packages())
            integration_summary = enabled_summary_for_prompt()
        except Exception:
            integration_summary = ""
        effective_guidelines = (
            "\n\n".join(
                filter(None, [self._base_guidelines, guidelines, integration_summary]),
            )
            or None
        )

        # Workspace-OAuth gate for the OAuth helper section — independent of
        # the integration-packages gate: a workspace-email assistant with
        # zero packages keeps the OAuth section. Cheap in-memory presence
        # check; never forces a network sync.
        has_workspace_oauth = False
        try:
            from unify.common.runtime_oauth import has_workspace_oauth_connection

            has_workspace_oauth = has_workspace_oauth_connection()
        except Exception:
            has_workspace_oauth = False

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] building system prompt")
        system_prompt = build_code_act_prompt(
            environments=sandbox_envs,
            tools=base_tools,
            can_store=effective_can_store,
            guidelines=effective_guidelines,
            discovery_first_policy=self.tool_policy is _USE_DEFAULT,
            include_external_app_integration=has_integration_packages,
            include_oauth_helper=has_workspace_oauth,
            persist=bool(persist),
        )
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] prompt built "
            f"({len(system_prompt)} chars, {len(base_tools)} tools)",
        )

        # Tool policy controls which tools are visible per turn, and whether a
        # tool call is required.  The static _filter_tools (can_compose,
        # can_store, can_spawn_sub_agents) is always applied regardless of
        # the dynamic policy.
        if self.tool_policy is None:
            # No dynamic policy -- only static filtering on every turn.
            def _static_only_policy(step: int, tools: Dict[str, Any]):
                return "auto", _filter_tools(tools)

            tool_policy: Optional[ToolPolicyFn] = _static_only_policy
        elif self.tool_policy is _USE_DEFAULT:
            # Default discovery-first policy (FM + GM + KM gates).
            _has_fm_tools = any(
                isinstance(k, str) and k.startswith("FunctionManager_")
                for k in base_tools.keys()
            )
            _has_gm_tools = any(
                isinstance(k, str) and k.startswith("GuidanceManager_")
                for k in base_tools.keys()
            )
            _has_km_tools = any(
                isinstance(k, str) and k.startswith("KnowledgeManager_")
                for k in base_tools.keys()
            )
            tool_policy = _default_tool_policy(
                _has_fm_tools,
                _has_gm_tools,
                _filter_tools,
                has_km_tools=_has_km_tools,
            )
        else:
            # Custom caller-provided policy.  Wrap it so that _filter_tools
            # is always applied first (static filters are never bypassed).
            _user_policy = self.tool_policy

            def _wrapped_policy(step: int, tools: Dict[str, Any]):
                return _user_policy(step, _filter_tools(tools))

            tool_policy = _wrapped_policy

        # Build an LLM client for this act() call. The profile is per-call so
        # concurrent runs on the same actor can use different models safely.
        client_model = act_llm_profile.model or self._model
        client = new_llm_client(
            client_model,
            purpose="planning",
            origin="CodeActActor.act",
            **act_llm_profile.client_kwargs,
        )
        if system_prompt:
            client.set_system_message(system_prompt)

        # Soft/partial discovery hosts often serialize families under
        # tool_choice=required. Inject a Unify-local completion mutator that
        # appends missing preferred discovery calls for the gated schema.
        if self.tool_policy is _USE_DEFAULT:
            _discovery_mutator = _build_discovery_parallel_mutator()
            _orig_generate = client.generate

            def _generate_with_discovery_mutator(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("completion_mutator", _discovery_mutator)
                return _orig_generate(*args, **kwargs)

            client.generate = _generate_with_discovery_mutator  # type: ignore[method-assign]

        tools = dict(base_tools)

        # Build event bus callbacks for clarification and notification tools
        # (the loop creates the tools; we just provide the event hooks).
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if clarification_enabled:
            # (None, None) still injects request_clarification; the tool then
            # uses per-call hidden queues so CM sees handle._clar_q events.
            _clar_queues = (
                (env_clarification_up_q, env_clarification_down_q)
                if caller_supplied_clarification_queues
                else (None, None)
            )

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        async def _on_notify(message: str):
            try:
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "CodeActActor",
                            "method": "act",
                            "action": "notification",
                            "message": message,
                        },
                    ),
                )
            except Exception:
                pass

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] starting async tool loop")
        run_meter = new_run_meter()
        delegate_token = current_task_execution_delegate.set(task_execution_delegate)
        meter_token = current_run_meter.set(run_meter)
        try:
            handle = start_async_tool_loop(
                client,
                request or initial_prompt,
                tools,
                loop_id=f"CodeActActor.act",
                parent_chat_context=_parent_chat_context,
                interrupt_llm_with_interjections=True,
                log_steps=True,
                tool_policy=tool_policy,
                response_format=response_format,
                persist=persist,
                preprocess_msgs=self._preprocess_msgs,
                prompt_caching=self._prompt_caching,
                extra_ask_tools=self._get_extra_ask_tools(),
                extra_compression_tools=(
                    ["store_skills"] if effective_can_store else None
                ),
                clarification_queues=_clar_queues,
                on_clarification_request=_on_clar_req,
                on_clarification_answer=_on_clar_ans,
                on_notify=_on_notify,
            )
        finally:
            current_run_meter.reset(meter_token)
            current_task_execution_delegate.reset(delegate_token)
        handle.run_meter = run_meter  # type: ignore[attr-defined]
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] loop started, returning handle",
        )

        # Wrap result() to run cleanup when the loop finishes
        _original_result = handle.result

        async def _result_with_cleanup() -> str:
            delegate_token = current_task_execution_delegate.set(
                task_execution_delegate,
            )
            try:
                try:
                    return await _original_result()
                finally:
                    current_task_execution_delegate.reset(delegate_token)
            finally:
                await _cleanup()

        handle.result = _result_with_cleanup  # type: ignore[assignment]

        # Wrap pause()/resume() to propagate to the browser agent
        if self._computer_primitives is not None:
            _cp: ComputerPrimitives = self._computer_primitives
            _original_pause = handle.pause
            _original_resume = handle.resume

            async def _pause_with_propagation(**kwargs: Any) -> None:
                await _original_pause(**kwargs)
                await _cp.pause()

            async def _resume_with_propagation(**kwargs: Any) -> None:
                await _cp.resume()
                await _original_resume(**kwargs)

            handle.pause = _pause_with_propagation  # type: ignore[assignment]
            handle.resume = _resume_with_propagation  # type: ignore[assignment]

            # Register the loop's interject queue so environmental state
            # changes (e.g. user remote control) are broadcast to this actor.
            _cp.register_interject_queue(handle._queue)
            _registered_queue[0] = handle._queue

        # Update agent context with handle reference
        new_ctx.handle = handle

        post_run_review_context = current_post_run_review_context.get()

        # Wrap in StorageCheckHandle for post-completion function review.
        # Persistent sessions additionally review at each completed turn —
        # a persist loop never self-completes, so without turn reviews a
        # recurring conversational deliverable would never distill.
        if effective_can_store or post_run_review_context is not None:
            handle = _StorageCheckHandle(
                inner=handle,
                actor=self,
                post_run_review_context=post_run_review_context,
                meter=run_meter,
                turn_reviews_enabled=effective_can_store and bool(persist),
            )
            # Tracked so ``close()`` can end a review still in flight. The
            # set is weak: a finished handle the caller has dropped must not
            # be kept alive by this bookkeeping.
            self._live_storage_handles.add(handle)

        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        # End any storage review still running before the resources it needs
        # are torn down below. Left alone, a review outlives the actor: it
        # keeps issuing inference against a closed venv pool and dead
        # sandboxes, which is how offline task pods stayed busy for days after
        # their run had already been recorded as finished.
        for storage_handle in list(self._live_storage_handles):
            await storage_handle.abandon_storage_review(
                reason="The actor running this review is shutting down.",
            )
        self._live_storage_handles.clear()

        # Close any in-process session sandboxes owned by the session executor.
        try:
            await self._session_executor.close()
        except Exception:
            pass

        # Clear session name registry.
        try:
            self._session_names.clear()
            self._session_names_rev.clear()
        except Exception:
            pass

        # Close the pools (terminates persistent subprocess/session connections)
        await self._venv_pool.close()
        await self._shell_pool.close()

        # The ComputerPrimitives backend is a process-wide singleton (one VM,
        # one screen).  Individual actors must not tear it down — the process
        # owns the lifecycle.
