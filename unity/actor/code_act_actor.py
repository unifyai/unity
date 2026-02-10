import asyncio
import contextvars
import copy
import functools
import inspect
import json
import traceback
import uuid
from datetime import datetime, timezone
from secrets import token_hex as _token_hex
import logging
from typing import (
    Any,
    Callable,
    Awaitable,
    Dict,
    Optional,
    Type,
    TYPE_CHECKING,
)
from pydantic import BaseModel

from unity.actor.base import BaseCodeActActor
from unity.actor.execution import (
    ExecutionResult,
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _validate_execution_params,
)
from unity.common.async_tool_loop import (
    AsyncToolLoopHandle,
    ChatContextPropagation,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.common.clarification_tools import add_clarification_tool_with_events
from unity.common.llm_client import new_llm_client
from unity.function_manager.primitives import ComputerPrimitives
from unity.actor.prompt_builders import build_code_act_prompt
from unity.events.manager_event_logging import log_manager_call
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from unity.common.hierarchical_logger import (
    build_hierarchy_label,
    log_boundary_event,
)
from unity.events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
)

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
    from unity.function_manager.function_manager import FunctionManager
    from unillm.types import PromptCacheParam


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
    `actor.act(task_description, entrypoint=<function_id>, persist=False)`.

    When an `entrypoint` is provided, CodeActActor resolves the function by id,
    injects it into the sandbox namespace, and executes it in an asyncio task.
    """

    def __init__(
        self,
        *,
        entrypoint_id: int,
        execution_task: asyncio.Task[Any],
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._entrypoint_id = int(entrypoint_id)
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._stopped = False
        self._on_finally = on_finally

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self) -> None:
        try:
            out = await self._execution_task
            if not self._stopped:
                self._result_str = str(out) if out is not None else ""
        except asyncio.CancelledError:
            self._stopped = True
            self._result_str = f"Entrypoint {self._entrypoint_id} was cancelled."
        except Exception as e:
            self._result_str = f"Error: {e}"
        finally:
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception:
                    pass
            self._completion_event.set()

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        status = "completed" if self.done() else "still running"
        client = new_llm_client()
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
            timeout=30,
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
        return self._result_str or ""

    async def next_clarification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Storage check: start a review loop and return its handle
# ---------------------------------------------------------------------------


def _start_storage_check_loop(
    *,
    trajectory: list[dict],
    ask_tools: dict,
    actor: "CodeActActor",
    original_result: str,
) -> "AsyncToolLoopHandle | None":
    """Start a loop that reviews a completed trajectory for reusable functions.

    Returns the loop handle so the caller can steer and await it, or
    ``None`` when there is no ``FunctionManager`` configured.
    """
    fm = actor.function_manager
    if fm is None:
        return None

    # ── Build sandbox-free FunctionManager tools ──────────────────────

    async def FunctionManager_search_functions(
        query: str,
        n: int = 5,
    ) -> Any:
        """Search for existing stored functions by semantic similarity."""
        return fm.search_functions(
            query=query,
            n=n,
            include_implementations=True,
        )

    async def FunctionManager_filter_functions(
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Any:
        """Filter existing stored functions using a filter expression."""
        return fm.filter_functions(
            filter=filter,
            offset=offset,
            limit=limit,
            include_implementations=True,
        )

    async def FunctionManager_list_functions(
        include_implementations: bool = False,
    ) -> Any:
        """List all stored functions."""
        return fm.list_functions(
            include_implementations=include_implementations,
        )

    async def FunctionManager_add_functions(
        implementations: str | list[str],
        *,
        language: str = "python",
        overwrite: bool = False,
    ) -> Any:
        """Store new reusable functions, or update existing ones by name.

        When ``overwrite=True`` and a function with the same name already
        exists, its implementation is replaced with the new one.
        """
        return fm.add_functions(
            implementations=implementations,
            language=language,
            overwrite=bool(overwrite),
        )

    async def FunctionManager_delete_functions(
        function_ids: list[int],
    ) -> Any:
        """Delete functions by their IDs.

        Use this to remove obsolete, redundant, or superseded functions
        from the store. Obtain ``function_id`` values from the results
        of search, filter, or list calls.
        """
        return fm.delete_function(function_id=function_ids)

    tools: Dict[str, Callable] = {
        "FunctionManager_search_functions": FunctionManager_search_functions,
        "FunctionManager_filter_functions": FunctionManager_filter_functions,
        "FunctionManager_list_functions": FunctionManager_list_functions,
        "FunctionManager_add_functions": FunctionManager_add_functions,
        "FunctionManager_delete_functions": FunctionManager_delete_functions,
    }

    # ── Wire ask_about_completed_tool from snapshot ───────────────────

    if ask_tools:
        completed_info: list[str] = []
        for name, fn in ask_tools.items():
            completed_info.append(f"- `{name}`")

        async def ask_about_completed_tool(
            tool_name: str,
            question: str,
        ) -> str:
            """Ask a follow-up question about a completed tool from the trajectory.

            Use this to inspect the internal reasoning or detailed results of
            any tool that ran during the completed execution.
            """
            fn = ask_tools.get(tool_name)
            if fn is None:
                return f"Tool '{tool_name}' not found. Available: {list(ask_tools.keys())}"
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

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(trajectory, indent=2, default=str)

    system_prompt = (
        "You are a function librarian. A CodeActActor has just completed a task. "
        "Your job is to review the execution trajectory and maintain the function "
        "library: store valuable new functions, improve existing ones, merge "
        "redundant entries, and remove obsolete ones.\n\n"
        "## Completed Trajectory\n\n"
        f"{trajectory_json}\n\n"
        "## Final Result\n\n"
        f"{original_result}\n\n"
        "## Instructions\n\n"
        "1. Review the trajectory for any Python functions that were composed "
        "and executed during the task.\n"
        "2. Search the existing function store "
        "(via `FunctionManager_search_functions`) to understand what already "
        "exists.\n"
        "3. Decide what actions (if any) would improve the library. You can:\n"
        "   - **Add** a genuinely new, reusable function "
        "(`FunctionManager_add_functions`).\n"
        "   - **Update** an existing function with a better implementation "
        "(`FunctionManager_add_functions` with `overwrite=True`).\n"
        "   - **Merge** two or more overlapping functions into a single, "
        "more general one: add the merged version, then delete the old "
        "entries (`FunctionManager_delete_functions`).\n"
        "   - **Delete** functions that are now redundant or superseded "
        "(`FunctionManager_delete_functions`).\n"
        "4. Prefer a clean, non-redundant library over a large one. Merging "
        "two similar functions into one general-purpose function is better "
        "than keeping both.\n"
        "5. Do NOT store trivial one-liners, test scaffolding, or functions "
        "that are too specific to this particular task to be reusable.\n"
        "6. When done (or if there is nothing worth changing), respond with a "
        "brief summary of what you did (or that nothing was needed)."
    )

    client = new_llm_client(
        actor._model,
        reasoning_effort=None,
        service_tier=None,
    )
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message="Review the trajectory and store any reusable functions.",
        tools=tools,
        loop_id="StorageCheck(CodeActActor.act)",
        max_steps=30,
        timeout=120,
    )


class _StorageCheckHandle(SteerableToolHandle):
    """Wraps an inner handle and runs a storage check after task completion.

    Lifecycle phases:

    * **task** -- the inner tool loop is running.  All steering methods
      forward to the inner handle.  Notifications from the inner handle
      are relayed to consumers.
    * **storage** -- the task has completed.  A notification carrying the
      original result has been emitted.  A second loop reviews the
      trajectory for reusable skills.  Steering operates on the storage
      loop.
    * **done** -- both phases have completed (or were stopped/skipped).
      ``result()`` resolves with the original task result.
    """

    def __init__(
        self,
        *,
        inner: "AsyncToolLoopHandle",
        actor: "CodeActActor",
    ) -> None:
        self._inner = inner
        self._actor = actor
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        self._completion_event = asyncio.Event()
        self._original_result: Optional[str] = None
        self._storage_handle: Optional["AsyncToolLoopHandle"] = None
        self._phase: str = "task"  # "task" | "storage" | "done"
        self._stopped: bool = False
        self._active_relay: Optional[asyncio.Task] = None

        # Start the two-phase lifecycle manager.
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())

    # ── Internal helpers ──────────────────────────────────────────────

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
        """Forward notifications from *source* into our queue until cancelled."""
        try:
            while True:
                notif = await source.next_notification()
                await self._notification_q.put(notif)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

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
        try:
            # ── Phase 1: task execution ───────────────────────────────
            self._active_relay = asyncio.create_task(
                self._relay_notifications_from(self._inner),
            )

            self._original_result = await self._inner.result()
            await self._cancel_relay()

            if self._stopped:
                return

            # Snapshot trajectory and ask tools (client/messages are still
            # valid after result() returns -- cleanup only resets context
            # vars and releases the semaphore).
            trajectory: list[dict] = []
            ask_tools: dict = {}
            try:
                client = getattr(self._inner, "_client", None)
                if client is not None:
                    trajectory = list(getattr(client, "messages", []) or [])
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

            # ── Transition: notify consumers ──────────────────────────
            self._phase = "storage"
            await self._notification_q.put(
                {
                    "type": "task_completed",
                    "result": self._original_result,
                    "message": (
                        f"Task completed with result:\n\n"
                        f"{self._original_result}\n\n"
                        f"The agent is now reviewing its execution "
                        f"trajectory to store reusable skills. This "
                        f"handle will remain active until skill "
                        f"consolidation finishes."
                    ),
                },
            )

            # ── Phase 2: storage check ────────────────────────────────
            storage_handle = _start_storage_check_loop(
                trajectory=trajectory,
                ask_tools=ask_tools,
                actor=self._actor,
                original_result=str(self._original_result),
            )

            if storage_handle is None:
                return

            self._storage_handle = storage_handle
            self._active_relay = asyncio.create_task(
                self._relay_notifications_from(self._storage_handle),
            )

            try:
                await self._storage_handle.result()
            except Exception:
                pass

            await self._cancel_relay()

        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._phase = "done"
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

        routing_client = new_llm_client()
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
            max_steps=5,
            timeout=60,
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

    async def result(self) -> str:
        await self._completion_event.wait()
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


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        computer_mode: str = "magnitude",
        timeout: float = 1000,
        agent_mode: str = "web",
        agent_server_url: str | None = None,
        computer_primitives: Optional["ComputerPrimitives"] = None,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        can_compose: bool = True,
        can_store: bool = True,
        can_spawn_sub_agents: bool = True,
        storage_check_on_return: bool = False,
        model: Optional[str] = None,
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
        prompt_caching: Optional["PromptCacheParam"] = None,
    ):
        """
        Initializes the CodeActActor.

        Args:
            computer_primitives: Optional existing ComputerPrimitives instance to reuse.
                           If provided, other computer-related params are ignored.
            environments: Optional list of execution environments. If None, defaults to
                [ComputerEnvironment, StateManagerEnvironment].
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
            agent_server_url: URL for the agent server. For desktop mode, pass the
                external VM's URL.
            can_spawn_sub_agents: When True, exposes a ``run_sub_agent`` tool that
                lets the LLM spawn inner CodeActActors to work on focused sub-tasks.
            model: Optional LLM model identifier (e.g. "claude-4.5-opus@anthropic").
                If None, uses SETTINGS.UNIFY_MODEL (default: "claude-4.5-opus@anthropic").
            preprocess_msgs: Optional callback to modify messages before each LLM call.
                Receives a list of message dicts and returns a modified list.
                Useful for pruning old messages, adding context, or transforming content.
            prompt_caching: Optional list of cache targets (e.g. ["system", "messages"]).
                Enables Anthropic prompt caching for the specified components to reduce
                costs and latency. Valid values: "tools", "system", "messages".
        """
        super().__init__(
            environments=environments,
            computer_primitives=computer_primitives,
            function_manager=function_manager,
            session_connect_url=session_connect_url,
            headless=headless,
            computer_mode=computer_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
        )

        # Create persistent pools that survive across act() calls
        from unity.function_manager.function_manager import VenvPool
        from unity.function_manager.shell_pool import ShellPool

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

        self._timeout = timeout
        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self.can_spawn_sub_agents: bool = bool(can_spawn_sub_agents)
        self.storage_check_on_return: bool = bool(storage_check_on_return)
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
            max_sessions_total=self._max_sessions_total,
            active_session_count=self._count_active_sessions_total(),
        )

    def _get_computer_tools(self) -> Dict[str, Callable]:
        """Extracts computer-related methods from the ComputerPrimitives."""
        if not self._computer_primitives:
            return {}
        return {
            "navigate": self._computer_primitives.navigate,
            "act": self._computer_primitives.act,
            "observe": self._computer_primitives.observe,
        }

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        async def execute_code(
            thought: str,
            code: Optional[str] = None,
            *,
            language: str,
            state_mode: str = "stateless",
            session_id: int | None = None,
            session_name: str | None = None,
            venv_id: int | None = None,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _parent_chat_context: list[dict] | None = None,
        ) -> Any:
            """
            Execute code in a specified language and state mode, optionally within a session.

            Use this tool for BOTH Python and shell execution. This is the unified
            "brain execution" tool for CodeActActor.

            Key concepts
            -----------
            - **language**: "python" | "bash" | "zsh" | "sh" | "powershell"
            - **state_mode**:
              - "stateless": no session; clean execution; no persistence
              - "stateful": persistent session; state accumulates
              - "read_only": reads from an existing session but does not persist changes
            - **session_id/session_name**:
              - only meaningful for stateful/read_only
              - for stateful: if omitted, defaults to **session_id=0** (the default session)
              - to create an additional stateful session, provide a fresh `session_name` (recommended)
                or an explicit `session_id` > 0
              - **Python session_id=0** is special:
                - If this tool is called from inside a running CodeAct `act()` loop, session 0 maps to the
                  **current per-call Python sandbox** (shared via the ContextVar binding).
                - If no sandbox is bound (e.g. calling the tool directly in a unit test), session 0 behaves
                  like a normal in-process Python session managed by the SessionExecutor.

            Best practices
            --------------
            - Use **stateful** when doing multi-step work (cd then ls; load data then analyze).
            - Use **stateless** for one-off checks or when you need isolation.
            - Use **read_only** to "peek" without mutating state (what-if exploration).
            - Use `list_sessions()` and `inspect_state()` to decide which session to use.

            Steerable handles
            -----------------
            State manager primitives (e.g. ``primitives.contacts.ask(...)``)
            return **steerable handles** — live in-flight operations that can
            be paused, resumed, interjected, or stopped from the outer loop.

            You control whether to steer an operation or let it complete
            internally, depending on what the task requires:

            - **Return the handle as the last expression** to hand control
              back to the outer loop. The handle is automatically adopted:
              progress is shown as intermediate output, dynamic steering
              helpers (``stop_*``, etc.) become available, and the final
              result replaces the placeholder when the inner operation
              completes. Use this when you may need to steer, monitor, or
              cancel the operation mid-flight.

              .. code-block:: python

                  # Last expression is the handle → adopted for steering
                  await primitives.contacts.ask(text="Who is Alice?")

            - **Await the result inside the code** to let the operation
              complete before returning. The outer loop receives only the
              finished result with no steering opportunity. Use this when
              you need the answer immediately for further processing within
              the same code block.

              .. code-block:: python

                  # Blocks until done; result available for further logic
                  handle = await primitives.contacts.ask(text="Who is Alice?")
                  answer = await handle.result()
                  print(f"Found: {answer}")

            Output
            ------
            Returns either a dict or an ExecutionResult object with the following fields:

            - **stdout**: For in-process Python, a List[TextPart | ImagePart] preserving
              rich output (text and images from print()/display()). For shell or venv
              execution, a plain string.
            - **stderr**: Same format as stdout (list for in-process Python, string otherwise).
            - **result**: The evaluated result of the last expression (Any), or None.
              If the last expression is a steerable handle, it is automatically
              adopted by the outer loop for mid-flight steering (see above).
            - **error**: Error message string if execution failed, otherwise None.
            - **language**: The language used for execution.
            - **state_mode**: The state mode used ("stateless", "stateful", or "read_only").
            - **session_id**: The session ID (int) if stateful/read_only, otherwise None.
            - **session_name**: The session name alias if one was assigned, otherwise None.
            - **venv_id**: The virtual environment ID if applicable, otherwise None.
            - **session_created**: True if a new session was created by this call.
            - **duration_ms**: Execution duration in milliseconds.
            - **computer_used**: True if computer primitives were invoked during execution.
            - **computer_state** (optional): Only present when computer_used is True and a
              computer environment is available. The textual metadata includes the URL
              and any error details. A screenshot, when available, is returned as an
              image block in the formatted tool output rather than embedded into JSON.

            For in-process Python execution with rich output, the result is wrapped in an
            ExecutionResult object (a Pydantic model implementing FormattedToolResult).
            """
            _ = thought  # Thought is logged by the LLM; not used programmatically.
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
                    "computer_used": False,
                }

            # ──────────────────────────────────────────────────────────────
            # Boundary wrapper: execute_code (lineage + events + terminal log)
            # ──────────────────────────────────────────────────────────────

            _suffix = _token_hex(2)
            _call_id = new_call_id()
            _parent = TOOL_LOOP_LINEAGE.get([])
            _parent_lineage = list(_parent) if isinstance(_parent, list) else []
            _hierarchy = [*_parent_lineage, "execute_code"]
            _hierarchy_label = build_hierarchy_label(_hierarchy, _suffix)
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
                        hierarchy_label=_hierarchy_label,
                        **payload,
                    )
                except Exception as e:
                    log_boundary_event(
                        _hierarchy_label,
                        f"Warning: failed to publish event: {type(e).__name__}: {e}",
                        icon="⚠️",
                        level="warning",
                    )

            try:
                await _pub_safe(phase="incoming")
            except Exception:
                pass
            log_boundary_event(_hierarchy_label, "Executing code...", icon="🛠️")

            out: dict[str, Any] | None = None
            tb_str: str | None = None
            exec_exc: Exception | None = None

            notification_q = _notification_up_q
            sandbox_id = None
            try:
                # Resolve / allocate sessions for stateful.
                if state_mode == "stateful":
                    if session_name:
                        resolved = self._resolve_session_name(session_name)
                        if resolved is not None:
                            language, venv_id, session_id = resolved
                        elif session_id is None:
                            # Allocate a new session id (reserve 0 for default sandbox).
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
                        # Default stateful session for each language/venv is session_id=0.
                        # This is especially important for Python because FunctionManager-injected
                        # callables are available in the default/bound Python sandbox (session 0).
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

                # Validate.
                err = self._validate_execution_params(
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=str(language),
                    venv_id=venv_id,
                )
                if err is not None:
                    out = err
                    return out

                # Inject per-tool notification queue into bound sandbox so notify() works.
                try:
                    sb_for_notifs = _CURRENT_SANDBOX.get()
                    sandbox_id = getattr(sb_for_notifs, "id", None)
                    if notification_q is not None:
                        sb_for_notifs.global_state["__notification_up_q__"] = (
                            notification_q
                        )
                except Exception:
                    pass

                if notification_q is not None and str(language) == "python":
                    try:
                        await notification_q.put(
                            {
                                "type": "execution_started",
                                "message": "execution_started",
                                "sandbox_id": sandbox_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        pass

                # Execute via SessionExecutor. Route primitives if available in current sandbox.
                primitives = None
                computer_primitives = self._computer_primitives
                try:
                    sb = _CURRENT_SANDBOX.get()
                    primitives = sb.global_state.get("primitives")
                    computer_primitives = sb.global_state.get(
                        "computer_primitives",
                        computer_primitives,
                    )
                except Exception:
                    pass

                _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                try:
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
                        if notification_q is not None and str(language) == "python":
                            try:
                                await notification_q.put(
                                    {
                                        "type": "execution_error",
                                        "message": "execution_error",
                                        "sandbox_id": sandbox_id,
                                        "error_kind": "exception",
                                        "traceback_preview": tb[:2000],
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    },
                                )
                            except Exception:
                                pass
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
                            "computer_used": False,
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

                # Attach computer state for Python runs when computer primitives were used.
                if (
                    str(out.get("language")) == "python"
                    and out.get("computer_used")
                    and self._computer_primitives is not None
                ):
                    try:
                        url = await self._computer_primitives.computer.get_current_url()
                        screenshot_b64 = (
                            await self._computer_primitives.computer.get_screenshot()
                        )
                        out["computer_state"] = {
                            "url": url,
                            "screenshot": screenshot_b64,
                        }
                    except Exception as e:
                        out["computer_state"] = {"error": str(e)}

                if notification_q is not None and str(language) == "python":
                    try:
                        _status = "ok" if not out.get("error") else "error"
                        await notification_q.put(
                            {
                                "type": "execution_finished",
                                "sandbox_id": sandbox_id,
                                "status": _status,
                                "message": f"execution_finished:{_status}",
                                "stdout_len": len(out.get("stdout") or ""),
                                "stderr_len": len(out.get("stderr") or ""),
                                "computer_used": bool(out.get("computer_used")),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        pass

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
                try:
                    if out is not None and out.get("error"):
                        await _pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(out.get("error")),
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

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_code": execute_code,
        }

        # Add FunctionManager tools (auto-inject callables into sandbox) if available.
        #
        # IMPORTANT:
        # These tools are called via JSON tool calls (not inside Python). They return
        # metadata to the LLM while injecting the matching function callables into the
        # sandbox global namespace so they can be executed immediately in Python code.
        if self.function_manager:

            async def FunctionManager_search_functions(
                query: str,
                n: int = 5,
            ) -> Any:
                """
                Search for functions by semantic similarity to a natural-language query.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after searching.
                """
                result = self.function_manager.search_functions(
                    query=query,
                    n=n,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_filter_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
            ) -> Any:
                """
                Filter functions using a Python-like filter expression.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after filtering.
                """
                result = self.function_manager.filter_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_list_functions(
                include_implementations: bool = False,
            ) -> Any:
                """
                List available functions.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after listing.
                """
                result = self.function_manager.list_functions(
                    include_implementations=include_implementations,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            tools["FunctionManager_search_functions"] = FunctionManager_search_functions
            tools["FunctionManager_filter_functions"] = FunctionManager_filter_functions
            tools["FunctionManager_list_functions"] = FunctionManager_list_functions

            async def FunctionManager_add_functions(
                implementations: str | list[str],
                *,
                language: str = "python",
                overwrite: bool = False,
                verify: Optional[dict[str, bool]] = None,
                preconditions: Optional[dict[str, dict]] = None,
            ) -> Any:
                """
                Add/store new functions into the FunctionManager.

                Notes
                -----
                - This tool is gated by CodeActActor's `can_store` flag (and can be disabled per-call).
                - Prefer using existing functions (search first) before adding new ones.
                """
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "FunctionManager is not configured on this actor.",
                    )
                return fm.add_functions(
                    implementations=implementations,
                    language=language,  # type: ignore[arg-type]
                    overwrite=bool(overwrite),
                    verify=(verify or {}),
                    preconditions=(preconditions or {}),
                )

            tools["FunctionManager_add_functions"] = FunctionManager_add_functions

            async def execute_function(
                function_name: str,
                call_kwargs: Optional[Dict[str, Any]] = None,
                _parent_chat_context: list[dict] | None = None,
            ) -> Dict[str, Any]:
                """
                Execute a stored function by name and return its result.

                This is the preferred way to run a known, pre-stored function
                without writing any code. The function is looked up in the
                FunctionManager by exact name and executed directly.

                When to use this vs execute_code
                ---------------------------------
                - Use **execute_function** when you know the exact function name
                  (from a prior search/list/filter call) and want to run it as-is.
                  This is simpler, faster, and avoids the overhead of a code sandbox.
                - Use **execute_code** when you need to compose multiple functions,
                  transform data between calls, or write any custom logic.

                Workflow
                -------
                1. Discover functions via `FunctionManager_search_functions`,
                   `FunctionManager_filter_functions`, or
                   `FunctionManager_list_functions`.
                2. Pick the best match by name.
                3. Call `execute_function(function_name=..., call_kwargs=...)`.

                Parameters
                ----------
                function_name : str
                    Exact name of the stored function to execute (as returned by
                    the FunctionManager discovery tools).
                call_kwargs : dict, optional
                    Keyword arguments to pass to the function. Omit or pass None /
                    an empty dict for functions that take no arguments.

                Returns
                -------
                dict
                    A dict with:
                    - **result**: The function's return value (any JSON-serializable type).
                    - **error**: Traceback string if the function raised, else None.
                    - **stdout**: Captured standard output (string).
                    - **stderr**: Captured standard error (string).
                """
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "FunctionManager is not configured on this actor.",
                    )

                # ── Lineage boundary (mirrors execute_code) ──────────────
                _ef_suffix = _token_hex(2)
                _ef_call_id = new_call_id()
                _ef_parent = TOOL_LOOP_LINEAGE.get([])
                _ef_parent_lineage = (
                    list(_ef_parent) if isinstance(_ef_parent, list) else []
                )
                _ef_hierarchy = [
                    *_ef_parent_lineage,
                    f"execute_function({function_name})",
                ]
                _ef_hierarchy_label = build_hierarchy_label(
                    _ef_hierarchy,
                    _ef_suffix,
                )
                _ef_lineage_token = TOOL_LOOP_LINEAGE.set(_ef_hierarchy)

                async def _ef_pub_safe(**payload: Any) -> None:
                    try:
                        await publish_manager_method_event(
                            _ef_call_id,
                            "CodeActActor",
                            "execute_function",
                            hierarchy=_ef_hierarchy,
                            hierarchy_label=_ef_hierarchy_label,
                            **payload,
                        )
                    except Exception as e:
                        log_boundary_event(
                            _ef_hierarchy_label,
                            f"Warning: failed to publish event: {type(e).__name__}: {e}",
                            icon="⚠️",
                            level="warning",
                        )

                try:
                    await _ef_pub_safe(phase="incoming")
                except Exception:
                    pass
                log_boundary_event(
                    _ef_hierarchy_label,
                    f"Executing function {function_name}...",
                    icon="🛠️",
                )

                primitives = None
                try:
                    env = self.environments.get("primitives")
                    if env is not None:
                        primitives = env.get_instance()
                except Exception:
                    primitives = None

                try:
                    result = await fm.execute_function(
                        function_name=function_name,
                        call_kwargs=call_kwargs,
                        primitives=primitives,
                        computer_primitives=self._computer_primitives,
                        venv_pool=self._venv_pool,
                        shell_pool=self._shell_pool,
                        state_mode="stateless",
                        _parent_chat_context=_parent_chat_context,
                    )
                except Exception as exc:
                    try:
                        await _ef_pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )
                    except Exception:
                        pass
                    raise
                else:
                    try:
                        await _ef_pub_safe(phase="outgoing", status="ok")
                    except Exception:
                        pass
                    return result
                finally:
                    try:
                        TOOL_LOOP_LINEAGE.reset(_ef_lineage_token)
                    except Exception:
                        pass

            tools["execute_function"] = execute_function

        # ───────────────────────── Session management tools ────────────────── #

        async def list_sessions(detail: str = "summary") -> Dict[str, Any]:
            """
            List all active sessions across all languages (Python + shell).

            Use this tool whenever you need to choose which session to use for a
            subsequent `execute_code(..., state_mode="stateful"/"read_only")` call.

            Parameters
            ----------
            detail:
                Controls how much information is returned per session:
                - "summary": metadata + a short `state_summary` string (default)
                - "full": best-effort enrichment using cheap inspection where available

            Returns
            -------
            dict:
                {"sessions": [ ... ]} where each entry includes (best-effort):
                - language: "python" | "bash" | "zsh" | "sh" | "powershell"
                - session_id: int (scoped per language + venv_id)
                - venv_id: int | None (Python only)
                - session_name: optional human-friendly alias (if registered)
                - created_at / last_used: timestamps when available
                - state_summary: a short human-readable summary (e.g. "3 names", "cwd=/repo")

            Notes
            -----
            - Session IDs are **scoped per (language, venv_id)**, so `python` session 0 and
              `bash` session 0 can coexist.
            - The default per-call Python sandbox is exposed as `python` session_id=0 (venv_id=None)
              when it is bound for the current call.
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

            This tool is for debugging and for deciding whether to:
            - continue in the same session (stateful)
            - start a fresh session (stateful without session_id/session_name)
            - run a one-off (stateless)
            - do a what-if (read_only)

            Parameters
            ----------
            session_name:
                Optional human-friendly alias for a session (preferred when available).
            session_id + language (+ optional venv_id):
                Directly identify a session. `session_id` is scoped per (language, venv_id).
            detail:
                "summary" | "names" | "full"
                - Prefer "summary" when you just need quick context.
                - Use "names" to see what variables exist without dumping values.
                - Use "full" sparingly (can be large; values are truncated/redacted best-effort).

            Defaults
            --------
            If no session selector is provided, this inspects the **current per-call Python sandbox**
            (python session_id=0, venv_id=None) when bound.

            Returns
            -------
            dict with:
            - session: {language, session_id, session_name, venv_id}
            - state: implementation-specific state representation (Python vars; shell cwd/env/functions/aliases)
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

            Use this to proactively manage resources when you are done with a session.
            This operation is **idempotent**: closing an already-closed/non-existent session
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
                - closed: bool
                - reason: "success" | "not_found" | "error"
                - session: {language, session_id, session_name}
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

            This is a blunt cleanup tool. Prefer `close_session(...)` when you only
            want to discard a specific polluted/unused session.

            Returns
            -------
            dict:
                - closed_count: int
                - languages: list[str] (languages that had sessions closed)
                - details: per-language counts
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

        tools["list_sessions"] = list_sessions
        tools["inspect_state"] = inspect_state
        tools["close_session"] = close_session
        tools["close_all_sessions"] = close_all_sessions

        # ───────────────────────── Sub-agent delegation tool ─────────────── #

        async def run_sub_agent(
            task: str,
            *,
            timeout: float | None = None,
            _parent_chat_context: list[dict] | None = None,
        ) -> str:
            """
            Spawn a sub-agent to work on a focused sub-task.

            The sub-agent is an independent CodeActActor with the same tools and
            environment access as the current agent. It runs the given task to
            completion and returns the final result as a string.

            When to use
            -----------
            - The overall task decomposes naturally into independent sub-problems
              that benefit from focused, isolated reasoning.
            - A sub-task requires multi-step work that would clutter or distract
              the main agent's context window.
            - You want to isolate a sub-task's execution state (sessions,
              variables) from the main agent's sandbox.

            When NOT to use
            ---------------
            - The task is simple enough to handle directly with ``execute_code``.
            - You need intermediate results from the sub-task to inform logic in
              the same code block (use ``execute_code`` with stateful sessions).
            - The sub-task is trivial (single tool call) — the overhead of a
              sub-agent is not worth it.

            Parameters
            ----------
            task : str
                A clear, self-contained description of what the sub-agent should
                accomplish. Be specific and include all necessary context, because
                the sub-agent does **not** share the parent agent's conversation
                history or session state.
            timeout : float, optional
                Maximum seconds for the sub-agent to complete. Defaults to half
                the parent agent's timeout, capped at 300 seconds.

            Returns
            -------
            str
                The sub-agent's final answer / result.
            """
            effective_timeout = (
                timeout
                if timeout is not None
                else min(self._timeout / 2, 300)
            )

            handle = await self.act(
                task,
                clarification_enabled=False,
                can_compose=True,
                can_store=False,
                can_spawn_sub_agents=False,
                storage_check_on_return=False,
            )

            try:
                result = await asyncio.wait_for(
                    handle.result(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                await handle.stop("Sub-agent timed out")
                result = (
                    f"Sub-agent timed out after {effective_timeout}s. "
                    f"The sub-task may have been too broad or complex."
                )

            return result

        tools["run_sub_agent"] = run_sub_agent

        return tools

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call("CodeActActor", "act", payload_key="description")
    async def act(
        self,
        description: str | dict | list[str | dict],
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        persist: Optional[bool] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        can_spawn_sub_agents: Optional[bool] = None,
        storage_check_on_return: Optional[bool] = None,
        **kwargs,
    ) -> SteerableToolHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        effective_can_compose = (
            self.can_compose if can_compose is None else bool(can_compose)
        )
        effective_can_store = self.can_store if can_store is None else bool(can_store)
        effective_can_spawn_sub_agents = (
            self.can_spawn_sub_agents
            if can_spawn_sub_agents is None
            else bool(can_spawn_sub_agents)
        )
        effective_storage_check = (
            self.storage_check_on_return
            if storage_check_on_return is None
            else bool(storage_check_on_return)
        )
        # storage_check_on_return only applies when both can_compose and can_store are True.
        if effective_storage_check and (
            not effective_can_compose or not effective_can_store
        ):
            effective_storage_check = False

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

        # Clarification queues:
        # - When enabled, we ensure the handle has queues (either provided by caller or newly created).
        # - When disabled, we do not provide queues and we do not wire queue injection into environments.
        clarification_up_q: Optional[asyncio.Queue[str]]
        clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            clarification_up_q = _clarification_up_q or asyncio.Queue()
            clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            clarification_up_q = None
            clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unity.actor.environments import (
                ComputerEnvironment as _ComputerEnvironment,
                StateManagerEnvironment as _StateManagerEnvironment,
            )
        except Exception:
            _ComputerEnvironment = None  # type: ignore
            _StateManagerEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _ComputerEnvironment is not None and isinstance(
                    env,
                    _ComputerEnvironment,
                ):
                    sandbox_envs[ns] = _ComputerEnvironment(
                        env.get_instance(),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
                if _StateManagerEnvironment is not None and isinstance(
                    env,
                    _StateManagerEnvironment,
                ):
                    sandbox_envs[ns] = _StateManagerEnvironment(
                        env.get_instance(),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard. If we can't acquire within 30s, treat as resource exhaustion.
        try:
            await asyncio.wait_for(
                self._act_semaphore.acquire(),
                timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "CodeActActor is at capacity (too many concurrent sessions). "
                "Try again later or reduce concurrency.",
            )
        sandbox = PythonExecutionSession(
            computer_primitives=self._computer_primitives,
            environments=sandbox_envs,
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
        )
        # Note: __notification_up_q__ is injected dynamically by execute_code
        # when it receives a _notification_up_q from the async tool loop.
        token = _CURRENT_SANDBOX.set(sandbox)

        # Set agent context for depth tracking and handle access
        parent_ctx = _CURRENT_AGENT_CONTEXT.get()
        new_ctx = AgentContext(
            depth=parent_ctx.depth + 1,
            agent_id=str(uuid.uuid4()),
            handle=None,  # Will be set after handle is created
        )
        ctx_token = _CURRENT_AGENT_CONTEXT.set(new_ctx)

        async def _cleanup() -> None:
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
                _CURRENT_AGENT_CONTEXT.reset(ctx_token)
            except Exception:
                pass
            try:
                self._act_semaphore.release()
            except Exception:
                pass

        # If an explicit FunctionManager entrypoint is provided (e.g., TaskScheduler task execution),
        # bypass the CodeAct LLM loop and run the function directly.
        if entrypoint is not None:
            entrypoint_id = int(entrypoint)
            args = list(entrypoint_args or [])
            kwargs_for_entrypoint = dict(entrypoint_kwargs or {})

            async def _run_entrypoint() -> Any:
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "CodeActActor cannot execute entrypoint: function_manager is None",
                    )

                out = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    return_callable=True,
                    namespace=sandbox.global_state,
                    also_return_metadata=True,
                )
                metadata = []
                if isinstance(out, dict):
                    metadata = list(out.get("metadata") or [])
                if not metadata:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                fn_name = metadata[0].get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} has no valid function name.",
                    )
                fn = sandbox.global_state.get(fn_name)
                if fn is None:
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} ({fn_name}) was not injected into the sandbox namespace.",
                    )

                res = fn(*args, **kwargs_for_entrypoint)
                if inspect.isawaitable(res):
                    res = await res
                return res

            entry_task = asyncio.create_task(_run_entrypoint())
            entry_handle = _CodeActEntrypointHandle(
                entrypoint_id=entrypoint_id,
                execution_task=entry_task,
                on_finally=_cleanup,
            )
            return entry_handle

        # Build the tool set for this call. When can_compose=False the LLM
        # may only discover and execute stored functions -- no arbitrary code,
        # no session management, no function persistence.
        _code_and_session_tools = {
            "execute_code",
            "list_sessions",
            "inspect_state",
            "close_session",
            "close_all_sessions",
        }

        def _filter_tools(tool_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Apply static per-call filters (can_compose, can_store, can_spawn_sub_agents)."""
            out = dict(tool_dict)
            if not effective_can_compose:
                for name in _code_and_session_tools:
                    out.pop(name, None)
                out.pop("FunctionManager_add_functions", None)
            if not effective_can_store:
                out.pop("FunctionManager_add_functions", None)
            if not effective_can_spawn_sub_agents:
                out.pop("run_sub_agent", None)
            return out

        base_tools = _filter_tools(self.get_tools("act"))

        # When execute_code is masked (can_compose=False), strip the
        # execute_code comparison from execute_function's docstring so the
        # LLM has no awareness that a code sandbox exists.
        if "execute_function" in base_tools and "execute_code" not in base_tools:
            base_tools["execute_function"].__doc__ = (
                "Execute a stored function by name and return its result.\n"
                "\n"
                "The function is looked up in the FunctionManager by exact name\n"
                "and executed directly. Only functions discovered via the\n"
                "FunctionManager discovery tools can be invoked.\n"
                "\n"
                "Workflow\n"
                "-------\n"
                "1. Discover functions via `FunctionManager_search_functions`,\n"
                "   `FunctionManager_filter_functions`, or\n"
                "   `FunctionManager_list_functions`.\n"
                "2. Pick the best match by name.\n"
                "3. Call `execute_function(function_name=..., call_kwargs=...)`.\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "function_name : str\n"
                "    Exact name of the stored function to execute (as returned by\n"
                "    the FunctionManager discovery tools).\n"
                "call_kwargs : dict, optional\n"
                "    Keyword arguments to pass to the function. Omit or pass None /\n"
                "    an empty dict for functions that take no arguments.\n"
                "\n"
                "Returns\n"
                "-------\n"
                "dict\n"
                "    A dict with:\n"
                "    - **result**: The function's return value (any JSON-serializable type).\n"
                "    - **error**: Traceback string if the function raised, else None.\n"
                "    - **stdout**: Captured standard output (string).\n"
                "    - **stderr**: Captured standard error (string).\n"
            )

        system_prompt = build_code_act_prompt(
            environments=sandbox_envs,
            tools=base_tools,
        )

        # Tool policy controls which tools are visible per turn, and whether a
        # tool call is required. Concerns:
        # 1) Static filters (can_compose, can_store) -- applied via _filter_tools.
        # 2) Function-first: on the first model turn, require a FunctionManager
        #    discovery call (search/filter/list) when those tools exist.
        _has_fm_tools = any(
            isinstance(k, str) and k.startswith("FunctionManager_")
            for k in base_tools.keys()
        )

        def _tool_policy(step: int, tools: Dict[str, Any]):
            filtered = _filter_tools(tools)

            # Function-first: on the first model turn, require a FunctionManager call
            # (search/filter/list) when those tools exist. This avoids the model skipping
            # the function library and going straight to execution.
            if step == 0 and _has_fm_tools:
                fm_only = {
                    k: v
                    for k, v in filtered.items()
                    if isinstance(k, str)
                    and k.startswith("FunctionManager_")
                    and k != "FunctionManager_add_functions"
                }
                if fm_only:
                    return "required", fm_only

            return "auto", filtered

        tool_policy = _tool_policy

        # Build an LLM client for this act() call
        client = new_llm_client(
            self._model,
            reasoning_effort=None,
            service_tier=None,
        )
        if system_prompt:
            client.set_system_message(system_prompt)

        # Add clarification tool when queues are supplied
        tools = dict(base_tools)
        if clarification_up_q is not None and clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                clarification_up_q,
                clarification_down_q,
                manager="CodeActActor",
                method="act",
                call_id=_call_id,
            )

        handle = start_async_tool_loop(
            client,
            description or initial_prompt,
            tools,
            loop_id=f"CodeActActor.act",
            propagate_chat_context=ChatContextPropagation.ALWAYS,
            parent_chat_context=_parent_chat_context,
            interrupt_llm_with_interjections=True,
            log_steps=True,
            max_steps=100,
            timeout=self._timeout,
            tool_policy=tool_policy,
            response_format=response_format,
            persist=persist,
            preprocess_msgs=self._preprocess_msgs,
            prompt_caching=self._prompt_caching,
        )

        # Wrap result() to run cleanup when the loop finishes
        _original_result = handle.result

        async def _result_with_cleanup() -> str:
            try:
                return await _original_result()
            finally:
                await _cleanup()

        handle.result = _result_with_cleanup  # type: ignore[assignment]

        # Update agent context with handle reference
        new_ctx.handle = handle

        # Wrap in StorageCheckHandle for post-completion function review.
        if effective_storage_check:
            handle = _StorageCheckHandle(inner=handle, actor=self)

        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
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

        if self._computer_primitives:
            self._computer_primitives.computer.stop()
