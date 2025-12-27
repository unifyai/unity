# conductor/conductor.py
from __future__ import annotations

from typing import Callable, Dict, Optional, TYPE_CHECKING

import asyncio
import json

import contextlib
import functools
import inspect
import weakref

from unity.session_details import SESSION_DETAILS


from ..conversation_manager.base import BaseConversationManagerHandle
from ..common.llm_client import new_llm_client
from ..common.llm_helpers import (
    methods_to_tool_dict,
    ToolSpec,
    short_id,
)
from ..common.llm_helpers import (
    canonicalize_handle_class_name as _canon_handle_name,
)
from ..common.async_tool_loop import start_async_tool_loop
from ..common.async_tool_loop import AsyncToolLoopHandle
from ..common.async_tool_loop import (
    SteerableToolHandle as _SteerableBase,
)
from .request_handle import ConductorRequestHandle
from .types import StateManager
from .prompt_builders import build_request_prompt
from .base import BaseConductor
from ..contact_manager.base import BaseContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..knowledge_manager.base import BaseKnowledgeManager
from ..guidance_manager.base import BaseGuidanceManager
from ..task_scheduler.base import BaseTaskScheduler
from ..task_scheduler.active_queue import ActiveQueue
from ..web_searcher.base import BaseWebSearcher
from ..file_manager.base import BaseGlobalFileManager
from ..actor.base import BaseActor
from ..actor.base import BaseActorHandle
from ..secret_manager.base import BaseSecretManager
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from .concurrency_guard import ActiveSessionRegistry
from ..common.sentinels import _DisabledSentinel
from ..settings import SETTINGS
from ..manager_registry import ManagerRegistry

if TYPE_CHECKING:  # type hints only
    from ..image_manager.types.image_refs import ImageRefs


class Conductor(BaseConductor):
    """
    Top-level façade that *can* own a maximum of *one* live plan at a time and exposes a
    unified tool surface (via ``request()``) which includes all read-only and write-capable
    tools across knowledge, tasks, contacts, transcripts, and other managers.
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Optional manager overrides – fall back to simulated defaults
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
        knowledge_manager: Optional[BaseKnowledgeManager] = None,
        guidance_manager: Optional[BaseGuidanceManager] = None,
        secret_manager: Optional[BaseSecretManager] = None,
        task_scheduler: Optional[BaseTaskScheduler] = None,
        web_searcher: Optional[BaseWebSearcher] = None,
        actor: Optional[BaseActor] = None,
        global_file_manager: Optional[BaseGlobalFileManager] = None,
        conversation_manager: Optional[BaseConversationManagerHandle] = None,
    ) -> None:
        """
        Args:
            description: A detailed description of the hypothetical scenario to simulate.
            log_events: Whether to log ManagerMethod events to the EventBus.
        """
        super().__init__()
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        # ── Managers – use provided instances or settings-driven defaults ──
        # When None is passed, ManagerRegistry resolves the IMPL from settings.
        # DISABLED sentinel explicitly disables optional managers.

        # Actor (foundational - cannot be disabled)
        self._actor = actor if actor is not None else ManagerRegistry.get_actor()

        # ContactManager (foundational - cannot be disabled)
        self._contact_manager = (
            contact_manager
            if contact_manager is not None
            else ManagerRegistry.get_contact_manager(
                description=description,
                simulation_guidance=simulation_guidance,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
            )
        )

        # TranscriptManager (foundational - cannot be disabled)
        self._transcript_manager = (
            transcript_manager
            if transcript_manager is not None
            else ManagerRegistry.get_transcript_manager(
                description=description,
                simulation_guidance=simulation_guidance,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                contact_manager=self._contact_manager,
            )
        )

        # TaskScheduler (foundational - cannot be disabled)
        self._task_scheduler = (
            task_scheduler
            if task_scheduler is not None
            else ManagerRegistry.get_task_scheduler(
                description=description,
                simulation_guidance=simulation_guidance,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                actor=self._actor,
            )
        )

        # ConversationManager (foundational - cannot be disabled)
        if conversation_manager is not None:
            self._cm_handle = conversation_manager
        elif SETTINGS.conversation.IMPL == "simulated":
            self._cm_handle = ManagerRegistry.get_conversation_manager_handle(
                description=description,
                simulation_guidance=simulation_guidance,
                assistant_id=SESSION_DETAILS.assistant.id,
                contact_id=SETTINGS.conversation.CONTACT_ID,
            )
        else:
            # Real ConversationManagerHandle requires a ConversationManager
            # which has complex dependencies - caller must provide it explicitly
            raise ValueError(
                "SETTINGS.conversation.IMPL='real' requires an explicit "
                "conversation_manager argument. Either pass a conversation_manager "
                "or set UNITY_CONVERSATION_IMPL='simulated'.",
            )

        # ── Optional managers (can be disabled via DISABLED sentinel or settings) ──

        # KnowledgeManager
        if isinstance(knowledge_manager, _DisabledSentinel):
            self._knowledge_manager = None
        elif knowledge_manager is not None:
            self._knowledge_manager = knowledge_manager
        elif not SETTINGS.knowledge.ENABLED:
            self._knowledge_manager = None
        else:
            self._knowledge_manager = ManagerRegistry.get_knowledge_manager(
                description=description,
                simulation_guidance=simulation_guidance,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
            )

        # GuidanceManager
        if isinstance(guidance_manager, _DisabledSentinel):
            self._guidance_manager = None
        elif guidance_manager is not None:
            self._guidance_manager = guidance_manager
        elif not SETTINGS.guidance.ENABLED:
            self._guidance_manager = None
        else:
            self._guidance_manager = ManagerRegistry.get_guidance_manager(
                description=description,
                simulation_guidance=simulation_guidance,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
            )

        # SecretManager
        if isinstance(secret_manager, _DisabledSentinel):
            self._secret_manager = None
        elif secret_manager is not None:
            self._secret_manager = secret_manager
        elif not SETTINGS.secret.ENABLED:
            self._secret_manager = None
        else:
            self._secret_manager = ManagerRegistry.get_secret_manager(
                description=description,
                simulation_guidance=simulation_guidance,
            )

        # WebSearcher
        if isinstance(web_searcher, _DisabledSentinel):
            self._web_searcher = None
        elif web_searcher is not None:
            self._web_searcher = web_searcher
        elif not SETTINGS.web.ENABLED:
            self._web_searcher = None
        else:
            self._web_searcher = ManagerRegistry.get_web_searcher(
                description=description,
                simulation_guidance=simulation_guidance,
            )

        # GlobalFileManager
        if isinstance(global_file_manager, _DisabledSentinel):
            self._file_manager: BaseGlobalFileManager | None = None
        elif global_file_manager is not None:
            self._file_manager = global_file_manager
        elif not SETTINGS.file.ENABLED:
            self._file_manager = None
        else:
            self._file_manager = ManagerRegistry.get_file_manager()

        #  Run-time state & tool-dict helpers
        self._active_task = None  # type: ignore
        self._session_guard = ActiveSessionRegistry()
        # Track live Conductor.request handles (weakly) for quick nested scans
        self._live_requests: "weakref.WeakSet[AsyncToolLoopHandle]" = weakref.WeakSet()

        # -------- build passive helpers (read-only managers) ---------- #
        # Build list of ask methods, filtering out None managers
        passive_methods = [
            # Foundational managers (always present)
            self._contact_manager.ask,
            self._transcript_manager.ask,
            self._task_scheduler.ask,
            self._cm_handle.ask,
            self._cm_handle.interject,
            self._cm_handle.get_full_transcript,
        ]
        # Optional managers (may be None)
        if self._knowledge_manager is not None:
            passive_methods.append(self._knowledge_manager.ask)
        if self._guidance_manager is not None:
            passive_methods.append(self._guidance_manager.ask)
        if self._secret_manager is not None:
            passive_methods.append(self._secret_manager.ask)
        if self._web_searcher is not None:
            passive_methods.append(self._web_searcher.ask)
        if self._file_manager is not None:
            passive_methods.append(self._file_manager.ask)

        passive = methods_to_tool_dict(*passive_methods, include_class_name=True)

        # -------- build active helpers (passive + writers) ------------ #
        # Build list of write methods, filtering out None managers
        active_methods = [
            # Foundational managers (always present)
            self._contact_manager.update,
            self._task_scheduler.update,
            ToolSpec(self._task_scheduler.execute, max_concurrent=1),
            self._actor.act,
            self.clear,
        ]
        # Optional managers (may be None)
        if self._knowledge_manager is not None:
            active_methods.append(self._knowledge_manager.update)
        if self._guidance_manager is not None:
            active_methods.append(self._guidance_manager.update)
        if self._secret_manager is not None:
            active_methods.append(self._secret_manager.update)
        if self._web_searcher is not None:
            active_methods.append(self._web_searcher.update)
        if self._file_manager is not None:
            active_methods.append(self._file_manager.organize)

        active = {
            **passive,  # read-only tools are also valid here
            **methods_to_tool_dict(*active_methods, include_class_name=True),
        }

        # Enforce mutual exclusion between Actor.act and TaskScheduler.execute by
        # tracking a single active handle and masking both tools while one is active.
        def _wrap_and_track(orig_callable, *, kind: str):
            @functools.wraps(orig_callable)
            async def _wrapper(*args, **kwargs):
                # 1) Reserve the global interactive slot
                kind_norm = "actor" if kind == "actor" else "execute"
                reserved = await self._session_guard.try_reserve(kind_norm)
                if not reserved:
                    return (
                        "An interactive session is already in progress. "
                        "Use interject/ask/stop on the current session instead of starting a new one."
                    )

                # 2) Call underlying tool and adopt the handle
                res = orig_callable(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    res = await res
                from unity.common.async_tool_loop import SteerableToolHandle  # type: ignore

                if isinstance(res, SteerableToolHandle):
                    # Track on Conductor and registry for masking and steering
                    self._active_task = res  # type: ignore[assignment]
                    await self._session_guard.adopt(res, kind_norm)

                    async def _clear_when_done(h):
                        try:
                            await h.result()
                        finally:
                            # Clear both registry and Conductor state if still owned by this handle
                            await self._session_guard.release_if(h)
                            if getattr(self, "_active_task", None) is h:
                                self._active_task = None  # type: ignore[assignment]

                    asyncio.create_task(_clear_when_done(res))
                else:
                    # No handle returned – release reservation immediately
                    await self._session_guard.release_if(None)
                return res

            # Preserve original signature/annotations so tool schema stays accurate
            try:
                _wrapper.__signature__ = inspect.signature(orig_callable)
                try:
                    ann = dict(getattr(orig_callable, "__annotations__", {}))
                    ann.pop("self", None)
                    _wrapper.__annotations__ = ann
                except Exception:
                    pass
            except Exception:
                pass
            return _wrapper

        # Locate canonical keys for the two entry-points (names include class prefixes)
        # Actor: resolve by bound-method identity rather than name so it works for any actor class
        actor_key = None
        try:
            for _k, _v in list(active.items()):
                fn = _v.fn if isinstance(_v, ToolSpec) else _v  # type: ignore[attr-defined]
                try:
                    # Identify bound method to the configured actor instance
                    if (
                        hasattr(fn, "__self__")
                        and getattr(fn, "__self__", None) is self._actor
                        and getattr(fn, "__name__", "") == "act"
                    ):
                        actor_key = _k
                        break
                except Exception:
                    continue
        except Exception:
            actor_key = None
        exec_key = next(
            (k for k in active if "taskscheduler_execute" in k.lower()),
            None,
        )

        if actor_key is not None:
            _orig = active[actor_key]
            if isinstance(_orig, ToolSpec):
                active[actor_key] = ToolSpec(
                    fn=_wrap_and_track(_orig.fn, kind="actor"),
                    max_concurrent=_orig.max_concurrent,
                    max_total_calls=_orig.max_total_calls,
                )
            else:
                active[actor_key] = _wrap_and_track(_orig, kind="actor")  # type: ignore[arg-type]
            # Provide a stable alias 'Actor_act' so the LLM consistently recognizes the entry-point
            if "Actor_act" not in active:
                active["Actor_act"] = active[actor_key]

        if exec_key is not None:
            _orig = active[exec_key]
            if isinstance(_orig, ToolSpec):
                active[exec_key] = ToolSpec(
                    fn=_wrap_and_track(_orig.fn, kind="execute"),
                    max_concurrent=_orig.max_concurrent,
                    max_total_calls=_orig.max_total_calls,
                )
            else:
                active[exec_key] = _wrap_and_track(_orig, kind="execute")  # type: ignore[arg-type]

        self.add_tools("request", active)

    # ------------------------------------------------------------------ #
    #  start_task – auto-start request loop to execute a task       #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseConductor.start_task, updated=())
    async def start_task(self, task_id: int, trigger_reason: str):
        """
        Return a steerable `Conductor.request` handle that immediately executes
        `TaskScheduler.execute` for the provided `task_id` without an initial LLM turn.

        Behaviour
        ---------
        - Seeds a minimal snapshot for the `Conductor.request` entrypoint with a
          single assistant tool_call targeting the exposed `TaskScheduler.execute` tool.
        - Deserialization triggers preflight backfill which schedules the tool call
          immediately (no LLM thinking step). The returned ActiveQueue handle is
          adopted with passthrough, so interject/pause/resume/stop/notifications
          work as usual via the returned handle.
        """

        # Resolve the exact tool name as exposed on the Conductor.request surface
        tools: Dict[str, Callable] = dict(self.get_tools("request"))
        exec_tool_name = next(
            (n for n in tools.keys() if "taskscheduler_execute" in n.lower()),
            None,
        )
        if exec_tool_name is None:
            raise ValueError(
                "TaskScheduler.execute tool is not available on request surface",
            )

        # Build a minimal v1 snapshot that instructs Conductor.request to call execute
        call_id = f"tc_{short_id(8)}"
        try:
            task_id_int = int(task_id)
        except Exception:
            # Be strict to avoid ambiguous execution routing
            raise ValueError("task_id must be an integer")

        snapshot = {
            "version": 1,
            "loop_id": f"{self.__class__.__name__}.request",
            "initial_user_message": (
                f"<This task has been *automatically* triggered due to {str(trigger_reason).strip()}>."
            ),
            "assistant": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": exec_tool_name,
                                "arguments": json.dumps({"task_id": task_id_int}),
                            },
                        },
                    ],
                },
            ],
            "tools": [],
        }

        # Deserialize into a live handle; preflight backfill will run the execute call immediately
        handle = AsyncToolLoopHandle.deserialize(snapshot)
        # Ensure handle is tracked for properties and cleaned up when finished
        self._register_live_request_handle(handle)
        return handle

    # ------------------------------------------------------------------ #
    #  clear – irreversible state wipe for a selected manager            #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseConductor.clear, updated=())
    def clear(self, target: StateManager) -> None:

        # Accept either an Enum member or its string value for robustness at runtime
        if isinstance(target, StateManager):
            key = target.value
        elif isinstance(target, str):
            try:
                key = StateManager(target).value
            except Exception:
                key = target
        else:
            raise TypeError("Invalid type for 'target'; expected StateManager or str.")

        # Map enum value to attribute name on this instance (prefixed underscore)
        attr_name = f"_{key}"
        manager = getattr(self, attr_name, None)
        if manager is None:
            # Build a robust label irrespective of whether `target` is an Enum or a string
            try:
                target_label = (
                    target.name if isinstance(target, StateManager) else str(target)
                )
            except Exception:
                target_label = str(target)
            raise ValueError(
                f"State manager '{target_label}' is not available on this Conductor instance.",
            )

        clear_fn = getattr(manager, "clear", None)
        if clear_fn is None or not callable(clear_fn):
            raise TypeError(
                f"State manager '{target.name}' does not expose a callable clear() method.",
            )

        clear_fn()

    # ------------------------------------------------------------------ #
    #  request  (write-capable)                                          #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseConductor.request, updated=())
    async def request(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        log_events: bool = False,
        rolling_summary_in_prompts: Optional[bool] = None,
    ):
        """
        Full-access entry-point – exposes every passive tool **plus** all
        write-capable helpers and `execute` (which unlocks plan steering).
        """
        should_log = self._log_events or log_events
        call_id = None

        if should_log:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "Conductor",
                "request",
                phase="incoming",
                request=text,
            )

        tools: Dict[str, Callable] = dict(self.get_tools("request"))

        if _clarification_up_q is not None or _clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if _clarification_up_q is None or _clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing.")
                await _clarification_up_q.put(question)
                return await _clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = new_llm_client()
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_request_prompt(tools, include_activity=include_activity),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.request.__name__}",
            parent_chat_context=_parent_chat_context,
            log_steps=_log_tool_steps,
            max_steps=None,
            timeout=None,
            # Hide Actor.act and TaskScheduler.execute while a session is active
            tool_policy=self._mask_act_execute_policy(),
            handle_cls=ConductorRequestHandle,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "Conductor",
                "request",
            )

        # Register this request handle for live scans and schedule cleanup when done
        self._register_live_request_handle(handle)

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result

        return handle

    # ----------------------------
    #  Internal: live handle registry
    # ----------------------------
    def _register_live_request_handle(self, handle) -> None:
        """Track the given request handle for fast property scans and auto-cleanup."""
        try:
            _h = getattr(handle, "_inner", handle)
            self._live_requests.add(_h)

            async def _cleanup_when_done(h):
                try:
                    await h.result()
                except Exception:
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        self._live_requests.discard(h)

            asyncio.create_task(_cleanup_when_done(_h))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Live handle discovery via nested_structure                         #
    # ------------------------------------------------------------------ #

    def _tree_has_any(self, node: dict, prefixes: tuple[str, ...]) -> bool:
        """Return True if any node in the nested_structure tree has a handle label starting with any prefix."""
        try:
            handle_label = str(node.get("handle", "")).strip()
        except Exception:
            handle_label = ""
        try:
            if any(handle_label.startswith(p) for p in prefixes):
                return True
        except Exception:
            pass
        try:
            for child in node.get("children", []) or []:
                if self._tree_has_any(child, prefixes):
                    return True
        except Exception:
            pass
        return False

    async def task_handle(self) -> Optional[ConductorRequestHandle]:
        """
        Return the live Conductor.request handle when a TaskScheduler.execute is active,
        detected purely via `nested_structure()` using a dynamically canonicalized handle prefix.
        """
        # Build the canonical handle label prefix dynamically (e.g., "ActiveQueue(")
        try:
            queue_prefix = f"{_canon_handle_name(ActiveQueue)}("
        except Exception:
            queue_prefix = "ActiveQueue("
        for h in list(getattr(self, "_live_requests", [])):
            try:
                if hasattr(h, "done") and h.done():
                    continue
                tree = await h.nested_structure()
            except Exception:
                continue
            if self._tree_has_any(tree, (queue_prefix,)):
                return h  # type: ignore[return-value]
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  High-level steering helpers                                       #
    # ------------------------------------------------------------------ #
    async def pause_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> dict:
        """
        Pause any in-flight interactive execution and announce the pause.
        """
        # Prefer a running TaskScheduler execution; else fall back to a direct Actor session
        target_handle = await self.task_handle()
        if target_handle is not None:
            target_tool = _canon_handle_name(ActiveQueue)
            interject_sig_source = ActiveQueue
        else:
            target_handle = await self.actor_handle()
            target_tool = _canon_handle_name(BaseActorHandle)
            interject_sig_source = BaseActorHandle
        if target_handle is None:
            return {"applied": [], "skipped": [], "status": {}}

        # Dynamic method names from abstract handle
        pause_name = getattr(_SteerableBase, "pause").__name__
        interject_name = getattr(_SteerableBase, "interject").__name__

        # Resolve images kwarg name dynamically from the concrete signature
        images_kw: str | None = None
        try:
            sig = inspect.signature(interject_sig_source.interject)  # type: ignore[attr-defined]
            for pname, param in sig.parameters.items():
                if pname in ("self", "message"):
                    continue
                if pname.lower() == "images":
                    images_kw = pname
                    break
        except Exception:
            images_kw = None

        child_message = f"<execution was paused due to {reason}>"
        interject_call: Dict[str, object] = {
            "method": interject_name,
            "args": child_message,
        }
        if images is not None and images_kw:
            interject_call["kwargs"] = {images_kw: images}
        spec: Dict[str, object] = {
            "children": [
                {
                    "tool": target_tool,
                    "steps": [
                        {"method": pause_name},
                        interject_call,
                    ],
                },
            ],
        }
        return await target_handle.nested_steer(spec)  # type: ignore[attr-defined]

    async def resume_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> dict:
        """
        Resume any in-flight interactive execution and announce the resume.
        """
        # Prefer a running TaskScheduler execution; else fall back to a direct Actor session
        target_handle = await self.task_handle()
        if target_handle is not None:
            target_tool = _canon_handle_name(ActiveQueue)
            interject_sig_source = ActiveQueue
        else:
            target_handle = await self.actor_handle()
            target_tool = _canon_handle_name(BaseActorHandle)
            interject_sig_source = BaseActorHandle
        if target_handle is None:
            return {"applied": [], "skipped": [], "status": {}}

        # Dynamic method names from abstract handle
        resume_name = getattr(_SteerableBase, "resume").__name__
        interject_name = getattr(_SteerableBase, "interject").__name__

        # Resolve images kwarg name dynamically from the concrete signature
        images_kw: str | None = None
        try:
            sig = inspect.signature(interject_sig_source.interject)  # type: ignore[attr-defined]
            for pname, param in sig.parameters.items():
                if pname in ("self", "message"):
                    continue
                if pname.lower() == "images":
                    images_kw = pname
                    break
        except Exception:
            images_kw = None

        child_message = f"<execution was resumed due to {reason}>"
        interject_call: Dict[str, object] = {
            "method": interject_name,
            "args": child_message,
        }
        if images is not None and images_kw:
            interject_call["kwargs"] = {images_kw: images}
        spec: Dict[str, object] = {
            "children": [
                {
                    "tool": target_tool,
                    "steps": [
                        interject_call,
                        {"method": resume_name},
                    ],
                },
            ],
        }
        return await target_handle.nested_steer(spec)  # type: ignore[attr-defined]

    async def actor_handle(self) -> Optional[ConductorRequestHandle]:
        """
        Return the live Conductor.request handle when an Actor session is active.
        If a TaskScheduler.execute session is active, return the same handle.
        """
        th = await self.task_handle()
        if th is not None:
            return th
        # Build the canonical handle label prefix dynamically (e.g., "ActorHandle(")
        try:
            actor_prefix = f"{_canon_handle_name(BaseActorHandle)}("
        except Exception:
            actor_prefix = "ActorHandle("
        for h in list(getattr(self, "_live_requests", [])):
            try:
                if hasattr(h, "done") and h.done():
                    continue
                tree = await h.nested_structure()
            except Exception:
                continue
            if self._tree_has_any(tree, (actor_prefix,)):
                return h  # type: ignore[return-value]
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Internal policy – mask Actor.act and TaskScheduler.execute while active
    # ------------------------------------------------------------------ #

    def _mask_act_execute_policy(self):
        def _policy(step_index: int, tools: Dict[str, Callable]):
            mode = "required" if step_index < 1 else "auto"
            filtered = dict(tools)

            try:
                active = getattr(self, "_active_task", None)
                if active is not None and not active.done():
                    # Remove both entry-points from the base toolkit; dynamic helpers remain available
                    actor_keys = [
                        k
                        for k in list(filtered)
                        if k.lower().startswith("actor_") and k.lower().endswith("act")
                    ]
                    exec_key = next(
                        (
                            k
                            for k in list(filtered)
                            if "taskscheduler_execute" in k.lower()
                        ),
                        None,
                    )
                    for ak in actor_keys:
                        filtered.pop(ak, None)
                    if exec_key:
                        filtered.pop(exec_key, None)
            except Exception:
                pass

            return mode, filtered

        return _policy
