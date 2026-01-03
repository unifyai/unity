from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Type

import unify
from pydantic import BaseModel

from unity.common.llm_client import new_llm_client
from unity.file_manager.managers.base import BaseFileManager
from unity.file_manager.base import BaseGlobalFileManager
from unity.common.llm_helpers import (
    methods_to_tool_dict,
)
from unity.common.clarification_tools import add_clarification_tool_with_events
from unity.common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.common.tool_spec import manager_tool
from unity.file_manager.prompt_builders import (
    build_global_file_manager_ask_prompt,
    build_global_file_manager_organize_prompt,
)


class GlobalFileManager(BaseGlobalFileManager):
    """Single-surface facade over multiple filesystem-specific FileManagers.

    This manager does not expose low-level operations itself. Instead, it
    presents a unified read-only "ask" surface and an "organize" surface that
    delegates to the underlying managers' class‑named tools (e.g.,
    ``LocalFileManager_ask``, ``LocalFileManager_organize``).
    """

    def __init__(self, managers: List[BaseFileManager]):
        """
        Construct a GlobalFileManager over multiple FileManager instances.

        Parameters
        ----------
        managers : list[BaseFileManager]
            Concrete FileManager instances to expose. Their tools are surfaced
            with class‑named entries via ``include_class_name=True`` so the LLM
            can choose the right manager directly without aliases.
        """
        super().__init__()
        self._managers: List[BaseFileManager] = list(managers)

        # Ask tools: list filesystems + per-manager ask surfaces (class‑named)
        ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.list_filesystems,
            include_class_name=False,
        )
        for mgr in self._managers:
            ask_tools.update(
                methods_to_tool_dict(
                    mgr.ask,
                    mgr.ask_about_file,
                    include_class_name=True,
                ),
            )
        self.add_tools("ask", ask_tools)

        # Organize tools: discovery via ask + per-manager organize (class‑named)
        organize_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.ask,
            include_class_name=False,
        )
        for mgr in self._managers:
            organize_tools.update(
                methods_to_tool_dict(
                    mgr.organize,
                    include_class_name=True,
                ),
            )
        self.add_tools("organize", organize_tools)

    # -------------------- Default tool loop policies -------------------- #
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Default ask-side tool policy (no-op, retain current tools)."""
        return ("auto", current_tools)

    @staticmethod
    def _default_organize_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    # Helpers

    def list_filesystems(self) -> List[str]:
        """Return the list of manager class names in deterministic order."""
        names = [
            getattr(m.__class__, "__name__", "FileManager") for m in self._managers
        ]
        return sorted(set(names))

    # Public surfaces
    @functools.wraps(BaseGlobalFileManager.ask, updated=())
    @manager_tool
    @log_manager_call("GlobalFileManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        """
        Read-only questions over the aggregated file view.

        Parameters
        ----------
        text : str
            Natural-language query.
        _return_reasoning_steps : bool, default False
            When True, ``handle.result()`` returns ``(answer, messages)``.
        _parent_chat_context : list[dict] | None
            Optional upstream chat messages to seed the loop.
        _clarification_up_q / _clarification_down_q : Any | None
            Duplex queues for interactive clarification.
        rolling_summary_in_prompts : bool | None
            Whether to include the rolling activity summary in prompts.
        _call_id : str | None
            Correlation id for event logging.

        Returns
        -------
        SteerableToolHandle
            Handle that yields the final answer and supports pause/resume/stop.
        """
        client = new_llm_client()
        tools = dict(self.get_tools("ask"))
        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="GlobalFileManager",
                method="ask",
                call_id=_call_id,
            )
        include_activity = (
            True if rolling_summary_in_prompts is None else rolling_summary_in_prompts
        )
        system_msg = build_global_file_manager_ask_prompt(
            tools,
            num_filesystems=len(self._managers),
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        use_semantic_cache = "both" if SETTINGS.UNITY_SEMANTIC_CACHE else None
        tool_policy_fn = (
            None
            if use_semantic_cache in ("read", "both")
            else self._default_ask_tool_policy
        )
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.ask",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=tool_policy_fn,
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.ask",
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            response_format=response_format,
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    @functools.wraps(BaseGlobalFileManager.organize, updated=())
    @log_manager_call("GlobalFileManager", "organize", payload_key="text")
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        """
        Plan and execute safe rename/move operations across filesystems.

        Parameters
        ----------
        text : str
            Natural-language organization request.
        _return_reasoning_steps : bool, default False
            When True, ``handle.result()`` returns ``(answer, messages)``.
        _parent_chat_context : list[dict] | None
            Optional upstream chat messages to seed the loop.
        _clarification_up_q / _clarification_down_q : Any | None
            Duplex queues for interactive clarification.
        rolling_summary_in_prompts : bool | None
            Whether to include the rolling activity summary in prompts.
        _call_id : str | None
            Correlation id for event logging.

        Returns
        -------
        SteerableToolHandle
            Handle yielding a summary of operations performed.
        """
        client = new_llm_client()
        tools = dict(self.get_tools("organize"))
        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="GlobalFileManager",
                method="organize",
                call_id=_call_id,
            )
        include_activity = (
            True if rolling_summary_in_prompts is None else rolling_summary_in_prompts
        )
        system_msg = build_global_file_manager_organize_prompt(
            tools,
            num_filesystems=len(self._managers),
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        tool_policy = self._default_organize_tool_policy
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.organize",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=tool_policy,
            response_format=response_format,
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    @functools.wraps(BaseGlobalFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        """
        Reset the GlobalFileManager view and all underlying managers.

        Behaviour
        ---------
        - Attempts to delete the GlobalFileManager's own Unify context if one is
          present or can be derived. This manager does not persist aggregated rows
          by default, but this step ensures any future or temporary contexts are
          cleaned up.
        - Calls ``clear()`` on each underlying filesystem‑specific manager so any
          per‑filesystem contexts and local caches are reset.
        - All errors are swallowed to keep ``clear()`` idempotent and safe to call
          in test setup/teardown.
        """
        # Clear a derived global context if present
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        global_ctx = f"{read_ctx}/FilesGlobal" if read_ctx else "FilesGlobal"
        unify.delete_context(global_ctx)

        # Fan‑out clear to all underlying managers
        try:
            for mgr in self._managers or []:
                try:
                    mgr.clear()
                except Exception:
                    continue
        except Exception:
            pass
