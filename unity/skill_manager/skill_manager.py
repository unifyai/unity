from __future__ import annotations

import asyncio
import functools
from typing import Any, Dict, Optional, List, Callable, Type
from pydantic import BaseModel

import unify

from ..common.llm_client import new_llm_client
from ..common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..events.manager_event_logging import log_manager_call

from ..manager_registry import ManagerRegistry
from ..function_manager.types.function import Function
from .prompt_builders import build_ask_prompt
from .base import BaseSkillManager


class SkillManager(BaseSkillManager):
    """
    Answers natural-language questions about the assistant's skills by exposing
    a read-only tool loop wired to the FunctionManager's listing/search methods.
    """

    def __init__(self) -> None:
        super().__init__()
        # Get FunctionManager via registry to ensure context exists
        self._function_manager = ManagerRegistry.get_function_manager()

        # Expose read-only FunctionManager methods directly (no wrappers)
        ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._function_manager.list_functions,
                self._function_manager.search_functions,
                self._function_manager.search_functions_by_similarity,
                self._function_manager.get_precondition,
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)

        # Cache function columns for prompt readability
        try:
            # Access Function schema fields; does not hit backend
            self._function_columns = [
                {k: str(v.annotation)} for k, v in Function.model_fields.items()
            ]
        except Exception:
            self._function_columns = []

    # Small helper – LLM client factory

    def _num_functions(self) -> int:
        """Return the total number of stored functions (skills)."""
        try:
            # Use backend metric directly over FunctionManager context
            ctxs = unify.get_active_context()
            read_ctx = ctxs["read"]
            fm_ctx = f"{read_ctx}/Functions" if read_ctx else "Functions"
            ret = unify.get_logs_metric(
                metric="count",
                key="function_id",
                context=fm_ctx,
            )
            return int(ret) if ret is not None else 0
        except Exception:
            return 0

    @functools.wraps(BaseSkillManager.ask, updated=())
    @log_manager_call("SkillManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        import asyncio  # local to avoid widening import surface at module import time

        tools = dict(self.get_tools("ask"))

        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    from ..events.event_bus import EVENT_BUS, Event

                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "SkillManager",
                                "method": "ask",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_answer(ans: str):
                try:
                    from ..events.event_bus import EVENT_BUS, Event

                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "SkillManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        client = new_llm_client()
        client.set_system_message(
            build_ask_prompt(
                tools,
                include_activity=True,
                num_functions=self._num_functions(),
                function_columns=self._function_columns,
            ),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle
