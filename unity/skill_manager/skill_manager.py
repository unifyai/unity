from __future__ import annotations

import json
import os
import functools
from typing import Any, Dict, Optional, List, Callable

import unify

from ..common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
    inject_broader_context,
)
from ..common.async_tool_loop import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.manager_event_logging import log_manager_call

from ..function_manager.function_manager import FunctionManager
from ..function_manager.types.function import Function
from .prompt_builders import build_ask_prompt
from .base import BaseSkillManager


class SkillManager(BaseSkillManager):
    """
    Answers natural-language questions about the assistant's skills by exposing
    a read-only tool loop wired to the FunctionManager's listing/search methods.
    """

    def __init__(self) -> None:
        # Ensure the FunctionManager context exists to allow column/schema access
        self._function_manager = FunctionManager()

        # Expose read-only tools to the LLM
        def _list_functions(
            *,
            include_implementations: bool = False,
            parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
            return self._function_manager.list_functions(
                include_implementations=include_implementations,
                parent_chat_context=parent_chat_context,
            )

        def _search_functions(
            *,
            filter: Optional[str] = None,
            offset: int = 0,
            limit: int = 100,
            parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        ) -> List[Dict[str, Any]]:  # type: ignore[override]
            return self._function_manager.search_functions(
                filter=filter,
                offset=offset,
                limit=limit,
                parent_chat_context=parent_chat_context,
            )

        def _search_functions_by_similarity(
            *,
            query: str,
            n: int = 5,
            parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        ) -> List[Dict[str, Any]]:  # type: ignore[override]
            return self._function_manager.search_functions_by_similarity(
                query=query,
                n=n,
                parent_chat_context=parent_chat_context,
            )

        def _get_precondition(
            *,
            function_name: str,
            parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        ) -> Optional[Dict[str, Any]]:  # type: ignore[override]
            return self._function_manager.get_precondition(
                function_name=function_name,
                parent_chat_context=parent_chat_context,
            )

        self._tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                _list_functions,
                _search_functions,
                _search_functions_by_similarity,
                _get_precondition,
                include_class_name=False,
            ),
        }

        # Cache function columns for prompt readability
        try:
            # Access Function schema fields; does not hit backend
            self._function_columns = [
                {k: str(v.annotation)} for k, v in Function.model_fields.items()
            ]
        except Exception:
            self._function_columns = []

    # Small helper – LLM client factory
    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

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
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        import asyncio  # local to avoid widening import surface at module import time

        tools = dict(self._tools)

        if clarification_up_q is not None and clarification_down_q is not None:

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
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        client = self._new_llm_client("gpt-5@openai")
        client.set_system_message(
            build_ask_prompt(
                tools,
                include_activity=True,
                num_functions=self._num_functions(),
                function_columns=self._function_columns,
            ),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle
