from __future__ import annotations

import asyncio
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
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.manager_event_logging import log_manager_call

from ..function_manager.simulated import SimulatedFunctionManager
from ..function_manager.types.function import Function
from .prompt_builders import build_ask_prompt
from .base import BaseSkillManager


class SimulatedSkillManager(BaseSkillManager):
    """
    Drop-in simulated SkillManager that still spins up a real tool loop wired to
    the read-only methods of SimulatedFunctionManager. Useful for offline demos.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
    ) -> None:
        # Store settings for parity with other simulated managers
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance

        self._function_manager = SimulatedFunctionManager(
            description=description,
            simulation_guidance=simulation_guidance,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
        )

        # Expose the simulated FunctionManager's read-only tools directly (no wrappers)
        self._tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._function_manager.list_functions,
                self._function_manager.search_functions,
                self._function_manager.search_functions_by_similarity,
                self._function_manager.get_precondition,
                include_class_name=False,
            ),
        }

        try:
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
        # Simulated manager returns a small constant for cosmetic display
        return 10

    @functools.wraps(BaseSkillManager.ask, updated=())
    @log_manager_call("SimulatedSkillManager", "ask", payload_key="question")
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
                                "manager": "SimulatedSkillManager",
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
                                "manager": "SimulatedSkillManager",
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
                include_activity=self._rolling_summary_in_prompts,
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
