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
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..function_manager.simulated import SimulatedFunctionManager
from ..function_manager.types.function import Function
from .prompt_builders import build_ask_prompt
from .base import BaseSkillManager
from ..common.simulated import (
    SimulatedLineage,
    SimulatedLog,
    maybe_tool_log_scheduled,
    SimulatedHandleMixin,
)


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
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

    def _num_functions(self) -> int:
        # Simulated manager returns a small constant for cosmetic display
        return 10

    @functools.wraps(BaseSkillManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:

        # Tool-style scheduled log (only when there is no parent lineage)
        sched = maybe_tool_log_scheduled(
            "SimulatedSkillManager.ask",
            "ask",
            {
                "text": text if isinstance(text, str) else repr(text),
                "has_clarification_channels": bool(
                    _clarification_up_q is not None
                    and _clarification_down_q is not None,
                ),
            },
        )
        # Prefer a stable lineage-aware label for subsequent steering logs
        label = (
            sched[0]
            if isinstance(sched, tuple) and len(sched) >= 1
            else SimulatedLineage.make_label("SimulatedSkillManager.ask")
        )

        tools = dict(self._tools)

        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                # Simulated, best-effort, human-facing log only
                try:
                    SimulatedLog.log_clarification_request(label, q)
                except Exception:
                    pass

            async def _on_answer(ans: str):
                # Simulated, best-effort, human-facing log only
                try:
                    SimulatedLog.log_clarification_answer(label, ans)
                except Exception:
                    pass

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
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

        inner_handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
        )

        if _return_reasoning_steps:
            original_result = inner_handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            inner_handle.result = wrapped_result  # type: ignore

        # Wrap the underlying tool-loop handle with a simulated logging proxy
        return _SimulatedSkillHandle(inner_handle, label)


class _SimulatedSkillHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    Thin proxy around the async tool loop handle that adds simulated logging
    consistent with other simulated managers (pause/resume/stop/interject/ask,
    clarifications and notifications).
    """

    def __init__(self, inner: SteerableToolHandle, log_label: str) -> None:
        self._inner = inner
        # Human-friendly, lineage-aware label for consistent logs
        self._log_label = (
            str(log_label)
            if log_label
            else SimulatedLineage.make_label(
                "SimulatedSkillManager.ask",
            )
        )

    # Steering methods ---------------------------------------------------------
    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Optional[str]:
        self._log_interject(message)
        try:
            return await self._inner.interject(
                message,
                parent_chat_context_cont=parent_chat_context_cont,
            )
        except TypeError:
            # Fallback for inner handles with a simpler signature
            return await self._inner.interject(message)  # type: ignore[arg-type]

    def stop(
        self,
        reason: Optional[str] = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> Optional[str]:
        self._log_stop(reason)
        try:
            return self._inner.stop(
                reason,
                parent_chat_context_cont=parent_chat_context_cont,
            )
        except TypeError:
            return self._inner.stop(reason)  # type: ignore[call-arg]

    async def pause(self) -> Optional[str]:
        self._log_pause()
        return await self._inner.pause()

    async def resume(self) -> Optional[str]:
        self._log_resume()
        return await self._inner.resume()

    def done(self) -> bool:
        return self._inner.done()

    async def result(self) -> str:
        return await self._inner.result()

    # Nested ask ---------------------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
        _return_reasoning_steps: bool = False,
        **kwargs,
    ) -> "SteerableToolHandle":
        # Build a concise child label and emit a simulated ask log
        try:
            child_label = SimulatedLineage.question_label(self._log_label)
        except Exception:
            child_label = f"Question({self._log_label})"
        try:
            SimulatedLog.log_request("ask", child_label, question)
        except Exception:
            pass

        # Delegate to the underlying handle
        try:
            inner = await self._inner.ask(
                question,
                parent_chat_context_cont=parent_chat_context_cont,
                images=images,
                _return_reasoning_steps=_return_reasoning_steps,
                **(kwargs or {}),
            )
        except TypeError:
            inner = await self._inner.ask(question)  # type: ignore[call-arg]
        # Wrap nested handle too so steering logs remain consistent
        return _SimulatedSkillHandle(inner, child_label)

    # Bottom-up event APIs -----------------------------------------------------
    async def next_clarification(self) -> dict:
        evt = await self._inner.next_clarification()
        try:
            msg = ""
            if isinstance(evt, dict):
                msg = str(evt.get("message", "")).strip()
            if msg:
                SimulatedLog.log_clarification_request(self._log_label, msg)
        except Exception:
            pass
        return evt

    async def next_notification(self) -> dict:
        evt = await self._inner.next_notification()
        try:
            message = ""
            if isinstance(evt, dict):
                message = str(evt.get("message", "")).strip()
            if message:
                SimulatedLog.log_notification(self._log_label, message)
        except Exception:
            pass
        return evt

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            SimulatedLog.log_clarification_answer(self._log_label, answer)
        except Exception:
            pass
        await self._inner.answer_clarification(call_id, answer)
