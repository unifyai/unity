from __future__ import annotations

from typing import List, Dict, Optional, Callable, Any, Tuple
import asyncio
import json
import functools
import os
import re

import unify

from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_call
from ..common.semantic_search import (
    fetch_top_k_by_references,
    backfill_rows,
)
from .base import BaseGuidanceManager
from .types.guidance import Guidance


class GuidanceManager(BaseGuidanceManager):
    """
    Concrete Guidance manager backed by Unify contexts and fields.
    """

    def __init__(self, *, rolling_summary_in_prompts: bool = True) -> None:
        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised  # local

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass

        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a GuidanceManager."

        self._ctx = f"{read_ctx}/Guidance" if read_ctx else "Guidance"

        # Ensure context/fields exist deterministically
        self._store = TableStore(
            self._ctx,
            unique_keys={"guidance_id": "int"},
            auto_counting={"guidance_id": None},
            description=(
                "Table of distilled guidance entries from transcripts and images."
            ),
            fields=model_to_fields(Guidance),
        )
        self._store.ensure_context()

        # Built-in fields derived from Guidance model
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Guidance.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        # Public tools
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._list_columns,
                self._filter,
                self._search,
                include_class_name=False,
            ),
        }
        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.ask,
                self._add_guidance,
                self._update_guidance,
                self._delete_guidance,
                self._create_custom_column,
                self._delete_custom_column,
                include_class_name=False,
            ),
        }

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Track custom fields seen/created during lifetime
        self._known_custom_fields: set[str] = set()
        try:
            existing_cols = self._get_columns()
            for col in existing_cols:
                if col not in self._REQUIRED_COLUMNS and not str(col).startswith("_"):
                    self._known_custom_fields.add(col)
        except Exception:
            pass

    # ------------------------------- Public API -------------------------------
    @functools.wraps(BaseGuidanceManager.ask, updated=())
    @log_manager_call("GuidanceManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "ask",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools=tools,
                num_items=self._num_items(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseGuidanceManager.update, updated=())
    @log_manager_call("GuidanceManager", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "update",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "update",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(
                tools,
                num_items=self._num_items(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # ------------------------------- Helpers ---------------------------------
    def _num_items(self) -> int:
        ret = unify.get_logs_metric(
            metric="count",
            key="guidance_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

    def _get_columns(self) -> Dict[str, str]:
        return self._store.get_columns()

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(
                f"'{column_name}' is a required column and cannot be recreated.",
            )
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column_name):
            raise ValueError(
                "column_name must be snake_case: start with a letter, then letters/digits/underscores",
            )
        if (
            getattr(self, "_known_custom_fields", None)
            and column_name in self._known_custom_fields
        ):
            raise ValueError(f"Column '{column_name}' already exists.")
        info: Dict[str, Any] = {"type": str(column_type), "mutable": True}
        if column_description is not None:
            info["description"] = column_description
        resp = unify.create_fields(fields={column_name: info}, context=self._ctx)
        try:
            self._known_custom_fields.add(column_name)
        except Exception:
            pass
        return resp

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(f"Cannot delete required column '{column_name}'.")
        resp = unify.delete_fields(fields=[column_name], context=self._ctx)
        try:
            if column_name in getattr(self, "_known_custom_fields", set()):
                self._known_custom_fields.discard(column_name)
        except Exception:
            pass
        return resp

    # ------------------------------- Private tools ----------------------------
    def _add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Dict[str, int]] = None,
    ) -> ToolOutcome:
        if not title and not content and not images:
            raise ValueError(
                "At least one field (title/content/images) must be provided.",
            )
        g = Guidance(
            title=title or "",
            content=content or "",
            images=images or {},
        )
        log = unify.log(
            context=self._ctx,
            **g.to_post_json(),
            new=True,
            mutable=True,
        )
        return {
            "outcome": "guidance created successfully",
            "details": {"guidance_id": log.entries["guidance_id"]},
        }

    def _update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Dict[str, int]] = None,
    ) -> ToolOutcome:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if images is not None:
            # Validate via model field-validator by constructing minimal model
            _ = Guidance(title=title or "tmp", content=content or "tmp", images=images)
            updates["images"] = _.images
        if not updates:
            raise ValueError("At least one field must be provided for an update.")

        ids = unify.get_logs(
            context=self._ctx,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with guidance_id {guidance_id}. Data integrity issue.",
            )
        unify.update_logs(
            logs=[ids[0]],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )
        return {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}}

    def _delete_guidance(self, *, guidance_id: int) -> ToolOutcome:
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to delete.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with guidance_id {guidance_id}. Data integrity issue.",
            )
        unify.delete_logs(context=self._ctx, logs=ids[0])
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    def _search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        allowed_fields = list(self._BUILTIN_FIELDS)
        rows = fetch_top_k_by_references(
            self._ctx,
            references,
            k=k,
            allowed_fields=allowed_fields,
        )
        filled = backfill_rows(
            self._ctx,
            rows,
            k,
            unique_id_field="guidance_id",
            allowed_fields=allowed_fields,
        )
        return [Guidance(**r) for r in filled]

    def _filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        from_fields = list(self._BUILTIN_FIELDS)
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            from_fields=from_fields,
        )
        return [Guidance(**lg.entries) for lg in logs]

    # ------------------------------- Policies ---------------------------------
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if step_index < 1 and "search" in current_tools:
            return ("required", {"search": current_tools["search"]})
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if step_index < 1 and "ask" in current_tools:
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)
