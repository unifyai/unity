from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional

import unify
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
from ..common.tool_outcome import ToolOutcome
from ..common.embed_utils import ensure_vector_column
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types import Secret
from .base import BaseSecretManager
from .prompt_builders import build_ask_prompt, build_update_prompt


def _mask_value_for_llm(value: str) -> str:
    """Return a short, safe mask when the LLM would otherwise see a raw secret."""
    if not value:
        return "<redacted>"
    # Show only type/length for debugging, but not content
    return f"<secret:{len(value)} chars>"


def _placeholder(name: str) -> str:
    return f"${{{name}}}"


class SecretManager(BaseSecretManager):
    """
    Manages a fixed-schema table of secrets. Ensures secrets are never exposed
    to LLMs directly. Public methods mirror other managers' design.
    """

    def __init__(self) -> None:
        # Resolve context and construct a single-table store
        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must match for SecretManager."
        self._ctx = f"{read_ctx}/Secrets"

        # Fixed schema derived from Secret model
        self._store = TableStore(
            self._ctx,
            unique_keys={"name": "str"},
            description="Key-value secrets with descriptions and embeddings.",
            fields=model_to_fields(Secret),
        )
        self._store.ensure_context()

        # Public tools
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._create_secret,
                self._update_secret,
                self._delete_secret,
                self._list_secret_keys,
                self._list_columns,
                self._filter_secrets,
                self._search_secrets,
                include_class_name=False,
            ),
        }
        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.ask,
                self._create_secret,
                self._update_secret,
                self._delete_secret,
                self._list_secret_keys,
                include_class_name=False,
            ),
        }

        # Ensure vector for description
        try:
            ensure_vector_column(
                self._ctx,
                embed_column="description_emb",
                source_column="description",
                derived_expr=None,
            )
        except Exception:
            pass

        # Unify storage is the single source of truth for secrets

    # --------------------- Public API --------------------- #
    @functools.wraps(BaseSecretManager.ask, updated=())
    @log_manager_call("SecretManager", "ask", payload_key="question")
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
        client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

        # Build tools for read-only inspection
        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "SecretManager",
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
                            "manager": "SecretManager",
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

        # System message via prompt builder
        client.set_system_message(
            build_ask_prompt(tools=tools),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.ask",
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

    @functools.wraps(BaseSecretManager.update, updated=())
    @log_manager_call("SecretManager", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "SecretManager",
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
                            "manager": "SecretManager",
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

        client.set_system_message(
            build_update_prompt(tools=tools),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.update",
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

    # --------------------- Tools (read-only) --------------------- #
    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        cols = self._store.get_columns()
        return cols if include_types else list(cols)

    def _search_secrets(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Secret]:
        # Simple implementation: prefer description vector; fallback to recent
        try:
            from ..common.semantic_search import (
                fetch_top_k_by_references,
                backfill_rows,
            )

            rows = fetch_top_k_by_references(
                self._ctx,
                references,
                k=k,
                allowed_fields=["name", "description"],
                row_filter=None,
            )
            filled = backfill_rows(
                self._ctx,
                rows,
                k,
                row_filter=None,
                unique_id_field="name",
                allowed_fields=["name", "description"],
            )
            return [
                Secret(
                    name=r.get("name"),
                    value="",
                    description=r.get("description", ""),
                )
                for r in filled
            ]
        except Exception:
            logs = unify.get_logs(context=self._ctx, limit=k)
            return [
                Secret(
                    name=lg.entries.get("name"),
                    value="",
                    description=lg.entries.get("description", ""),
                )
                for lg in logs
            ]

    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Secret]:
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            from_fields=["name", "description"],
        )
        # Never expose values in read tools
        return [
            Secret(
                name=lg.entries.get("name"),
                value="",
                description=lg.entries.get("description", ""),
            )
            for lg in logs
        ]

    def _list_secret_keys(self) -> List[str]:
        """Return all available secret names (keys) stored in Unify (sorted, unique)."""
        try:
            rows = unify.get_logs(context=self._ctx, from_fields=["name"], limit=100000)
        except Exception:
            rows = []
        names: set[str] = set()
        for lg in rows:
            nm = (lg.entries or {}).get("name")
            if isinstance(nm, str) and nm:
                names.add(nm)
        return sorted(names)

    # --------------------- Tools (mutations) --------------------- #
    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
    ) -> ToolOutcome:
        assert name and value, "Both name and value are required."
        # Enforce uniqueness of name
        existing = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing, f"Secret with name '{name}' already exists."

        # Write secret (store raw value in backend, but never surface to LLM)
        entries = {
            "name": name,
            "value": value,
            "description": description or "",
        }
        log = unify.log(context=self._ctx, **entries, new=True, mutable=True)

        return {"outcome": "secret created", "details": {"name": name}}

    def _update_secret(
        self,
        *,
        name: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ToolOutcome:
        # Find target log id
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        log_id = ids[0]

        updates: Dict[str, Any] = {}
        if description is not None:
            updates["description"] = description
        if value is not None:
            updates["value"] = value

        if not updates:
            raise ValueError("No updates provided.")

        unify.update_logs(
            logs=[log_id],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )

        return {"outcome": "secret updated", "details": {"name": name}}

    def _delete_secret(self, *, name: str) -> ToolOutcome:
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        unify.delete_logs(context=self._ctx, logs=ids[0])
        return {"outcome": "secret deleted", "details": {"name": name}}
