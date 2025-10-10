from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import unify

from unity.file_manager.managers.base import BaseFileManager
from unity.file_manager.base import BaseGlobalFileManager
from unity.common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from unity.common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from ..constants import is_readonly_ask_guard_enabled
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.events.event_bus import EVENT_BUS, Event
from unity.file_manager.prompt_builders import (
    build_global_file_manager_ask_prompt,
    build_global_file_manager_organize_prompt,
)


class GlobalFileManager(BaseGlobalFileManager):
    """Single-surface file manager over multiple per-filesystem managers.

    Retrieval (search/filter/list columns) is Unify-only via the underlying
    managers' Unify tools. Organize actions are routed to a specific manager's
    organize() which will call its adapter for mutations behind the scenes.
    """

    def __init__(self, managers_by_alias: Dict[str, BaseFileManager]):
        super().__init__()
        self._managers: Dict[str, BaseFileManager] = dict(managers_by_alias)

        # Ask tools: global aggregated retrieval + alias-specific passthrough
        ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._list_filesystems,
            self._list_columns,
            self._filter_files,
            self._search_files,
            include_class_name=True,
        )
        for alias, mgr in self._managers.items():

            async def _ask_alias(text: str, __alias=alias):  # type: ignore
                handle = await self._managers[__alias].ask(text)
                return await handle.result()

            async def _ask_about_file_alias(filename: str, question: str, __alias=alias):  # type: ignore
                handle = await self._managers[__alias].ask_about_file(
                    filename,
                    question,
                )
                return await handle.result()

            ask_tools[f"ask__{alias}"] = _ask_alias
            ask_tools[f"ask_about_file__{alias}"] = _ask_about_file_alias
        self.add_tools("ask", ask_tools)

        # Organize tools: expose global inspection + alias-specific organizing
        organize_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._list_filesystems,
            self._list_columns,
            self._filter_files,
            self._search_files,
            self._rename_file,
            self._move_file,
            self._delete_file,
            include_class_name=True,
        )
        for alias, mgr in self._managers.items():

            async def _organize_alias(text: str, __alias=alias):  # type: ignore
                handle = await self._managers[__alias].organize(text)
                return await handle.result()

            organize_tools[f"organize__{alias}"] = _organize_alias
        self.add_tools("organize", organize_tools)

    # Helpers
    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

    def _list_filesystems(self) -> List[str]:
        return list(self._managers.keys())

    @staticmethod
    def _strip_filesystem_prefix(path: str, filesystem: str) -> str:
        """
        Strip filesystem namespace prefix from a path.

        Examples:
            _strip_filesystem_prefix("/local/file.txt", "local") -> "file.txt"
            _strip_filesystem_prefix("local/file.txt", "local") -> "file.txt"
            _strip_filesystem_prefix("file.txt", "local") -> "file.txt"
        """
        path = str(path).lstrip("/")
        prefix = f"{filesystem}/"
        if path.startswith(prefix):
            return path[len(prefix) :]
        return path

    def _rename_file(
        self,
        *,
        filesystem: str,
        target_id_or_path: str,
        new_name: str,
    ) -> Dict[str, Any]:
        if filesystem not in self._managers:
            raise ValueError(f"Filesystem '{filesystem}' not found.")
        # Strip filesystem namespace prefix before delegating
        clean_path = self._strip_filesystem_prefix(target_id_or_path, filesystem)
        return self._managers[filesystem]._rename_file(  # type: ignore[attr-defined]
            target_id_or_path=clean_path,
            new_name=new_name,
        )

    def _move_file(
        self,
        *,
        filesystem: str,
        target_id_or_path: str,
        new_parent_path: str,
    ) -> Dict[str, Any]:
        if filesystem not in self._managers:
            raise ValueError(f"Filesystem '{filesystem}' not found.")
        # Strip filesystem namespace prefix from both paths before delegating
        clean_path = self._strip_filesystem_prefix(target_id_or_path, filesystem)
        clean_parent = self._strip_filesystem_prefix(new_parent_path, filesystem)
        return self._managers[filesystem]._move_file(  # type: ignore[attr-defined]
            target_id_or_path=clean_path,
            new_parent_path=clean_parent,
        )

    def _delete_file(self, *, filesystem: str, file_id: int) -> Dict[str, Any]:
        if filesystem not in self._managers:
            raise ValueError(f"Filesystem '{filesystem}' not found.")
        return self._managers[filesystem]._delete_file(file_id=file_id)  # type: ignore[attr-defined]

    # ----------------- Unify-backed aggregated retrieval ----------------- #
    def _list_columns(self, *, include_types: bool = True):
        # Use schema from the first manager (shared model) and add source column
        cols: Dict[str, Any] = {}
        try:
            for _, mgr in self._managers.items():
                cols = mgr._list_columns(include_types=True)  # type: ignore[attr-defined]
                break
        except Exception:
            cols = {}
        if include_types:
            out = dict(cols or {})
            out["source_filesystem"] = "str"
            return out
        keys = list((cols or {}).keys())
        if "source_filesystem" not in keys:
            keys.append("source_filesystem")
        return keys

    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ):
        aggregated: List[Dict[str, Any]] = []
        for alias, mgr in self._managers.items():
            try:
                rows = mgr._filter_files(filter=filter, offset=0, limit=limit)  # type: ignore[attr-defined]
                for r in rows:
                    e = (
                        r
                        if isinstance(r, dict)
                        else getattr(r, "model_dump", lambda: r.__dict__)()
                    )
                    e = dict(e)
                    fname = e.get("filename")
                    if isinstance(fname, str):
                        e["filename"] = f"/{alias}/{fname}"
                    e["source_filesystem"] = alias
                    aggregated.append(e)
            except Exception:
                continue
        return aggregated[offset : offset + limit]

    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ):
        per_alias: Dict[str, List[Dict[str, Any]]] = {}
        for alias, mgr in self._managers.items():
            try:
                rows = mgr._search_files(references=references, k=k)  # type: ignore[attr-defined]
                normalized: List[Dict[str, Any]] = []
                for r in rows:
                    e = (
                        r
                        if isinstance(r, dict)
                        else getattr(r, "model_dump", lambda: r.__dict__)()
                    )
                    e = dict(e)
                    fname = e.get("filename")
                    if isinstance(fname, str):
                        e["filename"] = f"/{alias}/{fname}"
                    e["source_filesystem"] = alias
                    normalized.append(e)
                per_alias[alias] = normalized
            except Exception:
                continue
        merged: List[Dict[str, Any]] = []
        while len(merged) < k and any(per_alias.values()):
            for alias in list(per_alias.keys()):
                batch = per_alias.get(alias) or []
                if batch:
                    merged.append(batch.pop(0))
                    if len(merged) >= k:
                        break
        return merged

    # Public surfaces
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
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")
        tools = dict(self.get_tools("ask"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "GlobalFileManager",
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
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "GlobalFileManager",
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
        include_activity = (
            True if rolling_summary_in_prompts is None else rolling_summary_in_prompts
        )
        system_msg = build_global_file_manager_ask_prompt(
            tools,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.ask",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

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
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")
        tools = dict(self.get_tools("organize"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "GlobalFileManager",
                                "method": "organize",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_answer(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "GlobalFileManager",
                                "method": "organize",
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
        include_activity = (
            True if rolling_summary_in_prompts is None else rolling_summary_in_prompts
        )
        system_msg = build_global_file_manager_organize_prompt(
            tools,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.organize",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle
