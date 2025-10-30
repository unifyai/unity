from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import unify

from unity.common.llm_client import new_llm_client
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
        """
        Create a GlobalFileManager over multiple filesystem-specific managers.

        Parameters
        ----------
        managers_by_alias : dict[str, BaseFileManager]
            Mapping of filesystem alias (e.g., "local", "drive", "interact") to the
            corresponding concrete FileManager instance. The aliases are used to
            namespace filenames in aggregated views (e.g., "/local/notes.txt").

        Notes
        -----
        - Registers separate tool surfaces for ``ask`` and ``organize``. For each
          underlying manager, lightweight alias-prefixed passthrough helpers are
          added (e.g., ``ask__local``, ``ask_about_file__drive``) so the LLM can
          route questions to a specific filesystem when needed.
        - Aggregated tools (``_list_columns``, ``_filter_files``, ``_search_files``)
          operate across all managers and add a synthetic ``source_filesystem``
          field for provenance.
        """
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

    def _list_filesystems(self) -> List[str]:
        """Return the list of configured filesystem aliases in deterministic order."""
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
        """
        Rename a file within a specific filesystem.

        Parameters
        ----------
        filesystem : str
            Alias of the target filesystem (must exist in this manager).
        target_id_or_path : str
            Adapter-specific identifier or path for the file to rename. May be
            namespaced ("/alias/path") or plain; any leading alias is stripped
            before delegation.
        new_name : str
            New filename without directory components.

        Returns
        -------
        dict
            Adapter-specific result from the underlying manager.
        """
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
        """
        Move a file to a new parent path within a specific filesystem.

        Parameters
        ----------
        filesystem : str
            Alias of the target filesystem.
        target_id_or_path : str
            Adapter-specific identifier or path of the file to move.
        new_parent_path : str
            Destination directory path (within the same filesystem).

        Returns
        -------
        dict
            Adapter-specific result from the underlying manager.
        """
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
        """
        Permanently delete a file from a specific filesystem.

        Parameters
        ----------
        filesystem : str
            Alias of the target filesystem.
        file_id : int
            Numeric id of the file to delete (adapter-specific).

        Returns
        -------
        dict
            Adapter-specific confirmation payload.
        """
        if filesystem not in self._managers:
            raise ValueError(f"Filesystem '{filesystem}' not found.")
        return self._managers[filesystem]._delete_file(file_id=file_id)  # type: ignore[attr-defined]

    # ----------------- Unify-backed aggregated retrieval ----------------- #
    def _list_columns(self, *, include_types: bool = True):
        """
        Return the global schema for files, augmented with ``source_filesystem``.

        Parameters
        ----------
        include_types : bool, default True
            When True, returns ``{column: type}``; otherwise returns a list of
            column names.

        Returns
        -------
        dict[str, Any] | list[str]
            Consolidated schema from the first manager (they share a model) with
            an added ``source_filesystem`` column.
        """
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
        """
        Filter files across all filesystems using a Python expression.

        Parameters
        ----------
        filter : str | None, default None
            Row-level predicate evaluated per underlying manager result row. The
            predicate may reference any column in the files table; use single
            quotes for string literals.
        offset : int, default 0
            Zero-based index of the first row to include (after aggregation).
        limit : int, default 100
            Maximum number of rows to return.

        Returns
        -------
        list[dict]
            Normalized rows with namespaced ``filename`` and ``source_filesystem``.
        """
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
                    fname = e.get("file_path")
                    if isinstance(fname, str):
                        e["file_path"] = f"/{alias}/{fname}"
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
        """
        Semantic search across all filesystems and interleave top results.

        Parameters
        ----------
        references : dict[str, str] | None, default None
            Mapping of ``source_expr → reference_text`` terms. Each source
            expression is a column or derived expression in the underlying
            manager. When omitted/empty, falls back to most recent files.
        k : int, default 10
            Maximum number of results to return after interleaving per-alias
            rankings.

        Returns
        -------
        list[dict]
            Up to ``k`` normalized rows (with ``filename`` namespaced and
            ``source_filesystem`` added).
        """
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
                    fname = e.get("file_path")
                    if isinstance(fname, str):
                        e["file_path"] = f"/{alias}/{fname}"
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
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

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
        # Best‑effort: clear a derived global context if present
        try:
            ctxs = unify.get_active_context()
            read_ctx = ctxs.get("read")
            global_ctx = f"{read_ctx}/FilesGlobal" if read_ctx else "FilesGlobal"
            try:
                unify.delete_context(global_ctx)
            except Exception:
                pass
        except Exception:
            pass

        # Fan‑out clear to all underlying managers
        try:
            for _, mgr in (self._managers or {}).items():
                try:
                    mgr.clear()
                except Exception:
                    continue
        except Exception:
            pass
