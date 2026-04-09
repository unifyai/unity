from __future__ import annotations

import asyncio
import uuid
import unify
import functools
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from unity.data_manager.data_manager import DataManager

import json
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.common.tool_outcome import ToolOutcome
from unity.common.token_utils import count_tokens_per_utf_byte
from unity.common import token_utils as _tok
from unity.common.grouping_helpers import build_grouped_dump_payload
from .types import ColumnType
from ..common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from .base import BaseKnowledgeManager
from ..events.manager_event_logging import log_manager_call
from .prompt_builders import (
    build_update_prompt,
    build_ask_prompt,
    build_refactor_prompt,
)
from ..common.tool_spec import read_only, manager_tool, ToolSpec
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..common.context_registry import ContextRegistry, TableContext
from ..common.llm_client import new_llm_client
from ..common.metrics_utils import reduce_logs
from ..events.event_bus import EVENT_BUS, Event

# Module delegations (split helpers for parity with ContactManager)
from .storage import (
    get_columns as _storage_get_columns,
    tables_overview as _storage_tables_overview,
    ctx_for_table as _storage_ctx_for_table,
    create_table as _storage_create_table,
    rename_table as _storage_rename_table,
    delete_tables as _storage_delete_tables,
)
from .search import (
    filter as _srch_filter,
    search as _srch_search,
    filter_join as _srch_filter_join,
    search_join as _srch_search_join,
    filter_multi_join as _srch_filter_multi_join,
    search_multi_join as _srch_search_multi_join,
)
from .ops import (
    add_rows as _op_add_rows,
    update_rows as _op_update_rows,
    delete_rows as _op_delete_rows,
    transform_column as _op_transform_column,
    copy_column as _op_copy_column,
    move_column as _op_move_column,
    create_empty_column as _op_create_empty_column,
    create_derived_column as _op_create_derived_column,
    delete_column as _op_delete_column,
    vectorize_column as _op_vectorize_column,
    rename_column as _op_rename_column,
)


class KnowledgeManager(BaseKnowledgeManager):
    class Config:
        required_contexts = [
            TableContext(
                name="Knowledge",
                description="Knowledge base for the assistant.",
            ),
        ]

    def __init__(
        self,
        *,
        rolling_summary_in_prompts: bool = True,
        include_contacts: bool = True,
        grouped: bool = False,
        # Table-dump heuristics
        full_table_dump: bool = False,
        per_table_dumps: bool = False,
        dump_safety_factor: float = 0.9,
        dump_timeout_s: float = 3.0,
        dump_scan_page_size: int = 1000,
        max_input_tokens_override: Optional[int] = None,
    ) -> None:
        """
        Initialise the KnowledgeManager.

        This manager reads/writes directly to Unify contexts for knowledge
        tables. When ``include_contacts`` is ``True`` it also exposes the
        root‑level ``Contacts`` table for cross‑table queries/joins.

        Parameters
        ----------
        rolling_summary_in_prompts : bool, default ``True``
            When enabled, inject a short rolling activity summary (sourced
            from ``MemoryManager``) into system prompts for LLM calls.
        include_contacts : bool, default ``True``
            When ``True``, link the root‑level ``Contacts`` table so that
            tools such as joins and filters can reference it via the special
            table name ``"Contacts"``.
        """
        super().__init__()
        self.include_in_multi_assistant_table = False
        # Allow ingestion/deprecation only within update/refactor flows
        refactor_tools = methods_to_tool_dict(
            # Ask
            ToolSpec(fn=self.ask, display_label="Querying notes"),
            # Tables
            ToolSpec(fn=self._create_table, display_label="Creating a new notes table"),
            ToolSpec(fn=self._rename_table, display_label="Renaming a notes table"),
            ToolSpec(fn=self._delete_tables, display_label="Deleting notes tables"),
            # Columns
            ToolSpec(fn=self._rename_column, display_label="Renaming a column"),
            ToolSpec(fn=self._copy_column, display_label="Copying a column"),
            ToolSpec(fn=self._move_column, display_label="Moving a column"),
            ToolSpec(fn=self._delete_column, display_label="Deleting a column"),
            ToolSpec(fn=self._create_empty_column, display_label="Adding a new column"),
            ToolSpec(
                fn=self._create_derived_column,
                display_label="Creating a derived column",
            ),
            ToolSpec(
                fn=self._transform_column,
                display_label="Transforming column values",
            ),
            ToolSpec(
                fn=self._vectorize_column,
                display_label="Indexing a column for search",
            ),
            # Rows
            ToolSpec(fn=self._delete_rows, display_label="Deleting rows"),
            ToolSpec(fn=self._update_rows, display_label="Updating rows"),
            include_class_name=False,
        )
        self.add_tools("refactor", refactor_tools)

        multi_table_ask_tools = methods_to_tool_dict(
            ToolSpec(fn=self._filter_join, display_label="Cross-referencing notes"),
            ToolSpec(fn=self._search_join, display_label="Searching across notes"),
            ToolSpec(
                fn=self._filter_multi_join,
                display_label="Cross-referencing multiple note tables",
            ),
            ToolSpec(
                fn=self._search_multi_join,
                display_label="Searching across multiple note tables",
            ),
            include_class_name=False,
        )
        self.add_tools("ask.multi_table", multi_table_ask_tools)

        # We decide in `ask` method whether to include the multi-table tools or not
        ask_tools = {
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._tables_overview,
                    display_label="Reviewing notes structure",
                ),
                ToolSpec(fn=self._filter, display_label="Filtering notes"),
                ToolSpec(fn=self._search, display_label="Searching notes"),
                ToolSpec(fn=self._reduce, display_label="Summarising notes"),
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)

        update_tools = {
            **refactor_tools,
            **methods_to_tool_dict(
                ToolSpec(fn=self._add_rows, display_label="Adding new entries"),
                include_class_name=False,
            ),
        }
        self.add_tools("update", update_tools)

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # When enabled, results returned by table tools are grouped client-side
        # into a nested structure to reduce duplication and token usage.
        self._group_results: bool = grouped

        # Heuristic dump settings
        self._full_table_dump: bool = full_table_dump
        self._per_table_dumps: bool = per_table_dumps
        self._dump_safety_factor: float = max(0.1, min(0.99, dump_safety_factor))
        self._dump_timeout_s: float = max(0.5, dump_timeout_s)
        self._dump_scan_page_size: int = max(50, dump_scan_page_size)
        self._max_input_tokens: int = (
            int(max_input_tokens_override)
            if max_input_tokens_override is not None
            else _tok.read_model_max_input_tokens()
        )

        # ------------------------------------------------------------------
        # Optional Contacts-table linkage
        # ------------------------------------------------------------------
        self._include_contacts: bool = include_contacts
        self._ctx = ContextRegistry.get_context(self, "Knowledge")

        # Only compute the Contacts context if the caller requested integration.
        self._contacts_ctx: Optional[str]
        if include_contacts:
            from unity.contact_manager.contact_manager import ContactManager

            self._contacts_ctx = ContextRegistry.get_context(ContactManager, "Contacts")
        else:
            self._contacts_ctx = None

        # Lazily-initialized DataManager for delegation
        self.__data_manager: Optional["DataManager"] = None

    @property
    def _data_manager(self) -> "DataManager":
        """
        Lazily-initialized DataManager instance for delegation.

        All low-level data operations (filter, search, insert, update, delete,
        joins, etc.) are delegated to the DataManager to ensure consistency
        and avoid direct ``unify`` calls in KnowledgeManager utilities.
        """
        if self.__data_manager is None:
            from unity.data_manager.data_manager import DataManager

            self.__data_manager = DataManager()
        return self.__data_manager

    async def _maybe_build_show_all_seed(
        self,
        message: Union[str, dict, List[Union[str, dict]]],
        tables_overview: Dict[str, Dict[str, Any]] | None = None,
    ) -> Optional[List[dict]]:
        """
        Decide whether to seed a synthetic first tool call `show_all` that dumps
        some or all tables, and if so return the seeded transcript. Otherwise None.
        Uses `unify.get_groups`-based unique value enumeration for token estimation.
        """
        try:
            if not (self._full_table_dump or self._per_table_dumps):
                return None

            if isinstance(message, list):
                user_text = (
                    next(
                        (
                            m.get("content")
                            for m in message
                            if isinstance(m, dict) and m.get("role") == "user"
                        ),
                        None,
                    )
                    or ""
                )
            elif isinstance(message, dict):
                user_text = str(message.get("content") or "")
            else:
                user_text = message

            overview = tables_overview or self._tables_overview()
            all_tables = list((overview or {}).keys())
            if not all_tables:
                return None

            table_to_ctx = {t: self._ctx_for_table(t) for t in all_tables}
            # Be robust to tables without a "columns" key in their overview entry
            table_to_columns: Dict[str, List[str]] = {
                t: list((lt.get("columns") or {}).keys()) for t, lt in overview.items()
            }

            est = await _tok.estimate_tables_tokens_parallel(
                table_to_ctx=table_to_ctx,
                table_to_columns=table_to_columns,
                max_input_tokens=self._max_input_tokens,
                safety_factor=self._dump_safety_factor,
                max_concurrency=4,
            )
            total_est = sum(est.values())
            budget = int(self._max_input_tokens * self._dump_safety_factor)

            selected: List[str] = []
            if self._full_table_dump and total_est <= budget:
                selected = list(all_tables)
            elif self._per_table_dumps and all_tables:
                per_tbl_threshold = int(
                    self._max_input_tokens
                    * self._dump_safety_factor
                    / (2 * max(1, len(all_tables))),
                )
                selected = [t for t, v in est.items() if v <= per_tbl_threshold]

            if not selected:
                return None

            payload, per_tbl_payload_tokens = build_grouped_dump_payload(
                table_to_ctx,
                selected,
                limit=self._dump_scan_page_size,
            )
            total_payload_tokens = count_tokens_per_utf_byte(payload)
            sum_per_table_tokens = sum(per_tbl_payload_tokens.values())
            if sum_per_table_tokens > budget or total_payload_tokens > budget:
                ranked = sorted(
                    selected,
                    key=lambda t: per_tbl_payload_tokens.get(t, 0),
                    reverse=True,
                )
                keep = list(selected)
                current_sum = sum_per_table_tokens
                # Drop largest tables until within budget using per-table token sizes
                while keep and current_sum > budget:
                    drop = ranked.pop(0)
                    if drop in keep:
                        keep.remove(drop)
                        current_sum -= per_tbl_payload_tokens.get(drop, 0)
                selected = keep
                # Rebuild payload for the reduced selection
                payload, per_tbl_payload_tokens = build_grouped_dump_payload(
                    {t: table_to_ctx[t] for t in selected},
                    selected,
                    limit=self._dump_scan_page_size,
                )

            if not selected:
                return None

            call_id = f"show_all_{uuid.uuid4().hex[:8]}"
            seeded = [
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "show_all", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "show_all",
                    "content": payload,
                },
            ]
            return seeded
        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error in _maybe_build_show_all_seed: {e}")
            return None

    # Helpers #
    # --------#

    def _ctx_for_table(self, table: str) -> str:
        """
        Return the fully‑qualified Unify context name for ``table``.

        When this instance was created with ``include_contacts=False`` any
        attempt to reference the ``Contacts`` table is rejected to avoid
        hidden cross‑coupling.

        Parameters
        ----------
        table : str
            Logical table name as used by this manager (e.g. ``"Products"``).
            The special name ``"Contacts"`` maps to the root‑level contacts
            context when contacts linkage is enabled.

        Returns
        -------
        str
            The fully‑qualified Unify context.

        Raises
        ------
        ValueError
            If ``table == "Contacts"`` but this instance was initialised with
            ``include_contacts=False``.
        """

        return _storage_ctx_for_table(self, table)

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_ASK_TOOL_IS_SEARCH
            and step_index < 1
            and "search" in current_tools
        ):
            return ("required", {"search": current_tools["search"]})
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
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

    @staticmethod
    def _default_refactor_tool_policy(
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

    # Public #
    # -------#

    # English-Text Command

    @functools.wraps(BaseKnowledgeManager.refactor, updated=())
    @log_manager_call(
        "KnowledgeManager",
        "refactor",
        payload_key="request",
        display_label="Reorganizing notes",
    )
    async def refactor(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> "SteerableToolHandle":
        """
        Structure-changing edits to tables, columns and rows.

        Parameters
        ----------
        text : str
            High-level description of the desired refactor (e.g., create/rename
            tables, transform columns, delete rows).
        _return_reasoning_steps : bool, default False
            When True, ``handle.result()`` returns ``(answer, messages)`` for
            debugging.
        _parent_chat_context : list[dict] | None
            Optional upstream chat messages to seed the loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Duplex queues for interactive clarification.
        rolling_summary_in_prompts : bool | None
            Whether to include the rolling activity summary in prompts.
        _call_id : str | None
            Correlation id for event logging.

        Returns
        -------
        SteerableToolHandle
            Handle that yields a natural-language summary of the refactor.
        """

        client = new_llm_client()

        # 1️⃣  Prepare toolset (and optional live clarification helper)
        tools = dict(self.get_tools("refactor"))

        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "refactor",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "refactor",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        # 2️⃣  Build & inject system prompt
        table_schemas_json = json.dumps(
            self._tables_overview(),
            indent=4,
            sort_keys=True,
        )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        client.set_system_message(
            build_refactor_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
                include_activity=include_activity,
            ).to_list(),
        )

        # 3️⃣  Launch interactive tool-use loop
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.refactor.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_refactor_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        # 4️⃣  Optionally wrap .result() to expose hidden reasoning
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore – runtime override

        return handle

    @functools.wraps(BaseKnowledgeManager.update, updated=())
    @log_manager_call(
        "KnowledgeManager",
        "update",
        payload_key="request",
        display_label="Updating notes",
    )
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        case_specific_instructions: str | None = None,
    ) -> "SteerableToolHandle":
        """
        Write-capable updates to knowledge tables (rows/columns).

        Parameters
        ----------
        text : str
            High-level description of the update request.
        _return_reasoning_steps : bool, default False
            When True, ``handle.result()`` returns ``(answer, messages)``.
        _parent_chat_context : list[dict] | None
            Optional upstream chat messages to seed the loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Duplex queues for interactive clarification.
        rolling_summary_in_prompts : bool | None
            Whether to include the rolling activity summary in prompts.
        _call_id : str | None
            Correlation id for event logging.
        case_specific_instructions : str | None
            Optional extra guidance injected into the system prompt.

        Returns
        -------
        SteerableToolHandle
            Handle that yields a summary of operations performed.
        """

        client = new_llm_client()

        # ── 1.  Expose tools + a *dynamic* request_clarification helper ──
        tools = dict(self.get_tools("update"))

        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "update",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "update",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        # ── 2.  Launch the interactive tool-use loop ──────────────────────
        # Add the system message with all tools
        table_schemas_json = json.dumps(
            self._tables_overview(),
            indent=4,
            sort_keys=True,
        )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
                include_activity=include_activity,
                case_specific_instructions=case_specific_instructions,
            ).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        # Optionally wrap .result() to expose reasoning
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    @functools.wraps(BaseKnowledgeManager.ask, updated=())
    @manager_tool
    @log_manager_call(
        "KnowledgeManager",
        "ask",
        payload_key="question",
        display_label="Checking notes",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        case_specific_instructions: str | None = None,
        _call_id: Optional[str] = None,
    ) -> "SteerableToolHandle":
        """
        Read-only questions over one or multiple knowledge tables.

        Parameters
        ----------
        text : str
            Natural-language question.
        _return_reasoning_steps : bool, default False
            When True, ``handle.result()`` returns ``(answer, messages)``.
        _parent_chat_context : list[dict] | None
            Optional upstream chat messages to seed the loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Duplex queues for interactive clarification.
        rolling_summary_in_prompts : bool | None
            Whether to include the rolling activity summary in prompts.
        case_specific_instructions : str | None
            Optional extra guidance injected into the system prompt.
        response_format : Any | None
            Optional JSON schema or dict-like structure to constrain the output.
        _call_id : str | None
            Correlation id for event logging.

        Returns
        -------
        SteerableToolHandle
            Handle that yields the final answer.
        """

        client = new_llm_client()

        # ── 1.  Expose tools + a *dynamic* request_clarification helper ──
        tables_overview = self._tables_overview()
        include_join_info = len(tables_overview) > 1

        # We decide in `ask` method whether to include the multi-table tools or not
        tools = dict(self.get_tools("ask"))
        if len(tables_overview) > 1:
            multi_table_tools = dict(self.get_tools("ask.multi_table"))
            tools.update(multi_table_tools)

        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "ask",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "KnowledgeManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        # ── 2.  Launch the interactive tool-use loop ──────────────────────
        # Add the system message with all tools
        table_schemas_json = json.dumps(tables_overview, indent=4, sort_keys=True)
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
                include_activity=include_activity,
                case_specific_instructions=case_specific_instructions,
                include_join_info=include_join_info,
            ).to_list(),
        )

        tool_policy_fn = self._default_ask_tool_policy
        # Maybe seed a synthetic `show_all` dump as the first tool call (ask only)
        text = await self._maybe_build_show_all_seed(text, tables_overview) or text

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=tool_policy_fn,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        # Optionally wrap .result() to expose reasoning
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # Helpers #
    # --------#

    def _get_columns(self, *, table: str) -> Dict[str, str]:
        """
        Return ``{column_name: column_type}`` for the given table.

        Parameters
        ----------
        table : str
            Logical table name (e.g. ``"Products"`` or ``"Contacts"`` when
            linkage is enabled).

        Returns
        -------
        dict[str, str]
            Mapping of column names to their Unify data types.
        """
        return _storage_get_columns(self, table=table)

    @functools.wraps(BaseKnowledgeManager.clear, updated=())
    def clear(self) -> None:
        """Drop all Knowledge-managed contexts and re-provision storage.

        Behaviour
        ---------
        - Deletes every child context under ``self._ctx`` (one per knowledge table).
        - Then re-provisions optional linked storage (e.g., Contacts) so future
          calls see a consistent schema.
        """
        km_prefix = f"{self._ctx}/"
        ctxs = unify.get_contexts(prefix=km_prefix)
        for full_ctx in list(ctxs.keys()):
            unify.delete_context(full_ctx)

        # Re-provision any required/linked storage

    # Tables

    def _create_table(
        self,
        *,
        name: str,
        description: str | None = None,
        columns: Dict[str, ColumnType] | None = None,
        unique_key_name: str = "row_id",
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, str]:
        """
        **Create** a brand-new table in the knowledge store.

        Parameters
        ----------
        name : str
                Canonical table name (must be unique within this manager).
        description : str | None, default ``None``
                Human-readable explanation of the table's purpose.
        columns : dict[str, ColumnType] | None
                Optional initial schema – mapping *column → type*.  If omitted an
                empty table is created and columns can be added later with
                :pyfunc:`_create_empty_column`. Colums names MUST be *snake case*.
                The column name `id` is reserved for internals, do *not* use this name.
        unique_key_name : str
                Every table *must* have a unique integer column which auto-increments
                upwards from 0. By default this is called `row_id`, but the name can
                be customized to be more descriptive for the table. For example,
                `team_id`, `company_id`, `product_id`, or anything else. This is
                managed automatically, it should not be included in the `columns`
                argument, and data is *never written* to this unique column.
        auto_counting : dict[str, Optional[str]] | None, default ``None``
                Optional auto-counting configuration for the table. If provided,
                the table will be configured to auto-count the provided columns.
                Keys are column names to auto-increment, values are parent counter
                names (None for independent counters) e.g.
                {
                    "company_id": None,
                    "department_id": "company_id",
                }

        Returns
        -------
        dict[str, str]
                Backend response describing success or failure (driver specific).
        """
        return _storage_create_table(
            self,
            name=name,
            description=description,
            columns=columns,
            unique_key_name=unique_key_name,
            auto_counting=auto_counting,
        )

    @read_only
    def _tables_overview(
        self,
        *,
        include_column_info: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Show the information for **all** tables.

        Parameters
        ----------
        include_column_info : bool, default ``True``
                When *True* each table entry also contains a
                ``"columns": {name: type}`` mapping.

        Returns
        -------
        dict[str, dict]
                Mapping ``table_name → {"description": str, "columns": {...}}``.
                If *include_column_info* is *False* the ``"columns"`` key is omitted.
        """
        return _storage_tables_overview(self, include_column_info=include_column_info)

    def _rename_table(
        self,
        *,
        old_name: str,
        new_name: str,
    ) -> Dict[str, str]:
        """
        **Rename** an existing table.

        Parameters
        ----------
        old_name : str
                Current table identifier.
        new_name : str
                New identifier (must not clash with existing tables).

        Returns
        -------
        dict[str, str]
                Backend acknowledgement / error message.
        """
        return _storage_rename_table(self, old_name=old_name, new_name=new_name)

    def _delete_tables(
        self,
        *,
        tables: Union[str, List[str]],
        startswith: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        **Drop** an entire table *and* all its rows.

        Parameters
        ----------
        tables : str | list[str]
                Target table name(s).
        startswith : str | None, default None
                If provided, also delete all tables whose names start with this prefix.

        Returns
        -------
        list[dict[str, str]]
            Confirmations / errors from the backend.
        """
        return _storage_delete_tables(self, tables=tables, startswith=startswith)

    # Columns

    def _create_empty_column(
        self,
        *,
        table: str,
        column_name: str,
        column_type: ColumnType | str,
    ) -> Dict[str, str]:
        """
        Add a **new, initially empty column** to *table*.

        Parameters
        ----------
        table : str
                Target table.
        column_name : str
                New column identifier, MUST be *snake case*.
                The column name `id` is reserved for internals, do *not* use this name.
        column_type : ColumnType | str
                Logical type, e.g. ``"str"``, ``"float"``, ``"datetime"``.

        Returns
        -------
        dict[str, str]
                Backend response.
        """
        return _op_create_empty_column(
            self,
            table=table,
            column_name=column_name,
            column_type=column_type,
        )

    def _create_derived_column(
        self,
        *,
        table: str,
        column_name: str,
        equation: str,
    ) -> Dict[str, str]:
        """
        Create a **derived column** whose value is computed from other columns
        via an arbitrary Python *equation*.

        Parameters
        ----------
        table : str
                Table to modify.
        column_name : str
                Name of the new derived column, MUST be *snake case*.
                The column name `id` is reserved for internals, do *not* use this name.
        equation : str
                Python expression evaluated per-row (column names appear as
                variables).  Example: ``(x**2 + y**2) ** 0.5``.

        Returns
        -------
        dict[str, str]
                Backend acknowledgement.
        """
        return _op_create_derived_column(
            self,
            table=table,
            column_name=column_name,
            equation=equation,
        )

    def _delete_column(
        self,
        *,
        table: str,
        column_name: str,
    ) -> Dict[str, str]:
        """
        **Remove** a column (and its data) from *table*.

        Parameters
        ----------
        table : str
                Table name.
        column_name : str
                Column to drop, MUST be *snake case*.

        Returns
        -------
        dict[str, str]
                Backend confirmation or error.
        """
        return _op_delete_column(self, table=table, column_name=column_name)

    def _rename_column(
        self,
        *,
        table: str,
        old_name: str,
        new_name: str,
    ) -> Dict[str, str]:
        """
        **Rename** a column inside *table*.

        Parameters
        ----------
        table : str
                Table identifier.
        old_name : str
                Existing column name, MUST be *snake case*.
        new_name : str
                Desired new name, MUST be *snake case*.
                The column name `id` is reserved for internals, do *not* use this name.

        Returns
        -------
        dict[str, str]
                Backend response.
        """
        # Short-circuit obvious no-op and invalid rename targets to avoid any backend call
        return _op_rename_column(
            self,
            table=table,
            old_name=old_name,
            new_name=new_name,
        )

    def _copy_column(
        self,
        *,
        source_table: str,
        column_name: str,
        dest_table: str,
    ) -> Dict[str, Any]:
        """
        Copy a column's values from one table to another.

        Parameters
        ----------
        source_table : str
            Table to read values from.
        column_name : str
            Column to copy.
        dest_table : str
            Destination table that will receive rows containing ``column_name``.

        Returns
        -------
        dict[str, str]
            Summary of the copy operation including counts and source/dest info.

        Notes
        -----
        Implemented by attaching the matching logs to the destination context
        via ``unify.add_logs_to_context``.
        """
        return _op_copy_column(
            self,
            source_table=source_table,
            column_name=column_name,
            dest_table=dest_table,
        )

    def _move_column(
        self,
        *,
        source_table: str,
        column_name: str,
        dest_table: str,
    ) -> Dict[str, Any]:
        """
        Move a column from one table to another.

        Parameters
        ----------
        source_table : str
            Source table.
        column_name : str
            Column to move.
        dest_table : str
            Destination table.

        Returns
        -------
        dict[str, str]
            Summary containing the copy and delete sub‑results.

        Notes
        -----
        Implemented as ``_copy_column`` followed by ``_delete_column`` on the
        source table.
        """
        return _op_move_column(
            self,
            source_table=source_table,
            column_name=column_name,
            dest_table=dest_table,
        )

    def _transform_column(
        self,
        *,
        table: str,
        column_name: str,
        equation: str,
    ) -> Dict[str, Any]:
        """
        Transform a column in‑place according to a Python ``equation``.

        Parameters
        ----------
        table : str
            Table to modify.
        column_name : str
            Column to transform.
        equation : str
            Per‑row Python expression where column names are variables.

        Returns
        -------
        dict[str, str]
            Summary of the create/delete/rename steps.

        Notes
        -----
        The operation is implemented as:
        1. Create a temporary derived column from ``equation``.
        2. Delete the original column.
        3. Rename the temporary column back to ``column_name``.
        """
        return _op_transform_column(
            self,
            table=table,
            column_name=column_name,
            equation=equation,
        )

    #  Row-level deletion

    def _delete_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Delete every log matching ``filter`` across one or more tables.

        Parameters
        ----------
        filter : str | None, default ``None``
            Row‑level predicate evaluated per table. ``None`` deletes nothing
            (no predicate).
        offset : int, default ``0``
            Pagination offset into each table before applying deletion.
        limit : int, default ``100``
            Maximum number of rows considered per table.
        tables : list[str] | None, default ``None``
            Subset of tables to scan; ``None`` means all tables managed by this
            instance (and optionally ``Contacts`` when linked).

        Returns
        -------
        dict[str, str]
            Mapping ``table_name → backend message / "no-op"``.
        """
        return _op_delete_rows(
            self,
            filter=filter,
            offset=offset,
            limit=limit,
            tables=tables,
        )

    # Row creation / update

    def _add_rows(
        self,
        *,
        table: str,
        rows: List[Dict[str, Any]],
    ) -> ToolOutcome:
        """
        **Insert** one or many rows into *table*.

        Missing columns are auto-created (type inferred via JSON schema
        rules) before the insert.

        Parameters
        ----------
        table : str
                Destination table.
        rows : list[dict[str, Any]]
                Sequence of row dictionaries. Dictionary keys (column names) MUST be *snake case*.

        Returns
        -------
        dict[str, str]
                Backend confirmation.
        """
        try:
            res = _op_add_rows(self, table=table, rows=rows)
        except Exception as e:
            return {"outcome": "error", "details": {"error": str(e)}}
        return {"outcome": "rows added successfully", "details": {"length": len(res)}}

    def _update_rows(
        self,
        *,
        table: str,
        updates: Dict[int, Dict[str, Any]],
    ) -> ToolOutcome:
        """
        Update existing rows identified by their table‑specific unique id.

        Parameters
        ----------
        table : str
            Target table.
        updates : dict[int, dict[str, Any]]
            Mapping of unique row ids (e.g. ``row_id``, ``team_id``) to a dict
            of new field values. Unspecified fields are left unchanged.

        Returns
        -------
        dict[str, str]
            Backend response from ``unify.update_logs``.
        """
        try:
            res = _op_update_rows(self, table=table, updates=updates)
        except Exception as e:
            return {"outcome": "error", "details": {"error": str(e)}}
        return {"outcome": "rows updated successfully", "details": res}

    # File ingestion / deprecation

    # async def _ingest_documents(
    #     self,
    #     *,
    #     filenames: Union[str, List[str]],
    #     table: str = "Content",
    #     replace_existing: bool = True,
    #     batch_size: int = 3,
    #     embed_along: bool = True,
    #     embedding_config: Dict[str, Any] | None = None,
    #     auto_counting: Dict[str, Optional[str]] | None = None,
    #     allowed_columns: List[str] | None = None,
    #     **parse_options: Any,
    # ) -> Dict[str, Any]:
    #     """
    #     Ingest one or more documents efficiently with streaming.
    #     This tool handles the complete workflow for document ingestion:
    #     1. Stream parse documents in batches
    #     2. Delete existing records that match (if replace_existing=True)
    #     3. Insert new records as they become available
    #     Args:
    #         filenames: Single filename (str) or list of filenames to ingest
    #         table: Target table (default: "content")
    #         replace_existing: Whether to delete old records first
    #         batch_size: Number of documents to parse in parallel
    #         **parse_options: Options passed to parser
    #     Returns:
    #         Dict with success status, per-file results, and aggregate statistics
    #     """
    #     try:
    #         if not self._file_manager:
    #             return {"success": False, "error": "FileManager not available"}

    #         # Normalize input to always be a list
    #         if isinstance(filenames, str):
    #             filenames = [filenames]

    #         if not filenames:
    #             return {"success": False, "error": "No filenames provided"}

    #         print(
    #             f"📄 Processing {len(filenames)} document{'s' if len(filenames) > 1 else ''} with batch size {batch_size}...",
    #         )

    #         # Initialize tracking
    #         total_inserted = 0
    #         total_deleted = 0
    #         file_results = {}
    #         batch_records = []
    #         batch_files = []
    #         processed_count = 0
    #         total_inserted_log_event_ids = []

    #         # Normalise allowed_columns to a set for fast membership tests
    #         allowed_columns_set = set(allowed_columns) if allowed_columns else None

    #         # Process documents as they complete parsing
    #         async for result in self._file_manager.parse_async(
    #             filenames,
    #             batch_size=batch_size,
    #             auto_counting=auto_counting,
    #             document_index_offset=0,
    #             **parse_options,
    #         ):
    #             filename = result.get("filename")

    #             if result["status"] == "error":
    #                 file_results[filename] = {
    #                     "filename": filename,
    #                     "success": False,
    #                     "error": result["error"],
    #                     "inserted": 0,
    #                     "deleted": 0,
    #                 }
    #                 continue

    #             records = result.get("records", [])
    #             if not records:
    #                 file_results[filename] = {
    #                     "filename": filename,
    #                     "success": False,
    #                     "error": "No records extracted",
    #                     "inserted": 0,
    #                     "deleted": 0,
    #                 }
    #                 continue

    #             # Delete existing records if requested
    #             deleted_count = 0
    #             if replace_existing and records:
    #                 first_record = records[0]
    #                 doc_filters = []

    #                 if doc_id := first_record.get("document_id"):
    #                     doc_filters.append(f"document_id == '{doc_id}'")

    #                 if file_path := first_record.get("file_path"):
    #                     # Clean up temp directory from path for matching
    #                     clean_file_path = file_path
    #                     if "/tmp/" in clean_file_path:
    #                         parts = clean_file_path.split("/tmp/")
    #                         if len(parts) > 1:
    #                             after_tmp = parts[1]
    #                             subparts = after_tmp.split("/", 1)
    #                             if len(subparts) > 1:
    #                                 clean_file_path = subparts[1]
    #                     # Use Python string method for pattern matching
    #                     doc_filters.append(f"file_path.endswith('{clean_file_path}')")

    #                 if doc_fingerprint := first_record.get("document_fingerprint"):
    #                     doc_filters.append(
    #                         f"document_fingerprint == '{doc_fingerprint}'",
    #                     )

    #                 if doc_filters:
    #                     filter_expr = " or ".join(f"({f})" for f in doc_filters)
    #                     try:
    #                         # Count records to be deleted using IDs-only
    #                         target_ctx = self._ctx_for_table(table)
    #                         ids_to_delete = unify.get_logs(
    #                             context=target_ctx,
    #                             filter=filter_expr,
    #                             return_ids_only=True,
    #                         )
    #                         deleted_count = len(ids_to_delete)

    #                         if deleted_count > 0:
    #                             self._delete_rows(tables=[table], filter=filter_expr)
    #                             total_deleted += deleted_count
    #                         print(
    #                             f"✅ Deleted {deleted_count} old records for {filename}",
    #                         )
    #                     except Exception as e:
    #                         print(
    #                             f"❌ Failed to delete old records for {filename}: {e}",
    #                         )

    #             # Add to batch
    #             # After deletion, filter records to allowed columns if provided
    #             if allowed_columns_set is not None:
    #                 filtered_records = []
    #                 for rec in records:
    #                     filtered = {
    #                         k: v for k, v in rec.items() if k in allowed_columns_set
    #                     }
    #                     filtered_records.append(filtered)
    #                 records = filtered_records

    #             batch_records.extend(records)
    #             batch_files.append(
    #                 {
    #                     "filename": filename,
    #                     "record_count": len(records),
    #                     "deleted_count": deleted_count,
    #                 },
    #             )
    #             processed_count += 1

    #             print(f"✅ Parsed {filename}: {len(records)} records")

    #             # Insert batch when we have processed batch_size documents or it's the last one
    #             if len(batch_files) >= batch_size or processed_count == len(filenames):
    #                 if batch_records:
    #                     try:
    #                         print(
    #                             f"📥 Inserting batch of {len(batch_records)} records from {len(batch_files)} documents...",
    #                         )
    #                         result = self._add_rows(table=table, rows=batch_records)
    #                         inserted_log_event_ids = [log.id for log in result]
    #                         total_inserted_log_event_ids.extend(inserted_log_event_ids)
    #                         total_inserted += len(batch_records)

    #                         # Update file results for this batch
    #                         for file_info in batch_files:
    #                             file_results[file_info["filename"]] = {
    #                                 "filename": file_info["filename"],
    #                                 "success": True,
    #                                 "inserted": file_info["record_count"],
    #                                 "deleted": file_info["deleted_count"],
    #                                 "error": None,
    #                             }

    #                         print(f"✅ Batch inserted successfully")

    #                         # Optional: embed along after this batch is inserted
    #                         if embed_along and embedding_config:
    #                             try:
    #                                 tables_cfg = embedding_config.get("tables", {})
    #                                 table_cfg = tables_cfg.get(table, {})
    #                                 to_embed = table_cfg.get("columns_to_embed", [])
    #                                 # Restrict embedding to rows just inserted in this batch
    #                                 for col in to_embed:
    #                                     src = col.get("source_column")
    #                                     dst = col.get("target_column")
    #                                     if src and dst:
    #                                         print(
    #                                             f"🔮 Embedding {table}.{src} -> {dst} (embed_along) for {len(inserted_log_event_ids)} rows",
    #                                         )
    #                                         self._vectorize_column(
    #                                             table=table,
    #                                             source_column=src,
    #                                             target_column_name=dst,
    #                                             from_ids=inserted_log_event_ids,
    #                                         )
    #                                         print(
    #                                             f"✅ Embedded {table}.{src} -> {dst} (embed_along) for {len(inserted_log_event_ids)} rows",
    #                                         )
    #                             except Exception as e:
    #                                 print(f"❌ Failed to embed along: {e}")

    #                     except Exception as e:
    #                         # Update file results for failed batch
    #                         for file_info in batch_files:
    #                             file_results[file_info["filename"]] = {
    #                                 "filename": file_info["filename"],
    #                                 "success": False,
    #                                 "inserted": 0,
    #                                 "deleted": file_info["deleted_count"],
    #                                 "error": f"Batch insertion failed: {str(e)}",
    #                             }
    #                         print(f"❌ Failed to insert batch: {e}")

    #                     # Clear batch for next set
    #                     batch_records = []
    #                     batch_files = []

    #         # Calculate summary statistics
    #         successful_files = sum(
    #             1 for fr in file_results.values() if fr.get("success", False)
    #         )
    #         failed_files = len(filenames) - successful_files

    #         return {
    #             "success": failed_files == 0,
    #             "total_files": len(filenames),
    #             "successful_files": successful_files,
    #             "failed_files": failed_files,
    #             "total_records": total_inserted,
    #             "total_inserted": total_inserted,
    #             "total_deleted": total_deleted,
    #             "file_results": list(file_results.values()),
    #             "inserted_log_event_ids": total_inserted_log_event_ids,
    #         }

    #     except Exception as e:
    #         return {"success": False, "error": str(e)}

    # Vector Search Helpers
    def _vectorize_column(
        self,
        table: str,
        source_column: str,
        target_column_name: str,
        *,
        from_ids: List[int] | None = None,
    ) -> None:
        """
        Ensure a vector column exists, creating it if necessary.

        Parameters
        ----------
        table : str
            The table to ensure the vector column in.
        source_column : str
            The existing column whose text will be embedded.
        target_column_name : str
            Name of the embedding column to create/ensure (snake case).

        Returns
        -------
        None
        """
        return _op_vectorize_column(
            self,
            table=table,
            source_column=source_column,
            target_column_name=target_column_name,
            from_ids=from_ids,
        )

    @read_only
    def _search(
        self,
        *,
        table: str,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search within a single knowledge table using one or more source expressions.

        Parameters
        ----------
        table : str
                The table to search within.
        references : Dict[str, str]
                Mapping from a source expression (plain column or derived Unify expression) to the
                reference text to compare against. Supports multiple expressions; when more than one
                is provided the ranking uses a sum of cosine distances over all terms.
        k : int, default 10
                Maximum number of rows to return. Must be <= 1000.
        filter : str | None, default ``None``
                Row-level predicate (evaluated with column names as variables).
                *None* returns all rows.

        Returns
        -------
        list[dict[str, Any]]
                Up to ``k`` rows sorted by ascending semantic distance (best match first).
                If similarity search yields fewer than ``k`` rows and there are more rows
                overall, the remainder is backfilled from ``unify.get_logs(limit=k)`` in
                returned order, skipping duplicates based on each table's unique id.
        """
        return _srch_search(
            self,
            table=table,
            references=references,
            k=k,
            filter=filter,
        )

    @read_only
    def _search_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over the result of joining two tables.

        Parameters
        ----------
        tables : list[str]
            Exactly two table names, e.g. `["A", "B"]`.

        join_expr : str
            Boolean join condition using the same table identifiers given in
            `tables`, e.g. `"A.user_id == B.user_id"`.

        select : dict[str, str]
            Mapping of source columns to output column names in the join
            result, e.g. `{ "A.user_id": "user_identifier", "B.score":
            "user_score" }`.

        mode : str, default "inner"
            Join mode. Typical values: "inner", "left", "right", "outer".

        left_where : str | None, default None
            Optional row-level predicate applied to the left table before the
            join, e.g. `"user_id == 1"`.

        right_where : str | None, default None
            Optional row-level predicate applied to the right table before the
            join.

        references : dict[str, str]
            Mapping of source expressions (columns or expressions in the join
            result) to reference text for semantic similarity. When multiple
            entries are provided, their scores are combined for ranking.

        k : int, default 5
            Maximum number of rows to return. Must be <= 1000.

        filter : str | None, default ``None``
                Row-level predicate (evaluated with column names as variables).
                *None* returns all rows.

        Returns
        -------
        list[dict[str, Any]]
            Up to `k` rows from the joined result, sorted by best semantic
            match first. If the similarity search yields fewer than `k` rows and
            there are more rows overall in the joined context, the remainder is
            backfilled from `unify.get_logs(limit=k)` in returned order, skipping
            duplicates based on the joined table's unique id.
        """
        return _srch_search_join(
            self,
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            references=references,
            k=k,
            filter=filter,
        )

    @read_only
    def _search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over the result of chaining multiple joins.

        Parameters
        ----------
        joins : list[dict]
            Ordered list of join steps. Each step supports the keys:

            - "tables" (list[str], required): Exactly two table names for this
              step. The special placeholder values "$prev", "__prev__", or
              "_" may be used to refer to the result of the previous step (not
              allowed in the first step).
            - "join_expr" (str, required): Join predicate for this step using
              the table identifiers declared in "tables".
            - "select" (dict[str, str], required): Mapping of source columns to
              output names for this step's result.
            - "mode" (str, optional): Join mode for this step (default:
              "inner").
            - "left_where" (str | None, optional): Row-level predicate applied
              to the left table of this step before joining.
            - "right_where" (str | None, optional): Row-level predicate applied
              to the right table of this step before joining.

        references : dict[str, str]
            Mapping of expressions in the final result to reference text for
            semantic similarity. Multiple entries are combined for ranking.

        k : int, default 5
            Maximum number of rows to return. Must be <= 1000.

        Returns
        -------
        list[dict[str, Any]]
            Up to `k` rows from the final joined result, best semantic match
            first. If the similarity search yields fewer than `k` rows and
            there are more rows overall in the final joined context, the
            remainder is backfilled from `unify.get_logs(limit=k)` in returned
            order, skipping duplicates based on the final context's unique id.
        """

        return _srch_search_multi_join(
            self,
            joins=joins,
            references=references,
            k=k,
            filter=filter,
        )

    # Search

    ## private helper

    @read_only
    def _filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        **Filter search** across one or more tables using a Python boolean
        expression.

        Parameters
        ----------
        filter : str | None, default ``None``
                Row-level predicate (evaluated with column names as variables).
                *None* returns all rows.
        offset : int, default ``0``
                Pagination offset (0-based).
        limit : int, default ``100``
                Maximum rows per table. Must be <= 1000.
        tables :  str | list[str]
                Subset of tables to scan; ``None`` → all tables.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
                Mapping ``table_name → [row_dict, …]``.
        """
        return _srch_filter(
            self,
            filter=filter,
            offset=offset,
            limit=limit,
            tables=tables,
        )

    @read_only
    def _reduce(
        self,
        *,
        table: str,
        metric: str,
        keys: str | List[str],
        filter: Optional[str | Dict[str, str]] = None,
        group_by: Optional[str | List[str]] = None,
    ) -> Any:
        """
        Compute reduction metrics over a single knowledge table.

        Parameters
        ----------
        table : str
            Logical table name managed by this KnowledgeManager (for example
            ``\"Content\"``, ``\"Products\"``, or ``\"Contacts\"`` when linkage
            is enabled).
        metric : str
            Reduction metric to compute. Supported values (case-insensitive) are
            ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
            ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
        keys : str | list[str]
            One or more numeric columns in ``table`` to aggregate. A single
            column name returns a scalar; a list of column names computes the
            metric independently per key and returns a ``{key -> value}``
            mapping.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s) using the same Python
            syntax as the :py:meth:`_filter` tool. When a string, the
            expression is applied uniformly; when a dict, each key maps to its
            own filter expression.
        group_by : str | list[str] | None, default None
            Optional column(s) to group by. Use a single column name for one
            grouping level, or a list such as ``[\"category\", \"row_id\"]`` to
            group hierarchically in that order. When provided, the result
            becomes a nested mapping keyed by group values, mirroring
            :func:`unify.get_logs_metric` behaviour.

        Returns
        -------
        Any
            Metric value(s) computed over the resolved table context:

            * Single key, no grouping  → scalar (float/int/str/bool).
            * Multiple keys, no grouping → ``dict[key -> scalar]``.
            * With grouping             → nested ``dict`` keyed by group values.
        """
        ctx = self._ctx_for_table(table)
        return reduce_logs(
            context=ctx,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    @read_only
    def _filter_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Join two tables and return rows from the joined result with optional filtering.

        Delegates to DataManager.filter_join via a single server-side
        round-trip; no temporary context is created.

        Parameters
        ----------
        tables : list[str]
            Exactly two table names, e.g. `["A", "B"]`.

        join_expr : str
            Boolean join condition using the same table identifiers given in
            `tables`, e.g. `"A.user_id == B.user_id"`.

        select : dict[str, str]
            Mapping of source columns to output column names in the joined
            result, e.g. `{ "A.user_id": "user_identifier", "B.score":
            "user_score" }`.

        mode : str, default "inner"
            Join mode. Typical values: "inner", "left", "right", "outer".

        left_where : str | None, default None
            Optional row-level predicate applied to the left table before the
            join.

        right_where : str | None, default None
            Optional row-level predicate applied to the right table before the
            join.

        result_where : str | None, default None
            Optional row-level predicate applied to the joined result when
            returning rows. This predicate may only reference the output column
            names created by `select`.

        result_limit : int, default 100
            Maximum number of rows to return. Must be <= 1000.

        result_offset : int, default 0
            Pagination offset into the result set.

        Returns
        -------
        list[dict[str, Any]]
            Rows from the joined result matching the provided filters.
        """

        return _srch_filter_join(
            self,
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )

    @read_only
    def _filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Chain together multiple joins, then return rows from the final joined result.

        Delegates to DataManager.filter_multi_join; temporary contexts
        are managed internally and cleaned up automatically.

        Parameters
        ----------
        joins : list[dict]
            Ordered list of join steps. Each step supports the keys:

            - "tables" (list[str], required): Exactly two table names for this
              step. The placeholders "$prev", "__prev__", or "_" may be used
              to refer to the result of the previous step (not valid in the
              first step).
            - "join_expr" (str, required): Join predicate for this step using
              the table identifiers declared in "tables".
            - "select" (dict[str, str], required): Mapping of source columns to
              output names for this step's result.
            - "mode" (str, optional): Join mode for this step (default:
              "inner").
            - "left_where" (str | None, optional): Row-level predicate applied
              to the left table of this step before joining.
            - "right_where" (str | None, optional): Row-level predicate applied
              to the right table of this step before joining.

        result_where : str | None, default None
            Optional row-level predicate applied when returning rows from the
            final joined result. This predicate may only reference the output
            column names created by the final step's `select` mapping.

        result_limit : int, default 100
            Maximum number of rows to return. Must be <= 1000.

        result_offset : int, default 0
            Pagination offset into the final result set.

        Returns
        -------
        list[dict[str, Any]]
            Rows from the final joined result matching the provided filters.
        """

        return _srch_filter_multi_join(
            self,
            joins=joins,
            result_where=result_where,
            result_limit=result_limit,
            result_offset=result_offset,
        )
