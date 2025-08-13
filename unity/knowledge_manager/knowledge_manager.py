import os
import asyncio
import uuid
import unify
import functools
import requests
from typing import Any, Dict, List, Optional, Callable, Union

import json
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..helpers import _handle_exceptions
from .types import ColumnType
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from ..helpers import _handle_exceptions
from .base import BaseKnowledgeManager
from .prompt_builders import (
    build_update_prompt,
    build_ask_prompt,
    build_refactor_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.semantic_search import ensure_vector_for_source, ensure_sum_cosine_column


class KnowledgeManager(BaseKnowledgeManager):
    def __init__(
        self,
        *,
        rolling_summary_in_prompts: bool = True,
        include_contacts: bool = True,
    ) -> None:
        """
        KnowledgeManager now **directly manipulates** the root-level
        ``Contacts`` table instead of calling the public ContactManager API.
        """

        self._refactor_tools = methods_to_tool_dict(
            # Ask
            self.ask,
            # Tables
            self._create_table,
            self._rename_table,
            self._delete_tables,
            # Columns
            self._rename_column,
            self._copy_column,
            self._move_column,
            self._delete_column,
            self._create_empty_column,
            self._create_derived_column,
            self._transform_column,
            self._vectorize_column,
            # Rows
            self._delete_rows,
            self._update_rows,
            include_class_name=False,
        )

        # ── immutable built-ins for *Contacts* ───────────────────────────
        self._CONTACT_REQUIRED_COLUMNS: set[str] = {
            "contact_id",
            "first_name",
            "surname",
            "email_address",
            "phone_number",
            "whatsapp_number",
            "description",
        }

        self._ask_tools = {
            **methods_to_tool_dict(
                self._tables_overview,
                self._filter,
                self._search,
                self._filter_join,
                self._filter_multi_join,
                self._search_join,
                self._search_multi_join,
            ),
        }

        self._update_tools = {
            **self._refactor_tools,
            **methods_to_tool_dict(
                self._add_rows,
                include_class_name=False,
            ),
        }

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # ------------------------------------------------------------------
        # Optional Contacts-table linkage
        # ------------------------------------------------------------------
        self._include_contacts: bool = include_contacts

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a KnowledgeManager."
        self._ctx = f"{read_ctx}/Knowledge" if read_ctx else "Knowledge"

        # Only compute the Contacts context if the caller requested integration.
        self._contacts_ctx = (
            (f"{read_ctx}/Contacts" if read_ctx else "Contacts")
            if include_contacts
            else None
        )

    # Helpers #
    # --------#

    def _ctx_for_table(self, table: str) -> str:
        """Return the correct Unify context for *table*.

        When this instance was created with ``include_contacts=False`` any
        attempt to reference the *Contacts* table is considered an error to
        avoid hidden cross-coupling between the knowledge store and the
        contact book.
        """

        if table == "Contacts":
            if not self._include_contacts or self._contacts_ctx is None:
                raise ValueError(
                    "This KnowledgeManager instance was initialised with include_contacts=False so it cannot access the Contacts table.",
                )
            return self._contacts_ctx

        return f"{self._ctx}/{table}"

    def _look_first_tool_policy(self, step: int, tls: Dict[str, Callable]):
        if step < 1:
            return "required", methods_to_tool_dict(
                self._filter,
                self._search,
                self._filter_join,
                self._search_join,
                include_class_name=False,
            )
        return "auto", tls

    # Public #
    # -------#

    # English-Text Command

    @functools.wraps(BaseKnowledgeManager.refactor, updated=())
    async def refactor(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> "SteerableToolHandle":

        # ── 0.  Emit *incoming* ManagerMethod event ──────────────────────
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "KnowledgeManager",
            "refactor",
            phase="incoming",
            command=text,
        )

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # 1️⃣  Prepare toolset (and optional live clarification helper)
        tools = dict(self._refactor_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "KnowledgeManager.refactor was invoked without both "
                        "clarification queues but the model requested one.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # 2️⃣  Build & inject system prompt
        table_schemas_json = json.dumps(self._tables_overview(), indent=4)
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
            ),
        )

        # 3️⃣  Launch interactive tool-use loop
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.refactor.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=self._look_first_tool_policy,
            preprocess_msgs=self._inject_broader_context,
        )

        # ── 3.  Add logging wrapper so every handle-interaction is traced ─
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "KnowledgeManager",
            "refactor",
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
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> "SteerableToolHandle":

        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "KnowledgeManager",
            "update",
            phase="incoming",
            request=text,
        )

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # ── 1.  Expose tools + a *dynamic* request_clarification helper ──
        tools = dict(self._update_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Query the user for more information, and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "TranscriptManager.ask was called without both "
                        "clarification queues but the model requested clarifications.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 2.  Launch the interactive tool-use loop ──────────────────────
        # Add the system message with all tools
        table_schemas_json = json.dumps(self._tables_overview(), indent=4)
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
            ),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=self._look_first_tool_policy,
            preprocess_msgs=self._inject_broader_context,
        )

        # ── 3a.  Add logging wrapper  ─────────────────────────────────────
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "KnowledgeManager",
            "update",
        )

        # ── 3b.  Optionally wrap .result() to expose reasoning  ───────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    @functools.wraps(BaseKnowledgeManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> "SteerableToolHandle":
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "KnowledgeManager",
            "ask",
            phase="incoming",
            question=text,
        )

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # ── 1.  Expose tools + a *dynamic* request_clarification helper ──
        tools = dict(self._ask_tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Query the user for more information, and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "KnowledgeManager.retrieve was called without both "
                        "clarification queues but the model requested clarifications.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 2.  Launch the interactive tool-use loop ──────────────────────
        # Add the system message with all tools
        table_schemas_json = json.dumps(self._tables_overview(), indent=4)
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
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
            preprocess_msgs=self._inject_broader_context,
        )

        # ── 3a.  Add logging wrapper  ─────────────────────────────────────
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "KnowledgeManager",
            "ask",
        )

        # ── 3b.  Optionally wrap .result() to expose reasoning  ───────────
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
        proj = unify.active_project()
        ctx = self._ctx_for_table(table)
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={ctx}"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

    # Private #
    # --------#

    # Tables

    def _create_table(
        self,
        *,
        name: str,
        description: str | None = None,
        columns: Dict[str, ColumnType] | None = None,
        unique_column_name: str = "row_id",
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
        unique_column_name : str
            Every table *must* have a unique integer column which auto-increments
            upwards from 0. By default this is called `row_id`, but the name can
            be customized to be more descriptive for the table. For example,
            `team_id`, `company_id`, `product_id`, or anything else. This is
            managed automatically, it should not be included in the `columns`
            argument, and data is *never written* to this unique column.

        Returns
        -------
        dict[str, str]
            Backend response describing success or failure (driver specific).
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{name}"
        unify.create_context(
            ctx,
            unique_column_ids=unique_column_name,
            description=description,
        )

        # Always add the generated primary-key unless the caller supplied it.
        if columns is None:
            columns = {}
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {"project": proj, "context": ctx, "fields": columns}
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

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
        tables = {
            k[len(f"{self._ctx}/") :]: {"description": v}
            for k, v in unify.get_contexts(prefix=f"{self._ctx}/").items()
        }

        # Optionally expose root-level Contacts when linkage is enabled.
        if (
            self._include_contacts
            and self._contacts_ctx is not None
            and self._contacts_ctx in unify.get_contexts()
        ):
            tables["Contacts"] = {
                "description": unify.get_contexts()[self._contacts_ctx],
            }
        if not include_column_info:
            return tables
        return {
            k: {**v, "columns": self._get_columns(table=k)} for k, v in tables.items()
        }

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
        proj = unify.active_project()
        old_name = f"{self._ctx}/{old_name}"
        new_name = f"{self._ctx}/{new_name}"
        url = f"{unify.BASE_URL}/project/{proj}/contexts/{old_name}/rename"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {"name": new_name}
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _delete_tables(
        self,
        *,
        tables: Union[str, List[str]],
        startswith: Optional[str] = None,
    ) -> Dict[str, str]:
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
        if isinstance(tables, str):
            tables = [tables]
        rets = list()
        for table in tables:
            rets.append(unify.delete_context(self._ctx_for_table(table)))
        if startswith is None:
            return rets
        contexts = unify.get_contexts(prefix=f"{self._ctx}/{startswith}")
        for ctx in contexts:
            rets.append(unify.delete_context(ctx))
        return rets

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
        proj = unify.active_project()
        ctx = self._ctx_for_table(table)
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": proj,
            "context": ctx,
            "fields": {column_name: column_type},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

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
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        equation = equation.replace("{", "{lg:")
        json_input = {
            "project": unify.active_project(),
            "context": self._ctx_for_table(table),
            "key": column_name,
            "equation": equation,
            "referenced_logs": {"lg": {"context": self._ctx_for_table(table)}},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        return response.json()

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
        table_ctx = unify.get_context(self._ctx_for_table(table))
        unique_column_name = table_ctx["unique_column_ids"]
        # Guard against removal of mandatory columns
        if (table == "Contacts" and column_name in self._CONTACT_REQUIRED_COLUMNS) or (
            table != "Contacts" and column_name == unique_column_name
        ):
            raise ValueError(
                f"❌  Column '{column_name}' is mandatory and cannot be deleted.",
            )

        url = f"{os.environ['UNIFY_BASE_URL']}/logs?delete_empty_logs=True"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": unify.active_project(),
            "context": self._ctx_for_table(table),
            "ids_and_fields": [[None, column_name]],
            "source_type": "all",
        }
        response = requests.request("DELETE", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

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
        proj = unify.active_project()
        ctx = self._ctx_for_table(table)
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/rename_field"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": proj,
            "context": ctx,
            "old_field_name": old_name,
            "new_field_name": new_name,
        }
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _copy_column(
        self,
        *,
        source_table: str,
        column_name: str,
        dest_table: str,
    ) -> Dict[str, str]:
        """
        **Copy** *column_name* from *source_table* to *dest_table*.

        Internally this attaches every log in *source_table* that contains
        *column_name* to the destination context via
        :pyfunc:`unify.add_logs_to_context`.
        """
        src_ctx = self._ctx_for_table(source_table)
        dest_ctx = self._ctx_for_table(dest_table)

        log_ids = unify.get_logs(
            context=src_ctx,
            filter=f"{column_name} is not None",
            limit=100_000,
            return_ids_only=True,
        )
        unify.add_logs_to_context(
            log_ids,
            context=dest_ctx,
            project=unify.active_project(),
        )
        return {
            "status": "copied",
            "rows": str(len(log_ids)),
            "from": source_table,
            "to": dest_table,
            "column": column_name,
        }

    def _move_column(
        self,
        *,
        source_table: str,
        column_name: str,
        dest_table: str,
    ) -> Dict[str, str]:
        """
        **Move** *column_name* from *source_table* to *dest_table*.

        Implemented as `_copy_column` + `_delete_column`.
        """
        copy_res = self._copy_column(
            source_table=source_table,
            column_name=column_name,
            dest_table=dest_table,
        )
        del_res = self._delete_column(table=source_table, column_name=column_name)
        return {
            "status": "moved",
            "copy_result": str(copy_res),
            "delete_result": str(del_res),
        }

    def _transform_column(
        self,
        *,
        table: str,
        column_name: str,
        equation: str,
    ) -> Dict[str, str]:
        """
        **Transform** *column_name* in-place according to *equation*.

        Steps:
        1. Create a temporary derived column from *equation*.
        2. Delete the original column.
        3. Rename the temporary column back to the original name.
        """
        tmp_name = f"tmp_{column_name}_{uuid.uuid4().hex[:8]}"

        create_res = self._create_derived_column(
            table=table,
            column_name=tmp_name,
            equation=equation,
        )
        delete_res = self._delete_column(table=table, column_name=column_name)
        rename_res = self._rename_column(
            table=table,
            old_name=tmp_name,
            new_name=column_name,
        )
        return {
            "status": "transformed",
            "create_result": str(create_res),
            "delete_result": str(delete_res),
            "rename_result": str(rename_res),
        }

    #  Row-level deletion

    def _delete_rows(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Delete every log **matching *filter*** across the chosen tables.

        Argspec mirrors `_search_knowledge`.
        """
        if tables is None:
            tables = list(self._tables_overview().keys())

        summaries: Dict[str, str] = {}
        for table in tables:
            ctx = self._ctx_for_table(table)
            log_ids = list(
                unify.get_logs(
                    context=ctx,
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    return_ids_only=True,
                ),
            )
            if not log_ids:
                summaries[table] = "no-op"
                continue

            res = unify.delete_logs(
                logs=log_ids,
                context=ctx,
                project=unify.active_project(),
                delete_empty_logs=True,
            )
            summaries[table] = res.get("message", str(res))

        return summaries

    # Row creation / update

    def _add_rows(
        self,
        *,
        table: str,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, str]:
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
        return unify.create_logs(
            context=self._ctx_for_table(table),
            entries=rows,
            batched=True,
        )

    def _update_rows(
        self,
        *,
        table: str,
        updates: Dict[int, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        **Update** existing rows identified by their *unique_id*.

        Parameters
        ----------
        table : str
            Target table.
        updates : dict[int, dict[str, Any]]
            Mapping of unique `row_id` rows to update (might be named `team_id`, `product_id` etc.)
            to the dict of new field values mapping column names to the new overwriting values.
        overwrite : bool, default ``False``
            When *True*, fields **not** mentioned in *entries* are cleared.
        """
        ctx = self._ctx_for_table(table)
        ctx_info = unify.get_context(ctx)
        unique_column_name = ctx_info["unique_column_ids"][0]
        unique_ids = sorted([int(k) for k in updates.keys()])
        log_ids: List[int] = sorted(
            unify.get_logs(
                context=ctx,
                filter=f"{unique_column_name} in {unique_ids}",
                return_ids_only=True,
            ),
        )
        entries = [updates[str(unique_id)] for unique_id in unique_ids]
        res = unify.update_logs(
            logs=log_ids,
            context=ctx,
            entries=entries,
            overwrite=True,
        )
        return res

    # Vector Search Helpers
    def _vectorize_column(
        self,
        table: str,
        source_column: str,
        target_column_name: str,
    ) -> None:
        """
        Ensure that a vector column exists in the given table. If it doesn't exist,
        create it as a derived column from the source column.

        Args:
            table (str): The name of the table to ensure the vector column in.
            source (str): The name of the column to derive the vector column from.
            column (str): The name of the vector column to ensure, MUST be *snake case*.
        """
        context = self._ctx_for_table(table)
        ensure_vector_column(
            context,
            embed_column=target_column_name,
            source_column=source_column,
        )

    def _search(
        self,
        *,
        table: str,
        references: Dict[str, str],
        k: int = 5,
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
        k : int, default 5
            Maximum number of rows to return.

        Returns
        -------
        list[dict[str, Any]]
            Rows sorted by ascending semantic distance (best match first).
        """
        import hashlib as _hashlib

        assert (
            isinstance(references, dict) and len(references) > 0
        ), "references must be a non-empty dict"

        context = self._ctx_for_table(table)

        # Ensure vectors for each source expression and collect (embed_col, ref_text)
        terms: list[tuple[str, str]] = []
        for source_expr, ref_text in references.items():
            embed_col = ensure_vector_for_source(context, source_expr)
            terms.append((embed_col, ref_text))

        # Fast-path: single expression → rank directly by cosine
        if len(terms) == 1:
            embed_col, ref_text = terms[0]
            escaped_ref = ref_text.replace("'", "\\'")
            logs = unify.get_logs(
                context=context,
                sorting={
                    f"cosine({embed_col}, embed('{escaped_ref}', model='{EMBED_MODEL}'))": "ascending",
                },
                limit=k,
                exclude_fields=[
                    fld
                    for fld in unify.get_fields(context=context).keys()
                    if fld.endswith("_emb")
                ],
            )
            return [lg.entries for lg in logs]

        # Multi-expression path: rank by sum of cosine distances
        canonical = "|".join(
            f"{key}=>{references[key]}" for key in sorted(references.keys())
        )
        sum_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
        sum_key = ensure_sum_cosine_column(context, terms, sum_hash)

        logs = unify.get_logs(
            context=context,
            sorting={sum_key: "ascending"},
            limit=k,
            exclude_fields=[
                fld
                for fld in unify.get_fields(context=context).keys()
                if fld.endswith("_emb")
            ],
        )
        return [lg.entries for lg in logs]

    def _search_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        references: Dict[str, str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search on the result of joining two tables, using the same
        mechanism as `_search` but operating within a temporary joined context.

        Parameters
        ----------
        .tables : str | list[str]
            Exactly two tables to join.
        .join_expr : str
            Expression linking aliases in the two tables.
        .select : dict[str, str]
            Projection mapping of fully-qualified input columns to output column
            names present in the joined table.
        .mode : str
            Join kind understood by Unify ("inner", "left", "right", "outer").
        .left_where / right_where : str | None
            Optional pre-join predicates applied to the left/right tables.
        .references : dict[str, str]
            Mapping from a source expression (plain column or derived Unify
            expression) to the reference text. Multiple entries will use a sum of
            cosine distances for ranking.
        .k : int
            Maximum number of rows to return from the joined table.

        Returns
        -------
        list[dict[str, Any]]
            Rows from the joined context sorted by semantic similarity.
        """

        # 1️⃣  Materialize the join into a temporary context
        dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
        dest_ctx = self._create_join(
            dest_table=dest_table,
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
        )

        try:
            # 2️⃣  Ensure vectors for each source expression within the joined context
            assert (
                isinstance(references, dict) and len(references) > 0
            ), "references must be a non-empty dict"
            terms: List[tuple[str, str]] = []
            for source_expr, ref_text in references.items():
                embed_col = ensure_vector_for_source(dest_ctx, source_expr)
                terms.append((embed_col, ref_text))

            # 3️⃣  Rank and fetch like `_search`
            if len(terms) == 1:
                embed_col, ref_text = terms[0]
                escaped_ref = ref_text.replace("'", "\\'")
                logs = unify.get_logs(
                    context=dest_ctx,
                    sorting={
                        f"cosine({embed_col}, embed('{escaped_ref}', model='{EMBED_MODEL}'))": "ascending",
                    },
                    limit=k,
                    exclude_fields=[
                        fld
                        for fld in unify.get_fields(context=dest_ctx).keys()
                        if fld.endswith("_emb")
                    ],
                )
                return [lg.entries for lg in logs]

            # Multi-expression: sum of cosine distances
            canonical = "|".join(
                f"{key}=>{references[key]}" for key in sorted(references.keys())
            )
            import hashlib as _hashlib

            sum_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
            sum_key = ensure_sum_cosine_column(dest_ctx, terms, sum_hash)

            logs = unify.get_logs(
                context=dest_ctx,
                sorting={sum_key: "ascending"},
                limit=k,
                exclude_fields=[
                    fld
                    for fld in unify.get_fields(context=dest_ctx).keys()
                    if fld.endswith("_emb")
                ],
            )
            return [lg.entries for lg in logs]
        finally:
            # 4️⃣  Clean up the temporary context best-effort
            try:
                unify.delete_context(dest_ctx)
            except Exception:
                pass

    def _search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Dict[str, str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the result of chaining multiple joins.

        Parameters
        ----------
        joins : list[dict]
            An ordered list where each element mirrors the kwargs of `_search_join`,
            but chained tables may use the `$prev` placeholder to reference the
            result of the previous join step. Each dict must include at least
            `"tables"`, `"join_expr"`, and `"select"`. Optional keys: `mode`,
            `left_where`, `right_where`.
        references : dict[str, str]
            Mapping from a source expression (plain column or derived Unify
            expression such as concatenations using `str({col})`) to the reference
            text. Multiple expressions will be ranked via the sum of cosine
            distances, mirroring `_search` and `_search_join` semantics.
        k : int, default 5
            Maximum number of rows to return.

        Returns
        -------
        list[dict[str, Any]]
            Rows from the final joined context sorted by semantic similarity.
        """

        if not joins:
            raise ValueError("`joins` must contain at least one join step.")
        if not isinstance(references, dict) or len(references) == 0:
            raise AssertionError("references must be a non-empty dict")

        tmp_prefix = f"_tmp_mjoin_{uuid.uuid4().hex[:6]}"
        tmp_tables: List[str] = []
        previous_table: Optional[str] = None

        for idx, step in enumerate(joins):
            local_step = step.copy()  # do not mutate caller's dict
            raw_tables = local_step.get("tables")
            raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
            if not isinstance(raw_tables, list) or len(raw_tables) != 2:
                raise ValueError(
                    f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
                )

            # Substitute `$prev` placeholder
            step_tables = [
                (previous_table if t in {"$prev", "__prev__", "_"} else t)
                for t in raw_tables
            ]
            if any(t is None for t in step_tables):
                raise ValueError(
                    "Misplaced `$prev` in first join – there is no previous result.",
                )

            # Fix-up join_expr & columns that reference `$prev`
            def _replace_prev(
                s: Optional[Union[str, List[str], Dict[str, str]]],
            ) -> Optional[Union[str, List[str], Dict[str, str]]]:
                if s is None or previous_table is None:
                    return s

                def repl(txt: str) -> str:
                    return (
                        txt.replace("$prev", previous_table)
                        .replace("__prev__", previous_table)
                        .replace("_.", f"{previous_table}.")
                    )

                if isinstance(s, str):
                    return repl(s)
                elif isinstance(s, dict):
                    return {repl(k): v for k, v in s.items()}
                return [repl(c) for c in s]

            join_expr = _replace_prev(local_step.get("join_expr"))
            select = _replace_prev(local_step.get("select"))

            # Destination table for this hop
            is_last = idx == len(joins) - 1
            dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
            tmp_tables.append(dest_table)

            # Materialise the join (no reads yet)
            self._create_join(
                dest_table=dest_table,
                tables=step_tables,
                join_expr=join_expr,  # type: ignore[arg-type]
                select=select,  # type: ignore[arg-type]
                mode=local_step.get("mode", "inner"),
                left_where=local_step.get("left_where"),
                right_where=local_step.get("right_where"),
            )

            previous_table = dest_table

        assert previous_table is not None  # mypy guard

        final_ctx = self._ctx_for_table(previous_table)

        try:
            # Ensure vectors for each source expression within the final joined context
            terms: List[tuple[str, str]] = []
            for source_expr, ref_text in references.items():
                embed_col = ensure_vector_for_source(final_ctx, source_expr)
                terms.append((embed_col, ref_text))

            # Rank and fetch like `_search` / `_search_join`
            if len(terms) == 1:
                embed_col, ref_text = terms[0]
                escaped_ref = ref_text.replace("'", "\\'")
                logs = unify.get_logs(
                    context=final_ctx,
                    sorting={
                        f"cosine({embed_col}, embed('{escaped_ref}', model='{EMBED_MODEL}'))": "ascending",
                    },
                    limit=k,
                    exclude_fields=[
                        fld
                        for fld in unify.get_fields(context=final_ctx).keys()
                        if fld.endswith("_emb")
                    ],
                )
                return [lg.entries for lg in logs]

            # Multi-expression: sum of cosine distances
            canonical = "|".join(
                f"{key}=>{references[key]}" for key in sorted(references.keys())
            )
            import hashlib as _hashlib

            sum_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
            sum_key = ensure_sum_cosine_column(final_ctx, terms, sum_hash)

            logs = unify.get_logs(
                context=final_ctx,
                sorting={sum_key: "ascending"},
                limit=k,
                exclude_fields=[
                    fld
                    for fld in unify.get_fields(context=final_ctx).keys()
                    if fld.endswith("_emb")
                ],
            )
            return [lg.entries for lg in logs]
        finally:
            # Clean up temporary contexts (best-effort)
            try:
                self._delete_tables(tables=tmp_tables)
            except Exception:
                pass

    # Search

    ## private helper

    def _create_join(
        self,
        *,
        dest_table: str,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
    ) -> str:
        """
        Create **one** temporary joined table on the Unify backend and return
        its fully-qualified context.

        This helper is intentionally *side-effect-only*: it **does not** read
        from or delete the created context.  Those concerns are left to the
        public APIs so they can enforce their own life-cycle semantics
        (single-use for `_search_join`, multi-step for `_search_multi_join`).
        """
        # 1️⃣  Resolve & validate the inputs
        if isinstance(tables, str):
            tables = [tables]
        if len(tables) != 2:
            raise ValueError("❌  Exactly TWO tables are required.")

        left_table, right_table = tables
        left_ctx, right_ctx = self._ctx_for_table(left_table), self._ctx_for_table(
            right_table,
        )

        # Optionally rewrite the pre-filters to the fully-qualified contexts
        def _rewrite_filter(expr: Optional[str], table: str, ctx: str) -> Optional[str]:
            return None if expr is None else expr.replace(table, ctx)

        left_where = _rewrite_filter(left_where, left_table, left_ctx)
        right_where = _rewrite_filter(right_where, right_table, right_ctx)

        # Fully-qualify the join expression / selected columns
        join_expr = join_expr.replace(left_table, left_ctx).replace(
            right_table,
            right_ctx,
        )
        select = {
            c.replace(left_table, left_ctx).replace(right_table, right_ctx): v
            for c, v in select.items()
        }

        # 3️⃣  Destination context
        dest_ctx = self._ctx_for_table(dest_table)

        # 4️⃣  Fire the REST request
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/join"
        headers = {
            "Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "project": unify.active_project(),
            "pair_of_args": (
                {
                    "context": left_ctx,
                    **({} if left_where is None else {"filter_expr": left_where}),
                },
                {
                    "context": right_ctx,
                    **({} if right_where is None else {"filter_expr": right_where}),
                },
            ),
            "join_expr": join_expr,
            "mode": mode,
            "new_context": dest_ctx,
            "columns": select,
        }

        resp = requests.request("POST", url, json=payload, headers=headers)
        _handle_exceptions(resp)

        return dest_ctx

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
            Maximum rows per table.
        tables :  str | list[str]
            Subset of tables to scan; ``None`` → all tables.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping ``table_name → [row_dict, …]``.
        """
        if tables is None:
            tables = self._tables_overview()
        elif isinstance(tables, str):
            tables = [tables]
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=self._ctx_for_table(table),
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    exclude_fields=[
                        k
                        for k in unify.get_fields(
                            context=self._ctx_for_table(table),
                        ).keys()
                        if k.endswith("_emb")
                    ],
                )
            ]
        return results

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
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        **Join two tables** server-side and query the derived context.
        Useful for querying data that transcends more than one table.

        Parameters
        ----------
        tables : str | list[str]
            Exactly **two** table names to join.
        join_expr : str | None
            Expression linking aliases in the two tables.
            For example "Departments.id == Employees.derpartment_id",
        select : dict[str, str]
            Column names to include in the resultant joined table, keys being the originals and values being the new names.
            For example: {'Students.id': 'student_id', 'Departments.id': 'department_id'}
        mode : str
            Join kind understood by Unify (``"inner"``, ``"left"``, ``"right"``, ``"outer"``).
        left_filter / right_filter : str | None
            Optional pre-join predicates on the left / right tables.
        result_where : str | None
            Predicate evaluated on the new table *after* the join.
            Can *only* include columns specified as the *values* in the 'select' argument,
            which dictates the columns that are present in the final joined table.
        result_limit, result_offset : int
            Pagination of the post-join rows.

        Returns
        -------
        list[dict[str, Any]]
            Rows from the *temporary* table following the join operation and
            then the `result_where` filtering and `result_limit` and `result_offset` pagination.
            The table is deleted immediately afterwards – this method is therefore **read-only**.

        Notes
        -----
        • *left_where* / *right_where* are applied **before** the join, one
          predicate per input table.
        • *result_where* is evaluated **after** the join on the **columns that
          survived projection** (``select``).
        • If you provide ``select=[]`` make sure every column mentioned in
          *result_where* is also listed there – otherwise you will receive a
          ``ValueError`` telling you which column(s) you forgot to project.
        """

        # ── helper to catch mismatches early ────────────────────────────
        def _qualified_refs(expr: str) -> set[str]:
            import re

            return set(
                m.group(0) for m in re.finditer(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr)
            )

        if result_where:
            missing = _qualified_refs(result_where) - set(select)
            if missing:
                raise ValueError(
                    "❌  `result_where` references column(s) that are not present in "
                    "`select`.  Either add them to `select` *or* move the predicate to "
                    "`left_where` / `right_where` as appropriate.  "
                    f"Missing: {', '.join(sorted(missing))}",
                )

        # 1️⃣  Materialise the join (helper handles validation & REST)
        dest_table = f"_tmp_join_{uuid.uuid4().hex[:8]}"
        dest_ctx = self._create_join(
            dest_table=dest_table,
            tables=tables,
            join_expr=join_expr,
            select=select,
            mode=mode,
            left_where=left_where,
            right_where=right_where,
        )

        # 2️⃣  Read from the derived context
        rows: List[Dict[str, Any]] = [
            log.entries
            for log in unify.get_logs(
                context=dest_ctx,
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
            )
        ]

        # 3️⃣  Clean-up
        try:
            unify.delete_context(dest_ctx)
        except Exception:
            # Best-effort – if it fails the tmp context will age-out later.
            pass

        return rows

    def _filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        **Chain together an arbitrary number of joins in one call**.

        Parameters
        ----------
        joins : list[dict]
            An *ordered* list where **each element mirrors the kwargs of
            `_search_join` except that chained tables use the '$prev' placeholder.
            Every dict *requires* the three arguments: "tables", "join_expr" and "select" as a minimum.

            Example::

                joins = [
                    {
                        "tables": ["Authors", "Books"],
                        "join_expr": "Authors.id == Books.author_id",
                        "select": {"Authors.id": "author_id", "Books.title": "book_title", "Books.id": "book_id"},
                        "mode": "inner",
                    },
                    {
                        "tables": ["$prev", "Reviews"],   # $prev → last result
                        "join_expr": "$prev.book_id == Reviews.book_id",
                        "select": {"$prev.book_title": "book_title", "Reviews.content": "review"}
                    },
                ]

        result_where, result_limit, result_offset
            Standard projection / pagination applied *after* the final join.

        Returns
        -------
        list[dict[str, Any]]
            The resultant data following the search operation on the serial table joins ``[row, …]``.
        """

        if not joins:
            raise ValueError("`joins` must contain at least one join step.")

        tmp_prefix = f"_tmp_mjoin_{uuid.uuid4().hex[:6]}"
        tmp_tables: List[str] = []
        previous_table: Optional[str] = None

        for idx, step in enumerate(joins):
            step = step.copy()  # do not mutate caller's dict
            raw_tables = step.get("tables")
            raw_tables = [raw_tables] if isinstance(raw_tables, str) else raw_tables
            if not isinstance(raw_tables, list) or len(raw_tables) != 2:
                raise ValueError(
                    f"Step {idx} must specify exactly TWO tables – got {raw_tables!r}",
                )

            # Substitute `$prev` placeholder
            step_tables = [
                (previous_table if t in {"$prev", "__prev__", "_"} else t)
                for t in raw_tables
            ]
            if any(t is None for t in step_tables):
                raise ValueError(
                    "Misplaced `$prev` in first join – there is no previous result.",
                )

            # Fix-up join_expr & columns that reference `$prev`
            def _replace_prev(
                s: Optional[Union[str, List[str], Dict[str, str]]],
            ) -> Optional[Union[str, List[str], Dict[str, str]]]:
                if s is None or previous_table is None:
                    return s
                repl = (
                    lambda txt: txt.replace("$prev", previous_table)
                    .replace("__prev__", previous_table)
                    .replace("_.", f"{previous_table}.")
                )
                if isinstance(s, str):
                    return repl(s)
                elif isinstance(s, dict):
                    return {repl(k): v for k, v in s.items()}
                return [repl(c) for c in s]

            join_expr = _replace_prev(step.get("join_expr"))
            select = _replace_prev(step.get("select"))

            # Destination table for this hop
            is_last = idx == len(joins) - 1
            dest_table = f"{tmp_prefix}_final" if is_last else f"{tmp_prefix}_{idx}"
            tmp_tables.append(dest_table)

            # Materialise the join (no reads yet)
            self._create_join(
                dest_table=dest_table,
                tables=step_tables,
                join_expr=join_expr,  # type: ignore[arg-type]
                select=select,  # type: ignore[arg-type]
                mode=step.get("mode", "inner"),
                left_where=step.get("left_where"),
                right_where=step.get("right_where"),
            )

            previous_table = dest_table

        assert previous_table is not None  # mypy guard

        # -------- 4.  Read final result ---------------------------------
        rows: List[Dict[str, Any]] = [
            log.entries
            for log in unify.get_logs(
                context=self._ctx_for_table(previous_table),
                filter=result_where,
                offset=result_offset,
                limit=result_limit,
            )
        ]

        # -------- 5.  Clean-up ------------------------------------------
        try:
            # do not delete the user-requested *persistent* table
            self._delete_tables(tables=tmp_tables)
        except Exception:
            # best-effort — leave garbage collection to Unify if this fails
            pass

        return rows

    # ────────────────────────────────────────────────────────────────────
    # Broader context helper
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace the `{broader_context}` placeholder inside *system* messages
        with a fresh snapshot from `MemoryManager` right before the LLM call."""

        import copy

        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local import to avoid cycles

        patched = copy.deepcopy(msgs)

        try:
            broader_ctx = MemoryManager.get_rolling_activity()
        except Exception:
            broader_ctx = ""

        for m in patched:
            if m.get("role") == "system" and "{broader_context}" in (
                m.get("content") or ""
            ):
                m["content"] = m["content"].replace("{broader_context}", broader_ctx)

        return patched
