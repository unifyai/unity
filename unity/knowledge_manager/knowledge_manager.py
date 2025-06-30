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

API_KEY = os.environ["UNIFY_KEY"]


class KnowledgeManager(BaseKnowledgeManager):
    def __init__(self) -> None:
        """
        KnowledgeManager now **directly manipulates** the root-level
        ``Contacts`` table instead of calling the public ContactManager API.
        """

        self._refactor_tools = methods_to_tool_dict(
            # Search
            self._search,
            self._nearest,
            # Tables
            self._tables_overview,
            self._create_table,
            self._rename_table,
            self._delete_table,
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
                self._search,
                self._nearest,
            ),
        }

        self._update_tools = {
            **self._refactor_tools,
            **methods_to_tool_dict(
                self._add_rows,
                include_class_name=False,
            ),
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a KnowledgeManager."
        self._ctx = f"{read_ctx}/Knowledge" if read_ctx else "Knowledge"
        self._contacts_ctx = f"{read_ctx}/Contacts" if read_ctx else "Contacts"

    # Helpers #
    # --------#

    def _ctx_for_table(self, table: str) -> str:
        """Return the correct Unify context for *table*."""
        return self._contacts_ctx if table == "Contacts" else f"{self._ctx}/{table}"

    def _look_first_tool_policy(self, step: int, tls: Dict[str, Callable]):
        if step < 1:
            return "required", methods_to_tool_dict(
                self._search,
                self._nearest,
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
    ) -> "SteerableToolHandle":

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
        client.set_system_message(
            build_refactor_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
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
    ) -> "SteerableToolHandle":

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
        client.set_system_message(
            build_update_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
            ),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=self._look_first_tool_policy,
        )

        # ── 3.  Optionally wrap .result() to expose reasoning  ────────────
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
    ) -> "SteerableToolHandle":
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
        client.set_system_message(
            build_ask_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        # ── 3.  Optionally wrap .result() to expose reasoning  ────────────
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        # Expose root-level Contacts
        if self._contacts_ctx in unify.get_contexts():
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"name": new_name}
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _delete_table(self, *, table: str) -> Dict[str, str]:
        """
        **Drop** an entire table *and* all its rows.

        Parameters
        ----------
        table : str
            Target table name.

        Returns
        -------
        dict[str, str]
            Confirmation / error from the backend.
        """
        return unify.delete_context(self._ctx_for_table(table))

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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        headers = {"Authorization": f"Bearer {API_KEY}"}
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
        unique_column_name = ctx_info["unique_column_ids"]
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

    def _nearest(
        self,
        *,
        tables: Optional[Union[str, List[str]]],
        source: str,
        text: str,
        k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform a **semantic nearest-neighbour search** over one or more
        tables using cosine similarity in embedding space.

        Parameters
        ----------
        tables : str | list[str]
            Candidate tables (each must contain *source* column); ``None`` → all tables.
        source : str
            Text column to embed (an auxiliary ``_<source>_emb`` column is
            auto-created if missing). MUST be *snake case*.
        text : str
            Query text to embed and compare against.
        k : int, default ``5``
            Number of nearest rows to return *per table*.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping ``table_name → [row, …]`` sorted by ascending distance.
        """
        # ToDo: convert to map function
        if tables is None:
            tables = self._tables_overview()
        elif isinstance(tables, str):
            tables = [tables]
        results = dict()
        for table in tables:
            context = self._ctx_for_table(table)
            column_emb = f"_{source}_emb"
            self._vectorize_column(table, source, column_emb)
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=context,
                    sorting={
                        f"cosine({column_emb}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
                    },
                    limit=k,
                    exclude_fields=[
                        k
                        for k in unify.get_fields(context=context).keys()
                        if k.endswith("_emb")
                    ],
                )
            ]
        return results

    # Search

    def _search(
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
