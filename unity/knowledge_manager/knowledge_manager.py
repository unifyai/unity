import os
import asyncio
import unify
import functools
import requests
from typing import Any, Dict, List, Optional

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
from ..events.event_bus import EventBus
from .base import BaseKnowledgeManager
from ..contact_manager.contact_manager import BaseContactManager, ContactManager
from .prompt_builders import build_store_prompt, build_retrieve_prompt

API_KEY = os.environ["UNIFY_KEY"]


class KnowledgeManager(BaseKnowledgeManager):

    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        traced: bool = True,
    ) -> None:
        """
        Responsible for *adding to*, *updating* and *searching through* all knowledge the assistant has stored in memory.
        """
        if contact_manager is None:
            contact_manager = ContactManager(EventBus())

        refactor_tools = methods_to_tool_dict(
            # Tables
            self._create_table,
            self._list_tables,
            self._rename_table,
            self._delete_table,
            # Columns
            self._create_empty_column,
            self._create_derived_column,
            self._rename_column,
            self._delete_column,
            include_class_name=False,
        )

        cm_ask = methods_to_tool_dict(contact_manager.ask, include_class_name=True)
        cm_update = methods_to_tool_dict(
            contact_manager.update,
            include_class_name=True,
        )

        self._retrieve_tools = {
            **cm_ask,
            **refactor_tools,
            **methods_to_tool_dict(
                self._search_knowledge,
                self._nearest_knowledge,
                include_class_name=False,
            ),
        }

        self._store_tools = {
            **cm_ask,
            **cm_update,
            **refactor_tools,
            **methods_to_tool_dict(
                self._add_data,
                include_class_name=False,
            ),
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a KnowledgeManager."
        self._ctx = f"{read_ctx}/Knowledge" if read_ctx else "Knowledge"

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Command

    @functools.wraps(BaseKnowledgeManager.store, updated=())
    def store(
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
        tools = dict(self._store_tools)

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
        table_schemas_json = json.dumps(self._list_tables(), indent=4)
        client.set_system_message(
            build_store_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
            ),
        )

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        # ── 3.  Optionally wrap .result() to expose reasoning  ────────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    @functools.wraps(BaseKnowledgeManager.retrieve, updated=())
    def retrieve(
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
        tools = dict(self._retrieve_tools)

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
        table_schemas_json = json.dumps(self._list_tables(), indent=4)
        client.set_system_message(
            build_retrieve_prompt(
                tools=tools,
                table_schemas_json=table_schemas_json,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
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
        ctx = f"{self._ctx}/{table}"
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
            :pyfunc:`_create_empty_column`.

        Returns
        -------
        dict[str, str]
            Backend response describing success or failure (driver specific).
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{name}"
        unify.create_context(ctx, description=description)
        if not columns:
            return
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"project": proj, "context": ctx, "fields": columns}
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _list_tables(
        self,
        *,
        include_columns: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Enumerate **all** tables managed by this instance.

        Parameters
        ----------
        include_columns : bool, default ``True``
            When *True* each table entry also contains a
            ``"columns": {name: type}`` mapping.

        Returns
        -------
        dict[str, dict]
            Mapping ``table_name → {"description": str, "columns": {...}}``.
            If *include_columns* is *False* the ``"columns"`` key is omitted.
        """
        tables = {
            k[len(f"{self._ctx}/") :]: {"description": v}
            for k, v in unify.get_contexts(prefix=f"{self._ctx}/").items()
        }
        if not include_columns:
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
        return unify.delete_context(f"{self._ctx}/{table}")

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
            New column identifier.
        column_type : ColumnType | str
            Logical type, e.g. ``"str"``, ``"float"``, ``"datetime"``.

        Returns
        -------
        dict[str, str]
            Backend response.
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{table}"
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
            Name of the new derived column.
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
            "context": f"{self._ctx}/{table}",
            "key": column_name,
            "equation": equation,
            "referenced_logs": {"lg": {"context": f"{self._ctx}/{table}"}},
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
            Column to drop.

        Returns
        -------
        dict[str, str]
            Backend confirmation or error.
        """
        url = f"{os.environ['UNIFY_BASE_URL']}/logs?delete_empty_logs=True"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": unify.active_project(),
            "context": f"{self._ctx}/{table}",
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
            Existing column name.
        new_name : str
            Desired new name.

        Returns
        -------
        dict[str, str]
            Backend response.
        """
        proj = unify.active_project()
        ctx = f"{self._ctx}/{table}"
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

    # Add Data

    def _add_data(
        self,
        *,
        table: str,
        data: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        **Insert** one or many rows into *table*.

        Missing columns are auto-created (type inferred via JSON schema
        rules) before the insert.

        Parameters
        ----------
        table : str
            Destination table.
        data : list[dict[str, Any]]
            Sequence of row dictionaries.

        Returns
        -------
        dict[str, str]
            Backend confirmation.
        """
        return unify.create_logs(
            context=f"{self._ctx}/{table}",
            entries=data,
            batched=True,  # NOTE: async logger can mess with the order of the data
        )

    # Vector Search Helpers
    def _ensure_table_vector(self, table: str, column: str, source: str) -> None:
        """
        Ensure that a vector column exists in the given table. If it doesn't exist,
        create it as a derived column from the source column.

        Args:
            table (str): The name of the table to ensure the vector column in.
            column (str): The name of the vector column to ensure.
            source (str): The name of the column to derive the vector column from.
        """
        context = f"{self._ctx}/{table}"
        ensure_vector_column(context, embed_column=column, source_column=source)

    def _nearest_knowledge(
        self,
        *,
        tables: List[str],
        source: str,
        text: str,
        k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform a **semantic nearest-neighbour search** over one or more
        tables using cosine similarity in embedding space.

        Parameters
        ----------
        tables : list[str]
            Candidate tables (each must contain *source* column).
        source : str
            Text column to embed (an auxiliary ``<source>_vec`` column is
            auto-created if missing).
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
        results = dict()
        for table in tables:
            context = f"{self._ctx}/{table}"
            column = f"{source}_vec"
            self._ensure_table_vector(table, column, source)
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=context,
                    sorting={
                        f"cosine({column}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
                    },
                    limit=k,
                )
            ]
        return results

    # Search

    def _search_knowledge(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
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
        tables : list[str] | None
            Subset of tables to scan; ``None`` → all tables.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping ``table_name → [row_dict, …]``.
        """
        if tables is None:
            tables = self._list_tables()
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=f"{self._ctx}/{table}",
                    filter=filter,
                    offset=offset,
                    limit=limit,
                )
            ]
        return results
