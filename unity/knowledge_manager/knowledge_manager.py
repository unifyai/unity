import os
import unify
import requests
from typing import Any, Dict, List, Optional, Union

import requests
import unify

from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..helpers import _handle_exceptions
from .types import ColumnType
from ..common.llm_helpers import tool_use_loop
from ..helpers import _handle_exceptions

API_KEY = os.environ["UNIFY_KEY"]


class KnowledgeManager:

    def __init__(self, *, traced: bool = True) -> None:
        """
        Responsible for *adding to*, *updating* and *searching through* all knowledge the assistant has stored in memory.
        """

        refactor_tools = {
            # Tables
            self._create_table.__name__: self._create_table,
            self._list_tables.__name__: self._list_tables,
            self._rename_table.__name__: self._rename_table,
            self._delete_table.__name__: self._delete_table,
            # Columns
            self._create_empty_column.__name__: self._create_empty_column,
            self._create_derived_column.__name__: self._create_derived_column,
            self._rename_column.__name__: self._rename_column,
            self._delete_column.__name__: self._delete_column,
        }

        self._store_tools = {
            **refactor_tools,
            self._add_data.__name__: self._add_data,
        }

        self._retrieve_tools = {
            **refactor_tools,
            self._search.__name__: self._search,
            self._nearest.__name__: self._nearest,
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert read_ctx == write_ctx, "read and write contexts must be the same when instantiating a KnowledgeManager."
        self._ctx = f"{read_ctx}/Knowledge" if read_ctx else "Knowledge"

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Command

    def store(self, text: str, *, return_reasoning_steps: bool = False) -> Any:
        """
        Take in any storage text command, and use the tools available (the *non-skipped* private methods of this class) to store the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information storage request, as a plain-text command.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the storage request.

        Returns:
            bool: Whether the storage request completed successfully.
        """
        from unity.knowledge_manager.sys_msgs import STORE

        client = unify.Unify("o4-mini@openai", cache=True)
        client.set_system_message(STORE)
        ans = tool_use_loop(client, text, self._store_tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    def retrieve(self, text: str, *, return_reasoning_steps: bool = False) -> str:
        """
        Take in any retrieval text command, and use the tools available (the *non-skipped* private methods of this class) to retireve the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information retrieval request, as a plain-text command.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the retrieval request.

        Returns:
            str: The result of the retrieval.
        """
        from unity.knowledge_manager.sys_msgs import RETRIEVE

        client = unify.Unify("o4-mini@openai", cache=True)
        client.set_system_message(RETRIEVE)
        ans = tool_use_loop(client, text, self._retrieve_tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

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
        name: str,
        *,
        description: Optional[str] = None,
        columns: Optional[Dict[str, ColumnType]] = None,
    ) -> Dict[str, str]:
        """
        Create a new table for storing long-term knowledge.

        Args:
            name (str): The name of the table to create.

            description (Optional[str]): A description of the table and the main purpose.

            columns (Optional[Dict[str, ColumnType]]): A dictionary of column names and their types.

        Returns:
            Dict[str, str]: Message explaining whether the table was created or not.
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
        include_columns: bool = False,
    ) -> Union[List[str], List[Dict[str, ColumnType]]]:
        """
        List the tables which are being used to store all knowledge.

        Args:
            include_columns (bool): Whether to include the columns and their types for each table in the returned list.

        Returns:
            List[Dict[str, Dict[str, Union[str, ColumnType]]]]: Table names and their descriptions, and optionally also column names and types.
        """
        tables = {
            k[len(f"{self._ctx}/"):]: {"description": v}
            for k, v in unify.get_contexts(prefix=f"{self._ctx}/").items()
        }
        if not include_columns:
            return tables
        return {
            k: {**v, "columns": self._get_columns(table=k)} for k, v in tables.items()
        }

    def _rename_table(self, *, old_name: str, new_name: str) -> Dict[str, str]:
        """
        Rename the table.

        Args:
            old_name (str): The old name of the table.

            new_name (str): The new name for the table.

        Returns:
            Dict[str, str]: Message explaining whether the table was renamed or not.
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

    def _delete_table(self, table: str) -> Dict[str, str]:
        """
        Delete the specified table, and all of its data from the knowledge store.

        Args:
            name (str): The name of the table to delete.

        Returns:
            Dict[str, str]: Message explaining whether the table was deleted or not.
        """
        return unify.delete_context(f"{self._ctx}/{table}")

    # Columns

    def _create_empty_column(
        self,
        *,
        table: str,
        column_name: str,
        column_type: str,
    ) -> Dict[str, str]:
        """
        Adds an empty column to the table, which is initialized with `None` values.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            column_type (str): The type of the column to add.

        Returns:
            Dict[str, str]: Message explaining whether the column was created or not.
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
        Create a new column in the table, derived from the other columns in the table.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            equation (str): The equation to use to derive the column. This is arbitrary Python code, with column names expressed as standard variables. For example, if a table includes two float columns `x` and `y`, then an equation of "(x**2 + y**2)**0.5" would be a valid, computing the length. Indexing is also supported `x[0]` for valid types `dict`, `list`, `str` etc., as is `len(x)`, casting to str via `str(x)` etc. The expression just needs to be valid Python with the column names as variables.

        Returns:
            Dict[str, str]: Message explaining whether the column was created or not.
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

    def _delete_column(self, *, table: str, column_name: str) -> Dict[str, str]:
        """
        Delete column from the table, and all of the data.

        Args:
            table (str): The name of the table to delete the column from.

            column_name (str): The name of the column to delete.

        Returns:
            Dict[str, str]: Message explaining whether the column was deleted or not.
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
        Rename the specified column.

        Args:
            table (str): The name of the table to rename the column in.

            old_name (str): The name of the column to rename.

            new_name (str): The new name for the column.

        Returns:
            Dict[str, str]: Message explaining whether the column was renamed or not.
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
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    # Add Data

    def _add_data(self, *, table: str, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Add data to the specified table. Will automatically create new columns if any keys are not present in the table already.

        Args:
            table (str): The name of the table to add the data to.

            data (List[Dict[str, Any]]): The data to add to the table.

        Returns:
            Dict[str, str]: Message explaining whether the data was added or not.
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

    def _nearest(
        self,
        *,
        tables: List[str],
        column: str,
        source: str,
        text: str,
        k: int = 5,
    ) -> List[unify.Log]:
        """
        Find the k nearest entries in the table to the given text using vector embeddings.

        Args:
            table (str): The name of the table to search in.
            column (str): The name of the vector column to use for similarity search.
            source (str): The name of the column to derive the vector column from.
            text (str): The query text to find similar entries to.
            k (int): The number of results to return.

        Returns:
            List[unify.Log]: The k nearest log entries to the query text.
        """
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            context = f"{self._ctx}/{table}"
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

    def _search(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply the filter to all of the specified tables, and return the results following the filter.

        Args:
            filter (Optional[str]): Arbitrary Python logical expressions which evaluate to `bool`, with column names expressed as standard variables. For example, if a table includes two integer columns `x` and `y`, then a filter expression of "x > 3 and y < 2" would be a valid. Indexing is also supported `x[0]` for valid types `dict`, `list`, `str` etc., as is `len(x)`, casting to str via `str(x)` etc. The expression just needs to be valid Python with the column names as variables.

            offset (int): The offset to start the search from, in the paginated result.

            limit (int): The number of rows to return, in the paginated result.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are table names and values are lists, where each item in the list is a dict representing a row in the table.
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
