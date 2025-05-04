import os
import unify
import requests
import threading
from typing import List, Tuple, Any, Dict, Optional, Union
from knowledge.types import ColumnType
from llm_helpers import tool_use_loop

API_KEY = os.environ["UNIFY_KEY"]


class KnowledgeManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for *adding to*, *updating* and *searching through* all knowledge the assistant has stored in memory.
        """
        super().__init__(daemon=daemon)

        refactor_tools = {
            # Tables
            self._create_table.__name__: self._create_table,
            self._list_tables.__name__: self._list_tables,
            self._rename_table.__name__: self._rename_table,
            self._delete_table.__name__: self._delete_table,
            # Columns
            self._create_empty_column.__name__: self._create_empty_column,
            self._create_derived_column.__name__: self._create_derived_column,
            self._list_columns.__name__: self._list_columns,
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
        }

    # Public #
    # -------#

    # English-Text Command

    def store(self, text: str) -> Any:
        """
        Take in any storage text command, and use the tools available (the *non-skipped* private methods of this class) to store the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information storage request, as a plain-text command.

        Returns:
            bool: Whether the storage request completed successfully.
        """
        from knowledge.sys_msgs import STORE

        client = unify.Unify("o4-mini@openai")
        client.set_system_message(STORE)
        return tool_use_loop(client, text, self._store_tools)

    def retrieve(self, text: str) -> str:
        """
        Take in any retrieval text command, and use the tools available (the *non-skipped* private methods of this class) to retireve the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information retrieval request, as a plain-text command.

        Returns:
            str: The result of the retrieval.
        """
        from knowledge.sys_msgs import RETRIEVE

        client = unify.Unify("o4-mini@openai")
        client.set_system_message(RETRIEVE)
        return tool_use_loop(client, text, self._retrieve_tools)

    # Helpers #
    # --------#

    def _get_columns(self, table):
        proj = unify.active_project()
        ctx = f"Knowledge/{table}"
        url = f"https://api.unify.ai/v0/logs/fields?project={proj}&context={ctx}"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.request("GET", url, headers=headers)
        if not response.ok:
            raise response.json()
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
    ) -> str:
        """
        Create a new table for storing long-term knowledge.

        Args:
            name (str): The name of the table to create.

            description (Optional[str]): A description of the table.

            columns (Optional[Dict[str, ColumnType]]): A dictionary of column names and their types.

        Returns:
            str: The name of the table that was created.
        """
        proj = unify.active_project()
        ctx = f"Knowledge/{name}"
        unify.create_context(ctx, description=description)
        if not columns:
            return
        url = f"https://api.unify.ai/v0/project/{proj}/contexts/{ctx}/columns"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"columns": columns}
        response = requests.request("POST", url, json=json_input, headers=headers)
        if not response.ok:
            raise response.json()
        return response.json()

    def _list_tables(
        self,
        include_columns: bool = False,
    ) -> Union[List[str], List[Dict[str, ColumnType]]]:
        """
        List the tables which are being used to store all knowledge.

        Args:
            include_columns (bool): Whether to include the columns in the returned list.

        Returns:
            Union[List[str], List[Dict[str, ColumnType]]]: A list of table names.
        """
        tables = {
            k.lstrip("Knowledge/"): {"description": v}
            for k, v in unify.get_contexts(prefix="Knowledge/").items()
        }
        if not include_columns:
            return tables
        return {k: {**v, "columns": self._get_columns(k)} for k, v in tables.items()}

    def _rename_table(self, old_name: str, new_name: str) -> None:
        """
        Rename the table.

        Args:
            old_name (str): The old name of the table.

            new_name (str): The new name for the table.
        """
        proj = unify.active_project()
        old_name = f"Knowledge/{old_name}"
        new_name = f"Knowledge/{new_name}"
        url = f"https://api.unify.ai/v0/project/{proj}/contexts/{old_name}/rename"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"name": new_name}
        response = requests.request("PATCH", url, json=json_input, headers=headers)
        if not response.ok:
            raise response.json()
        return response.json()

    def _delete_table(self, table: str) -> None:
        """
        Delete the specified table, and all of its data from the knowledge store.

        Args:
            name (str): The name of the table to delete.
        """
        unify.delete_context(f"Knowledge/{table}")

    # Columns

    def _list_columns(
        self,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        List the columns for each specified table.

        Args:
            tables (List[str]): The tables to list the columns for. Default is all tables.

        Returns:
            List[Tuple[str, str]]: A  dict, where keys are table names and values are lists of tuples, where each tuple contains the column name and type.
        """
        raise NotImplemented

    def _create_empty_column(
        self,
        table: str,
        column_name: str,
        column_type: str,
    ) -> None:
        """
        Adds an empty column to the table, which is initialized with `None` values.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            column_type (str): The type of the column to add.
        """
        proj = unify.active_project()
        ctx = f"Knowledge/{table}"
        url = f"https://api.unify.ai/v0/project/{proj}/contexts/{ctx}/columns"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {"columns": {column_name: column_type}}
        response = requests.request("POST", url, json=json_input, headers=headers)
        if not response.ok:
            raise response.json()
        return response.json()

    def _create_derived_column(
        self,
        table: str,
        column_name: str,
        equation: str,
    ):
        """
        Create a new column in the table, derived from the other columns in the table.

        Args:
            table (str): The name of the table to add the column to.

            column_name (str): The name of the column to add.

            equation (str): The equation to use to derive the column.
        """
        url = "https://api.unify.ai/v0/logs/derived"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        equation = equation.replace("{", "{lg:")
        json_input = {
            "project": unify.active_project(),
            "context": f"Knowledge/{table}",
            "key": column_name,
            "equation": equation,
            "referenced_logs": {"lg": {"context": f"Knowledge/{table}"}},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        return response.json()

    def _delete_column(self, table: str, column_name: str):
        """
        Delete column from the table, and all of the data.

        Args:
            table (str): The name of the table to delete the column from.
        """
        url = "https://api.unify.ai/v0/logs?delete_empty_logs=True"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": unify.active_project(),
            "context": f"Knowledge/{table}",
            "ids_and_fields": [[None, column_name]],
            "source_type": "all",
        }
        response = requests.request("DELETE", url, json=json_input, headers=headers)
        if not response.ok:
            raise response.json()
        return response.json()

    def _rename_column(self, table: str, old_name: str, new_name: str):
        """
        Rename the specified column.

        Args:
            table (str): The name of the table to rename the column in.

            old_name (str): The name of the column to rename.
        """
        proj = unify.active_project()
        ctx = f"Knowledge/{table}"
        url = "https://api.unify.ai/v0/logs/rename_field"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        json_input = {
            "project": proj,
            "context": ctx,
            "old_field_name": old_name,
            "new_field_name": new_name,
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        if not response.ok:
            raise response.json()
        return response.json()

    # Add Data

    def _add_data(self, table: str, data: List[Dict[str, Any]]):
        """
        Add data to the specified table. Will automatically create new columns if any keys are not present in the table already.

        Args:
            table (str): The name of the table to add the data to.

            data (List[Dict[str, Any]]): The data to add to the table.
        """
        return unify.create_logs(
            context=f"Knowledge/{table}",
            entries=data,
        )

    # Search

    def _search(self, query: Optional[str] = None, tables: Optional[List[str]] = None):
        """
        Search the query through all of the tables, and return the results across all tables.
        """
        if tables is None:
            tables = self._list_tables()
        # ToDo: convert to map function
        results = dict()
        for table in tables:
            results[table] = [
                log.entries
                for log in unify.get_logs(
                    context=f"Knowledge/{table}",
                    filter=query,
                )
            ]
        return results
