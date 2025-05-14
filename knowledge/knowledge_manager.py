import os
import unify
import threading
from typing import List, Any, Dict, Optional, Union
from knowledge.types import ColumnType
from llm_helpers import tool_use_loop



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

    def store(self, text: str, *, return_reasoning_steps: bool = False) -> Any:
        """
        Take in any storage text command, and use the tools available (the *non-skipped* private methods of this class) to store the information, refactoring the table and column schema along the way if needed.

        Args:
            text (str): The information storage request, as a plain-text command.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the storage request.

        Returns:
            bool: Whether the storage request completed successfully.
        """
        from knowledge.sys_msgs import STORE

        client = unify.Unify("o4-mini@openai", cache=True, traced=True)
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
        from knowledge.sys_msgs import RETRIEVE

        client = unify.Unify("o4-mini@openai", cache=True, traced=True)
        client.set_system_message(RETRIEVE)
        ans = tool_use_loop(client, text, self._retrieve_tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Helpers #
    # --------#

    def _get_columns(self, *, table: str) -> Dict[str, str]:
        ret = unify.get_fields(context=f"Knowledge/{table}")
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
        ctx = f"Knowledge/{name}"
        unify.create_context(ctx, description=description)
        if not columns:
            return
        return unify.create_fields(columns, context=ctx)

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
            k.lstrip("Knowledge/"): {"description": v}
            for k, v in unify.get_contexts(prefix="Knowledge/").items()
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
        return unify.rename_context(f"Knowledge/{old_name}", f"Knowledge/{new_name}")

    def _delete_table(self, table: str) -> Dict[str, str]:
        """
        Delete the specified table, and all of its data from the knowledge store.

        Args:
            name (str): The name of the table to delete.

        Returns:
            Dict[str, str]: Message explaining whether the table was deleted or not.
        """
        return unify.delete_context(f"Knowledge/{table}")

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
        return unify.create_fields({column_name: column_type}, context=f"Knowledge/{table}")

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
        equation = equation.replace("{", "{lg:")
        context = f"Knowledge/{table}"
        return unify.update_derived_log(
            target=None,  # TODO required argument
            key=column_name,
            equation=equation,
            referenced_logs={"lg": {"context": context}},
            context=context,
        )

    def _delete_column(self, *, table: str, column_name: str) -> Dict[str, str]:
        """
        Delete column from the table, and all of the data.

        Args:
            table (str): The name of the table to delete the column from.

            column_name (str): The name of the column to delete.

        Returns:
            Dict[str, str]: Message explaining whether the column was deleted or not.
        """
        return unify.delete_fields([column_name], context=f"Knowledge/{table}")

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
        ctx = f"Knowledge/{table}"
        return unify.rename_field(ctx, old_name, new_name)

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
            context=f"Knowledge/{table}",
            entries=data,
            batched=True,  # NOTE: async logger can mess with the order of the data
        )

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
                    context=f"Knowledge/{table}",
                    filter=filter,
                    offset=offset,
                    limit=limit,
                )
            ]
        return results
