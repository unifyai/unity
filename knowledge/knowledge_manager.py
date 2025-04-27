import threading
from typing import List, Tuple, Any, Dict, Optional


class KnowledgeManager(threading.Thread):

    def __init__(self):
        """
        Responsible for *adding to*, *updating* and *searching through* all knowledge the assistant has stored in memory.
        """
        raise NotImplemented

    # Public #
    # -------#

    # English-Text Command

    def perform_action(text: str) -> Any:
        """
        Take in any text command, and use the tools available (the *non-skipped* private methods of this class) to perform the action.

        Args:
            text (str): The text command to perform.

        Returns:
            Any: The result of the action.
        """
        raise NotImplemented

    # Private #
    # --------#

    def _list_tables() -> List[str]:
        """
        List the tables which are currently being used to store all agent knowledge.
        """
        raise NotImplemented

    def _list_columns(
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

    def _rename_table():
        pass

    def _rename_column():
        pass
