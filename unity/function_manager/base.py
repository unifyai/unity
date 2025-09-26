from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..singleton_registry import SingletonABCMeta


class BaseFunctionManager(ABC, metaclass=SingletonABCMeta):
    """
    Public contract for a function catalogue that stores and retrieves
    user-supplied Python functions and their metadata.

    Implementations may use a real backing store (e.g. Unify logs), an
    in-memory mock, or a simulated LLM – but they all expose the same
    public methods documented below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """
        Validate, compile and persist one or more function implementations.

        Returns
        -------
        Dict[str, str]
            Mapping of function name to status string, e.g.
            ``{"my_func": "added"}`` or ``{"my_func": "error: <msg>"}``.
        """

    @abstractmethod
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary keyed by function name.

        Each value contains:
        - ``function_id`` – unique identifier for the function
        - ``argspec`` – full signature, e.g. ``(x: int, y: int) -> int``
        - ``docstring`` – cleaned docstring or empty string
        - ``implementation`` – full source code (only when
          ``include_implementations=True``)
        """

    @abstractmethod
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the stored precondition for a given function.

        Parameters
        ----------
        function_name : str
            The name of the function.

        Returns
        -------
        dict | None
            The precondition dictionary or ``None`` if not found.
        """

    @abstractmethod
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete a function by id.

        When ``delete_dependents`` is ``True`` the operation must also remove
        every function that calls the target (recursive cascade).
        """

    @abstractmethod
    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Flexible, Python-expression filtering over stored function metadata
        (e.g. ``name``, ``argspec``, ``docstring``, ``calls``).

        Examples
        --------
        >>> mgr.search_functions(filter="'price' in docstring and 'sum' in calls")
        >>> mgr.search_functions(filter="name.startswith('get_')")
        """

    @abstractmethod
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity.

        Parameters
        ----------
        query : str
            The natural language query to search for.
        n : int, default 5
            The number of similar functions to return.

        Returns
        -------
        list[dict]
            A list containing up to ``n`` functions ordered by similarity.
        """
