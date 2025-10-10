from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..singleton_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING


class BaseFunctionManager(ABC, metaclass=SingletonABCMeta):
    """
    Public contract for a function catalogue that stores and retrieves
    user‑supplied Python functions and their metadata.

    Overview
    --------
    Implementations may talk to a real database (e.g. Unify logs), an
    in‑memory mock, or a purely simulated LLM – but they all expose the
    same public methods documented below.

    Data Model
    ----------
    All function records conform to the Pydantic model
    ``unity.function_manager.types.function.Function`` (referred to as
    "Function" in the method docs). Implementations may return either
    instances of this model or JSON‑serialisable dictionaries whose keys
    and value types match the model schema. Fields that are not
    applicable to a particular operation (e.g. ``implementation`` when
    not requested) may be omitted or set to suitable defaults by the
    implementation, but the schema serves as the single source of truth
    for field names and types.
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

        Signature
        ---------
        add_functions(
            *,
            implementations: str | list[str],
            preconditions: dict[str, dict] | None = None,
        ) -> dict[str, str]

        Parameters
        ----------
        implementations : str | list[str]
            One or more full Python function source strings. Each string must
            contain exactly one top‑level ``def`` (or ``async def``) starting at
            column 0. The implementation may call built‑ins and object methods
            but must not call other user‑defined functions in the same batch.
        preconditions : dict[str, dict] | None, default ``None``
            Optional mapping from function name → precondition payload. The
            payload is stored as the ``precondition`` field on the corresponding
            ``Function`` record. The expected shape matches the
            ``Function.precondition`` type (``dict[str, Any] | None``).

        Returns
        -------
        dict[str, str]
            Mapping of function name to status string, e.g.
            ``{"my_func": "added"}`` or ``{"my_func": "error: <message>"}``.

        Notes
        -----
        - Implementations should persist records that conform to the
          ``Function`` model (including ``name``, ``function_id``, ``argspec``,
          ``docstring``, ``implementation``, ``calls``, ``embedding_text`` and
          ``precondition``) and ensure that failures for one function do not
          prevent other valid functions in the same batch from being added.
        """

    @abstractmethod
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of function name to function metadata.

        Signature
        ---------
        list_functions(
            *,
            include_implementations: bool = False,
        ) -> dict[str, dict[str, Any]]

        Parameters
        ----------
        include_implementations : bool, default ``False``
            When ``True``, values include the full source code in the
            ``implementation`` field. When ``False``, implementations may be
            omitted to reduce payload size.

        Returns
        -------
        dict[str, Function]
            Mapping of function name → record conforming to the
            ``Function`` schema. Implementations MAY return actual Pydantic
            ``Function`` instances or JSON‑serialisable dicts matching that
            schema. When ``include_implementations=False``, the
            ``implementation`` field may be omitted.
        """

    @abstractmethod
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the stored precondition for a given function.

        Signature
        ---------
        get_precondition(*, function_name: str) -> dict[str, Any] | None

        Parameters
        ----------
        function_name : str
            The canonical function ``name`` (as stored in the corresponding
            ``Function`` record).

        Returns
        -------
        dict[str, Any] | None
            The ``Function.precondition`` payload if present, otherwise ``None``.
        """

    @abstractmethod
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete a function by its unique identifier.

        Signature
        ---------
        delete_function(
            *,
            function_id: int,
            delete_dependents: bool = True,
        ) -> dict[str, str]

        Parameters
        ----------
        function_id : int
            Identifier of the function to delete (``Function.function_id``).
        delete_dependents : bool, default ``True``
            When ``True``, also remove every function that directly or
            transitively calls the target (recursive cascade).

        Returns
        -------
        dict[str, str]
            Status mapping, typically ``{<function_name>: "deleted"}``.
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
        Filter stored function metadata using a Python‑expression.

        Signature
        ---------
        search_functions(
            *,
            filter: str | None = None,
            offset: int = 0,
            limit: int = 100,
        ) -> list[dict[str, Any]]

        Parameters
        ----------
        filter : str | None, default ``None``
            A boolean expression evaluated per row with fields of the
            ``Function`` model in scope (e.g. ``name``, ``argspec``,
            ``docstring``, ``calls``). When ``None``, returns all rows subject
            to pagination.
        offset : int, default ``0``
            Zero‑based index of the first result to return.
        limit : int, default ``100``
            Maximum number of results to return.

        Returns
        -------
        list[Function]
            A list of records conforming to the ``Function`` schema. An
            implementation may return actual ``Function`` instances or
            JSON‑serialisable dicts matching the model.

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
        Search for functions by semantic similarity to a natural‑language query.

        Signature
        ---------
        search_functions_by_similarity(
            *,
            query: str,
            n: int = 5,
        ) -> list[dict[str, Any]]

        Parameters
        ----------
        query : str
            Natural‑language text describing the desired function(s).
        n : int, default ``5``
            Number of similar results to return.

        Returns
        -------
        list[dict[str, Any]]
            Up to ``n`` results ordered by similarity. Each element SHOULD
            include the fields of the ``Function`` model (as a record or a
            dict matching the schema) and MAY include an additional ``score``
            field (``float``) representing the similarity (lower distance or
            higher similarity depending on implementation).
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseFunctionManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
