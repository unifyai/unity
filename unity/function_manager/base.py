from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager

# Supported function languages
FunctionLanguage = Literal["python", "bash", "zsh", "sh", "powershell"]

# State modes for Python function execution
StateMode = Literal["stateful", "read_only", "stateless"]


class BaseFunctionManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract for a function catalogue that stores and retrieves
    user‑supplied functions and their metadata.

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

    _as_caller_description: str = "the FunctionManager, managing user-defined functions"

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        language: FunctionLanguage = "python",
        preconditions: Optional[Dict[str, Dict]] = None,
        verify: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, str]:
        """
        Validate, compile and persist one or more function implementations.

        Signature
        ---------
        add_functions(
            *,
            implementations: str | list[str],
            language: Literal["python", "bash", "zsh", "sh", "powershell"] = "python",
            preconditions: dict[str, dict] | None = None,
            verify: dict[str, bool] | None = None,
        ) -> dict[str, str]

        Parameters
        ----------
        implementations : str | list[str]
            One or more function source strings. For Python, each string must
            contain exactly one top‑level ``def`` (or ``async def``) starting at
            column 0. For shell languages, the script should include metadata
            comments at the top (see Notes).
        language : Literal["python", "bash", "zsh", "sh", "powershell"], default ``"python"``
            The language/interpreter for the function(s). All implementations
            in a single call must be the same language.
        preconditions : dict[str, dict] | None, default ``None``
            Optional mapping from function name → precondition payload. The
            payload is stored as the ``precondition`` field on the corresponding
            ``Function`` record. The expected shape matches the
            ``Function.precondition`` type (``dict[str, Any] | None``).
        verify : dict[str, bool] | None, default ``None``
            Optional mapping from function name → verification requirement.
            If a function name is present and mapped to ``True`` (default) or ``False``,
            it sets the ``verify`` field on the ``Function`` record.

        Returns
        -------
        dict[str, str]
            Mapping of function name to status string, e.g.
            ``{"my_func": "added"}`` or ``{"my_func": "error: <message>"}``.

        Notes
        -----
        - For Python functions: implementations are validated via AST parsing
          and executed to extract signatures and docstrings automatically.
        - For shell functions (bash, zsh, sh, powershell): metadata is extracted
          from comments at the top of the script using these patterns::

              # @name: my_function
              # @args: (input_file output_file --verbose)
              # @description: Brief description of what the function does

          The ``@name`` comment is required for shell functions. ``@args`` and
          ``@description`` are optional.
        - Implementations should persist records that conform to the
          ``Function`` model and ensure that failures for one function do not
          prevent other valid functions in the same batch from being added.
        """

    @abstractmethod
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of function name to function metadata.

        Signature
        ---------
        list_functions(
            *,
            include_implementations: bool = False,
            return_callable: bool = False,
            namespace: dict[str, Any] | None = None,
            also_return_metadata: bool = False,
        ) -> dict[str, dict[str, Any]] | dict[str, Callable[..., Any]] | dict[str, Any]

        Parameters
        ----------
        include_implementations : bool, default ``False``
            When ``True``, values include the full source code in the
            ``implementation`` field. When ``False``, implementations may be
            omitted to reduce payload size.
        return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``namespace``.
        namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``return_callable=True``. Required when ``return_callable=True``.
        also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": <...>, "metadata": <...>}``.

        Returns
        -------
        dict[str, Function] | dict[str, Callable[..., Any]] | dict[str, Any]
            - When ``return_callable=False``: mapping of function name → record
              conforming to the ``Function`` schema (as dicts or Function objects).
              When ``include_implementations=False``, the ``implementation`` field
              may be omitted.
            - When ``return_callable=True``: mapping of function name → callable.
              Callables MAY be in-process functions or proxy callables for functions
              that must execute in an isolated virtual environment (implementation‑defined).
            - When ``also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding mappings.

        Raises
        ------
        ValueError
            If ``return_callable=True`` but ``namespace`` is ``None``.
        ValueError
            If ``also_return_metadata=True`` but ``return_callable`` is ``False``.
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
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
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
            include_implementations: bool = True,
            return_callable: bool = False,
            namespace: dict[str, Any] | None = None,
            also_return_metadata: bool = False,
        ) -> list[dict[str, Any]] | list[Callable[..., Any]] | dict[str, Any]

        Parameters
        ----------
        filter : str | None, default ``None``
            A boolean expression evaluated per row with fields of the
            ``Function`` model in scope (e.g. ``name``, ``argspec``,
            ``docstring``, ``depends_on``). When ``None``, returns all rows subject
            to pagination.
        offset : int, default ``0``
            Zero‑based index of the first result to return.
        limit : int, default ``100``
            Maximum number of results to return. Must be <= 1000.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``namespace``.
        namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``return_callable=True``. Required when ``return_callable=True``.
        also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[Function] | list[Callable[..., Any]] | dict[str, Any]
            - When ``return_callable=False``: list of records conforming to the
              ``Function`` schema (as dicts or Function objects). When
              ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``return_callable=True`` but ``namespace`` is ``None``.
        ValueError
            If ``also_return_metadata=True`` but ``return_callable`` is ``False``.

        Examples
        --------
        >>> mgr.search_functions(filter="'price' in docstring and 'sum' in depends_on")
        >>> mgr.search_functions(filter="name.startswith('get_')")
        """

    @abstractmethod
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
        include_implementations: bool = True,
        return_callable: bool = False,
        namespace: Optional[Dict[str, Any]] = None,
        also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity to a natural‑language query.

        Signature
        ---------
        search_functions_by_similarity(
            *,
            query: str,
            n: int = 5,
            include_implementations: bool = True,
            return_callable: bool = False,
            namespace: dict[str, Any] | None = None,
            also_return_metadata: bool = False,
        ) -> list[dict[str, Any]] | list[Callable[..., Any]] | dict[str, Any]

        Parameters
        ----------
        query : str
            Natural‑language text describing the desired function(s).
        n : int, default ``5``
            Number of similar results to return.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``namespace``.
        namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``return_callable=True``. Required when ``return_callable=True``.
        also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[dict[str, Any]] | list[Callable[..., Any]] | dict[str, Any]
            - When ``return_callable=False``: up to ``n`` results ordered by similarity.
              Each element SHOULD include the fields of the ``Function`` model and MAY
              include an additional ``score`` field (``float``) representing similarity.
              When ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``return_callable=True`` but ``namespace`` is ``None``.
        ValueError
            If ``also_return_metadata=True`` but ``return_callable`` is ``False``.
        """

    @abstractmethod
    async def execute_function(
        self,
        *,
        function_name: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        target_venv_id: Optional[int] = ...,
        state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
        session_id: int = 0,
        venv_pool: Optional[Any] = None,
        shell_pool: Optional[Any] = None,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a stored function by name with optional venv and state mode overrides.

        Signature
        ---------
        execute_function(
            *,
            function_name: str,
            call_kwargs: dict[str, Any] | None = None,
            target_venv_id: int | None = USE_FUNCTION_DEFAULT,
            state_mode: Literal["stateful", "read_only", "stateless"] = "stateless",
            session_id: int = 0,
            venv_pool: VenvPool | None = None,
            shell_pool: ShellPool | None = None,
            primitives: Any | None = None,
            computer_primitives: Any | None = None,
        ) -> dict[str, Any]

        Parameters
        ----------
        function_name : str
            Name of the function to execute (must exist in the function table).
        call_kwargs : dict[str, Any] | None, default ``None``
            Keyword arguments to pass to the function. For Python functions, these
            are passed as keyword arguments. For shell functions, they may be converted
            to positional arguments or environment variables depending on the argspec.
        target_venv_id : int | None, default ``USE_FUNCTION_DEFAULT``
            Override the execution environment (Python functions only):
            - ``USE_FUNCTION_DEFAULT`` (``...``): Use the function's stored ``venv_id``
              from the function table. This is the default behavior.
            - ``None``: Execute in the default Python environment (no custom venv).
            - ``int``: Execute in this specific venv_id, regardless of what's
              stored in the function table.

            This allows running simple/compatible functions in a different venv
            than they were originally associated with. The caller is responsible
            for ensuring the target venv has the required packages.
            Ignored for shell functions.
        state_mode : Literal["stateful", "read_only", "stateless"], default ``"stateless"``
            Controls how global state is handled during execution:
            - ``"stateless"``: Executes with fresh globals/no inherited state.
              Every execution starts with a clean environment. This is the default
              for backward compatibility and is useful for pure functions that should
              not depend on or affect session state.
            - ``"stateful"``: Uses a persistent globals dict (in-process) or subprocess
              connection (venv). Variables and state from previous executions persist.
              Enables Jupyter-notebook-style incremental development. Requires
              ``venv_pool`` for venv functions, ``shell_pool`` for shell functions.
              For in-process Python functions (no venv), state is stored internally.
            - ``"read_only"``: Reads the current state from the persistent session
              but executes in a fresh environment. Changes are not persisted.
              Useful for "what-if" exploration. Requires the appropriate pool for
              venv/shell functions.

            All three modes are supported for both in-process (no venv) and
            subprocess (venv) Python function execution.
        session_id : int, default ``0``
            The session ID within the execution environment. Multiple sessions allow
            independent stateful execution contexts. Each session has its own process
            and state, enabling concurrent "notebook panes" with isolated state.
            Only applies to ``state_mode="stateful"`` or ``state_mode="read_only"``.
        venv_pool : VenvPool | None, default ``None``
            The VenvPool instance for stateful Python execution. Required when
            ``state_mode="stateful"`` or ``state_mode="read_only"`` and the function
            is Python with a venv. If not provided for these modes, an error is raised.
        shell_pool : ShellPool | None, default ``None``
            The ShellPool instance for stateful shell execution. Required when
            ``state_mode="stateful"`` or ``state_mode="read_only"`` and the function
            is a shell script. If not provided for these modes, an error is raised.
        primitives : Any | None, default ``None``
            The Primitives instance for RPC access to state managers.
        computer_primitives : Any | None, default ``None``
            The ComputerPrimitives instance for browser/desktop RPC access.

        Returns
        -------
        dict[str, Any]
            Execution result with keys:
            - ``result``: The return value (Python) or exit code (shell).
            - ``error``: Error message if execution failed, ``None`` otherwise.
            - ``stdout``: Captured stdout from the function.
            - ``stderr``: Captured stderr from the function.

        Raises
        ------
        ValueError
            If the function does not exist or has no implementation.
        ValueError
            If state_mode requires a pool but none is provided.

        Examples
        --------
        >>> # Execute Python function statefully
        >>> result = await fm.execute_function(
        ...     function_name="my_func",
        ...     call_kwargs={"x": 1},
        ...     state_mode="stateful",
        ...     venv_pool=venv_pool,
        ... )

        >>> # Execute shell function statefully
        >>> result = await fm.execute_function(
        ...     function_name="my_shell_func",
        ...     state_mode="stateful",
        ...     shell_pool=shell_pool,
        ... )

        >>> # Execute statelessly - fresh environment every time
        >>> result = await fm.execute_function(
        ...     function_name="pure_func",
        ...     state_mode="stateless",
        ... )
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Sentinel for "use the function's default venv_id"
USE_FUNCTION_DEFAULT = ...

# Attach centralised docstring
BaseFunctionManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
