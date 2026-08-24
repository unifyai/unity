from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager

# Supported function languages
FunctionLanguage = Literal["python", "bash", "zsh", "sh", "powershell"]

# State modes for Python function execution
StateMode = Literal["stateful", "read_only", "stateless"]


class BaseFunctionManager(BaseStateManager):
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
    ``unify.function_manager.types.function.Function`` (referred to as
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
        contracts: Optional[Dict[str, Dict[str, Any]]] = None,
        fixtures: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
        venv_id: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Validate, compile and persist one or more function implementations.

        Parameters
        ----------
        implementations : str | list[str]
            Function source strings. Python: exactly one top-level ``def`` /
            ``async def`` per string, starting at column 0; signatures and
            docstrings are extracted automatically. Shell (bash, zsh, sh,
            powershell): metadata comes from leading comments — ``# @name:``
            (required), ``# @args:`` and ``# @description:`` (optional).
        language : default ``"python"``
            Language/interpreter; one language per call.
        preconditions : dict[str, dict] | None
            Mapping of function name → precondition payload, stored as the
            record's ``precondition`` field.
        contracts : dict[str, dict] | None
            Mapping of function name → ``{"postconditions": [...]}``: boolean
            Python expressions over ``result`` (the return value) and
            ``kwargs`` (the call's keyword arguments) checked after every
            call, e.g. ``["isinstance(result, list)"]``. Only comparison /
            boolean operators and basic builtins (``len``, ``all``, ``any``,
            ``isinstance``, ``min``, ``max``, ``sum``, …) are allowed. Input
            and output JSON schemas are derived from type hints, so hint
            every parameter and the return type. Postconditions survive
            ``overwrite=True`` unless a new ``contracts`` entry replaces them.
        fixtures : dict[str, list[dict]] | None
            Mapping of function name → recorded ``{"args": {...}, "result":
            ...}`` pairs the function must reproduce. Pure functions only (no
            I/O; effect class ``safe_noop``); replayed whenever the
            implementation changes, and a mismatch rejects the change with
            ``FixtureRegressionError`` naming the failing pair.
        overwrite : bool, default ``False``
            When ``True``, an existing name is updated in place with a stable
            ``function_id`` so existing references (task entrypoints,
            guidance links) keep resolving — always prefer this over
            delete-and-re-add. When ``False``, existing names are skipped and
            reported as duplicates.
        raise_on_error : bool, default ``True``
            Raise ``ValueError`` naming the failed functions and errors,
            instead of reporting them in the result. Failures for one
            function never block the rest of the batch.
        venv_id : int | None
            **Required** when a function imports third-party packages (beyond
            the standard library and the execution environment); otherwise
            ``ValueError`` is raised. Create a venv via ``add_venv`` first
            and pass the returned ID.

        Returns
        -------
        dict[str, str]
            Function name → status, e.g. ``{"my_func": "added"}`` or
            ``{"my_func": "error: <message>"}``.

        Anti-patterns
        -------------
        - Silent early returns or empty/"ok" results with no ``PHASE`` /
          ``SKIP`` / ``SOFT_FAIL`` trail — soft failures need stdlib
          ``logging`` calls with those markers (not user-facing
          notifications), and the trail must survive distillation from a
          live trajectory.
        - Nesting ``asyncio.run(...)`` inside sync helpers / entrypoints —
          offline Jobs and actor sandboxes already own a loop. Prefer
          ``async def`` + ``await``, or the injected ``run_coro_sync`` helper.
        """

    @abstractmethod
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of function name to function metadata.

        Each record conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        include_implementations : bool, default ``False``
            When ``True``, values include the full source code in the
            ``implementation`` field. When ``False``, implementations may be
            omitted to reduce payload size.
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": <...>, "metadata": <...>}``.

        Returns
        -------
        dict[str, Function] | dict[str, Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: mapping of function name → record
              conforming to the ``Function`` schema (as dicts or Function objects).
              When ``include_implementations=False``, the ``implementation`` field
              may be omitted.
            - When ``_return_callable=True``: mapping of function name → callable.
              Callables MAY be in-process functions or proxy callables for functions
              that must execute in an isolated virtual environment (implementation‑defined).
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding mappings.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.
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
    def reconcile_dependencies(
        self,
        *,
        function_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Refresh structured link debt for compositional dependencies.

        Audits each selected compositional function's declared ``depends_on``
        names against the current compositional and primitive catalogues.
        Existing ``stale_reasons`` for dependencies that resolve again are
        removed; missing dependencies are recorded without changing
        ``depends_on``.

        Parameters
        ----------
        function_ids : list[int] | None
            Optional subset to audit; when omitted, checks all compositional
            functions.

        Returns
        -------
        dict[str, Any]
            Outcome with checked and stale function identifiers.
        """

    @abstractmethod
    def filter_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_implementations: bool = True,
        destination: Optional[str] = ...,  # type: ignore[assignment]
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter stored function metadata using a Python expression.

        Each result conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        filter : str | None, default ``None``
            A boolean expression evaluated per row with fields of the
            ``Function`` model in scope (e.g. ``name``, ``argspec``,
            ``docstring``, ``depends_on``). When ``None``, returns all rows subject
            to pagination. Supported grammar: comparisons (==, !=, <, <=, >, >=),
            membership tests (in / not in), and boolean combinators (and, or,
            not) over field names and literal values, plus a fixed set of
            helpers (``len()``, string methods like ``.lower()`` /
            ``.startswith()``, ``embed()``). Arbitrary Python calls outside
            that set — e.g. ``' '.join(depends_on)`` or a list comprehension —
            are rejected.
        offset : int, default ``0``
            Zero‑based index of the first result to return.
        limit : int, default ``100``
            Maximum number of results to return. Must be <= 1000.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        destination : str | None, optional
            When omitted, reads federate across personal and shared
            Compositional roots (personal-first). When provided, restrict the
            Compositional read to one root: ``None`` / ``"personal"`` for the
            personal catalog, or ``"team:<id>"`` for a shared team catalog.
            Use this when resolving a ``function_id`` that is only unique
            within one destination (e.g. symbolic task entrypoints).
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[Function] | list[Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: list of records conforming to the
              ``Function`` schema (as dicts or Function objects). When
              ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``_return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.

        Examples
        --------
        >>> mgr.filter_functions(filter="'price' in docstring and 'sum' in depends_on")
        >>> mgr.filter_functions(filter="name.startswith('get_')")
        """

    @abstractmethod
    def search_functions(
        self,
        *,
        query: str = "",
        n: int = 5,
        include_implementations: bool = True,
        include_dormant: bool = False,
        _return_callable: bool = False,
        _namespace: Optional[Dict[str, Any]] = None,
        _also_return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity to a natural‑language query.

        Results are ranked like memory, not just like an index: semantic
        similarity dominates, and a function's *standing* — how often and
        how recently it has actually been used, judged against its own
        usage rhythm — acts as the tiebreaker. Functions whose standing has
        fully lapsed drop out of results entirely (they still exist and
        still run; ``filter_functions``/``list_functions`` always see the
        whole store, and ``include_dormant=True`` brings them back here).
        Freshly stored functions surface normally: creation counts as a
        first use.

        Every result carries the ranking components in the open:
        ``_similarity`` (semantic match, 0–1), ``_standing`` (usage-based
        memory strength, 0–1 — recency against the function's own rhythm ×
        log-saturating call count), and ``_retrieval_score`` (the combined
        rank, ``similarity × (floor + (1−floor) × standing)``), beside the
        raw trace (``usage_calls``, ``usage_last_called_at``). Read them:
        a highly similar result with near-zero standing is a plausible but
        long-unexercised skill, while a moderate match with high standing
        is a battle-tested recent workhorse.

        Each result conforms to the ``Function`` schema and includes a
        ``guidance_ids`` field — a list of identifiers for related guidance
        entries that describe compositional procedures using these functions.

        Parameters
        ----------
        query : str, default ``""``
            Natural‑language text describing the desired function(s). An empty
            query is allowed (soft models sometimes omit it during discovery)
            and returns a broad backfilled sample rather than failing.
        n : int, default ``5``
            Number of similar results to return.
        include_implementations : bool, default ``True``
            When ``True``, results include the full source code in the
            ``implementation`` field. When ``False``, implementations are
            omitted to reduce payload size.
        include_dormant : bool, default ``False``
            When ``True``, functions whose usage standing has fully lapsed
            are ranked and returned like any other instead of being dropped
            from the results. Use when hunting for a skill you believe was
            stored long ago but has not been exercised recently.
        _return_callable : bool, default ``False``
            When ``True``, return Python callables instead of metadata dicts.
            Implementations SHOULD inject the resulting callables (and any of their
            transitive dependencies) into the provided ``_namespace``.
        _namespace : dict[str, Any] | None, default ``None``
            Target namespace dict for dependency injection when
            ``_return_callable=True``. Required when ``_return_callable=True``.
        _also_return_metadata : bool, default ``False``
            When ``True`` (and only valid with ``_return_callable=True``), return a
            dict containing both callables and metadata:
            ``{"callables": [...], "metadata": [...]}``.

        Returns
        -------
        list[dict[str, Any]] | list[Callable[..., Any]] | dict[str, Any]
            - When ``_return_callable=False``: up to ``n`` results ordered by similarity.
              Each element SHOULD include the fields of the ``Function`` model and MAY
              include an additional ``score`` field (``float``) representing similarity.
              When ``include_implementations=False``, the ``implementation`` field
              is omitted.
            - When ``_return_callable=True``: list of callables corresponding to the
              returned records.
            - When ``_also_return_metadata=True``: a dict with keys ``callables`` and
              ``metadata`` containing the two corresponding lists.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_also_return_metadata=True`` but ``_return_callable`` is ``False``.
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
        extra_namespaces: Optional[Dict[str, Any]] = None,
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
            extra_namespaces: dict[str, Any] | None = None,
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
        extra_namespaces : dict[str, Any] | None, default ``None``
            Named objects to inject into the function's execution namespace.
            For in-process Python execution, all entries are injected into the
            globals dict. For venv/subprocess execution, ``"primitives"`` and
            ``"primitives"`` entries are bridged via RPC; other entries
            are only available in-process.

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

        >>> # Execute with extra namespaces (e.g. sub-agent environment)
        >>> result = await fm.execute_function(
        ...     function_name="my_func",
        ...     extra_namespaces={"primitives": prims, "sub_agents": agent_env},
        ... )
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Sentinel for "use the function's default venv_id"
USE_FUNCTION_DEFAULT = ...

# Attach centralised docstring
BaseFunctionManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
