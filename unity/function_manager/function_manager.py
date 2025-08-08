import ast
import builtins
import inspect
import threading
from typing import Dict, List, Set, Union, Tuple, Any, Optional
import unify
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from .types.function import Function
from ..common.model_to_fields import model_to_fields


class FunctionManager(threading.Thread):
    """
    Keeps a catalogue of user-supplied Python functions that can reference
    one another.  Each function is stored in the `unify` backend so that it
    can be listed, searched and cleanly deleted (optionally cascading to
    dependants).
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #
    _FUNC_EMB = "_function_embedding"

    def __init__(self, *, daemon: bool = True, traced: bool = False) -> None:
        super().__init__(daemon=daemon)
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}

        # Internal monotonically-increasing function-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a function.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FunctionManager."
        self._ctx = f"{read_ctx}/Functions" if read_ctx else "Functions"

        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_column_ids="function_id",
                description="List of functions, with all function details stored.",
            )
            fields = model_to_fields(Function)
            unify.create_fields(
                fields,
                context=self._ctx,
            )
        # Add tracing
        if traced:
            self = unify.traced(self)

    @property
    def _allowed_calls(self) -> Set[str]:
        """
        Dynamically generates the set of all allowed function and method calls.
        """
        standard_builtins = {
            "range",
            "enumerate",
            "len",
            "str",
            "min",
            "max",
            "zip",
            "sum",
            "sorted",
            "abs",
            "round",
            "pow",
            "divmod",
            "int",
            "float",
            "complex",
            "bool",
            "list",
            "tuple",
            "set",
            "dict",
            "reversed",
            "slice",
            "all",
            "any",
            "chr",
            "ord",
            "isinstance",
            "issubclass",
            "id",
        }

        # Lazy import to avoid circular dependency: the ActionProvider is only
        # imported when this property is accessed, which happens *after* both
        # modules have finished initial loading.  This breaks the circular
        # import chain between FunctionManager <-> planner package.
        from unity.planner.action_provider import ActionProvider  # noqa: WPS433,E402

        action_provider_methods = {
            name
            for name, _ in inspect.getmembers(ActionProvider, inspect.isfunction)
            if not name.startswith("_")
        }

        allowed_globals = {"action_provider"}

        return standard_builtins | action_provider_methods | allowed_globals

    @property
    def _disallowed_builtins(self) -> Set[str]:
        """Built-ins that are not in our explicit allow-list."""
        return set(dir(builtins)) - self._allowed_calls

    def _parse_implementation(
        self,
        source: str,
    ) -> Tuple[str, ast.Module, ast.FunctionDef, str]:
        """
        Common syntactic checks (unchanged, but now returns the stripped
        source verbatim so we can persist it later).
        """
        stripped = source.lstrip("\n")
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith((" ", "\t")):
            raise ValueError(
                "Function definition must start at column 0 (no indentation).",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error:\n{e.text}") from e

        if len(tree.body) != 1 or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef] = tree.body[0]
        if fn_node.col_offset != 0:
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    def _collect_function_calls(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Set[str]:
        calls: Set[str] = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                name = self._format_callable_name(node.func)
                if name:
                    calls.add(name)
        return calls

    def _format_callable_name(self, callable_node: ast.AST) -> Optional[str]:
        """Return a best-effort fully qualified name for a callable.

        Handles both simple names (e.g., ``foo()``) and nested attributes
        (e.g., ``a.b.c()``). If the base of the attribute chain is not a simple
        ``ast.Name`` (e.g., ``get().b()``), this falls back to ``ast.unparse``
        when available.
        """
        # Simple function call: foo()
        if isinstance(callable_node, ast.Name):
            return callable_node.id

        # Attribute access: a.b.c()
        if isinstance(callable_node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = callable_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
            # Fallback to unparse for complex bases like calls/subscripts
            try:
                return ast.unparse(callable_node)
            except Exception:
                pass
            return ".".join(reversed(parts)) if parts else None

        try:
            return ast.unparse(callable_node)
        except Exception:
            return None

    def _validate_function_calls(
        self,
        fn_name: str,
        calls: Set[str],
        provided_names: Set[str],
    ) -> None:
        allowed = self._allowed_calls
        disallowed_builtins = self._disallowed_builtins
        for called in calls:
            if called in disallowed_builtins:
                raise ValueError(
                    f"Built-in '{called}' is not permitted in {fn_name}().",
                )
            if called not in provided_names and called not in allowed:
                raise ValueError(
                    f"{fn_name}() references unknown function '{called}'. "
                    "All referenced functions must be provided together.",
                )

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    def _get_log_by_function_id(self, *, function_id: int) -> unify.Log:
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"function_id == {function_id}",
        )
        assert len(logs) == 1, f"No function with id {function_id!r} exists."
        return logs[0]

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    # 1. Add / register ------------------------------------------------- #

    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
    ) -> Dict[str, str]:
        """
        Validate, compile and persist one or more function implementations.

        Returns
        -------
        Dict[str, str]  –  ``{<name>: "added" | "error: <msg>"}``
        """
        if isinstance(implementations, str):
            implementations = [implementations]

        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        for source in implementations:
            parsed.append(self._parse_implementation(source))

        provided_names = {name for name, *_ in parsed}

        # Deep validation
        for name, tree, node, _ in parsed:
            calls = self._collect_function_calls(node)
            self._validate_function_calls(name, calls, provided_names)

        # Compile & persist
        results: Dict[str, str] = {}

        for name, _, node, source in parsed:
            namespace: Dict[str, object] = {}
            exec(source, namespace)
            fn_obj = namespace[name]

            signature = str(inspect.signature(fn_obj))
            docstring = inspect.getdoc(fn_obj) or ""
            calls = list(self._collect_function_calls(node))

            # Create a combined string for embedding
            embedding_text = (
                f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
            )

            unify.log(
                context=self._ctx,
                name=name,
                argspec=signature,
                docstring=docstring,
                implementation=source,
                calls=calls,
                embedding_text=embedding_text,
                new=True,
            )

            results[name] = "added"
        return results

    # 2. Listing -------------------------------------------------------- #

    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary keyed by *function name*.

        Each value contains:

        * **argspec**   – full signature, e.g. ``(x: int, y: int) -> int``
        * **docstring** – cleaned docstring or empty string
        * **implementation** – full source code (only when
          ``include_implementations=True``)
        """
        entries: Dict[str, Dict[str, Any]] = {}
        for log in unify.get_logs(context=self._ctx):
            data = {
                "argspec": log.entries["argspec"],
                "docstring": log.entries["docstring"],
            }
            if include_implementations:
                data["implementation"] = log.entries["implementation"]
            entries[log.entries["name"]] = data
        return entries

    # 3. Deletion ------------------------------------------------------- #

    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete a function by *id*.  If `delete_dependents` is ``True`` (the
        default) then every function that calls the target is recursively
        removed as well.
        """
        log = self._get_log_by_function_id(function_id=function_id)
        target_name = log.entries["name"]

        # Identify dependants (direct callers)
        if delete_dependents:
            dependants = unify.get_logs(
                context=self._ctx,
                filter=f"'{target_name}' in calls",
            )
            for dep in dependants:
                if dep.entries["function_id"] == function_id:
                    continue  # skip the target itself
                self.delete_function(
                    function_id=dep.entries["function_id"],
                    delete_dependents=True,
                )

        unify.delete_logs(
            context=self._ctx,
            logs=log.id,
        )
        return {target_name: "deleted"}

    # 4. Search --------------------------------------------------------- #

    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Flexible, *Python-expression* filtering over every stored column
        (`name`, `argspec`, `docstring`, `calls`, …).

        Examples
        --------
        >>> mgr.search_functions(filter="'price' in docstring and 'sum' in calls")
        >>> mgr.search_functions(filter="name.startswith('get_')")
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [lg.entries for lg in logs]

    # 5. Semantic Search ------------------------------------------------ #
    def _ensure_function_embedding(self) -> None:
        """
        Ensure that the function embedding column exists.
        """
        ensure_vector_column(
            self._ctx,
            embed_column=self._FUNC_EMB,
            source_column="embedding_text",
        )

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
        List[Dict[str, Any]]
            A list of the n most similar functions.
        """
        self._ensure_function_embedding()
        escaped_query = query.replace("'", "\\'")
        logs = unify.get_logs(
            context=self._ctx,
            sorting={
                f"cosine({self._FUNC_EMB}, embed('{escaped_query}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=n,
            exclude_fields=[self._FUNC_EMB],
        )
        return [lg.entries for lg in logs]
