import ast
import inspect
import functools
from typing import Dict, List, Set, Union, Tuple, Any, Optional
import unify
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column, list_private_fields
from ..common.sandbox_utils import create_sandbox_globals
from .types.function import Function
from .base import BaseFunctionManager
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore


class FunctionManager(BaseFunctionManager):
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
        # No thread behavior; keep parameter for backward compatibility
        self._daemon = daemon
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}

        # Internal monotonically-increasing function-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a function.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FunctionManager."
        self._ctx = f"{read_ctx}/Functions" if read_ctx else "Functions"

        # Ensure functions context and fields exist deterministically
        self._store = TableStore(
            self._ctx,
            unique_keys={"function_id": "int"},
            auto_counting={"function_id": None},
            description="List of functions, with all function details stored.",
            fields=model_to_fields(Function),
        )
        self._store.ensure_context()
        # Add tracing
        if traced:
            self = unify.traced(self)

    @property
    def _dangerous_builtins(self) -> Set[str]:
        """
        A minimal set of truly dangerous built-ins that should never be allowed.
        These could compromise security or system integrity.
        """
        return {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",  # File system access should go through proper APIs
            "input",  # No interactive input in automated functions
            "breakpoint",  # No debugging breakpoints
            "exit",
            "quit",
        }

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
        """
        Validates function calls to prevent functions from calling other user-defined functions.

        Allows:
        - Built-in functions from the allowed list
        - Any method calls on objects (e.g., action_provider.*, call_handle.*, call.*)

        Disallows:
        - Direct calls to any user-defined functions
        - Disallowed built-in functions
        """
        dangerous = self._dangerous_builtins

        for called in calls:
            # Allow all method calls (anything with a dot)
            # This includes action_provider.*, call_handle.*, obj.method(), etc.
            if "." in called:
                continue

            # Block only truly dangerous built-ins
            if called in dangerous:
                raise ValueError(
                    f"Dangerous built-in '{called}' is not permitted in {fn_name}(). "
                    f"Functions cannot use: {', '.join(sorted(dangerous))}",
                )

            # Block direct calls to other user-defined functions
            # (but not built-ins or exception classes)
            if called in provided_names:
                raise ValueError(
                    f"{fn_name}() cannot call user-defined function '{called}'. "
                    "Functions must not call other user-defined functions.",
                )

            # Everything else is allowed - including all built-ins, exception classes, etc.

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    def _get_log_by_function_id(self, *, function_id: int) -> unify.Log:
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"function_id == {function_id}",
            exclude_fields=list_private_fields(self._ctx),
        )
        assert len(logs) == 1, f"No function with id {function_id!r} exists."
        return logs[0]

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    # 1. Add / register ------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:

        if preconditions is None:
            preconditions = {}
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
            namespace = create_sandbox_globals()
            exec(source, namespace)
            fn_obj = namespace[name]

            signature = str(inspect.signature(fn_obj))
            docstring = inspect.getdoc(fn_obj) or ""
            calls = list(self._collect_function_calls(node))

            # Create a combined string for embedding
            embedding_text = (
                f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
            )
            precondition = preconditions.get(name)

            unify.log(
                context=self._ctx,
                name=name,
                argspec=signature,
                docstring=docstring,
                implementation=source,
                calls=calls,
                embedding_text=embedding_text,
                precondition=precondition,
                new=True,
            )

            results[name] = "added"
        return results

    # 2. Listing -------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:

        entries: Dict[str, Dict[str, Any]] = {}
        for log in unify.get_logs(
            context=self._ctx,
            exclude_fields=list_private_fields(self._ctx),
        ):
            data = {
                "function_id": log.entries["function_id"],
                "argspec": log.entries["argspec"],
                "docstring": log.entries["docstring"],
            }
            if include_implementations:
                data["implementation"] = log.entries["implementation"]
            entries[log.entries["name"]] = data
        return entries

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"name == '{function_name}'",
            limit=1,
            exclude_fields=list_private_fields(self._ctx),
        )
        if not logs:
            return None

        return logs[0].entries.get("precondition")

    # 3. Deletion ------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:

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

    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:

        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            exclude_fields=list_private_fields(self._ctx),
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

    @functools.wraps(BaseFunctionManager.search_functions_by_similarity, updated=())
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:

        self._ensure_function_embedding()
        escaped_query = query.replace("'", "\\'")
        logs = unify.get_logs(
            context=self._ctx,
            sorting={
                f"cosine({self._FUNC_EMB}, embed('{escaped_query}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=n,
            exclude_fields=list_private_fields(self._ctx),
        )
        return [lg.entries for lg in logs]
