import ast
import builtins
import inspect
import threading
from typing import Dict, List, Set, Union, Tuple, Any, Optional

import unify


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

    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}

    # ------------------------------------------------------------------ #
    #  Static validation helpers                                          #
    # ------------------------------------------------------------------ #

    _ALLOWED_BUILTINS: Set[str] = {
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
    _DISALLOWED_BUILTINS: Set[str] = set(dir(builtins)) - _ALLOWED_BUILTINS

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

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: ast.FunctionDef = tree.body[0]
        if fn_node.col_offset != 0:
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    @staticmethod
    def _ensure_no_imports(tree: ast.Module, fn_name: str) -> None:
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
            raise ValueError(f"Imports are not allowed (found in {fn_name}()).")

    def _collect_function_calls(self, fn_node: ast.FunctionDef) -> Set[str]:
        calls: Set[str] = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                match node.func:
                    case ast.Name(id=ident):
                        calls.add(ident)
                    case ast.Attribute():
                        raise ValueError(
                            f"Attribute call '{ast.unparse(node.func)}' "
                            f"is not allowed in {fn_node.name}().",
                        )
        return calls

    def _validate_function_calls(
        self,
        fn_name: str,
        calls: Set[str],
        provided_names: Set[str],
    ) -> None:
        for called in calls:
            if called in self._DISALLOWED_BUILTINS:
                raise ValueError(
                    f"Built-in '{called}' is not permitted in {fn_name}().",
                )
            if called not in provided_names and called not in self._ALLOWED_BUILTINS:
                raise ValueError(
                    f"{fn_name}() references unknown function '{called}'. "
                    "All referenced functions must be provided together.",
                )

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    _CTX = "Functions"

    def _get_log_by_function_id(self, *, function_id: int) -> unify.Log:
        log_ids = unify.get_logs(
            context=self._CTX,
            filter=f"function_id == {function_id}",
            return_ids_only=True,
        )
        assert len(log_ids) == 1, f"No function with id {function_id!r} exists."
        return log_ids[0]

    def _next_function_id(self) -> int:
        if self._CTX not in unify.get_contexts():
            return 0
        logs = unify.get_logs(context=self._CTX)
        return max(lg.entries["function_id"] for lg in logs) + 1 if logs else 0

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
            self._ensure_no_imports(tree, name)
            calls = self._collect_function_calls(node)
            self._validate_function_calls(name, calls, provided_names)

        # Compile & persist
        results: Dict[str, str] = {}
        next_id = self._next_function_id()

        for name, _, node, source in parsed:
            namespace: Dict[str, object] = {}
            exec(source, namespace)
            fn_obj = namespace[name]

            signature = str(inspect.signature(fn_obj))
            docstring = inspect.getdoc(fn_obj) or ""
            calls = list(self._collect_function_calls(node))

            unify.log(
                context=self._CTX,
                function_id=next_id,
                name=name,
                argspec=signature,
                docstring=docstring,
                implementation=source,
                calls=calls,
                new=True,
            )

            results[name] = "added"
            next_id += 1

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
        for log in unify.get_logs(context=self._CTX):
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
                context=self._CTX,
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
            context=self._CTX,
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
        return [
            lg.entries
            for lg in unify.get_logs(
                context=self._CTX,
                filter=filter,
                offset=offset,
                limit=limit,
            )
        ]
