import ast
import builtins
import threading
from typing import Dict, List, Set, Union, Tuple


class FunctionManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for managing the set of re-usable functions, available when planning how to execute tasks.
        """
        super().__init__(daemon=daemon)
        # ToDo: implement the tools
        self._tools = {}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                   #
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
        • Must be valid Python.
        • Must contain exactly one top-level ``def foo(...):``.
        • That ``def`` must start in column 0 (no indentation).
        """
        stripped = source.lstrip("\n")  # ignore blank leading lines
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith((" ", "\t")):
            raise ValueError(
                "Function definition must start at column 0 (no indentation).",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            # Preserve original guidance for other syntax problems
            raise ValueError(f"Syntax error:\n{e.text}") from e

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: ast.FunctionDef = tree.body[0]
        if fn_node.col_offset != 0:  # catches inner whitespace
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    def _ensure_no_imports(self, tree: ast.Module, fn_name: str) -> None:
        """Fail if any `import` or `from ... import` nodes are present."""
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
            raise ValueError(f"Imports are not allowed (found in {fn_name}()).")

    def _collect_function_calls(self, fn_node: ast.FunctionDef) -> Set[str]:
        """
        Returns every *raw* name that appears as a call,
        e.g. ``foo()`` → "foo".
        Attribute calls like ``math.sin()`` raise immediately.
        """
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
        """
        • Only `_ALLOWED_BUILTINS` are permitted from builtins.
        • Every other call must reference a function supplied in *this*
          `add_functions` invocation.
        """
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

    # Public #
    # -------#

    # English-Text update request

    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
    ) -> Dict[str, str]:
        """
        Validate and register user-supplied function implementations.

        See docstring on the private helpers for validation rules.

        Returns
        -------
        Dict[str, str]
            ``{"<func_name>": "added" | "error: <msg>"}``
        """
        if isinstance(implementations, str):
            implementations = [implementations]

        # First pass – parse each source, grab the name, store results
        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        for source in implementations:
            parsed.append(self._parse_implementation(source))

        provided_names = {name for name, *_ in parsed}

        # Second pass – deep validation
        for name, tree, node, _ in parsed:
            self._ensure_no_imports(tree, name)
            calls = self._collect_function_calls(node)
            self._validate_function_calls(name, calls, provided_names)

        # Third pass – compile & register
        results: Dict[str, str] = {}
        for name, _, _, source in parsed:
            namespace: Dict[str, object] = {}
            exec(source, namespace)
            results[name] = "added"
        return results
