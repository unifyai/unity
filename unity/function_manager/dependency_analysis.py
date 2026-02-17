"""AST-based dependency detection for stored functions.

This module is the **storage-time** half of the dependency pipeline.  It
analyses a function's AST to discover which other functions and environment
namespaces the code calls, and records those names in the ``depends_on``
field of the stored function record.

The **runtime** counterpart is ``FunctionManager._inject_dependencies``,
which reads ``depends_on`` and injects the corresponding objects into the
execution namespace:

* Bare names (``"helper"``) → the stored implementation is exec'd.
* Dotted names (``"actor.act"``) → the root namespace object is constructed
  via ``registry.construct_sandbox_root()``.

The set of recognised environment namespace roots (``"primitives"``,
``"computer_primitives"``, ``"actor"``) is passed in as
*environment_namespaces* by the caller (``FunctionManager.add_functions``).
"""

from __future__ import annotations

import ast
from typing import FrozenSet, Optional, Set


class DependencyVisitor(ast.NodeVisitor):
    """AST visitor that collects dependency names from a function body.

    Captures:
    - Direct calls: ``foo()``
    - Indirect calls via variable assignment: ``f = foo; f()``
    - Returned function references: ``return foo``
    - Callables passed as arguments: ``bar(callback=foo)``
    - Dotted environment calls: ``primitives.contacts.ask(...)``,
      ``computer_primitives.screenshot(...)``, ``actor.act(...)``

    Dotted calls are only captured when the root segment matches one of the
    *environment_namespaces* provided at construction time.  The full dotted
    name (e.g. ``"actor.act"``) is recorded in ``depends_on`` so that
    ``_inject_dependencies`` can resolve the root namespace at runtime.
    """

    def __init__(
        self,
        known_function_names: Set[str],
        *,
        environment_namespaces: FrozenSet[str] = frozenset(),
    ):
        self.known_function_names = known_function_names
        self.environment_namespaces = environment_namespaces
        self.dependencies: Set[str] = set()
        self._assignment_map: dict[str, str] = {}

    @staticmethod
    def _resolve_dotted_name(node: ast.AST) -> Optional[str]:
        """Resolve an ast.Attribute chain to a dotted string like 'primitives.contacts.ask'."""
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def visit_Assign(self, node: ast.Assign):
        # Only track simple assignments: target_var = potential_func_name
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_var = node.targets[0].id
            if isinstance(node.value, ast.Name):
                assigned_name = node.value.id
                # Check if the assigned name is one of the functions we manage
                if assigned_name in self.known_function_names:
                    self._assignment_map[target_var] = assigned_name
                elif target_var in self._assignment_map:
                    del self._assignment_map[target_var]
            elif target_var in self._assignment_map:
                del self._assignment_map[target_var]

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_node = node.func
        called_name: Optional[str] = None

        # Direct call -> func_name()
        if isinstance(func_node, ast.Name):
            func_name = func_node.id
            if func_name in self.known_function_names:
                called_name = func_name
            elif func_name in self._assignment_map:
                called_name = self._assignment_map[func_name]

        # Dotted call -> primitives.contacts.ask(), computer_primitives.act(), etc.
        elif isinstance(func_node, ast.Attribute) and self.environment_namespaces:
            dotted = self._resolve_dotted_name(func_node)
            if dotted:
                root = dotted.split(".")[0]
                if root in self.environment_namespaces:
                    called_name = dotted

        if called_name:
            self.dependencies.add(called_name)

        self._collect_callable_args(node)
        self.generic_visit(node)

    def _collect_callable_args(self, node: ast.Call):
        """Collect function references passed as positional or keyword arguments."""
        for arg in node.args:
            if isinstance(arg, ast.Name):
                if arg.id in self.known_function_names:
                    self.dependencies.add(arg.id)
                elif arg.id in self._assignment_map:
                    self.dependencies.add(self._assignment_map[arg.id])

        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name):
                name = keyword.value.id
                if name in self.known_function_names:
                    self.dependencies.add(name)
                elif name in self._assignment_map:
                    self.dependencies.add(self._assignment_map[name])

    def visit_Return(self, node: ast.Return):
        # Return statement -> return func_name or return var
        if isinstance(node.value, ast.Name):
            returned_name = node.value.id
            if returned_name in self.known_function_names:
                self.dependencies.add(returned_name)
            elif returned_name in self._assignment_map:
                self.dependencies.add(self._assignment_map[returned_name])

        self.generic_visit(node)


def _collect_known_names(
    expr: Optional[ast.AST],
    known_function_names: Set[str],
) -> Set[str]:
    if expr is None:
        return set()
    deps: Set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in known_function_names:
            deps.add(node.id)
    return deps


def collect_dependencies_from_function_node(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    known_function_names: Set[str],
    *,
    environment_namespaces: FrozenSet[str] = frozenset(),
) -> Set[str]:
    """
    Collect dependencies for a single top-level function node.

    Includes body-call dependencies (via DependencyVisitor) plus additional references that
    must resolve at *definition time*:
    - Decorators (e.g. `@my_decorator`)
    - Annotations (e.g. `x: typing.Annotated[int, validator]`)

    When *environment_namespaces* is provided, dotted calls whose root segment matches
    one of the namespaces (e.g. ``primitives.contacts.ask``, ``actor.act``) are also
    captured as dependencies.
    """
    visitor = DependencyVisitor(
        known_function_names,
        environment_namespaces=environment_namespaces,
    )
    visitor.visit(fn_node)
    deps: Set[str] = set(visitor.dependencies)

    # Decorators can be `@name`, `@name(...)`, etc.
    for dec in getattr(fn_node, "decorator_list", []) or []:
        deps |= _collect_known_names(dec, known_function_names)

    # Function signature annotations + return annotation
    args = fn_node.args
    for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        deps |= _collect_known_names(
            getattr(arg, "annotation", None),
            known_function_names,
        )
    deps |= _collect_known_names(
        getattr(args.vararg, "annotation", None),
        known_function_names,
    )
    deps |= _collect_known_names(
        getattr(args.kwarg, "annotation", None),
        known_function_names,
    )
    deps |= _collect_known_names(
        getattr(fn_node, "returns", None),
        known_function_names,
    )

    deps.discard(fn_node.name)
    return deps


def collect_dependencies_from_source(
    source: str,
    known_function_names: Set[str],
    *,
    environment_namespaces: FrozenSet[str] = frozenset(),
) -> Set[str]:
    """Parse a single-function source string and return its dependency names."""
    try:
        tree = ast.parse(source)
    except Exception:
        return set()
    if len(tree.body) != 1 or not isinstance(
        tree.body[0],
        (ast.FunctionDef, ast.AsyncFunctionDef),
    ):
        return set()
    node: ast.FunctionDef | ast.AsyncFunctionDef = tree.body[0]
    return collect_dependencies_from_function_node(
        node,
        known_function_names,
        environment_namespaces=environment_namespaces,
    )
