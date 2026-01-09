from __future__ import annotations

import ast
from typing import Optional, Set


class DependencyVisitor(ast.NodeVisitor):
    """
    Stateful dependency collector for FunctionManager functions.

    Captures:
    - Direct calls: `foo()`
    - Indirect calls via variable assignment: `f = foo; f()`
    - Returned function references: `return foo` or `return f` (where f maps to foo)
    """

    def __init__(self, known_function_names: Set[str]):
        self.known_function_names = known_function_names
        self.dependencies: Set[str] = set()
        self._assignment_map: dict[str, str] = {}

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

        if called_name:
            self.dependencies.add(called_name)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        # Return statement -> return func_name or return var
        if isinstance(node.value, ast.Name):
            returned_name = node.value.id
            if returned_name in self.known_function_names:
                self.dependencies.add(returned_name)
            elif returned_name in self._assignment_map:
                self.dependencies.add(self._assignment_map[returned_name])

        self.generic_visit(node)


def _collect_known_names(expr: Optional[ast.AST], known_function_names: Set[str]) -> Set[str]:
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
) -> Set[str]:
    """
    Collect dependencies for a single top-level function node.

    Includes body-call dependencies (via DependencyVisitor) plus additional references that
    must resolve at *definition time*:
    - Decorators (e.g. `@my_decorator`)
    - Annotations (e.g. `x: typing.Annotated[int, validator]`)
    """
    visitor = DependencyVisitor(known_function_names)
    visitor.visit(fn_node)
    deps: Set[str] = set(visitor.dependencies)

    # Decorators can be `@name`, `@name(...)`, etc.
    for dec in getattr(fn_node, "decorator_list", []) or []:
        deps |= _collect_known_names(dec, known_function_names)

    # Function signature annotations + return annotation
    args = fn_node.args
    for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        deps |= _collect_known_names(getattr(arg, "annotation", None), known_function_names)
    deps |= _collect_known_names(getattr(args.vararg, "annotation", None), known_function_names)
    deps |= _collect_known_names(getattr(args.kwarg, "annotation", None), known_function_names)
    deps |= _collect_known_names(getattr(fn_node, "returns", None), known_function_names)

    deps.discard(fn_node.name)
    return deps


def collect_dependencies_from_source(
    source: str,
    known_function_names: Set[str],
) -> Set[str]:
    """Parse a single-function source string and return its dependency names."""
    try:
        tree = ast.parse(source)
    except Exception:
        return set()
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    node: ast.FunctionDef | ast.AsyncFunctionDef = tree.body[0]
    return collect_dependencies_from_function_node(node, known_function_names)

