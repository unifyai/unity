from __future__ import annotations

import ast
from typing import Optional


class _RewriteGetAndAttributes(ast.NodeTransformer):
    """
    Transform filter expressions into a backend-friendly subset:
    - Rewrite `obj.get("key")` → `obj["key"]` when possible.
    - Rewrite attribute access `obj.key` → `obj["key"]`.
    - Transform `isinstance(x, T)` → `type(x) is T` and
      `isinstance(x, (T1, T2))` → `type(x) in (T1, T2)`.
    """

    def __init__(self) -> None:
        pass

    def visit_Attribute(self, node: ast.Attribute):  # type: ignore[override]
        # Rewrite obj.key → obj["key"] to match how nested dict data is stored.
        new_value = self.visit(node.value)
        return ast.Subscript(
            value=new_value,
            slice=ast.Constant(node.attr),
            ctx=ast.Load(),
        )

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        # Special case handled BEFORE generic traversal so we don't rewrite
        # `.get` into a subscript on the attribute access itself.
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    base = self.visit(node.func.value)
                    key = node.args[0].value
                    return ast.Subscript(
                        value=base,
                        slice=ast.Constant(key),
                        ctx=ast.Load(),
                    )
        except Exception:
            pass

        # isinstance(x, T) → type(x) is T; isinstance(x, (T1, T2)) → type(x) in (...)
        try:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) >= 2
            ):
                target = self.visit(node.args[0])
                type_expr = self.visit(node.args[1])
                type_call = ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[target],
                    keywords=[],
                )
                if isinstance(type_expr, ast.Tuple):
                    return ast.Compare(
                        left=type_call,
                        ops=[ast.In()],
                        comparators=[type_expr],
                    )
                else:
                    return ast.Compare(
                        left=type_call,
                        ops=[ast.Is()],
                        comparators=[type_expr],
                    )
        except Exception:
            pass

        # Default: traverse children and leave as-is
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node: ast.Subscript):  # type: ignore[override]
        # First, recursively visit children so Attribute/Call rewrites happen
        new_value = self.visit(node.value)
        new_slice = self.visit(node.slice)
        return ast.Subscript(value=new_value, slice=new_slice, ctx=ast.Load())


def normalize_filter_expr(expr: Optional[str]) -> Optional[str]:
    """
    Best‑effort normalization of a filter expression for the backend.

    Transformations performed:
    - Attribute access: `obj.key`                → `obj["key"]`
    - Mapping get:    `obj.get("key")`         → `obj["key"]`
    - isinstance:     `isinstance(x, T)`        → `type(x) is T`
                     `isinstance(x, (T1,T2))`   → `type(x) in (T1, T2)`

    Unknown constructs are left unchanged; callers should pass the result to the
    backend and allow upstream logic to handle any remaining errors.
    """
    if expr is None or not isinstance(expr, str) or not expr.strip():
        return expr

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return expr

    rewriter = _RewriteGetAndAttributes()
    try:
        rewritten = rewriter.visit(tree)
    except Exception:
        return expr

    ast.fix_missing_locations(rewritten)

    try:
        return ast.unparse(rewritten)
    except Exception:
        return expr
