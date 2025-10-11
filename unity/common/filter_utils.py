from __future__ import annotations

import ast
from typing import Optional, Tuple


class _FilterState:
    """Holds validation flags discovered during AST transforms/visits."""

    def __init__(self) -> None:
        self.has_disallowed_call: bool = False
        self.has_disallowed_slice: bool = False
        self.has_disallowed_attribute: bool = False


class _RewriteGetAndAttributes(ast.NodeTransformer):
    """
    Transform filter expressions into a backend-friendly subset:
    - Rewrite `obj.get("key")` → `obj["key"]` when possible.
    - Rewrite attribute access `obj.key` → `obj["key"]`.
    - Replace all other calls with `None` and flag them for validation to handle upstream.

    This keeps the output expression syntactically valid while allowing a later
    validation pass to decide whether to accept it or to degrade safely.
    """

    def __init__(self, state: _FilterState) -> None:
        self.state = state

    def visit_Attribute(self, node: ast.Attribute):  # type: ignore[override]
        # Rewrite obj.key → obj["key"] to match how nested dict data is stored.
        new_value = self.visit(node.value)
        return ast.Subscript(
            value=new_value,
            slice=ast.Constant(node.attr),
            ctx=ast.Load(),
        )

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        # Allowed special case: obj.get("key") → obj["key"] (ignore default arg)
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                # One positional arg (string key) → rewrite; ignore keywords
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
            # Fall through to generic handling below
            pass

        # Any other call is disallowed in Unify filter expressions – flag and replace with None
        self.state.has_disallowed_call = True
        return ast.Constant(value=None)

    def visit_Subscript(self, node: ast.Subscript):  # type: ignore[override]
        # First, recursively visit children so Attribute/Call rewrites happen
        new_value = self.visit(node.value)
        new_slice = self.visit(node.slice)
        return ast.Subscript(value=new_value, slice=new_slice, ctx=ast.Load())


class _ValidateAllowedSubset(ast.NodeVisitor):
    """
    Validate the rewritten AST belongs to a conservative, backend-supported subset.

    Allowed constructs:
    - BoolOps: and/or; UnaryOp: not
    - Compare ops: ==, !=, <, <=, >, >=, in, not in, is, is not
    - Names, Constants (str/int/float/bool/None)
    - Tuples/Lists of allowed literals
    - Subscript where slice is a string literal (no slicing, no computed indices)
    """

    ALLOWED_COMPARE_OPS = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    def __init__(self, state: _FilterState) -> None:
        self.state = state

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        # Any remaining calls are disallowed
        self.state.has_disallowed_call = True

    def visit_Attribute(self, node: ast.Attribute):  # type: ignore[override]
        # Attributes should have been rewritten already; any left are disallowed
        self.state.has_disallowed_attribute = True

    def visit_Subscript(self, node: ast.Subscript):  # type: ignore[override]
        # Visit children first
        self.generic_visit(node)
        # Only allow subscript with a string literal key
        sl = node.slice
        # Python 3.11+: slice is an expr, not ast.Index
        is_const_str = isinstance(sl, ast.Constant) and isinstance(
            getattr(sl, "value", None),
            str,
        )
        if not is_const_str:
            # Any `a[:10]` or computed index should be flagged
            self.state.has_disallowed_slice = True

    def visit_BinOp(self, node: ast.BinOp):  # type: ignore[override]
        # Arithmetic is not part of our conservative subset
        self.state.has_disallowed_attribute = True

    def visit_UnaryOp(self, node: ast.UnaryOp):  # type: ignore[override]
        if not isinstance(node.op, ast.Not):
            self.state.has_disallowed_attribute = True
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):  # type: ignore[override]
        if not isinstance(node.op, (ast.And, ast.Or)):
            self.state.has_disallowed_attribute = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):  # type: ignore[override]
        for op in node.ops:
            if not isinstance(op, self.ALLOWED_COMPARE_OPS):
                self.state.has_disallowed_attribute = True
        self.generic_visit(node)

    def visit_List(self, node: ast.List):  # type: ignore[override]
        for elt in node.elts:
            if not isinstance(elt, (ast.Constant, ast.Name)):
                self.state.has_disallowed_attribute = True
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):  # type: ignore[override]
        for elt in node.elts:
            if not isinstance(elt, (ast.Constant, ast.Name)):
                self.state.has_disallowed_attribute = True
        self.generic_visit(node)


def normalize_and_validate_filter_expr(
    expr: Optional[str],
) -> Tuple[Optional[str], bool]:
    """
    Normalize and validate a filter expression string for Unify backend.

    Returns (normalized_expr, is_safe). When `is_safe` is False, the caller
    SHOULD avoid submitting the expression to the backend (e.g., degrade to an
    empty result locally) to prevent tool failures.

    Normalizations performed:
    - obj.get("key")          → obj["key"]
    - attribute obj.key        → obj["key"]

    Disallowed constructs (will set is_safe=False):
    - Any function/method calls (after the allowed get() rewrite)
    - Any slicing (e.g., value[:10]) or computed indices
    - Arithmetic or unsupported unary/bool ops
    - Remaining attribute nodes (should have been rewritten)
    """
    if expr is None or not isinstance(expr, str) or not expr.strip():
        return expr, True

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        # Fallback: do not submit malformed expressions to backend
        return expr, False

    state = _FilterState()

    # 1) Rewrite .get() and attribute accesses
    rewriter = _RewriteGetAndAttributes(state)
    try:
        rewritten = rewriter.visit(tree)
    except Exception:
        return expr, False

    ast.fix_missing_locations(rewritten)

    # 2) Validate allowed subset
    validator = _ValidateAllowedSubset(state)
    try:
        validator.visit(rewritten)
    except Exception:
        # Any unexpected traversal error → conservative block
        return expr, False

    is_safe = not (
        state.has_disallowed_call
        or state.has_disallowed_slice
        or state.has_disallowed_attribute
    )

    # 3) Produce normalized string
    try:
        normalized = ast.unparse(rewritten)
    except Exception:
        # As a last resort, fallback to the original
        normalized = expr

    return normalized, is_safe
