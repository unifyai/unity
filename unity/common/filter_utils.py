from __future__ import annotations

from typing import Optional


def normalize_filter_expr(expr: Optional[str]) -> Optional[str]:
    """
    Pass filter expressions through unmodified.

    This function currently serves as a passthrough for filter expressions.
    Any filtering logic or validation will be handled on the client-side
    before reaching the backend.
    """
    return expr
