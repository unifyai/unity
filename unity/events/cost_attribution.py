"""Per-user cost attribution via ContextVar.

When set, LLM costs are attributed to the listed platform user IDs instead of
the assistant's supervisor.  Falls back to SESSION_DETAILS.user.id when None.

Each asyncio.Task inherits its caller's context, so concurrent act() calls
get isolated attribution automatically.
"""

from contextvars import ContextVar
from typing import Optional

COST_ATTRIBUTION: ContextVar[Optional[list[str]]] = ContextVar(
    "COST_ATTRIBUTION",
    default=None,
)
