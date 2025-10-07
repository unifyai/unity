from dataclasses import dataclass
from typing import Callable, Optional, Dict, Union

# ─────────────────────────────────────────────────────────────────────────────
# 0.  metadata wrapper - lets us attach `max_concurrent` to a tool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolSpec:
    """Wrap the callable with optional metadata (e.g., max_concurrent)."""

    fn: Callable
    max_concurrent: Optional[int] = None  # «None» ⇒ unlimited
    # Hidden per-loop quota: when set, the tool will only be callable
    # `max_total_calls` times within a single async tool-use loop. Once
    # exhausted, the tool is silently hidden from the exposed schema and
    # any additional invocations are minimally acknowledged without
    # revealing quota details to the LLM.
    max_total_calls: Optional[int] = None
    read_only: Optional[bool] = None

    # Let a ToolSpec be invoked like the underlying callable (nice for tests)
    def __call__(self, *a, **kw):  # pragma: no cover
        return self.fn(*a, **kw)


def normalise_tools(
    raw: Dict[str, Union[Callable, "ToolSpec"]],
) -> Dict[str, "ToolSpec"]:
    """Return a uniform ``dict[name → ToolSpec]`` from callables or ToolSpec values."""
    out: Dict[str, ToolSpec] = {}
    for n, v in raw.items():
        if isinstance(v, ToolSpec):
            out[n] = v
        else:
            out[n] = ToolSpec(fn=v, read_only=getattr(v, "_tool_spec_read_only", None))
    return out


def read_only(fn: Callable) -> Callable:
    """Mark a tool as read-only; eligible for semantic-cache re-execution."""
    setattr(fn, "_tool_spec_read_only", True)
    return fn
