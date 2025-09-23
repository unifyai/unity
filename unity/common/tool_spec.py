from dataclasses import dataclass
from typing import Callable, Optional, Dict, Union

# ─────────────────────────────────────────────────────────────────────────────
# 0.  metadata wrapper - lets us attach `max_concurrent` to a tool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolSpec:
    """
    Wrap the real *callable* together with optional metadata.

    Only ``max_concurrent`` is required today but we deliberately keep this
    extensible – adding cost caps, rate limits, auth scopes, … later will not
    change any external API.
    """

    fn: Callable
    max_concurrent: Optional[int] = None  # «None» ⇒ unlimited
    # Hidden per-loop quota: when set, the tool will only be callable
    # `max_total_calls` times within a single async tool-use loop. Once
    # exhausted, the tool is silently hidden from the exposed schema and
    # any additional invocations are minimally acknowledged without
    # revealing quota details to the LLM.
    max_total_calls: Optional[int] = None

    # Let a ToolSpec be invoked like the underlying callable (nice for tests)
    def __call__(self, *a, **kw):  # pragma: no cover
        return self.fn(*a, **kw)


def normalise_tools(
    raw: Dict[str, Union[Callable, "ToolSpec"]],
) -> Dict[str, "ToolSpec"]:
    """
    Accept the *legacy* ``dict[name → callable]`` or the new
    ``dict[name → ToolSpec]`` and always return a *uniform*
    ``dict[name → ToolSpec]``.
    """
    out: Dict[str, ToolSpec] = {}
    for n, v in raw.items():
        out[n] = v if isinstance(v, ToolSpec) else ToolSpec(fn=v)
    return out
