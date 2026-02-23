from secrets import token_hex
from ..llm_helpers import short_id
from contextvars import ContextVar

# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])

# Bridge for suffix consistency: when a decorator (log_manager_call,
# log_manager_result) or boundary wrapper generates a suffix for the
# ManagerMethod incoming event, it stores that suffix here *before*
# spawning the tool loop.  LoopConfig.__init__ consumes it (and resets
# to None) so that the loop's hierarchy_label carries the same suffix
# as the boundary event.  Without this, two independent token_hex(2)
# calls would produce different suffixes for the same operation.
_PENDING_LOOP_SUFFIX: ContextVar[str | None] = ContextVar(
    "_PENDING_LOOP_SUFFIX",
    default=None,
)

from . import images as _images

# Public re-exports
LIVE_IMAGES_REGISTRY = _images.LIVE_IMAGES_REGISTRY
LIVE_IMAGES_LOG = _images.LIVE_IMAGES_LOG

__all__ = [
    "TOOL_LOOP_LINEAGE",
    "_PENDING_LOOP_SUFFIX",
    "LoopConfig",
    "LIVE_IMAGES_REGISTRY",
    "LIVE_IMAGES_LOG",
]


class LoopConfig:
    def __init__(self, loop_id, lineage, parent_lineage):
        self._loop_id = loop_id if loop_id is not None else short_id()
        # Consume _PENDING_LOOP_SUFFIX if a decorator/boundary set one for us;
        # otherwise generate a fresh random suffix.
        _pending = _PENDING_LOOP_SUFFIX.get(None)
        _suffix = _pending if _pending else token_hex(2)
        # Embed the suffix into the leaf segment so the hierarchy array is
        # self-describing for unique parent-child linking (every segment
        # carries per-invocation identity).
        if lineage is not None:
            self._lineage = list(lineage)
            if self._lineage:
                self._lineage[-1] = f"{self._lineage[-1]}({_suffix})"
        else:
            self._lineage = [*parent_lineage, f"{self._loop_id}({_suffix})"]
        # hierarchy_label is now trivially derived — no separate suffix logic.
        # TODO: remove hierarchy_label from payloads once frontend migrates.
        self._label = (
            "->".join(self._lineage) if self._lineage else f"{self._loop_id}({_suffix})"
        )

    @property
    def loop_id(self):
        return self._loop_id

    @property
    def lineage(self):
        return self._lineage

    @property
    def label(self):
        return self._label
