from secrets import token_hex
from ..llm_helpers import short_id
from contextvars import ContextVar

# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])

from . import images as _images

# Public re-exports
LIVE_IMAGES_REGISTRY = _images.LIVE_IMAGES_REGISTRY
LIVE_IMAGES_LOG = _images.LIVE_IMAGES_LOG

__all__ = [
    "TOOL_LOOP_LINEAGE",
    "LoopConfig",
    "LIVE_IMAGES_REGISTRY",
    "LIVE_IMAGES_LOG",
]


class LoopConfig:
    def __init__(self, loop_id, lineage, parent_lineage):
        self._loop_id = loop_id if loop_id is not None else short_id()
        self._lineage = (
            list(lineage) if lineage is not None else [*parent_lineage, self._loop_id]
        )
        # Human-friendly label composed from lineage, with a short per-loop suffix
        # e.g. "TaskScheduler.execute->TaskScheduler.ask(x2ab)"
        _base = "->".join(self._lineage) if self._lineage else (self._loop_id or "")
        _suffix = token_hex(2)  # 4 hex chars
        self._label = f"{_base}({_suffix})"

    @property
    def loop_id(self):
        return self._loop_id

    @property
    def lineage(self):
        return self._lineage

    @property
    def label(self):
        return self._label
