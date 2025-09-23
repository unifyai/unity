from contextvars import ContextVar
from ..llm_helpers import short_id

# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])


class LoopConfig:
    def __init__(self, loop_id, lineage, parent_lineage):
        self._loop_id = loop_id if loop_id is not None else short_id()
        self._lineage = (
            list(lineage) if lineage is not None else [*parent_lineage, self._loop_id]
        )
        self._label = (
            "->".join(self._lineage) if self._lineage else (self._loop_id or "")
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
