import asyncio
import time
from typing import Optional
from contextvars import ContextVar
import string
import secrets

from ...constants import LOGGER


def short_id(length=4):
    alphabet = string.ascii_lowercase + string.digits  # base36
    return "".join(secrets.choice(alphabet) for _ in range(length))


# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])


class TimeoutTimer:
    def __init__(
        self,
        timeout: Optional[int],
        max_steps: Optional[int],
        raise_on_limit: bool,
        client,
    ):
        self._timeout = timeout
        self._client = client
        self._max_steps = max_steps
        self._raise_on_limit = raise_on_limit
        self.reset()

    def remaining_time(self) -> Optional[float]:
        if self._timeout is None:
            return None

        return self._timeout - (time.perf_counter() - self.last_activity_ts)

    def reset(self):
        """Refresh the rolling timeout."""
        self.last_activity_ts = time.perf_counter()
        self.last_msg_count = (
            0 if not self._client.messages else len(self._client.messages)
        )

    def has_exceeded_time(self) -> bool:
        """
        Return whether we exceeded the timeout threshold, raises Exception if raise_on_limit is set
        """
        if self._timeout is None:
            return False

        ret = time.perf_counter() - self.last_activity_ts > self._timeout
        if self._raise_on_limit and ret:
            raise asyncio.TimeoutError(
                f"Loop exceeded {self._timeout}s wall-clock limit",
            )
        return ret

    def has_exceeded_msgs(self) -> bool:
        """
        Return whether we exceeded the messages threshold, raises Exception if raise_on_limit is set
        """
        if self._max_steps is None:
            return False

        ret = len(self._client.messages) >= self._max_steps
        if self._raise_on_limit and ret:
            raise RuntimeError(
                f"Conversation exceeded max_steps={self._max_steps} "
                f"(len(client.messages)={len(self._client.messages)})",
            )
        return ret


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


class LoopLogger:
    def __init__(self, cfg: LoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)
