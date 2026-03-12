import asyncio

from unity.logger import LOGGER


def log_task_exc(task: asyncio.Task) -> None:
    try:
        task.result()  # re-raises if failed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        LOGGER.exception("Slow-brain task failed: %s", e)


class Debouncer:
    def __init__(self, name: str | None = None):
        self.running_task: asyncio.Task = None
        self.pending_task: asyncio.Task = None
        self._name = name
        self._pending_label: str = ""
        self.was_queued: bool = False
        self.running_task_started_at: float = 0.0
        self.running_task_trace_meta: dict = {}

    async def submit(
        self,
        async_fn,
        args: tuple = None,
        kwargs: dict = None,
        delay=0,
        cancel_running=False,
        label: str = "",
        trace_meta: dict | None = None,
    ):
        args, kwargs = args or (), kwargs or {}

        had_pending = self.pending_task is not None and not self.pending_task.done()
        has_running = self.running_task is not None and not self.running_task.done()
        old_label = self._pending_label

        await self._cancel_tasks(running=cancel_running)

        if self._name and not cancel_running:
            new_type = f"{label} " if label else ""
            if had_pending and has_running:
                old_type = f" (replacing {old_label})" if old_label else ""
                LOGGER.info(
                    f"🚦 [{self._name}] {new_type}request queued{old_type}",
                )
            elif has_running:
                LOGGER.info(
                    f"🚦 [{self._name}] {new_type}request queued",
                )

        async def wait_for_running_task():
            if delay > 0:
                await asyncio.sleep(delay)
            queued = self.running_task is not None and not self.running_task.done()
            try:
                # Wait for any currently running task to complete.
                # Use asyncio.shield() to protect the running task from being
                # cancelled if THIS pending task is cancelled. In Python 3.11+,
                # cancelling a task that awaits another task will also cancel
                # the inner task - shield() prevents this propagation.
                if self.running_task and not self.running_task.done():
                    await asyncio.shield(self.running_task)
            except asyncio.CancelledError:
                # CancelledError can come from two sources:
                # 1. The running task was cancelled (e.g., cancel_running=True)
                # 2. This pending task was cancelled (debounced by a newer submit)
                #
                # In case 1, we should proceed to create a new running task.
                # In case 2, we should NOT proceed - let the newer pending task handle it.
                if self.running_task and self.running_task.cancelled():
                    # Running task was cancelled, proceed to create new one
                    pass
                else:
                    # We (the pending task) were cancelled, re-raise to stop
                    raise
            self.was_queued = queued
            self.running_task_started_at = asyncio.get_event_loop().time()
            self.running_task_trace_meta = trace_meta or {}
            self.running_task = asyncio.create_task(async_fn(*args, **kwargs))
            self.running_task.add_done_callback(log_task_exc)
            self.pending_task = None
            self._pending_label = ""

        self.pending_task = asyncio.create_task(wait_for_running_task())
        self._pending_label = label

    async def _cancel_tasks(self, pending=True, running=False):
        if running:
            if self.running_task and not self.running_task.done():
                self.running_task.cancel()
                try:
                    await self.running_task
                except asyncio.CancelledError:
                    pass
        if pending:
            if self.pending_task and not self.pending_task.done():
                self.pending_task.cancel()
                try:
                    await self.pending_task
                except asyncio.CancelledError:
                    pass
