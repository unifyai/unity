import asyncio
import traceback

def log_task_exc(task: asyncio.Task) -> None:
    try:
        task.result()          # re-raises if failed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        traceback.print_exc()

class Debouncer:
    def __init__(self):
        self.running_task: asyncio.Task = None
        self.pending_task: asyncio.Task = None
    
    async def submit(self, async_fn, args: tuple=None, kwargs: dict=None, delay=0, cancel_running=False):
        # cancel pending task (debounce) and, optionally, cancel running task as well
        args, kwargs = args or (), kwargs or {}
        await self._cancel_tasks(running=cancel_running)
        # scheduele a new task to run
        async def wait_for_running_task():
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                # this will attempt to wait for any currently running tasks
                # if it was already cancelled because of `cancel_running` will just throw a CancelledError
                # if not then its gonna wait till the currently running task finishes
                if self.running_task and not self.running_task.done():
                    await self.running_task
            except asyncio.CancelledError:
                pass
            # create a running task after delay (if it was not cancelled by a new event being emitted)
            self.running_task = asyncio.create_task(async_fn(*args, **kwargs))
            self.running_task.add_done_callback(log_task_exc)
            self.pending_task = None
        self.pending_task = asyncio.create_task(wait_for_running_task())

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
                