import json
import queue
import asyncio
from functools import wraps
from task_managers.task_manager import TaskManager
from controller.controller import Controller
from planner.planner import Planner

import unify


def _wrap_sync_method(fn: callable, name: str):

    @wraps(fn)
    def _wrapped(*a, **kw):
        ret = fn(*a, **kw)
        is_put = fn.__name__ in ("put", "put_nowait")
        if is_put or ret is not None:
            unify.log(
                context="Queues",
                queue=name,
                method=fn.__name__,
                content=(
                    json.dumps({"args": a, "kwargs": kw}, indent=4)
                    if is_put
                    else json.dumps(ret, indent=4)
                ),
            )
        return ret

    return _wrapped


def _wrap_async_method(fn, name: str):

    @wraps(fn)
    async def _wrapped(*a, **kw):
        ret = await fn(*a, **kw)
        is_put = fn.__name__ in ("put", "put_nowait")
        if is_put or ret is not None:
            unify.log(
                context="Queues",
                queue=name,
                method=fn.__name__,
                content=(
                    json.dumps({"args": a, "kwargs": kw}, indent=4)
                    if is_put
                    else json.dumps(ret, indent=4)
                ),
            )
        return ret

    return _wrapped


def _log_queue(q):
    if isinstance(q, asyncio.Queue):
        _wrap_method = _wrap_async_method
    elif isinstance(q, queue.Queue):
        _wrap_method = _wrap_sync_method
    else:
        raise Exception(f"Expected queue, but found {type(q)}")
    q.put = _wrap_method(q.put, q.name)
    q.put_nowait = _wrap_method(q.put_nowait, q.name)
    q.get = _wrap_method(q.get, q.name)
    q.get_nowait = _wrap_method(q.get_nowait, q.name)


class BusManager:

    def __init__(self) -> None:

        # Queues #
        # -------#

        # the latest (windowed) user-agent transcript, updated after every new exchange
        self._transcript_q: queue.Queue[list[str]] = queue.Queue()
        self._transcript_q.name = "transcript_q"
        _log_queue(self._transcript_q)

        # low-level browser actions, in text form
        self._text_action_q: queue.Queue[str] = queue.Queue()
        self._text_action_q.name = "text_action_q"
        _log_queue(self._text_action_q)

        # user task requests, in text form
        self._text_task_q: queue.Queue[str] = queue.Queue()
        self._text_task_q.name = "text_task_q"
        _log_queue(self._text_task_q)

        # lower-level browser commands
        self._browser_command_q: queue.Queue[str] = queue.Queue()
        self._browser_command_q.name = "browser_command_q"
        _log_queue(self._browser_command_q)

        # playwright browser state
        self._browser_state_q: queue.Queue[str] = queue.Queue()
        self._browser_state_q.name = "browser_state_q"
        _log_queue(self._browser_state_q)

        # actions which have been completed, referenced by their title
        self._action_completion_q: queue.Queue[str] = queue.Queue()
        self._action_completion_q.name = "action_completion_q"
        _log_queue(self._action_completion_q)

        # tasks which have been completed, referenced by their title
        self._task_completion_q: asyncio.Queue[str] = asyncio.Queue()
        self._task_completion_q.name = "task_completion_q"
        _log_queue(self._task_completion_q)

        # Managers #
        # ---------#

        # re-organizes and schedules task, based on transcripts
        self._task_manager = TaskManager(
            # [reads from]: detect when a task trigger + change is requested from transcript
            self._transcript_q,
            # [writes to]: parses intent from transcript + sends clear text commands
            self._text_task_q,
        )

        # handles hierarchical task planning + decomposition
        self._planner = Planner(
            # [read from]: take high-level text-based tasks and decomposes into low-level text-based actions
            self._text_task_q,
            # [writes to]: send these low-level text-based actions to the controller
            self._text_action_q,
            # [reads from]: determines when the low-level actions are completed
            self._action_completion_q,
            # [writes to]: writes incremental task progress, so the user-facing assistant stays updated
            self._task_completion_q,
        )

        # handles text -> low-level browser commands
        self._controller = Controller(
            # [reads from]: take low-level text-based actions and convert to browser actions
            self._text_action_q,
            # [reads from]: use the browser state as context for text->action controller
            self._browser_state_q,
            # [writes to]: send the browser commands for the browser worker to execute
            self._browser_command_q,
            # [writes to]: sends the name of the completed action, once it is completed
            self._action_completion_q,
        )

    def start(self):
        self._task_manager.start()
        self._planner.start()
        self._controller.start()

    # Properties #
    # -----------#

    @property
    def transcript_q(self) -> queue.Queue[list[str]]:
        return self._transcript_q

    @property
    def task_completion_q(self) -> asyncio.Queue[str]:
        return self._task_completion_q
