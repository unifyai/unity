import queue
import inspect
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from asyncio import AbstractEventLoop

from task_managers.task_manager import TaskManager
from controller.controller import Controller
from planner.planner import Planner
from helpers import _find_project_frame

import unify
from constants import SESSION_ID

PRINT_LOCK = threading.Lock()


def _redacted(obj):
    if isinstance(obj, dict):
        return {
            k: ("{image}" if k == "screenshot" else _redacted(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redacted(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_redacted(v) for v in obj)
    return obj


def _wrap_sync_method(fn: callable, name: str):

    @wraps(fn)
    def _wrapped(*a, **kw):
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        ret = fn(*a, **kw)
        is_put = fn.__name__ in ("put", "put_nowait")
        if is_put or ret is not None:
            if name != "browser_state_q":  # constantly streaming
                caller_frame = _find_project_frame(inspect.currentframe().f_back)
                if caller_frame is not None:
                    fpath = (
                        f"{Path(caller_frame.f_code.co_filename).resolve()}"
                        f":{caller_frame.f_lineno}"
                    )
                else:
                    fpath = "?"
                PRINT_LOCK.acquire()
                if is_put:
                    print(
                        f"\n⬇️ {name}.{fn.__name__}(args={_redacted(a)}, kw={_redacted(kw)})",
                        f"[⏱️ {t}])",
                        f"[🗂️ {fpath}]",
                        f"[🧵thread-{threading.get_ident()}]\n",
                    )
                else:
                    print(
                        f"\n⬆️ {name}.{fn.__name__}() -> {_redacted(ret)}",
                        f"[⏱️ {t}])",
                        f"[🗂️ {fpath}]",
                        f"[🧵thread-{threading.get_ident()}]\n",
                    )
                PRINT_LOCK.release()
            unify.log(
                context="Events",
                session_id=SESSION_ID,
                queue=name,
                method=fn.__name__,
                content={"args": a, "kwargs": kw} if is_put else {"return": ret},
            )
        return ret

    return _wrapped


def _wrap_async_method(fn, name: str):

    @wraps(fn)
    async def _wrapped(*a, **kw):
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        ret = await fn(*a, **kw)
        is_put = fn.__name__ in ("put", "put_nowait")
        if is_put or ret is not None:
            if name != "browser_state_q":  # constantly streaming
                caller_frame = _find_project_frame(inspect.currentframe().f_back)
                if caller_frame is not None:
                    fpath = (
                        f"{Path(caller_frame.f_code.co_filename).resolve()}"
                        f":{caller_frame.f_lineno}"
                    )
                else:
                    fpath = "?"
                PRINT_LOCK.acquire()
                if is_put:
                    print(
                        f"\n⬇️ {name}.{fn.__name__}(args={_redacted(a)}, kw={_redacted(kw)})",
                        f"[⏱️ {t}])",
                        f"[🗂️ {fpath}]",
                        f"[🧵thread-{threading.get_ident()}]\n",
                    )
                else:
                    print(
                        f"\n⬆️ {name}.{fn.__name__}() -> {_redacted(ret)}",
                        f"[⏱️ {t}])",
                        f"[🗂️ {fpath}]",
                        f"[🧵thread-{threading.get_ident()}]\n",
                    )
                PRINT_LOCK.release()
            unify.log(
                context="Events",
                session_id=SESSION_ID,
                queue=name,
                method=fn.__name__,
                content={"args": a, "kwargs": kw} if is_put else {"return": ret},
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
    q.put_nowait = _wrap_sync_method(q.put_nowait, q.name)
    q.get = _wrap_method(q.get, q.name)
    q.get_nowait = _wrap_sync_method(q.get_nowait, q.name)


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

    def set_coms_asyncio_loop(self, loop: AbstractEventLoop) -> None:
        self._coms_asyncio_loop = loop

    def _create_managers(self):
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
            # enables writing to action_completion_q, which lives in another asyncio loop
            self._coms_asyncio_loop,
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
        self._create_managers()
        self._task_manager.start()
        self._planner.start()
        self._controller.start()

    # Properties #
    # -----------#

    @property
    def transcript_q(self) -> queue.Queue[list[str]]:
        return self._transcript_q

    @property
    def text_action_q(self) -> queue.Queue[str]:
        return self._text_action_q

    @property
    def text_task_q(self) -> queue.Queue[str]:
        return self._text_task_q

    @property
    def browser_command_q(self) -> queue.Queue[str]:
        return self._browser_command_q

    @property
    def browser_state_q(self) -> queue.Queue[str]:
        return self._browser_state_q

    @property
    def action_completion_q(self) -> queue.Queue[str]:
        return self._action_completion_q

    @property
    def task_completion_q(self) -> asyncio.Queue[str]:
        return self._task_completion_q
