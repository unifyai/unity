import queue
import inspect
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from asyncio import AbstractEventLoop

from off_the_shelf import BrowserAssistant, BrowserController
from task_manager.task_manager import TaskManager
from controller.controller import Controller
from planner.planner import Planner
from helpers import _find_project_frame

import unify
from constants import SESSION_ID, LOGGER

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
                    LOGGER.info(
                        f"\n⬇️ {name}.{fn.__name__}(args={_redacted(a)}, kw={_redacted(kw)}) [⏱️ {t}]) [🗂️ {fpath}] [🧵thread-{threading.get_ident()}]\n",
                    )
                else:
                    LOGGER.info(
                        f"\n⬆️ {name}.{fn.__name__}() -> {_redacted(ret)} [⏱️ {t}]) [🗂️ {fpath}] [🧵thread-{threading.get_ident()}]\n",
                    )
                PRINT_LOCK.release()
            unify.log(
                context="Events",
                session_id=SESSION_ID,
                queue=name,
                method=fn.__name__,
                content={"args": a, "kwargs": kw} if is_put else {"return": ret},
                new=True,
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
                    LOGGER.info(
                        f"\n⬇️ {name}.{fn.__name__}(args={_redacted(a)}, kw={_redacted(kw)}) [⏱️ {t}]) [🗂️ {fpath}] [🧵thread-{threading.get_ident()}]\n",
                    )
                else:
                    LOGGER.info(
                        f"\n⬆️ {name}.{fn.__name__}() -> {_redacted(ret)} [⏱️ {t}]) [🗂️ {fpath}] [🧵thread-{threading.get_ident()}]\n",
                    )
                PRINT_LOCK.release()
            unify.log(
                context="Events",
                session_id=SESSION_ID,
                queue=name,
                method=fn.__name__,
                content={"args": a, "kwargs": kw} if is_put else {"return": ret},
                new=True,
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

    def __init__(self, with_browser_use: bool = False) -> None:
        self._with_browser_use = with_browser_use

    def set_coms_asyncio_loop(self, loop: AbstractEventLoop) -> None:
        self._coms_asyncio_loop = loop

    def _create_managers(self):
        if self._with_browser_use:
            # helps manipulate the browser
            self._browser_assistant = BrowserAssistant()

            # helps stop, pause, resume the browser assistant
            self._browser_controller = BrowserController()

            # Connect controller to assistant
            self._browser_controller.set_browser_assistant(self._browser_assistant)
        else:
            # re-organizes and schedules task, based on transcripts
            self._task_manager = TaskManager()

            # handles hierarchical task planning + decomposition
            self._planner = Planner()

            # handles text -> low-level browser commands
            self._controller = Controller()

    def start(self):
        self._create_managers()
        if self._with_browser_use:
            self._browser_assistant.start()
            self._browser_controller.start()
        else:
            self._task_manager.start()
            self._planner.start()
            self._controller.start()
