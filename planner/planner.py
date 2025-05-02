import queue
import asyncio
import threading
import time
from typing import List, Optional, Any, Callable
from asyncio import AbstractEventLoop

from .context import context as planner_context
from . import zero_shot, update_handler
from .verifier import Verifier
from .code_rewriter import rewrite_function
from .primitives import set_runtime_controls
import logging


class Planner(threading.Thread):

    def __init__(
        self,
        task_update_queue: "queue.Queue[str]",
        text_action_q: "queue.Queue[str]",
        action_completion_q: "queue.Queue[str]",
        task_completion_q: asyncio.Queue[str],
        coms_asyncio_loop: AbstractEventLoop,
        browser_state_broadcast_q: Optional["queue.Queue[Any]"] = None,
        *,
        daemon: bool = True,
    ) -> None:
        """
        Receives a stream of user inputs related to this task (can either be high-level or low-level guidance), and must stream a series of low-level actions to the controller, as quickly and efficiently as possible, in order to complete the task.

        Args:
            task_update_queue (queue.Queue[List[str]]): Where the text-based user updates for the task come from, as well as the initial task request.
            text_action_q (queue.Queue[List[str]]): Where the low-level text actions are sent.
            action_completion_q (queue.Queue[List[str]]): Where the completion status of the low-level text actions come from.
            task_completion_q (asyncio.Queue[str]): Where we inform the user that the *overall* task is complete.
            coms_asyncio_loop (AbstractEventLoop): The asyncio loop for the user-facing agent. Need for task_completion_q.
            browser_state_broadcast_q (queue.Queue[Any], optional): Queue containing browser state snapshots.
        """
        super().__init__(daemon=daemon)
        self._text_task_q = task_update_queue
        self._text_action_q = text_action_q
        self._action_completion_q = action_completion_q
        self._task_completion_q = task_completion_q
        self._coms_asyncio_loop = coms_asyncio_loop

        # Set up PlannerContext with browser state broadcast queue
        if browser_state_broadcast_q:
            planner_context.set_broadcast_queue(browser_state_broadcast_q)

        # New instance variables for plan management
        self.current_plan = None
        self.paused = False

        self._plan_module = None
        self._plan_thread = None

        # Runtime control mechanisms
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._call_queue = queue.Queue()

    def _run_plan_wrapper(self, fn):
        """
        Wrapper function that manages call stack and executes the plan function.

        Args:
            fn: The root function of the plan to execute
        """
        try:
            planner_context.push_frame(fn.__name__)
            fn()
            status = "Task completed successfully"
        except Exception as exc:
            logging.exception("Plan crashed in %s", fn.__name__)
            status = f"Task crashed: {exc!r}"
        finally:
            planner_context.pop_frame()

        self._coms_asyncio_loop.call_soon_threadsafe(
            self._task_completion_q.put_nowait, status
        )

    def run(self) -> None:
        while True:
            # Priority 0: Check for reimplement requests from Verifier (highest priority)
            try:
                reimplement_queue = Verifier.get_reimplement_queue()
                fn_obj = reimplement_queue.get_nowait()

                # Pause the plan execution
                self._pause()

                # backward‐compatible: if a name was enqueued
                if isinstance(fn_obj, str):
                    fn_obj = getattr(self._plan_module, fn_obj, None)
                if callable(fn_obj):
                    rewrite_function(fn_obj)

                # Unpause the plan
                self._resume()
            except queue.Empty:
                pass

            # Priority 1: Check for new task events (non-blocking)
            try:
                task_description = self._text_task_q.get(block=False)
                if task_description is None:
                    break
                # Handle payloads that may come as (meta, description)
                if isinstance(task_description, tuple):
                    task_description = task_description[1]
                self._handle_task_event(task_description)
            except queue.Empty:
                pass

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

    def _pause(self):
        """
        Pause plan execution by clearing the pause event.
        Primitives will block until the event is set again.
        """
        self._pause_event.clear()
        self.paused = True

    def _resume(self):
        """
        Resume plan execution by setting the pause event.
        This allows blocked primitives to continue execution.
        """
        self._pause_event.set()
        self.paused = False

    def _handle_task_event(self, task_description):
        """
        Handle incoming task events from the task update queue.
        Determines whether to start a new plan or update existing plan.

        Args:
            task_description: The description of the task to plan for
        """
        if self.current_plan is None:
            # If no plan exists, start a new one
            self._start_new_plan(task_description)
        else:
            # If a plan exists, handle as an update
            update_handler.handle_update(self, task_description)

    def _start_new_plan(self, task_description):
        """
        Start a new plan based on the task description.
        Uses zero-shot planning to generate a sequence of actions.
        Launches a daemon thread to execute the plan.

        Args:
            task_description: The description of the task to plan for
        """
        # Set up queues for primitives to block on (do this only once)
        from .primitives import set_queues

        set_queues(self._text_action_q, self._action_completion_q)

        # Set up runtime control mechanisms
        set_runtime_controls(self._pause_event, self._call_queue)

        # Create a new plan using zero-shot planning
        module, root_fn = zero_shot.create_initial_plan(task_description)
        self._plan_module = module
        self.current_plan = module
        self._root_fn = root_fn

        # Update the current plan in PlannerContext
        planner_context.set_current_plan(self.current_plan)

        # Ensure planner is not paused
        self.paused = False

        # Launch a daemon thread to execute the plan
        self._plan_thread = threading.Thread(
            target=self._run_plan_wrapper, args=(self._root_fn,), daemon=True
        )
        self._plan_thread.start()
