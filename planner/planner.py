import queue
import asyncio
import threading
import time
from typing import List, Optional, Any
from asyncio import AbstractEventLoop

from .context import context as planner_context
from . import zero_shot, update_handler
from .verifier import Verifier
from .code_rewriter import CodeRewriter


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
        self.plan_stack = []

    def run(self) -> None:
        while True:
            # Priority 0: Check for reimplement requests from Verifier (highest priority)
            try:
                reimplement_queue = Verifier.get_reimplement_queue()
                function_name = reimplement_queue.get_nowait()

                # Pause the plan execution
                self.paused = True

                # Find the function node by name in the current plan
                if self.current_plan:
                    function_node = self.current_plan.get_function_by_name(
                        function_name
                    )
                    if function_node:
                        # Rewrite the plan section using CodeRewriter
                        CodeRewriter.rewrite_plan_section(function_node)
                        if self.current_plan:
                            self.current_plan._expected_primitive = None

                # Clear the queue entry by consuming it (already done with get_nowait)
                # Unpause the plan
                self.paused = False
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

            # Priority 2: Check for action completions (non-blocking)
            try:
                action_string = self._action_completion_q.get(block=False)
                if self.current_plan:
                    # Pass the raw action string to mark_action_done
                    self.current_plan.mark_action_done(action_string)
            except queue.Empty:
                pass

            # Priority 3: If not paused and plan is ready, send next action
            if not self.paused and self.current_plan and self.current_plan.ready():
                next_action = self.current_plan.next_action()
                if next_action:
                    # Send the action string to the action queue
                    self._text_action_q.put(next_action)
                else:
                    # Plan is complete
                    self._coms_asyncio_loop.call_soon_threadsafe(
                        self._task_completion_q.put_nowait,
                        "Task completed successfully",
                    )

                    # Reset paused state
                    self.paused = False

                    # Pop from plan stack if available
                    if self.plan_stack:
                        self.current_plan = self.plan_stack.pop()
                        planner_context.set_current_plan(self.current_plan)
                        self.current_plan._expected_primitive = None
                    else:
                        self.current_plan = None
                        planner_context.set_current_plan(None)
                        self.paused = False

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

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

        Args:
            task_description: The description of the task to plan for
        """
        # Create a new plan using zero-shot planning
        self.current_plan = zero_shot.create_initial_plan(task_description)

        # Set up queues for primitives to block on
        from .primitives import set_queues

        set_queues(self._text_action_q, self._action_completion_q)

        # Set the task completion queue
        self.current_plan.task_completion_q = self._task_completion_q

        # Update the current plan in PlannerContext
        planner_context.set_current_plan(self.current_plan)

        # Ensure planner is not paused
        self.paused = False
