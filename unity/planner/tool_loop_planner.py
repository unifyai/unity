import asyncio
import enum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type
import os
import json
import copy
import uuid

from unity.common.llm_helpers import (
    start_async_tool_use_loop,
    AsyncToolLoopHandle,
)
from unity.controller.controller import Controller
from unify import AsyncUnify


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Dummy ComsManager (will be replaced with the real ComsManager)
class ComsManager:
    async def communicate(self, description: str) -> str:
        logging.info(f"Dummy ComsManager.communicate called with: {description}")
        await asyncio.sleep(0.1)
        return f"Communication task initiated for: {description}"


class _PlannerState(enum.Enum):
    IDLE = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()


class ToolLoopPlanner:
    """
    An asyncio, single-task orchestrator that uses start_async_tool_use_loop
    with a defined set of tools (act, observe, communicate).
    It allows for task execution, control (stop, pause, resume),
    and interaction (interject, query, clarification handling).
    """

    def __init__(
        self,
        base_system_prompt: str = "You are a helpful assistant. Use the available tools to complete the user's request. Prioritize completing the primary request.",
    ) -> None:
        self._state: _PlannerState = _PlannerState.IDLE
        self._loop_handle: Optional[AsyncToolLoopHandle] = None
        self._paused_context: Optional[List[dict]] = None
        self._task_id: Optional[str] = None  # To track the current task

        self._controller = Controller()
        self._coms_manager = ComsManager()

        self._base_system_prompt = base_system_prompt
        self._client: Optional[AsyncUnify] = None

        # Clarification queues
        self._clar_up_q: asyncio.Queue[str] = asyncio.Queue()
        self._clar_down_q: asyncio.Queue[str] = asyncio.Queue()

        self._tools: Dict[str, Callable[..., Awaitable[Any]]] = self._build_tools()

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the tool mapping for the async tool loop."""

        async def act_tool(action: str) -> str:
            """Performs a browser action based on a textual description."""
            logger.info(
                f"Planner: Calling Controller.act with '{action}' for task {self._task_id}",
            )
            result = await self._controller.act(action)
            # Controller.act can return Optional[List[str]] or Optional[str]
            if isinstance(result, list):
                res_str = ", ".join(result)
                return (
                    f"Executed actions: {res_str}"
                    if result
                    else "No action taken by controller."
                )
            return (
                f"Executed action: {result}"
                if result
                else "No action taken by controller."
            )

        async def observe_tool(query: str, response_format_str: str = "str") -> str:
            """Observes the current browser state and answers a query.
            response_format_str can be 'str', 'bool', 'int', 'float'."""
            logger.info(
                f"Planner: Calling Controller.observe with query '{query}', format '{response_format_str}' for task {self._task_id}",
            )

            format_map: Dict[str, Type] = {
                "str": str,
                "bool": bool,
                "int": int,
                "float": float,
            }
            actual_response_format = format_map.get(response_format_str.lower(), str)

            result = await self._controller.observe(
                query,
                response_format=actual_response_format,
            )
            return str(result)  # Ensure string return for the LLM

        async def communicate_tool(description: str) -> str:
            """Initiates a communication task (e.g., send email, SMS)."""
            logger.info(
                f"Planner: Calling ComsManager.communicate with '{description}' for task {self._task_id}",
            )
            return await self._coms_manager.communicate(description)

        async def request_clarification_tool(question: str) -> str:
            """Used by the assistant to ask for clarification from the end-user."""
            logger.info(
                f"Planner: LLM requesting clarification for task {self._task_id}: '{question}'",
            )
            await self._clar_up_q.put(question)
            answer = await self._clar_down_q.get()
            logger.info(
                f"Planner: User provided clarification for task {self._task_id}: '{answer}'",
            )
            return answer

        return {
            "act": act_tool,
            "observe": observe_tool,
            "communicate": communicate_tool,
            "request_clarification": request_clarification_tool,
        }

    @property
    def status(self) -> str:
        """Returns the current status of the planner: 'idle', 'running', or 'paused'."""
        return self._state.name.lower()

    @property
    def clarification_questions(self) -> asyncio.Queue[str]:
        """Queue for receiving clarification questions from the planner's internal LLM."""
        return self._clar_up_q

    @staticmethod
    def _drain_queue(q: "asyncio.Queue[Any]") -> None:
        try:
            while True:
                q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def answer_clarification(self, answer: str) -> None:
        """Provides an answer to a pending clarification question from the planner's LLM."""
        # No state check needed here, as an answer might be provided even if the loop was paused
        # after asking the question. The queue mechanism will handle it.
        await self._clar_down_q.put(answer)
        logger.info(
            f"Planner: Clarification answer '{answer}' queued for task {self._task_id}.",
        )

    def _fresh_llm_client(
        self,
        messages_to_load: Optional[List[dict]] = None,
    ) -> AsyncUnify:
        """Creates a new AsyncUnify client."""
        client = AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        if self._base_system_prompt:
            client.set_system_message(self._base_system_prompt)
        if messages_to_load:
            if (
                self._base_system_prompt
                and messages_to_load
                and messages_to_load[0].get("role") == "system"
            ):
                client.append_messages(messages_to_load[1:])
            elif messages_to_load:
                client.append_messages(messages_to_load)
        return client

    def _attach_completion_callback(self, handle: AsyncToolLoopHandle) -> None:
        """Sets up a callback for when the tool loop finishes or is cancelled."""
        current_task_id = (
            self._task_id
        )  # Capture task_id at the time of handle creation

        async def _finalizer():
            try:
                await handle.result()  # Wait for the task to complete or be cancelled
                logger.info(f"Planner: Task {current_task_id} completed successfully.")
            except asyncio.CancelledError:
                logger.info(f"Planner: Task {current_task_id} was cancelled.")
            except Exception as e:
                logger.error(
                    f"Planner: Task {current_task_id} failed with exception: {e}",
                    exc_info=True,
                )
            finally:
                # Only modify state if this finalizer is for the *current* active handle
                if self._loop_handle is handle:
                    # If the state is RUNNING, it means it finished normally or errored, not paused.
                    if self._state == _PlannerState.RUNNING:
                        self._set_state(_PlannerState.IDLE)
                    # If it was PAUSED, pause() itself changed the state, so finalizer doesn't reset to IDLE.
                    self._loop_handle = None
                    self._client = None
                    if self._state == _PlannerState.IDLE:
                        self._task_id = None

        asyncio.create_task(_finalizer())

    def _set_state(self, new_state: _PlannerState) -> None:
        if self._state != new_state:  # Log only actual changes
            logger.info(
                f"Planner: State changing from {self._state.name} to {new_state.name} for task {self._task_id}",
            )
            self._state = new_state

    async def execute(self, task_description: str) -> str:
        """
        Starts a new task execution with the given description.
        Raises RuntimeError if another task is already running.
        Returns the final string result from the tool loop.
        """

        if self._state == _PlannerState.RUNNING:
            raise RuntimeError(
                f"Cannot execute new task. Task {self._task_id} is already running. Stop or pause it first.",
            )

        # Drain any pending clarification questions and answers
        self._drain_queue(self._clar_up_q)
        self._drain_queue(self._clar_down_q)

        self._task_id = str(uuid.uuid4())
        logger.info(f"Planner: Executing new task {self._task_id}: {task_description}")
        self._paused_context = None

        self._client = self._fresh_llm_client()

        self._loop_handle = start_async_tool_use_loop(
            client=self._client,
            message=task_description,
            tools=self._tools,  # Includes request_clarification
            interrupt_llm_with_interjections=True,
            log_steps=False,  # Enable for detailed debugging
        )

        self._attach_completion_callback(self._loop_handle)
        self._set_state(_PlannerState.RUNNING)

        try:
            result_str: str = await self._loop_handle.result()
            return result_str
        except asyncio.CancelledError:
            # This path is taken if stop() or pause() was called before result() completed.
            # The _finalizer callback handles state updates.
            logger.info(
                f"Planner: Task {self._task_id} execution was externally cancelled (likely by stop/pause).",
            )
            if self._state == _PlannerState.PAUSED:  # Check if pause was the cause
                return "Task paused."
            return "Task stopped."
        except Exception as e:
            logger.error(
                f"Planner: Task {self._task_id} execution failed: {e}",
                exc_info=True,
            )
            self._set_state(_PlannerState.IDLE)
            self._task_id = None
            raise  # Re-throw the exception after cleaning up state

    async def stop(self) -> None:
        """Stops the currently running or paused task and resets the planner to idle."""
        if self._state == _PlannerState.IDLE:
            logger.info("Planner: No task to stop, already idle.")
            return

        logger.info(
            f"Planner: Stopping task {self._task_id} (current state: {self.status}).",
        )
        current_handle = self._loop_handle

        self._set_state(_PlannerState.IDLE)  # Tentatively set to IDLE
        self._paused_context = None

        if current_handle:
            current_handle.stop()
            try:
                # Wait for the task associated with the handle to acknowledge cancellation.
                await current_handle._task  # Accessing private _task, common in handle examples
            except asyncio.CancelledError:
                logger.info(
                    f"Planner: Loop for task {self._task_id} acknowledged cancellation by stop().",
                )
            except Exception as e:
                logger.error(
                    f"Planner: Error during stop's await of task {self._task_id}: {e}",
                    exc_info=True,
                )
            finally:
                # Finalizer will also run, ensure consistent state.
                # If the finalizer hasn't cleared _loop_handle yet (e.g. if stop called after finalizer started)
                if self._loop_handle is current_handle:
                    self._loop_handle = None

        self._client = None
        self._task_id = None  # Task is definitively stopped
        logger.info("Planner: Task stopped and reset to idle.")

    async def pause(self) -> None:
        """Pauses the currently running task, saving its context for resumption."""
        if self._state != _PlannerState.RUNNING:
            raise RuntimeError(
                f"Planner: No running task to pause (current state: {self.status}).",
            )
        if not self._loop_handle or not self._client:
            # This should ideally not happen if state is RUNNING
            logger.error(
                "Planner: Inconsistent state during pause - running but no loop_handle or client.",
            )
            self._set_state(_PlannerState.IDLE)  # Reset to a safe state
            return

        logger.info(f"Planner: Pausing task {self._task_id}.")

        self._loop_handle.stop()  # Signal the loop to cancel
        try:
            await self._loop_handle._task  # Wait for cancellation to be processed by the loop
        except asyncio.CancelledError:
            logger.info(f"Planner: Loop for task {self._task_id} cancelled for pause.")
        except Exception as e:
            logger.error(
                f"Planner: Error awaiting loop cancellation for pause of task {self._task_id}: {e}",
                exc_info=True,
            )

        # Capture context after the loop has stopped
        if self._client:
            self._paused_context = copy.deepcopy(self._client.messages)

        self._set_state(_PlannerState.PAUSED)
        self._loop_handle = None
        self._client = None  # Client state is saved in _paused_context
        logger.info(f"Planner: Task {self._task_id} paused.")

    async def resume(self) -> None:
        """Resumes a task that was previously paused."""
        if self._state != _PlannerState.PAUSED:
            raise RuntimeError(
                f"Planner: No paused task to resume (current state: {self.status}).",
            )
        if not self._paused_context:
            # This should not happen if state is PAUSED
            logger.error(
                "Planner: Cannot resume, no paused context found despite PAUSED state.",
            )
            self._set_state(_PlannerState.IDLE)  # Reset to a safe state
            return

        logger.info(f"Planner: Resuming task {self._task_id}.")

        self._client = self._fresh_llm_client(self._paused_context)
        self._loop_handle = start_async_tool_use_loop(
            client=self._client,
            message="Continue with the previous task, considering the history. What is the next step or tool call?",
            tools=self._tools,
            propagate_chat_context=True,
            interrupt_llm_with_interjections=True,
            log_steps=False,  # Enable for detailed debugging
        )

        self._attach_completion_callback(self._loop_handle)
        self._set_state(_PlannerState.RUNNING)
        self._paused_context = None  # Context is now live in the new loop
        logger.info(f"Planner: Task {self._task_id} resumed.")

    async def interject(self, message: str) -> None:
        """Sends an interjection (new user message) to the currently running task."""
        if self._state != _PlannerState.RUNNING:
            raise RuntimeError(
                f"Planner: No running task to interject (current state: {self.status}).",
            )
        if not self._loop_handle:
            # Should not happen if state is RUNNING
            logger.error(
                "Planner: Inconsistent state during interject - running but no loop_handle.",
            )
            return

        logger.info(f"Planner: Interjecting message to task {self._task_id}: {message}")
        await self._loop_handle.interject(message)

    async def query(self, question: str) -> str:
        """
        Sends a query as an interjection to the running task and attempts to
        return the next assistant response that is not a tool call.
        """
        if self._state != _PlannerState.RUNNING:
            raise RuntimeError(
                f"Planner: No running task to query (current state: {self.status}).",
            )
        if not self._loop_handle or not self._client:
            # Should not happen if state is RUNNING
            logger.error(
                "Planner: Inconsistent state during query - running but no loop_handle or client.",
            )
            return "Error: Planner in inconsistent state."

        logger.info(f"Planner: Querying task {self._task_id}: {question}")

        # Snapshot message count *before* interjection
        # start_async_tool_use_loop appends user messages to client.messages when interject is called
        start_idx = len(self._client.messages)
        await self._loop_handle.interject(question)
        # After interject, the new user message is at self._client.messages[start_idx]

        async def _wait_for_reply() -> str:
            timeout_seconds = 30
            poll_interval = 0.2
            end_time = asyncio.get_event_loop().time() + timeout_seconds

            while asyncio.get_event_loop().time() < end_time:
                # Iterate from where the interjected message would be, to the end.
                # The actual assistant reply will be after the interjected user message.
                # So, we look for an assistant message *after* the (new) user message.
                # The interjected message gets added, then LLM processes, then assistant replies.
                # client.messages list grows.
                current_messages = self._client.messages
                for i in range(
                    start_idx + 1,
                    len(current_messages),
                ):  # Look for assistant message after interjection
                    msg = current_messages[i]
                    if (
                        msg.get("role") == "assistant"
                        and msg.get("content")
                        and not msg.get("tool_calls")
                    ):
                        logger.info(
                            f"Planner: Query response for task {self._task_id}: {msg['content']}",
                        )
                        return msg["content"]
                await asyncio.sleep(poll_interval)

            logger.warning(
                f"Planner: Timeout waiting for query response for task {self._task_id}.",
            )
            raise asyncio.TimeoutError(
                f"Planner: Timeout waiting for query response for task {self._task_id}.",
            )

        return await _wait_for_reply()
