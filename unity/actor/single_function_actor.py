"""
A minimal actor that executes a single function or primitive.

This actor is useful for:
- Testing that stored functions work correctly
- Deploying rigid, pre-defined workflows with no interactive elements
- Integration testing of the function/computer_primitives pipeline
- Executing action primitives (state manager methods) directly

The actor can execute either user-defined functions from the FunctionManager
or action primitives (like ContactManager.ask, TaskScheduler.execute, etc.).

Supports optional verification via LLM to check if the function achieved its goal.

Steerable Function Support
--------------------------
If the executed function returns a SteerableHandle (e.g., from CodeActActor.act()
or start_async_tool_loop()), the handle is detected and all steering operations
(interject, stop, pause, resume, ask) are forwarded to the inner handle. This
enables compositional functions that wrap other actors to be fully steerable.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel, Field

from unity.common.async_tool_loop import (
    SteerableHandle,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.common.llm_client import new_llm_client
from unity.function_manager.execution_env import create_execution_globals
from unity.manager_registry import ManagerRegistry
from unity.function_manager.primitives import get_primitive_callable

from unity.function_manager.primitives import ComputerPrimitives
from .base import BaseActor, BaseActorHandle

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unity.function_manager.function_manager import FunctionManager


class SingleFunctionVerificationResult(BaseModel):
    """Structured output for single function verification."""

    success: bool = Field(
        ...,
        description="True if the function appears to have achieved its goal based on the return value and context.",
    )
    reason: str = Field(
        ...,
        description="A concise explanation of why verification succeeded or failed.",
    )


class SingleFunctionActorHandle(BaseActorHandle):
    """
    A handle for a single function execution with steerable forwarding support.

    This handle provides the standard steerable interface. For non-steerable
    functions, pause/resume/interject are no-ops. For steerable functions
    (those that return a SteerableHandle), all operations are forwarded to
    the inner handle, enabling full steering support.

    Steerable Detection
    -------------------
    When the executed function returns a SteerableHandle (detected via isinstance),
    this handle becomes a transparent forwarder - all steering operations are
    delegated to the inner handle. This enables compositional functions that
    wrap CodeActActor or other steerable actors to be fully steerable.
    """

    def __init__(
        self,
        function_name: str,
        function_id: Optional[int],
        execution_task: asyncio.Task,
        is_primitive: bool = False,
        verify: bool = False,
        goal: Optional[str] = None,
        docstring: Optional[str] = None,
        actor: Optional["SingleFunctionActor"] = None,
    ):
        self._function_name = function_name
        self._function_id = function_id
        self._is_primitive = is_primitive
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._error_str: Optional[str] = None
        self._stopped = False
        self._verify = verify
        self._goal = goal
        self._docstring = docstring
        self._actor = actor
        self._verification_passed: Optional[bool] = None
        self._verification_reason: Optional[str] = None

        # Steerable forwarding support
        self._inner_handle: Optional[SteerableHandle] = None
        self._function_returned = asyncio.Event()

        # Start monitoring the task
        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self):
        """Monitor the execution task, detect steerable handles, and manage completion."""
        try:
            result = await self._execution_task

            # Check if the result is a steerable handle
            if isinstance(result, SteerableHandle):
                # Store the inner handle for forwarding
                self._inner_handle = result
                logger.info(
                    f"Function '{self._function_name}' returned steerable handle, "
                    f"forwarding operations to inner handle.",
                )
                # DO NOT set completion_event - the inner handle manages completion
                # The function_returned event signals that we have the inner handle
                return

            # Non-steerable: process result as before
            if not self._stopped:
                result_str = (
                    str(result)
                    if result is not None
                    else "Function completed successfully."
                )

                # Run verification if enabled (only for non-steerable)
                if self._verify and self._actor is not None:
                    try:
                        verification = await self._actor._verify_execution(
                            function_name=self._function_name,
                            goal=self._goal,
                            docstring=self._docstring,
                            return_value=result,
                        )
                        self._verification_passed = verification.success
                        self._verification_reason = verification.reason

                        if not verification.success:
                            self._error_str = (
                                f"Verification failed: {verification.reason}"
                            )
                            self._result_str = (
                                f"Function '{self._function_name}' executed but verification failed: "
                                f"{verification.reason}"
                            )
                            logger.warning(
                                f"Verification failed for '{self._function_name}': {verification.reason}",
                            )
                        else:
                            self._result_str = result_str
                            logger.info(
                                f"Verification passed for '{self._function_name}': {verification.reason}",
                            )
                    except Exception as e:
                        logger.error(
                            f"Verification error for '{self._function_name}': {e}",
                        )
                        self._error_str = f"Verification error: {e}"
                        self._result_str = f"Function '{self._function_name}' executed but verification errored: {e}"
                else:
                    self._result_str = result_str

            # Non-steerable: mark as complete
            self._completion_event.set()

        except asyncio.CancelledError:
            self._result_str = f"Function '{self._function_name}' was cancelled."
            self._stopped = True
            self._completion_event.set()
        except Exception as e:
            self._error_str = str(e)
            self._result_str = f"Function '{self._function_name}' failed: {e}"
            self._completion_event.set()
        finally:
            # Always signal that the function call has returned
            self._function_returned.set()

    async def result(self) -> str:
        # Wait for the function to return first
        await self._function_returned.wait()

        # If steerable, forward to inner handle
        if self._inner_handle is not None:
            return await self._inner_handle.result()

        # Non-steerable: wait for completion and return stored result
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return self._result_str or f"Function '{self._function_name}' completed."

    def done(self) -> bool:
        # If we have an inner handle, delegate to it
        if self._inner_handle is not None:
            return self._inner_handle.done()
        # Non-steerable: check completion event
        return self._completion_event.is_set()

    @functools.wraps(SteerableToolHandle.stop, updated=())
    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """Stop the function execution. Forwards to inner handle if steerable."""
        # If we already have an inner handle, forward to it
        if self._inner_handle is not None:
            await self._inner_handle.stop(reason)
            return

        # If function hasn't returned yet, cancel the task
        if not self._function_returned.is_set():
            self._stopped = True
            self._execution_task.cancel()

            # Wait for cancellation to complete
            try:
                await asyncio.wait_for(self._function_returned.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

            self._result_str = (
                f"Function '{self._function_name}' stopped."
                if not reason
                else f"Function '{self._function_name}' stopped: {reason}"
            )

    @functools.wraps(SteerableToolHandle.pause, updated=())
    async def pause(self) -> Optional[str]:
        """Pause the execution. Forwards to inner handle if steerable."""
        # Wait for function to return so we know if it's steerable
        await self._function_returned.wait()

        if self._inner_handle is not None:
            return await self._inner_handle.pause()

        # Non-steerable: no-op
        return f"Pause acknowledged (no effect on non-steerable function '{self._function_name}')."

    @functools.wraps(SteerableToolHandle.resume, updated=())
    async def resume(self) -> Optional[str]:
        """Resume the execution. Forwards to inner handle if steerable."""
        # Wait for function to return so we know if it's steerable
        await self._function_returned.wait()

        if self._inner_handle is not None:
            return await self._inner_handle.resume()

        # Non-steerable: no-op
        return f"Resume acknowledged (no effect on non-steerable function '{self._function_name}')."

    @functools.wraps(SteerableHandle.interject, updated=())
    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Interject into the execution. Forwards to inner handle if steerable."""
        # Wait for function to return so we know if it's steerable
        await self._function_returned.wait()

        # If steerable, forward to inner handle
        if self._inner_handle is not None:
            await self._inner_handle.interject(
                message,
                _parent_chat_context_cont=_parent_chat_context_cont,
            )

    @functools.wraps(SteerableHandle.ask, updated=())
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableHandle:
        """Query the status. Forwards to inner handle if steerable."""
        # Wait for function to return so we know if it's steerable
        await self._function_returned.wait()

        # If steerable, forward to inner handle
        if self._inner_handle is not None:
            return await self._inner_handle.ask(
                question,
                _parent_chat_context=_parent_chat_context,
            )

        # Non-steerable: return status-reporting loop
        client = new_llm_client()
        id_info = "(primitive)" if self._is_primitive else f"(ID: {self._function_id})"
        client.set_system_message(
            f"You are reporting on the status of a function execution. "
            f"The function '{self._function_name}' {id_info} is currently running. "
            f"Respond briefly to the user's question.",
        )

        status = "completed" if self.done() else "still running"
        client.append_messages(
            [
                {
                    "role": "user",
                    "content": f"Status: Function is {status}. Question: {question}",
                },
            ],
        )

        return start_async_tool_loop(
            client=client,
            message=question,
            tools={},
            loop_id=f"SingleFunctionAsk({self._function_name})",
            max_consecutive_failures=1,
            timeout=30,
        )

    # Event APIs

    async def next_clarification(self) -> dict:
        """Get next clarification. Forwards to inner handle if steerable."""
        await self._function_returned.wait()

        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "next_clarification",
        ):
            return await self._inner_handle.next_clarification()

        # Non-steerable: no clarifications
        await asyncio.Event().wait()  # Wait forever
        return {}

    async def next_notification(self) -> dict:
        """Get next notification. Forwards to inner handle if steerable."""
        await self._function_returned.wait()

        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "next_notification",
        ):
            return await self._inner_handle.next_notification()

        # Non-steerable: no notifications
        await asyncio.Event().wait()  # Wait forever
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Answer a clarification. Forwards to inner handle if steerable."""
        await self._function_returned.wait()

        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "answer_clarification",
        ):
            return await self._inner_handle.answer_clarification(call_id, answer)

        # Non-steerable: no clarifications to answer
        return None

    def get_history(self) -> list[dict]:
        """Get conversation history. Forwards to inner handle if steerable."""
        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "get_history",
        ):
            return self._inner_handle.get_history()
        return []

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "clarification_up_q",
        ):
            return self._inner_handle.clarification_up_q
        return None

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "clarification_down_q",
        ):
            return self._inner_handle.clarification_down_q
        return None

    @property
    def is_steerable(self) -> bool:
        """Return True if this handle wraps a steerable inner handle."""
        return self._inner_handle is not None

    @property
    def inner_handle(self) -> Optional[SteerableHandle]:
        """Return the inner steerable handle if available."""
        return self._inner_handle


class SingleFunctionActor(BaseActor):
    """
    A minimal actor that executes a single function or primitive.

    This actor is designed for:
    - Testing stored functions
    - Deploying rigid, pre-defined workflows
    - Cases where interactive steering is not needed
    - Executing action primitives (state manager methods) directly

    The actor finds and executes a single function or primitive, either by
    explicit ID/name or by semantic search matching the description.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        function_manager: Optional["FunctionManager"] = None,
        headless: bool = True,
        computer_mode: str = "magnitude",
        agent_mode: str = "web",
        agent_server_url: str | None = None,
    ):
        """
        Initialize the SingleFunctionActor.

        Args:
            computer_primitives: Optional existing ComputerPrimitives. If not provided,
                           one will be created with the given parameters.
            function_manager: Optional FunctionManager instance. If not provided,
                            uses the singleton.
            headless: Whether to run in headless mode.
            computer_mode: Computer backend mode ("magnitude" or "mock").
            agent_mode: Agent mode for ComputerPrimitives ("web" or "desktop").
            agent_server_url: URL for the agent server. For desktop mode, pass the
                external VM's URL.
        """
        if computer_primitives is not None:
            self._computer_primitives = computer_primitives
        else:
            self._computer_primitives = ComputerPrimitives(
                headless=headless,
                computer_mode=computer_mode,
                agent_mode=agent_mode,
                agent_server_url=agent_server_url,
            )

        self._function_manager = (
            function_manager or ManagerRegistry.get_function_manager()
        )

    def _get_function_by_id(
        self,
        function_id: int,
        namespace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get a user-defined function by its ID, injecting dependencies into namespace."""
        result = self._function_manager.filter_functions(
            filter=f"function_id == {function_id}",
            include_implementations=True,
            return_callable=True,
            namespace=namespace,
            also_return_metadata=True,
        )
        metadata = result.get("metadata", [])
        if not metadata:
            raise ValueError(f"No function found with ID {function_id}")
        return metadata[0]

    def _get_primitive_by_name(self, primitive_name: str) -> Dict[str, Any]:
        """Get a primitive by its qualified name (e.g., 'ContactManager.ask')."""
        self._function_manager.sync_primitives()
        primitives = self._function_manager.list_primitives()
        if primitive_name not in primitives:
            raise ValueError(f"No primitive found with name '{primitive_name}'")
        return primitives[primitive_name]

    def _search_function(
        self,
        description: str,
        namespace: Dict[str, Any],
        include_primitives: bool = True,
    ) -> Dict[str, Any]:
        """Search for the best matching function, injecting dependencies into namespace."""
        result = self._function_manager.search_functions(
            query=description,
            n=1,
            include_primitives=include_primitives,
            return_callable=True,
            namespace=namespace,
            also_return_metadata=True,
        )
        metadata = result.get("metadata", [])
        if not metadata:
            raise ValueError(f"No function found matching description: {description}")
        return metadata[0]

    def _create_execution_globals(self) -> Dict[str, Any]:
        """Create the globals dict for function execution."""
        globals_dict = create_execution_globals()
        globals_dict["computer_primitives"] = self._computer_primitives
        return globals_dict

    async def _execute_primitive(
        self,
        primitive_data: Dict[str, Any],
        **call_kwargs,
    ) -> Any:
        """Execute a primitive (state manager method)."""
        name = primitive_data.get("name")

        fn = get_primitive_callable(primitive_data, self._computer_primitives)
        if fn is None:
            raise ValueError(f"Could not resolve primitive '{name}' to a callable")

        # Call the primitive
        if inspect.iscoroutinefunction(fn):
            return await fn(**call_kwargs)
        else:
            return fn(**call_kwargs)

    async def _execute_function(
        self,
        function_data: Dict[str, Any],
        namespace: Dict[str, Any],
        **call_kwargs,
    ) -> Any:
        """Execute a user-defined function."""
        implementation = function_data.get("implementation")
        name = function_data.get("name")
        venv_id = function_data.get("venv_id")

        if not implementation:
            raise ValueError(f"Function '{name}' has no implementation")

        if venv_id is not None:
            return await self._execute_in_custom_venv(
                implementation=implementation,
                name=name,
                venv_id=venv_id,
                call_kwargs=call_kwargs,
            )

        return await self._execute_in_process(
            implementation=implementation,
            name=name,
            namespace=namespace,
            call_kwargs=call_kwargs,
        )

    async def _execute_in_process(
        self,
        implementation: str,
        name: str,
        namespace: Dict[str, Any],
        call_kwargs: Dict[str, Any],
    ) -> Any:
        """Execute a function in the current process."""
        exec(implementation, namespace)

        short_name = name.split(".")[-1] if "." in name else name
        fn = namespace.get(short_name)
        if fn is None:
            raise ValueError(f"Function '{short_name}' not found after execution")

        if inspect.iscoroutinefunction(fn):
            return await fn(**call_kwargs)
        else:
            return fn(**call_kwargs)

    async def _execute_in_custom_venv(
        self,
        implementation: str,
        name: str,
        venv_id: int,
        call_kwargs: Dict[str, Any],
    ) -> Any:
        """Execute a function in a custom virtual environment subprocess.

        The subprocess has access to both `primitives` and `computer_primitives`
        via RPC calls back to the main process.
        """
        logger.info(
            f"Executing function '{name}' in custom venv (ID: {venv_id})",
        )

        # Determine if the function is async by checking the implementation
        is_async = "async def" in implementation

        # Get primitives for RPC access
        from unity.function_manager.primitives import Primitives

        primitives = Primitives()

        # Execute in the custom venv with RPC support
        result = await self._function_manager.execute_in_venv(
            venv_id=venv_id,
            implementation=implementation,
            call_kwargs=call_kwargs,
            is_async=is_async,
            primitives=primitives,
            computer_primitives=self._computer_primitives,
        )

        # Log any captured output
        if result.get("stdout"):
            logger.debug(f"Function '{name}' stdout: {result['stdout']}")
        if result.get("stderr"):
            logger.warning(f"Function '{name}' stderr: {result['stderr']}")

        # Handle errors
        if result.get("error"):
            raise RuntimeError(
                f"Function '{name}' failed in venv {venv_id}: {result['error']}",
            )

        return result.get("result")

    async def _verify_execution(
        self,
        function_name: str,
        goal: Optional[str],
        docstring: Optional[str],
        return_value: Any,
    ) -> SingleFunctionVerificationResult:
        """
        Use an LLM to verify that a function execution achieved its goal.

        Args:
            function_name: The name of the function that was executed.
            goal: The high-level goal/description for the execution.
            docstring: The function's docstring describing what it should do.
            return_value: The value returned by the function.

        Returns:
            A SingleFunctionVerificationResult indicating success/failure.
        """
        client = new_llm_client()
        client.set_system_message(
            "You are a verification assistant. Your job is to determine whether a function "
            "execution succeeded based on its return value, goal, and docstring. "
            "Be pragmatic: if the return value indicates the function completed its task, "
            "mark it as successful. Only mark as failed if there's clear evidence of failure "
            "(e.g., error messages, None when a value was expected, explicit failure indicators).",
        )

        # Build verification prompt
        prompt_parts = [f"## Function: `{function_name}`"]

        if goal:
            prompt_parts.append(f"\n## Goal\n{goal}")

        if docstring:
            prompt_parts.append(f"\n## Docstring\n{docstring}")

        prompt_parts.append(f"\n## Return Value\n```\n{repr(return_value)}\n```")

        prompt_parts.append(
            "\n## Task\n"
            "Based on the above information, determine if the function achieved its goal. "
            "Respond with your assessment.",
        )

        verification_prompt = "\n".join(prompt_parts)

        try:
            # Use start_async_tool_loop with response_format for structured output
            handle = start_async_tool_loop(
                client=client,
                message=verification_prompt,
                tools={},
                loop_id=f"SingleFunctionVerify({function_name})",
                max_consecutive_failures=1,
                timeout=60,
                response_format=SingleFunctionVerificationResult,
            )
            result_json = await handle.result()
            # Parse the JSON response into the Pydantic model
            import json

            result_dict = json.loads(result_json)
            return SingleFunctionVerificationResult.model_validate(result_dict)
        except Exception as e:
            logger.error(f"Verification LLM call failed: {e}")
            # Default to success if verification itself fails (don't block execution)
            return SingleFunctionVerificationResult(
                success=True,
                reason=f"Verification skipped due to error: {e}",
            )

    async def act(
        self,
        description: Optional[str] = None,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        function_id: Optional[int] = None,
        primitive_name: Optional[str] = None,
        include_primitives: bool = True,
        call_kwargs: Optional[Dict[str, Any]] = None,
        verify: Optional[bool] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ) -> SingleFunctionActorHandle:
        """
        Execute a single function or primitive.

        Args:
            description: Natural language description of what to do.
                        Used for semantic search if function_id/primitive_name not provided.
                        Required when neither function_id nor primitive_name is specified.
            function_id: Optional explicit function ID to execute.
                        If provided, skips the search step.
            primitive_name: Optional explicit primitive name (e.g., 'ContactManager.ask').
                           If provided, skips the search step.
            include_primitives: If True (default), include primitives in semantic search.
            call_kwargs: Optional keyword arguments to pass to the function/primitive.
            verify: Optional verification flag. If None, uses the function's own verify flag.
                   If True, forces verification. If False, skips verification.
            _parent_chat_context: Ignored (no conversation context needed).
            _clarification_up_q: Ignored (no clarifications).
            _clarification_down_q: Ignored (no clarifications).

        Returns:
            A SingleFunctionActorHandle for monitoring the execution.

        Raises:
            ValueError: If no matching function or primitive is found, or if no
                       selection method (description, function_id, primitive_name) is provided.
        """
        call_kwargs = call_kwargs or {}

        if function_id is None and primitive_name is None and description is None:
            raise ValueError(
                "Must provide at least one of: description, function_id, or primitive_name",
            )

        globals_dict = self._create_execution_globals()

        if primitive_name is not None:
            function_data = self._get_primitive_by_name(primitive_name)
            logger.info(
                f"SingleFunctionActor: Executing primitive '{primitive_name}'",
            )
        elif function_id is not None:
            function_data = self._get_function_by_id(
                function_id,
                namespace=globals_dict,
            )
            logger.info(
                f"SingleFunctionActor: Executing function ID {function_id} "
                f"({function_data.get('name')})",
            )
        else:
            function_data = self._search_function(
                description,
                namespace=globals_dict,
                include_primitives=include_primitives,
            )
            logger.info(
                f"SingleFunctionActor: Found '{function_data.get('name')}' "
                f"for description: '{description}'",
            )

        function_name = function_data.get("name", "unknown")
        is_primitive = function_data.get("is_primitive", False)
        fid = function_data.get("function_id")
        docstring = function_data.get("docstring")

        if verify is None:
            should_verify = (
                function_data.get("verify", True) if not is_primitive else False
            )
        else:
            should_verify = verify

        if should_verify:
            logger.info(f"Verification enabled for '{function_name}'")

        if is_primitive:
            execution_task = asyncio.create_task(
                self._execute_primitive(function_data, **call_kwargs),
            )
        else:
            execution_task = asyncio.create_task(
                self._execute_function(function_data, globals_dict, **call_kwargs),
            )

        return SingleFunctionActorHandle(
            function_name=function_name,
            function_id=fid,
            execution_task=execution_task,
            is_primitive=is_primitive,
            verify=should_verify,
            goal=description,
            docstring=docstring,
            actor=self,
        )

    async def close(self):
        """Clean up resources."""
        if self._computer_primitives:
            try:
                self._computer_primitives.computer.stop()
            except Exception:
                pass
