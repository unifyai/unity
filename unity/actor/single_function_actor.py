"""
A minimal actor that executes a single function from the FunctionManager.

This actor is useful for:
- Testing that stored functions work correctly
- Deploying rigid, pre-defined workflows with no interactive elements
- Integration testing of the function/action_provider pipeline
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Dict, Optional

from unity.common.async_tool_loop import SteerableToolHandle, start_async_tool_loop
from unity.common.llm_client import new_llm_client
from unity.common.sandbox_utils import create_sandbox_globals
from unity.function_manager.function_manager import FunctionManager

from ..task_scheduler.base import BaseActiveTask
from .action_provider import ActionProvider
from .base import BaseActor, BaseActorHandle

logger = logging.getLogger(__name__)


class SingleFunctionActorHandle(BaseActiveTask, BaseActorHandle):
    """
    A minimal handle for a single function execution.

    This handle provides the standard steerable interface but with simplified
    behavior: pause/resume/interject are no-ops since a single function
    execution cannot be meaningfully paused mid-flight.
    """

    def __init__(
        self,
        function_name: str,
        function_id: int,
        execution_task: asyncio.Task,
    ):
        self._function_name = function_name
        self._function_id = function_id
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._error_str: Optional[str] = None
        self._stopped = False

        # Start monitoring the task
        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self):
        """Monitor the execution task and set completion when done."""
        try:
            result = await self._execution_task
            if not self._stopped:
                self._result_str = (
                    str(result)
                    if result is not None
                    else "Function completed successfully."
                )
        except asyncio.CancelledError:
            self._result_str = f"Function '{self._function_name}' was cancelled."
            self._stopped = True
        except Exception as e:
            self._error_str = str(e)
            self._result_str = f"Function '{self._function_name}' failed: {e}"
        finally:
            self._completion_event.set()

    @functools.wraps(BaseActiveTask.result, updated=())
    async def result(self) -> str:
        await self._completion_event.wait()
        if self._error_str:
            return f"Error: {self._error_str}"
        return self._result_str or f"Function '{self._function_name}' completed."

    @functools.wraps(BaseActiveTask.done, updated=())
    def done(self) -> bool:
        return self._completion_event.is_set()

    @functools.wraps(BaseActiveTask.stop, updated=())
    async def stop(self, reason: Optional[str] = None, **kwargs) -> str:
        """Cancel the function execution if still running."""
        if self._completion_event.is_set():
            return await self.result()

        self._stopped = True
        self._execution_task.cancel()

        # Wait for cancellation to complete
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        self._result_str = (
            f"Function '{self._function_name}' stopped."
            if not reason
            else f"Function '{self._function_name}' stopped: {reason}"
        )
        return self._result_str

    @functools.wraps(BaseActiveTask.pause, updated=())
    async def pause(self) -> str:
        """No-op: single function execution cannot be paused."""
        return f"Pause acknowledged (no effect on single function '{self._function_name}')."

    @functools.wraps(BaseActiveTask.resume, updated=())
    async def resume(self) -> str:
        """No-op: single function execution cannot be resumed."""
        return f"Resume acknowledged (no effect on single function '{self._function_name}')."

    @functools.wraps(BaseActiveTask.interject, updated=())
    async def interject(self, message: str, **kwargs) -> str:
        """No-op: single function execution cannot be interjected."""
        return f"Interjection acknowledged (no effect on single function '{self._function_name}')."

    @functools.wraps(BaseActiveTask.ask, updated=())
    async def ask(self, question: str) -> SteerableToolHandle:
        """Returns a simple response about the function status."""
        client = new_llm_client()
        client.set_system_message(
            f"You are reporting on the status of a function execution. "
            f"The function '{self._function_name}' (ID: {self._function_id}) is currently running. "
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

    # Additional BaseActiveTask methods that may be expected

    async def next_clarification(self) -> dict:
        """No clarifications for single function execution."""
        await asyncio.Event().wait()  # Wait forever
        return {}

    async def next_notification(self) -> dict:
        """No notifications for single function execution."""
        await asyncio.Event().wait()  # Wait forever
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """No clarifications to answer."""

    def get_history(self) -> list[dict]:
        """No conversation history for single function execution."""
        return []

    @property
    def clarification_up_q(self) -> Optional[asyncio.Queue[str]]:
        return None

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        return None


class SingleFunctionActor(BaseActor):
    """
    A minimal actor that executes a single function from the FunctionManager.

    This actor is designed for:
    - Testing stored functions
    - Deploying rigid, pre-defined workflows
    - Cases where interactive steering is not needed

    The actor finds and executes a single function, either by explicit ID
    or by semantic search matching the description.
    """

    def __init__(
        self,
        action_provider: Optional[ActionProvider] = None,
        function_manager: Optional[FunctionManager] = None,
        headless: bool = True,
        browser_mode: str = "magnitude",
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
    ):
        """
        Initialize the SingleFunctionActor.

        Args:
            action_provider: Optional existing ActionProvider. If not provided,
                           one will be created with the given parameters.
            function_manager: Optional FunctionManager instance. If not provided,
                            uses the singleton.
            headless: Whether to run browser in headless mode.
            browser_mode: Browser backend mode ("magnitude" or "legacy").
            agent_mode: Agent mode for ActionProvider.
            agent_server_url: URL for the agent server.
        """
        if action_provider is not None:
            self._action_provider = action_provider
        else:
            self._action_provider = ActionProvider(
                headless=headless,
                browser_mode=browser_mode,
                agent_mode=agent_mode,
                agent_server_url=agent_server_url,
            )

        self._function_manager = function_manager or FunctionManager()

    def _get_function_by_id(self, function_id: int) -> Dict[str, Any]:
        """Get a function by its ID."""
        functions = self._function_manager.list_functions(include_implementations=True)
        for name, data in functions.items():
            if data.get("function_id") == function_id:
                return {"name": name, **data}
        raise ValueError(f"No function found with ID {function_id}")

    def _search_function(self, description: str) -> Dict[str, Any]:
        """Search for the best matching function by description."""
        results = self._function_manager.search_functions_by_similarity(
            query=description,
            n=1,
        )
        if not results:
            raise ValueError(f"No function found matching description: {description}")
        return results[0]

    def _create_execution_globals(self) -> Dict[str, Any]:
        """Create the globals dict for function execution."""
        globals_dict = create_sandbox_globals()

        # Inject the action_provider
        globals_dict["action_provider"] = self._action_provider

        # Also inject commonly needed managers/utilities that functions might use
        # This makes the execution environment rich enough for real-world functions
        try:
            from unity.contact_manager.contact_manager import ContactManager

            globals_dict["ContactManager"] = ContactManager
        except ImportError:
            pass

        try:
            from unity.knowledge_manager.knowledge_manager import KnowledgeManager

            globals_dict["KnowledgeManager"] = KnowledgeManager
        except ImportError:
            pass

        try:
            from unity.secret_manager.secret_manager import SecretManager

            globals_dict["SecretManager"] = SecretManager
        except ImportError:
            pass

        return globals_dict

    async def _execute_function(
        self,
        function_data: Dict[str, Any],
        **call_kwargs,
    ) -> Any:
        """Execute a function with the given data."""
        implementation = function_data.get("implementation")
        name = function_data.get("name")

        if not implementation:
            raise ValueError(f"Function '{name}' has no implementation")

        # Create execution environment
        globals_dict = self._create_execution_globals()

        # Compile and exec the function definition
        exec(implementation, globals_dict)

        # Get the function object
        fn = globals_dict.get(name)
        if fn is None:
            raise ValueError(f"Function '{name}' not found after execution")

        # Call the function
        if inspect.iscoroutinefunction(fn):
            return await fn(**call_kwargs)
        else:
            return fn(**call_kwargs)

    async def act(
        self,
        description: str,
        *,
        function_id: Optional[int] = None,
        call_kwargs: Optional[Dict[str, Any]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ) -> SingleFunctionActorHandle:
        """
        Execute a single function from the FunctionManager.

        Args:
            description: Natural language description of what to do.
                        Used for semantic search if function_id not provided.
            function_id: Optional explicit function ID to execute.
                        If provided, skips the search step.
            call_kwargs: Optional keyword arguments to pass to the function.
            _parent_chat_context: Ignored (no conversation context needed).
            _clarification_up_q: Ignored (no clarifications).
            _clarification_down_q: Ignored (no clarifications).

        Returns:
            A SingleFunctionActorHandle for monitoring the execution.

        Raises:
            ValueError: If no matching function is found.
        """
        call_kwargs = call_kwargs or {}

        # Find the function
        if function_id is not None:
            function_data = self._get_function_by_id(function_id)
            logger.info(
                f"SingleFunctionActor: Executing function ID {function_id} "
                f"({function_data.get('name')})",
            )
        else:
            function_data = self._search_function(description)
            logger.info(
                f"SingleFunctionActor: Found function '{function_data.get('name')}' "
                f"for description: '{description}'",
            )

        function_name = function_data.get("name", "unknown")
        fid = function_data.get("function_id", -1)

        # Create the execution task
        execution_task = asyncio.create_task(
            self._execute_function(function_data, **call_kwargs),
        )

        # Return the handle
        return SingleFunctionActorHandle(
            function_name=function_name,
            function_id=fid,
            execution_task=execution_task,
        )

    async def close(self):
        """Clean up resources."""
        if self._action_provider:
            try:
                self._action_provider.browser.stop()
            except Exception:
                pass
