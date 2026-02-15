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

When no explicit ``call_kwargs`` are provided but a ``request`` is given,
the actor uses a lightweight LLM step to inspect the matched function's
signature and docstring and generate the appropriate keyword arguments from
the natural-language request.

Execution results are packaged as ``ExecutionResult`` objects (the same structured
format used by ``CodeActActor``), capturing stdout, stderr, the return value, and
any errors.  If the executed function returns a steerable handle, it is detected
via ``_extract_nested_handle`` and all steering operations on the outer handle
are forwarded to it.  The intermediate ``ExecutionResult`` (with stdout/stderr
captured before the handle was returned) is published as a notification.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel, Field

from unity.actor.execution import ExecutionResult, TextPart, execute_callable
from unity.common._async_tool.tools_data import _extract_nested_handle
from unity.common.async_tool_loop import (
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


class GeneratedCallKwargs(BaseModel):
    """Structured output for LLM-generated function arguments."""

    call_kwargs: Dict[str, Any] = Field(
        ...,
        description="The keyword arguments to pass to the function, matching its parameter names and types.",
    )


class SingleFunctionActorHandle(BaseActorHandle):
    """
    A handle for a single function execution.

    If the executed function returns a ``SteerableToolHandle`` (e.g., from
    ``CodeActActor.act()`` or ``start_async_tool_loop()``), it is detected
    via ``_extract_nested_handle`` and all steering operations are forwarded
    to it.  The intermediate ``ExecutionResult`` (with the handle replaced
    by a sentinel) is published as a notification so callers can observe
    stdout / stderr captured before the handle was returned.

    For non-steerable functions, steering operations are no-ops and
    ``result()`` returns the ``ExecutionResult`` directly.
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
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        self._function_name = function_name
        self._function_id = function_id
        self._is_primitive = is_primitive
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._execution_result: Optional[ExecutionResult] = None
        self._stopped = False
        self._verify = verify
        self._goal = goal
        self._docstring = docstring
        self._actor = actor
        self._verification_passed: Optional[bool] = None
        self._verification_reason: Optional[str] = None

        # Clarification queues stored locally; forwarded to inner handle when
        # a steerable function is detected.
        self._clarification_up_q_local = _clarification_up_q
        self._clarification_down_q_local = _clarification_down_q

        # Nested handle support
        self._inner_handle: Optional[SteerableToolHandle] = None
        self._handle_ready = asyncio.Event()
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self):
        """Await the execution task, detect nested handles, run verification, store result."""
        try:
            execution_result: ExecutionResult = await self._execution_task

            if self._stopped:
                self._execution_result = execution_result
                return

            # Check for a nested steerable handle inside the result.
            raw_dict = {
                "stdout": execution_result.stdout,
                "stderr": execution_result.stderr,
                "result": execution_result.result,
                "error": execution_result.error,
            }
            nested_handle, cleaned = _extract_nested_handle(raw_dict)

            if nested_handle is not None:
                # _extract_nested_handle returns a list of (handle, label)
                # tuples when handles are nested inside a container. Extract
                # the first handle for single-handle forwarding.
                if isinstance(nested_handle, list):
                    nested_handle = nested_handle[0][0]
                self._inner_handle = nested_handle
                logger.info(
                    f"Function '{self._function_name}' returned steerable handle, "
                    f"forwarding steering to inner handle.",
                )
                # Publish the cleaned ExecutionResult as a notification so
                # callers can observe any stdout/stderr captured before the
                # handle was returned.
                cleaned_result = ExecutionResult(**cleaned)
                await self._notification_q.put(
                    {
                        "type": "intermediate_result",
                        "content": cleaned_result,
                    },
                )
                self._handle_ready.set()
                # Do NOT set _completion_event -- the inner handle manages
                # completion and result() forwards to it.
                return

            # Non-steerable: run verification if enabled.
            # Skip verification when the function already raised an exception —
            # the traceback is more useful than an LLM opinion about a None
            # return value, and we must not overwrite it.
            if execution_result.error:
                logger.warning(
                    "Function '%s' raised an exception, skipping verification. "
                    "Error:\n%s",
                    self._function_name,
                    execution_result.error,
                )
            elif self._verify and self._actor is not None:
                try:
                    verification = await self._actor._verify_execution(
                        function_name=self._function_name,
                        goal=self._goal,
                        docstring=self._docstring,
                        return_value=execution_result.result,
                    )
                    self._verification_passed = verification.success
                    self._verification_reason = verification.reason

                    if not verification.success:
                        execution_result.error = (
                            f"Verification failed: {verification.reason}"
                        )
                        logger.warning(
                            f"Verification failed for '{self._function_name}': {verification.reason}",
                        )
                    else:
                        logger.info(
                            f"Verification passed for '{self._function_name}': {verification.reason}",
                        )
                except Exception as e:
                    logger.error(
                        f"Verification error for '{self._function_name}': {e}",
                    )
                    execution_result.error = f"Verification error: {e}"

            self._execution_result = execution_result

        except asyncio.CancelledError:
            self._stopped = True
            self._execution_result = ExecutionResult(
                error=f"Function '{self._function_name}' was cancelled.",
            )
        except Exception as e:
            self._execution_result = ExecutionResult(
                error=f"Function '{self._function_name}' failed: {e}",
            )
        finally:
            self._handle_ready.set()
            self._completion_event.set()

    async def result(self):
        await self._handle_ready.wait()
        if self._inner_handle is not None:
            return await self._inner_handle.result()
        await self._completion_event.wait()
        return self._execution_result or ExecutionResult(
            error=f"Function '{self._function_name}' produced no result.",
        )

    def done(self) -> bool:
        if self._inner_handle is not None:
            return self._inner_handle.done()
        return self._completion_event.is_set()

    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        if self._inner_handle is not None:
            await self._inner_handle.stop(reason)
            return
        if self._completion_event.is_set():
            return
        self._stopped = True
        self._execution_task.cancel()
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    async def pause(self) -> Optional[str]:
        if self._inner_handle is not None:
            return await self._inner_handle.pause()
        return None

    async def resume(self) -> Optional[str]:
        if self._inner_handle is not None:
            return await self._inner_handle.resume()
        return None

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        if self._inner_handle is not None:
            await self._inner_handle.interject(
                message,
                _parent_chat_context_cont=_parent_chat_context_cont,
            )

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        if self._inner_handle is not None:
            return await self._inner_handle.ask(
                question,
                _parent_chat_context=_parent_chat_context,
            )

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

    async def next_clarification(self) -> dict:
        # Wait for the inner handle to be resolved first.
        await self._handle_ready.wait()
        if self._inner_handle is not None:
            return await self._inner_handle.next_clarification()
        # Non-steerable functions don't produce clarifications; block forever.
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        if self._inner_handle is not None:
            return await self._inner_handle.answer_clarification(call_id, answer)
        return None

    def get_history(self) -> list[dict]:
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
        return self._clarification_up_q_local

    @property
    def clarification_down_q(self) -> Optional[asyncio.Queue[str]]:
        if self._inner_handle is not None and hasattr(
            self._inner_handle,
            "clarification_down_q",
        ):
            return self._inner_handle.clarification_down_q
        return self._clarification_down_q_local

    @property
    def is_steerable(self) -> bool:
        return self._inner_handle is not None

    @property
    def inner_handle(self) -> Optional[SteerableToolHandle]:
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
    explicit ID/name or by semantic search matching the request.

    When no explicit ``call_kwargs`` are provided and the matched function
    has parameters, the actor uses a lightweight LLM step to inspect the
    function's signature and docstring alongside the user's ``request``
    and generates appropriate keyword arguments automatically.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        function_manager: Optional["FunctionManager"] = None,
        headless: bool = True,
        computer_mode: str = "magnitude",
        agent_mode: str = "desktop",
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
        query: str,
        namespace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Search for the best matching function, injecting dependencies into namespace."""
        result = self._function_manager.search_functions(
            query=query,
            n=1,
            return_callable=True,
            namespace=namespace,
            also_return_metadata=True,
        )
        metadata = result.get("metadata", [])
        if not metadata:
            raise ValueError(f"No function found matching: {query}")
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
    ) -> ExecutionResult:
        """Execute a primitive (state manager method) with stdout capture."""
        name = primitive_data.get("name")

        fn = get_primitive_callable(primitive_data, self._computer_primitives)
        if fn is None:
            raise ValueError(f"Could not resolve primitive '{name}' to a callable")

        out = await execute_callable(fn, **call_kwargs)
        return ExecutionResult(**out)

    async def _execute_function(
        self,
        function_data: Dict[str, Any],
        namespace: Dict[str, Any],
        **call_kwargs,
    ) -> ExecutionResult:
        """Execute a user-defined function with stdout capture."""
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
    ) -> ExecutionResult:
        """Execute a function in the current process with stdout capture."""
        exec(implementation, namespace)

        short_name = name.split(".")[-1] if "." in name else name
        fn = namespace.get(short_name)
        if fn is None:
            raise ValueError(f"Function '{short_name}' not found after execution")

        out = await execute_callable(fn, **call_kwargs)
        return ExecutionResult(**out)

    async def _execute_in_custom_venv(
        self,
        implementation: str,
        name: str,
        venv_id: int,
        call_kwargs: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute a function in a custom virtual environment subprocess.

        The subprocess has access to both ``primitives`` and ``computer_primitives``
        via RPC calls back to the main process.
        """
        logger.info(
            f"Executing function '{name}' in custom venv (ID: {venv_id})",
        )

        is_async = "async def" in implementation

        from unity.function_manager.primitives import Primitives

        primitives = Primitives()

        out = await self._function_manager.execute_in_venv(
            venv_id=venv_id,
            implementation=implementation,
            call_kwargs=call_kwargs,
            is_async=is_async,
            primitives=primitives,
            computer_primitives=self._computer_primitives,
        )

        # Venv subprocess returns plain strings for stdout/stderr;
        # normalize to List[TextPart] for ExecutionResult compatibility.
        for key in ("stdout", "stderr"):
            val = out.get(key)
            if isinstance(val, str):
                out[key] = [TextPart(text=val)] if val else []

        return ExecutionResult(**out)

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
            handle = start_async_tool_loop(
                client=client,
                message=verification_prompt,
                tools={},
                loop_id=f"SingleFunctionVerify({function_name})",
                max_consecutive_failures=1,
                timeout=60,
                response_format=SingleFunctionVerificationResult,
            )
            result = await handle.result()
            if isinstance(result, SingleFunctionVerificationResult):
                return result

            result_dict = json.loads(result)
            return SingleFunctionVerificationResult.model_validate(result_dict)
        except Exception as e:
            logger.error(f"Verification LLM call failed: {e}")
            return SingleFunctionVerificationResult(
                success=True,
                reason=f"Verification skipped due to error: {e}",
            )

    async def _generate_call_kwargs(
        self,
        function_name: str,
        argspec: Optional[str],
        docstring: Optional[str],
        request: str,
        parent_chat_context: list[dict] | None = None,
        guidelines: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Use an LLM to generate keyword arguments for a function call.

        When the caller provides a natural-language ``request`` but no
        explicit ``call_kwargs``, this method inspects the function's signature
        (``argspec``) and ``docstring`` and asks an LLM to produce the correct
        keyword arguments.

        Args:
            function_name: Name of the matched function.
            argspec: The function's parameter signature string (e.g. ``(query: str, limit: int = 10)``).
            docstring: The function's docstring.
            request: The user's natural-language request.
            parent_chat_context: Optional conversation history for additional context.
            guidelines: Optional meta-guidance on how to approach argument generation.

        Returns:
            A dictionary of keyword arguments to pass to the function.
        """
        client = new_llm_client()
        system_msg = (
            "You are an argument-generation assistant. Given a function's signature, "
            "docstring, and the user's natural-language request, produce the keyword "
            "arguments that should be passed to the function.\n\n"
            "Rules:\n"
            "- Only include parameters that are present in the function's signature.\n"
            "- Omit parameters that have suitable defaults and don't need overriding.\n"
            "- Use the user's request as the primary source for argument values.\n"
            "- If the function takes a single string parameter (e.g. `query`, `goal`, "
            "`description`, `message`, `prompt`, `question`, `text`, `input`), "
            "pass the user's full request as that parameter.\n"
            "- Return an empty dict {} if the function takes no arguments."
        )
        if guidelines:
            system_msg += f"\n\nAdditional guidelines to follow:\n{guidelines}"
        client.set_system_message(system_msg)

        prompt_parts = [f"## Function: `{function_name}`"]

        if argspec:
            prompt_parts.append(f"\n## Signature\n```\n{function_name}{argspec}\n```")

        if docstring:
            prompt_parts.append(f"\n## Docstring\n{docstring}")

        prompt_parts.append(f"\n## User Request\n{request}")

        if parent_chat_context:
            # Include a brief summary of recent conversation for context
            recent = parent_chat_context[-5:]  # last 5 messages at most
            context_lines = []
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    # Truncate long messages
                    truncated = content[:200] + "..." if len(content) > 200 else content
                    context_lines.append(f"  {role}: {truncated}")
            if context_lines:
                prompt_parts.append(
                    "\n## Recent Conversation Context\n" + "\n".join(context_lines),
                )

        prompt_parts.append(
            "\n## Task\n"
            "Generate the keyword arguments to pass to this function. "
            "Return them as a JSON object in the `call_kwargs` field.",
        )

        generation_prompt = "\n".join(prompt_parts)

        try:
            handle = start_async_tool_loop(
                client=client,
                message=generation_prompt,
                tools={},
                loop_id=f"SingleFunctionGenerateArgs({function_name})",
                max_consecutive_failures=1,
                timeout=30,
                response_format=GeneratedCallKwargs,
            )
            result = await handle.result()
            if isinstance(result, GeneratedCallKwargs):
                return result.call_kwargs

            result_dict = json.loads(result)
            parsed = GeneratedCallKwargs.model_validate(result_dict)
            return parsed.call_kwargs
        except Exception as e:
            logger.error(
                f"Argument generation LLM call failed for '{function_name}': {e}. "
                f"Falling back to empty kwargs.",
            )
            return {}

    async def act(
        self,
        request: Optional[str] = None,
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        function_id: Optional[int] = None,
        primitive_name: Optional[str] = None,
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
            request: Natural language request specifying what to do.
                        Used for semantic search if function_id/primitive_name not provided.
                        Also used for LLM-based argument generation when ``call_kwargs``
                        is not explicitly provided.
                        Required when neither function_id nor primitive_name is specified.
            guidelines: Optional meta-guidance on *how* to approach the task, as
                opposed to *what* to do. When provided, these are included in the
                LLM context for argument generation and verification steps.
            function_id: Optional explicit function ID to execute.
                        If provided, skips the search step.
            primitive_name: Optional explicit primitive name (e.g., 'ContactManager.ask').
                           If provided, skips the search step.
            call_kwargs: Optional keyword arguments to pass to the function/primitive.
                        If not provided and the matched function has parameters, an LLM
                        step generates appropriate arguments from ``request``.
            verify: Optional verification flag. If None, uses the function's own verify flag.
                   If True, forces verification. If False, skips verification.
            _parent_chat_context: Optional conversation context from the caller (e.g.
                ConversationManager state snapshot). Always injected into the
                execution globals as ``__parent_chat_context__`` (``None`` when
                not provided) so compositional functions can forward it to
                inner actors via bare name access.
            _clarification_up_q: Optional queue for clarification requests flowing
                upward from inner tool loops. Injected into execution globals as
                ``__clarification_up_q__`` (``None`` when ``clarification_enabled``
                is False or not provided) and also wired to the returned handle.
            _clarification_down_q: Optional queue for clarification answers flowing
                downward to inner tool loops.  Injected into execution globals as
                ``__clarification_down_q__`` (``None`` when ``clarification_enabled``
                is False or not provided) and also wired to the returned handle.

        Returns:
            A SingleFunctionActorHandle for monitoring the execution.

        Raises:
            ValueError: If no matching function or primitive is found, or if no
                       selection method (request, function_id, primitive_name) is provided.
        """
        # Determine whether the caller explicitly provided call_kwargs.
        caller_provided_kwargs = call_kwargs is not None
        call_kwargs = call_kwargs or {}

        if function_id is None and primitive_name is None and request is None:
            raise ValueError(
                "Must provide at least one of: request, function_id, or primitive_name",
            )

        globals_dict = self._create_execution_globals()

        # ── Inject parent chat context ─────────────────────────────────────
        # Always inject (even as None) so bare name access inside the
        # executed function never raises NameError. The function's
        # __globals__ *is* this namespace (set by exec()), so a plain
        # ``__parent_chat_context__`` reference resolves here directly —
        # no need for globals() which is deliberately excluded from the
        # safe builtins to preserve sandbox integrity.
        globals_dict["__parent_chat_context__"] = _parent_chat_context

        # ── Resolve clarification queues based on clarification_enabled ────
        if clarification_enabled:
            clar_up = _clarification_up_q
            clar_down = _clarification_down_q
        else:
            clar_up = None
            clar_down = None

        globals_dict["__clarification_up_q__"] = clar_up
        globals_dict["__clarification_down_q__"] = clar_down

        # ── Function / primitive resolution ────────────────────────────────
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
                request,
                namespace=globals_dict,
            )
            logger.info(
                f"SingleFunctionActor: Found '{function_data.get('name')}' "
                f"for request: '{request}'",
            )

        function_name = function_data.get("name", "unknown")
        is_primitive = function_data.get("is_primitive", False)
        fid = function_data.get("function_id")
        docstring = function_data.get("docstring")
        argspec = function_data.get("argspec")

        if verify is None:
            should_verify = (
                function_data.get("verify", True) if not is_primitive else False
            )
        else:
            should_verify = verify

        if should_verify:
            logger.info(f"Verification enabled for '{function_name}'")

        # ── LLM-based argument generation ──────────────────────────────────
        # When the caller did not provide explicit call_kwargs and we have a
        # request, use an LLM to map the request to the function's
        # parameters.  Skip when:
        #  - call_kwargs were explicitly provided (even if empty)
        #  - request is None (direct ID/name execution with no request)
        #  - the function takes no parameters (argspec is "()" or empty)
        needs_arg_generation = (
            not caller_provided_kwargs
            and request is not None
            and argspec is not None
            and argspec.strip() not in ("()", "")
        )

        if needs_arg_generation:
            logger.info(
                f"SingleFunctionActor: Generating call_kwargs for '{function_name}' "
                f"from request via LLM",
            )
            call_kwargs = await self._generate_call_kwargs(
                function_name=function_name,
                argspec=argspec,
                docstring=docstring,
                request=request,
                parent_chat_context=_parent_chat_context,
                guidelines=guidelines,
            )
            logger.info(
                f"SingleFunctionActor: Generated call_kwargs for '{function_name}': "
                f"{call_kwargs}",
            )

        # ── Execution ──────────────────────────────────────────────────────
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
            goal=request,
            docstring=docstring,
            actor=self,
            _clarification_up_q=clar_up,
            _clarification_down_q=clar_down,
        )

    async def close(self):
        """Clean up resources."""
        if self._computer_primitives:
            try:
                self._computer_primitives.computer.stop()
            except Exception:
                pass
