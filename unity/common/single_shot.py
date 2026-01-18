"""Single-shot LLM tool decision.

A lightweight alternative to the full async tool loop for cases where you need
exactly one LLM decision that optionally calls tools. No looping, no
final_answer, no steering handles.

"Single-shot" refers to a single THINKING step - the LLM makes one decision,
but that decision may include multiple parallel tool calls.

Use cases:
- Reactive event handlers (ConversationManager responding to SMS/email/call)
- Simple classification/routing decisions
- Any "pick actions and execute them" pattern
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type, Union

import unillm
from pydantic import BaseModel

from .llm_helpers import method_to_schema
from .llm_client import pydantic_to_json_schema_response_format
from .tool_spec import ToolSpec, normalise_tools


@dataclass
class ToolExecution:
    """Result of executing a single tool.

    Attributes
    ----------
    name : str
        Name of the tool that was called.
    args : dict[str, Any]
        Arguments passed to the tool.
    result : Any
        Return value from executing the tool.
    """

    name: str
    args: dict[str, Any]
    result: Any


@dataclass
class SingleShotResult:
    """Result of a single-shot tool decision (one thinking step, possibly multiple tools).

    Attributes
    ----------
    tools : list[ToolExecution]
        All tool executions from this decision. Empty if no tools were called.
    text_response : str | None
        Text content from the LLM response (if any). This is populated when the
        LLM returns content alongside or instead of tool calls.
    structured_output : BaseModel | None
        Parsed structured output when response_format was provided.

    Properties (backward compatibility)
    -----------------------------------
    tool_name : str | None
        Name of the first tool called, or None if no tools called.
    tool_args : dict[str, Any] | None
        Arguments of the first tool, or None if no tools called.
    tool_result : Any
        Result of the first tool, or None if no tools called.
    """

    tools: List[ToolExecution] = field(default_factory=list)
    text_response: str | None = None
    structured_output: BaseModel | None = None

    @property
    def tool_name(self) -> str | None:
        """First tool name for backward compatibility."""
        return self.tools[0].name if self.tools else None

    @property
    def tool_args(self) -> dict[str, Any] | None:
        """First tool args for backward compatibility."""
        return self.tools[0].args if self.tools else None

    @property
    def tool_result(self) -> Any:
        """First tool result for backward compatibility."""
        return self.tools[0].result if self.tools else None


async def single_shot_tool_decision(
    client: unillm.AsyncUnify,
    message: str | dict | list,
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    tool_choice: str = "auto",
    include_class_name: bool = False,
    response_format: Type[BaseModel] | None = None,
) -> SingleShotResult:
    """Make a single LLM call, execute all selected tools, return the results.

    "Single-shot" means one thinking step - the LLM makes one decision, but that
    decision may include multiple parallel tool calls which are all executed.

    This is a lightweight alternative to `start_async_tool_loop` for cases where:
    - You need exactly one decision per event
    - No multi-step reasoning is required
    - No steering (pause/resume/interject) is needed
    - The actions themselves ARE the answer (no final_answer step)

    Parameters
    ----------
    client : unillm.AsyncUnify
        A configured LLM client. The system message should already be set via
        `client.set_system_message()` before calling this function.
    message : str | dict | list
        The user message to send to the LLM. Can be a simple string, a dict with
        role/content, or a list of message dicts for multi-turn context.
    tools : dict[str, Callable | ToolSpec]
        Mapping of tool names to callables (or ToolSpec wrappers). These will be
        converted to OpenAI-compatible function schemas automatically.
    tool_choice : str, default "auto"
        Controls whether the LLM must call a tool:
        - "auto": LLM decides whether to call tools or respond with text
        - "required": LLM must call at least one tool
        - "none": LLM cannot call tools (text response only)
    include_class_name : bool, default False
        Whether to include the class name prefix in tool schemas (e.g.,
        "ContactManager_filter_contacts" vs "filter_contacts").
    response_format : Type[BaseModel] | None, default None
        Optional Pydantic model for structured output. When provided, the LLM
        response content will be parsed into this model and returned in
        `structured_output`. This can be combined with tools - the model can
        return structured JSON AND call tools in the same turn.

    Returns
    -------
    SingleShotResult
        A dataclass containing:
        - tools: list of ToolExecution (name, args, result) for all tools called
        - text_response: any text content from the LLM response
        - structured_output: parsed response_format model (or None)

        For backward compatibility, also provides properties:
        - tool_name: first tool name (or None)
        - tool_args: first tool args (or None)
        - tool_result: first tool result (or None)
    """
    import asyncio
    import inspect

    # Normalise tools to ToolSpec for consistent handling
    normalised = normalise_tools(tools)

    # Build OpenAI-compatible tool schemas
    schemas = []
    for name, spec in normalised.items():
        schema = method_to_schema(
            spec.fn,
            tool_name=name,
            include_class_name=include_class_name,
        )
        schemas.append(schema)

    # Normalise message to list format
    if isinstance(message, str):
        messages = [{"role": "user", "content": message}]
    elif isinstance(message, dict):
        messages = [message]
    else:
        messages = list(message)

    # Build generate kwargs
    gen_kwargs: dict[str, Any] = {
        "messages": messages,
        "stateful": True,
    }
    if schemas:
        gen_kwargs["tools"] = schemas
        gen_kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        gen_kwargs["response_format"] = pydantic_to_json_schema_response_format(
            response_format,
        )

    # Single LLM call
    await client.generate(**gen_kwargs)

    # The client appends the assistant response to client.messages
    # Extract it from there (this is how async_tool_loop does it)
    messages = client.messages
    if not messages:
        raise RuntimeError("LLM client returned no messages")
    msg = messages[-1]

    # Extract text response (if any)
    text_response = None
    structured_output = None
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        text_response = content
        # Parse as structured output if response_format was provided
        if response_format is not None:
            try:
                parsed = json.loads(content)
                structured_output = response_format.model_validate(parsed)
            except (json.JSONDecodeError, Exception):
                # If parsing fails, leave structured_output as None
                pass

    # Check for tool calls
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        # No tools called - return text-only result
        return SingleShotResult(
            tools=[],
            text_response=text_response,
            structured_output=structured_output,
        )

    # Execute ALL tool calls concurrently
    tool_calls_list = list(tool_calls)

    async def execute_tool_call(call: dict) -> ToolExecution:
        """Execute a single tool call and return the result."""
        fn_info = call.get("function", {})
        fn_name = fn_info.get("name")
        fn_args_raw = fn_info.get("arguments", "{}")

        # Parse arguments
        if isinstance(fn_args_raw, str):
            fn_args = json.loads(fn_args_raw) if fn_args_raw else {}
        else:
            fn_args = fn_args_raw or {}

        # Get the callable
        if not fn_name or fn_name not in normalised:
            raise ValueError(f"LLM called unknown tool: {fn_name}")
        spec = normalised[fn_name]
        fn = spec.fn

        # Execute (handle both sync and async callables)
        if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):
            result = await fn(**fn_args)
        else:
            result = fn(**fn_args)

        return ToolExecution(name=fn_name, args=fn_args, result=result)

    # Execute all tool calls concurrently
    tool_executions = await asyncio.gather(
        *[execute_tool_call(call) for call in tool_calls_list],
    )

    return SingleShotResult(
        tools=list(tool_executions),
        text_response=text_response,
        structured_output=structured_output,
    )
