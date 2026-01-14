"""Single-shot LLM tool decision.

A lightweight alternative to the full async tool loop for cases where you need
exactly one LLM decision that optionally calls one tool. No looping, no
final_answer, no steering handles.

Use cases:
- Reactive event handlers (ConversationManager responding to SMS/email/call)
- Simple classification/routing decisions
- Any "pick one action and execute it" pattern
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, Union

import unillm
from pydantic import BaseModel

from .llm_helpers import method_to_schema
from .llm_client import pydantic_to_json_schema_response_format
from .tool_spec import ToolSpec, normalise_tools


@dataclass
class SingleShotResult:
    """Result of a single-shot tool decision.

    Attributes
    ----------
    tool_name : str | None
        Name of the tool that was called, or None if the LLM chose not to call any tool.
    tool_args : dict[str, Any] | None
        Arguments passed to the tool, or None if no tool was called.
    tool_result : Any
        Return value from executing the tool, or None if no tool was called.
    text_response : str | None
        Text content from the LLM response (if any). This is populated when the
        LLM returns content alongside or instead of tool calls.
    structured_output : BaseModel | None
        Parsed structured output when response_format was provided.
    """

    tool_name: str | None
    tool_args: dict[str, Any] | None
    tool_result: Any
    text_response: str | None
    structured_output: BaseModel | None = None


async def single_shot_tool_decision(
    client: unillm.AsyncUnify,
    message: str | dict | list,
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    tool_choice: str = "auto",
    include_class_name: bool = False,
    response_format: Type[BaseModel] | None = None,
) -> SingleShotResult:
    """Make a single LLM call, optionally execute one tool, return the result.

    This is a lightweight alternative to `start_async_tool_loop` for cases where:
    - You need exactly one decision per event
    - No multi-step reasoning is required
    - No steering (pause/resume/interject) is needed
    - The action itself IS the answer (no final_answer step)

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
        - "auto": LLM decides whether to call a tool or respond with text
        - "required": LLM must call exactly one tool
        - "none": LLM cannot call tools (text response only)
    include_class_name : bool, default False
        Whether to include the class name prefix in tool schemas (e.g.,
        "ContactManager_filter_contacts" vs "filter_contacts").
    response_format : Type[BaseModel] | None, default None
        Optional Pydantic model for structured output. When provided, the LLM
        response content will be parsed into this model and returned in
        `structured_output`. This can be combined with tools - the model can
        return structured JSON AND call a tool in the same turn.

    Returns
    -------
    SingleShotResult
        A dataclass containing:
        - tool_name: which tool was called (or None)
        - tool_args: arguments passed to the tool (or None)
        - tool_result: return value from the tool (or None)
        - text_response: any text content from the LLM response
        - structured_output: parsed response_format model (or None)
    """
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
        # No tool called - return text-only result
        return SingleShotResult(
            tool_name=None,
            tool_args=None,
            tool_result=None,
            text_response=text_response,
            structured_output=structured_output,
        )

    # Execute the first tool call only (single-shot = one action)
    tool_calls_list = list(tool_calls)
    call = tool_calls_list[0]
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
    import asyncio
    import inspect

    if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):
        result = await fn(**fn_args)
    else:
        result = fn(**fn_args)

    return SingleShotResult(
        tool_name=fn_name,
        tool_args=fn_args,
        tool_result=result,
        text_response=text_response,
        structured_output=structured_output,
    )
