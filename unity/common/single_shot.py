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
    exclusive_tools: set[str] | None = None,
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
    exclusive_tools : set[str] | None, default None
        Tool names that must appear at most once per LLM turn. If the LLM calls
        an exclusive tool more than once, ALL instances are rejected without
        execution and replaced with error results.

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
    import time as _ss_time
    import logging as _ss_logging

    _ss_logger = _ss_logging.getLogger("unity")
    _ss_t0 = _ss_time.perf_counter()

    def _ss_ms() -> str:
        return f"{(_ss_time.perf_counter() - _ss_t0) * 1000:.0f}ms"

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
    _ss_logger.debug(
        f"⏱️ [single_shot +{_ss_ms()}] calling generate ({len(schemas)} tools)",
    )
    await client.generate(**gen_kwargs)
    _ss_logger.debug(f"⏱️ [single_shot +{_ss_ms()}] generate returned")

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

    # Some providers/adapters(eg: Anthropic) represent structured output as a `json_tool_call`
    # tool invocation (with arguments matching `response_format`) instead of
    # placing JSON into `message.content`. If present, parse it and exclude it
    # from tool execution.
    if response_format is not None:
        try:
            for _tc in list(tool_calls):
                _fn = (_tc.get("function") or {}) if isinstance(_tc, dict) else {}
                if _fn.get("name") != "json_tool_call":
                    continue
                _args = _fn.get("arguments", "{}")
                _payload = (
                    json.loads(_args) if isinstance(_args, str) else (_args or {})
                )
                if isinstance(_payload, dict):
                    try:
                        structured_output = response_format.model_validate(_payload)
                    except Exception:
                        # Best-effort: if it doesn't validate, ignore and proceed.
                        pass
        except Exception:
            pass

    # Execute ALL tool calls concurrently
    tool_calls_list = list(tool_calls)

    def _parse_json_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, str):
            return json.loads(raw) if raw else {}
        return raw or {}

    def _unwrap_json_tool_call(call: dict) -> list[dict]:
        """
        Expand provider wrapper tool-calls into real tool calls.

        Why this exists
        --------------
        Some LLM providers (or provider adapters) represent tool calling as a
        *single* wrapper tool invocation (commonly named `json_tool_call`) whose
        arguments describe the *actual* tool(s) to call. This is compatible with
        OpenAI-style tool schemas but requires unwrapping before execution.

        This function returns a list of OpenAI-style tool call dicts:
            {"function": {"name": <tool_name>, "arguments": <json-string|dict>}, ...}
        """
        fn_info = call.get("function", {}) or {}
        fn_name = fn_info.get("name")
        if fn_name != "json_tool_call":
            return [call]

        args = _parse_json_args(fn_info.get("arguments", "{}"))

        # Common shapes we accept:
        # 1) {"name": "...", "arguments": {...}}  (single tool)
        # 2) {"tool_name": "...", "tool_args": {...}} (single tool)
        # 3) {"tool_calls": [ {"name": "...", "arguments": {...}}, ... ]} (multi tool)
        # 4) {"tool_calls": [ {"function": {"name": "...", "arguments": {...}}}, ... ]} (already OpenAI-like)
        if isinstance(args, dict):
            if "tool_calls" in args:
                inner = args.get("tool_calls")
                if isinstance(inner, str):
                    inner = json.loads(inner) if inner else []
                if not isinstance(inner, list):
                    raise ValueError(
                        f"json_tool_call.tool_calls must be a list, got {type(inner)}",
                    )
                expanded: list[dict] = []
                for item in inner:
                    if not isinstance(item, dict):
                        continue
                    if "function" in item:
                        # Already tool-call shaped
                        expanded.append(item)
                        continue
                    # Normalize {"name":..., "arguments":...}
                    name = item.get("name") or item.get("tool_name")
                    item_args = item.get("arguments", item.get("tool_args", {}))
                    expanded.append(
                        {
                            "type": "function",
                            "id": call.get("id", ""),
                            "function": {
                                "name": name,
                                "arguments": item_args,
                            },
                        },
                    )
                return expanded

            # Single tool
            name = args.get("name") or args.get("tool_name")
            inner_args = args.get("arguments", args.get("tool_args", {}))
            if name:
                return [
                    {
                        "type": "function",
                        "id": call.get("id", ""),
                        "function": {"name": name, "arguments": inner_args},
                    },
                ]

        # If this `json_tool_call` doesn't describe an inner tool, treat it as a
        # non-executable wrapper (commonly used to carry structured output).
        return []

    async def execute_tool_call(call: dict) -> ToolExecution:
        """Execute a single tool call and return the result."""
        fn_info = call.get("function", {})
        fn_name = fn_info.get("name")
        fn_args = _parse_json_args(fn_info.get("arguments", "{}"))

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
    # Expand wrapper tool calls (e.g. `json_tool_call`) into real tool calls first.
    expanded_calls: list[dict] = []
    for c in tool_calls_list:
        expanded_calls.extend(_unwrap_json_tool_call(c))

    _tool_names = [(c.get("function") or {}).get("name", "?") for c in expanded_calls]

    # Reject exclusive tools that appear more than once.
    violated_tools: set[str] = set()
    if exclusive_tools:
        from collections import Counter

        counts = Counter(_tool_names)
        violated_tools = {name for name in exclusive_tools if counts.get(name, 0) > 1}

    _ss_logger.debug(
        f"⏱️ [single_shot +{_ss_ms()}] executing {len(expanded_calls)} tools: {_tool_names}",
    )

    if violated_tools:
        error_msg = (
            f"Rejected: multiple calls to exclusive tool(s) "
            f"{violated_tools} in a single turn. None were executed."
        )
        _ss_logger.warning(f"⏱️ [single_shot] {error_msg}")

        async def _execute_or_reject(call: dict) -> ToolExecution:
            fn_info = call.get("function", {})
            fn_name = fn_info.get("name")
            fn_args = _parse_json_args(fn_info.get("arguments", "{}"))
            if fn_name in violated_tools:
                return ToolExecution(
                    name=fn_name,
                    args=fn_args,
                    result={"error": error_msg},
                )
            return await execute_tool_call(call)

        tool_executions = await asyncio.gather(
            *[_execute_or_reject(call) for call in expanded_calls],
        )
    else:
        tool_executions = await asyncio.gather(
            *[execute_tool_call(call) for call in expanded_calls],
        )
    _ss_logger.debug(f"⏱️ [single_shot +{_ss_ms()}] all tools completed")

    return SingleShotResult(
        tools=list(tool_executions),
        text_response=text_response,
        structured_output=structured_output,
    )
