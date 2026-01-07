from __future__ import annotations

import ast
import asyncio
import base64
import copy
import contextlib
import datetime
import enum
import functools
import inspect
import json
import logging
import sys
import textwrap
import traceback

from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type
import typing
import types
import weakref
import unillm
from pydantic import BaseModel, Field

from unity.common.llm_client import new_llm_client
import importlib.util
import importlib.machinery
import uuid
from pathlib import Path
import contextvars
from unity.common.async_tool_loop import (
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.manager_registry import ManagerRegistry
from unity.actor.base import (
    BaseActor,
    BaseActorHandle,
)
from unity.task_scheduler.base import BaseActiveTask
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.primitives import ComputerPrimitives
from unity.actor.environments.base import BaseEnvironment, ToolMetadata
import unity.actor.prompt_builders as prompt_builders
from unity.controller.browser_backends import BrowserAgentError, MagnitudeBrowserBackend
from unity.common._async_tool.loop_config import (
    LIVE_IMAGES_REGISTRY,
    LIVE_IMAGES_LOG,
)
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.conversation_manager.handle import ConversationManagerHandle
from unity.actor.steerable_tool_pane import SteerableToolPane
from unity.common.task_execution_context import (
    TaskExecutionDelegate,
    current_task_execution_delegate,
)

current_run_id_var = contextvars.ContextVar("hp_run_id", default=0)
current_interaction_sink_var = contextvars.ContextVar(
    "hp_interaction_sink",
    default=None,
)
current_invocation_id_var = contextvars.ContextVar("hp_invocation_id", default="none")

logger = logging.getLogger(__name__)

DIAGNOSTIC_MODE = True


@dataclass
class VerificationWorkItem:
    """A package of all necessary context to verify a single function's execution."""

    ordinal: int
    function_name: str
    parent_stack: tuple
    func_source: str
    docstring: str
    func_sig_str: str
    pre_state: dict
    post_state: dict
    interactions: list
    return_value_repr: str
    cache_miss_counter: int
    exit_seq: int
    start_seq: int = -1
    full_call_stack_tuple: tuple = field(default_factory=tuple)
    scoped_context_snapshot: dict = field(default_factory=dict)
    # Pane event capture (index-based boundaries; robust under concurrency)
    pane_events: list = field(default_factory=list)
    pane_event_boundary: int = 0


@dataclass
class VerificationHandle:
    """A handle to a single, in-flight verification task."""

    item: VerificationWorkItem
    task: asyncio.Task


class _HierarchicalActorDelegate:
    """Run-scoped delegate for routing task execution through the same Actor.

    This is set in `HierarchicalActorHandle._initialize_and_run()` via a ContextVar
    so that nested task execution (e.g., TaskScheduler.execute) can be routed back
    into the *same* HierarchicalActor instance that is already running.
    """

    def __init__(self, actor: "HierarchicalActor", handle: "HierarchicalActorHandle"):
        self.actor = actor
        self.handle = handle

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None,
        parent_chat_context: list[dict] | None,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
        images: Any | None = None,
        **kwargs: Any,
    ) -> SteerableToolHandle:
        """Route task execution through the same Actor instance."""
        return await self.actor.act(
            description=task_description,
            entrypoint=entrypoint,
            _parent_chat_context=parent_chat_context,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
            persist=False,
            images=images,
            **kwargs,
        )


class PlanRuntime:
    """A runtime object injected into the plan to manage execution state."""

    def __init__(self):
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._checkpoint_event = asyncio.Event()
        self._waiting_for_actor = False

        self.action_counter = 0
        self.cache_miss_counter: List[int] = []
        self.path_context: List[str] = []
        self.call_stacks = defaultdict(list)
        self.frame_id_counter = 0
        self.execution_mode: str = "fresh_start"

        self._loop_context_stack: List[Tuple[str, int]] = []
        self._loop_id_counters: Dict[str, int] = {}

    async def checkpoint(self, label: str = ""):
        """
        A cooperative yield point. It yields control, honors pauses,
        and then signals the actor and waits for release.
        """
        await asyncio.sleep(0)
        await self._pause_event.wait()

        self._waiting_for_actor = True
        self._checkpoint_event.set()
        try:
            while self._waiting_for_actor:
                await asyncio.sleep(0.01)
                await self._pause_event.wait()
        finally:
            self._checkpoint_event.clear()

    def _release_from_checkpoint(self):
        """Called by the actor to allow the plan to continue to the next checkpoint."""
        self._waiting_for_actor = False

    def pause(self):
        """Pauses the plan's execution at the next checkpoint."""
        self._pause_event.clear()

    def resume(self):
        """Resumes a paused plan."""
        self._pause_event.set()

    def push_path_context(self, context_id: str):
        """Pushes a new context (e.g., loop or branch) onto the execution path stack."""
        self.path_context.append(context_id)

    def pop_path_context(self):
        """Pops the latest context from the execution path stack."""
        if self.path_context:
            self.path_context.pop()

    def push_frame(self, run_id: int, func_name: str) -> tuple:
        """Pushes a new frame with a unique token onto the correct run's stack."""
        self.frame_id_counter += 1
        frame_token = (self.frame_id_counter, func_name)
        self.call_stacks[run_id].append(frame_token)
        return frame_token

    def pop_frame(self, run_id: int, frame_token: tuple):
        """Safely pops a frame, only if the run_id and token match."""
        stack = self.call_stacks.get(run_id)

        if not stack:
            logger.warning(
                f"[STACK_GUARD] Attempted to pop from an empty or unknown stack for run_id={run_id}",
            )
            return

        if stack[-1] != frame_token:
            logger.warning(
                f"[STACK_GUARD] Ignoring stale pop for run_id={run_id}. Expected {frame_token}, found {stack[-1]}.",
            )
            return

        stack.pop()

    def get_current_stack_tuple(self, run_id: int) -> tuple:
        """Gets the function names from the current run's stack as a tuple."""
        return tuple(
            func_name for frame_id, func_name in self.call_stacks.get(run_id, [])
        )

    def start_loop_context(self, loop_id: str):
        """Called by sanitized code BEFORE a loop starts."""
        logger.debug(f"LOOP_CONTEXT: Starting loop '{loop_id}'.")
        self._loop_context_stack.append((loop_id, -1))

    def increment_loop_iteration(self, loop_id: str):
        """Called by sanitized code at the START of each loop iteration."""
        if not self._loop_context_stack:
            logger.warning(
                f"LOOP_CONTEXT: Tried to increment iteration for '{loop_id}' but stack is empty.",
            )
            return
        current_loop_id, current_iteration = self._loop_context_stack[-1]
        if current_loop_id != loop_id:
            logger.warning(
                f"LOOP_CONTEXT: Mismatch! Expected loop '{current_loop_id}', got '{loop_id}'.",
            )
        new_iteration = current_iteration + 1
        self._loop_context_stack[-1] = (loop_id, new_iteration)
        logger.debug(
            f"LOOP_CONTEXT: Incrementing loop '{loop_id}' to iteration {new_iteration}.",
        )

    def end_loop_context(self, loop_id: str):
        """Called by sanitized code AFTER a loop finishes."""
        logger.debug(f"LOOP_CONTEXT: Ending loop '{loop_id}'.")
        if not self._loop_context_stack:
            logger.warning(
                f"LOOP_CONTEXT: Tried to end loop '{loop_id}' but stack is empty.",
            )
            return
        ended_loop_id, ended_iteration = self._loop_context_stack.pop()
        if ended_loop_id != loop_id:
            logger.warning(
                f"LOOP_CONTEXT: Mismatch! Popped loop '{ended_loop_id}' when ending '{loop_id}'.",
            )

    def get_current_loop_context_tuple(self) -> Tuple[Tuple[str, int], ...]:
        """Gets the current loop nesting state as a tuple for the cache key."""
        return tuple(self._loop_context_stack)

    def reset_loop_ids(self):
        """Resets counters used for generating unique loop IDs."""
        self._loop_id_counters.clear()


def format_pydantic_model(
    model: BaseModel,
    title: Optional[str] = None,
    indent: int = 0,
) -> str:
    """
    Generic pretty printer for any Pydantic model.

    Args:
        model: The Pydantic model instance to format
        title: Optional title to display above the model data
        indent: Base indentation level (number of spaces)

    Returns:
        A formatted string representation of the model
    """
    lines = []
    indent_str = " " * indent
    if title:
        lines.append(f"{indent_str}{title}:")
    else:
        lines.append(f"{indent_str}{model.__class__.__name__}:")
    model_data = model.model_dump(exclude_none=True)

    for field_name, field_value in model_data.items():
        field_indent = " " * (indent + 2)

        if isinstance(field_value, str):
            if "\n" in field_value:
                lines.append(f"{field_indent}{field_name}:")
                value_lines = field_value.split("\n")
                for value_line in value_lines:
                    lines.append(f"{field_indent}  {value_line}")
            else:
                lines.append(f"{field_indent}{field_name}: {field_value}")
        elif isinstance(field_value, (list, dict)):
            lines.append(f"{field_indent}{field_name}:")
            json_str = json.dumps(field_value, indent=2)
            for json_line in json_str.split("\n"):
                lines.append(f"{field_indent}  {json_line}")
        else:
            lines.append(f"{field_indent}{field_name}: {field_value}")

    return "\n".join(lines)


class ReplanFromParentException(Exception):
    """Raised by the @verify decorator when a function's goal is misguided."""

    def __init__(
        self,
        message,
        reason: Optional[str] = None,
        failed_interactions: Optional[List] = None,
    ):
        super().__init__(message)
        self.reason = reason if reason else message
        self.failed_interactions = failed_interactions


class _ForcedRetryException(Exception):
    """Internal exception to force a retry loop after a successful reimplementation."""


class _ControlledInterruptionException(Exception):
    """Raised to signal that a function's execution should be stopped and retried due to a user interjection."""


class FatalVerificationError(Exception):
    """Raised when verification results in a fatal, unrecoverable error."""


class _StrictBaseModel(BaseModel):
    """
    BaseModel configured for OpenAI Structured Outputs (strict JSON schema).

    OpenAI requires `additionalProperties: false` on all object schemas used for
    `response_format`. Pydantic defaults allow extras unless configured.
    """

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class VerificationAssessment(_StrictBaseModel):
    """Structured output for the _check_state_against_goal LLM call."""

    status: str = Field(
        ...,
        description="Outcome: 'ok', 'reimplement_local', 'replan_parent', 'fatal_error', or 'request_clarification'.",
    )
    reason: str = Field(..., description="A concise explanation for the status.")
    clarification_question: Optional[str] = Field(
        None,
        description="The specific question to ask the user if status is 'request_clarification'.",
    )


class ImplementationDecision(_StrictBaseModel):
    """A structured decision for how to proceed with a function implementation."""

    action: typing.Literal[
        "implement_function",
        "replan_parent",
        "skip_function",
        "request_clarification",
    ] = Field(
        ...,
        description="The chosen action: 'implement_function' to provide new code, 'replan_parent' to escalate the failure, 'skip_function' to bypass the current step, or 'request_clarification' to ask the user for help.",
    )
    code: Optional[str] = Field(
        None,
        description="The Python code for the function. Required if action is 'implement_function'.",
    )
    reason: str = Field(
        ...,
        description="A concise justification for the chosen action. If replanning the parent, this reason will be passed up.",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="A clear, specific question to ask the user. Required if action is 'request_clarification'.",
    )


class FunctionPatch(_StrictBaseModel):
    """Represents a single function's code to be updated in the plan."""

    function_name: str = Field(
        ...,
        description="The name of the function to be replaced.",
    )
    new_code: str = Field(
        ...,
        description="The full, new source code for this function, including the signature.",
    )


class CacheStepRange(_StrictBaseModel):
    """Specifies a range of steps within a function to invalidate."""

    function_name: str = Field(..., description="The name of the function to target.")
    from_step_inclusive: int = Field(
        ...,
        description="The per-function action_counter step to start invalidating from (inclusive).",
    )


class CacheInvalidateSpec(_StrictBaseModel):
    """
    LLM's proposal for selective cache invalidation after a plan modification.
    The runtime will apply these and still enforce safety guardrails (e.g., impure propagation).
    """

    invalidate_functions: List[str] = Field(
        default_factory=list,
        description="A list of function names whose entire cache should be cleared.",
    )
    invalidate_steps: List[CacheStepRange] = Field(
        default_factory=list,
        description="Invalidate only the tail of a function: from the specified step number onward (inclusive).",
    )


class _InterjectionBroadcastFilter(_StrictBaseModel):
    """Strict schema for broadcast routing filters in `InterjectionDecision`."""

    statuses: list[
        typing.Literal[
            "running",
            "paused",
            "waiting_for_clarification",
            "completed",
            "failed",
            "stopped",
        ]
    ] = Field(
        default_factory=lambda: ["running", "paused", "waiting_for_clarification"],
        description=(
            "Handle statuses to include in broadcast routing. Defaults to in-flight statuses."
        ),
    )
    origin_tool_prefixes: Optional[List[str]] = Field(
        None,
        description="Only target handles whose origin_tool starts with any of these prefixes.",
    )
    capabilities: Optional[List[str]] = Field(
        None,
        description="Only target handles that declare ALL of these capabilities.",
    )
    created_after_step: Optional[int] = Field(
        None,
        description="Only target handles created at or after this origin_step.",
    )
    created_before_step: Optional[int] = Field(
        None,
        description="Only target handles created at or before this origin_step.",
    )


class InterjectionDecision(_StrictBaseModel):
    """A structured decision for how to proceed with a user interjection."""

    action: typing.Literal[
        "modify_task",
        "replace_task",
        "explore_detached",
        "clarify",
        "complete_task",
        "refactor_and_generalize",
    ] = Field(..., description="The chosen action based on the user's interjection.")
    reason: str = Field(..., description="A brief justification for the chosen action.")
    patches: Optional[List[FunctionPatch]] = Field(
        None,
        description="A list of functions to be updated. Required for 'modify_task'.",
    )
    cache: Optional[CacheInvalidateSpec] = Field(
        None,
        description="An optional, surgical plan for invalidating the cache to enable a more efficient replay.",
    )
    new_goal: Optional[str] = Field(
        None,
        description="The goal for the new or detached task.",
    )
    clarification_question: Optional[str] = Field(
        None,
        description="A question to ask the user for clarification.",
    )
    generalization_context: Optional[str] = Field(
        None,
        description="The context for generalization, e.g., 'all other employees in the folder' or 'Sam Parker'.",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Routing to in-flight steerable handles (SteerableToolPane)
    # ──────────────────────────────────────────────────────────────────────
    routing_action: Optional[
        typing.Literal["none", "targeted", "broadcast_filtered"]
    ] = Field(
        None,
        description=(
            "How to route this interjection to in-flight handles. "
            "'none': No routing (default). "
            "'targeted': Route to specific handle_ids. "
            "'broadcast_filtered': Broadcast to handles matching filter criteria."
        ),
    )
    target_handle_ids: Optional[List[str]] = Field(
        None,
        description=(
            "List of handle_ids to target when routing_action='targeted'. "
            "Use this when the interjection is relevant to specific in-flight operations."
        ),
    )
    broadcast_filter: Optional[_InterjectionBroadcastFilter] = Field(
        None,
        description=(
            "Filter criteria for broadcast routing when routing_action='broadcast_filtered'. "
            "All filters are inclusive-only (whitelist). If omitted, defaults to all in-flight interjectable handles."
        ),
    )
    routed_message: Optional[str] = Field(
        None,
        description=(
            "Optional custom message to send to routed handles. "
            "If omitted, the original interjection message is used."
        ),
    )


class SandboxMergeDecision(BaseModel):
    """A structured decision on whether to merge sandbox findings."""

    modification_needed: bool = Field(
        ...,
        description="True if the main plan should be modified based on the sandbox results.",
    )
    reason: str = Field(..., description="A brief justification for the decision.")
    modification_request: Optional[str] = Field(
        None,
        description="If true, the user's request, rephrased as a direct instruction to modify the main plan.",
    )


class RefactorDecision(BaseModel):
    """The structured output from the refactoring LLM call."""

    refactored_code: str = Field(
        ...,
        description="The complete, refactored Python code block.",
    )
    deduced_precondition: PreconditionDecision = Field(
        ...,
        description="The necessary starting state for the refactored plan.",
    )


class PreconditionDecision(BaseModel):
    """A structured decision on a function's precondition."""

    status: typing.Literal["ok", "not_applicable"] = Field(
        ...,
        description="Whether a precondition was identified or not.",
    )
    url: Optional[str] = Field(
        None,
        description="The URL of the page that must be present for the function to run.",
    )
    description: Optional[str] = Field(
        None,
        description="A description of the page state that must be present for the function to run.",
    )


class _HierarchicalHandleState(enum.Enum):
    """Manages the detailed lifecycle state of a hierarchical plan."""

    IDLE = enum.auto()
    EXPLORING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    PAUSED_FOR_MODIFICATION = enum.auto()
    PAUSED_FOR_INTERJECTION = enum.auto()
    COMPLETED = enum.auto()
    STOPPED = enum.auto()
    ERROR = enum.auto()


async def llm_call(
    client: unillm.AsyncUnify,
    prompt: str,
    screenshot: bytes | str | None = None,
    images: Optional[dict[str, Any]] = None,
    static_prompt: Optional[str] = None,
) -> str:
    """
    Convenience wrapper for a simple, stateless LLM call with optional prompt caching.

    This helper automatically resets the client's message history before making
    the call to ensure no context is leaked from previous interactions.

    Args:
        client: The AsyncUnify client to use for the LLM call
        prompt: The dynamic prompt content (user message)
        screenshot: Optional screenshot to include in the prompt
        images: Optional dictionary of image handles to include
        static_prompt: Optional static content to cache (sent as system message with cache_control)

    Returns:
        The LLM's response as a string (automatically extracts content from ChatCompletion
        if the client has return_full_completion=True)

    Note:
        When static_prompt is provided, the static content
        is sent as a system message with LiteLLM's cache_control directive. This enables
        provider-agnostic prompt caching across OpenAI, Anthropic, Gemini, etc.
        The static prompt should be ≥2,048 tokens for optimal caching benefits.
    """
    # Vertex/Gemini context caching has provider-side minimums. Empirically, attempting to
    # start caching below ~1024 tokens can hard-fail requests with INVALID_ARGUMENT.
    #
    # We therefore only include `cache_control` when the static prompt is "definitely"
    # large enough to be worth caching (and to avoid provider errors). Otherwise we still
    # send `static_prompt` as a normal system message, but without cache directives.
    #
    # NOTE: we intentionally use a conservative character threshold (rather than a token
    # estimator) because tokenization varies by provider/model.
    _CACHE_CONTROL_MIN_CHARS = 9000
    _use_cache_control = (
        bool(static_prompt) and len(static_prompt) >= _CACHE_CONTROL_MIN_CHARS
    )

    client.reset_messages()
    user_content = [{"type": "text", "text": prompt}]

    if screenshot:
        if isinstance(screenshot, str):
            screenshot_b64 = screenshot
        else:
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                },
            },
        )

    if images:
        for key, handle in images.items():
            try:
                image_bytes = handle.raw()
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                )
            except Exception as e:
                logger.warning(f"Could not process image for prompt: {e}")

    if static_prompt:
        if _use_cache_control:
            system_content = [
                {
                    "type": "text",
                    "text": static_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        else:
            system_content = [{"type": "text", "text": static_prompt}]

        messages_to_send = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    else:
        messages_to_send = [{"role": "user", "content": user_content}]

    try:
        response = await client.generate(messages=messages_to_send)
    except Exception as _e:
        raise

    if static_prompt and _use_cache_control:
        try:
            usage = None

            if hasattr(response, "usage"):
                usage = response.usage
            if usage:
                cached = 0

                if hasattr(usage, "prompt_tokens_details") and hasattr(
                    usage.prompt_tokens_details,
                    "cached_tokens",
                ):
                    cached = usage.prompt_tokens_details.cached_tokens or 0
                elif hasattr(usage, "cache_read_input_tokens"):
                    cached = usage.cache_read_input_tokens or 0

                if cached > 0:
                    logger.debug(f"✓ Prompt cache hit: {cached:,} tokens cached")
                logger.debug(f"Usage: {usage}")
        except Exception as e:
            logger.debug(f"Could not access cache metadata: {e}")

    if isinstance(response, str):
        return response
    else:
        return response.choices[0].message.content


class PlanSanitizer(ast.NodeTransformer):
    """
    AST transformer to enforce security and correctness of plan code.
    - Ensures every `async def` function is decorated with `@verify`.
    - Injects checkpoint calls (`_cp`) and interrupt probes (`_int`).
    - Safely wraps awaited tool calls and sub-tasks.
    - Recursively applies transformations to nested functions.
    """

    def __init__(self, plan: "HierarchicalActorHandle"):
        self._plan = plan
        self._function_context: list[str] = []
        self._is_in_async_context = False
        self._defined_functions = set()
        self._counters: dict[str, dict[str, int]] = {}
        # Active environment namespaces for this plan (e.g. {"computer_primitives", "primitives"}).
        # Any awaited call whose label starts with one of these namespaces is treated as a tool call
        # and will receive checkpoints/interrupt probes via instrumentation.
        self._tool_call_namespaces: set[str] = set()
        try:
            envs = getattr(plan, "environments", None)
            if isinstance(envs, dict):
                self._tool_call_namespaces = {
                    k for k in envs.keys() if isinstance(k, str)
                }
            else:
                actor_envs = getattr(getattr(plan, "actor", None), "environments", None)
                if isinstance(actor_envs, dict):
                    self._tool_call_namespaces = {
                        k for k in actor_envs.keys() if isinstance(k, str)
                    }
        except Exception:
            self._tool_call_namespaces = set()

    def _label_is_tool_call(self, label: str) -> bool:
        """Returns True when a call label belongs to an active environment namespace."""
        if not label:
            return False
        root = label.split(".", 1)[0]
        return root in self._tool_call_namespaces

    def _make_call_node(self, func_id: str, args: list) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=func_id, ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    def _make_await_expr_node(self, call_node: ast.Call) -> ast.Expr:
        return ast.Expr(value=ast.Await(value=call_node))

    def _make_cp_node(self, label: str) -> ast.Expr:
        return self._make_await_expr_node(
            self._make_call_node("_cp", [ast.Constant(value=label)]),
        )

    def _make_int_node(self, func_name: str) -> ast.Expr:
        return self._make_await_expr_node(
            self._make_call_node("_int", [ast.Constant(value=func_name)]),
        )

    def _call_to_label(self, call: ast.Call) -> str:
        """Derives a readable label from a call node."""
        f = call.func
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        return ".".join(reversed(parts)) if parts else "call"

    def _make_runtime_call_expr(self, method: str, args: List[ast.AST]) -> ast.Expr:
        """Helper to create `runtime.method(...)` expressions."""
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="runtime", ctx=ast.Load()),
                    attr=method,
                    ctx=ast.Load(),
                ),
                args=args,
                keywords=[],
            ),
        )

    def _enter_function(
        self,
        node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ):
        """Helper to manage entering a function's scope."""
        self._function_context.append(node.name)
        self._counters[node.name] = {
            "loop": 0,
            "if": 0,
            "try": 0,
        }

    def _exit_function(self):
        """Helper to manage exiting a function's scope."""
        if self._function_context:
            self._function_context.pop()

    def _get_counter(self, counter_type: str) -> int:
        """Gets and increments a counter for the current function scope."""
        current_func = (
            self._function_context[-1] if self._function_context else "global"
        )
        if current_func not in self._counters:
            self._counters[current_func] = {"loop": 0, "if": 0, "try": 0}
        self._counters[current_func][counter_type] += 1
        return self._counters[current_func][counter_type]

    @property
    def _current_function_name(self) -> str:
        """Property for backward compatibility."""
        return self._function_context[-1] if self._function_context else "global"

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._plan.top_level_function_names.clear()
        for sub_node in node.body:
            if isinstance(sub_node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                self._defined_functions.add(sub_node.name)
                self._plan.top_level_function_names.add(sub_node.name)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handles regular (non-async) function definitions and their nesting."""
        self._enter_function(node)
        parent_is_async = self._is_in_async_context
        self._is_in_async_context = False

        self.generic_visit(node)

        self._is_in_async_context = parent_is_async
        self._exit_function()
        return node

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        self._enter_function(node)
        parent_is_async = self._is_in_async_context
        self._is_in_async_context = True

        cleaned_body = []
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AugAssign)
                and isinstance(stmt.target, ast.Attribute)
                and hasattr(stmt.target.value, "id")
                and stmt.target.value.id == "runtime"
                and stmt.target.attr == "action_counter"
            ):
                continue

            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Await)
                and isinstance(stmt.value.value, ast.Call)
                and isinstance(stmt.value.value.func, ast.Name)
                and stmt.value.value.func.id in {"_cp", "_int"}
            ):
                continue

            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and hasattr(stmt.value.func.value, "id")
                and stmt.value.func.value.id == "runtime"
                and stmt.value.func.attr in {"push_path_context", "pop_path_context"}
            ):
                continue

            cleaned_body.append(stmt)
        node.body = cleaned_body

        has_verify_decorator = any(
            isinstance(d, ast.Name) and d.id == "verify" for d in node.decorator_list
        )
        should_skip_verify = node.name in self._plan.functions_skip_verify
        if not has_verify_decorator and not should_skip_verify:
            node.decorator_list.insert(0, ast.Name(id="verify", ctx=ast.Load()))

        entry_probes = [
            self._make_cp_node(f"Enter function: {node.name}"),
            self._make_int_node(node.name),
        ]

        offset = (
            1
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            )
            else 0
        )
        node.body[offset:offset] = entry_probes

        self.generic_visit(node)

        self._is_in_async_context = parent_is_async
        self._exit_function()
        return node

    def _wrap_block_with_context(
        self,
        block: List[ast.stmt],
        context_id: str,
    ) -> List[ast.stmt]:
        """Injects push/pop context calls around a block of statements GUARANTEED to run."""
        if not block:
            return []

        push_call = self._make_runtime_call_expr(
            "push_path_context",
            [ast.Constant(value=context_id)],
        )
        pop_call = self._make_runtime_call_expr("pop_path_context", [])

        finalized_block = ast.Try(
            body=[push_call] + block,
            handlers=[],
            orelse=[],
            finalbody=[pop_call],
        )
        return [finalized_block]

    def visit_If(self, node: ast.If) -> ast.If:
        if_id = f"if_{self._get_counter('if')}"
        self.generic_visit(node)

        node.body = self._wrap_block_with_context(node.body, f"{if_id}_true")
        if node.orelse:
            node.orelse = self._wrap_block_with_context(node.orelse, f"{if_id}_false")

        return node

    def visit_Try(self, node: ast.Try) -> ast.Try:
        try_id = f"try_{self._get_counter('try')}"
        self.generic_visit(node)

        node.body = self._wrap_block_with_context(node.body, f"{try_id}_try")
        for i, handler in enumerate(node.handlers):
            handler.body = self._wrap_block_with_context(
                handler.body,
                f"{try_id}_except_{i}",
            )
        if node.finalbody:
            node.finalbody = self._wrap_block_with_context(
                node.finalbody,
                f"{try_id}_finally",
            )

        return node

    def _get_loop_id(self, loop_node: ast.AST) -> str:
        """Generates a unique ID for a loop within the current function."""
        loop_type = type(loop_node).__name__.lower()
        current_func = self._current_function_name

        if current_func not in self._counters:
            self._counters[current_func] = {}
        if loop_type not in self._counters[current_func]:
            self._counters[current_func][loop_type] = 0

        self._counters[current_func][loop_type] += 1
        count = self._counters[current_func][loop_type]

        return f"{loop_type}_{count}"

    def _wrap_loop_with_context(
        self,
        node: typing.Union[ast.While, ast.For, ast.AsyncFor],
    ) -> ast.Try:
        """Injects start, increment, and end context calls around a loop."""
        loop_id = self._get_loop_id(node)

        # 1. Create call nodes for runtime methods
        start_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="runtime", ctx=ast.Load()),
                    attr="start_loop_context",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            ),
        )

        increment_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="runtime", ctx=ast.Load()),
                    attr="increment_loop_iteration",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            ),
        )

        end_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="runtime", ctx=ast.Load()),
                    attr="end_loop_context",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            ),
        )

        self.generic_visit(node)

        probes = []
        if self._is_in_async_context:
            probes = [
                self._make_cp_node(
                    f"Enter {type(node).__name__} in {self._current_function_name}",
                ),
                self._make_int_node(self._current_function_name),
            ]

        new_body = [increment_call] + probes + node.body
        node.body = new_body

        wrapped_loop = ast.Try(
            body=[start_call, node],
            handlers=[],
            orelse=[],
            finalbody=[end_call],
        )
        ast.fix_missing_locations(wrapped_loop)

        return wrapped_loop

    def visit_For(self, node: ast.For) -> ast.Try:
        return self._wrap_loop_with_context(node)

    def visit_While(self, node: ast.While) -> ast.Try:
        return self._wrap_loop_with_context(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.Try:
        return self._wrap_loop_with_context(node)

    def visit_Expr(self, node: ast.Expr) -> list[ast.AST] | ast.Expr:
        """Handles top-level 'await' expressions."""
        if not (
            isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call)
        ):
            return self.generic_visit(node)

        call = node.value.value
        label = self._call_to_label(call)
        is_tool_call = self._label_is_tool_call(label) or (
            isinstance(call.func, ast.Name) and call.func.id in self._defined_functions
        )

        if is_tool_call:
            return self._instrument_awaited_call(call, node)

        return self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> ast.Await:
        """Handles awaited calls inside expressions by wrapping them."""
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)

        call = node.value
        label = self._call_to_label(call)

        if self._label_is_tool_call(label) or (
            isinstance(call.func, ast.Name) and call.func.id in self._defined_functions
        ):
            new_call = self._make_call_node(
                "_around_cp",
                [ast.Constant(value=label), call],
            )
            node.value = new_call

        return self.generic_visit(node)

    def _instrument_awaited_call(
        self,
        call_node: ast.Call,
        full_statement_node: ast.AST,
    ) -> list[ast.AST]:
        """Helper to wrap a tool call statement with instrumentation."""
        label = self._call_to_label(call_node)

        return [
            self._make_cp_node(f"Before: {label}"),
            full_statement_node,
            self._make_cp_node(f"After: {label}"),
            self._make_int_node(self._current_function_name),
        ]

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST] | ast.Assign:
        """Handles assignment statements with awaited tool calls."""
        if not (
            isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call)
        ):
            return self.generic_visit(node)

        call = node.value.value
        label = self._call_to_label(call)
        is_tool_call = self._label_is_tool_call(label) or (
            isinstance(call.func, ast.Name) and call.func.id in self._defined_functions
        )

        if is_tool_call:
            return self._instrument_awaited_call(call, node)

        return self.generic_visit(node)


class FunctionReplacer(ast.NodeTransformer):
    """AST transformer to replace a function definition in a module, including nested functions."""

    def __init__(self, target_name: str, new_function_node: ast.AST):
        """
        Initializes the transformer.

        Args:
            target_name: The name of the function to replace.
            new_function_node: The new AST node for the function.
        """
        self.target_name = target_name
        self.new_function_node = new_function_node
        self.replaced = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visits and potentially replaces a synchronous function definition."""
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visits and potentially replaces an asynchronous function definition."""
        if node.name == self.target_name:
            self.replaced = True
            return self.new_function_node
        return self.generic_visit(node)


class _HistoryCapturingHandleProxy(SteerableToolHandle):
    def __init__(
        self,
        real_handle: SteerableToolHandle,
        plan: "HierarchicalActorHandle",
        handle_id: str | None,
        call_repr: str,
        cache_key: tuple,
        meta: dict,
    ):
        self._real_handle = real_handle
        self._plan = plan
        self._handle_id = handle_id
        self._call_repr = call_repr
        self._cache_key = cache_key
        self._meta = meta

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_handle, name)

    async def stop(
        self,
        reason: str | None = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        return await self._real_handle.stop(
            reason,
            parent_chat_context_cont=parent_chat_context_cont,
        )

    async def pause(self):
        return await self._real_handle.pause()

    async def resume(self):
        return await self._real_handle.resume()

    async def done(self):
        return await self._real_handle.done()

    async def next_clarification(self) -> dict:
        return await self._real_handle.next_clarification()

    async def next_notification(self) -> dict:
        return await self._real_handle.next_notification()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return await self._real_handle.answer_clarification(call_id, answer)

    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        return await self._real_handle.ask(
            question,
            parent_chat_context_cont=parent_chat_context_cont,
        )

    async def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        return await self._real_handle.interject(
            message,
            parent_chat_context_cont=parent_chat_context_cont,
        )

    async def result(self) -> str:
        if not isinstance(self._cache_key, tuple):
            raise TypeError(
                f"Expected cache_key to be a tuple, got {type(self._cache_key)}",
            )

        result_cache_key_list = list(self._cache_key)
        if len(result_cache_key_list) > 3 and isinstance(result_cache_key_list[3], str):
            result_cache_key_list[3] += "->result()"
            result_cache_key = tuple(result_cache_key_list)
        else:
            logger.warning(
                f"Unexpected cache key structure for result caching: {self._cache_key}. "
                f"Using modified repr.",
            )
            result_cache_key = (*self._cache_key, "->result()")

        logger.debug(
            f"RESULT: Using cache key: {result_cache_key}",
        )
        if result_cache_key in self._plan.idempotency_cache:
            cached_data = self._plan.idempotency_cache[result_cache_key]
            cached_result = cached_data["result"]
            cached_interaction = cached_data["interaction_log"]

            self._plan.action_log.append(
                f"CACHE HIT: Using cached result for {self._call_repr} -> .result()",
            )
            logger.debug(f"CACHE HIT for: {self._call_repr} -> .result()")

            interactions_log = current_interaction_sink_var.get()
            if interactions_log is not None:
                if len(cached_interaction) < 4:
                    cached_interaction = (*cached_interaction, [])
                interactions_log.append(cached_interaction)

            return cached_result

        self._plan.action_log.append(
            f"CACHE MISS: Executing {self._call_repr} -> .result()",
        )
        logger.debug(f"CACHE MISS for: {self._call_repr} -> .result()")

        final_result = await self._real_handle.result()

        sub_loop_history = []
        if hasattr(self._real_handle, "get_history"):
            sub_loop_history = self._real_handle.get_history()

        interaction_to_cache = (
            "handle_method_call",
            f"{self._call_repr} -> .result()",
            str(final_result),
            sub_loop_history,
        )

        interactions_log = current_interaction_sink_var.get()
        if interactions_log is not None:
            interactions_log.append(interaction_to_cache)

        self._plan.idempotency_cache[result_cache_key] = {
            "result": final_result,
            "interaction_log": interaction_to_cache,
            "meta": self._meta,
        }

        # Best-effort completion marking for pane (only reliable when `.result()` is awaited).
        try:
            if (
                getattr(self._plan, "pane", None) is not None
                and self._handle_id is not None
            ):
                await self._plan.pane._cleanup_handle(
                    self._handle_id,
                    emit_completed=True,
                )
        except Exception as e:
            logger.debug(
                f"Pane completion marking failed for handle_id={self._handle_id}: {e}",
            )

        return final_result


class _SteerableToolHandleProxy:
    """
    A proxy for SteerableToolHandle to intercept its method calls, log
    them for verification, and apply idempotency caching.
    """

    def __init__(
        self,
        real_handle: SteerableToolHandle,
        plan: "HierarchicalActorHandle",
        handle_name: str,
        handle_id: str,
    ):
        self._real_handle = real_handle
        self._plan = plan
        self._handle_name = handle_name
        self._handle_id = handle_id

    def __getattr__(self, name: str) -> Any:
        """
        Intercepts attribute access on the handle (e.g., call_handle.ask).
        """
        real_attr = getattr(self._real_handle, name)
        if not callable(real_attr):
            return real_attr

        def handle_method_logic(
            call_repr: str,
            is_cached: bool,
            cached_data: dict | None,
            tool_name: str,
            original_cache_key: tuple,
        ):
            if is_cached:
                cached_result_value = cached_data["result"]
                cached_interaction = cached_data["interaction_log"]
                self._plan.action_log.append(
                    f"CACHE HIT: Using cached result for {call_repr}",
                )
                logger.debug(f"CACHE HIT for: {call_repr}")
                interactions_log = current_interaction_sink_var.get()
                if interactions_log is not None:
                    if len(cached_interaction) < 4:
                        cached_interaction = (*cached_interaction, [])
                    interactions_log.append(cached_interaction)

                if (
                    isinstance(cached_result_value, dict)
                    and "handle_id" in cached_result_value
                ):
                    handle_id = cached_result_value["handle_id"]
                    real_handle = self._plan.live_handles.get(handle_id)
                    if not real_handle:
                        logger.error(
                            f"Cache consistency error: Could not find live handle for ID {handle_id} "
                            f"during cache hit for {call_repr}. Forcing cache miss.",
                        )
                        return None

                    meta = cached_data.get("meta", {})

                    return _HistoryCapturingHandleProxy(
                        real_handle,
                        self._plan,
                        handle_id,
                        call_repr,
                        original_cache_key,
                        meta,
                    )
                else:
                    return cached_result_value
            return None

        async def async_method_wrapper(*args, **kwargs):
            self._plan.runtime.action_counter += 1
            tool_name = f"{self._handle_name}:{self._handle_id}.{name}"
            call_repr = f"{self._handle_name}[{self._handle_id[:8]}].{name}({self._plan.actor._serialize_args(args, kwargs)})"

            cache_key = self._plan.actor._generate_cache_key(
                self._plan,
                tool_name,
                args,
                kwargs,
            )

            if cache_key in self._plan.idempotency_cache:
                cached_result = handle_method_logic(
                    call_repr,
                    True,
                    self._plan.idempotency_cache[cache_key],
                    tool_name,
                    cache_key,
                )
                if cached_result is not None:
                    return cached_result

            if self._plan.runtime.cache_miss_counter:
                self._plan.runtime.cache_miss_counter[-1] += 1
            self._plan.action_log.append(f"CACHE MISS: Executing {call_repr}")
            logger.debug(f"CACHE MISS for: {call_repr}")

            output = await real_attr(*args, **kwargs)

            # Best-effort completion marking for pane: when `.result()` is awaited on a top-level handle proxy.
            if name == "result":
                try:
                    if getattr(self._plan, "pane", None) is not None:
                        await self._plan.pane._cleanup_handle(
                            self._handle_id,
                            emit_completed=True,
                        )
                except Exception as e:
                    logger.debug(
                        f"Pane completion marking failed for handle_id={self._handle_id}: {e}",
                    )

            if isinstance(output, SteerableToolHandle):
                sub_handle_name = f"{name}_handle"
                sub_handle_id = str(uuid.uuid4())
                self._plan.live_handles[sub_handle_id] = output

                # Register nested handles with the pane (best-effort).
                try:
                    if getattr(self._plan, "pane", None) is not None:
                        capabilities: list[str] = []
                        if hasattr(output, "interject"):
                            capabilities.append("interjectable")
                        if hasattr(output, "pause") and hasattr(output, "resume"):
                            capabilities.append("pausable")
                        if hasattr(output, "ask"):
                            capabilities.append("askable")
                        if hasattr(output, "stop"):
                            capabilities.append("stoppable")
                        if hasattr(output, "answer_clarification"):
                            capabilities.append("clarifiable")

                        await self._plan.pane.register_handle(
                            handle=output,
                            handle_id=sub_handle_id,
                            parent_handle_id=self._handle_id,
                            origin_tool=tool_name,
                            origin_step=self._plan.runtime.action_counter,
                            environment_namespace="handle_methods",
                            capabilities=capabilities,
                            call_stack=str(cache_key[0]) if cache_key else None,
                        )
                except Exception as e:
                    logger.debug(f"Pane nested registration failed: {e}")

                result_to_cache = {"handle_id": sub_handle_id}

                initial_interaction = (
                    "handle_method_call",
                    call_repr,
                    f"Returned handle: {output.__class__.__name__} (ID: {sub_handle_id})",
                )
                interactions_log = current_interaction_sink_var.get()
                if interactions_log is not None:
                    interactions_log.append(initial_interaction)

                # URL capture happens at the tool-call proxy level, not at the handle
                # method level (which may be domain-agnostic and not have browser access).
                url = None

                meta = {
                    "call_stack": cache_key[0],
                    "path": cache_key[1],
                    "function": cache_key[0][-1] if cache_key[0] else None,
                    "step": self._plan.runtime.action_counter,
                    "tool": tool_name,
                    "url": url,
                    "impure": False,
                }

                self._plan.idempotency_cache[cache_key] = {
                    "result": result_to_cache,
                    "interaction_log": initial_interaction,
                    "meta": meta,
                }

                return _HistoryCapturingHandleProxy(
                    output,
                    self._plan,
                    sub_handle_id,
                    call_repr,
                    cache_key,
                    meta,
                )
            else:
                # Special-case `.result()`: capture sub-loop history (when available) as a 4th tuple element.
                # This allows verification prompts to treat handle history as "low-level trace" evidence.
                if name == "result":
                    sub_loop_history: list[Any] = []
                    if hasattr(self._real_handle, "get_history"):
                        try:
                            sub_loop_history = self._real_handle.get_history()  # type: ignore[attr-defined]
                        except Exception:
                            sub_loop_history = []
                    interaction_to_cache = (
                        "handle_method_call",
                        call_repr,
                        str(output),
                        sub_loop_history,
                    )
                else:
                    interaction_to_cache = (
                        "handle_method_call",
                        call_repr,
                        str(output),
                    )
                interactions_log = current_interaction_sink_var.get()
                if interactions_log is not None:
                    interactions_log.append(interaction_to_cache)

                # URL capture happens at the tool-call proxy level, not at the handle
                # method level (which may be domain-agnostic and not have browser access).
                url = None

                meta = {
                    "call_stack": cache_key[0],
                    "path": cache_key[1],
                    "function": cache_key[0][-1] if cache_key[0] else None,
                    "step": self._plan.runtime.action_counter,
                    "tool": tool_name,
                    "url": url,
                    "impure": False,
                }

                self._plan.idempotency_cache[cache_key] = {
                    "result": output,
                    "interaction_log": interaction_to_cache,
                    "meta": meta,
                }
                return output

        def sync_method_wrapper(*args, **kwargs):
            self._plan.runtime.action_counter += 1
            tool_name = f"{self._handle_name}:{self._handle_id}.{name}"
            call_repr = f"{self._handle_name}[{self._handle_id[:8]}].{name}({self._plan.actor._serialize_args(args, kwargs)})"

            cache_key = self._plan.actor._generate_cache_key(
                self._plan,
                tool_name,
                args,
                kwargs,
            )

            if cache_key in self._plan.idempotency_cache:
                cached_result = handle_method_logic(
                    call_repr,
                    True,
                    self._plan.idempotency_cache[cache_key],
                    tool_name,
                    cache_key,
                )
                if cached_result is not None:
                    return cached_result

            if self._plan.runtime.cache_miss_counter:
                self._plan.runtime.cache_miss_counter[-1] += 1
            self._plan.action_log.append(f"CACHE MISS: Executing {call_repr}")
            logger.debug(f"HANDLE CACHE MISS for: {call_repr}")

            output = real_attr(*args, **kwargs)
            # Special-case `.result()`: best-effort capture of sub-loop history (when available).
            if name == "result":
                sub_loop_history: list[Any] = []
                if hasattr(self._real_handle, "get_history"):
                    try:
                        sub_loop_history = self._real_handle.get_history()  # type: ignore[attr-defined]
                    except Exception:
                        sub_loop_history = []
                interaction_to_cache = (
                    "handle_method_call",
                    call_repr,
                    str(output),
                    sub_loop_history,
                )
            else:
                interaction_to_cache = ("handle_method_call", call_repr, str(output))
            interactions_log = current_interaction_sink_var.get()
            if interactions_log is not None:
                interactions_log.append(interaction_to_cache)

            meta = {
                "call_stack": cache_key[0],
                "path": cache_key[1],
                "function": cache_key[0][-1] if cache_key[0] else None,
                "step": self._plan.runtime.action_counter,
                "tool": tool_name,
                "url": None,
                "impure": False,
            }

            self._plan.idempotency_cache[cache_key] = {
                "result": output,
                "interaction_log": interaction_to_cache,
                "meta": meta,
            }
            return output

        return (
            async_method_wrapper
            if inspect.iscoroutinefunction(real_attr)
            else sync_method_wrapper
        )


class _ToolProviderProxy:
    """
    Generic proxy for any tool provider (browser, state managers, and future custom
    environments).

    Intercepts tool calls to apply:
    - idempotency caching (multi-dimensional: stack/loop/path/step)
    - interaction logging for verification
    - steerable handle tracking
    - run-id gating to prevent stale calls

    Browser-specific features are conditionally enabled when the provided
    `environment` looks like a `ComputerEnvironment`:
    - Magnitude log capture (if the browser backend is `MagnitudeBrowserBackend`)
    - `wait`/`context` kwargs injection
    - URL + post-action screenshot capture for cache metadata
    - scoped context injection into the `reason` tool
    """

    def __init__(
        self,
        real_instance: Any,
        plan: "HierarchicalActorHandle",
        namespace: str,
        environment: BaseEnvironment | None = None,
    ):
        self._real_instance = real_instance
        self._plan = plan
        self._namespace = namespace
        self._environment = environment
        self._is_browser_env = (
            environment is not None
            and environment.__class__.__name__ == "ComputerEnvironment"
        )

    def _handle_name_for_tool(self, tool_method_name: str) -> str:
        # Preserve historical naming for browser primitives and manager primitives.
        if self._is_browser_env:
            return f"{tool_method_name}_handle"
        manager_name = self._namespace.split(".")[-1]
        return f"{manager_name}_{tool_method_name}_handle"

    def __getattr__(self, name: str) -> Any:
        real_attr = getattr(self._real_instance, name)

        # Special-case: the conversation manager handle must remain steerable and tracked.
        if name == "conversation_manager" and isinstance(
            real_attr,
            ConversationManagerHandle,
        ):
            handle_id = "cm_handle"
            if handle_id not in self._plan.live_handles:
                self._plan.live_handles[handle_id] = real_attr
            return _SteerableToolHandleProxy(
                real_attr,
                self._plan,
                "conversation_manager",
                handle_id,
            )

        if not callable(real_attr):
            return real_attr

        async def async_wrapper(*args, **kwargs):
            # Browser-only: inject scoped local context into reason() calls.
            if self._is_browser_env and name == "reason":
                try:
                    scoped_context_dict = (
                        self._plan.actor._get_scoped_context_from_plan_state(self._plan)
                    )
                    scoped_context_str = (
                        self._plan.actor._format_scoped_context_for_prompt(
                            scoped_context_dict,
                        )
                    )

                    user_provided_context = kwargs.get("context", "")

                    kwargs["context"] = (
                        "###CALL STACK CONTEXT\n"
                        "The following is a scoped view of the plan's source code, centered on the current point of execution.\n"
                        "Use this to understand the local context and make your decision.\n\n"
                        f"{scoped_context_str}\n\n---\n\n"
                        "### USER-PROVIDED CONTEXT\n"
                        f"{user_provided_context}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to inject scoped context into 'reason' tool: {e}",
                    )

            # Browser-only: support `wait=` without polluting cache keys.
            wait = True
            if self._is_browser_env:
                wait = kwargs.pop("wait", True)

            ctx_run_id = current_run_id_var.get()
            plan_run_id = self._plan.run_id
            if ctx_run_id != plan_run_id:
                logger.warning(
                    f"Blocked stale tool call to '{self._namespace}.{name}' "
                    f"(context run_id={ctx_run_id} != plan run_id={plan_run_id}).",
                )
                self._plan.action_log.append(
                    f"Blocked stale tool call to '{self._namespace}.{name}' (context run_id={ctx_run_id} != plan run_id={plan_run_id}).",
                )
                raise asyncio.CancelledError("Stale tool call blocked by run_id gate.")

            interactions_log = current_interaction_sink_var.get()
            self._plan.runtime.action_counter += 1

            tool_name = f"{self._namespace}.{name}"

            # Inject clarification channels into manager calls when the plan supports
            # clarifications. This enables nested manager tool-loops to request
            # clarification without requiring plan code to thread queue objects.
            #
            # IMPORTANT: do NOT allow these queue objects to pollute cache keys or logs.
            clarification_injected = False
            if (not self._is_browser_env) and getattr(
                self._plan,
                "clarification_enabled",
                False,
            ):
                try:
                    if (
                        "_clarification_up_q" not in kwargs
                        and "_clarification_down_q" not in kwargs
                    ):
                        kwargs["_clarification_up_q"] = getattr(
                            self._plan,
                            "clarification_up_q",
                            None,
                        )
                        kwargs["_clarification_down_q"] = getattr(
                            self._plan,
                            "clarification_down_q",
                            None,
                        )
                        clarification_injected = True
                except Exception:
                    clarification_injected = False

            kwargs_for_cache = dict(kwargs)
            if clarification_injected:
                kwargs_for_cache.pop("_clarification_up_q", None)
                kwargs_for_cache.pop("_clarification_down_q", None)

            if DIAGNOSTIC_MODE:
                run_id = current_run_id_var.get()
                invoc_id = current_invocation_id_var.get()
                diag_prefix = f"[run_id={run_id} invoc={invoc_id}]"
            else:
                diag_prefix = ""

            call_repr = f"{tool_name}({self._plan.actor._serialize_args(args, kwargs_for_cache)})"
            if diag_prefix:
                if self._is_browser_env:
                    logger.info(
                        f"{diag_prefix} 🐍 PYTHON: Executing {call_repr} (wait={wait})",
                    )
                else:
                    logger.info(f"{diag_prefix} 🐍 PYTHON: Executing {call_repr}")

            cache_key = self._plan.actor._generate_cache_key(
                self._plan,
                tool_name,
                args,
                kwargs_for_cache,
            )

            if cache_key in self._plan.idempotency_cache:
                cached_data = self._plan.idempotency_cache[cache_key]
                cached_result_id = cached_data["result"]
                cached_interaction = cached_data["interaction_log"]

                if (
                    self._is_browser_env
                    and isinstance(cached_interaction, tuple)
                    and len(cached_interaction) < 4
                ):
                    cached_interaction = (*cached_interaction, [])

                self._plan.action_log.append(
                    (
                        f"{diag_prefix} CACHE HIT: Using cached result for {call_repr}"
                        if diag_prefix
                        else f"CACHE HIT: Using cached result for {call_repr}"
                    ),
                )
                logger.debug(f"{diag_prefix} CACHE HIT for key: {cache_key}")
                if interactions_log is not None:
                    interactions_log.append(cached_interaction)

                if (
                    isinstance(cached_interaction, tuple)
                    and len(cached_interaction) >= 3
                    and cached_interaction[0] == "tool_call"
                    and "Returned handle" in str(cached_interaction[2])
                ):
                    real_handle = self._plan.live_handles.get(cached_result_id)
                    if not real_handle:
                        raise RuntimeError(
                            f"Cache consistency error: Could not find live handle for ID {cached_result_id}",
                        )
                    handle_name = self._handle_name_for_tool(name)
                    return _SteerableToolHandleProxy(
                        real_handle,
                        self._plan,
                        handle_name,
                        cached_result_id,
                    )

                return cached_result_id

            if self._plan.runtime.cache_miss_counter:
                self._plan.runtime.cache_miss_counter[-1] += 1
            self._plan.action_log.append(
                (
                    f"{diag_prefix} CACHE MISS: Executing {call_repr}"
                    if diag_prefix
                    else f"CACHE MISS: Executing {call_repr}"
                ),
            )
            logger.debug(f"{diag_prefix} CACHE MISS for key: {cache_key}")

            magnitude_logs: list[Any] = []
            backend = None
            is_magnitude = False

            if self._is_browser_env:
                try:
                    backend = self._real_instance.browser.backend
                    is_magnitude = isinstance(backend, MagnitudeBrowserBackend)
                except Exception:
                    backend = None
                    is_magnitude = False

            if is_magnitude and backend is not None:
                capture_q = asyncio.Queue()
                backend._current_capture_queue = capture_q
            else:
                capture_q = None

            try:
                if self._is_browser_env and name != "reason":
                    func_name = (
                        self._plan.call_stack[-1] if self._plan.call_stack else "global"
                    )
                    context = {"function_name": func_name, "run_id": self._plan.run_id}
                    tool_output = await real_attr(
                        *args,
                        wait=wait,
                        context=context,
                        **kwargs,
                    )
                else:
                    tool_output = await real_attr(*args, **kwargs)
            except BrowserAgentError as e:
                if self._is_browser_env and e.error_type == "cancelled":
                    logger.info(
                        f"🔴 Action interrupted by immediate pause: {call_repr}",
                    )
                    raise _ControlledInterruptionException(
                        f"Action '{call_repr}' interrupted by immediate pause.",
                    )
                raise
            finally:
                if is_magnitude and backend is not None and capture_q is not None:
                    # Allow final logs to flush into the capture queue.
                    await asyncio.sleep(0.25)

                    backend._current_capture_queue = None
                    while not capture_q.empty():
                        magnitude_logs.append(capture_q.get_nowait())

            result_to_cache = tool_output
            return_value = tool_output
            interaction_str = str(tool_output)

            if isinstance(tool_output, SteerableToolHandle):
                handle_name = self._handle_name_for_tool(name)
                handle_id = str(uuid.uuid4())
                self._plan.live_handles[handle_id] = tool_output
                result_to_cache = handle_id
                interaction_str = f"Returned handle: {tool_output.__class__.__name__}"
                return_value = _SteerableToolHandleProxy(
                    tool_output,
                    self._plan,
                    handle_name,
                    handle_id,
                )

                # Register new handle with pane (best-effort).
                try:
                    if getattr(self._plan, "pane", None) is not None:
                        capabilities: list[str] = []
                        if hasattr(tool_output, "interject"):
                            capabilities.append("interjectable")
                        if hasattr(tool_output, "pause") and hasattr(
                            tool_output,
                            "resume",
                        ):
                            capabilities.append("pausable")
                        if hasattr(tool_output, "ask"):
                            capabilities.append("askable")
                        if hasattr(tool_output, "stop"):
                            capabilities.append("stoppable")
                        if hasattr(tool_output, "answer_clarification"):
                            capabilities.append("clarifiable")

                        await self._plan.pane.register_handle(
                            handle=tool_output,
                            handle_id=handle_id,
                            parent_handle_id=None,
                            origin_tool=tool_name,
                            origin_step=self._plan.runtime.action_counter,
                            environment_namespace=self._namespace,
                            capabilities=capabilities,
                            call_stack=str(cache_key[0]) if cache_key else None,
                        )
                except Exception as e:
                    logger.debug(f"Pane registration failed for tool handle: {e}")

            if self._is_browser_env:
                interaction_to_cache = (
                    "tool_call",
                    call_repr,
                    interaction_str,
                    magnitude_logs,
                )
            else:
                interaction_to_cache = ("tool_call", call_repr, interaction_str)

            if interactions_log is not None:
                interactions_log.append(interaction_to_cache)

            url = None
            if self._is_browser_env:
                try:
                    url = await self._real_instance.browser.get_current_url()
                except Exception:
                    url = None

            meta_impure = False
            try:
                meta_obj = self._plan.actor.tool_metadata.get(tool_name)
                if meta_obj is not None:
                    meta_impure = bool(getattr(meta_obj, "is_impure", False))
            except Exception:
                pass

            meta: dict[str, Any] = {
                "call_stack": cache_key[0],
                "path": cache_key[1],
                "function": cache_key[0][-1] if cache_key[0] else None,
                "step": self._plan.runtime.action_counter,
                "tool": tool_name,
                "url": url,
                "impure": meta_impure,
            }

            if meta["impure"] and self._is_browser_env:
                try:
                    post_screenshot = await self._real_instance.browser.get_screenshot()
                    meta["post_state_screenshot"] = post_screenshot
                except Exception as e:
                    logger.warning(f"Failed to capture post-action screenshot: {e}")

            self._plan.idempotency_cache[cache_key] = {
                "result": result_to_cache,
                "interaction_log": interaction_to_cache,
                "meta": meta,
            }

            return return_value

        def sync_wrapper(*args, **kwargs):
            """Synchronous wrapper for logging and calling sync tools."""
            ctx_run_id = current_run_id_var.get()
            plan_run_id = self._plan.run_id
            if ctx_run_id != plan_run_id:
                logger.warning(
                    f"Blocked stale tool call to '{self._namespace}.{name}' "
                    f"(context run_id={ctx_run_id} != plan run_id={plan_run_id}).",
                )
                self._plan.action_log.append(
                    f"Blocked stale tool call to '{self._namespace}.{name}' (context run_id={ctx_run_id} != plan run_id={plan_run_id}).",
                )
                raise asyncio.CancelledError("Stale tool call blocked by run_id gate.")

            interactions_log = current_interaction_sink_var.get()
            self._plan.runtime.action_counter += 1
            tool_name = f"{self._namespace}.{name}"

            if DIAGNOSTIC_MODE:
                run_id = current_run_id_var.get()
                invoc_id = current_invocation_id_var.get()
                diag_prefix = f"[run_id={run_id} invoc={invoc_id}]"
            else:
                diag_prefix = ""

            call_repr = f"{tool_name}({self._plan.actor._serialize_args(args, kwargs)})"
            if diag_prefix:
                logger.info(f"{diag_prefix} 🐍 PYTHON: Executing {call_repr}")

            cache_key = self._plan.actor._generate_cache_key(
                self._plan,
                tool_name,
                args,
                kwargs,
            )

            if cache_key in self._plan.idempotency_cache:
                cached_data = self._plan.idempotency_cache[cache_key]
                cached_result_id = cached_data["result"]
                cached_interaction = cached_data["interaction_log"]
                self._plan.action_log.append(
                    (
                        f"{diag_prefix} CACHE HIT: Using cached result for {call_repr}"
                        if diag_prefix
                        else f"CACHE HIT: Using cached result for {call_repr}"
                    ),
                )
                logger.debug(f"{diag_prefix} CACHE HIT for key: {cache_key}")
                if interactions_log is not None:
                    interactions_log.append(cached_interaction)

                if (
                    isinstance(cached_interaction, tuple)
                    and len(cached_interaction) >= 3
                    and cached_interaction[0] == "tool_call"
                    and "Returned handle" in str(cached_interaction[2])
                ):
                    real_handle = self._plan.live_handles.get(cached_result_id)
                    if not real_handle:
                        raise RuntimeError(
                            f"Cache consistency error: Could not find live handle for ID {cached_result_id}",
                        )
                    handle_name = self._handle_name_for_tool(name)
                    return _SteerableToolHandleProxy(
                        real_handle,
                        self._plan,
                        handle_name,
                        cached_result_id,
                    )
                return cached_result_id

            if self._plan.runtime.cache_miss_counter:
                self._plan.runtime.cache_miss_counter[-1] += 1
            self._plan.action_log.append(
                (
                    f"{diag_prefix} CACHE MISS: Executing {call_repr}"
                    if diag_prefix
                    else f"CACHE MISS: Executing {call_repr}"
                ),
            )
            logger.debug(f"{diag_prefix} CACHE MISS for key: {cache_key}")

            result = real_attr(*args, **kwargs)

            result_to_cache = result
            return_value = result
            interaction_str = str(result)

            if isinstance(result, SteerableToolHandle):
                handle_name = self._handle_name_for_tool(name)
                handle_id = str(uuid.uuid4())
                self._plan.live_handles[handle_id] = result
                result_to_cache = handle_id
                interaction_str = f"Returned handle: {result.__class__.__name__}"
                return_value = _SteerableToolHandleProxy(
                    result,
                    self._plan,
                    handle_name,
                    handle_id,
                )

                # Register new handle with pane (best-effort). sync_wrapper can't await.
                try:
                    if getattr(self._plan, "pane", None) is not None:
                        capabilities: list[str] = []
                        if hasattr(result, "interject"):
                            capabilities.append("interjectable")
                        if hasattr(result, "pause") and hasattr(result, "resume"):
                            capabilities.append("pausable")
                        if hasattr(result, "ask"):
                            capabilities.append("askable")
                        if hasattr(result, "stop"):
                            capabilities.append("stoppable")
                        if hasattr(result, "answer_clarification"):
                            capabilities.append("clarifiable")

                        t = asyncio.create_task(
                            self._plan.pane.register_handle(
                                handle=result,
                                handle_id=handle_id,
                                parent_handle_id=None,
                                origin_tool=tool_name,
                                origin_step=self._plan.runtime.action_counter,
                                environment_namespace=self._namespace,
                                capabilities=capabilities,
                                call_stack=str(cache_key[0]) if cache_key else None,
                            ),
                            name=f"pane_register_{handle_id[:8]}",
                        )
                        self._plan._child_tasks.add(t)
                except Exception:
                    pass

            interaction_to_cache = ("tool_call", call_repr, interaction_str)
            if interactions_log is not None:
                interactions_log.append(interaction_to_cache)

            meta_impure = False
            try:
                meta_obj = self._plan.actor.tool_metadata.get(tool_name)
                if meta_obj is not None:
                    meta_impure = bool(getattr(meta_obj, "is_impure", False))
            except Exception:
                pass

            meta = {
                "call_stack": cache_key[0],
                "path": cache_key[1],
                "function": cache_key[0][-1] if cache_key[0] else None,
                "step": self._plan.runtime.action_counter,
                "tool": tool_name,
                "url": None,
                "impure": meta_impure,
            }

            self._plan.idempotency_cache[cache_key] = {
                "result": result_to_cache,
                "interaction_log": interaction_to_cache,
                "meta": meta,
            }

            return return_value

        return async_wrapper if inspect.iscoroutinefunction(real_attr) else sync_wrapper


class _PrimitivesProxy:
    """Proxy for `unity.function_manager.primitives.Primitives`.\n\n    Exposes managers under `primitives.<manager>` and wraps each manager with\n    `_ToolProviderProxy` to ensure calls are logged/cached/handle-tracked.\n\n    Special-case: `primitives.computer` returns the already-injected\n    `computer_primitives` proxy so browser tools remain properly proxied.\n"""

    def __init__(self, real_primitives: Any, plan: "HierarchicalActorHandle"):
        self._real_primitives = real_primitives
        self._plan = plan

    def __getattr__(self, name: str) -> Any:
        if name == "computer":
            # Ensure browser operations remain proxied even if plan uses primitives.computer.
            cp = self._plan.execution_namespace.get("computer_primitives")
            if cp is not None:
                return cp
        real_attr = getattr(self._real_primitives, name)
        # Wrap managers (they are objects with .ask/.update/... methods).
        if name in {
            "contacts",
            "transcripts",
            "knowledge",
            "tasks",
            "secrets",
            "guidance",
            "web",
            "files",
        }:
            env = None
            if hasattr(self._plan, "environments") and isinstance(
                getattr(self._plan, "environments", None),
                dict,
            ):
                env = self._plan.environments.get("primitives")
            elif hasattr(self._plan.actor, "environments"):
                env = self._plan.actor.environments.get("primitives")
            return _ToolProviderProxy(
                real_instance=real_attr,
                plan=self._plan,
                namespace=f"primitives.{name}",
                environment=env,
            )
        return real_attr


class _VenvFunctionProxy:
    """
    A proxy that wraps venv functions as atomic, opaque callables.

    Venv functions run in a separate subprocess via FunctionManager.execute_in_venv.
    They are treated like external primitives - visible to the LLM (name, argspec,
    docstring) but not steppable or verifiable at the code level.

    This allows HierarchicalActor to use venv functions alongside regular functions,
    while respecting the isolation requirements of custom virtual environments.
    """

    def __init__(
        self,
        function_manager: "FunctionManager",
        func_data: Dict[str, Any],
        plan: "HierarchicalActorHandle",
        primitives: Any,
        computer_primitives: Any,
    ):
        self._function_manager = function_manager
        self._func_data = func_data
        self._plan = plan
        self._primitives = primitives
        self._computer_primitives = computer_primitives
        self._name = func_data.get("name", "unknown")
        self._venv_id = func_data.get("venv_id")
        self._implementation = func_data.get("implementation", "")
        self._docstring = func_data.get("docstring", "")
        self._argspec = func_data.get("argspec", "")

        # Set function metadata for introspection
        self.__name__ = self._name
        self.__doc__ = self._docstring

    async def __call__(self, **kwargs) -> Any:
        """
        Execute the venv function via subprocess.

        This method is called when the generated plan invokes the function.
        It uses FunctionManager.execute_in_venv for isolated execution.
        """
        # Log the call for verification/debugging
        call_repr = (
            f"{self._name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
        )
        self._plan.action_log.append(f"Venv function call: {call_repr}")

        # Determine if function is async (check implementation for 'async def')
        is_async = "async def " in self._implementation

        try:
            result = await self._function_manager.execute_in_venv(
                venv_id=self._venv_id,
                implementation=self._implementation,
                call_kwargs=kwargs,
                is_async=is_async,
                primitives=self._primitives,
                computer_primitives=self._computer_primitives,
            )

            # Log the result
            if result.get("error"):
                error_msg = f"Venv function error: {result['error']}"
                self._plan.action_log.append(error_msg)
                if result.get("stderr"):
                    self._plan.action_log.append(f"Stderr: {result['stderr']}")
                raise RuntimeError(result["error"])

            if result.get("stdout"):
                self._plan.action_log.append(f"Venv stdout: {result['stdout']}")

            return result.get("result")

        except Exception as e:
            self._plan.action_log.append(f"Venv function exception: {e}")
            raise


class HierarchicalActorHandle(BaseActiveTask, BaseActorHandle):
    """
    Represents and executes a single, dynamically generated hierarchical plan.

    This class is a steerable handle managing the plan's lifecycle, including
    generation, execution, self-correction, and modification.
    """

    def __init__(
        self,
        actor: "HierarchicalActor",
        goal: Optional[str] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        parent_chat_context: Optional[str] = None,
        max_escalations: Optional[int] = None,
        max_local_retries: Optional[int] = None,
        persist: bool = True,
        images: Optional[dict[str, Any]] = None,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        dedicated_computer_primitives: Optional[ComputerPrimitives] = None,
        can_compose: bool = True,
        can_store: bool = True,
    ):
        """
        Initializes the Hierarchical Plan active task.

        Args:
            actor: The parent HierarchicalActor instance.
            goal: The high-level user goal for this plan.
            clarification_up_q: Queue for sending clarification questions to the user.
            clarification_down_q: Queue for receiving answers from the user.
            parent_chat_context: The chat context from a parent process, if any.
            max_escalations: Max number of strategic replans before pausing.
            max_local_retries: Max number of tactical retries for a function.
            persist: If True, plan will pause for interjections after completion. If False, plan will complete immediately.
            images: Optional mapping of source-scoped keys to ImageHandle objects.
            entrypoint: Optional. If provided, bypasses LLM plan generation
                and directly executes the function from the FunctionManager as the main plan.
            dedicated_computer_primitives: Optional. If provided, use this action provider for the plan.
                If not provided, use the actor's action provider instead.
            can_compose: When True, allows the plan to generate new code on the fly.
                When False, only pre-existing functions can be executed via entrypoint.
            can_store: When True, allows verified functions to be persisted to FunctionManager.
                When False, functions are executed but not stored.
        """
        self.actor = actor
        self.goal = goal
        self.images = images or {}
        self.plan_source_code: Optional[str] = None
        self.execution_namespace: Dict[str, Any] = {}
        self.persist = persist
        self.entrypoint = entrypoint
        self.entrypoint_args = entrypoint_args or []
        self.entrypoint_kwargs = entrypoint_kwargs or {}
        self.dedicated_computer_primitives = dedicated_computer_primitives
        self.can_compose = can_compose
        self.can_store = can_store

        # Plan-local environments map (used for proxy injection + verification evidence).
        # This must reflect the plan's dedicated computer primitives (when provided),
        # rather than the actor's shared browser session.
        self.environments: dict[str, BaseEnvironment] = {}
        try:
            if hasattr(actor, "environments") and isinstance(actor.environments, dict):
                # Shallow copy is fine: environments are lightweight descriptors.
                self.environments = dict(actor.environments)
        except Exception:
            self.environments = {}

        if self.dedicated_computer_primitives is not None:
            # Only override the browser environment when a dedicated session is used.
            # This ensures env.capture_state() reflects the dedicated session.
            from unity.actor.environments import ComputerEnvironment

            self.environments["computer_primitives"] = ComputerEnvironment(
                self.dedicated_computer_primitives,
            )

        self.idempotency_cache: Dict[tuple, Any] = {}
        self.live_handles: Dict[str, SteerableToolHandle] = {}
        self.runtime = PlanRuntime()
        self.call_stack: List[str] = []
        self.action_log: List[str] = []
        self._pane_supervisor_task: Optional[asyncio.Task] = None
        self.last_verified_function_name: Optional[str] = None
        self.last_verified_url: Optional[str] = None
        self.last_verified_screenshot: Optional[str | bytes] = None
        self.function_source_map: Dict[str, str] = {}
        self.clean_function_source_map: Dict[str, str] = {}
        self.top_level_function_names: set[str] = set()
        self.functions_skip_verify: set[str] = set()
        self.interaction_stack: List[List[Tuple[str, str, Optional[str]]]] = []
        self.escalation_count = 0
        self._is_complete = False
        self._execution_task: Optional[asyncio.Task] = None
        self._state = _HierarchicalHandleState.IDLE

        self.run_id = 0
        self.invocation_counter = 0
        self.interruption_request: Optional[Dict[str, str]] = None
        self._interject_lock = asyncio.Lock()
        self._completion_event = asyncio.Event()
        self.skipped_functions: set = set()

        if self.persist:
            self._done_events = asyncio.Queue()
            self.cumulative_interactions: List[Tuple] = []
            self._last_summarized_interaction_count: int = 0

        self._child_tasks: set[asyncio.Task] = set()

        # One pane per actor run (deterministic runtime component).
        self.pane = SteerableToolPane(run_id=str(self.run_id))

        self.verif_seq: int = 0
        self.pending_verifications: "OrderedDict[int, VerificationHandle]" = (
            OrderedDict()
        )
        self._verification_lock = asyncio.Lock()
        self._recovery_task: Optional[asyncio.Task] = None
        self._recovery_target_ordinal: Optional[int] = None
        self.is_verifying_post_completion: bool = False
        self._recovery_in_progress: bool = False

        self._execution_task = asyncio.create_task(self._initialize_and_run())
        self.MAX_ESCALATIONS = max_escalations or 2
        self.MAX_LOCAL_RETRIES = max_local_retries or 3

        self._temp_file_path: Optional[Path] = None
        self._module_name: str = f"hp_plan_{uuid.uuid4().hex}"
        self._module: Optional[types.ModuleType] = None
        self._module_spec: Optional[importlib.machinery.ModuleSpec] = None

        self._final_result_str: Optional[str] = None
        self.parent_chat_context = parent_chat_context
        self.clarification_up_q = clarification_up_q
        self.clarification_down_q = clarification_down_q
        self.clarification_enabled = (
            clarification_up_q is not None and clarification_down_q is not None
        )

        self._img_token = None
        self._imglog_token = None
        if self.images:
            id_map: dict[int, Any] = {}
            for _k, _ih in self.images.items():
                try:
                    _iid = int(getattr(_ih, "image_id", -1))
                    if _iid >= 0:
                        id_map[_iid] = _ih
                except Exception:
                    continue
            self._img_token = LIVE_IMAGES_REGISTRY.set(id_map)

            seed_log: list[str] = []
            try:
                for _k, _ih in self.images.items():
                    try:
                        _iid = int(getattr(_ih, "image_id", -1))
                    except Exception:
                        _iid = -1
                    seed_log.append(f"user_message:{_iid}:{_k}")
            except Exception:
                pass
            self._imglog_token = LIVE_IMAGES_LOG.set(seed_log)

        self.plan_generation_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )
        self.verification_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )
        self.implementation_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )
        self.summarization_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )
        self.modification_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )
        self.ask_client: unillm.AsyncUnify = new_llm_client(
            return_full_completion=True,
            reasoning_effort=None,
            service_tier=None,
        )

    def _get_computer_primitives(self) -> ComputerPrimitives:
        cp = (
            self.dedicated_computer_primitives
            if self.dedicated_computer_primitives is not None
            else getattr(self.actor, "computer_primitives", None)
        )
        if cp is None:
            raise RuntimeError(
                "computer_primitives is not available: this actor/plan was configured without a browser environment.",
            )
        return cp

    def _set_final_result(self, result: str):
        """Sets the final result and the completion event."""
        if not self._completion_event.is_set():
            self._final_result_str = result
            self._is_complete = True
            self._completion_event.set()

    def _set_state(self, new_state: _HierarchicalHandleState):
        """Sets the plan state and logs the transition."""
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        stack = traceback.format_stack()
        caller_info = " -> ".join(
            line.strip().split("\n")[-1].strip() for line in stack[-3:-1]
        )
        logger.debug(
            f"STATE_TRANSITION: {old_state.name} -> {new_state.name} (Caller: {caller_info})",
        )
        self.action_log.append(f"STATE CHANGE: {old_state.name} -> {new_state.name}")

    async def _initialize_and_run(
        self,
        mode: str = "fresh_start",
    ):
        """
        Manages the entire lifecycle of the plan from initialization to completion.
        """
        token = current_run_id_var.set(self.run_id)
        delegate_token = None
        try:
            delegate: TaskExecutionDelegate = _HierarchicalActorDelegate(
                actor=self.actor,
                handle=self,
            )
            delegate_token = current_task_execution_delegate.set(delegate)
        except Exception:
            delegate_token = None

        self.runtime.execution_mode = mode
        try:
            # Ensure pane run_id stays correlated with the plan run_id (run_id may change across reruns).
            try:
                self.pane.run_id = str(self.run_id)
            except Exception:
                pass

            if not self._is_complete:
                self._set_state(_HierarchicalHandleState.RUNNING)

            # Start the pane supervisor early so it can react to clarifications during execution.
            if self._pane_supervisor_task is None or self._pane_supervisor_task.done():
                self._pane_supervisor_task = asyncio.create_task(
                    self._pane_event_supervisor(),
                    name=f"PaneSupervisor-{self._module_name}",
                )
                self._child_tasks.add(self._pane_supervisor_task)

            if self.plan_source_code is None:
                # Determine the entrypoint: explicit, semantic search, or generate plan
                resolved_entrypoint = self.entrypoint
                entrypoint_via_search = False

                if resolved_entrypoint is None and not self.can_compose:
                    # can_compose=False with no explicit entrypoint: use semantic search
                    if not self.actor.function_manager:
                        raise ValueError(
                            "can_compose=False requires a FunctionManager to search for functions.",
                        )
                    self.action_log.append(
                        f"Searching for best matching function for goal: '{self.goal}'",
                    )
                    search_results = (
                        self.actor.function_manager.search_functions_by_similarity(
                            query=self.goal,
                            n=1,
                            include_primitives=False,
                        )
                    )
                    if not search_results:
                        raise ValueError(
                            f"No functions found matching goal: '{self.goal}'. "
                            f"Add functions to FunctionManager or set can_compose=True.",
                        )
                    best_match = search_results[0]
                    resolved_entrypoint = best_match.get("function_id")
                    entrypoint_via_search = True
                    self.action_log.append(
                        f"Selected function '{best_match.get('name')}' (ID: {resolved_entrypoint}) via semantic search.",
                    )

                if resolved_entrypoint is not None:
                    if not entrypoint_via_search:
                        self.action_log.append(
                            f"Bypassing LLM generation. Using entrypoint function_id {resolved_entrypoint}.",
                        )
                    if not self.actor.function_manager:
                        raise ValueError(
                            "Entrypoint was provided, but no FunctionManager is available to fetch the function.",
                        )

                    search_results = self.actor.function_manager.search_functions(
                        filter=f"function_id == {resolved_entrypoint}",
                        limit=1,
                    )
                    if not search_results:
                        raise ValueError(
                            f"Entrypoint function_id {resolved_entrypoint} not found in FunctionManager.",
                        )

                    entrypoint_func_data = search_results[0]
                    entrypoint_code = entrypoint_func_data.get("implementation")
                    entrypoint_name = entrypoint_func_data.get("name")
                    entrypoint_verify = entrypoint_func_data.get("verify", True)

                    if not entrypoint_code or not entrypoint_name:
                        raise ValueError(
                            f"Invalid function data for entrypoint {resolved_entrypoint}.",
                        )

                    # Skip verification for entrypoint wrapper if verify=False
                    if not entrypoint_verify:
                        self.functions_skip_verify.add(entrypoint_name)
                        # self.functions_skip_verify.add("main_plan")

                    # Build argument string for entrypoint call from plan-provided args/kwargs
                    def _render_value(v: Any) -> str:
                        return repr(v)

                    args_list = [_render_value(v) for v in (self.entrypoint_args or [])]
                    for k, v in (self.entrypoint_kwargs or {}).items():
                        args_list.append(f"{k}={_render_value(v)}")
                    rendered_args = ", ".join(args_list)

                    synthetic_main = f"""
async def main_plan():
    '''Auto-generated main_plan to run entrypoint function {entrypoint_name}.'''
    return await {entrypoint_name}({rendered_args})
"""
                    base_code = f"{entrypoint_code}\n\n{synthetic_main}"

                    self.action_log.append(
                        f"Injecting entrypoint '{entrypoint_name}' and its dependencies.",
                    )
                    full_code, skip_verify = await self.actor._inject_library_functions(
                        base_code,
                    )
                    self.functions_skip_verify.update(skip_verify)

                    self.plan_source_code = self.actor._sanitize_code(full_code, self)
                    self.action_log.append("Entrypoint plan sanitized and ready.")

                else:
                    # can_compose=True: generate plan from goal
                    self.action_log.append("Generating plan from goal...")
                    self.plan_source_code = await self.actor._generate_initial_plan(
                        plan=self,
                        goal=self.goal,
                    )
                    self.action_log.append("Initial plan generated successfully.")
            else:
                self.action_log.append("Proceeding with existing plan source code.")

            await self.actor._prepare_execution_environment(self)
            await self._start_main_execution_loop()
        except Exception as e:
            logger.error(f"Plan initialization failed: {e}", exc_info=True)
            self._set_state(_HierarchicalHandleState.ERROR)
            self._set_final_result(f"ERROR: Plan initialization failed: {e}")
        finally:
            try:
                current_run_id_var.reset(token)
            except Exception:
                pass

            try:
                if delegate_token is not None:
                    current_task_execution_delegate.reset(delegate_token)
            except Exception:
                pass

    async def _pane_event_supervisor(self) -> None:
        """Concurrent supervisor reacting to pane events.

        Policy: be conservative for clarifications. If a clarification channel exists,
        pause the plan, ask the user, forward the answer via `pane.answer_clarification`,
        then resume. Notifications are logged (no additional LLM turns).
        """

        try:
            while not self._completion_event.is_set():
                wake = asyncio.create_task(self.pane._events_q.get())
                done, pending = await asyncio.wait(
                    {wake, asyncio.create_task(self._completion_event.wait())},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

                # Completed: exit
                if wake not in done:
                    break

                try:
                    idx = wake.result()
                except Exception:
                    continue

                try:
                    event = self.pane._events_log[idx]
                except Exception:
                    continue

                if event.get("type") == "steering_applied":
                    payload = event.get("payload") or {}
                    method = payload.get("method")
                    status = payload.get("status")
                    handle_id = event.get("handle_id")

                    interactions_log = current_interaction_sink_var.get()
                    if interactions_log is not None:
                        interactions_log.append(
                            (
                                "pane_steering",
                                str(method),
                                f"handle_id={handle_id} status={status}",
                            ),
                        )
                    try:
                        self.action_log.append(
                            f"PANE steering_applied: method={method} handle_id={handle_id} status={status}",
                        )
                    except Exception:
                        pass

                elif event.get("type") == "clarification":
                    await self._handle_pane_clarification_event(event)
                elif event.get("type") == "notification":
                    try:
                        self.action_log.append(
                            f"PANE notification from {event.get('handle_id')}: {event.get('payload')}",
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.debug(f"Pane supervisor failed: {e}", exc_info=True)

    async def _handle_pane_clarification_event(self, event: dict[str, Any]) -> None:
        """Handle a single clarification event from the pane."""

        handle_id = str(event.get("handle_id", ""))
        payload = event.get("payload") or {}
        call_id = str(payload.get("call_id", ""))
        question = str(payload.get("question", ""))
        origin_tool = ""
        try:
            origin = event.get("origin") or {}
            origin_tool = str(origin.get("origin_tool", ""))
        except Exception:
            origin_tool = ""

        self.action_log.append(
            f"PANE clarification from {origin_tool or '<unknown>'}: {question}",
        )

        if self.clarification_enabled:
            # Pause the plan (non-immediate: avoids browser interrupt assumptions).
            with contextlib.suppress(Exception):
                await self.pause(immediate=False)

            # Ask user via queues, then answer the clarification.
            await self.clarification_up_q.put(question)
            user_answer = await self.clarification_down_q.get()
            self.action_log.append(
                f"PANE clarification answered by user: {user_answer}",
            )

            # Forward into the specific in-flight handle via the pane.
            await self.pane.answer_clarification(handle_id, call_id, user_answer)

            # Also record as an interaction for verification traces (best-effort).
            interactions_log = current_interaction_sink_var.get()
            if interactions_log is not None:
                interactions_log.append(
                    (
                        "pane_steering",
                        "answer_clarification",
                        f"handle_id={handle_id} call_id={call_id}",
                    ),
                )

            with contextlib.suppress(Exception):
                await self.resume()
        else:
            # No user clarification channel available: best-effort forward a neutral response.
            await self.pane.answer_clarification(
                handle_id,
                call_id,
                "No user clarification channel is available. Please proceed with your best judgment.",
            )

    async def _start_main_execution_loop(self):
        """
        Starts the primary, stateful execution loop that advances the plan
        one checkpoint at a time and can hold at checkpoints.
        """
        main_fn_name = self._get_main_function_name()
        if not main_fn_name:
            raise RuntimeError("Could not determine main entry point 'main_plan'.")

        main_fn = self.execution_namespace[main_fn_name]
        main_task = asyncio.create_task(
            main_fn(),
            name=f"MainPlanTask-{self._module_name}",
        )
        self._child_tasks.add(main_task)

        while not main_task.done():
            checkpoint_waiter = asyncio.create_task(
                self.runtime._checkpoint_event.wait(),
            )
            self._child_tasks.add(checkpoint_waiter)

            done, pending = await asyncio.wait(
                {main_task, checkpoint_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )

            self._child_tasks.remove(checkpoint_waiter)
            if checkpoint_waiter in pending:
                checkpoint_waiter.cancel()

            if main_task in done:
                break

            if checkpoint_waiter in done:
                if self._state == _HierarchicalHandleState.RUNNING:
                    self.runtime._release_from_checkpoint()

        if main_task in self._child_tasks:
            self._child_tasks.remove(main_task)

        try:
            result = await main_task
            self._final_result_str = str(result)

            if self.persist:
                self.action_log.append(
                    f"Main plan execution concluded with result: {result}. Verifying final steps in background...",
                )

                self.is_verifying_post_completion = True
                self._set_state(_HierarchicalHandleState.PAUSED_FOR_INTERJECTION)
                if hasattr(self, "_done_events") and not self._done_events.empty():
                    try:
                        event_to_signal = self._done_events.get_nowait()
                        event_to_signal.set()
                    except asyncio.QueueEmpty:
                        pass
                return
            else:
                self.action_log.append(
                    f"Main plan execution finished with result: {result}.",
                )
                if self.pending_verifications:
                    self.action_log.append(
                        f"Waiting for {len(self.pending_verifications)} pending verification(s)...",
                    )
                    await asyncio.gather(
                        *[h.task for h in self.pending_verifications.values()],
                        return_exceptions=True,
                    )

                    if self._recovery_in_progress:
                        self.action_log.append(
                            "Recovery initiated during final verification. Aborting completion.",
                        )
                        logger.info(
                            "Recovery initiated during final verification. Aborting plan completion to allow recovery task to proceed.",
                        )
                        return

                    self.action_log.append("All background verifications complete.")

                await self._cancel_all_background_tasks()
                self._set_state(_HierarchicalHandleState.COMPLETED)
                self._set_final_result(str(result))
                return

        except Exception as e:
            if not isinstance(e, asyncio.CancelledError):
                logger.error(
                    f"Plan execution failed with unhandled exception: {e}",
                    exc_info=True,
                )
                self._set_state(_HierarchicalHandleState.ERROR)
                self.action_log.append(f"ERROR: Plan execution failed: {e}")
                self._set_final_result(f"ERROR: Plan execution failed: {e}")

    async def _handle_dynamic_implementation(self, function_name: str, **kwargs):
        """
        Orchestrates the dynamic implementation or modification of a function.

        Args:
            function_name: The name of the function to implement.
            **kwargs: Additional context for implementation, including:
                - replan_reason: The string explaining why this is happening.
                - clarification_question: The question that was asked to the user.
                - clarification_answer: The answer received from the user.
                - call_stack_snapshot: Snapshot of the call stack at time of failure.
                - scoped_context_snapshot: Snapshot of the scoped context at time of failure.
                - ...and other existing kwargs.
        """
        if not self.can_compose:
            raise RuntimeError(
                f"Cannot dynamically implement '{function_name}': can_compose=False. "
                f"All functions must have complete implementations when this flag is disabled.",
            )

        reason = kwargs.get(
            "replan_reason",
            "First-time implementation from NotImplementedError.",
        )
        self.action_log.append(
            f"IMPLEMENTATION CONTEXT for '{function_name}': {reason}",
        )

        decision = await self.actor._dynamic_implement(
            plan=self,
            function_name=function_name,
            **kwargs,
        )

        if decision.action == "implement_function":
            self.action_log.append(
                f"Decision: Implementing function '{function_name}'. Reason: {decision.reason}",
            )
            if not decision.code:
                raise ValueError(
                    "Action 'implement_function' requires the 'code' field but it was missing.",
                )
            self._update_plan_with_new_code(function_name, decision.code)

        elif decision.action == "skip_function":
            self.action_log.append(
                f"Decision: Skipping function '{function_name}'. Reason: {decision.reason}",
            )
            self.skipped_functions.add(function_name)

        elif decision.action == "request_clarification":
            self.action_log.append(
                f"Decision: Requesting user clarification for '{function_name}'. Reason: {decision.reason}",
            )

            if not self.clarification_enabled:
                self.action_log.append(
                    "Clarification requested but no clarification channel available. Attempting to implement with best guess.",
                )
                await self._handle_dynamic_implementation(
                    function_name,
                    replan_reason="Clarification channel not available. Please proceed with your best guess.",
                    call_stack_snapshot=kwargs.get(
                        "call_stack_snapshot",
                    ),
                    scoped_context_snapshot=kwargs.get(
                        "scoped_context_snapshot",
                    ),
                )
            else:
                question = (
                    decision.clarification_question
                    or "I need help understanding how to implement this function."
                )
                self.action_log.append(f"Asking user: {question}")

                await self.clarification_up_q.put(question)
                answer = await self.clarification_down_q.get()
                self.action_log.append(f"User answered: {answer}")

                await self._handle_dynamic_implementation(
                    function_name,
                    clarification_question=question,
                    clarification_answer=answer,
                    call_stack_snapshot=kwargs.get(
                        "call_stack_snapshot",
                    ),
                    scoped_context_snapshot=kwargs.get(
                        "scoped_context_snapshot",
                    ),
                )

        elif decision.action == "replan_parent":
            self.action_log.append(
                f"Decision: Escalating to replan parent of '{function_name}'. Reason: {decision.reason}",
            )
            raise ReplanFromParentException(
                f"Child function '{function_name}' requested parent replan.",
                reason=decision.reason,
            )
        else:
            raise ValueError(
                f"Unknown ImplementationDecision action: {decision.action}",
            )

    def _get_main_function_name(self) -> str | None:
        """
        Parses the plan's source code to find the main entry point.

        Returns:
            The name of the main function ('main_plan') or None if not found.
        """
        try:
            tree = ast.parse(self.plan_source_code or "")
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "main_plan"
                ):
                    return node.name
        except SyntaxError:
            return None
        return None

    def _get_node_key(self, node: ast.AST) -> Optional[tuple]:
        """
        Generates a unique, comparable key for a top-level AST node.

        This key is used to identify and replace nodes during a code merge.
        - Definitions (functions, classes) are keyed by type and name.
        - Imports are keyed by their full string representation.
        - Global assignments are keyed by the variable name.
        - Other nodes (like module-level docstrings) are keyed by their content.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("def", node.name)
        elif isinstance(node, ast.ClassDef):
            return ("class", node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            return ("import", ast.unparse(node))
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                return ("assign", node.targets[0].id)
        return ("other", ast.unparse(node))

    def _is_docstring_or_pass(self, stmt, is_first=False):
        """
        Check if a statement is a docstring or pass statement that should be skipped during merge.

        Args:
            stmt: The AST statement to check
            is_first: Whether this is the first statement in the function body

        Returns:
            True if the statement should be skipped, False otherwise
        """
        if isinstance(stmt, ast.Pass):
            return True

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                return True

        return False

    def _update_plan_with_new_code(self, function_name: str, new_code: str):
        """
        Updates the plan's source code by performing an in-place replacement of a
        function, whether it is top-level or nested. This uses an AST transformer
        to ensure correctness.

        Args:
            function_name: The name of the function to replace.
            new_code: The full source code of the new function implementation.
        """
        self.action_log.append(f"Updating implementation of '{function_name}'.")

        try:
            new_function_tree = ast.parse(textwrap.dedent(new_code))
            if not new_function_tree.body or not isinstance(
                new_function_tree.body[0],
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                raise ValueError("New code must contain a single function definition.")
            new_function_node = new_function_tree.body[0]

            parent_function_name = None
            for top_level_func in self.top_level_function_names:
                if top_level_func == function_name:
                    continue
                parent_source = self.clean_function_source_map.get(top_level_func, "")
                if (
                    f"def {function_name}(" in parent_source
                    or f"async def {function_name}(" in parent_source
                ):
                    parent_function_name = top_level_func
                    break

            if parent_function_name:
                parent_source = self.clean_function_source_map[parent_function_name]
                parent_tree = ast.parse(parent_source)

                replacer = FunctionReplacer(function_name, new_function_node)
                modified_parent_tree = replacer.visit(parent_tree)
                ast.fix_missing_locations(modified_parent_tree)

                new_parent_source = ast.unparse(modified_parent_tree)
                self.clean_function_source_map[parent_function_name] = new_parent_source
            else:
                self.clean_function_source_map[function_name] = new_code
                if function_name not in self.top_level_function_names:
                    self.top_level_function_names.add(function_name)

            original_sanitized_tree = ast.parse(self.plan_source_code or "pass")
            reconstructed_parts = [
                ast.unparse(node)
                for node in original_sanitized_tree.body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for func_name in sorted(list(self.top_level_function_names)):
                if func_name in self.clean_function_source_map:
                    reconstructed_parts.append(
                        self.clean_function_source_map[func_name],
                    )

            new_unsanitized_code = "\n\n".join(reconstructed_parts)

            self.plan_source_code = self.actor._sanitize_code(
                new_unsanitized_code,
                self,
            )
            self.actor._load_plan_module(self)

        except (SyntaxError, ValueError, RuntimeError) as e:
            logger.error(
                f"AST-based code replacement for '{function_name}' failed: {e}",
                exc_info=True,
            )
            raise

    async def result(self) -> str:
        """
        Waits for the plan to complete and returns its final result.

        Returns:
            The final result string of the plan.
        """
        await self._completion_event.wait()
        if self._state == _HierarchicalHandleState.ERROR:
            import traceback

            raise RuntimeError(f"Plan failed in state ERROR: {traceback.format_exc()}")
        return (
            self._final_result_str
            or f"Plan finished in state {self._state.name} without a result."
        )

    def done(self) -> bool:
        """
        Checks if the plan has completed.

        Returns:
            True if the plan is in a terminal state, False otherwise.
        """
        return self._is_complete

    async def awaiting_next_instruction(self) -> str:
        """
        Waits until the plan has completed its current unit of work and is
        paused waiting for the next instruction. Returns a summary of actions
        performed since the last call to this method.
        """
        if not self.persist:
            raise RuntimeError(
                "The .awaiting_next_instruction() handle is only available when the plan is started with persist=True.",
            )

        completion_event = asyncio.Event()
        await self._done_events.put(completion_event)
        await completion_event.wait()

        start_index = self._last_summarized_interaction_count
        new_interactions = self.cumulative_interactions[start_index:]

        if not new_interactions:
            return "No new actions were performed since the last update."

        formatted_interactions = []
        for interaction in new_interactions:
            kind, act, obs, *logs = interaction
            logs = logs[0] if logs else []
            log_entry = f"- Action: `{act}` with result `{obs or 'N/A'}`"
            if logs:
                log_details = "\n".join([f"    {line}" for line in logs])
                log_entry += f"\n  - Agent Logs:\n{log_details}"
            formatted_interactions.append(log_entry)

        summary = await self._summarize_log_chunk("\n".join(formatted_interactions))

        self._last_summarized_interaction_count = len(self.cumulative_interactions)
        return summary

    def _spawn_async_verification(self, item: VerificationWorkItem):
        """
        Creates and tracks a new background task that performs real verification.
        """

        async def verification_runner():
            """
            Performs the actual verification, including the cache-only fast path,
            and routes the result to the appropriate handler.
            """
            try:
                computer_primitives: ComputerPrimitives | None = None
                try:
                    computer_primitives = self._get_computer_primitives()
                except Exception:
                    computer_primitives = None

                if (
                    item.start_seq >= 0
                    and item.exit_seq > item.start_seq
                    and computer_primitives is not None
                    and isinstance(
                        computer_primitives.browser.backend,
                        MagnitudeBrowserBackend,
                    )
                ):
                    backend = computer_primitives.browser.backend
                    current_seq_cursor = item.start_seq + 1

                    for idx, interaction in enumerate(item.interactions):
                        if interaction[0] == "tool_call":
                            call_repr = interaction[1]
                            if (
                                "computer_primitives.act" in call_repr
                                or "computer_primitives.navigate" in call_repr
                            ):
                                target_seq = current_seq_cursor
                                current_seq_cursor += 1

                                try:
                                    logs = await backend.await_sequence_logs(target_seq)
                                    if logs:
                                        new_interaction = (
                                            interaction[0],
                                            interaction[1],
                                            interaction[2],
                                            logs,
                                        )
                                        item.interactions[idx] = new_interaction
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to sync logs for seq {target_seq}: {e}",
                                    )

                    try:
                        await backend.barrier(up_to_seq=item.exit_seq)

                        new_screenshot = await backend.get_screenshot()
                        new_url = await backend.get_current_url()

                        # Refresh evidence from browser environment after barrier.
                        # (The work item stores evidence per-environment namespace.)
                        if "computer_primitives" in item.post_state and isinstance(
                            item.post_state["computer_primitives"],
                            dict,
                        ):
                            item.post_state["computer_primitives"][
                                "screenshot"
                            ] = new_screenshot
                            item.post_state["computer_primitives"]["url"] = new_url
                    except Exception as e:
                        logger.warning(f"Failed to refresh post-state after sync: {e}")

                assessment = None
                if item.cache_miss_counter == 0:
                    assessment = VerificationAssessment(
                        status="ok",
                        reason="Skipped verification for fully cached replay step.",
                    )
                else:
                    # Only attempt browser barriers when a browser environment is active.
                    computer_primitives: ComputerPrimitives | None = None
                    if (
                        "computer_primitives" in (self.actor.environments or {})
                        and getattr(self.actor, "computer_primitives", None) is not None
                    ):
                        try:
                            computer_primitives = self._get_computer_primitives()
                        except Exception:
                            computer_primitives = None

                    if computer_primitives is not None and isinstance(
                        computer_primitives.browser.backend,
                        MagnitudeBrowserBackend,
                    ):
                        await computer_primitives.browser.backend.barrier(
                            up_to_seq=item.exit_seq,
                        )

                    assessment = await self.actor._check_state_against_goal(
                        plan=self,
                        function_name=item.function_name,
                        function_docstring=item.docstring,
                        function_source_code=item.func_source,
                        interactions=item.interactions,
                        evidence=item.post_state,
                        function_return_value=item.return_value_repr,
                    )

                if assessment.status == "ok":
                    await self._on_verification_success(item, assessment)
                else:
                    await self._on_verification_failure(item, assessment)

            except asyncio.CancelledError:
                logger.warning(
                    f"[V-TASK-{item.ordinal}] Verification for '{item.function_name}' was cancelled.",
                )
                self.action_log.append(
                    f"Verification for '{item.function_name}' was cancelled",
                )
            except Exception as e:
                logger.error(
                    f"[V-TASK-{item.ordinal}] Verification task for '{item.function_name}' crashed: {e}",
                    exc_info=True,
                )
                assessment = VerificationAssessment(
                    status="fatal_error",
                    reason=f"Verification task crashed: {e}",
                )
                await self._on_verification_failure(item, assessment)
            finally:
                self.pending_verifications.pop(item.ordinal, None)
                self._child_tasks.discard(asyncio.current_task())

        task = asyncio.create_task(
            verification_runner(),
            name=f"Verify-{item.ordinal}-{item.function_name}",
        )
        self.pending_verifications[item.ordinal] = VerificationHandle(
            item=item,
            task=task,
        )
        self._child_tasks.add(task)

    async def _on_verification_success(
        self,
        item: VerificationWorkItem,
        assessment: VerificationAssessment,
    ):
        """Handles the side-effects of a successful verification."""
        logger.info(
            f"[V-TASK-{item.ordinal}] Verification SUCCEEDED for '{item.function_name}'. Reason: {assessment.reason}",
        )

        self.last_verified_function_name = item.function_name

        # Extract browser evidence if available (post-state is per-environment evidence).
        browser_evidence = item.post_state.get("computer_primitives")
        if isinstance(browser_evidence, dict):
            self.last_verified_url = browser_evidence.get("url")
            self.last_verified_screenshot = browser_evidence.get("screenshot")
        else:
            self.last_verified_url = None
            self.last_verified_screenshot = None

        if hasattr(self, "cumulative_interactions"):
            self.cumulative_interactions.extend(item.interactions)

        if not self.can_store:
            logger.debug(
                f"Skipping function persistence for '{item.function_name}' (can_store=False).",
            )
            return

        is_top_level_function = item.function_name in self.top_level_function_names

        if (
            is_top_level_function
            and self.actor.function_manager
            and item.function_name != "main_plan"
        ):
            try:
                func_tree = ast.parse(
                    self.clean_function_source_map[item.function_name],
                )
                func_node = func_tree.body[0]

                if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_node.decorator_list = [
                        d
                        for d in func_node.decorator_list
                        if not (isinstance(d, ast.Name) and d.id == "verify")
                    ]

                clean_func_source = ast.unparse(func_tree)
                existing_funcs = self.actor.function_manager.list_functions(
                    include_implementations=True,
                )
                is_duplicate = any(
                    item.function_name == data.get("name")
                    for data in existing_funcs.values()
                )

                precondition_prompt = prompt_builders.build_precondition_prompt(
                    function_source_code=clean_func_source,
                    interactions_log=json.dumps(
                        item.interactions,
                        indent=2,
                    ),
                    has_entry_screenshot=bool(
                        isinstance(item.pre_state.get("computer_primitives"), dict)
                        and item.pre_state.get("computer_primitives", {}).get(
                            "screenshot",
                        )
                        is not None,
                    ),
                    environments=self.actor.environments,
                )

                self.summarization_client.set_response_format(PreconditionDecision)
                try:
                    entry_screenshot = None
                    entry_browser_evidence = item.pre_state.get("computer_primitives")
                    if isinstance(entry_browser_evidence, dict):
                        entry_screenshot = entry_browser_evidence.get("screenshot")

                    decision_str = await llm_call(
                        self.summarization_client,
                        precondition_prompt,
                        screenshot=entry_screenshot,
                    )
                    precondition_data = PreconditionDecision.model_validate_json(
                        decision_str,
                    )
                finally:
                    self.summarization_client.reset_response_format()

                preconditions_for_fm = {}
                if precondition_data.status == "ok" and (
                    precondition_data.url or precondition_data.description
                ):
                    preconditions_for_fm[item.function_name] = {
                        "url": precondition_data.url,
                        "description": precondition_data.description,
                    }

                if not is_duplicate:
                    self.action_log.append(
                        f"Persisting verified function '{item.function_name}' as a new skill.",
                    )
                    logger.info(
                        f"Adding function '{item.function_name}' to FunctionManager.",
                    )
                    self.actor.function_manager.add_functions(
                        implementations=[clean_func_source],
                        preconditions=preconditions_for_fm,
                    )
                else:
                    self.action_log.append(
                        f"Skipping persistence for '{item.function_name}'; identical skill already exists.",
                    )
                    logger.info(
                        f"Skipping adding function '{item.function_name}' to FunctionManager; identical function already exists.",
                    )
            except Exception as e:
                self.action_log.append(
                    f"WARNING: Could not persist function '{item.function_name}': {e}",
                )
                logger.warning(
                    f"Could not add function '{item.function_name}' to FunctionManager: {e}",
                )

    async def _on_verification_failure(
        self,
        item: VerificationWorkItem,
        assessment: VerificationAssessment,
    ):
        """
        Handles a failed verification, including preemption logic and targeting
        the correct function for replanning (parent or self).
        """
        if not self.can_compose:
            error_msg = (
                f"Verification failed for '{item.function_name}' (status={assessment.status}): "
                f"{assessment.reason}. Cannot recover because can_compose=False. "
                f"All functions must execute successfully without re-implementation when this flag is disabled."
            )
            logger.error(error_msg)
            self._set_state(_HierarchicalHandleState.ERROR)
            self._set_final_result(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        if self.is_verifying_post_completion:
            logger.info(
                f"Verification failed for '{item.function_name}' after plan completion. Re-opening plan for recovery...",
            )
            self.action_log.append(
                f"Post-completion verification failed for '{item.function_name}'. Re-opening plan to recover.",
            )
            self._set_state(_HierarchicalHandleState.RUNNING)
            self.is_verifying_post_completion = False

        async with self._verification_lock:
            failing_ordinal = item.ordinal
            target_function_name = item.function_name
            recovery_ordinal_for_preemption = failing_ordinal

            if assessment.status == "replan_parent":
                try:
                    stack = item.full_call_stack_tuple
                    current_index = stack.index(item.function_name)
                    if current_index > 0:
                        target_function_name = stack[current_index - 1]
                        self.action_log.append(
                            f"Verification failure of '{item.function_name}' (ord={failing_ordinal}) "
                            f"escalated to replan parent: '{target_function_name}'.",
                        )
                        logger.info(
                            f"[V-TASK-{failing_ordinal}] Escalating failure to replan parent '{target_function_name}'.",
                        )
                        recovery_ordinal_for_preemption = failing_ordinal
                    else:
                        self.action_log.append(
                            f"WARNING: '{item.function_name}' requested replan_parent but is top-level. "
                            f"Attempting reimplement_local instead.",
                        )
                        logger.warning(
                            f"[V-TASK-{failing_ordinal}] Cannot replan parent of top-level '{item.function_name}'. "
                            f"Treating as reimplement_local.",
                        )
                        target_function_name = item.function_name
                        assessment.status = "reimplement_local"
                        recovery_ordinal_for_preemption = failing_ordinal

                except ValueError:
                    logger.error(
                        f"Could not find '{item.function_name}' in its own stack snapshot: {stack}. Treating as reimplement_local.",
                    )
                    target_function_name = item.function_name
                    assessment.status = "reimplement_local"
                    recovery_ordinal_for_preemption = failing_ordinal
            if self._recovery_task:
                if recovery_ordinal_for_preemption < (
                    self._recovery_target_ordinal or float("inf")
                ):
                    logger.warning(
                        f"Preempting recovery for target ordinal {self._recovery_target_ordinal} "
                        f"with earlier failure detection point (ord={recovery_ordinal_for_preemption}) "
                        f"targeting function '{target_function_name}'.",
                    )
                    self.action_log.append(
                        f"PREEMPTION: Earlier failure trigger (ord={recovery_ordinal_for_preemption}, target='{target_function_name}') "
                        f"preempts recovery for target ordinal {self._recovery_target_ordinal}.",
                    )
                    self._recovery_task.cancel()
                    try:
                        await self._recovery_task
                    except asyncio.CancelledError:
                        pass
                else:
                    logger.info(
                        f"Ignoring failure trigger at ordinal {recovery_ordinal_for_preemption} "
                        f"(target='{target_function_name}') because recovery for earlier/equivalent target "
                        f"(ord={self._recovery_target_ordinal}) is in progress.",
                    )
                    self.action_log.append(
                        f"Ignoring failure trigger at ord={recovery_ordinal_for_preemption} "
                        f"(target='{target_function_name}') because recovery for earlier/equivalent target "
                        f"(ord={self._recovery_target_ordinal}) is in progress.",
                    )
                    return
            self._recovery_in_progress = True

            logger.critical(
                f"[V-TASK-{item.ordinal}] VERIFICATION FAILED. Status: {assessment.status}, "
                f"Reason: {assessment.reason}. Initiating recovery targeting function '{target_function_name}' "
                f"(using ordinal {recovery_ordinal_for_preemption} for preemption check).",
            )
            self.action_log.append(
                f"Async Verification for {item.function_name} (ord={item.ordinal}): FAILED - Status: {assessment.status}, "
                f"Reason: '{assessment.reason}'. Initiating recovery targeting '{target_function_name}'.",
            )

            await self._cancel_verifications_after(item.ordinal)
            self._recovery_target_ordinal = recovery_ordinal_for_preemption

            self._recovery_task = asyncio.create_task(
                self._perform_verification_recovery(
                    item,
                    assessment,
                    target_function_name_override=target_function_name,
                ),
                name=f"Recovery-TriggerOrd{failing_ordinal}-TargetFunc-{target_function_name}",
            )

    async def _perform_verification_recovery(
        self,
        item: VerificationWorkItem,
        assessment: VerificationAssessment,
        target_function_name_override: str,
    ):
        """
        Orchestrates the full rollback, fix, rewind, and restart process,
        targeting the specified function for the fix.

        Args:
            item: The verification work item containing original failure context & snapshots.
            assessment: The verification assessment with status and reason.
            target_function_name_override: The function to target for reimplementation
                (may be the parent when handling replan_parent).
        """
        try:
            run_id_being_cancelled = self.run_id
            await self._cancel_and_wait_for_task(
                self._execution_task,
                f"verification failure targeting '{target_function_name_override}'",
            )
            # Browser interruption is only applicable when a browser environment is active.
            if (
                "computer_primitives" in (self.actor.environments or {})
                and getattr(self.actor, "computer_primitives", None) is not None
            ):
                await self._clear_browser_queue_for_run(run_id_being_cancelled)
                computer_primitives = self._get_computer_primitives()
                if hasattr(
                    computer_primitives.browser.backend,
                    "interrupt_current_action",
                ):
                    await computer_primitives.browser.backend.interrupt_current_action()

            await self._handle_dynamic_implementation(
                function_name=target_function_name_override,
                replan_reason=assessment.reason,
                status=assessment.status,
                failed_item=item,
                call_stack_snapshot=list(item.full_call_stack_tuple),
                scoped_context_snapshot=item.scoped_context_snapshot,
            )
            self.action_log.append(
                f"COURSE CORRECTION: Launching recovery agent to restore state before '{target_function_name_override}' was executed.",
            )
            logger.info(
                f"Launching course correction agent for verification recovery targeting '{target_function_name_override}'.",
            )
            try:
                # Extract browser screenshot for course correction.
                target_screenshot = None
                entry_browser_evidence = item.pre_state.get("computer_primitives")
                if isinstance(entry_browser_evidence, dict):
                    target_screenshot = entry_browser_evidence.get("screenshot")

                trajectory = []
                for interaction in item.interactions:
                    if len(interaction) > 1:
                        trajectory.append(interaction[1])
                    else:
                        trajectory.append(str(interaction))

                if (
                    target_screenshot
                    and trajectory
                    and "computer_primitives" in (self.actor.environments or {})
                    and getattr(self.actor, "computer_primitives", None) is not None
                ):
                    await self.actor._run_course_correction_agent(
                        plan=self,
                        target_screenshot=target_screenshot,
                        trajectory=trajectory,
                    )
                    self.action_log.append(
                        "COURSE CORRECTION: Recovery agent completed successfully.",
                    )
                    logger.info(
                        "Course correction for verification recovery completed successfully.",
                    )
                else:
                    logger.warning(
                        f"Missing target screenshot or trajectory for course correction. "
                        f"target_screenshot={bool(target_screenshot)}, trajectory_len={len(trajectory)}",
                    )
                    self.action_log.append(
                        "WARNING: Skipping course correction due to missing data. Proceeding with replay from current state.",
                    )
            except Exception as e:
                logger.error(
                    f"Course correction for verification recovery failed: {e}",
                    exc_info=True,
                )
                self.action_log.append(
                    f"WARNING: Course correction failed: {e}. Proceeding with replay from current state.",
                )
            try:
                target_index = item.full_call_stack_tuple.index(
                    target_function_name_override,
                )
                parent_stack_for_invalidation = item.full_call_stack_tuple[
                    :target_index
                ]
            except ValueError:
                logger.warning(
                    f"Could not find target function '{target_function_name_override}' in stack snapshot "
                    f"{item.full_call_stack_tuple} during cache invalidation derivation. "
                    f"Falling back to original item's parent_stack.",
                    exc_info=True,
                )
                parent_stack_for_invalidation = item.parent_stack

            self._invalidate_cache_from_function(
                function_name=target_function_name_override,
                parent_stack=parent_stack_for_invalidation,
            )
            self._restart_execution_loop(
                f"replay_after_recovery_targeting_{target_function_name_override}",
            )

        except Exception as e:
            logger.error(
                f"Critical error during verification recovery targeting '{target_function_name_override}': {e}",
                exc_info=True,
            )
            self._set_state(_HierarchicalHandleState.ERROR)
            self._set_final_result(
                f"ERROR: Unrecoverable error during verification recovery targeting '{target_function_name_override}': {e}",
            )
        finally:
            self._recovery_task = None
            self._recovery_target_ordinal = None
            self._recovery_in_progress = False

    async def _cancel_verifications_after(self, ord_: int):
        """Cancels all pending verification tasks with an ordinal greater than the given one."""

        to_cancel = [o for o in self.pending_verifications if o > ord_]
        if to_cancel:
            logger.info(f"Cancelling {len(to_cancel)} subsequent verification tasks.")
            self.action_log.append(
                f"Cancelling {len(to_cancel)} subsequent verification tasks.",
            )

        for ordinal in to_cancel:
            handle = self.pending_verifications.pop(ordinal, None)
            if handle and not handle.task.done():
                handle.task.cancel()

                try:
                    await handle.task
                except asyncio.CancelledError:

                    self.action_log.append(
                        f"Verification for '{handle.item.function_name}' was cancelled (ord={ordinal}).",
                    )
            if handle:
                self._child_tasks.discard(handle.task)

    def _invalidate_cache_from_function(self, function_name: str, parent_stack: tuple):
        """
        Invalidates cache entries for the target function and all subsequent calls
        within the same call chain, based on the provided stack context.
        """
        try:
            full_stack_list = list(parent_stack) + [function_name]
            stack_depth = len(full_stack_list)

            if not stack_depth:
                logger.warning(
                    "Attempted cache invalidation with an empty stack. Skipping.",
                )
                return

            self.action_log.append(
                f"CACHE INVALIDATION: Recovery initiating invalidation from function "
                f"'{function_name}' at stack depth {stack_depth} (Stack: {' -> '.join(full_stack_list)}).",
            )
            logger.info(
                f"Invalidating cache from function '{function_name}' (stack: {' -> '.join(full_stack_list)}) onwards.",
            )

            keys_to_delete = set()
            invalidated_functions_logged = set()

            for key, value in self.idempotency_cache.items():
                key_call_stack = (
                    key[0]
                    if isinstance(key, tuple)
                    and len(key) > 0
                    and isinstance(key[0], tuple)
                    else None
                )

                if not key_call_stack or len(key_call_stack) < stack_depth:
                    continue

                is_match = True
                for i in range(stack_depth):
                    if key_call_stack[i] != full_stack_list[i]:
                        is_match = False
                        break

                if is_match:
                    keys_to_delete.add(key)
                    if (
                        value.get("meta")
                        and value["meta"].get("function")
                        not in invalidated_functions_logged
                    ):
                        invalidated_functions_logged.add(value["meta"]["function"])

            if keys_to_delete:
                logger.info(
                    f"Invalidating {len(keys_to_delete)} cache entries related to call stack starting from '{function_name}'. "
                    f"Affected functions in log: {', '.join(sorted(list(invalidated_functions_logged)))}",
                )
                self.action_log.append(
                    f"CACHE INVALIDATION: Identified {len(keys_to_delete)} entries to remove.",
                )

                invalidated_handles = set()

                for key in keys_to_delete:
                    entry = self.idempotency_cache.pop(key, None)
                    if entry:
                        result_is_handle_id = (
                            isinstance(entry.get("result"), dict)
                            and "handle_id" in entry["result"]
                        )
                        interaction_mentions_handle = (
                            "Returned handle"
                            in entry.get("interaction_log", [None, None, ""])[2]
                        )

                        if result_is_handle_id or interaction_mentions_handle:
                            handle_id = (
                                entry.get("result", {}).get("handle_id")
                                if result_is_handle_id
                                else None
                            )
                            if handle_id and isinstance(handle_id, str):
                                invalidated_handles.add(handle_id)
                                self.live_handles.pop(handle_id, None)
            else:
                logger.info(
                    f"No cache entries found matching the stack prefix for recovery targeting '{function_name}'.",
                )
                self.action_log.append(
                    f"CACHE INVALIDATION: No matching entries found for '{function_name}'.",
                )

        except Exception as e:
            logger.warning(
                f"Selective cache invalidation during recovery failed unexpectedly: {e}. "
                f"Clearing entire cache as a fallback to ensure safety.",
                exc_info=True,
            )
            self.action_log.append(
                f"CACHE INVALIDATION: Error during selective invalidation: {e}. Clearing entire cache!",
            )
            self.idempotency_cache.clear()
            self.live_handles.clear()

    def _resolve_invalidation_keys(
        self,
        decision: InterjectionDecision,
    ) -> set[tuple]:
        """
        Resolve which cache keys to invalidate by combining:
        (A) the LLM's proposal (functions + intra-function tails), and
        (B) the impure guardrail for safety.

        This method is domain-agnostic and works for any tool type (browser, state
        managers, custom functions) based purely on cache metadata.
        """
        cache = self.idempotency_cache
        all_keys = list(cache.keys())
        spec = getattr(decision, "cache", None)

        proposed = set()
        if spec:
            for k in all_keys:
                meta = cache[k].get("meta", {})
                if not meta:
                    continue
                if meta.get("function") in spec.invalidate_functions:
                    proposed.add(k)
                    continue
                for r in spec.invalidate_steps:
                    if (
                        r.function_name == meta.get("function")
                        and meta.get("step", 0) >= r.from_step_inclusive
                    ):
                        proposed.add(k)
                        break

        impure_guard = set()
        invalidated_impure_indices = {
            i
            for i, k in enumerate(all_keys)
            if k in proposed and cache[k].get("meta", {}).get("impure")
        }
        if invalidated_impure_indices:
            latest_impure_index = max(invalidated_impure_indices)
            for i, k in enumerate(all_keys):
                if i > latest_impure_index:
                    impure_guard.add(k)

        final_keys = proposed | impure_guard
        return final_keys

    def _restart_execution_loop(self, mode: str):
        """Restart the execution loop with a new run_id. Assumes the caller has already cancelled and awaited the previous task."""
        old_run_id = self.run_id
        self.run_id += 1
        logger.info(
            f"🔄 RUN TRANSITION: run_id={old_run_id} -> run_id={self.run_id} ({mode})",
        )
        self.action_log.append(
            f"RESTART: Restarting execution loop (old_run_id={old_run_id} → new_run_id={self.run_id}, reason: {mode}).",
        )

        asyncio.create_task(self._cancel_all_background_tasks())

        if old_run_id in self.runtime.call_stacks:
            del self.runtime.call_stacks[old_run_id]
        self.interaction_stack.clear()
        self.call_stack.clear()
        self.runtime.path_context.clear()
        self.runtime._loop_context_stack.clear()
        self.interaction_stack.append([])

        self._execution_task = asyncio.create_task(
            self._initialize_and_run(mode=mode),
            name=f"MainPlanTask-{self._module_name}-Run{self.run_id}",
        )

    async def _cancel_and_wait_for_task(
        self,
        task: Optional[asyncio.Task],
        reason: str,
    ):
        """Robustly cancels a task and waits for it to finish."""
        if task and not task.done():
            self.action_log.append(
                f"CANCEL: Requesting cancellation of task {task.get_name()} due to: {reason}.",
            )
            logger.debug(
                f"Requesting cancellation of task {task.get_name()} (run_id={self.run_id}) due to: {reason}.",
            )
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                self.action_log.append(
                    f"CANCEL: Task {task.get_name()} confirmed cancelled.",
                )
                logger.debug(f"Task {task.get_name()} confirmed cancelled.")
            except Exception as e:
                self.action_log.append(
                    f"WARNING: Unexpected error waiting for task {task.get_name()} cancellation: {e}",
                )
                logger.warning(
                    f"Unexpected error waiting for task {task.get_name()} cancellation: {e}",
                    exc_info=True,
                )
        elif task:
            logger.debug(
                f"Task {task.get_name()} was already done when cancellation requested.",
            )
        else:
            logger.debug(f"No task provided for cancellation ({reason}).")

    async def _clear_browser_queue_for_run(self, run_id_to_clear: int):
        """
        Instructs the browser backend to clear pending commands for a specific run_id.
        This prevents stale commands from an old execution run from executing after
        the run has been cancelled and a new one has started.

        This is a no-op if no browser environment is active.
        """
        if "computer_primitives" not in (self.actor.environments or {}):
            logger.debug(
                "No browser environment active; skipping browser queue clearing.",
            )
            return

        try:
            backend = self._get_computer_primitives().browser.backend
        except Exception as e:
            logger.debug(f"Could not access browser backend to clear queue: {e}")
            return

        if not hasattr(backend, "clear_pending_commands"):
            logger.debug(
                "Browser backend does not have 'clear_pending_commands'. Skipping queue clearing.",
            )
            return

        try:
            self.action_log.append(
                f"BROWSER: Clearing pending commands for cancelled run_id={run_id_to_clear}.",
            )
            logger.info(
                f"Clearing pending browser commands for cancelled run_id={run_id_to_clear}.",
            )
            await backend.clear_pending_commands(run_id=run_id_to_clear)
            self.action_log.append(
                f"BROWSER: Pending commands cleared for run_id={run_id_to_clear}.",
            )
            logger.info(f"Pending commands cleared for run_id={run_id_to_clear}.")
        except Exception as e:
            self.action_log.append(
                f"WARNING: Failed to clear browser command queue for run_id={run_id_to_clear}: {e}",
            )
            logger.warning(
                f"Failed to clear browser command queue for run_id={run_id_to_clear}: {e}",
                exc_info=True,
            )

    async def _cancel_all_background_tasks(self):
        """Gracefully cancels all in-flight verification and recovery tasks."""
        logger.debug("Cancelling all background verification and recovery tasks.")
        self.action_log.append(
            "Cancelling all background verification and recovery tasks.",
        )

        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()

        # Cancel pane supervisor (independent of verification/recovery task presence).
        if (
            self._pane_supervisor_task is not None
            and not self._pane_supervisor_task.done()
        ):
            self._pane_supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pane_supervisor_task
        self._pane_supervisor_task = None

        verifications_to_cancel = [
            handle
            for handle in self.pending_verifications.values()
            if not handle.task.done()
        ]

        if not verifications_to_cancel and (
            not self._recovery_task or self._recovery_task.done()
        ):
            return

        for handle in verifications_to_cancel:
            handle.task.cancel()
            self.action_log.append(
                f"Verification for '{handle.item.function_name}' was cancelled",
            )

        all_tasks = [handle.task for handle in verifications_to_cancel] + (
            [self._recovery_task] if self._recovery_task else []
        )
        await asyncio.gather(*all_tasks, return_exceptions=True)

        if self._recovery_task:
            self._child_tasks.discard(self._recovery_task)

        self.pending_verifications.clear()
        self._recovery_task = None

    async def _summarize_log_chunk(self, summaries: str) -> str:
        """Uses an LLM to summarize a list of action log entries."""
        if not summaries:
            return "Actions completed successfully."

        prompt = textwrap.dedent(
            f"""
            The following is a log of actions from an autonomous agent.
            Summarize these actions concisely in a single sentence from the first-person perspective (e.g., "I have successfully navigated to the website and then searched for cookies.").

            ACTION LOG:
            ---
            {summaries}
            ---
        """,
        )
        summary = await llm_call(self.summarization_client, prompt)
        return summary.strip()

    def _cleanup_temp_file(self):
        """
        Clean up temporary file and module references.

        This method ensures proper resource cleanup to prevent accumulation
        of temporary files and modules in memory.
        """
        try:
            if self._img_token:
                LIVE_IMAGES_REGISTRY.reset(self._img_token)
                self._img_token = None
            if self._imglog_token:
                LIVE_IMAGES_LOG.reset(self._imglog_token)
                self._imglog_token = None

            if self._temp_file_path and self._temp_file_path.exists():
                self._temp_file_path.unlink(missing_ok=True)
                logger.debug(f"Deleted temporary plan file: {self._temp_file_path}")
                self._temp_file_path = None

            if self._module_name and self._module_name in sys.modules:
                del sys.modules[self._module_name]
                logger.debug(
                    f"Removed plan module from sys.modules: {self._module_name}",
                )

            self._module = None
            self._module_spec = None

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def interject(
        self,
        message: str,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
    ) -> str:
        """
        Processes a user interjection by using an LLM to decide on the best course of action.
        """

        await self._cancel_all_background_tasks()

        logger.debug(
            f"INTERJECT: Interjection received {message}. Current state: {self._state.name}",
        )
        self.action_log.append(
            f"INTERJECT: Interjection received {message}. Current state: {self._state.name}",
        )
        if not self._is_valid_method("interject"):
            return "Cannot interject: plan not running."

        async with self._interject_lock:
            # Conditionally interrupt browser if a browser environment is active.
            if "computer_primitives" in (self.actor.environments or {}):
                try:
                    computer_primitives = self._get_computer_primitives()
                    if hasattr(
                        computer_primitives.browser.backend,
                        "interrupt_current_action",
                    ):
                        await computer_primitives.browser.backend.interrupt_current_action()
                except Exception as e:
                    logger.debug(f"Could not interrupt browser action: {e}")
            await self.pause()
            decision = None
            try:
                context_dict = self.actor._get_scoped_context_from_plan_state(self)
                scoped_context_str = self.actor._format_scoped_context_for_prompt(
                    context_dict,
                )

                # Snapshot steerable pane state for routing decisions (best-effort).
                pane_snapshot: dict[str, Any] | None = None
                try:
                    if hasattr(self, "pane") and self.pane is not None:
                        all_handles = await self.pane.list_handles(status=None)
                        in_flight = [
                            h
                            for h in all_handles
                            if h.get("status")
                            in ("running", "paused", "waiting_for_clarification")
                        ]
                        pending_clars = await self.pane.get_pending_clarifications()
                        pane_snapshot = {
                            "active_handles": in_flight,
                            "pending_clarifications_count": len(pending_clars),
                        }
                except Exception as e:
                    logger.debug(f"Could not build pane snapshot for interjection: {e}")

                static_prompt, dynamic_prompt = (
                    prompt_builders.build_interjection_prompt(
                        interjection=message,
                        parent_chat_context=self.parent_chat_context,
                        scoped_context=scoped_context_str,
                        call_stack=self.call_stack,
                        action_log=self.action_log[-10:],
                        goal=self.goal,
                        idempotency_cache=self.idempotency_cache,
                        tools=self.actor.tools,
                        environments=self.actor.environments,
                        images=images,
                        pane_snapshot=pane_snapshot,
                    )
                )

                self.modification_client.set_response_format(InterjectionDecision)
                try:
                    decision_str = await llm_call(
                        self.modification_client,
                        dynamic_prompt,
                        static_prompt=static_prompt,
                        images=images,
                    )
                    decision = InterjectionDecision.model_validate_json(decision_str)
                    logger.debug(
                        f"{format_pydantic_model(decision, title='INTERJECTION DECISION', indent=2)}",
                    )
                finally:
                    self.modification_client.reset_response_format()

                self.action_log.append(
                    f"Interjection Decision: {decision.action} - {decision.reason}",
                )

                # Apply routing to in-flight handles via pane.
                # Do this *before* executing potentially expensive interjection actions (e.g. modify_task),
                # to minimize propagation latency to in-flight manager loops.
                routing_status = await self._apply_interjection_routing(
                    decision=decision,
                    original_message=message,
                    images=images,
                )
                status_message = await self._execute_interjection_decision(decision)
                if routing_status:
                    status_message = f"{status_message}\n{routing_status}"

                return status_message

            except Exception as e:
                logger.error(f"Error during interjection handling: {e}", exc_info=True)
                self.action_log.append(f"ERROR during interjection: {e}")
                return f"Error processing interjection: {e}"
            finally:
                # Resume by default after interjection handling, except when we are
                # intentionally waiting for follow-up user input or replacing the plan.
                #
                # `modify_task` only blocks resume when patches are present; when no patches
                # are provided, we treat it as a no-op.
                should_resume = decision is None or (
                    decision.action not in ("replace_task", "clarify")
                    and not (decision.action == "modify_task" and decision.patches)
                )
                if self._state == _HierarchicalHandleState.PAUSED and should_resume:
                    await self.resume()

    async def _apply_interjection_routing(
        self,
        *,
        decision: InterjectionDecision,
        original_message: str,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
    ) -> str:
        """Apply routing from an interjection decision to in-flight handles via the pane.

        This is deterministic runtime logic: it does not perform an additional LLM call.
        Returns a short status message summarizing routing actions (or '' for no routing).
        """

        if not hasattr(self, "pane") or self.pane is None:
            return ""

        routing_action = getattr(decision, "routing_action", None) or "none"
        if routing_action == "none":
            return ""

        routed_message = getattr(decision, "routed_message", None) or original_message

        try:
            if routing_action == "targeted":
                target_ids = getattr(decision, "target_handle_ids", None) or []
                if not target_ids:
                    logger.warning(
                        "Interjection routing requested targeted but no target_handle_ids provided",
                    )
                    return ""

                # Best-effort status summary: use current pane registry snapshot (metadata only).
                try:
                    known = {
                        h["handle_id"]: h.get("status")
                        for h in await self.pane.list_handles(status=None)
                    }
                except Exception:
                    known = {}

                results: list[str] = []
                for hid in target_ids:
                    # Always call pane.interject so the pane emits a canonical `steering_applied` event
                    # (even when the handle is not found or already terminal).
                    await self.pane.interject(
                        hid,
                        routed_message,
                        parent_chat_context_cont=self.parent_chat_context,
                        images=images,
                    )
                    st = known.get(hid)
                    if st is None:
                        results.append(
                            f"- {hid}: dispatched (handle not found at snapshot time)",
                        )
                    elif st in ("completed", "failed", "stopped"):
                        results.append(
                            f"- {hid}: dispatched (handle already {st} at snapshot time)",
                        )
                    else:
                        results.append(f"- {hid}: dispatched")

                status = (
                    f"Routed interjection to {len(target_ids)} handle(s):\n"
                    + "\n".join(
                        results,
                    )
                )
                self.action_log.append(f"ROUTING: {status}")
                return status

            if routing_action == "broadcast_filtered":
                from unity.actor.steerable_tool_pane import BroadcastFilter

                # Micro-optimization / log hygiene: if there are no in-flight handles at all,
                # don't attempt a broadcast (avoids "0 handle(s)" noise).
                try:
                    all_handles = await self.pane.list_handles(status=None)
                    in_flight = [
                        h
                        for h in all_handles
                        if h.get("status")
                        in ("running", "paused", "waiting_for_clarification")
                    ]
                    if not in_flight:
                        return ""
                except Exception:
                    # Best-effort; if introspection fails, proceed with broadcast.
                    pass

                filter_dict = getattr(decision, "broadcast_filter", None) or {}
                statuses = filter_dict.get(
                    "statuses",
                    ["running", "paused", "waiting_for_clarification"],
                )
                bfilter = BroadcastFilter(
                    statuses=statuses,
                    origin_tool_prefixes=filter_dict.get("origin_tool_prefixes"),
                    capabilities=filter_dict.get("capabilities"),
                    created_after_step=filter_dict.get("created_after_step"),
                    created_before_step=filter_dict.get("created_before_step"),
                )
                result = await self.pane.broadcast_interject(
                    routed_message,
                    filter=bfilter,
                    parent_chat_context_cont=self.parent_chat_context,
                    images=images,
                )
                status = f"Broadcast interjection to {int(result.get('count') or 0)} handle(s)"
                self.action_log.append(f"ROUTING: {status}")
                return status

            logger.warning(f"Unknown routing_action: {routing_action}")
            return ""

        except Exception as e:
            logger.error(f"Error applying interjection routing: {e}", exc_info=True)
            msg = f"Routing error: {e}"
            self.action_log.append(f"ROUTING ERROR: {msg}")
            return msg

    async def _execute_interjection_decision(
        self,
        decision: InterjectionDecision,
    ) -> str:
        """Executes the action decided by the Interjection Handler LLM."""
        if decision.action == "modify_task" and decision.patches:
            self.action_log.append("Executing stateful decision: modify_task.")

            run_id_being_cancelled = self.run_id
            await self._cancel_and_wait_for_task(
                self._execution_task,
                "modify_task interjection",
            )
            await self._clear_browser_queue_for_run(run_id_being_cancelled)

            original_call_stack = list(self.call_stack)
            first_modified_function_index = -1
            first_modified_function_name = None
            for i, func_name in enumerate(original_call_stack):
                if any(p.function_name == func_name for p in decision.patches):
                    first_modified_function_index = i
                    first_modified_function_name = func_name
                    break

            try:
                keys_to_delete = self._resolve_invalidation_keys(
                    decision,
                )
            except Exception as e:
                logger.warning(
                    f"Error during selective cache invalidation: {e}. For safety, clearing the entire cache.",
                )
                keys_to_delete = set(self.idempotency_cache.keys())

            target_screenshot = None
            trajectory = []
            if keys_to_delete:
                logger.info(
                    f"Invalidating {len(keys_to_delete)} cache entries due to interjection.",
                )
                self.action_log.append(
                    f"Invalidating {len(keys_to_delete)} cache entries due to interjection.",
                )

                all_keys_sorted = sorted(
                    self.idempotency_cache.keys(),
                    key=lambda k: self.idempotency_cache[k]
                    .get("meta", {})
                    .get("step", 0),
                )

                last_valid_key = None
                for key in all_keys_sorted:
                    if key not in keys_to_delete:
                        last_valid_key = key
                    else:
                        break

                if last_valid_key:
                    last_valid_entry = self.idempotency_cache.get(last_valid_key)
                    if last_valid_entry and "meta" in last_valid_entry:
                        target_screenshot = last_valid_entry["meta"].get(
                            "post_state_screenshot",
                        )

                invalidated_keys_sorted = sorted(
                    [k for k in all_keys_sorted if k in keys_to_delete],
                    key=lambda k: self.idempotency_cache[k]
                    .get("meta", {})
                    .get("step", 0),
                )
                for key in invalidated_keys_sorted:
                    entry = self.idempotency_cache.get(key)
                    if entry and "interaction_log" in entry:
                        call_repr = (
                            entry["interaction_log"][1]
                            if len(entry["interaction_log"]) > 1
                            else str(entry)
                        )
                        trajectory.append(call_repr)

            invalidated_handles = set()
            for key in keys_to_delete:
                entry = self.idempotency_cache.pop(key, None)
                if (
                    entry
                    and isinstance(entry.get("result"), str)
                    and entry["interaction_log"][2].startswith("Returned handle")
                ):
                    hid = entry["result"]
                    invalidated_handles.add(hid)
                    self.live_handles.pop(hid, None)

            if invalidated_handles:
                logger.info(
                    f"Cleaning up cached method calls for {len(invalidated_handles)} invalidated handles.",
                )
                self.action_log.append(
                    f"Cleaning up cached method calls for {len(invalidated_handles)} invalidated handles.",
                )
                for k in list(self.idempotency_cache.keys()):
                    meta_tool = (
                        self.idempotency_cache.get(k, {})
                        .get("meta", {})
                        .get("tool", "")
                    )
                    if any(f":{hid}." in meta_tool for hid in invalidated_handles):
                        self.idempotency_cache.pop(k, None)

            # Course correction is only applicable for browser-based workflows.
            if (
                trajectory
                and target_screenshot
                and "computer_primitives" in (self.actor.environments or {})
            ):
                self.action_log.append(
                    f"COURSE CORRECTION: Launching recovery agent to reverse {len(trajectory)} invalidated actions.",
                )
                logger.info(
                    f"Launching course correction agent to reverse {len(trajectory)} actions.",
                )
                try:
                    await self.actor._run_course_correction_agent(
                        plan=self,
                        target_screenshot=target_screenshot,
                        trajectory=trajectory,
                    )
                    self.action_log.append(
                        "COURSE CORRECTION: Recovery agent completed successfully.",
                    )
                    logger.info("Course correction completed successfully.")
                except Exception as e:
                    logger.error(
                        f"Course correction failed: {e}",
                        exc_info=True,
                    )
                    self.action_log.append(
                        f"WARNING: Course correction failed: {e}. Proceeding with replay from current state.",
                    )
            elif trajectory and not target_screenshot:
                logger.debug(
                    "Skipping course correction: actions were invalidated but no screenshot state was available. "
                    "Relying on cache replay for state recovery.",
                )

            modification_summary = ", ".join(
                [p.function_name for p in decision.patches],
            )
            self.action_log.append(
                f"Applying patches for functions: {modification_summary}",
            )

            for patch in decision.patches:
                self._update_plan_with_new_code(patch.function_name, patch.new_code)

            modification_reason = decision.reason
            if self.goal:
                new_goal = (
                    f"{self.goal}\n\nIMPORTANT UPDATE: The user has provided a new instruction to modify the "
                    f"plan: '{modification_reason}'"
                )
                self.action_log.append(
                    f"Updating plan goal to reflect interjection. New goal: '{new_goal}'",
                )
                self.goal = new_goal
            else:
                self.goal = f"Incrementally taught plan:\n- {modification_reason}"

            self._restart_execution_loop("modify_task interjection")

            if self._state in (
                _HierarchicalHandleState.PAUSED,
                _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            ):
                self.runtime.resume()

            return f"Plan modification for '{modification_summary}' applied. Resuming execution from a clean state."

        elif decision.action == "replace_task":
            self.action_log.append(
                f"Executing decision: replace_task with new goal: '{decision.new_goal}'",
            )
            logger.debug(
                f"Replace task triggered. New goal: '{decision.new_goal}'. Proceeding to re-initialize plan.",
            )

            run_id_being_cancelled = self.run_id
            await self._cancel_and_wait_for_task(
                self._execution_task,
                "replace_task interjection",
            )
            await self._clear_browser_queue_for_run(run_id_being_cancelled)

            self.goal = decision.new_goal
            self.plan_source_code = None
            self.idempotency_cache.clear()
            self.live_handles.clear()
            self.runtime = PlanRuntime()
            self.call_stack.clear()
            self.skipped_functions.clear()
            self._is_complete = False
            self._completion_event.clear()
            self._final_result_str = None

            self._execution_task = asyncio.create_task(
                self._initialize_and_run(mode="fresh_after_replace_task"),
            )
            return (
                f"Plan has been re-initialized with a new goal: '{decision.new_goal}'"
            )

        elif decision.action == "refactor_and_generalize":
            self.action_log.append(
                f"Executing decision: refactor_and_generalize. Context: '{decision.generalization_context}'",
            )
            logger.debug(
                "Refactor and generalize triggered. Generating a refactored plan and restarting execution.",
            )

            run_id_being_cancelled = self.run_id
            await self._cancel_and_wait_for_task(
                self._execution_task,
                "refactor_and_generalize interjection",
            )
            await self._clear_browser_queue_for_run(run_id_being_cancelled)

            monolithic_code = (
                "\n\n".join(
                    self.clean_function_source_map.values(),
                )
                if self.clean_function_source_map
                else ""
            )
            current_url = None
            if "computer_primitives" in (self.actor.environments or {}):
                try:
                    computer_primitives = self._get_computer_primitives()
                    current_url = await computer_primitives.browser.get_current_url()
                except Exception as e:
                    logger.debug(f"Could not get current URL: {e}")
            refactor_prompt = prompt_builders.build_refactor_prompt(
                monolithic_code=monolithic_code,
                generalization_request=decision.generalization_context,
                action_log="\n".join(self.action_log),
                current_url=current_url,
                tools=self.actor.tools,
                environments=self.actor.environments,
            )

            self.plan_generation_client.set_response_format(RefactorDecision)
            try:
                response_str = await llm_call(
                    self.plan_generation_client,
                    refactor_prompt,
                )
                refactor_decision = RefactorDecision.model_validate_json(response_str)
            finally:
                self.plan_generation_client.reset_response_format()

            modification_reason = decision.reason
            if self.goal:
                new_goal = f"{self.goal}\n\nIMPORTANT UPDATE: The user has generalized the task with a new subject: '{modification_reason}'"
                self.action_log.append(
                    f"Updating plan goal to reflect generalization. New goal: '{new_goal}'",
                )
                self.goal = new_goal
            else:
                self.goal = f"Incrementally taught and generalized plan:\n- {modification_reason}"

            self.action_log.append("Replacing old plan with newly refactored version.")
            self.plan_source_code = self.actor._sanitize_code(
                refactor_decision.refactored_code,
                self,
            )
            self.actor._load_plan_module(self)

            self.action_log.append(
                "CACHE INVALIDATION: Clearing entire cache after refactoring to ensure a clean state for the new, generalized plan.",
            )
            logger.info("Clearing idempotency cache due to refactor_and_generalize.")
            self.idempotency_cache.clear()

            self._restart_execution_loop("refactor_and_generalize interjection")

            return "Plan successfully refactored. Resuming execution with the new modular plan."

        elif decision.action == "explore_detached":
            self.action_log.append(
                f"Executing decision: explore_detached for goal: '{decision.new_goal}'",
            )

            # Tab isolation is only applicable for browser-based workflows.
            use_tab_isolation = "computer_primitives" in (self.actor.environments or {})

            computer_primitives = None
            original_tab_index = None
            original_url = None
            try:
                if use_tab_isolation:
                    try:
                        computer_primitives = self._get_computer_primitives()

                        class TabState(BaseModel):
                            current_tab_index: int | None = Field(
                                None,
                                description="The index of the current tab. Return None if the tab index cannot be determined from the visible content.",
                            )

                        try:
                            original_tab_index = await computer_primitives.observe(
                                "Look at the browser tabs at the top of the screen. What is the numerical index (starting from 0) of the currently active/selected tab? If you cannot see clear tab indicators or determine the active tab index, return null for current_tab_index.",
                                response_format=TabState,
                            )
                            original_url = (
                                await computer_primitives.browser.get_current_url()
                            )
                        except Exception as e:
                            self.action_log.append(
                                f"SANDBOX: Could not record tab state: {e}",
                            )
                            original_tab_index = TabState(current_tab_index=0)
                            original_url = (
                                await computer_primitives.browser.get_current_url()
                            )

                        self.action_log.append(
                            "SANDBOX: Opening new tab for exploration",
                        )
                        await computer_primitives.act(
                            f"Open a new tab navigating to the url {original_url} and ensure the new tab is active",
                        )
                    except Exception as e:
                        logger.warning(
                            f"Tab isolation failed: {e}. Proceeding without tab isolation.",
                            exc_info=True,
                        )
                        use_tab_isolation = False
                else:
                    self.action_log.append(
                        "SANDBOX: Running detached exploration without tab isolation (no browser environment active)",
                    )

                self.action_log.append(
                    f"SANDBOX: Starting sub-plan for goal: '{decision.new_goal}'",
                )
                sandbox_plan = HierarchicalActorHandle(
                    actor=self.actor,
                    goal=decision.new_goal,
                    parent_chat_context=self.parent_chat_context,
                    clarification_up_q=self.clarification_up_q,
                    clarification_down_q=self.clarification_down_q,
                    persist=False,
                )

                sandbox_result = await sandbox_plan.result()
                self.action_log.append(
                    f"Sandbox plan finished with result: {sandbox_result}",
                )

                self.action_log.append(
                    "SANDBOX: Analyzing sandbox results for potential merge.",
                )

                merge_prompt = prompt_builders.build_sandbox_merge_prompt(
                    main_goal=self.goal or "This is a teaching session",
                    main_plan_source=self.plan_source_code
                    or "No plan source available",
                    sandbox_goal=decision.new_goal,
                    sandbox_result=str(sandbox_result),
                )

                self.modification_client.set_response_format(SandboxMergeDecision)
                try:
                    decision_str = await llm_call(
                        self.modification_client,
                        merge_prompt,
                    )
                    merge_decision = SandboxMergeDecision.model_validate_json(
                        decision_str,
                    )
                finally:
                    self.modification_client.reset_response_format()

                if merge_decision.modification_needed:
                    self.action_log.append(
                        f"SANDBOX MERGE: Decision to modify main plan. Reason: {merge_decision.reason}",
                    )

                    new_interjection = InterjectionDecision(
                        action="modify_task",
                        reason=merge_decision.reason,
                        patches=[
                            FunctionPatch(
                                function_name=self._get_main_function_name()
                                or "main_plan",
                                new_code=merge_decision.modification_request or "",
                            ),
                        ],
                    )
                    self._sandbox_merge_result = "Detached exploration completed and findings are being merged into the main plan."
                    self._pending_merge_interjection = new_interjection
                else:
                    self.action_log.append(
                        "SANDBOX MERGE: No modifications needed. Resuming main plan.",
                    )
                    self._sandbox_merge_result = (
                        "Detached exploration completed. Resuming main plan."
                    )
                    self._pending_merge_interjection = None

            finally:
                if use_tab_isolation and original_tab_index is not None:
                    self.action_log.append("SANDBOX: Returning to original tab")
                    try:
                        if computer_primitives is None:
                            computer_primitives = self._get_computer_primitives()
                        tab_index = (
                            original_tab_index.current_tab_index
                            if original_tab_index.current_tab_index is not None
                            else 0
                        )
                        await computer_primitives.act(
                            f"Switch to tab {tab_index} which was on the url {original_url} to go back to the original tab",
                        )
                        self.action_log.append("SANDBOX: Returned to original tab")
                    except Exception as e:
                        self.action_log.append(
                            f"SANDBOX: Error returning to original tab: {e}",
                        )

            if (
                hasattr(self, "_pending_merge_interjection")
                and self._pending_merge_interjection
            ):
                await self._execute_interjection_decision(
                    self._pending_merge_interjection,
                )
                delattr(self, "_pending_merge_interjection")

            result_msg = getattr(
                self,
                "_sandbox_merge_result",
                "Detached exploration completed. Resuming main plan.",
            )
            if hasattr(self, "_sandbox_merge_result"):
                delattr(self, "_sandbox_merge_result")

            return result_msg

        elif decision.action == "clarify":
            self.action_log.append(f"Executing decision: clarify.")
            if decision.clarification_question:
                await self.clarification_up_q.put(decision.clarification_question)
                return "Clarification requested from user to determine next steps."
            else:
                return "Clarification needed, but no question was provided."

        elif decision.action == "complete_task":
            self.action_log.append("Executing decision: complete_task.")
            return await self.stop(final_result="Plan completed by user instruction.")

        # If the LLM chooses modify_task without patches, treat it as "no plan change needed".
        if decision.action == "modify_task" and not decision.patches:
            self.action_log.append(
                "Interjection chose modify_task but provided no patches; treating as no-op.",
            )
            return "No plan modifications provided; continuing execution."

        return "Error: Unknown or unsupported interjection action."

    async def stop(
        self,
        final_result: str | None = None,
        *,
        reason: str | None = None,
        cancel: bool | None = None,
    ) -> str:
        """
        Stops the plan's execution permanently.
        In persist mode, this is a graceful shutdown. Otherwise, it is a hard cancel.

        Parameters
        ----------
        final_result : str | None
            Optional final message to record as the plan's result.
        reason : str | None
            Optional human-readable reason appended to the final result.
        cancel : bool | None
            If True, perform a hard cancel (STOPPED). If False, perform a
            graceful stop (COMPLETED). If None, preserve legacy behaviour:
            COMPLETED when `persist` is True, else STOPPED.

        Returns:
            A status message.
        """

        if self.dedicated_computer_primitives is not None:
            try:
                logger.info(
                    "Stopping dedicated action provider session for plan %s.",
                    self._module_name,
                )
                self.dedicated_computer_primitives.browser.stop()
            except Exception as exc:
                logger.warning(
                    "Failed to stop dedicated action provider session for plan %s: %s",
                    self._module_name,
                    exc,
                    exc_info=True,
                )
            finally:
                self.dedicated_computer_primitives = None

        await self._cancel_all_background_tasks()
        if self._is_complete:
            return f"Plan already in terminal state: {self._state.name}."

        # Cleanup pane (watchers) best-effort.
        try:
            if getattr(self, "pane", None) is not None:
                await self.pane.cleanup()
        except Exception as e:
            logger.debug(f"Pane cleanup failed: {e}", exc_info=True)

        if final_result is not None:
            base_msg = final_result
        else:
            base_msg = (
                "Plan was cancelled by user."
                if cancel is True
                else "Plan was stopped by user."
            )
        result_str = base_msg if not reason else f"{base_msg} Reason: {reason}"

        self.action_log.append(f"stop() called. Final result: '{result_str}'")
        self._cleanup_temp_file()

        if cancel is None:
            if self.persist:
                self._set_state(_HierarchicalHandleState.COMPLETED)
                self._set_final_result(result_str)
                try:
                    if hasattr(self, "_done_events") and not self._done_events.empty():
                        event_to_signal = self._done_events.get_nowait()
                        event_to_signal.set()
                except Exception:
                    pass
            else:
                self._set_state(_HierarchicalHandleState.STOPPED)
                self._set_final_result(result_str)
        else:
            if cancel is False:

                self._set_state(_HierarchicalHandleState.COMPLETED)
                self._set_final_result(result_str)
                try:
                    if hasattr(self, "_done_events") and not self._done_events.empty():
                        event_to_signal = self._done_events.get_nowait()
                        event_to_signal.set()
                except Exception:
                    pass
            else:

                self._set_state(_HierarchicalHandleState.STOPPED)
                self._set_final_result(result_str)

        try:
            self.runtime._release_from_checkpoint()
        except Exception:
            pass
        try:
            self.runtime.resume()
        except Exception:
            pass
        return result_str

    async def pause(self, immediate: bool = False) -> str:
        """
        Pauses the plan's execution.

        Args:
            immediate: If True, interrupts any currently executing browser action.

        Returns:
            A status message.
        """
        if self._state in (
            _HierarchicalHandleState.RUNNING,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        ):
            if immediate:
                computer_primitives = self._get_computer_primitives()
                if computer_primitives and hasattr(
                    computer_primitives.browser,
                    "backend",
                ):
                    backend = computer_primitives.browser.backend
                    if hasattr(backend, "interrupt_current_action"):
                        logger.info("⚡ Sending interrupt to browser action...")
                        await backend.interrupt_current_action()

            self.runtime.pause()
            self._set_state(_HierarchicalHandleState.PAUSED)
            pause_type = "immediately" if immediate else "by user"
            self.action_log.append(f"Plan paused {pause_type}.")
            return "Plan paused."

        if self._state in (
            _HierarchicalHandleState.PAUSED,
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.STOPPED,
            _HierarchicalHandleState.ERROR,
        ):
            return f"Plan already in state {self._state.name}, no action taken."

        return f"Cannot pause in state {self._state.name}."

    async def resume(self) -> str:
        """
        Resumes a paused plan.

        Returns:
            A status message.
        """
        if self._state == _HierarchicalHandleState.PAUSED:
            self.runtime.resume()
            self._set_state(_HierarchicalHandleState.RUNNING)
            self.action_log.append("Plan resumed by user.")
            return "Plan resumed."
        return f"Cannot resume from state {self._state.name}."

    async def ask(self, question: str) -> SteerableToolHandle:
        """
        Asks a question about the current state of the plan by creating a new,
        isolated tool loop that returns a handle to its result. This loop
        has access to a query tool to answer questions about the agent's
        actions and memory.
        """
        full_context_log = "\n".join(f"- {log}" for log in self.action_log)

        # Capture evidence from all active environments
        evidence: Dict[str, Any] = {}
        for env_namespace, env in self.actor.environments.items():
            try:
                evidence[env_namespace] = await env.capture_state()
            except Exception as e:
                logger.warning(f"Failed to capture evidence from {env_namespace}: {e}")
                evidence[env_namespace] = {"type": "error", "error": str(e)}

        system_message = prompt_builders.build_ask_prompt(
            goal=self.goal,
            state=self._state.name,
            call_stack=" -> ".join(self.call_stack) or "None",
            context_log=full_context_log,
            question=question,
            environments=self.actor.environments,
            evidence=evidence,
        )

        self.ask_client.reset_messages()
        self.ask_client.reset_system_message()
        self.ask_client.set_system_message(system_message)

        # Conditional query tool based on environment
        if "computer_primitives" in self.actor.environments:

            async def query_tool(query: str) -> str:
                """Query the agent's memory and action history."""
                try:
                    return await self._get_computer_primitives().browser_query(query)
                except Exception as e:
                    return f"Error querying browser: {e}"

        else:

            async def query_tool(query: str) -> str:
                """Query the agent's memory and action history."""
                # Fallback: search through action log and interactions
                matching_logs = [
                    log for log in self.action_log if query.lower() in log.lower()
                ]
                if matching_logs:
                    return "\n".join(matching_logs[-5:])  # Last 5 matches
                return "No matching information found in action log."

        tools = {"query": query_tool}
        handle = start_async_tool_loop(
            client=self.ask_client,
            message=question,
            tools=tools,
        )

        self.action_log.append(f"USER ASKED: {question}")
        return handle

    async def next_clarification(self) -> dict:
        """Awaits the next clarification question from the running plan."""
        if not self.clarification_enabled:
            await asyncio.Event().wait()
            return {}
        question = await self.clarification_up_q.get()
        return {"question": question}

    async def next_notification(self) -> dict:
        """
        Awaits the next notification from the running plan.
        NOTE: This is not implemented for HierarchicalActorHandle and will wait indefinitely.
        """
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """
        Provides an answer to a pending clarification question.
        The call_id is ignored as this handle only manages one clarification channel.
        """
        if self.clarification_enabled:
            await self.clarification_down_q.put(answer)

    def _is_valid_method(self, name: str) -> bool:
        """
        Checks if a given control method is valid in the current plan state.

        Args:
            name: The name of the method to check.

        Returns:
            True if the method is valid, False otherwise.
        """
        if name == "stop":
            return not self._is_complete
        if name == "pause":
            return self._state == _HierarchicalHandleState.RUNNING
        if name == "resume":
            return self._state == _HierarchicalHandleState.PAUSED
        if name == "ask":
            return self._state not in (
                _HierarchicalHandleState.IDLE,
                _HierarchicalHandleState.EXPLORING,
            )
        if name == "interject":
            return self._state in (
                _HierarchicalHandleState.RUNNING,
                _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            )
        return False


class HierarchicalActor(BaseActor):
    """
    Orchestrates task execution by generating and managing Python code.

    This actor takes a high-level goal, generates a Python script representing
    the plan, and then executes it in a controlled, self-correcting manner.
    """

    def __init__(
        self,
        function_manager: Optional["FunctionManager"] = None,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        max_escalations: Optional[int] = None,
        max_local_retries: Optional[int] = None,
        timeout: Optional[int] = 1000,
        browser_mode: str = "magnitude",
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
        *,
        connect_now: bool = False,
        can_compose: bool = True,
        can_store: bool = True,
        environments: Optional[list["BaseEnvironment"]] = None,
    ):
        """
        Initializes the HierarchicalActor.

        Args:
            function_manager: Manages a library of reusable functions.
            controller: The browser controller for executing `act` and `observe`.
            session_connect_url: URL for connecting to an existing browser session.
            headless: Whether to run the browser in headless mode.
            max_escalations: Default max number of strategic replans for plans.
            max_local_retries: Default max number of tactical retries for plans.
            timeout: Default timeout for plan execution.
            browser_mode: The browser mode to use. Can be "legacy" or "magnitude".
            agent_mode: The agent mode to use. Can be "browser" or "desktop".
            agent_server_url: The URL of the agent server to use. Can be used to connect to a remote client.
            connect_now: When False (default), defer any browser/agent connections until first use.
            can_compose: When True (default), allows the actor to generate new code on the fly.
                When False, the actor can only execute pre-existing functions via entrypoint.
            can_store: When True (default), allows verified functions to be persisted
                to the FunctionManager as reusable skills. When False, functions are
                executed but not stored.
            environments: Optional list of execution environments. If None, defaults to
                [ComputerEnvironment, StateManagerEnvironment].
        """
        # TODO: enable auto fetch desktop_url later
        # agent_server_url = self._get_desktop_url(agent_server_url)
        self.function_manager = (
            function_manager or ManagerRegistry.get_function_manager()
        )
        self._session_connect_url = session_connect_url
        self._headless = headless
        self._browser_mode = browser_mode
        self._agent_mode = agent_mode
        self._agent_server_url = agent_server_url
        self._connect_now = connect_now
        self.can_compose = can_compose
        self.can_store = can_store
        # Only construct ComputerPrimitives when a browser environment is actually configured.
        # If `environments` are provided explicitly, we must not implicitly introduce browser deps.
        self.computer_primitives: Optional[ComputerPrimitives] = None

        # Pluggable environments (domain-agnostic tool providers).
        if environments is None:
            from unity.actor.environments import (
                ComputerEnvironment,
                StateManagerEnvironment,
            )
            from unity.function_manager.primitives import Primitives

            self.computer_primitives = ComputerPrimitives(
                session_connect_url=session_connect_url,
                headless=headless,
                browser_mode=browser_mode,
                agent_mode=agent_mode,
                agent_server_url=agent_server_url,
                connect_now=connect_now,
            )
            primitives = Primitives()
            environments = [
                ComputerEnvironment(self.computer_primitives),
                StateManagerEnvironment(primitives),
            ]

        self.environments: dict[str, "BaseEnvironment"] = {
            env.namespace: env for env in environments
        }

        # If the provided environments include a browser environment, expose its instance
        # (used by some internal helpers and by plans that request a dedicated session).
        if (
            self.computer_primitives is None
            and "computer_primitives" in self.environments
        ):
            try:
                cp = self.environments["computer_primitives"].get_instance()
                if isinstance(cp, ComputerPrimitives):
                    self.computer_primitives = cp
            except Exception:
                self.computer_primitives = None

        # Metadata for tool purity/steerability (used by proxies; not by prompts).
        self.tool_metadata: dict[str, ToolMetadata] = {}
        for env in self.environments.values():
            try:
                self.tool_metadata.update(env.get_tools())
            except Exception as e:
                logger.warning(f"Failed to load tools for environment '{env}': {e}")

        # Tools exposed to prompt builders (mapping of tool name -> callable).
        #
        # IMPORTANT:
        # - We MUST honor the tool surface declared by each environment via `env.get_tools()`.
        # - Tool names are the fully-qualified names used inside plan execution
        #   (e.g. "computer_primitives.navigate", "primitives.contacts.ask").
        # - We enforce uniqueness across environments to avoid ambiguity in prompts.
        self.tools: dict[str, Any] = {}
        _tool_owners: dict[str, str] = {}

        def _resolve_tool_callable(
            *,
            env_namespace: str,
            instance: Any,
            tool_name: str,
        ) -> Any:
            if not tool_name.startswith(f"{env_namespace}."):
                raise ValueError(
                    "Environment tool name must be fully-qualified and match its environment namespace. "
                    f"Got tool_name='{tool_name}' for env_namespace='{env_namespace}'.",
                )
            # Strip the namespace and follow dotted attributes on the instance.
            # Example: env_namespace="primitives", tool_name="primitives.contacts.ask"
            # → path=["contacts", "ask"].
            attr_path = tool_name.split(".")[1:]
            target = instance
            for part in attr_path:
                if not hasattr(target, part):
                    raise AttributeError(
                        f"Tool '{tool_name}' could not be resolved: missing attribute '{part}' "
                        f"while traversing path {attr_path!r}.",
                    )
                target = getattr(target, part)
            if not callable(target):
                raise TypeError(
                    f"Tool '{tool_name}' resolved to a non-callable object of type {type(target)!r}.",
                )
            return target

        for env_namespace, env in self.environments.items():
            instance = env.get_instance()
            try:
                tools_metadata = env.get_tools()
            except Exception as e:
                logger.warning(
                    "Failed to load tools for environment '%s' while building prompt tools: %s",
                    env_namespace,
                    e,
                )
                continue

            for declared_name, meta in tools_metadata.items():
                tool_name = getattr(meta, "name", declared_name)
                tool_callable = _resolve_tool_callable(
                    env_namespace=env_namespace,
                    instance=instance,
                    tool_name=tool_name,
                )
                if tool_name in self.tools:
                    raise ValueError(
                        "Tool name collision detected while extracting environment tools: "
                        f"'{tool_name}' is provided by both '{_tool_owners[tool_name]}' and '{env_namespace}'. "
                        "Environment tool names must be unique.",
                    )
                self.tools[tool_name] = tool_callable
                _tool_owners[tool_name] = env_namespace
        self.max_escalations = max_escalations or 2
        self.max_local_retries = max_local_retries or 3
        self.timeout = timeout
        self._plan_handles: weakref.WeakSet = weakref.WeakSet()

    def _format_scoped_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Formats the dictionary from _get_scoped_context into a string for an LLM prompt.
        """
        prompt_parts = []

        if context.get("parent_source"):
            prompt_parts.append(
                "### Parent Source (Caller)\n"
                "This is the function that called the current one.\n"
                "```python\n"
                f"{context['parent_source']}\n"
                "```",
            )

        if context.get("current_source"):
            prompt_parts.append(
                "### Current Function Source\n"
                "This is the function currently being executed or implemented.\n"
                "```python\n"
                f"{context['current_source']}\n"
                "```",
            )

        if context.get("children_source"):
            children_str = []
            for name, source in context["children_source"].items():
                children_str.append(
                    f"# Child Function: {name}\n" f"{source}",
                )
            prompt_parts.append(
                "### Children Source (Functions it may call)\n"
                "This is the source code of other plan functions that the current function might call. "
                "```python\n"
                f"{chr(10).join(children_str)}\n"
                "```",
            )

        return f"\n\n{'-' * 3}\n\n".join(prompt_parts)

    def _get_scoped_context_from_plan_state(
        self,
        plan: "HierarchicalActorHandle",
    ) -> Dict[str, Any]:
        """
        Builds a scoped context dictionary using the plan's current call stack
        and source maps, without needing a live frame object.
        """
        context = {
            "parent_source": None,
            "current_source": None,
            "children_source": {},
        }

        if not plan.call_stack:
            return context
        try:
            current_func_name = plan.call_stack[-1]
            context["current_source"] = plan.clean_function_source_map.get(
                current_func_name,
            )

            if len(plan.call_stack) > 1:
                parent_func_name = plan.call_stack[-2]
                context["parent_source"] = plan.clean_function_source_map.get(
                    parent_func_name,
                )
        except IndexError:
            pass

        if context["current_source"]:

            class CallVisitor(ast.NodeVisitor):
                def __init__(self, plan_functions):
                    self.plan_functions = plan_functions
                    self.called_functions = set()

                def visit_Call(self, node: ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.plan_functions:
                            self.called_functions.add(node.func.id)
                    self.generic_visit(node)

            try:
                tree = ast.parse(context["current_source"])
                plan_function_names = set(plan.clean_function_source_map.keys())
                visitor = CallVisitor(plan_function_names)
                visitor.visit(tree)

                for child_name in visitor.called_functions:
                    if child_name in plan.clean_function_source_map:
                        context["children_source"][child_name] = (
                            plan.clean_function_source_map[child_name]
                        )
            except (SyntaxError, KeyError) as e:
                logger.warning(
                    f"Could not parse AST for child detection from plan state: {e}",
                )

        return context

    def _get_desktop_url(self, agent_server_url: str) -> str:
        """
        Resolve desktop_url from the orchestrator by assistant full name.

        Steps:
        - GET <UNIFY_BASE_URL>/assistant with Authorization: Bearer <UNIFY_KEY>
        - Match the assistant whose full name matches env ASSISTANT_NAME as "<first> <last>"
        - Return its "desktop_url" field

        Falls back to the provided agent_server_url on any failure.
        """
        orchestra_url = SETTINGS.UNIFY_BASE_URL
        unify_key = SESSION_DETAILS.unify_key
        assistant_name = SESSION_DETAILS.assistant.name.strip()

        if not orchestra_url or not unify_key or not assistant_name:
            return agent_server_url

        try:
            from unify.utils import http

            url = f"{orchestra_url.rstrip('/')}/assistant"
            headers = {"Authorization": f"Bearer {unify_key}"}
            resp = http.get(url, headers=headers, timeout=30, raise_for_status=False)
            if not (200 <= resp.status_code < 300):
                return agent_server_url
            try:
                payload = resp.json()
            except Exception:
                return agent_server_url

            assistants = []
            if isinstance(payload, list):
                assistants = payload
            elif isinstance(payload, dict) and isinstance(payload.get("info"), list):
                assistants = payload.get("info", [])
            else:
                return agent_server_url

            for a in assistants:
                try:
                    first = (a.get("first_name") or a.get("first") or "").strip()
                    last = (
                        a.get("surname") or a.get("last_name") or a.get("last") or ""
                    ).strip()
                    full = f"{first} {last}".strip()
                    if full == assistant_name:
                        desktop_url = a.get("desktop_url")
                        if isinstance(desktop_url, str) and desktop_url.strip():
                            return desktop_url
                        return agent_server_url
                except Exception:
                    continue
        except Exception:
            return agent_server_url

        return agent_server_url

    def _sanitize_code(self, code: str, plan: HierarchicalActorHandle) -> str:
        """
        Parses, sanitizes, and unparses code to enforce security.

        Args:
            code: The Python code string to sanitize.

        Returns:
            The sanitized code string.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    plan.clean_function_source_map[node.name] = ast.unparse(node)
            sanitizer = PlanSanitizer(plan)
            sanitized_tree = sanitizer.visit(tree)
            ast.fix_missing_locations(sanitized_tree)
            return ast.unparse(sanitized_tree)
        except SyntaxError as e:
            logger.error(f"Generated code failed sanitization: {e}")
            raise

    def _serialize_args(self, args: tuple, kwargs: dict) -> str:
        """Robustly serializes tool arguments for the cache key."""
        try:
            # NOTE: could make this more robust with a JSON serializer
            return repr((args, kwargs))
        except Exception:
            return f"ARGS:{str(args)}_KWARGS:{str(kwargs)}"

    def _generate_cache_key(
        self,
        plan: HierarchicalActorHandle,
        tool_name: str,
        args: tuple,
        kwargs: dict,
    ) -> tuple:
        """
        Generates the composite cache key for a tool call, including loop context.

        Key Structure: (call_stack, loop_context, branch_context, step_in_func, tool, args)
        This explicitly separates loop state from branch state for robust caching across iterations.
        """

        run_id = current_run_id_var.get()
        call_stack_tuple = plan.runtime.get_current_stack_tuple(run_id)

        loop_context_tuple = plan.runtime.get_current_loop_context_tuple()

        branch_path_tuple = tuple(plan.runtime.path_context)

        step_counter = plan.runtime.action_counter

        serialized_args = self._serialize_args(args, kwargs)

        cache_key = (
            call_stack_tuple,
            loop_context_tuple,
            branch_path_tuple,
            step_counter,
            tool_name,
            serialized_args,
        )

        logger.debug(
            f"Generated Cache Key: call_stack={call_stack_tuple}, loop_context={loop_context_tuple}, branch_path={branch_path_tuple}, step={step_counter}, tool={tool_name}",
        )
        return cache_key

    async def act(
        self,
        description: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        persist: bool = True,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        entrypoint: Optional[int] = None,
        new_session: bool = False,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        **kwargs,
    ) -> HierarchicalActorHandle:
        """
        Creates and starts a new HierarchicalActorHandle active task.

        Args:
            description: The high-level goal for the task.
            parent_chat_context: Chat context from a parent process.
            clarification_up_q: Queue for sending clarification questions.
            clarification_down_q: Queue for receiving clarification answers.
            persist: If True, plan will pause for interjections after completion. If False, plan will complete immediately.
            images: Optional mapping of source-scoped keys to ImageHandle objects.
            entrypoint: Optional. If provided, bypasses LLM plan generation
                and directly executes the specified function from the FunctionManager.
            new_session: If True, creates a new browser/desktop session for this plan. If False (default), reuses the actor's shared session.
            can_compose: If provided, overrides the actor's default can_compose setting
                for this call. When False, the actor can only execute pre-existing
                functions via entrypoint.
            can_store: If provided, overrides the actor's default can_store setting
                for this call. When False, verified functions are not persisted.

        Returns:
            An active handle to the running HierarchicalActorHandle.
        """
        dedicated_computer_primitives = None
        if new_session:
            dedicated_computer_primitives = ComputerPrimitives(
                session_connect_url=self._session_connect_url,
                headless=self._headless,
                browser_mode=self._browser_mode,
                agent_mode=self._agent_mode,
                agent_server_url=self._agent_server_url,
                connect_now=self._connect_now,
            )

        effective_can_compose = (
            can_compose if can_compose is not None else self.can_compose
        )
        effective_can_store = can_store if can_store is not None else self.can_store

        plan_handle = HierarchicalActorHandle(
            actor=self,
            goal=description,
            parent_chat_context=_parent_chat_context,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            max_escalations=self.max_escalations,
            max_local_retries=self.max_local_retries,
            persist=persist,
            images=images,
            entrypoint=entrypoint,
            entrypoint_args=entrypoint_args,
            entrypoint_kwargs=entrypoint_kwargs,
            dedicated_computer_primitives=dedicated_computer_primitives,
            can_compose=effective_can_compose,
            can_store=effective_can_store,
        )
        setattr(plan_handle, "__passthrough__", True)
        self._plan_handles.add(plan_handle)
        return plan_handle

    def _load_plan_module(self, plan: HierarchicalActorHandle):
        """
        Load plan source code as a module from a temporary file.
        """
        plans_dir = Path.cwd() / ".unity_plans"
        plans_dir.mkdir(exist_ok=True)

        if plan._temp_file_path is None:
            plan._temp_file_path = plans_dir / f"{plan._module_name}.py"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        goal_lines = plan.goal.split("\n") if plan.goal else ["No goal specified"]
        formatted_goal = "\n        # ".join(goal_lines)

        header = textwrap.dedent(
            f"""
        # Hierarchical Plan Script
        # Plan ID: {plan._module_name}
        # Goal: {formatted_goal}
        # Last Updated: {timestamp}
        #
        # This script is auto-generated and executed by the HierarchicalActor.
        # It is updated dynamically during the execution lifecycle.

        """,
        )

        full_script_content = f"{header}\n{plan.plan_source_code or 'pass'}"

        plan._temp_file_path.write_text(full_script_content)
        logger.info(f"Plan source code written to: {plan._temp_file_path}")

        if plan._module is None:
            spec = importlib.util.spec_from_file_location(
                plan._module_name,
                plan._temp_file_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Failed to create module spec for {plan._temp_file_path}",
                )

            plan._module = importlib.util.module_from_spec(spec)
            plan._module_spec = spec

            plan._module.__name__ = plan._module_name
            plan._module.__file__ = str(plan._temp_file_path)

            sys.modules[plan._module_name] = plan._module

            module_dict = plan._module.__dict__
            module_dict.update(plan.execution_namespace)

            spec.loader.exec_module(plan._module)
        else:
            if plan._module_name in sys.modules:
                del sys.modules[plan._module_name]

            spec = importlib.util.spec_from_file_location(
                plan._module_name,
                plan._temp_file_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Failed to recreate module spec for {plan._temp_file_path}",
                )

            plan._module = importlib.util.module_from_spec(spec)
            plan._module_spec = spec

            loader = spec.loader

            plan._module.__dict__.clear()

            plan._module.__name__ = plan._module_name
            plan._module.__file__ = str(plan._temp_file_path)
            plan._module.__package__ = None
            plan._module.__loader__ = loader
            plan._module.__spec__ = spec

            plan._module.__dict__.update(plan.execution_namespace)

            sys.modules[plan._module_name] = plan._module
            loader.exec_module(plan._module)

        plan.execution_namespace = plan._module.__dict__

        plan.function_source_map.clear()
        if plan.plan_source_code:
            try:
                tree = ast.parse(plan.plan_source_code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        plan.function_source_map[node.name] = ast.get_source_segment(
                            plan.plan_source_code,
                            node,
                        )
            except Exception:
                pass

    def _create_sandbox_globals(self) -> Dict[str, Any]:
        """Creates safe, sandboxed globals for plan execution."""
        from unity.function_manager.execution_env import create_execution_globals

        return create_execution_globals()

    async def _prepare_execution_environment(self, plan: HierarchicalActorHandle):
        """
        Prepares the sandboxed execution environment for a plan.
        """
        sandbox_globals = self._create_sandbox_globals()

        async def request_clarification_primitive(question: str) -> str:
            """Allows the plan to ask for clarification during execution."""
            if not plan.clarification_enabled:
                raise RuntimeError("Clarification is not supported in this context.")
            plan.action_log.append(f"Asking clarification: {question}")
            call_repr = f"request_clarification('{question}')"
            try:
                await asyncio.wait_for(
                    plan.clarification_up_q.put(question),
                    timeout=5,
                )
                answer = await asyncio.wait_for(
                    plan.clarification_down_q.get(),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                raise FatalVerificationError(
                    "Timed out waiting for user clarification.",
                )
            plan.action_log.append(f"Received clarification: {answer}")
            interaction_to_log = ("tool_call", call_repr, answer)
            interactions_log = current_interaction_sink_var.get()
            if interactions_log is not None:
                interactions_log.append(interaction_to_log)
            return answer

        plan.execution_namespace.clear()
        plan.execution_namespace.update(sandbox_globals)
        plan.execution_namespace["_cp"] = plan.runtime.checkpoint

        # Determine active environments (plan-local overrides actor defaults).
        active_envs: dict[str, BaseEnvironment] = {}
        if hasattr(plan, "environments") and isinstance(
            getattr(plan, "environments", None),
            dict,
        ):
            active_envs = plan.environments
        elif hasattr(self, "environments") and isinstance(
            getattr(self, "environments", None),
            dict,
        ):
            active_envs = self.environments

        computer_primitives: ComputerPrimitives | None = None
        if "computer_primitives" in active_envs:
            # Only resolve when actually configured; avoids implicit browser deps.
            computer_primitives = plan._get_computer_primitives()

        async def _int(func_name: str):
            req = plan.interruption_request
            if req and any(
                patch.function_name == func_name for patch in req.get("patches", [])
            ):
                plan.interruption_request = None
                raise _ControlledInterruptionException(
                    req.get("reason", "Interjection"),
                )

        plan.execution_namespace["_int"] = _int

        async def _around_cp(label: str, awaitable):
            await plan.runtime.checkpoint(f"Before: {label}")
            try:
                return await awaitable
            finally:
                await plan.runtime.checkpoint(f"After: {label}")

        plan.execution_namespace["_around_cp"] = _around_cp

        primitives_real: Any = None
        if "primitives" in active_envs:
            primitives_real = active_envs["primitives"].get_instance()

        # Inject all environment instances into the sandbox.
        # IMPORTANT: `computer_primitives` must come from the plan (it may be dedicated),
        # even if the actor has an environment instance for it.
        injected_namespaces: set[str] = set()

        if active_envs:
            for env_namespace, env in active_envs.items():
                if env_namespace == "computer_primitives":
                    if computer_primitives is None:
                        continue
                    plan.execution_namespace[env_namespace] = _ToolProviderProxy(
                        real_instance=computer_primitives,
                        plan=plan,
                        namespace=env_namespace,
                        environment=env,
                    )
                    injected_namespaces.add(env_namespace)
                    continue
                if env_namespace == "primitives":
                    # Keep the raw instance available for internal plumbing (e.g. venv calls),
                    # but expose only the proxied surface to plan code.
                    if primitives_real is not None:
                        plan.execution_namespace["_primitives"] = primitives_real
                        plan.execution_namespace[env_namespace] = _PrimitivesProxy(
                            primitives_real,
                            plan,
                        )
                        injected_namespaces.add(env_namespace)
                    continue

                # Future environments: inject via the unified tool-provider proxy.
                try:
                    plan.execution_namespace[env_namespace] = _ToolProviderProxy(
                        real_instance=env.get_instance(),
                        plan=plan,
                        namespace=env_namespace,
                        environment=env,
                    )
                    injected_namespaces.add(env_namespace)
                except Exception as e:
                    logger.warning(
                        f"Failed to inject environment '{env_namespace}' into execution namespace: {e}",
                    )

        # Keep existing helper injections (request_clarification, verify, etc.)
        plan.execution_namespace.update(
            {
                "request_clarification": request_clarification_primitive,
                "runtime": plan.runtime,
                "verify": self._create_verify_decorator(plan),
                "ReplanFromParentException": ReplanFromParentException,
                "_ForcedRetryException": _ForcedRetryException,
                "FatalVerificationError": FatalVerificationError,
                "_ControlledInterruptionException": _ControlledInterruptionException,
            },
        )

        # Inject venv functions as atomic callable proxies
        # These run in subprocess via execute_in_venv, treated like external primitives
        await self._inject_venv_function_proxies(
            plan,
            computer_primitives,
            primitives=plan.execution_namespace.get("_primitives"),
        )

        self._load_plan_module(plan)

    async def _inject_venv_function_proxies(
        self,
        plan: HierarchicalActorHandle,
        computer_primitives: Any = None,
        primitives: Any = None,
    ):
        """
        Inject venv functions as callable proxies into the execution namespace.

        Venv functions (those with venv_id != None) run in isolated subprocess
        environments. They are treated as atomic, opaque callables - the LLM can
        see their signature and docstring but cannot step into or verify their code.

        This allows generated plans to call venv functions like any other function,
        while respecting the isolation requirements of custom virtual environments.
        """
        if not self.function_manager:
            return

        try:
            # Query all venv functions
            venv_functions = self.function_manager.search_functions(
                filter="venv_id != None",
                limit=1000,
            )

            if not venv_functions:
                logger.debug("No venv functions found to inject.")
                return

            logger.info(f"Injecting {len(venv_functions)} venv function proxies.")

            # Get primitives for RPC access from venv subprocess.
            # Reuse the same `primitives` object injected into the plan, if available,
            # so behavior is consistent across plan code and venv calls.
            if primitives is None:
                from unity.function_manager.primitives import Primitives

                primitives = Primitives()

            for func_data in venv_functions:
                func_name = func_data.get("name")
                if not func_name:
                    continue

                # Create proxy that wraps the venv function
                proxy = _VenvFunctionProxy(
                    function_manager=self.function_manager,
                    func_data=func_data,
                    plan=plan,
                    primitives=primitives,
                    computer_primitives=computer_primitives,
                )

                # Inject into execution namespace
                plan.execution_namespace[func_name] = proxy
                logger.debug(f"Injected venv function proxy: {func_name}")

        except Exception as e:
            logger.warning(f"Failed to inject venv function proxies: {e}")

    async def _run_course_correction_agent(
        self,
        plan: HierarchicalActorHandle,
        target_screenshot: bytes,
        trajectory: list[str],
    ) -> None:
        """Spawns a new CodeActActor instance as a sub-agent to perform state recovery."""
        logger.info(
            f"[COURSE_CORRECTION] Starting course correction agent.",
        )
        logger.debug(f"[COURSE_CORRECTION] Trajectory: {trajectory}")
        from .code_act_actor import CodeActActor

        computer_primitives = plan._get_computer_primitives()

        logger.info("[COURSE_CORRECTION] Getting current screenshot...")
        current_screenshot = await computer_primitives.browser.get_screenshot()
        logger.info("[COURSE_CORRECTION] Got current screenshot.")

        if isinstance(current_screenshot, str):
            current_screenshot = base64.b64decode(current_screenshot)
        if isinstance(target_screenshot, str):
            target_screenshot = base64.b64decode(target_screenshot)

        logger.info(
            "[COURSE_CORRECTION] Screenshots decoded. Preparing image manager...",
        )

        image_manager = ManagerRegistry.get_image_manager()
        ids = image_manager.add_images(
            [
                {
                    "data": base64.b64encode(current_screenshot).decode("ascii"),
                    "caption": "Current browser state (before correction)",
                },
                {
                    "data": base64.b64encode(target_screenshot).decode("ascii"),
                    "caption": "Target browser state (after correction)",
                },
            ],
        )
        current_id, target_id = ids[0], ids[1]

        images_for_sub_agent = ImageRefs(
            [
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=current_id),
                    annotation="Current state: where the browser is now",
                ),
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=target_id),
                    annotation="Target state: where the browser should be",
                ),
            ],
        )

        formatted_trajectory = "\n".join(f"- `{action}`" for action in trajectory)
        correction_goal = f"""
        ### Your Mission: Course Correction
        Your goal is to restore the browser's state by writing and executing Python code in an iterative loop.

        **CONTEXT:**
        - **Current State:** The first image provided. This is the state you are starting from.
        - **Target State:** The second image provided. This is the state you must return to.
        - **Trajectory:** The following actions led from the target state to the current state. Use this as a guide for what to reverse:
        {formatted_trajectory}

        **YOUR WORKFLOW:**
        1.  **Analyze:** Review the injected image overview (tool result) in the transcript that lists all available images with their annotations and metadata. Compare them to determine if a correction is needed. If they are the same, you are done.
        2.  **Plan & Execute:** Write Python code using the `computer_primitives` to reverse the trajectory. This may take multiple steps. After each step, you will receive new visual feedback to guide your next action.
        3.  **Complete:** When the browser state visually matches the target state, your work is done.
        """

        logger.info("COURSE CORRECTION: Starting recovery sub-agent...")

        correction_agent = CodeActActor(
            computer_primitives=computer_primitives,
            timeout=300,
        )

        try:
            correction_plan = await correction_agent.act(
                description=correction_goal,
                images=images_for_sub_agent,
                persist=False,
            )

            await correction_plan.result()
            logger.info("COURSE CORRECTION: Recovery sub-agent finished successfully.")
        finally:
            pass

    def _create_verify_decorator(self, plan: HierarchicalActorHandle):
        """
        Creates the @verify decorator for a given plan instance.

        The decorator wraps each function in the plan to implement the
        execution, verification, and self-correction loop *locally*,
        including handling escalations to parent functions.
        """

        def verify(fn):
            """The actual decorator that wraps plan functions."""

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                """The wrapper that performs verification and correction."""
                # Only resolve browser primitives when a browser environment is active.
                # Primitives-only deployments should not require (or even have) a browser.
                active_envs = (
                    plan.environments
                    if hasattr(plan, "environments")
                    and isinstance(getattr(plan, "environments", None), dict)
                    else plan.actor.environments
                )
                computer_primitives: ComputerPrimitives | None = None
                if "computer_primitives" in (active_envs or {}):
                    try:
                        computer_primitives = plan._get_computer_primitives()
                    except Exception:
                        computer_primitives = None
                while True:
                    context_rid = current_run_id_var.get()
                    plan_rid = plan.run_id
                    if context_rid != plan_rid:
                        logger.warning(
                            f"Blocked stale function call to '{fn.__name__}'. "
                            f"Context run_id={context_rid} does not match plan run_id={plan_rid}.",
                        )
                        plan.action_log.append(
                            f"Blocked stale function call to '{fn.__name__}'. Context run_id={context_rid} does not match plan run_id={plan_rid}.",
                        )
                        raise asyncio.CancelledError(
                            f"Stale function call to '{fn.__name__}' blocked by run_id gate.",
                        )

                    func_name = fn.__name__
                    if func_name in plan.skipped_functions:
                        plan.action_log.append(f"SKIPPING function '{func_name}'.")
                        plan.skipped_functions.remove(func_name)
                        return

                    plan.invocation_counter += 1
                    invocation_id = f"{func_name}_{plan.invocation_counter}"

                    frame_token = plan.runtime.push_frame(plan.run_id, func_name)
                    plan.call_stack.append(func_name)

                    local_interactions = []
                    parent_sink = current_interaction_sink_var.get(None)

                    run_id_token = current_run_id_var.set(plan.run_id)
                    sink_token = current_interaction_sink_var.set(local_interactions)
                    invoc_token = current_invocation_id_var.set(invocation_id)

                    diag_prefix = (
                        f"[run_id={plan.run_id} invoc={invocation_id}]"
                        if DIAGNOSTIC_MODE
                        else ""
                    )

                    args_repr = [repr(a) for a in args]
                    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                    all_args = ", ".join(args_repr + kwargs_repr)
                    plan.action_log.append(
                        f"{diag_prefix} -> Entering '{func_name}' with args: ({all_args})",
                    )

                    logger.info(f"{diag_prefix} VERIFY: Entering '{func_name}'")
                    parent_action_counter = plan.runtime.action_counter
                    plan.runtime.action_counter = 0

                    step_cache_miss_counter = 0
                    plan.runtime.cache_miss_counter.append(0)

                    # TODO: remove this if preconditions are no longer needed in favor of the new sub-agent course correction
                    # if not plan.runtime.execution_mode.startswith("replay_"):
                    #     await self._ensure_precondition(plan, func_name)

                    # Capture pane event boundary (index-based, robust for concurrent watcher emissions).
                    pane_event_idx_before = 0
                    try:
                        if getattr(plan, "pane", None) is not None:
                            pane_event_idx_before = len(plan.pane._events_log)
                    except Exception:
                        pane_event_idx_before = 0

                    # Gather pre-execution evidence from all active environments.
                    pre_state: dict[str, Any] = {}
                    active_envs = (
                        plan.environments
                        if hasattr(plan, "environments")
                        and isinstance(getattr(plan, "environments", None), dict)
                        else plan.actor.environments
                    )
                    for env_namespace, env in active_envs.items():
                        try:
                            pre_state[env_namespace] = await env.capture_state()
                        except Exception as e:
                            logger.error(
                                f"Failed to capture pre-state from {env_namespace}: {e}",
                            )
                            pre_state[env_namespace] = {
                                "type": "error",
                                "error": str(e),
                            }

                    start_seq = -1
                    if computer_primitives is not None and isinstance(
                        computer_primitives.browser.backend,
                        MagnitudeBrowserBackend,
                    ):
                        start_seq = computer_primitives.browser.backend.current_seq

                    last_error_reason = ""
                    result = None
                    try:
                        for i in range(plan.MAX_LOCAL_RETRIES):
                            plan.runtime.action_counter = 0
                            if i > 0:
                                local_interactions.clear()
                            try:

                                frame = inspect.currentframe()
                                current_fn_for_execution = None
                                try:
                                    if (
                                        frame
                                        and frame.f_back
                                        and func_name in frame.f_back.f_locals
                                    ):
                                        current_fn_for_execution = (
                                            frame.f_back.f_locals[func_name]
                                        )

                                    elif func_name in plan.execution_namespace:
                                        current_fn_for_execution = (
                                            plan.execution_namespace[func_name]
                                        )

                                    else:
                                        raise KeyError(
                                            f"Function '{func_name}' not found in local or global scope.",
                                        )
                                finally:
                                    del frame

                                captured_run_id = current_run_id_var.get()
                                result = await inspect.unwrap(current_fn_for_execution)(
                                    *args,
                                    **kwargs,
                                )
                                if captured_run_id != plan.run_id:
                                    logger.warning(
                                        f"Discarding stale verification for '{func_name}' from a previous run (ID: {captured_run_id}).",
                                    )
                                    plan.action_log.append(
                                        f"Stale verification for '{func_name}' discarded.",
                                    )
                                    raise _ControlledInterruptionException(
                                        "Stale verification.",
                                    )

                                break

                            except _ControlledInterruptionException as e:
                                plan.action_log.append(
                                    f"{diag_prefix} Retrying '{func_name}' Reason: {e}",
                                )
                                logger.info(
                                    f"{diag_prefix} Retrying '{func_name}' Reason: {e}",
                                )
                                local_interactions.clear()
                                continue

                            except _ForcedRetryException:
                                plan.action_log.append(
                                    f"{diag_prefix} Retrying '{func_name}' after successful reimplementation.",
                                )
                                logger.info(
                                    f"{diag_prefix} Retrying '{func_name}' after successful reimplementation.",
                                )
                                local_interactions.clear()
                                continue

                            except NotImplementedError as e:
                                plan.action_log.append(
                                    f"{diag_prefix} '{func_name}' not implemented. Implementing JIT.",
                                )
                                logger.info(
                                    f"{diag_prefix} '{func_name}' not implemented. Implementing JIT.",
                                )
                                last_error_reason = str(e) or "Function is a stub."
                                await plan._handle_dynamic_implementation(
                                    func_name,
                                    replan_reason=f"Implement from stub: {last_error_reason}",
                                    call_stack_snapshot=list(
                                        plan.runtime.get_current_stack_tuple(
                                            plan.run_id,
                                        ),
                                    ),
                                    scoped_context_snapshot=self._get_scoped_context_from_plan_state(
                                        plan,
                                    ),
                                )
                                local_interactions.clear()
                                continue

                            except ReplanFromParentException as e:
                                plan.action_log.append(
                                    f"Child of '{func_name}' requested strategic replan. Reason: {e.reason}",
                                )
                                last_error_reason = e.reason

                                existing_code = plan.clean_function_source_map.get(
                                    func_name,
                                )

                                current_call_stack_snapshot = list(
                                    plan.runtime.get_current_stack_tuple(plan.run_id),
                                )
                                current_scoped_context_snapshot = (
                                    self._get_scoped_context_from_plan_state(plan)
                                )

                                await plan._handle_dynamic_implementation(
                                    func_name,
                                    replan_reason=last_error_reason,
                                    existing_code_for_modification=existing_code,
                                    call_stack_snapshot=current_call_stack_snapshot,
                                    scoped_context_snapshot=current_scoped_context_snapshot,
                                )
                                local_interactions.clear()
                                continue

                            except FatalVerificationError:
                                raise

                            except (BrowserAgentError, Exception) as e:
                                logger.error(
                                    f"Function '{func_name}' failed with a runtime error on attempt {i+1}: {e}",
                                    exc_info=True,
                                )
                                last_error_reason = traceback.format_exc()
                                existing_code = plan.clean_function_source_map.get(
                                    func_name,
                                )
                                await plan._handle_dynamic_implementation(
                                    func_name,
                                    replan_reason=f"Function crashed. Fix bug:\n{last_error_reason}",
                                    existing_code_for_modification=existing_code,
                                    call_stack_snapshot=list(
                                        plan.runtime.get_current_stack_tuple(
                                            plan.run_id,
                                        ),
                                    ),
                                    scoped_context_snapshot=self._get_scoped_context_from_plan_state(
                                        plan,
                                    ),
                                )
                                local_interactions.clear()
                                continue
                        else:
                            if plan.clarification_enabled:
                                plan.action_log.append(
                                    f"Function '{func_name}' has failed all {plan.MAX_LOCAL_RETRIES} retries. Asking user for guidance.",
                                )
                                clarification_question = (
                                    f"I've been unable to complete the step '{func_name}'. "
                                    f"The last issue was: {last_error_reason}. How should I proceed?"
                                )
                                user_answer = await plan.execution_namespace[
                                    "request_clarification"
                                ](clarification_question)
                                plan.action_log.append(
                                    f"Received user guidance: {user_answer}",
                                )

                                await plan._handle_dynamic_implementation(
                                    func_name,
                                    replan_reason=f"Function failed all retries. User provided new guidance: {user_answer}",
                                    clarification_question=clarification_question,
                                    clarification_answer=user_answer,
                                    call_stack_snapshot=list(
                                        plan.runtime.get_current_stack_tuple(
                                            plan.run_id,
                                        ),
                                    ),
                                    scoped_context_snapshot=self._get_scoped_context_from_plan_state(
                                        plan,
                                    ),
                                )
                                plan.action_log.append(
                                    f"Restarting execution of '{func_name}' after user guidance.",
                                )
                                continue
                            else:
                                raise ReplanFromParentException(
                                    f"Function '{func_name}' failed after {plan.MAX_LOCAL_RETRIES} retries.",
                                    reason=f"Final error:\n{last_error_reason}",
                                )

                        exit_seq = -1
                        if computer_primitives is not None and isinstance(
                            computer_primitives.browser.backend,
                            MagnitudeBrowserBackend,
                        ):
                            exit_seq = computer_primitives.browser.backend.current_seq

                        # Gather post-execution evidence from all active environments.
                        post_state: dict[str, Any] = {}
                        active_envs = (
                            plan.environments
                            if hasattr(plan, "environments")
                            and isinstance(getattr(plan, "environments", None), dict)
                            else plan.actor.environments
                        )
                        for env_namespace, env in active_envs.items():
                            try:
                                post_state[env_namespace] = await env.capture_state()
                            except Exception as e:
                                logger.warning(
                                    f"Failed to capture post-state from {env_namespace}: {e}",
                                )
                                post_state[env_namespace] = {
                                    "type": "error",
                                    "error": str(e),
                                }

                        if plan.runtime.cache_miss_counter:
                            step_cache_miss_counter = (
                                plan.runtime.cache_miss_counter.pop()
                            )

                        try:
                            frame = inspect.currentframe()
                            target_fn_obj = None
                            try:
                                if (
                                    frame
                                    and frame.f_back
                                    and func_name in frame.f_back.f_locals
                                ):
                                    target_fn_obj = frame.f_back.f_locals[func_name]
                                elif func_name in plan.execution_namespace:
                                    target_fn_obj = plan.execution_namespace[func_name]
                            finally:
                                del frame

                            captured_docstring = (
                                inspect.getdoc(target_fn_obj) if target_fn_obj else ""
                            )
                            captured_sig_str = (
                                str(inspect.signature(target_fn_obj))
                                if target_fn_obj
                                else "()"
                            )
                        except Exception:
                            captured_docstring = ""
                            captured_sig_str = "()"

                        captured_full_stack_tuple = (
                            plan.runtime.get_current_stack_tuple(plan.run_id)
                        )
                        captured_scoped_context_snapshot = (
                            self._get_scoped_context_from_plan_state(plan)
                        )

                        # Capture pane events since boundary (index-based slice).
                        captured_pane_events: list[Any] = []
                        pane_event_idx_after = pane_event_idx_before
                        try:
                            if getattr(plan, "pane", None) is not None:
                                pane_event_idx_after = len(plan.pane._events_log)
                                captured_pane_events = list(
                                    plan.pane._events_log[
                                        pane_event_idx_before:pane_event_idx_after
                                    ],
                                )
                        except Exception:
                            captured_pane_events = []
                            pane_event_idx_after = pane_event_idx_before

                        plan.verif_seq += 1
                        item = VerificationWorkItem(
                            ordinal=plan.verif_seq,
                            function_name=func_name,
                            parent_stack=(
                                captured_full_stack_tuple[:-1]
                                if captured_full_stack_tuple
                                else ()
                            ),
                            func_source=plan.function_source_map.get(func_name, ""),
                            docstring=captured_docstring,
                            func_sig_str=captured_sig_str,
                            pre_state=pre_state,
                            post_state=post_state,
                            interactions=copy.deepcopy(local_interactions),
                            return_value_repr=repr(result),
                            cache_miss_counter=step_cache_miss_counter,
                            exit_seq=exit_seq,
                            start_seq=start_seq,
                            full_call_stack_tuple=captured_full_stack_tuple,
                            scoped_context_snapshot=captured_scoped_context_snapshot,
                            pane_events=captured_pane_events,
                            pane_event_boundary=pane_event_idx_after,
                        )

                        plan._spawn_async_verification(item)

                        return result

                    finally:
                        if parent_sink is not None:
                            parent_sink.extend(local_interactions)

                        # These contextvars should normally be reset within the same context
                        # they were set in. In rare cases (e.g. GeneratorExit/unraisable cleanup),
                        # Python may finalize this coroutine in a different context.
                        try:
                            current_run_id_var.reset(run_id_token)
                        except ValueError as e:
                            logger.warning(
                                f"{diag_prefix} Failed to reset hp_run_id contextvar: {e}",
                            )
                        try:
                            current_interaction_sink_var.reset(sink_token)
                        except ValueError as e:
                            logger.warning(
                                f"{diag_prefix} Failed to reset hp_interaction_sink contextvar: {e}",
                            )
                        try:
                            current_invocation_id_var.reset(invoc_token)
                        except ValueError as e:
                            logger.warning(
                                f"{diag_prefix} Failed to reset hp_invocation_id contextvar: {e}",
                            )
                        plan.runtime.pop_frame(plan.run_id, frame_token)
                        if plan.call_stack and plan.call_stack[-1] == func_name:
                            plan.call_stack.pop()

                        plan.action_log.append(
                            f"[run_id={plan.run_id} invoc={invocation_id}] <- Exiting '{func_name}'",
                        )
                        plan.runtime.action_counter = parent_action_counter

                        if plan.runtime.cache_miss_counter:
                            plan.runtime.cache_miss_counter[
                                -1
                            ] += step_cache_miss_counter

            return wrapper

        return verify

    async def _inject_library_functions(self, base_code: str) -> tuple[str, set[str]]:
        """
        Injects necessary library function implementations using the
        dependency list stored by FunctionManager.

        Returns:
            A tuple of (injected_code, functions_to_skip_verify) where:
            - injected_code: The code with library functions injected
            - functions_to_skip_verify: Set of function names that should not have @verify decorator
        """
        if not self.function_manager:
            logger.debug("No FunctionManager available, skipping library injection.")
            return base_code, set()

        # Pre-fetch all functions in a single batch to avoid recursive backend calls.
        # TODO: Use semantic similarity based on base_code docstrings to fetch only
        # relevant functions instead of fetching all functions. This would be more
        # efficient for large function libraries and would automatically scope the
        # injection to semantically related skills.
        all_functions_map: Optional[Dict[str, Dict[str, Any]]] = None
        try:
            logger.info(
                "Pre-fetching all functions from FunctionManager for fast injection...",
            )
            all_functions_data = self.function_manager.search_functions(
                limit=1000,
            )  # TODO: use `search_functions_by_similarity`
            all_functions_map = {
                func_data["name"]: func_data for func_data in all_functions_data
            }
            logger.info(
                f"✅ Pre-fetched {len(all_functions_map)} functions in a single call.",
            )
        except Exception as e:
            logger.warning(
                f"Failed to pre-fetch all functions, falling back to individual queries: {e}",
            )
            all_functions_map = None

        final_code_parts: List[str] = []
        injected_functions: Set[str] = set()
        functions_to_skip_verify: Set[str] = set()
        functions_to_inject_queue: List[str] = []
        queued_functions: Set[str] = set()

        try:
            tree = ast.parse(base_code)
            initial_references: Set[str] = set()
            defined_function_names: Set[str] = set(
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    initial_references.add(node.func.id)
                elif isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        if isinstance(node.value, ast.Name):
                            initial_references.add(node.value.id)
                elif isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                    initial_references.add(node.value.id)

            common_non_lib_names = {
                "computer_primitives",
                "asyncio",
                "print",
                "verify",
                "super",
                "NotImplementedError",
                "ValueError",
                "Exception",
                "RuntimeError",
                "range",
                "len",
                "isinstance",
                "getattr",
                "setattr",
                "hasattr",
                "BaseModel",
                "Field",
                "HttpUrl",
                "Literal",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
                "Optional",
                "List",
                "Dict",
                "Any",
                "Union",
                "Type",
                "Callable",
            }
            initial_references.difference_update(common_non_lib_names)
            initial_references = {f for f in initial_references if "." not in f}

            for name in initial_references:
                if name in defined_function_names:
                    continue
                if name not in queued_functions:
                    functions_to_inject_queue.append(name)
                    queued_functions.add(name)

        except SyntaxError as e:
            logger.warning(f"Could not parse base plan for function injection: {e}")
            return base_code, set()
        except Exception as e:
            logger.error(
                f"Unexpected error during initial scan for injection: {e}",
                exc_info=True,
            )
            return base_code, set()

        processed_count = 0
        MAX_INJECTIONS = 200

        while functions_to_inject_queue and processed_count < MAX_INJECTIONS:
            processed_count += 1
            function_name = functions_to_inject_queue.pop(0)

            if function_name in injected_functions:
                continue

            # Use pre-fetched map for O(1) lookup instead of individual backend call
            library_func_data = None
            if all_functions_map is not None:
                library_func_data = all_functions_map.get(function_name)
            else:
                # Fallback to individual query if pre-fetch failed
                try:
                    search_results = self.function_manager.search_functions(
                        filter=f"name == '{function_name}'",
                        limit=1,
                    )
                    if search_results:
                        library_func_data = search_results[0]
                except Exception as e:
                    logger.error(
                        f"Error searching FunctionManager for '{function_name}': {e}",
                        exc_info=True,
                    )
                    continue

            if not library_func_data:
                continue

            # Skip venv functions - they're injected as callable proxies, not code
            if library_func_data.get("venv_id") is not None:
                logger.debug(
                    f"Skipping venv function '{function_name}' - injected as proxy.",
                )
                injected_functions.add(function_name)  # Mark as handled
                continue

            func_code = library_func_data.get("implementation")
            dependencies = library_func_data.get("calls", [])

            if not func_code:
                logger.warning(
                    f"No implementation found for '{function_name}' in FunctionManager.",
                )
                continue

            final_code_parts.insert(0, f"\n{func_code}\n")
            injected_functions.add(function_name)

            # Track functions that should skip verification
            verify_flag = library_func_data.get("verify", True)
            if not verify_flag:
                functions_to_skip_verify.add(function_name)
                logger.debug(
                    f"Function '{function_name}' has verify=False, will skip @verify decorator",
                )

            if isinstance(dependencies, list):
                for dep_name in dependencies:
                    if (
                        dep_name not in injected_functions
                        and dep_name not in queued_functions
                    ):
                        if dep_name not in common_non_lib_names and "." not in dep_name:
                            functions_to_inject_queue.append(dep_name)
                            queued_functions.add(dep_name)

        if processed_count >= MAX_INJECTIONS:
            logger.warning(
                f"Function injection stopped after {MAX_INJECTIONS} functions due to safety limit. Injected {len(injected_functions)} functions. Check dependencies if this seems too low.",
            )
        else:
            logger.info(
                f"Function injection complete. Injected {len(injected_functions)} functions total.",
            )

        final_code_parts.append(base_code)
        return "".join(final_code_parts), functions_to_skip_verify

    async def _generate_initial_plan(
        self,
        plan: HierarchicalActorHandle,
        goal: str,
    ) -> str:
        """
        Generates the initial Python script for the plan from a user goal.

        Args:
            plan: The HierarchicalActorHandle instance to generate the initial plan for.
            goal: The high-level user goal.

        Returns:
            A string containing the generated Python code for the plan.
        """
        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            try:
                if self.function_manager:
                    try:
                        relevant_functions = (
                            self.function_manager.search_functions_by_similarity(
                                query=goal,
                                n=20,
                                include_primitives=False,
                            )
                        )
                        existing_functions = {f["name"]: f for f in relevant_functions}
                    except Exception as e:
                        logger.warning(
                            f"Could not retrieve functions from FunctionManager: {e}",
                        )
                        existing_functions = {}
                else:
                    existing_functions = {}

                prompt = prompt_builders.build_initial_plan_prompt(
                    goal=goal,
                    tools=self.tools,
                    environments=self.environments,
                    existing_functions=existing_functions,
                    retry_msg=(
                        ""
                        if attempt == 0
                        else f"Last attempt failed: {last_error}. Please fix."
                    ),
                    images=plan.images,
                )
                response = await llm_call(
                    plan.plan_generation_client,
                    prompt,
                    images=plan.images,
                )
                base_code = (
                    response.strip().replace("```python", "").replace("```", "").strip()
                )
                logger.debug(
                    f"LLM response for initial plan (attempt {attempt+1}):\n\n--- LLM RAW RESPONSE START ---\n{response}\n--- LLM RAW RESPONSE END ---\n\n",
                )

                full_code, skip_verify = await self._inject_library_functions(base_code)
                plan.functions_skip_verify.update(skip_verify)

                return self._sanitize_code(full_code, plan)

            except SyntaxError as e:
                last_error = f"{e}\nProblematic Code:\n---\n{base_code}\n---"
                logger.error(
                    f"Attempt {attempt+1} to generate plan failed. Reason: {last_error}",
                )
                if attempt == max_retries - 1:
                    raise
        raise RuntimeError("Failed to generate a valid plan after multiple retries.")

    async def _dynamic_implement(
        self,
        plan: HierarchicalActorHandle,
        function_name: str,
        call_stack_snapshot: Optional[list[str]] = None,
        scoped_context_snapshot: Optional[dict] = None,
        **kwargs,
    ) -> ImplementationDecision:
        """
        Generates and returns an ImplementationDecision for a stub function.
        Includes a retry loop to handle LLM-generated syntax errors.
        """
        max_retries = 3
        last_syntax_error = ""

        for attempt in range(max_retries):
            replan_reason = kwargs.get(
                "replan_reason",
                "First-time implementation from NotImplementedError.",
            )
            if last_syntax_error:
                replan_reason += (
                    "\n\nCRITICAL: Your previous attempt to generate code failed with a "
                    f"SyntaxError. You MUST fix this error. Details:\n{last_syntax_error}"
                )

            browser_screenshot = None
            if (
                "computer_primitives" in (self.environments or {})
                and self.computer_primitives is not None
            ):
                try:
                    browser_screenshot = (
                        await self.computer_primitives.browser.get_screenshot()
                    )
                except Exception as e:
                    logger.warning(f"Could not get browser screenshot: {e}")

            call_stack_list_for_prompt = call_stack_snapshot or []
            scoped_context_str_for_prompt = self._format_scoped_context_for_prompt(
                scoped_context_snapshot or {},
            )

            failed_item: Optional[VerificationWorkItem] = kwargs.get("failed_item")
            if failed_item:
                func_sig = failed_item.func_sig_str
                docstring = failed_item.docstring or "No docstring available."
            else:
                try:
                    target_fn_obj = plan.execution_namespace.get(function_name)
                    docstring = (
                        inspect.getdoc(target_fn_obj) or "No docstring provided."
                    )
                    func_sig = (
                        str(inspect.signature(target_fn_obj)) if target_fn_obj else "()"
                    )
                except Exception:
                    docstring = "Docstring not available."
                    func_sig = "()"

            existing_functions = {}
            if self.function_manager:
                try:
                    query = f"{function_name} {docstring}"
                    relevant_functions = (
                        self.function_manager.search_functions_by_similarity(
                            query=query,
                            n=3,
                            include_primitives=False,
                        )
                    )
                    existing_functions = {f["name"]: f for f in relevant_functions}
                    if existing_functions:
                        plan.action_log.append(
                            f"Found {len(existing_functions)} relevant skills for implementing '{function_name}'.",
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve functions from FunctionManager for dynamic_implement: {e}",
                    )

            recent_transcript = None
            try:
                # TODO: Add this in case the plan needs the full transcript as context(https://app.clickup.com/t/86c4unzg9)
                # recent_transcript = await self.computer_primitives.transcript_manager.ask(
                #     "Provide a summary of the most recent conversational turns."
                # )
                pass
            except Exception as e:
                logger.warning(f"Could not fetch recent transcript: {e}")

            static_prompt, dynamic_prompt = (
                prompt_builders.build_dynamic_implement_prompt(
                    goal=plan.goal,
                    scoped_context=scoped_context_str_for_prompt,
                    call_stack=call_stack_list_for_prompt,
                    function_name=function_name,
                    function_sig=func_sig,
                    function_docstring=docstring,
                    clarification_question=kwargs.get("clarification_question"),
                    clarification_answer=kwargs.get("clarification_answer"),
                    replan_context=replan_reason,
                    has_browser_screenshot=browser_screenshot is not None,
                    tools=self.tools,
                    existing_functions=existing_functions,
                    environments=self.environments,
                    recent_transcript=recent_transcript,
                    parent_chat_context=plan.parent_chat_context,
                    images=plan.images,
                )
            )
            plan.implementation_client.set_response_format(ImplementationDecision)
            try:
                response_str = await llm_call(
                    plan.implementation_client,
                    dynamic_prompt,
                    static_prompt=static_prompt,
                    screenshot=browser_screenshot,
                    images=plan.images,
                )
                decision = ImplementationDecision.model_validate_json(response_str)
                logger.debug(
                    f"\n{format_pydantic_model(decision, title='IMPLEMENTATION DECISION', indent=2)}",
                )
                if decision.action == "implement_function":
                    if not decision.code:
                        raise ValueError(
                            "Action 'implement_function' requires the 'code' field.",
                        )

                    try:
                        clean_code = (
                            decision.code.strip()
                            .replace("```python", "")
                            .replace("```", "")
                            .strip()
                        )
                        full_code, skip_verify = await self._inject_library_functions(
                            clean_code,
                        )
                        plan.functions_skip_verify.update(skip_verify)
                        ast.parse(
                            textwrap.dedent(full_code),
                        )
                        decision.code = full_code
                        return decision
                    except SyntaxError as e:
                        last_syntax_error = f"Invalid Python code provided.\nError: {e}\nProblematic Code Snippet:\n---\n{decision.code}\n---"
                        logger.error(
                            f"Attempt {attempt + 1} failed: {last_syntax_error}",
                        )
                        if attempt == max_retries - 1:
                            raise e
                        continue

                return decision

            finally:
                plan.implementation_client.reset_response_format()

        raise RuntimeError(
            "Failed to generate a valid implementation after multiple retries.",
        )

    async def _check_state_against_goal(
        self,
        plan: HierarchicalActorHandle,
        function_name: str,
        function_docstring: str | None,
        function_source_code: str | None,
        interactions: list,
        evidence: dict[str, Any],
        function_return_value: Any = None,
        clarification_question: Optional[str] = None,
        clarification_answer: Optional[str] = None,
    ) -> VerificationAssessment:
        """
        Uses an LLM to assess if a function's execution achieved its goal.

        Args:
            plan: The active plan instance.
            function_name: The name of the function being verified.
            function_docstring: The docstring of the function.
            function_source_code: The source code of the function.
            interactions: A log of interactions that occurred.
            evidence: Dictionary of evidence from all active environments.
                Keys are environment namespaces (e.g., "computer_primitives", "primitives").
                Values are evidence dicts returned by env.capture_state().
            function_return_value: The return value of the function.
            clarification_question: An optional question that was previously asked.
            clarification_answer: An optional answer that was received.

        Returns:
            A VerificationAssessment object with the outcome.
        """
        recent_transcript = None
        try:
            # TODO: Add this in case the plan needs the full transcript as context(https://app.clickup.com/t/86c4unzg9)
            # recent_transcript = await self.computer_primitives.transcript_manager.ask(
            #     "Provide a summary of the most recent conversational turns."
            # )
            pass
        except Exception as e:
            logger.warning(f"Could not fetch recent transcript: {e}")

        context_dict = self._get_scoped_context_from_plan_state(plan)
        scoped_context_str = self._format_scoped_context_for_prompt(context_dict)

        # Extract screenshot for the LLM call if available (backward-compatible image input).
        screenshot = None
        browser_evidence = evidence.get("computer_primitives")
        if isinstance(browser_evidence, dict) and "screenshot" in browser_evidence:
            screenshot = browser_evidence.get("screenshot")

        static_prompt, dynamic_prompt = prompt_builders.build_verification_prompt(
            goal=plan.goal,
            function_name=function_name,
            function_docstring=function_docstring,
            scoped_context=scoped_context_str,
            interactions=interactions,
            evidence=evidence,
            function_return_value=function_return_value,
            recent_transcript=recent_transcript,
            parent_chat_context=plan.parent_chat_context,
            clarification_question=clarification_question,
            clarification_answer=clarification_answer,
            environments=self.environments,
        )

        plan.verification_client.set_response_format(VerificationAssessment)

        try:
            response_str = await llm_call(
                plan.verification_client,
                dynamic_prompt,
                static_prompt=static_prompt,
                screenshot=screenshot,
            )
            assessment = VerificationAssessment.model_validate_json(response_str)
            return assessment
        except Exception as e:
            logger.error(
                f"Failed to parse verification assessment: {e}. Raw response: {response_str if 'response_str' in locals() else 'N/A'}",
                exc_info=True,
            )
            return VerificationAssessment(
                status="fatal_error",
                reason=f"LLM provided malformed assessment: {str(e)}",
            )
        finally:
            plan.verification_client.reset_response_format()

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        plan: HierarchicalActorHandle = None
        for plan in self._plan_handles:
            await plan.stop()
        if self.computer_primitives is not None:
            self.computer_primitives.browser.stop()
        self._plan_handles.clear()
