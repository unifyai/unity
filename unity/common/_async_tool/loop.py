import asyncio
import unillm
import json
import inspect
import copy

from typing import (
    Dict,
    Union,
    Callable,
    Tuple,
    Any,
    Set,
    Optional,
    TYPE_CHECKING,
)
from contextlib import suppress
from pydantic import BaseModel

from ...logger import LOGGER
from ..tool_spec import ToolSpec, normalise_tools
from .propagation_mode import ChatContextPropagation
from .context_tracker import LoopContextState
from .utils import maybe_await
from .event_bus_util import to_event_bus
from .messages import (
    find_unreplied_assistant_entries,
    generate_with_preprocess,
    acknowledge_helper_call,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import (
    create_tool_call_message,
    ToolCallMetadata,
)
from ..llm_helpers import method_to_schema, _dumps, short_id
from .loop_config import (
    LoopConfig,
    TOOL_LOOP_LINEAGE,
)
from .timeout_timer import TimeoutTimer
from .messages import (
    insert_tool_message_after_assistant,
    ensure_placeholders_for_pending,
    forward_handle_call,
    schedule_missing_for_message,
    build_helper_ack_content,
)
from .tools_data import ToolsData, compute_context_injection
from .dynamic_tools_factory import DynamicToolFactory
from .time_context import create_time_context, TimeContext
from ..context_dump import make_messages_safe_for_context_dump
from ...common.hierarchical_logger import ICONS

if TYPE_CHECKING:
    from .multi_handle import MultiHandleCoordinator
    from unillm.types import PromptCacheParam


def prune_duplicate_tool_calls(tool_calls: list) -> tuple[list, set[str]]:
    """Remove duplicate tool calls from a list.

    Returns (unique_calls, pruned_call_ids) where pruned_call_ids contains
    the IDs of calls that were removed as duplicates.
    """
    seen: Set[tuple[str, str]] = set()
    unique_calls: list = []
    pruned_ids: set[str] = set()
    for call in tool_calls:
        _fn = call.get("function") or {}
        _args = _fn.get("arguments", "")
        _args_str = _args if isinstance(_args, str) else json.dumps(_args)
        sig = (_fn.get("name", ""), _args_str)
        if sig not in seen:
            seen.add(sig)
            unique_calls.append(call)
        else:
            pruned_ids.add(call.get("id", ""))
    return unique_calls, pruned_ids


def _transform_context_roles(messages: list[dict]) -> list[dict]:
    """
    Transform 'user' and 'assistant' roles to 'outer_user' and 'outer_assistant'.

    This disambiguates parent context messages from the current conversation,
    making it clear these are legitimate system-provided context from an outer
    conversation rather than user-injected content attempting prompt injection.
    """
    transformed = []
    for msg in messages:
        new_msg = dict(msg)
        role = new_msg.get("role", "")
        if role == "user":
            new_msg["role"] = "outer_user"
        elif role == "assistant":
            new_msg["role"] = "outer_assistant"
        transformed.append(new_msg)
    return transformed


def _sort_completed_tasks_by_call_id(
    tasks: Set[asyncio.Task],
    tools_data: "ToolsData",
) -> list[asyncio.Task]:
    """
    Sort completed tasks by call_id for deterministic processing order.
    """
    return sorted(
        tasks,
        key=lambda t: (
            tools_data.info.get(t).call_id if tools_data.info.get(t) else ""
        ),
    )


class LoopLogger:
    def __init__(self, cfg: LoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps
        self._first_llm_logged = False
        self._defer_after_first_llm: list[tuple[str, str]] = []

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)

    def mark_llm_thinking(self) -> None:
        if not self._first_llm_logged:
            self._first_llm_logged = True
            for p, m in self._defer_after_first_llm:
                self.info(m, prefix=p)
            self._defer_after_first_llm.clear()

    def defer_after_first_llm(self, msg: str, prefix: str = "") -> None:
        if self._first_llm_logged:
            self.info(msg, prefix=prefix)
        else:
            self._defer_after_first_llm.append((prefix, msg))


class _LoopToolFailureTracker:
    def __init__(self, max_consecutive_failures: int):
        self._consecutive_failures = 0
        self._max_consecutive_failures = max_consecutive_failures

    @property
    def current_failures(self):
        return self._consecutive_failures

    @property
    def max_failures(self):
        return self._max_consecutive_failures

    def has_exceeded_failures(self) -> bool:
        return self._consecutive_failures >= self._max_consecutive_failures

    def increment_failures(self):
        self._consecutive_failures += 1

    def reset_failures(self):
        self._consecutive_failures = 0


def _check_valid_response_format(response_format: Any):
    # Require a Pydantic model class – anything else is a configuration error.
    if not (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    ):
        raise TypeError(
            "response_format must be a Pydantic BaseModel subclass (e.g. MySchema).",
        )

    return response_format.model_json_schema()


async def async_tool_loop_inner(
    client: unillm.AsyncUnify,
    message: str | dict | list[str | dict],
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[dict | str],
    cancel_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: ChatContextPropagation = ChatContextPropagation.LLM_DECIDES,
    parent_chat_context: Optional[list[dict]] = None,
    caller_description: Optional[str] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Union[
            Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]],
            Callable[
                [int, Dict[str, Callable], list[str]],
                Tuple[str, Dict[str, Callable]],
            ],
        ]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
    multi_handle_coordinator: Optional["MultiHandleCoordinator"] = None,
    prompt_caching: Optional["PromptCacheParam"] = None,
    time_awareness: bool = False,
    extra_ask_tools: Optional[Dict[str, Callable]] = None,
) -> str:
    r"""
    Orchestrate an *interactive* "function-calling" dialogue between an LLM
    and a set of Python callables until the model yields a **final** plain-
    text answer.

    Key design points
    -----------------
    • **Concurrency** – every tool suggested by the model is wrapped in its
      own ``asyncio.Task`` so multiple long-running calls may advance in
      parallel; the loop always waits only for the *first* one to finish.

    • **Interruptibility** – the outer caller may:
        – set ``cancel_event`` → graceful shutdown (all tasks cancelled &
          awaited, then ``asyncio.CancelledError`` is re-raised);
        – queue ``interject_queue.put(text)`` → a new *user* turn injected
          just before the *next* LLM step without disturbing already running
          tools.

    • **Robustness** – exceptions inside tools are caught, serialised, and
      shown to the model; after ``max_consecutive_failures`` consecutive
      crashes the whole loop aborts with ``RuntimeError`` (prevents infinite
      failure ping-pong).

    • **Low coupling** – all transport (e.g. websockets, HTTP) can live
      outside; an optional ``event_bus`` lets a UI or logger subscribe to
      every message without the loop having to know who is listening.

    Parameters
    ----------
    client : ``unillm.AsyncUnify``
        Pre-initialised Unify client that provides ``append_messages`` and
        ``generate``.  All tokens sent to / received from the LLM flow
        through this object.

    message : ``str | dict | list[str | dict]``
        The very first user prompt that kicks-off the whole interactive
        session, or a batch of already-structured messages to seed the
        conversation before backfilling unresolved tool calls.

    tools : ``dict[str, Callable]``
        A mapping ``name → function`` describing every callable the LLM may
        invoke.  Each function must be fully type-hinted and have a concise
        docstring – these are automatically converted to a *tool schema*
        via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str | dict]``
        Thread-safe channel through which the *outer* application can push
        additional user turns at any time (e.g. the human changes their
        mind mid-generation). When a dict is provided it should follow the
        shape {"message": str, "_parent_chat_context_continued": list[dict]}.

    cancel_event : ``asyncio.Event``
        Flips to *set* when the outer caller wants graceful shutdown.  The
        loop then cancels every running task and propagates
        ``asyncio.CancelledError`` upstream.

    max_consecutive_failures : ``int``, default ``3``
        Hard safety valve: after this many back-to-back exceptions coming
        from tools the loop bails out with ``RuntimeError`` to avoid an
        infinite crash-and-retry ping-pong.

    ignore_tool_duplicates : ``bool``, default ``True``
        Deduplicates model-requested tool calls that have *identical*
        ``function.name`` **and** argument JSON.  Duplicates are pruned
        **in-place** before ever touching chat history or being scheduled.

    interrupt_llm_with_interjection : ``bool``, default ``True``
        Controls latency to fresh user input.  When *True* any in-flight
        ``client.generate`` is cancelled the moment a new user turn arrives
        so the assistant can pivot instantly.  When *False* the loop waits
        for the model to finish (legacy behaviour).

    propagate_chat_context : ``bool``, default ``True``
        If *True*, the entire conversation state of **this** loop is
        threaded into any child tool that accepts a
        ``parent_chat_context`` keyword argument.
        If *True*, the entire conversation state of **this** loop is threaded
        into any child tool via the *internal-only* ``parent_chat_context``
        argument.  This parameter is added automatically and is **not**
        exposed to the LLM.

     tool_policy : ``Callable | None``, default ``None``
         Optional callable that *dynamically* controls tool exposure **and**
         whether a tool call is **required** on a given turn.  Receives the
         current turn index (starting at ``0``) and the full mapping
         ``{name → callable}``.  It must return a tuple ``(policy, tools)``
         where ``policy`` is either ``"auto"`` or ``"required"`` (fed straight
         into ``tool_choice``) and ``tools`` is the possibly-filtered mapping
         of base tools visible on that turn.

    parent_chat_context : ``list[dict] | None``
        Nested chat structure passed from an **outer** loop.  When
        ``propagate_chat_context`` is enabled, this initial context is forwarded
        to inner tools on their first call, with subsequent calls receiving only
        incremental updates (new messages since the last call) to avoid token waste.

    log_steps : ``bool | str``, default ``True``
        Controls verbosity of step logging to ``LOGGER``:
          • ``False`` – no logging
          • ``True``  – log everything except system messages
          • ``"full"`` – log everything including system messages

    timeout : ``int | None``, default ``None``
        Activity-based timeout in seconds. The timer resets after each
        observable event (LLM response, tool completion, interjection).
        This timeout guards against hung user-defined tools, NOT slow LLM
        inference. LLM providers have their own timeout mechanisms; if an
        LLM call is in-flight, the loop will wait for it to complete before
        checking the timeout. When ``None``, no timeout is enforced.

    raise_on_limit : ``bool``, default ``False``
        If ``True``, raises ``asyncio.TimeoutError`` or ``RuntimeError``
        when the timeout or max_steps limit is exceeded. If ``False``,
        the loop terminates gracefully with a summary message.

    persist : ``bool``, default ``False``
        If ``True``, the loop does not terminate when the LLM produces content
        without tool calls. Instead, it blocks waiting for the next interjection
        via the ``interject_queue``. When an interjection arrives, the LLM is
        granted another turn. This enables a single persistent loop that can
        process multiple events over time, rather than terminating after each
        "final answer". The loop only terminates when explicitly stopped via
        ``cancel_event`` or ``stop_event``.

    time_awareness : ``bool``, default ``True``
        If ``True``, a time-context system message is injected at the start
        of the conversation and refreshed after each tool completion, giving
        the LLM awareness of wall-clock time and tool execution durations.
        If ``False``, the time-context table is omitted entirely and no
        tool-timing tracking is performed.

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """
    # Loop identity / lineage
    cfg = LoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    # Expose the resolved label (with 4-hex suffix) to the outer handle so steering logs
    # (stop/pause/resume/interject/ask) share the same label as the tool loop.
    with suppress(Exception):
        if outer_handle_container and outer_handle_container[0] is not None:
            setattr(outer_handle_container[0], "_log_label", cfg.label)
            # Also expose the resolved lineage list so event payloads can include the full
            # parent->child stack even when called outside the tool loop ContextVar scope.
            setattr(outer_handle_container[0], "_log_hierarchy", list(cfg.lineage))
    logger = LoopLogger(cfg, log_steps)

    # Wire inline log-file pointers: when UNILLM_LOG_DIR is set, each LLM call
    # writes a request+response file.  The callback prints the finalized path
    # in the terminal with the full parent lineage so the developer can click
    # through to the exact I/O for each step.
    if log_steps:
        client.set_on_log_file(
            lambda path: logger.info(f"→ {path}", prefix=ICONS["llm_log_file"]),
        )

    # ── Time context for time-awareness ──────────────────────────────────────
    # Capture the conversation start time and track tool execution timings.
    time_ctx: Optional[TimeContext] = create_time_context() if time_awareness else None
    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)

    # ── Reasoning model compatibility ────────────────────────────────────────────
    # Provider-specific thinking mode compliance is handled automatically by
    # unillm's provider preprocessing. The async tool loop is provider-agnostic.

    def _apply_reasoning_model_compat(gen_kwargs: dict, tool_choice: str) -> Callable:
        """Handle reasoning model compatibility. Returns effective preprocess."""
        # All provider-specific compliance is handled by unillm's preprocessing.
        return preprocess_msgs

    # normalise optional graceful stop event
    stop_event = stop_event or asyncio.Event()

    _initial_user_message = copy.deepcopy(message)

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # `set_response_format` still happens at the end of the loop.
    # NOTE: This hint is added as a new system message (not mutating the original)
    # and is appended later via _msg_dispatcher.append_msgs().
    _response_format_hint: str | None = None
    if response_format is not None:
        try:
            _schema = _check_valid_response_format(response_format)
            _response_format_hint = (
                "## Response Format\n"
                "NOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                + json.dumps(_schema, indent=2)
            )
        except Exception as _exc:  # noqa: BLE001
            logger.error(f"response_format hint failed: {_exc!r}")

    # ── User visibility guidance ──────────────────────────────────────────────
    # Explain to the model what the end-user can and cannot see. This guidance
    # is injected as a system message ONLY when the first interjection arrives,
    # not at the start of the loop. This keeps the LLM focused on the task at
    # hand until an interjection actually occurs.
    #
    # The guidance helps the model understand:
    # 1. The user does NOT see any intermediate tool calls or tool results
    # 2. The user only sees the initial request and any interjection messages
    # 3. The user sees the final plain-text response from the assistant
    #
    # Appended to the global system message via LiteLLM preprocessing.
    # -------------------------------------------------------------------------
    _user_visibility_guidance = (
        "## User Visibility Context\n"
        "IMPORTANT: The end-user who initiated this conversation can ONLY see:\n"
        "1. Their original request and any follow-up messages they send (interjections)\n"
        "2. Any notifications you emit (status updates, progress indicators, etc.)\n"
        "3. Any clarification requests you send asking for more information\n"
        "4. Your FINAL plain-text response at the end of this tool-use session\n\n"
        "The user CANNOT see:\n"
        "- Any intermediate tool calls you make\n"
        "- Any tool results or outputs\n"
        "- Any assistant messages that include tool_calls\n\n"
        "When the user sends follow-up messages (interjections) during your tool-use "
        "session, these appear as regular user messages. Consider and incorporate ALL "
        "user interjections in your final response. Later interjections should override "
        "earlier ones if there are any conflicting comments or requests."
    )
    _visibility_guidance_injected = False

    # ── runtime guards ────────────────────────────────────────────────────
    # rolling timeout ----------------------------------------------------
    timer: TimeoutTimer = TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)
    parent_chat_context_safe = make_messages_safe_for_context_dump(parent_chat_context)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context_safe:
                logger.info(
                    f"Parent Context: {json.dumps(parent_chat_context_safe, indent=4)}",
                    prefix=ICONS["tool_seeding"],
                )
            logger.info(
                f"System Message: {client.system_message}",
                prefix=ICONS["system_message"],
            )
        # Log user message (skip if seeding with a batch - per-item logs are emitted below)
        if not isinstance(message, list):
            logger.info(f"User Message: {message}", prefix=ICONS["user_message"])

    # ── 0-a. Inject **system** header with runtime context ─────────────────────
    #
    # Consolidate caller context and parent chat context into a single system
    # message at the start of the conversation. This explains:
    # 1. Who the "user" is (which manager is calling this loop)
    # 2. The broader conversation context (for nested loops)
    #
    # The special marker ``_runtime_context=True`` lets us identify this message
    # later. For backwards compatibility, ``_ctx_header=True`` is also set.
    # -------------------------------------------------------------------------

    # Derive caller description from lineage if not explicitly provided
    _effective_caller_description = caller_description
    if _effective_caller_description is None and lineage and len(lineage) >= 2:
        # The parent caller is the second-to-last entry in the lineage
        # (the last entry is this loop's own id)
        try:
            parent_label = lineage[-2]
            # Extract class name from "ClassName.method" or "ClassName.method(id)"
            parent_class = parent_label.split(".")[0].split("(")[0]
            # Strip common prefixes like "Simulated", "Base", "V3" etc.
            for prefix in ("Simulated", "Base"):
                if parent_class.startswith(prefix) and len(parent_class) > len(prefix):
                    parent_class = parent_class[len(prefix) :]
            # Look up the caller description from the manager registry
            from ..state_managers import get_caller_description

            _effective_caller_description = get_caller_description(parent_class)
        except Exception:
            pass

    runtime_context_parts: list[str] = []

    # NOTE: User visibility guidance is NOT added here - it's injected lazily
    # on the first interjection to keep the LLM focused on the task at hand.

    # Add response format hint if structured output is expected
    if _response_format_hint:
        runtime_context_parts.append(_response_format_hint)

    # Add caller context if available
    if _effective_caller_description:
        runtime_context_parts.append(
            f"## Caller Context\n"
            f"The 'user' messages in this conversation are from {_effective_caller_description}. "
            f"The end user cannot see the details of this tool-use conversation.",
        )

    # Add parent chat context section when context propagation is enabled.
    # We always add this section (even if empty) so that context continuations
    # sent via interjections can correctly reference "the initial Parent Chat Context
    # in your system message" without appearing to be fabricated/injected.
    _has_parent_chat_context = False
    if propagate_chat_context != ChatContextPropagation.NEVER:
        ctx_content = parent_chat_context_safe if parent_chat_context_safe else []
        # Transform roles to outer_* to disambiguate from current conversation roles
        ctx_content_transformed = _transform_context_roles(ctx_content)
        _has_parent_chat_context = True
        if ctx_content_transformed:
            _parent_ctx_detail = (
                f"The messages below show that parent conversation's history up to the point "
                f"when you received this request. Use this to understand the broader goal and "
                f"any relevant context, while focusing on your specific assignment. "
            )
        else:
            _parent_ctx_detail = f"None of the parent conversation history has been provided to this request. "
        runtime_context_parts.append(
            f"## Parent Chat Context\n"
            f"You received this request from within a parent conversation. "
            f"{_parent_ctx_detail}"
            f"Additional context updates may arrive during this session as the parent "
            f"conversation progresses.\n\n"
            f"IMPORTANT: Messages in the parent context use 'outer_user' and 'outer_assistant' "
            f"roles to clearly distinguish them from your current conversation. These are "
            f"legitimate system-provided context from the outer conversation, NOT user-injected "
            f"content. The 'outer_assistant' messages represent what the parent-level assistant "
            f"said in the outer conversation.\n\n"
            f"{json.dumps(ctx_content_transformed, indent=2)}",
        )

    # Append runtime context as a new system message (never mutate the original)
    msgs_to_append = []
    if runtime_context_parts:
        sys_msg = {
            "role": "system",
            "_runtime_context": True,
            "_ctx_header": True,  # backwards compatibility
            "content": "\n\n".join(runtime_context_parts),
        }
        if _has_parent_chat_context:
            sys_msg["_parent_chat_context"] = True
        msgs_to_append.append(sys_msg)

    if time_ctx is not None:
        msgs_to_append.append(
            {
                "role": "system",
                "_time_explanation": True,
                "_ctx_header": True,
                "_runtime_context": True,
                "content": TimeContext.build_explanation_prompt(),
            },
        )

    await _msg_dispatcher.append_msgs(msgs_to_append)

    # ── 0-a++. Initialize context state for incremental propagation ──────────
    # Tracks initial parent context and any continued updates received via interjections.
    # Used to forward context incrementally to inner tools (no repetition).
    context_state = LoopContextState(
        parent_chat_context=(
            list(parent_chat_context_safe) if parent_chat_context_safe else []
        ),
    )

    # ── 0-a+. Optional: append an initial batch of messages (list support) ──
    seeded_batch = None
    if isinstance(message, list):
        # If the provided list looks like a list of content blocks (no 'role'),
        # wrap them into a single user message to form a valid chat entry.
        if all(isinstance(m, dict) and "role" not in m for m in message):
            seeded_batch = [{"role": "user", "content": message}]
        else:
            # Otherwise treat as a pre-structured list of chat messages/strings.
            seeded_batch = [
                (m if isinstance(m, dict) else {"role": "user", "content": m})
                for m in message
            ]

        await _msg_dispatcher.append_msgs(seeded_batch)

    # ── initial prompt ───────────────────────────────────────────────────────
    # ── 0-b. Coerce tools → ToolSpec & helper lambdas ───────────────────────
    #
    # • «tools_data.normalized» holds the *canonical* mapping name → ToolSpec
    # • helper for the active-count of one tool (cheap O(#pending))
    # • helper that answers "may we launch / advertise *this* tool right now?"
    #   by comparing the live count with max_concurrent.
    # -----------------------------------------------------------------------

    # Initialise loop state early so preflight backfill can schedule tasks
    tools_data: ToolsData = ToolsData(
        tools,
        client=client,
        logger=logger,
        time_ctx=time_ctx,
        extra_ask_tools=extra_ask_tools,
    )

    consecutive_failures = _LoopToolFailureTracker(max_consecutive_failures)
    assistant_meta: Dict[int, Dict[str, Any]] = {}
    step_index: int = 0  # per assistant turn
    called_tools: list[str] = []  # tool names in invocation order across all turns

    # Pre-compute whether tool_policy accepts a third positional arg
    # (called_tools history) so we avoid per-turn introspection overhead.
    _policy_accepts_history = False
    if tool_policy is not None:
        with suppress(Exception):
            _sig = inspect.signature(tool_policy)
            _positional_kinds = (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            _n_positional = sum(
                1 for p in _sig.parameters.values() if p.kind in _positional_kinds
            )
            _policy_accepts_history = _n_positional >= 3
    # Expose live task_info mapping on the current Task so outer handles/tests
    # can introspect currently running nested handles (used by ask/stop helpers).
    with suppress(Exception):
        _self_task = asyncio.current_task()
        if _self_task is not None:
            setattr(_self_task, "task_info", tools_data.info)  # type: ignore[attr-defined]
            # Also expose the map of clarification channels so handle-level methods
            # can route answers programmatically without involving the LLM.
            setattr(
                _self_task,
                "clarification_channels",
                tools_data.clarification_channels,
            )
            # Expose ask_tools snapshot so handle.ask() can propagate to inner handles.
            setattr(_self_task, "get_ask_tools", tools_data.get_ask_tools)  # type: ignore[attr-defined]
            # Expose completed tool metadata (including handle refs) for downstream consumers.
            setattr(_self_task, "get_completed_tool_metadata", lambda: dict(tools_data._completed_askable_tools))  # type: ignore[attr-defined]

    # Preflight repair: backfill any pre-existing assistant tool_calls without replies
    with suppress(Exception):
        unreplied = find_unreplied_assistant_entries(client)
        if unreplied:
            # backfill for all such assistant messages (oldest → newest)
            for entry in unreplied:
                amsg = entry["assistant_msg"]
                # Before scheduling, drop any over-quota tool calls in this message
                tools_data.prune_over_quota_tool_calls(amsg)
                # De-duplicate tool calls if pruning is enabled
                if prune_tool_duplicates and amsg.get("tool_calls"):
                    unique, pruned = prune_duplicate_tool_calls(amsg["tool_calls"])
                    if pruned:
                        amsg["tool_calls"] = unique
                        entry["missing"] = [
                            cid for cid in entry["missing"] if cid not in pruned
                        ]
                missing_ids = set(entry["missing"])
                if not missing_ids:
                    continue
                await schedule_missing_for_message(
                    amsg,
                    missing_ids,
                    tools_data=tools_data,
                    context_state=context_state,
                    propagate_chat_context=propagate_chat_context,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )

    # ── helper: synthesize mirrored helper tool_calls (no LLM step) ───────────
    # Centralized steering: target selection + per-child dispatcher
    def _select_steering_targets(
        method: str,
        payload: dict | None,
    ) -> list[Tuple[asyncio.Task, "ToolCallMetadata"]]:
        """
        Choose which child tool calls should receive a steering signal.
        Policy:
          - clarify: target the specified call_id only (exact or suffix match)
          - pause/resume/stop: target ALL children
          - interject/ask/custom: not auto-forwarded to children
        """
        base = str(method or "").lower().strip()
        payload = payload or {}
        selected: list[Tuple[asyncio.Task, ToolCallMetadata]] = []
        # Clarify always targets a single child by id
        if base == "clarify":
            try:
                target_call_id = payload.get("call_id")
            except Exception:
                target_call_id = None
            if isinstance(target_call_id, str) and target_call_id:
                for t, inf in list(tools_data.info.items()):
                    try:
                        if str(inf.call_id) == target_call_id or str(
                            inf.call_id,
                        ).endswith(target_call_id):
                            selected.append((t, inf))
                            break
                    except Exception:
                        continue
            return selected
        # Control signals go to all children
        if base in ("pause", "resume", "stop"):
            for t, inf in list(tools_data.info.items()):
                try:
                    # Include even when no handle is adopted yet, so pause/resume can toggle pause_event
                    selected.append((t, inf))
                except Exception:
                    continue
            return selected
        # interject/ask/custom methods are not auto-forwarded to children
        return selected

    async def _dispatch_steering_to_child(
        method: str,
        payload: dict | None,
        inf: "ToolCallMetadata",
    ) -> None:
        """
        Execute a steering operation on a single child according to standard conventions:
          - interject: prefer the private interject_queue; else call handle.interject(...)
          - ask: call handle.ask(...)
          - pause/resume: call handle.pause()/resume() when available; else toggle pause_event
          - stop: call handle.stop(...)
          - clarify: put answer onto clarification down-queue (by call_id)
          - default: best-effort generic forward to the handle
        """
        base = str(method or "").lower().strip()
        args = dict(payload or {})
        h = getattr(inf, "handle", None)
        # interject
        if base == "interject":
            try:
                new_text = args.get("content") if isinstance(args, dict) else None
                if new_text is None and isinstance(args, dict):
                    new_text = args.get("message")
            except Exception:
                new_text = None
            iq = getattr(inf, "interject_queue", None)
            if iq is not None:
                await iq.put(new_text)
                return
            if h is not None:
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "interject",
                    args if isinstance(args, dict) else {},
                    fallback_positional_keys=["content", "message"],
                )
            return
        # ask
        if base == "ask":
            # Do not forward ask here. The outer ask() starts a dedicated inspection
            # loop and symbolically injects ask_* tool calls which adopt and run
            # nested ask handles. Forwarding here would duplicate those calls.
            return
        # pause
        if base == "pause":
            if h is not None and hasattr(h, "pause"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "pause",
                    args if isinstance(args, dict) else {},
                )
                return
            ev = getattr(inf, "pause_event", None)
            if ev is not None:
                ev.clear()
            return
        # resume
        if base == "resume":
            if h is not None and hasattr(h, "resume"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "resume",
                    args if isinstance(args, dict) else {},
                )
                return
            ev = getattr(inf, "pause_event", None)
            if ev is not None:
                ev.set()
            return
        # stop
        if base == "stop":
            if h is not None and hasattr(h, "stop"):
                await forward_handle_call(  # type: ignore[name-defined]
                    h,
                    "stop",
                    args if isinstance(args, dict) else {},
                    fallback_positional_keys=["reason"],
                )
            return
        # clarify
        if base == "clarify":
            with suppress(Exception):
                _cid = str(inf.call_id)
                _clar_map = tools_data.clarification_channels
                # Prefer exact id; fall back to suffix lookup
                if _cid in _clar_map:
                    down_q = _clar_map[_cid][1]
                else:
                    down_q = None
                    for k, (_u, _d) in list(_clar_map.items()):
                        if str(k).endswith(_cid[-6:]):
                            down_q = _d
                            break
                if down_q is not None:
                    await down_q.put((args or {}).get("answer"))
            return
        # default: best-effort generic forward
        if h is not None:
            # Remove control keys (custom steering metadata)
            try:
                args.pop("_custom", None)
                aliases = list(args.pop("_aliases", []) or [])
            except Exception:
                aliases = []
            try:
                fb_keys = tuple(args.pop("_fallback", ()) or ())
            except Exception:
                fb_keys = ()
            # Build method candidates: original, aliases, then base
            try:
                original_name = str(method or "")
            except Exception:
                original_name = base
            candidates: list[str] = []
            if original_name:
                candidates.append(original_name)
            for nm in aliases:
                if isinstance(nm, str) and nm:
                    candidates.append(nm)
            if base and base not in candidates:
                candidates.append(base)
            # Try each candidate method in order
            for nm in candidates:
                try:
                    attr = getattr(h, nm, None)
                    if not callable(attr):
                        continue
                    await forward_handle_call(  # type: ignore[name-defined]
                        h,
                        nm,
                        args if isinstance(args, dict) else {},
                        fallback_positional_keys=fb_keys,
                    )
                    return
                except Exception:
                    continue

    async def _synthesize_mirrored_helper_calls(
        method: str,
        payload: dict | None = None,
    ) -> None:
        """
        Create an assistant message containing helper tool_calls that mirror a steering
        command and immediately insert acknowledgement tool messages, then forward the
        steering to the target child handles. This does NOT call the LLM.
        """
        payload = payload or {}
        # NEW: allow "inject-only" mode so we do not double-execute child steering
        inject_only = False
        try:
            inject_only = bool(payload.get("_inject_only"))
        except Exception:
            inject_only = False

        # Generic: allow special banner deferral sentinels without tool acks
        base_name = ""
        try:
            base_name = str(method or "").lower().strip()
        except Exception:
            base_name = ""
        if base_name == "_banner_after_first_llm":
            text = ""
            prefix = ""
            try:
                text = str((payload or {}).get("text") or "")
                prefix = str((payload or {}).get("prefix") or "")
            except Exception:
                text, prefix = "", ""
            if text:
                try:
                    logger.defer_after_first_llm(text, prefix=prefix)
                except Exception:
                    pass
            return

        # Defer stop log (and optional banner) until after first LLM thinking
        if base_name == "stop":
            reason_txt = ""
            try:
                r = payload.get("reason")
                if isinstance(r, str) and r:
                    reason_txt = r
            except Exception:
                reason_txt = ""
            suffix = f" – reason: {reason_txt}" if reason_txt else ""
            try:
                logger.defer_after_first_llm(
                    f"Stop requested{suffix}",
                    prefix=ICONS["stop_requested"],
                )
            except Exception:
                pass
            # Optional generic banner payload to chain after stop (e.g., "Serialization complete")
            try:
                banner = payload.get("_after_first_llm_banner")
                if isinstance(banner, dict):
                    btxt = str(banner.get("text") or "")
                    bpf = str(banner.get("prefix") or "")
                    if btxt:
                        logger.defer_after_first_llm(btxt, prefix=bpf)
            except Exception:
                pass

        # Minimal transcript-only path: when a helper label is provided, synthesize
        # a single helper tool_call and acknowledgement, then return (no dispatch).
        try:
            helper_label = payload.get("helper_label")
        except Exception:
            helper_label = None
        if isinstance(helper_label, str) and helper_label:
            try:
                base = str(method or "").lower().strip()
            except Exception:
                base = ""
            if base:
                try:
                    call_id = f"mirror_{short_id(6)}"
                except Exception:
                    call_id = "mirror_unknown"
                # Minimal args for readability
                args_json: dict[str, Any] = {}
                try:
                    if base == "interject":
                        msg = payload.get("message") or payload.get("content")
                        if msg is not None:
                            args_json["content"] = msg
                    elif base == "stop" and "reason" in payload:
                        args_json["reason"] = payload.get("reason")
                except Exception:
                    pass
                helper_name = f"{base}_{helper_label}_{str(call_id)[-6:]}"
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": helper_name,
                                "arguments": json.dumps(args_json or {}),
                            },
                        },
                    ],
                }
                await _msg_dispatcher.append_msgs([assistant_msg])
                with suppress(Exception):
                    await to_event_bus(assistant_msg, cfg)
                assistant_meta[id(assistant_msg)] = {"results_count": 0}
                # Ack
                with suppress(Exception):
                    await acknowledge_helper_call(  # type: ignore[name-defined]
                        assistant_msg,
                        call_id,
                        helper_name,
                        json.dumps(args_json or {}),
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
            return

        # Select targets via central policy
        targets: list[Tuple[asyncio.Task, ToolCallMetadata]] = _select_steering_targets(
            method,
            payload if isinstance(payload, dict) else {},
        )
        if not targets:
            return

        # Build one assistant message with multiple tool_calls
        tool_calls = []
        args_by_id: dict[str, Any] = {}
        for _t, inf in targets:
            try:
                base = str(method or "").lower().strip()
                helper_name = f"{base}_{inf.name}_{str(inf.call_id)[-6:]}"
                # Build full forward kwargs for dispatch (strip control keys)
                try:
                    forward_args = dict(payload or {})
                except Exception:
                    forward_args = {}
                for _k in ("_custom", "_aliases", "_fallback"):
                    try:
                        forward_args.pop(_k, None)
                    except Exception:
                        pass

                # Minimal helper args for transcript readability
                args_json: dict[str, Any] = {}
                if base == "interject":
                    msg = payload.get("message") or payload.get("content")
                    if msg is not None:
                        args_json["content"] = msg
                elif base == "ask":
                    q = payload.get("question")
                    if q is not None:
                        args_json["question"] = q
                elif base == "stop":
                    if "reason" in payload:
                        args_json["reason"] = payload.get("reason")
                elif base == "clarify":
                    if "answer" in payload:
                        args_json["answer"] = payload.get("answer")
                # pause/resume carry no helper args
                call_id = f"mirror_{short_id(6)}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": helper_name,
                            "arguments": json.dumps(args_json or {}),
                        },
                    },
                )
                # Use full forward kwargs for dispatch
                args_by_id[call_id] = (helper_name, forward_args, inf)
            except Exception:
                continue
        if not tool_calls:
            return

        # Append assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        await _msg_dispatcher.append_msgs([assistant_msg])
        with suppress(Exception):
            await to_event_bus(assistant_msg, cfg)
        assistant_meta[id(assistant_msg)] = {"results_count": 0}

        # Insert ack tool messages and forward steering immediately to target handles
        for call in tool_calls:
            try:
                cid = call.get("id")
                if not isinstance(cid, str):
                    continue
                name, args, inf = args_by_id.get(cid, (None, None, None))
                if not isinstance(name, str):
                    continue
                # Ack message
                with suppress(Exception):
                    await acknowledge_helper_call(  # type: ignore[name-defined]
                        assistant_msg,
                        cid,
                        name,
                        call["function"].get("arguments", "{}"),
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                # Forward steering to child handle or channels
                # Centralized steering dispatch (unless inject-only)
                if (not inject_only) and (inf is not None):
                    base = str(method or "").lower().strip()
                    await _dispatch_steering_to_child(base, args, inf)
            except Exception:
                continue

    # ── initial **user** message (single-message path)
    if seeded_batch is None:
        if isinstance(message, dict):
            initial_user_msg = message
        else:
            initial_user_msg = {"role": "user", "content": message}
        if time_ctx is not None and isinstance(initial_user_msg.get("content"), str):
            initial_user_msg["content"] = time_ctx.prefix_user_message(
                initial_user_msg["content"],
            )
        await _msg_dispatcher.append_msgs([initial_user_msg])

    # ── helper: graceful early-exit when limits are hit ────────────────────
    async def _handle_limit_reached(reason: str) -> str:
        """
        Gracefully terminate the loop when *timeout* or *max_steps* are
        exceeded and `raise_on_limit` is *False*:
          • stop every pending tool (via handle.stop() if available)
          • cancel waiter coroutines
          • append a short assistant notice
        """
        for task in list(tools_data.pending):
            with suppress(Exception):
                inf = tools_data.info.get(task)
                if inf is not None and inf.handle is not None and hasattr(inf.handle, "stop"):  # type: ignore[attr-defined]
                    await maybe_await(inf.handle.stop())
            if not task.done():
                task.cancel()
        await asyncio.gather(*tools_data.pending, return_exceptions=True)
        tools_data.pending.clear()

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}",
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix=ICONS["early_exit"])
        return notice["content"]

    # ── small local helpers to dedupe repeated logic ─────────────────────────
    def _pretty(tool_name: str, payload: Any) -> str:
        return ToolsData._pretty_tool_payload(tool_name, payload)

    async def _handle_clarification(
        src_task: asyncio.Task,
        question_payload: Any,
    ) -> None:
        question_text = ""
        try:
            if isinstance(question_payload, dict):
                question_text = question_payload.get("question", "")
            else:
                question_text = str(question_payload)
        except Exception:
            question_text = str(question_payload)

        call_id = tools_data.info[src_task].call_id
        tool_name = tools_data.info[src_task].name

        # mark the task as waiting
        tools_data.info[src_task].waiting_for_clarification = True

        # ensure/refresh single placeholder for this call-id
        ph = tools_data.info[src_task].tool_reply_msg
        if ph is None:
            ph = create_tool_call_message(
                name=f"clarification_request_{call_id}",
                call_id=call_id,
                content="",
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                tools_data.info[src_task].assistant_msg,
                ph,
                client,
                _msg_dispatcher,
            )
            tools_data.info[src_task].tool_reply_msg = ph

        ph["name"] = f"clarification_request_{call_id}"
        ph["content"] = (
            "Tool incomplete, please answer the following to continue tool execution:\n"
            f"{question_text}"
        )

        # Log the clarification request as a first-class event
        try:
            logger.info(
                f"Clarification requested – {tool_name}: {question_text}",
                prefix=ICONS["clarification"],
            )
        except Exception:
            pass

        # Forward programmatic clarification event to outer handle
        with suppress(Exception):
            outer = outer_handle_container[0] if outer_handle_container else None
            if outer is not None and hasattr(outer, "_clar_q"):
                await outer._clar_q.put(
                    {
                        "type": "clarification",
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "question": question_text,
                    },
                )

    async def _handle_notification(src_task: asyncio.Task, payload: Any) -> None:
        call_id = tools_data.info[src_task].call_id
        tool_name = tools_data.info[src_task].name

        pretty = ToolsData._pretty_tool_payload(tool_name, payload)

        # Emit a concise human-friendly notification log line immediately
        try:
            if isinstance(payload, dict):
                _msg_txt = str(
                    payload.get("message") or payload.get("status") or payload,
                )
            else:
                _msg_txt = str(payload)
            logger.info(
                f"Notification from {tool_name}: {_msg_txt}",
                prefix=ICONS["notification"],
            )
        except Exception:
            pass

        placeholder = tools_data.info[src_task].tool_reply_msg
        if placeholder is None:
            placeholder = create_tool_call_message(
                name=tool_name,
                call_id=call_id,
                content=pretty,
            )
            await insert_tool_message_after_assistant(
                assistant_meta,
                tools_data.info[src_task].assistant_msg,
                placeholder,
                client,
                _msg_dispatcher,
            )
            tools_data.info[src_task].tool_reply_msg = placeholder
        else:
            placeholder["content"] = pretty

        # Forward programmatic notification event to the outer handle
        with suppress(Exception):
            outer = outer_handle_container[0] if outer_handle_container else None
            if outer is not None and hasattr(outer, "_notification_q"):
                event_payload = (
                    payload if isinstance(payload, dict) else {"message": str(payload)}
                )
                await outer._notification_q.put(
                    {
                        "type": "notification",
                        "call_id": call_id,
                        "tool_name": tool_name,
                        **event_payload,
                    },
                )

    # Set to *True* whenever the loop must grant the LLM an immediate turn
    # before waiting again (user interjection, clarification answer, etc.).
    llm_turn_required = False
    # When a patient interjection (trigger_immediate_llm_turn=False) arrives while
    # the LLM is already thinking, remember to grant exactly one extra LLM step
    # after the current step completes (unless another event already triggers a turn).
    deferred_llm_turn = False

    # Loop returns immediately upon the final assistant message (no persist mode)

    try:
        while True:
            # ── 0-Ø. Main loop tick start ─────────────────────────────────────

            # ── 0-α-P. Global *pause* gate  ────────────────────────────
            # Keep handling tool completions & cancellation, but *never*
            # let the LLM speak while we're paused.
            if not pause_event.is_set():
                # While paused, process any MIRROR steering sentinels immediately so control
                # signals (pause/resume/stop/etc.) still reach child handles without waiting.
                try:
                    while True:
                        try:
                            _extra = interject_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if isinstance(_extra, dict) and "_mirror" in _extra:
                            _ms = _extra.get("_mirror") or {}
                            _m = _ms.get("method")
                            _kw = _ms.get("kwargs") or {}
                            if isinstance(_m, str) and _m:
                                # Merge control keys into payload for routing/dispatch
                                try:
                                    merged = dict(_kw if isinstance(_kw, dict) else {})
                                except Exception:
                                    merged = {}
                                try:
                                    if _ms.get("_custom"):
                                        merged["_custom"] = True
                                except Exception:
                                    pass
                                try:
                                    if "_aliases" in _ms:
                                        merged["_aliases"] = list(
                                            _ms.get("_aliases") or [],
                                        )
                                except Exception:
                                    pass
                                try:
                                    if "_fallback" in _ms:
                                        merged["_fallback"] = list(
                                            _ms.get("_fallback") or [],
                                        )
                                except Exception:
                                    pass
                                await _synthesize_mirrored_helper_calls(_m, merged)
                            continue
                        else:
                            # Re-queue non-mirror entries for later processing once resumed
                            await interject_queue.put(_extra)
                            break
                except Exception:
                    pass
                # While paused, proactively schedule any unreplied assistant tool_calls
                # so base tools start in paused state and placeholders appear.
                with suppress(Exception):
                    if True:
                        if unreplied := find_unreplied_assistant_entries(client):
                            last_problem = unreplied[-1]
                            amsg = last_problem["assistant_msg"]
                            missing_ids = set(last_problem["missing"])
                            if id(amsg) not in assistant_meta:
                                await schedule_missing_for_message(
                                    amsg,
                                    missing_ids,
                                    tools_data=tools_data,
                                    context_state=context_state,
                                    propagate_chat_context=propagate_chat_context,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    initial_paused=True,
                                )
                                # Ensure placeholders exist immediately
                                await ensure_placeholders_for_pending(
                                    tools_data=tools_data,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    time_ctx=time_ctx,
                                )
                # Give any pending tool tasks a chance to finish OR wait until the
                # loop is resumed / cancelled.  Every coroutine is wrapped in an
                # asyncio.Task so `asyncio.wait()` is happy.
                if tools_data.pending:
                    pause_waiter = asyncio.create_task(
                        pause_event.wait(),
                        name="PauseEventWait",
                    )
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="CancelEventWait",
                    )
                    waiters = tools_data.pending | {
                        pause_waiter,
                        cancel_waiter,
                    }

                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # helper-task cleanup so they don't dangle
                    for w in (pause_waiter, cancel_waiter):
                        if w not in done and not w.done():
                            w.cancel()
                            await asyncio.gather(w, return_exceptions=True)

                    # tool finished?
                    for t in _sort_completed_tasks_by_call_id(
                        done & tools_data.pending,
                        tools_data,
                    ):
                        await tools_data.process_completed_task(
                            task=t,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )
                    if cancel_event.is_set():
                        # Cancellation requested – rely on mirrored stop to have
                        # already reached children; abort loop gracefully.
                        raise asyncio.CancelledError
                    # No graceful stop path
                    continue  # remain paused: do not allow the LLM to speak while paused
                else:
                    # nothing running – just idle until resumed or cancelled
                    # Before idling, schedule any missing tool replies from last assistant turn
                    with suppress(Exception):
                        if unreplied := find_unreplied_assistant_entries(client):
                            last_problem = unreplied[-1]
                            amsg = last_problem["assistant_msg"]
                            missing_ids = set(last_problem["missing"])
                            if id(amsg) not in assistant_meta:
                                await schedule_missing_for_message(
                                    amsg,
                                    missing_ids,
                                    tools_data=tools_data,
                                    context_state=context_state,
                                    propagate_chat_context=propagate_chat_context,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    initial_paused=True,
                                )
                                await ensure_placeholders_for_pending(
                                    tools_data=tools_data,
                                    assistant_meta=assistant_meta,
                                    client=client,
                                    msg_dispatcher=_msg_dispatcher,
                                    time_ctx=time_ctx,
                                )
                    done, _ = await asyncio.wait(
                        {
                            asyncio.create_task(
                                pause_event.wait(),
                                name="PauseEventWait",
                            ),
                            asyncio.create_task(
                                cancel_event.wait(),
                                name="CancelEventWait",
                            ),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # resumed?
                    if pause_event.is_set():
                        continue  # back to main loop, un-paused

                    # cancelled?
                    if cancel_event.is_set():
                        # Cancellation requested – rely on mirrored stop to have
                        # already reached children; abort loop gracefully.
                        raise asyncio.CancelledError
                    # remain paused
                    continue  # top-of-loop, still paused

            # 0-α. **Global timeout**
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # 0-β. **Chat history length**
            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            # 0-γ. Repair any outstanding assistant tool_calls missing replies
            #      before we allow new user interjections to be appended.
            with suppress(Exception):
                # Only consider the very latest assistant with missing replies first
                if unreplied := find_unreplied_assistant_entries(client):
                    last_problem = unreplied[-1]
                    amsg = last_problem["assistant_msg"]
                    missing_ids = set(last_problem["missing"])
                    # Skip if we already scheduled for this assistant turn
                    if id(amsg) not in assistant_meta:
                        await schedule_missing_for_message(
                            amsg,
                            missing_ids,
                            tools_data=tools_data,
                            context_state=context_state,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                        )

            # ── 0. Drain *all* queued interjections, allowed at any time ──
            # NOTE: We must do this *before* waiting on tool completion so a
            # fast typist can still sneak in a question while long-running
            # tools are in flight.  Doing it here keeps latency <1π loop.
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # NEW: Optional policy override for LLM turn scheduling
                llm_policy = "immediate"
                try:
                    if isinstance(extra, dict):
                        llm_policy = str(extra.get("_llm_turn") or "immediate")
                except Exception:
                    llm_policy = "immediate"
                if llm_policy == "none":
                    # Do not schedule an LLM turn
                    pass
                elif llm_policy == "deferred":
                    try:
                        deferred_llm_turn = True
                    except Exception:
                        pass
                else:
                    # Default immediate: schedule a turn and clear any prior deferral
                    llm_turn_required = True
                    try:
                        deferred_llm_turn = False
                    except Exception:
                        pass
                # Mirrored steering sentinel: synthesize helper tool_calls immediately
                try:
                    if isinstance(extra, dict) and "_mirror" in extra:
                        _ms = extra.get("_mirror") or {}
                        _m = _ms.get("method")
                        _kw = _ms.get("kwargs") or {}
                        if isinstance(_m, str) and _m:
                            try:
                                merged = dict(_kw if isinstance(_kw, dict) else {})
                            except Exception:
                                merged = {}
                            try:
                                if _ms.get("_custom"):
                                    merged["_custom"] = True
                            except Exception:
                                pass
                            try:
                                if "_aliases" in _ms:
                                    merged["_aliases"] = list(_ms.get("_aliases") or [])
                            except Exception:
                                pass
                            try:
                                if "_fallback" in _ms:
                                    merged["_fallback"] = list(
                                        _ms.get("_fallback") or [],
                                    )
                            except Exception:
                                pass
                            await _synthesize_mirrored_helper_calls(_m, merged)
                            continue
                except Exception:
                    pass
                # Special sentinel: request immediate LLM turn without creating a new system message
                try:
                    if isinstance(extra, dict) and extra.get("_replay"):
                        # Do not append any message; just grant the next LLM turn
                        # and proceed. This preserves transcript fidelity after resume.
                        llm_turn_required = True
                        continue
                except Exception:
                    pass
                # Build system message based on the user-visible history stored on the outer handle.
                history_lines: list[str] = []
                try:
                    outer_handle = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    uvh = (
                        getattr(outer_handle, "_user_visible_history", [])
                        if outer_handle
                        else []
                    )
                    for _m in uvh:
                        role = _m.get("role")
                        _content = _m.get("content")
                        if isinstance(_content, dict):
                            _text = str(_content.get("message", "")).strip()
                        else:
                            _text = str(_content or "").strip()
                        if role in ("user", "assistant") and _text:
                            history_lines.append(f"{role}: {_text}")
                except Exception:
                    # Fallback to just the original user prompt if available
                    try:
                        first_user = next(
                            (
                                m.get("content", "")
                                for m in client.messages
                                if m.get("role") == "user"
                            ),
                            "",
                        )
                        if first_user:
                            history_lines = [f"user: {first_user}"]
                    except Exception:
                        history_lines = []

                # Support dict-style interjections carrying continued parent context.
                # Interjections are sent as user messages (not system messages) for
                # broad provider compatibility. User-visibility context is in the
                # topmost system message.
                if isinstance(extra, dict):
                    _msg_text = str(extra.get("message", "")).strip()
                    _ctx_cont = extra.get(
                        "_parent_chat_context_continued",
                    ) or extra.get(
                        "_parent_chat_context_continuted",  # legacy typo support
                    )
                else:
                    _msg_text = str(extra)
                    _ctx_cont = None

                # Log a single concise interjection line
                try:
                    logger.info(
                        f"Interjection received: {_msg_text}",
                        prefix=ICONS["interjection"],
                    )
                except Exception:
                    pass

                # Record continued context in our state for incremental propagation
                if _ctx_cont:
                    _ctx_cont = make_messages_safe_for_context_dump(_ctx_cont)
                    context_state.receive_context_continuation(_ctx_cont)
                    # Forward to active inner tool handles that opted into context
                    # Tools that set include_parent_chat_context=False initially
                    # should not receive context continuations either.
                    for task, info in tools_data.info.items():
                        if info.interject_queue is not None and info.context_opted_in:
                            with suppress(Exception):
                                # Forward the continued context to the inner handle
                                info.interject_queue.put_nowait(
                                    {
                                        "message": "",  # Empty message, just context update
                                        "_parent_chat_context_continued": _ctx_cont,
                                        "_context_only": True,  # Flag to indicate context-only update
                                    },
                                )
                                context_state.mark_cont_forwarded_to_tool(info.call_id)

                # On the FIRST interjection, inject user visibility guidance as a
                # system message so the model understands why a user message is
                # appearing mid-tool-execution and what the user can/cannot see.
                if not _visibility_guidance_injected:
                    await _msg_dispatcher.append_msgs(
                        [
                            {
                                "role": "system",
                                "_visibility_guidance": True,
                                "content": _user_visibility_guidance,
                            },
                        ],
                    )
                    _visibility_guidance_injected = True

                # Send interjection as user message(s).
                # If context continuation is present, inject it as a separate user message
                # tagged with _ctx_header so the current LLM sees it but it's filtered out
                # when building cur_msgs for inner tool forwarding.
                msgs_to_append: list[dict] = []
                if _ctx_cont:
                    # Transform roles to outer_* to disambiguate from current conversation
                    ctx_cont_transformed = _transform_context_roles(_ctx_cont)
                    ctx_cont_content = (
                        "## Parent Chat Context (continued)\n"
                        "This is the next incremental chunk of the parent conversation since the "
                        "last context update (either the initial Parent Chat Context in your system "
                        "message, or the previous continued context chunk). These messages arrived "
                        "while you have been working on this request and may be relevant. Use this "
                        "to stay informed of any updates or new information from the parent conversation. "
                        "As explained in the system message, 'outer_user' and 'outer_assistant' roles "
                        "indicate messages from the parent conversation.\n\n"
                        f"{json.dumps(ctx_cont_transformed, indent=2)}"
                    )
                    msgs_to_append.append(
                        {
                            "role": "user",
                            "_ctx_header": True,
                            "content": ctx_cont_content,
                        },
                    )
                # Only append user message if there's actual content
                if _msg_text:
                    _user_content = (
                        time_ctx.prefix_user_message(_msg_text)
                        if time_ctx is not None
                        else _msg_text
                    )
                    msgs_to_append.append({"role": "user", "content": _user_content})
                if msgs_to_append:
                    await _msg_dispatcher.append_msgs(msgs_to_append)
                # Update history only if there was user message content
                if _msg_text:
                    last_valid_user_history = history_lines + [f"user: {_msg_text}"]

                # Append this interjection to the user-visible history for future context
                with suppress(Exception):
                    if outer_handle:
                        outer_handle._user_visible_history.append(
                            {
                                "role": "user",
                                "content": (
                                    {
                                        "message": _msg_text,
                                        "_parent_chat_context_continued": _ctx_cont,
                                    }
                                    if isinstance(extra, dict) and _ctx_cont
                                    else _msg_text
                                ),
                            },
                        )

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # If a child just asked for clarification we also want to give
            # the LLM a chance to react immediately.
            # Skip this whole block if the model already needs to speak.
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if tools_data.pending and not llm_turn_required:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                notif_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in tools_data.pending:
                    # Only listen for *new* clarification questions.
                    # If the task is already awaiting an answer,
                    # `waiting_for_clarification` will be True.
                    info = tools_data.info[_t]
                    if info.waiting_for_clarification:
                        continue

                    # Always listen for clarification requests when a queue is provided
                    if info.clar_up_queue is not None:
                        w = asyncio.create_task(
                            info.clar_up_queue.get(),
                            name="ClarificationQueueGet",
                        )
                        clar_waiters[w] = _t

                    # Always listen for notifications when a queue is provided
                    if info.notification_queue is not None:
                        pw = asyncio.create_task(
                            info.notification_queue.get(),
                            name="NotificationQueueGet",
                        )
                        notif_waiters[pw] = _t
                waiters = (
                    tools_data.pending
                    | set(clar_waiters)
                    | set(notif_waiters)
                    | {cancel_waiter, interject_w}
                )

                # ── honour global *timeout* while we wait for tools ───────────
                if timer.has_exceeded_time():
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=timer.remaining_time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ── hit the timeout while waiting? ────────────────────────────
                if not done:
                    # nothing completed → the wait *timed out*
                    if raise_on_limit:
                        raise asyncio.TimeoutError(
                            f"Loop exceeded {timeout}s wall-clock limit",
                        )
                    else:
                        return await _handle_limit_reached(
                            f"timeout ({timeout}s) exceeded",
                        )

                # ── ensure *unused* auxiliary waiters don't linger ──────────
                # If one helper won the race we *must* cancel/await the other
                # so that it cannot consume the next interjection invisibly.
                for aux in (
                    interject_w,
                    cancel_waiter,
                    *clar_waiters.keys(),
                    *notif_waiters.keys(),
                ):
                    if aux not in done and not aux.done():
                        aux.cancel()
                        await asyncio.gather(aux, return_exceptions=True)

                if interject_w in done:
                    # re-queue so branch 0 will handle user turn immediately
                    await interject_queue.put(interject_w.result())
                    continue  # → loop, will be processed in 0.

                if cancel_waiter in done:
                    # Cancellation wins; mirrored stop is the only propagation path.
                    raise asyncio.CancelledError  # cancellation wins
                # No graceful stop path

                # ── clarification request bubbled up from a child tool ──────────────
                if done & clar_waiters.keys():
                    for cw in done & clar_waiters.keys():
                        await _handle_clarification(clar_waiters[cw], cw.result())

                    # let the assistant answer immediately
                    # Process any notifications that arrived in the same tick
                    if done & notif_waiters.keys():
                        for pw in done & notif_waiters.keys():
                            await _handle_notification(notif_waiters[pw], pw.result())

                    llm_turn_required = True
                    continue

                # ── progress update bubbled up from a child tool (non-blocking) ─────
                if done & notif_waiters.keys():
                    for pw in done & notif_waiters.keys():
                        await _handle_notification(notif_waiters[pw], pw.result())
                    # Require an immediate LLM turn (same behaviour as clarification)
                    llm_turn_required = True

                needs_turn = False
                # Only process completion for actual tool tasks; exclude helper waiters
                for task in _sort_completed_tasks_by_call_id(
                    done & tools_data.pending,
                    tools_data,
                ):
                    if await tools_data.process_completed_task(
                        task=task,
                        consecutive_failures=consecutive_failures,
                        outer_handle_container=outer_handle_container,
                        assistant_meta=assistant_meta,
                        msg_dispatcher=_msg_dispatcher,
                    ):
                        needs_turn = True

                # Other tools may still be running.
                if needs_turn:
                    llm_turn_required = True
                if tools_data.pending:
                    continue  # jump to top-of-loop

            # ── B: wait for remaining tools before asking the LLM again,
            # unless the model already deserves a turn
            if tools_data.pending and not llm_turn_required:
                # Ensure placeholders exist for any pending calls before the next assistant turn
                await ensure_placeholders_for_pending(
                    tools_data=tools_data,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                    time_ctx=time_ctx,
                )
                continue  # still waiting for other tool tasks

            # ── Continue scheduling / planning ────────────────────────────────

            # ── C.  Add temporary tools so the LLM can **continue** or **cancel**
            #       any still‑running tool calls ────────────────────────────────
            #
            # For each pending ``asyncio.Task`` we synthesise two VERY small helper
            # tools and expose them to the model on the *next* LLM step.  Each
            # helper's docstring is a single line that embeds **both** the name of
            # the original function **and** the concrete arguments it was invoked
            # with – this gives the agent just enough context without overwhelming
            # the token budget.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # 1.  Build the *static* part of the toolkit **fresh on every turn**
            #     so that concurrency changes (tasks finishing, stopping, …)
            #     are immediately reflected in what the LLM can see.
            # ------------------------------------------------------------------

            # 0.  Decide policy & tool-subset for this turn  ───────────────
            if tool_policy is not None:
                _tools_snapshot = {n: s.fn for n, s in tools_data.normalized.items()}
                try:
                    if _policy_accepts_history:
                        tool_choice_mode, filtered = tool_policy(
                            step_index,
                            _tools_snapshot,
                            list(called_tools),
                        )
                    else:
                        tool_choice_mode, filtered = tool_policy(
                            step_index,
                            _tools_snapshot,
                        )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", _tools_snapshot
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            # Force tool usage when a response_format is required so the model
            # must submit the final JSON via the response-submission tool.
            # This preserves flexible tool use while guaranteeing typed completion.
            if response_format is not None and tool_choice_mode != "required":
                tool_choice_mode = "required"

            # When tools are in-flight, force tool_choice=required so the LLM
            # must call a real tool (check_status_*, cancel_*, etc.) rather
            # than ending the loop.  The response tool is masked in this
            # situation (see below), so the only options are real tools.
            _has_pending_tools = bool(tools_data.pending)
            if _has_pending_tools and tool_choice_mode != "required":
                tool_choice_mode = "required"

            visible_base_tools_schema = [
                method_to_schema(
                    spec.fn,
                    name,
                    expose_context_control=(
                        propagate_chat_context == ChatContextPropagation.LLM_DECIDES
                    ),
                    has_parent_context=bool(parent_chat_context),
                )
                for name, spec in policy_tools_norm.items()
                if tools_data.concurrency_ok(name) and tools_data.quota_ok(name)
            ]

            # Inject the response-submission tool when response_format is set
            # AND no other tools are in-flight.  This tool is semantically
            # "end the current turn" (the tool-call analogue of a bare text
            # response).  When tools are still running we intentionally mask
            # it so the LLM must interact with them (via check_status_*,
            # cancel_*, etc.) rather than silently killing them.
            #
            # Name varies by mode:
            #   persist=True  → "send_response"  (signals turn completion,
            #                    loop continues waiting for next interjection)
            #   persist=False → "final_response"  (terminates the loop)
            _response_tool_name = "send_response" if persist else "final_response"

            if response_format is not None and not _has_pending_tools:
                if persist:
                    _response_tool_desc = (
                        "Submit your structured response for the current "
                        "request in the required JSON format. This signals "
                        "that you have completed the current work and are "
                        "ready for the next instruction. Do not use this "
                        "for progress updates — those should be sent via "
                        "notifications while work is still ongoing."
                    )
                else:
                    _response_tool_desc = (
                        "Submit your final response in the required JSON "
                        "format. The response can be a complete result, a "
                        "partial result, or a message indicating you cannot "
                        "proceed (e.g., 'I cannot help with that.'). "
                        "Calling this tool terminates the conversation."
                    )
                try:
                    _answer_schema = _check_valid_response_format(response_format)

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": True,
                            "function": {
                                "name": _response_tool_name,
                                "description": _response_tool_desc,
                                "parameters": {
                                    "type": "object",
                                    "properties": {"answer": _answer_schema},
                                    "required": ["answer"],
                                },
                            },
                        },
                    )
                except Exception as _injection_exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to inject {_response_tool_name} tool: {_injection_exc!r}",
                    )

            # Inject multi-handle `final_response` tool when coordinator is present.
            # This tool requires request_id to specify which request is being answered.
            # Unlike response_format mode, this is always available (tools may be shared).
            if multi_handle_coordinator is not None:
                visible_base_tools_schema.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "final_response",
                            "description": (
                                "Submit the final response for a specific request. "
                                "Use this to complete a request when you have the result. "
                                "Each request must be answered exactly once."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "request_id": {
                                        "type": "integer",
                                        "description": "The ID of the request being answered (from [Request N] tag).",
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "The final answer text for this request.",
                                    },
                                },
                                "required": ["request_id", "answer"],
                            },
                        },
                    },
                )
                # Also inject `ask_user_clarification` for routing clarifications to specific requests
                visible_base_tools_schema.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_user_clarification",
                            "description": (
                                "Ask a specific user for clarification. Use this when you need "
                                "more information from the user who submitted a particular request."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "request_id": {
                                        "type": "integer",
                                        "description": "The ID of the request whose user should receive the question.",
                                    },
                                    "question": {
                                        "type": "string",
                                        "description": "The clarification question to ask the user.",
                                    },
                                },
                                "required": ["request_id", "question"],
                            },
                        },
                    },
                )

            # Yield to allow just-scheduled tool tasks to complete (especially
            # those that immediately return a SteerableToolHandle). This ensures
            # dynamic helpers are generated with the handle's docstrings.
            await asyncio.sleep(0)

            # Process any tools that completed during the yield
            for task in list(tools_data.pending):
                if task.done():
                    with suppress(Exception):
                        await tools_data.process_completed_task(
                            task=task,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )

            dynamic_tool_factory = DynamicToolFactory(tools_data)
            dynamic_tool_factory.generate()
            dynamic_tools = dynamic_tool_factory.dynamic_tools
            # Keep ToolsData's reference to dynamic_tools up-to-date so
            # get_ask_tools() always reflects the latest set of helpers.
            tools_data._dynamic_tools_ref = dynamic_tools

            # Register callback to refresh helpers when a handle is adopted mid-loop
            def _refresh_helpers_for_task(task: asyncio.Task) -> None:
                with suppress(Exception):
                    dynamic_tool_factory._process_task(task)
                    dynamic_tools.update(dynamic_tool_factory.dynamic_tools)

            tools_data._on_handle_adopted = _refresh_helpers_for_task

            # If any task is currently waiting for clarification, hide the
            # global `wait` helper to ensure the model proceeds to request
            # clarification rather than idling. This avoids deadlocks where
            # no interjection arrives and a tool is blocked awaiting input.
            try:
                if any(
                    getattr(_inf, "waiting_for_clarification", False)
                    for _inf in tools_data.info.values()
                ):
                    dynamic_tools.pop("wait", None)
            except Exception:
                pass

            # make sure every pending call already has a *tool* reply ──
            #  (a placeholder) before we let the assistant speak again.
            await ensure_placeholders_for_pending(
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
                time_ctx=time_ctx,
            )

            # Merge helpers into the visible toolkit for the upcoming LLM step
            # For steering methods (ask/interject) on tools that opted into context,
            # expose include_parent_chat_context_cont in LLM_DECIDES mode
            _expose_ctx_cont_control = (
                propagate_chat_context == ChatContextPropagation.LLM_DECIDES
            )
            tmp_tools = visible_base_tools_schema + [
                method_to_schema(
                    fn,
                    include_class_name=include_class_in_dynamic_tool_names,
                    # Expose include_parent_chat_context for dynamic tools that accept
                    # _parent_chat_context (currently only ask_* tools). This lets the
                    # LLM opt out of context propagation for inspection loops.
                    expose_context_control=_expose_ctx_cont_control,
                    has_parent_context=bool(parent_chat_context),
                    # Expose context continuation control for steering methods when:
                    # 1. Propagation mode is LLM_DECIDES
                    # 2. The function is a steering method (ask/interject)
                    # 3. The underlying tool opted into context initially
                    expose_context_cont_control=(
                        _expose_ctx_cont_control
                        and getattr(fn, "__supports_context_propagation__", False)
                        and getattr(fn, "__context_opted_in__", False)
                    ),
                )
                for fn in dynamic_tools.values()
            ]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                logger.info(f"LLM thinking…", prefix=ICONS["llm_thinking"])
                logger.mark_llm_thinking()

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                    "prompt_caching": prompt_caching,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["parallel_tool_calls"] = (
                        max_parallel_tool_calls > 1
                    )

                llm_task = asyncio.create_task(
                    generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(_gen_kwargs, tool_choice_mode),
                        **_gen_kwargs,
                    ),
                    name="LLMGenerate",
                )
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )

                # ➋ …but ALSO watch the tool tasks that were still pending
                pending_snapshot = set(tools_data.pending)
                # Listen for clarification and notification events while the LLM is thinking
                clar_waiters2: Dict[asyncio.Task, asyncio.Task] = {}
                notif_waiters2: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in pending_snapshot:
                    _inf = tools_data.info[_t]
                    # Clarifications: only for new requests
                    if (
                        _inf is not None
                        and not getattr(_inf, "waiting_for_clarification", False)
                        and _inf.clar_up_queue is not None
                    ):
                        cw2 = asyncio.create_task(
                            _inf.clar_up_queue.get(),
                            name="ClarificationQueueGet",
                        )
                        clar_waiters2[cw2] = _t
                    # Notifications: always listen when provided
                    if _inf is not None and _inf.notification_queue is not None:
                        pw2 = asyncio.create_task(
                            _inf.notification_queue.get(),
                            name="NotificationQueueGet",
                        )
                        notif_waiters2[pw2] = _t

                done, _ = await asyncio.wait(
                    pending_snapshot
                    | set(clar_waiters2.keys())
                    | set(notif_waiters2.keys())
                    | {llm_task, interject_w, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Helper cleanup: cancel auxiliary waiters only.
                # NOTE: llm_task is deliberately NOT cancelled here. Each branch
                # below decides whether to cancel the LLM based on context:
                # - Tool finished → cancel LLM (needs new context)
                # - Immediate interjection → cancel LLM (user wants immediate response)
                # - Patient interjection → DO NOT cancel (let LLM finish naturally)
                # - Clarification/notification → cancel LLM (needs to surface event)
                # - Cancellation requested → cancel LLM (explicit stop)
                for tsk in (
                    interject_w,
                    cancel_waiter,
                    *clar_waiters2.keys(),
                    *notif_waiters2.keys(),
                ):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                await asyncio.gather(
                    interject_w,
                    cancel_waiter,
                    *clar_waiters2.keys(),
                    *notif_waiters2.keys(),
                    return_exceptions=True,
                )

                # 0️⃣ A *different* tool finished before the LLM answered -----
                if done & pending_snapshot:  # ← NEW
                    # — cancel the half-finished reasoning step
                    if not llm_task.done():
                        llm_task.cancel()
                    for aux in (interject_w, cancel_waiter):
                        if aux not in done and not aux.done():
                            aux.cancel()
                    await asyncio.gather(
                        llm_task,
                        interject_w,
                        cancel_waiter,
                        return_exceptions=True,
                    )
                    # — handle each newly-finished task exactly as branch A does
                    needs_turn = False
                    for task in _sort_completed_tasks_by_call_id(
                        done & pending_snapshot,
                        tools_data,
                    ):
                        if await tools_data.process_completed_task(
                            task=task,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        ):
                            needs_turn = True

                    # …then restart the main loop so the model sees the new info
                    if needs_turn:  # assistant speaks only if needed
                        llm_turn_required = True
                    continue

                # 1️⃣ user interjected → restart immediately
                if interject_w in done:
                    _payload = None
                    try:
                        _payload = interject_w.result()
                    except Exception:
                        _payload = None
                    # Default to immediate behaviour unless explicitly disabled per interjection
                    _immediate = True
                    try:
                        if isinstance(_payload, dict):
                            _immediate = bool(
                                _payload.get("trigger_immediate_llm_turn", True),
                            )
                    except Exception:
                        _immediate = True
                    # Re-queue the payload so it is processed by the main drain path
                    await interject_queue.put(_payload)
                    if _immediate:
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        continue  # top of loop
                    # Patient mode: allow the in-flight LLM call to finish organically
                    # and ensure we schedule exactly one subsequent LLM turn after completion.
                    deferred_llm_turn = True
                    # Wait for the LLM to complete naturally (don't cancel it)
                    if not llm_task.done():
                        await asyncio.gather(llm_task, return_exceptions=True)

                # 2️⃣ clarification bubbled up while the LLM was thinking →
                #    cancel current LLM step, surface the clarification request,
                #    then restart the loop so the next assistant turn can ingest it.
                if done & set(clar_waiters2.keys()):
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    for cw in done & set(clar_waiters2.keys()):
                        await _handle_clarification(clar_waiters2[cw], cw.result())
                    llm_turn_required = True
                    continue

                # 3️⃣ notification bubbled up while the LLM was thinking →
                #    cancel current LLM step, surface the notification,
                #    then restart the loop so the next assistant turn can ingest it.
                if done & set(notif_waiters2.keys()):
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    for pw in done & set(notif_waiters2.keys()):
                        await _handle_notification(notif_waiters2[pw], pw.result())
                    llm_turn_required = True
                    continue

                # 2️⃣ cancellation requested
                if cancel_waiter in done:
                    # Only escalate when the cancellation flag is actually set.
                    if cancel_event.is_set():
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        raise asyncio.CancelledError

                # 3️⃣ LLM finished normally
                if llm_task.cancelled():
                    raise asyncio.CancelledError
                if llm_task.exception():
                    try:
                        llm_task.result()
                    except Exception as e:
                        raise Exception(
                            f"LLM call failed: {type(e).__name__}: {e}",
                        ) from e

                    # Clarification request bubbled up while LLM thinking
                    if done & set(clar_waiters2.keys()):
                        for cw in done & set(clar_waiters2.keys()):
                            await _handle_clarification(clar_waiters2[cw], cw.result())
                        llm_turn_required = True

                    # Notification bubbled up while LLM thinking
                    if done & set(notif_waiters2.keys()):
                        for pw in done & set(notif_waiters2.keys()):
                            await _handle_notification(notif_waiters2[pw], pw.result())
                        llm_turn_required = True

            else:
                # ––––– legacy *blocking* mode ––––––––––––––––––––––––––––
                try:
                    _gen_kwargs = {
                        "return_full_completion": True,
                        "tools": tmp_tools,
                        "tool_choice": tool_choice_mode,
                        "stateful": True,
                        "prompt_caching": prompt_caching,
                    }
                    if max_parallel_tool_calls is not None:
                        _gen_kwargs["parallel_tool_calls"] = (
                            max_parallel_tool_calls > 1
                        )

                    _result = await generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(_gen_kwargs, tool_choice_mode),
                        **_gen_kwargs,
                    )
                except Exception as e:
                    raise Exception(
                        f"LLM call failed: {type(e).__name__}: {e}",
                    ) from e

            msg = client.messages[-1]
            await to_event_bus(msg, cfg)

            # LLM responded - reset the activity-based timeout. The timeout is
            # designed to catch hung tools, not slow LLM inference. LLM providers
            # have their own timeout mechanisms; our timeout only guards against
            # user-defined tools that may hang indefinitely.
            timer.reset()

            if log_steps:
                with suppress(Exception):
                    # Pretty-print tool_call arguments in assistant messages for readability
                    from .utils import (
                        try_parse_json as _try_parse_json,
                    )  # local import to avoid cycles

                    _msg_for_logging = copy.deepcopy(msg)
                    _tcs = _msg_for_logging.get("tool_calls") or []
                    for _tc in _tcs:
                        _fn = _tc.get("function", {})
                        _fn["arguments"] = _try_parse_json(_fn.get("arguments"))
                    logger.info(
                        f"{json.dumps(_msg_for_logging, indent=4)}",
                        prefix=ICONS["llm_response"],
                    )

            # ── timeout guard (post-LLM) ───────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # LLM has just spoken – reset the flag
            llm_turn_required = False
            # one full assistant turn completed
            step_index += 1

            # ── E.  Launch any new tool calls  ──────────────────────────────
            # NOTE: The model returned `tool_calls`.  For *each* call we:
            #   1. JSON-parse the arguments once (costly in Python – do it
            #      outside the worker thread).
            #   2. Wrap sync functions in `asyncio.to_thread` so the event
            #      loop is never blocked by CPU / I/O.
            #   3. Create an `asyncio.Task` and remember contextual metadata
            #      in `task_info` so we can later insert the result in the
            #      exact chronological position.
            #   4. Keep a pristine copy of the original `tool_calls` list;
            #      step A temporarily hides it to avoid "naked" unresolved
            #      calls flashing in the UI, and restores it once *any*
            #      result for that assistant turn is ready.
            # Finally we `continue` so control jumps back to *branch A*
            # where we wait for the **first** task / cancel / interjection.
            if msg["tool_calls"]:
                # ── De-duplicate tool calls (optional) ────────────────────────
                if prune_tool_duplicates:
                    unique, _ = prune_duplicate_tool_calls(msg["tool_calls"])
                    if len(unique) != len(msg["tool_calls"]):
                        msg["tool_calls"] = unique

                # Always ensure over-quota tool calls are removed regardless of
                # deduplication settings, before any scheduling occurs.
                tools_data.prune_over_quota_tool_calls(msg)

                # If pruning removed all calls and left a placeholder notice, inject a user turn
                # so the model is prompted to continue. This prevents Assistant->Assistant history
                # violations on strict models.
                if not msg.get(
                    "tool_calls",
                ) and "(Tool calls were removed due to quota limits)" in str(
                    msg.get("content") or "",
                ):
                    # Use 'user' role to ensure robust alternation for all providers
                    sys_notice = {
                        "role": "user",
                        "content": "System notification: The tool calls in your last response were blocked due to quota limits. Please modify your plan or conclude.",
                    }
                    await _msg_dispatcher.append_msgs([sys_notice])

                _persist_response_emitted = False
                _persist_response_content = (
                    None  # captured by send_response for surfacing
                )

                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    called_tools.append(name)

                    # Parse arguments - handle both string and dict formats
                    _raw_args = call["function"]["arguments"]
                    if isinstance(_raw_args, str):
                        args = json.loads(_raw_args)
                    else:
                        args = _raw_args if isinstance(_raw_args, dict) else {}

                    # Special-case: handle response-submission tool
                    # (send_response in persist mode, final_response otherwise)
                    _is_response_tool = (
                        name in ("final_response", "send_response")
                        and response_format is not None
                    )
                    if _is_response_tool:
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            # Validate payload with the provided Pydantic model.
                            response_format.model_validate(payload)

                            # Cancel any in-flight tools before returning.
                            # In persist mode this block should be unreachable
                            # (response tool is masked while tools are pending)
                            # but guard defensively.
                            if tools_data.pending and not persist:
                                logger.info(
                                    f"{name} called while {len(tools_data.pending)} "
                                    f"task(s) are in-flight. Auto-cancelling to terminate.",
                                    prefix=ICONS["auto_cancel"],
                                )
                                await tools_data.cancel_pending_tasks()

                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=_dumps(payload, indent=4),
                            )

                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            if persist:
                                # Treat as current-turn response; don't terminate.
                                _persist_response_emitted = True
                                _persist_response_content = json.dumps(payload)
                                break  # exit the for-loop over tool_calls
                            return json.dumps(payload)
                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    "⚠️ Validation failed – proceeding with standard formatting step.\n"
                                    + str(_exc)
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # Special-case: handle generic response tool (no response_format)
                    # With the injection branch removed, this path is only reachable
                    # if the LLM hallucinates a response tool call.  Handle defensively.
                    _is_generic_response = (
                        name in ("final_response", "send_response")
                        and response_format is None
                        and multi_handle_coordinator is None
                    )
                    if _is_generic_response:
                        answer = args.get("answer") if isinstance(args, dict) else None
                        if answer is None:
                            answer = str(args) if args else ""

                        # Cancel any in-flight tools before returning.
                        if tools_data.pending and not persist:
                            logger.info(
                                f"{name} called while {len(tools_data.pending)} "
                                f"task(s) are in-flight. Auto-cancelling to terminate.",
                                prefix=ICONS["auto_cancel"],
                            )
                            await tools_data.cancel_pending_tasks()

                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=answer,
                        )

                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        if persist:
                            _persist_response_emitted = True
                            _persist_response_content = answer
                            break
                        return answer

                    # Special-case: handle multi-handle response tool
                    _is_multi_response = (
                        name == "final_response"
                        and multi_handle_coordinator is not None
                    )
                    if _is_multi_response:
                        try:
                            request_id = args.get("request_id")
                            answer = args.get("answer")

                            if request_id is None:
                                raise ValueError(
                                    "Missing 'request_id' in tool arguments.",
                                )
                            if answer is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            request_id = int(request_id)

                            # Validate request_id
                            error_msg = multi_handle_coordinator.validate_request_id(
                                request_id,
                            )
                            if error_msg:
                                tool_msg = create_tool_call_message(
                                    name=name,
                                    call_id=call["id"],
                                    content=f"⚠️ Error: {error_msg}",
                                )
                                await insert_tool_message_after_assistant(
                                    assistant_meta,
                                    msg,
                                    tool_msg,
                                    client,
                                    _msg_dispatcher,
                                )
                                continue

                            # Complete the request
                            multi_handle_coordinator.complete_request(
                                request_id,
                                str(answer),
                            )

                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=f"Request {request_id} completed successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            logger.info(
                                f"Request {request_id} completed with answer: {answer[:100]}{'...' if len(answer) > 100 else ''}",
                                prefix=ICONS["completed"],
                            )

                            # Check if all requests are done - if so, loop will terminate
                            # at the next iteration when it checks should_terminate()
                            continue

                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=f"⚠️ Error processing {name}: {_exc}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # Special-case: handle multi-handle `ask_user_clarification` tool
                    if (
                        name == "ask_user_clarification"
                        and multi_handle_coordinator is not None
                    ):
                        try:
                            request_id = args.get("request_id")
                            question = args.get("question")

                            if request_id is None:
                                raise ValueError(
                                    "Missing 'request_id' in tool arguments.",
                                )
                            if question is None:
                                raise ValueError(
                                    "Missing 'question' in tool arguments.",
                                )

                            request_id = int(request_id)

                            # Route the clarification to the appropriate request's queue
                            multi_handle_coordinator.route_clarification_to_request(
                                request_id,
                                {
                                    "type": "clarification",
                                    "request_id": request_id,
                                    "question": str(question),
                                },
                            )

                            tool_msg = create_tool_call_message(
                                name="ask_user_clarification",
                                call_id=call["id"],
                                content=f"Clarification question sent to request {request_id}. Waiting for user response.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name="ask_user_clarification",
                                call_id=call["id"],
                                content=f"⚠️ Error: {_exc}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # ── Special-case dynamic helpers ──────────────────────
                    # • wait        → acknowledge, list running tasks, no scheduling
                    # • cancel_*    → cancel underlying task & purge metadata
                    # Normalise tool-call name defensively
                    lname = str(name or "").strip()
                    lname_cf = lname.casefold()

                    if lname_cf == "wait":
                        # When there ARE pending tools, prune the wait call to avoid
                        # transcript clutter - the loop will naturally wait for them.
                        if tools_data.pending:
                            try:
                                logger.info(
                                    "Assistant chose `wait` – no-op; not persisting to transcript.",
                                    prefix=ICONS["wait"],
                                )
                            except Exception:
                                pass

                            # Prune the `wait` tool call using a shared helper
                            with suppress(Exception):
                                from .messages import (
                                    prune_wait_tool_call as _prune_wait,
                                )

                                _prune_wait(msg, call["id"], client=client)

                            # After acknowledging a wait, do NOT grant an immediate LLM turn.
                            # The loop should now wait for any pending tools or interjections.
                            continue

                        # When there are NO pending tools, pruning would cause an
                        # infinite cache loop (same conversation → same cached response).
                        # Instead, insert a factual tool response. This:
                        # 1. Changes the conversation state (breaks cache)
                        # 2. Is purely informational (no prescriptive instructions)
                        # 3. Remains accurate even if interjections arrive later
                        try:
                            logger.info(
                                "Assistant called `wait` with no pending tools.",
                                prefix=ICONS["wait"],
                            )
                        except Exception:
                            pass

                        tool_msg = create_tool_call_message(
                            name="wait",
                            call_id=call["id"],
                            content="No tasks are currently running.",
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    elif lname_cf.startswith("stop_") and not lname_cf.startswith(
                        "_stop_tasks",
                    ):
                        # Helper names are of the form: stop_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # ── locate & cancel the underlying coroutine ──────
                        task_to_cancel = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if str(info.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        orig_fn = (
                            tools_data.info[task_to_cancel].name
                            if task_to_cancel
                            else "unknown"
                        )
                        arg_json = (
                            tools_data.info[task_to_cancel].call_dict["function"][
                                "arguments"
                            ]
                            if task_to_cancel
                            else "{}"
                        )
                        pretty_name = f"stop   {orig_fn}({arg_json})"

                        # Parse payload to forward extras to handle.stop if available
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                        if "payload" not in locals():
                            payload = {}

                        # ── gracefully shut down any *nested* async-tool loop first (central dispatcher) ──────
                        if task_to_cancel:
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "stop",
                                    payload if isinstance(payload, dict) else {},
                                    tools_data.info[task_to_cancel],
                                )

                        # ── then cancel the waiter coroutine itself ───────────────────────────
                        if task_to_cancel and not task_to_cancel.done():
                            task_to_cancel.cancel()
                        if task_to_cancel:
                            tools_data.pop_task(task_to_cancel)

                    # Acknowledge only when a live target was actually affected
                    if lname_cf.startswith("stop_") and task_to_cancel:
                        with suppress(Exception):
                            try:
                                reason_txt = payload.get("reason")
                            except Exception:
                                reason_txt = ""

                            tool_msg = create_tool_call_message(
                                name=pretty_name,
                                call_id=call["id"],
                                content=f"The tool call [{call_id_suffix}] has been stopped successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            continue  # helper handled for a live target

                    # ── _pause helper ────────────────────────────────────────────────
                    elif lname_cf.startswith("pause_") and not lname_cf.startswith(
                        "_pause_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if str(info.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task].name if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task].call_dict["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"pause {orig_fn}({arg_json})"

                        # Forward via central dispatcher (pause)
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                        if "payload" not in locals():
                            payload = {}

                        if tgt_task:
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "pause",
                                    payload,
                                    tools_data.info[tgt_task],
                                )

                            tool_msg = create_tool_call_message(
                                name=pretty_name,
                                call_id=call["id"],
                                content=f"The tool call [{call_id_suffix}] has been paused successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue  # helper handled for live target; otherwise fall through

                    # ── _resume helper ───────────────────────────────────────────────
                    elif lname_cf.startswith("resume_") and not lname_cf.startswith(
                        "_resume_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if str(info.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task].name if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task].call_dict["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"resume {orig_fn}({arg_json})"

                        # Forward via central dispatcher (resume)
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                        if "payload" not in locals():
                            payload = {}

                        if tgt_task:
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "resume",
                                    payload,
                                    tools_data.info[tgt_task],
                                )

                        if tgt_task:
                            tool_msg = create_tool_call_message(
                                name=pretty_name,
                                call_id=call["id"],
                                content=f"The tool call [{call_id_suffix}] has been resumed successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue  # helper handled (live target); otherwise fall through to base

                    elif lname_cf.startswith("clarify_"):
                        # Helper names are of the form: clarify_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]
                        ans = args["answer"]

                        # ── find the underlying pending task (if still alive) ───────────────
                        tgt_task = next(  # ← NEW
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        # Deliver via central dispatcher, then clear waiting flag
                        if tgt_task:
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "clarify",
                                    {"answer": ans},
                                    tools_data.info[tgt_task],
                                )
                                # ✔️ the tool is un-blocked – start watching it again
                                for _t, _inf in tools_data.info.items():
                                    if str(_inf.call_id).endswith(call_id_suffix):
                                        _inf.waiting_for_clarification = False
                                        break

                        if tgt_task:
                            # Always publish a tool reply acknowledging the clarify helper
                            tool_reply_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    f"Clarification answer sent upstream: {ans!r}\n"
                                    "⏳ Waiting for the original tool to finish…"
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_reply_msg,
                                client,
                                _msg_dispatcher,
                            )
                            # Store the clarify helper's reply so that when the tool
                            # completes, the final result goes here (not into the
                            # clarification_request_* message which preserves the question)
                            tools_data.info[tgt_task].clarify_placeholder = (
                                tool_reply_msg
                            )
                            continue  # handled clarify helper for live target

                    elif lname_cf.startswith("interject_"):
                        # helper signature mirrors downstream handle.interject (content plus any extras)
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                            new_text = payload.get("content") or payload.get("message")
                            if new_text is None:
                                new_text = ""
                        if "payload" not in locals():
                            payload = {}
                            new_text = "<unparsable>"

                        # Helper names are of the form: interject_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # locate the underlying long-running task
                        tgt_task = next(
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.call_id).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        pretty_name = (
                            f"interject {tools_data.info[tgt_task].name}({new_text})"
                            if tgt_task
                            else name
                        )

                        # ― forward via central dispatcher -------------
                        if tgt_task:
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "interject",
                                    payload,
                                    tools_data.info[tgt_task],
                                )

                            # ― emit a tool message so the chat log stays tidy ---
                            tool_msg = create_tool_call_message(
                                name=pretty_name,
                                call_id=call["id"],
                                content=f'Guidance "{new_text}" forwarded to the running tool.',
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue  # handled interject helper for live target

                    # (ask_* helpers are treated as base dynamic tools; no special-case here)

                    # Respect hidden per-tool total-call quotas (pre-pruned); guard
                    if tools_data.has_exceeded_quota_for_tool(name):
                        continue

                    # Respect *per-tool* concurrency limits  ────────────────
                    if tools_data.has_exceeded_concurrent_limit_for_tool(name):
                        # Concurrency cap reached → immediately insert a
                        # *tool-error* message and **do not** schedule.
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                f"⚠️ Cannot start '{name}': "
                                f"max_concurrent={tools_data.normalized[name].max_concurrent} "
                                "already reached. Wait for an existing call to "
                                "finish or stop one before retrying."
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    # first check any dynamic helpers we generated for long-running handles
                    if name in dynamic_tools:
                        # Global dispatchers (e.g. ask_about_completed_tool) are not
                        # tied to any specific live call — skip the suffix check.
                        _is_global_dispatcher = name == "ask_about_completed_tool"
                        # Disambiguation: only treat as a dynamic helper when its suffix targets a live call
                        _helper_targets_live = True
                        if not _is_global_dispatcher:
                            try:
                                _suffix = str(name).split("_")[-1]
                                _helper_targets_live = any(
                                    str(inf.call_id).endswith(_suffix)
                                    for inf in tools_data.info.values()
                                )
                            except Exception:
                                _helper_targets_live = True
                        if _helper_targets_live:
                            fn = dynamic_tools[name]

                            # ── build **extra** kwargs (chat context + queue) for dynamic helper ──
                            #
                            # Context propagation for dynamic helpers:
                            # - ask_* tools: Need FULL initial context (like base tools), since they
                            #   spawn new inspection loops. Use is_continuation_only=False.
                            # - interject_* tools: Need only continuation context (tool already has
                            #   initial context). Use is_continuation_only=True.
                            # - stop_*, pause_*, resume_*: No context needed.
                            #
                            target_call_id = (
                                name.split("_")[-1] if "_" in name else call["id"]
                            )
                            # Find the target tool's metadata to check context_opted_in
                            target_info = None
                            for _t, _inf in tools_data.info.items():
                                if str(_inf.call_id).endswith(target_call_id):
                                    target_info = _inf
                                    break

                            sig = inspect.signature(fn)
                            params = sig.parameters
                            has_varkw = any(
                                p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in params.values()
                            )

                            # Determine if this is an ask_* tool (needs full context) vs
                            # interject_* (needs continuation only) vs other (no context)
                            is_ask_tool = name.startswith("ask_")
                            is_interject_tool = name.startswith("interject_")
                            accepts_parent_ctx = (
                                "_parent_chat_context" in params or has_varkw
                            )
                            accepts_parent_ctx_cont = (
                                "_parent_chat_context_cont" in params or has_varkw
                            )

                            # Use shared helper for context injection
                            if is_ask_tool:
                                # ask_* tools spawn new loops - need full initial context
                                extra_kwargs, context_opted_in = (
                                    compute_context_injection(
                                        args=args,
                                        propagate_chat_context=propagate_chat_context,
                                        context_state=context_state,
                                        client_messages=client.messages,
                                        call_id=f"ask_{target_call_id}_{call['id']}",
                                        accepts_parent_ctx=accepts_parent_ctx,
                                        accepts_parent_ctx_cont=accepts_parent_ctx_cont,
                                        is_continuation_only=False,
                                    )
                                )
                            elif is_interject_tool:
                                # interject_* tools add to existing loops - need continuation only
                                extra_kwargs, _ = compute_context_injection(
                                    args=args,
                                    propagate_chat_context=propagate_chat_context,
                                    context_state=context_state,
                                    client_messages=client.messages,
                                    call_id=f"interject_{target_call_id}_{call['id']}",
                                    accepts_parent_ctx=False,  # Don't inject full context
                                    accepts_parent_ctx_cont=accepts_parent_ctx_cont,
                                    target_context_opted_in=(
                                        target_info.context_opted_in
                                        if target_info
                                        else None
                                    ),
                                    is_continuation_only=True,
                                )
                            else:
                                # Other steering tools (stop, pause, resume) - no context
                                # Still pop the control params so they don't get forwarded
                                args.pop("include_parent_chat_context", None)
                                args.pop("include_parent_chat_context_cont", None)
                                extra_kwargs = {}
                                context_opted_in = False

                            filtered_extras = {
                                k: v
                                for k, v in extra_kwargs.items()
                                if k in params or has_varkw
                            }
                            # Forward ALL call args verbatim. Let the callee raise if unsupported.
                            allowed_call_args = args
                            merged_kwargs = {**allowed_call_args, **filtered_extras}

                            if asyncio.iscoroutinefunction(fn):
                                coro = fn(**merged_kwargs)
                            else:
                                coro = asyncio.to_thread(fn, **merged_kwargs)

                            # (Argument pretty-printing now handled in assistant message logs only)

                            call_dict = {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                            # If this dynamic helper is marked as write-only, acknowledge immediately
                            # and run fire-and-forget without tracking in pending/task_info.
                            if getattr(fn, "__write_only__", False):
                                with suppress(Exception):
                                    tool_msg = create_tool_call_message(
                                        name=name,
                                        call_id=call["id"],
                                        content=build_helper_ack_content(
                                            name,
                                            call["function"]["arguments"],
                                        ),
                                    )
                                    await insert_tool_message_after_assistant(
                                        assistant_meta,
                                        msg,
                                        tool_msg,
                                        client,
                                        _msg_dispatcher,
                                    )
                                with suppress(Exception):
                                    asyncio.create_task(coro, name=f"ToolCall_{name}")
                                continue

                            # Scheduling dynamic helper call
                            t = asyncio.create_task(coro, name=f"ToolCall_{name}")
                            metadata = ToolCallMetadata(
                                name=name,
                                call_id=call["id"],
                                assistant_msg=msg,
                                call_dict=call_dict,
                                call_idx=idx,
                                is_interjectable=False,
                                is_dynamic=True,
                                chat_context=extra_kwargs.get("_parent_chat_context"),
                                pause_event=None,
                                # Debug helpers for failure logging
                                tool_schema=method_to_schema(
                                    fn,
                                    include_class_name=include_class_in_dynamic_tool_names,
                                ),
                                llm_arguments=allowed_call_args,
                                raw_arguments_json=call["function"]["arguments"],
                                # Track context opt-in for adopted handles (important for ask_*)
                                context_opted_in=context_opted_in,
                            )
                            tools_data.save_task(
                                coro=t,
                                metadata=metadata,
                            )
                        else:
                            # Target task was already removed (e.g., by a prior
                            # stop_ helper in the same assistant message). Insert
                            # a no-op acknowledgement so the transcript stays valid.
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    f"No-op: target task for '{name}' is no longer active."
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                    else:
                        # ── Unknown/unavailable tool fallback ─────────────────────
                        # If the tool doesn't exist OR wasn't visible on this turn
                        # (e.g., the model hallucinated a tool name, or the tool was
                        # hidden by tool_policy), insert an error tool response to
                        # keep the transcript valid. Without this, the assistant
                        # message would have an unresolved tool_call, causing
                        # subsequent LLM calls to fail.
                        if name not in policy_tools_norm:
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    f"⚠️ Error: Tool '{name}' is not available. "
                                    "The tool may have been removed or does not exist. "
                                    "Please proceed without using this tool."
                                ),
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        # Use shared helper for base tools
                        await tools_data.schedule_base_tool_call(
                            msg,
                            name=name,
                            args_json=call["function"]["arguments"],
                            call_id=call["id"],
                            call_idx=idx,
                            context_state=context_state,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                            initial_paused=not pause_event.is_set(),
                        )

                if _persist_response_emitted:
                    pass  # fall through to section F → persist wait
                else:
                    # metadata for orderly insertion
                    assistant_meta[id(msg)] = {
                        "results_count": 0,
                    }

                    # Immediately insert placeholder tool replies for every newly scheduled call
                    #  to satisfy API ordering even if a user interjection arrives instantly.
                    try:
                        await ensure_placeholders_for_pending(
                            assistant_msg=msg,
                            tools_data=tools_data,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                            time_ctx=time_ctx,
                        )
                    except Exception as _ph_exc:
                        logger.error(
                            f"Failed to insert immediate placeholders: {_ph_exc!r}",
                        )

                    continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Three scenarios reach this block:
            #   • `pending` **non-empty** and NOT all blocked on clarification
            #     → older tool tasks are still in flight; loop back to wait.
            #   • `pending` **non-empty** but ALL blocked on clarification
            #     → the LLM decided to end without answering; cancel blocked
            #     tasks so we can exit gracefully instead of deadlocking.
            #   • `pending` empty → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if tools_data.pending:
                # Check if ALL pending tasks are blocked waiting for clarification.
                # If the LLM returned content (no tool calls) while tasks are waiting
                # for clarification, the LLM has decided to end the conversation
                # without answering. Cancel those blocked tasks to avoid deadlock.
                blocked_on_clar = [
                    t
                    for t in tools_data.pending
                    if getattr(
                        tools_data.info.get(t),
                        "waiting_for_clarification",
                        False,
                    )
                ]
                not_blocked = [
                    t for t in tools_data.pending if t not in blocked_on_clar
                ]

                if blocked_on_clar and not not_blocked:
                    # ALL pending tasks are blocked on clarification - cancel them
                    logger.info(
                        f"LLM returned content while {len(blocked_on_clar)} task(s) "
                        f"await clarification. Cancelling blocked tasks to exit.",
                        prefix=ICONS["auto_cancel"],
                    )
                    for t in blocked_on_clar:
                        t.cancel()
                    await asyncio.gather(*blocked_on_clar, return_exceptions=True)
                    for t in blocked_on_clar:
                        tools_data.pending.discard(t)
                    # Fall through to return the final answer
                else:
                    if persist:
                        # In persist mode, never cancel in-flight tools due
                        # to a bare text response.  Loop back to Section A
                        # which properly races tool completions,
                        # notifications, interjections, and cancellation.
                        continue
                    # LLM gave text-only response while tools are in-flight.
                    # This is a valid termination signal - cancel all running
                    # tasks and return the LLM's response.
                    logger.info(
                        f"LLM returned text-only response while {len(not_blocked)} "
                        f"task(s) are in-flight. Auto-cancelling to terminate.",
                        prefix=ICONS["auto_cancel"],
                    )
                    await tools_data.cancel_pending_tasks()
                    # Fall through to return the final answer

            # If a patient interjection arrived during the last LLM step, or if there
            # are unprocessed interjections queued, process them before returning.
            try:
                if deferred_llm_turn or not interject_queue.empty():
                    deferred_llm_turn = False
                    continue  # drain interjections at top-of-loop; grants one extra LLM turn
            except Exception:
                pass

            # ── timeout guard (final turn) ──────────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            final_content = msg["content"]

            # ── multi-handle mode: check if all requests are done ──
            if multi_handle_coordinator is not None:
                if multi_handle_coordinator.should_terminate():
                    # All requests completed/cancelled and persist=False
                    logger.info(
                        "Multi-handle mode: all requests completed, terminating loop.",
                        prefix=ICONS["completed"],
                    )
                    multi_handle_coordinator.close()
                    return final_content  # Return last assistant content (may be empty)
                else:
                    # Still have pending requests - continue waiting
                    logger.info(
                        f"Multi-handle mode: {multi_handle_coordinator.registry.pending_count()} request(s) still pending.",
                        prefix=ICONS["pending"],
                    )
                    # Wait for next interjection or tool completion
                    continue

            # ── persist mode: wait for next interjection instead of returning ──
            if persist:
                # Surface the turn-complete response to the outer handle so the
                # ConversationManager can distinguish "response (awaiting input)"
                # from in-progress "notification" events.
                _response_to_surface = (
                    _persist_response_content
                    if _persist_response_content is not None
                    else final_content
                )
                _outer = outer_handle_container[0] if outer_handle_container else None
                if (
                    _outer is not None
                    and hasattr(_outer, "_notification_q")
                    and _response_to_surface
                ):
                    await _outer._notification_q.put(
                        {
                            "type": "response",
                            "content": _response_to_surface,
                        },
                    )
                # Reset for the next turn
                _persist_response_content = None
                _persist_response_emitted = False

                logger.info(
                    "Persist mode: waiting for next interjection...",
                    prefix=ICONS["pause"],
                )
                # Block until an interjection arrives or cancellation is requested
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="PersistCancelWait",
                )
                interject_waiter = asyncio.create_task(
                    interject_queue.get(),
                    name="PersistInterjectWait",
                )
                done, pending = await asyncio.wait(
                    {cancel_waiter, interject_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Clean up the waiter that didn't finish
                for p in pending:
                    p.cancel()
                    await asyncio.gather(p, return_exceptions=True)

                # Check if we were cancelled
                if cancel_event.is_set():
                    raise asyncio.CancelledError

                # An interjection arrived - put it back in the queue for normal processing
                # at the top of the loop
                if interject_waiter in done:
                    try:
                        interjection = interject_waiter.result()
                        await interject_queue.put(interjection)
                        logger.info(
                            "Persist mode: interjection received, resuming loop",
                            prefix=ICONS["resume"],
                        )
                    except Exception:
                        pass
                # Reset timer for the new "turn"
                timer.reset()
                continue  # Back to top of loop to process the interjection

            return final_content  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
        # NOTE: Caller (or parent task) requested cancellation.  We propagate
        # the signal to *all* running tool tasks first so each can release
        # resources cleanly.  Only after every task has finished/aborted do
        # we re-raise the same `CancelledError`, preserving expected asyncio
        # semantics for upstream callers.
        await tools_data.cancel_pending_tasks()
        raise
    finally:
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
