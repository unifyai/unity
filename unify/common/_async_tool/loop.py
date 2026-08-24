import asyncio
import unillm
import hashlib
import json
import inspect
import copy
from dataclasses import dataclass, field

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
from .utils import maybe_await, get_handle_paused_state
from .event_bus_util import to_event_bus
from ...events.types.tool_loop import ToolLoopKind
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
from ..llm_helpers import (
    DEFAULT_TOOL_SCHEMA_STRICT,
    method_to_schema,
    _dumps,
    short_id,
)
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
    is_mutable,
    is_loop_authored_message,
    loop_user_notice,
    extract_substantive_text,
    compact_reviewed_messages,
    strip_reasoning_payloads,
    _rebaseline_watermark_hash,
)
from .tools_data import (
    ToolsData,
    compute_context_injection,
)
from .dynamic_tools_factory import DynamicToolFactory
from .time_context import create_time_context, TimeContext
from .context_compression import (
    compress_context,
    _COMPRESSION_SIGNAL,
    context_over_threshold,
)
from .response_format import (
    NormalizedResponseFormat,
    normalize_response_format,
)
from ..context_dump import make_messages_safe_for_context_dump
from ...common.hierarchical_logger import ICONS

if TYPE_CHECKING:
    from .multi_handle import MultiHandleCoordinator
    from unillm.types import PromptCacheParam


@dataclass
class ToolLoopRuntimeState:
    call_counts: Dict[str, int] = field(default_factory=dict)
    called_tools: list[str] = field(default_factory=list)
    step_index: int = 0
    consecutive_failures: int = 0
    message_count_offset: int = 0
    # Refused calls, tallied two ways so that working out an argspec is free
    # while repeating a rejected call is not. Both persist for the run: a call
    # refused identically three times is not being learned from, and a success
    # elsewhere does not change that.
    refusals_by_call: Dict[str, int] = field(default_factory=dict)
    refusals_by_complaint: Dict[str, int] = field(default_factory=dict)
    pending_stop_reason: Optional[str] = None


def _parse_tool_policy_result(
    result: Any,
) -> Tuple[str, Dict[str, Callable], bool]:
    """Normalize a ``tool_policy`` return value.

    Accepted shapes:
      - ``(mode, tools)``
      - ``(mode, tools, {"eager": bool, ...})``

    ``eager=True`` means: after the model schedules tool calls on this turn,
    immediately grant another LLM turn (without waiting for those tools to
    finish) for as long as subsequent policy evaluations keep returning
    ``eager=True``.  Default is ``False`` (wait for tool results).

    While ``eager=True``, the loop also withholds ``compress_context`` from the
    visible tool schema (except on the forced over-threshold compression path)
    so gated required policies cannot be bypassed by compressing context.
    """
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise TypeError(
            f"tool_policy must return (mode, tools[, opts]), got {type(result)!r}",
        )
    mode, tools = result[0], result[1]
    eager = False
    if len(result) >= 3:
        opts = result[2]
        if isinstance(opts, dict):
            eager = bool(opts.get("eager", False))
        else:
            eager = bool(opts)
    return str(mode), tools, eager


def _is_cache_miss_error(exc: BaseException | None) -> bool:
    """True when *exc* (or anything in its cause/context chain) is a
    read-only LLM-cache miss (``unillm.caching.CacheMissError``)."""
    from unillm.caching import CacheMissError

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, CacheMissError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False


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
        self._thinking_emitted = False

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def debug(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.debug(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)

    def begin_thinking(self) -> None:
        self._thinking_emitted = False
        if not self._first_llm_logged:
            self._first_llm_logged = True
            for p, m in self._defer_after_first_llm:
                self.info(m, prefix=p)
            self._defer_after_first_llm.clear()

    def emit_thinking_with_path(self, path) -> None:
        self._thinking_emitted = True
        self.info(f"LLM thinking… → {path}", prefix=ICONS["llm_thinking"])

    def emit_thinking_fallback(self) -> None:
        if not self._thinking_emitted:
            self._thinking_emitted = True
            self.info("LLM thinking…", prefix=ICONS["llm_thinking"])

    def defer_after_first_llm(self, msg: str, prefix: str = "") -> None:
        if self._first_llm_logged:
            self.info(msg, prefix=prefix)
        else:
            self._defer_after_first_llm.append((prefix, msg))


class _LoopToolFailureTracker:
    def __init__(
        self,
        max_consecutive_failures: int,
        runtime_state: ToolLoopRuntimeState,
    ):
        self._runtime_state = runtime_state
        self._max_consecutive_failures = max_consecutive_failures

    @property
    def current_failures(self):
        return self._runtime_state.consecutive_failures

    @property
    def max_failures(self):
        return self._max_consecutive_failures

    def has_exceeded_failures(self) -> bool:
        return (
            self._runtime_state.consecutive_failures >= self._max_consecutive_failures
        )

    def increment_failures(self):
        self._runtime_state.consecutive_failures += 1

    def reset_failures(self):
        self._runtime_state.consecutive_failures = 0

    # ── refused calls ──────────────────────────────────────────────────────
    #
    # A refusal is not a fault the way an unexpected exception is: converging on
    # an argspec means being told what is wrong and trying again, so counting
    # refusals against `max_consecutive_failures` would abort exactly the
    # behaviour the messages exist to produce. What is never progress is
    # repetition, so refusals are counted by what repeats rather than by how
    # many there are.

    # The same call, refused and sent again unchanged. Nothing was read.
    IDENTICAL_CALL_LIMIT = 3
    # The same complaint, however the arguments are dressed. Something is being
    # varied, but not the part the refusal is about.
    SAME_COMPLAINT_LIMIT = 6

    def note_refusal(self, *, tool_name: str, args: Any, message: str) -> Optional[str]:
        """Record a refused call; return why to stop, or ``None`` to continue."""
        state = self._runtime_state

        call_key = f"{tool_name}::{_fingerprint(args)}"
        state.refusals_by_call[call_key] = state.refusals_by_call.get(call_key, 0) + 1
        seen = state.refusals_by_call[call_key]
        if seen >= self.IDENTICAL_CALL_LIMIT:
            return self._stop(
                f"{tool_name} was called with the same arguments and refused "
                f"{seen} times, so the refusal is not being read. Last refusal: "
                f"{message}",
            )

        complaint_key = f"{tool_name}::{_fingerprint(message)}"
        state.refusals_by_complaint[complaint_key] = (
            state.refusals_by_complaint.get(complaint_key, 0) + 1
        )
        same_complaint = state.refusals_by_complaint[complaint_key]
        if same_complaint >= self.SAME_COMPLAINT_LIMIT:
            return self._stop(
                f"{tool_name} was refused {same_complaint} times for the same "
                f"reason while the arguments varied around it, so the part being "
                f"varied is not the part at fault. Refusal: {message}",
            )

        return None

    def _stop(self, reason: str) -> str:
        self._runtime_state.pending_stop_reason = reason
        return reason

    def stop_reason(self) -> Optional[str]:
        """Why the loop should end now, or ``None`` to keep going."""
        if self._runtime_state.pending_stop_reason is not None:
            return self._runtime_state.pending_stop_reason
        if self.has_exceeded_failures():
            return "Aborted after too many consecutive tool failures."
        return None


def _fingerprint(value: Any) -> str:
    """Stable short digest of *value*, so tallies cost no memory of their own."""
    try:
        rendered = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        rendered = repr(value)
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()[:16]


def _check_valid_response_format(response_format: Any) -> dict[str, Any]:
    """Return the JSON Schema for ``final_response``/``send_response`` ``answer``.

    Accepts a Pydantic ``BaseModel`` subclass, a JSON Schema dict, a simplified
    ``{field: type}`` dict, or a JSON string encoding one of those dicts.
    """
    normalized = normalize_response_format(response_format)
    if normalized is None:
        raise TypeError("response_format is required")
    return normalized.answer_json_schema


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
    interrupt_llm_on_tool_completion: bool = True,
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
            Callable[
                [int, Dict[str, Callable]],
                Union[
                    Tuple[str, Dict[str, Callable]],
                    Tuple[str, Dict[str, Callable], Dict[str, Any]],
                ],
            ],
            Callable[
                [int, Dict[str, Callable], list[str]],
                Union[
                    Tuple[str, Dict[str, Callable]],
                    Tuple[str, Dict[str, Callable], Dict[str, Any]],
                ],
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
    enable_compression: bool = True,
    extra_compression_tools: Optional[list[str]] = None,
    clarification_queues: Optional[Tuple["asyncio.Queue", "asyncio.Queue"]] = None,
    on_clarification_request: Optional[Callable[[str], Any]] = None,
    on_clarification_answer: Optional[Callable[[str], Any]] = None,
    on_notify: Optional[Callable[[str], Any]] = None,
    runtime_state: Optional[ToolLoopRuntimeState] = None,
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

    propagate_chat_context : ``ChatContextPropagation``, default ``LLM_DECIDES``
        Controls whether a filtered snapshot of this loop's conversation
        (genuine user turns and substantive assistant text only) is threaded
        into child tools that accept a ``_parent_chat_context`` keyword
        argument.  ``ALWAYS`` injects on every such call, ``NEVER`` on none,
        and ``LLM_DECIDES`` exposes an ``include_parent_chat_context``
        parameter the model may set to ``true`` — omission means no context.
        The ``_parent_chat_context`` argument itself is injected
        automatically and is **not** exposed to the LLM.

     tool_policy : ``Callable | None``, default ``None``
         Optional callable that *dynamically* controls tool exposure **and**
         whether a tool call is **required** on a given turn.  Receives the
         current turn index (starting at ``0``) and the full mapping
         ``{name → callable}`` (and optionally the list of previously called
         tool names as a third argument).  It must return
         ``(policy, tools)`` or ``(policy, tools, {"eager": bool})`` where
         ``policy`` is either ``"auto"`` or ``"required"`` (fed straight into
         ``tool_choice``) and ``tools`` is the possibly-filtered mapping of
         base tools visible on that turn.  When ``eager`` is ``True``, the
         loop grants another LLM turn immediately after scheduling tool
         calls (without waiting for them to finish), for as long as the
         policy keeps returning ``eager=True``.  Eager turns also withhold
         ``compress_context`` from the visible schema (forced over-threshold
         compression is unchanged).  Omit ``eager`` (or set it ``False``) to
         keep the default wait-for-results behaviour.

    parent_chat_context : ``list[dict] | None``
        Nested chat structure passed from an **outer** loop.  When a tool
        call opts into context (or ``propagate_chat_context`` is ``ALWAYS``),
        the filtered snapshot of this context is forwarded to that inner tool
        on its first call, with subsequent calls receiving only incremental
        updates (new messages since the last call) to avoid token waste.

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
            setattr(outer_handle_container[0], "_loop_cfg", cfg)
    logger = LoopLogger(cfg, log_steps)

    # Wire inline log-file pointers: when UNILLM_LOG_DIR is set, each LLM call
    # writes a request+response file.  The pending callback fires at the START
    # of each generate() call (before inference), letting us combine the
    # "LLM thinking…" message with the log file path into a single line.
    if log_steps:
        client.set_on_log_file_pending(
            lambda path: logger.emit_thinking_with_path(path),
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

    # Normalize response_format once. LLM-supplied nested tool args may pass a
    # JSON Schema dict / JSON string rather than a Pydantic class; accept those
    # so final_response can be injected. Unsupported values disable structured
    # mode rather than forcing tool_choice=required with no escape hatch.
    _rf_norm: Optional[NormalizedResponseFormat] = None
    if response_format is not None:
        try:
            _rf_norm = normalize_response_format(response_format)
        except Exception as _exc:  # noqa: BLE001
            logger.error(
                f"response_format normalization failed ({_exc!r}); "
                f"continuing without structured-output mode.",
            )
            _rf_norm = None

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # the response-submission tool happens during the loop.
    # NOTE: This hint is added as a new system message (not mutating the original)
    # and is appended later via _msg_dispatcher.append_msgs().
    _response_format_hint: str | None = None
    if _rf_norm is not None:
        _response_format_hint = (
            "## Response Format\n"
            "NOTE: After completing all tool calls, submit your final answer via "
            "the response tool as JSON that conforms to the following schema. "
            "Do NOT include any extra keys or commentary.\n"
            + json.dumps(_rf_norm.answer_json_schema, indent=2)
        )

    runtime_state = runtime_state or ToolLoopRuntimeState()

    # ── runtime guards ────────────────────────────────────────────────────
    # A run with no step ceiling ends only when the model chooses to stop, so
    # one that never converges keeps calling tools — and billing — forever.
    # Fall back to the configured ceiling when a caller expresses no opinion,
    # which bounds every entry point at once; an explicit ``max_steps`` still
    # wins for callers that legitimately need more.
    if max_steps is None:
        from unify.settings import SETTINGS as _SETTINGS

        configured_max_steps = _SETTINGS.UNIFY_MAX_TOOL_LOOP_STEPS
        max_steps = configured_max_steps if configured_max_steps > 0 else None

    # rolling timeout ----------------------------------------------------
    timer: TimeoutTimer = TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
        message_count_offset=runtime_state.message_count_offset,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)
    parent_chat_context_safe = make_messages_safe_for_context_dump(parent_chat_context)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context_safe:
                from .utils import format_json_for_log

                logger.info(
                    f"Parent Context: {format_json_for_log(parent_chat_context_safe)}",
                    prefix=ICONS["tool_seeding"],
                )
            logger.info(
                f"System Message: {client.system_message}",
                prefix=ICONS["system_message"],
            )
        # Log request (skip if seeding with a batch - per-item logs are emitted below)
        if not isinstance(message, list):
            logger.info(f"Request: {message}", prefix=ICONS["request"])

    import time as _setup_time

    _setup_t0 = _setup_time.perf_counter()

    def _setup_elapsed() -> str:
        return f"{(_setup_time.perf_counter() - _setup_t0) * 1000:.0f}ms"

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

    logger.debug(
        f"[setup +{_setup_elapsed()}] context built, appending system msgs ({len(msgs_to_append)} msgs)",
    )
    await _msg_dispatcher.append_msgs(msgs_to_append)
    logger.debug(f"[setup +{_setup_elapsed()}] system msgs appended")

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

        logger.debug(
            f"[setup +{_setup_elapsed()}] appending seeded batch ({len(seeded_batch)} msgs)",
        )
        await _msg_dispatcher.append_msgs(seeded_batch)
        logger.debug(f"[setup +{_setup_elapsed()}] seeded batch appended")

    # ── initial prompt ───────────────────────────────────────────────────────
    # ── 0-b. Coerce tools → ToolSpec & helper lambdas ───────────────────────
    #
    # • «tools_data.normalized» holds the *canonical* mapping name → ToolSpec
    # • helper for the active-count of one tool (cheap O(#pending))
    # • helper that answers "may we launch / advertise *this* tool right now?"
    #   by comparing the live count with max_concurrent.
    # -----------------------------------------------------------------------

    # ── Inject loop-owned tools when the caller opted in ────────────────
    if clarification_queues is not None:
        from ..llm_helpers import make_request_clarification_tool

        _clar_up_q, _clar_down_q = clarification_queues
        tools["request_clarification"] = make_request_clarification_tool(
            _clar_up_q,
            _clar_down_q,
            on_request=on_clarification_request,
            on_answer=on_clarification_answer,
        )

    if on_notify is not None:
        from ..llm_helpers import make_send_notification_tool

        tools["send_notification"] = make_send_notification_tool(on_notify=on_notify)

    # Initialise loop state early so preflight backfill can schedule tasks
    logger.debug(f"[setup +{_setup_elapsed()}] initialising ToolsData")
    tools_data: ToolsData = ToolsData(
        tools,
        client=client,
        logger=logger,
        time_ctx=time_ctx,
        extra_ask_tools=extra_ask_tools,
        call_counts=runtime_state.call_counts,
    )
    logger.debug(
        f"[setup +{_setup_elapsed()}] ToolsData ready ({len(tools_data.normalized)} tools)",
    )

    _alias_lookup = {
        name: spec.display_label
        for name, spec in tools_data.normalized.items()
        if spec.display_label
    }
    cfg.tool_alias_lookup = _alias_lookup or None

    consecutive_failures = _LoopToolFailureTracker(
        max_consecutive_failures,
        runtime_state,
    )
    assistant_meta: Dict[int, Dict[str, Any]] = {}

    _max_input_tokens = unillm.get_max_input_tokens(client.endpoint)
    _over_threshold = False
    _full_completion: Any = None

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
    logger.debug(f"[setup +{_setup_elapsed()}] preflight repair start")
    with suppress(Exception):
        unreplied = find_unreplied_assistant_entries(client)
        if unreplied:
            # backfill for all such assistant messages (oldest → newest).
            # Each entry is repaired independently, inside its own
            # try/except: prune_over_quota_tool_calls can now raise (a
            # below-watermark mutation refused on a resumed client whose
            # watermark carried over), and a single blanket suppress around
            # the whole loop would silently abandon every entry after the
            # one that raised instead of just skipping it.
            for entry in unreplied:
                try:
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
                except Exception as exc:
                    logger.error(
                        f"Preflight repair failed for one assistant entry; "
                        f"continuing with the rest: {exc}",
                        prefix="🚨",
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
                _ctx_cont = (
                    args.get("_parent_chat_context_cont")
                    if isinstance(args, dict)
                    else None
                )
                if _ctx_cont is None:
                    # No continuation context to carry — keep forwarding the
                    # bare text exactly as before. Plenty of simple tools
                    # declare `_interject_queue` and just do
                    # `await _interject_queue.get()` expecting the raw
                    # string; wrapping unconditionally would break that
                    # contract for every interject that has nothing to do
                    # with context propagation.
                    await iq.put(new_text)
                else:
                    # Match AsyncToolLoopHandle.interject's own queue payload
                    # shape exactly (unify/common/async_tool_loop.py) only
                    # when there's actually context to carry, so a call
                    # routed through this queue shortcut carries the same
                    # continuation context as one routed through
                    # handle.interject() below — bypassing the handle must
                    # not silently drop it.
                    await iq.put(
                        {
                            "message": new_text,
                            "_parent_chat_context_continued": _ctx_cont,
                            "trigger_immediate_llm_turn": (
                                args.get("trigger_immediate_llm_turn", True)
                                if isinstance(args, dict)
                                else True
                            ),
                            "suppress_response_notification": (
                                args.get("suppress_response_notification", False)
                                if isinstance(args, dict)
                                else False
                            ),
                        },
                    )
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

        # Select targets via central policy
        targets: list[Tuple[asyncio.Task, ToolCallMetadata]] = _select_steering_targets(
            method,
            payload if isinstance(payload, dict) else {},
        )
        if not targets:
            return

        base = str(method or "").lower().strip()

        def _steer_payload_for(base_action: str) -> Optional[str]:
            if base_action == "interject":
                return payload.get("message") or payload.get("content")
            if base_action == "ask":
                return payload.get("question")
            if base_action == "stop":
                return payload.get("reason")
            if base_action == "clarify":
                return payload.get("answer")
            return None  # pause/resume carry no payload

        # Build one assistant message with one `steer` tool_call per target —
        # same structured-args shape the LLM itself would emit, so acking and
        # dispatching this programmatic steering path go through the exact
        # same `steer` schema/transcript convention, not a parallel one.
        tool_calls = []
        args_by_id: dict[str, Any] = {}
        for _t, inf in targets:
            try:
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

                steer_args: dict[str, Any] = {
                    "call_id": inf.call_id,
                    "action": base,
                }
                _pl = _steer_payload_for(base)
                if _pl is not None:
                    steer_args["payload"] = _pl

                call_id = f"mirror_{short_id(6)}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "steer",
                            "arguments": json.dumps(steer_args),
                        },
                    },
                )
                # Use full forward kwargs for dispatch
                args_by_id[call_id] = (forward_args, inf)
            except Exception:
                continue
        if not tool_calls:
            return

        # Append assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        await _msg_dispatcher.append_msgs([assistant_msg])
        with suppress(Exception):
            await to_event_bus(assistant_msg, cfg, kind=ToolLoopKind.STEERING_HELPER)
        assistant_meta[id(assistant_msg)] = {"results_count": 0}

        # Insert ack tool messages and forward steering immediately to target handles
        for call in tool_calls:
            try:
                cid = call.get("id")
                if not isinstance(cid, str):
                    continue
                args, inf = args_by_id.get(cid, (None, None))
                # Ack message
                with suppress(Exception):
                    await acknowledge_helper_call(  # type: ignore[name-defined]
                        assistant_msg,
                        cid,
                        "steer",
                        call["function"].get("arguments", "{}"),
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                # Forward steering to child handle or channels
                # Centralized steering dispatch (unless inject-only)
                if (not inject_only) and (inf is not None):
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

        # Coalesce-then-freeze into a [clarification <call_id>] tail message —
        # never the tool_reply_msg pending stub, which stays byte-frozen once
        # sent. The model answers off this tail message via clarify_<call_id>.
        await tools_data.record_clarification(
            tools_data.info[src_task],
            call_id,
            question_text,
            _msg_dispatcher,
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

        # Coalesce-then-freeze into a separate [progress <call_id>] tail
        # message — never the tool_reply_msg placeholder, which must stay
        # byte-frozen once sent. This is the site behind the observed
        # 0%-cache pair: rewriting the placeholder in place, mid-history,
        # broke the cached prefix on virtually every turn a sub-agent ran.
        await tools_data.record_progress(
            tools_data.info[src_task],
            call_id,
            pretty,
            _msg_dispatcher,
        )

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
    # Bounded retries for a terminal turn that returns empty content with no
    # substantive answer anywhere else in the conversation to fall back on.
    _empty_final_answer_retries = 0
    _MAX_EMPTY_FINAL_ANSWER_RETRIES = 1

    # Loop returns immediately upon the final assistant message (no persist mode)
    logger.debug(f"[setup +{_setup_elapsed()}] entering main loop")

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
            _suppress_persist_response = False
            _had_interjections = False
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                _is_sentinel = isinstance(extra, dict) and (
                    "_mirror" in extra
                    or "_transcript_note" in extra
                    or "_compact_transcript" in extra
                    or extra.get("_replay")
                )
                # Transcript-note sentinel: a background process (e.g. a
                # storage review) leaves a loop-authored note in the
                # transcript without granting an LLM turn — the model reads
                # it whenever it next speaks.
                if isinstance(extra, dict) and "_transcript_note" in extra:
                    try:
                        _note = str(
                            (extra.get("_transcript_note") or {}).get("text") or "",
                        )
                        if _note:
                            await _msg_dispatcher.append_msgs(
                                [loop_user_notice(_note)],
                            )
                    except Exception:
                        pass
                    continue
                # Transcript-compaction sentinel: a completed storage review
                # has consolidated the covered turns, so their raw tool
                # payloads shed their bulk. Processed only at drain points —
                # never mid-dispatch.
                if isinstance(extra, dict) and "_compact_transcript" in extra:
                    try:
                        _n = int(
                            (extra.get("_compact_transcript") or {}).get(
                                "reviewed_messages",
                            )
                            or 0,
                        )
                        if _n > 0:
                            compact_reviewed_messages(client, _n)
                    except Exception:
                        pass
                    continue
                if not _is_sentinel:
                    if not _had_interjections:
                        _had_interjections = True
                        _suppress_persist_response = True
                    if not isinstance(extra, dict) or not extra.get(
                        "suppress_response_notification",
                        False,
                    ):
                        _suppress_persist_response = False

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
                    # Tools that did not opt into context initially should not
                    # receive context continuations either.
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
                # Shared trigger with record_progress/record_clarification's own
                # call into the same method (same flag on tools_data), so a loop
                # that gets a real interjection before any status message still
                # only pays for one injection.
                await tools_data._ensure_visibility_guidance_injected(_msg_dispatcher)

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
                        loop_user_notice(ctx_cont_content, _ctx_header=True),
                    )
                # Only append user message if there's actual content
                if _msg_text:
                    _user_content = (
                        time_ctx.prefix_user_message(_msg_text)
                        if time_ctx is not None
                        else _msg_text
                    )
                    msgs_to_append.append(
                        {
                            "role": "user",
                            "_interjection": True,
                            "content": _user_content,
                        },
                    )
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
                _completed_tools = done & tools_data.pending
                if _completed_tools:
                    logger.debug(
                        f"⏱️ [ToolLoop] {len(_completed_tools)} tool task(s) completed, "
                        f"{len(tools_data.pending) - len(_completed_tools)} still pending",
                    )
                for task in _sort_completed_tasks_by_call_id(
                    _completed_tools,
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
            logger.debug(
                f"[setup +{_setup_elapsed()}] tool policy eval (step={runtime_state.step_index})",
            )
            # Eager policies (e.g. discovery-first gates) keep the model on a
            # narrow required tool subset.  Track that so we do not leak
            # compress_context into the schema as an escape hatch.
            _policy_eager = False
            if tool_policy is not None:
                _tools_snapshot = {n: s.fn for n, s in tools_data.normalized.items()}
                try:
                    if _policy_accepts_history:
                        _policy_result = tool_policy(
                            runtime_state.step_index,
                            _tools_snapshot,
                            list(runtime_state.called_tools),
                        )
                    else:
                        _policy_result = tool_policy(
                            runtime_state.step_index,
                            _tools_snapshot,
                        )
                    tool_choice_mode, filtered, _policy_eager = (
                        _parse_tool_policy_result(
                            _policy_result,
                        )
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {runtime_state.step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", _tools_snapshot
                    _policy_eager = False
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            # When tools are in-flight, force tool_choice=required so the LLM
            # must call a real tool (steer, wait, ask_about_completed_tool,
            # etc.) rather than ending the loop. The response tool stays in
            # the schema but is refused at execution time while anything is
            # pending (see the steer()/response-tool execution branches
            # below), so "required" still only leaves live options.
            _has_pending_tools = bool(tools_data.pending)
            if _has_pending_tools and tool_choice_mode != "required":
                tool_choice_mode = "required"

            logger.debug(
                f"[setup +{_setup_elapsed()}] building tool schemas ({len(policy_tools_norm)} tools)",
            )
            _compress_schema = (
                method_to_schema(compress_context, "compress_context")
                if enable_compression
                else None
            )

            if _over_threshold and enable_compression:
                if _has_pending_tools:
                    # Over threshold, pending tools → no base tools, no
                    # compress_context (can't compress mid-flight). Only the
                    # static surface (wait, steer, ask_about_completed_tool)
                    # remains visible.
                    visible_base_tools_schema = []
                    _threshold_msg = (
                        "Context window is nearly full. "
                        "You cannot start new tools. Wait for in-flight tools to complete and then call "
                        "`compress_context` to free up context."
                    )
                    if log_steps == "full":
                        logger.info(
                            f"Context over threshold (pending in-flight tools): {_threshold_msg}",
                            prefix=ICONS["summarize"],
                        )
                    await _msg_dispatcher.append_msgs(
                        [loop_user_notice(_threshold_msg)],
                    )
                else:
                    # Over threshold, no pending → compress_context plus
                    # any caller-specified extra compression tools (pulled
                    # from the full tool set so policy gates are bypassed).
                    visible_base_tools_schema = [_compress_schema]
                    if extra_compression_tools:
                        visible_base_tools_schema.extend(
                            method_to_schema(
                                spec.fn,
                                name,
                                expose_context_control=(
                                    propagate_chat_context
                                    == ChatContextPropagation.LLM_DECIDES
                                ),
                                has_parent_context=bool(parent_chat_context),
                            )
                            for name, spec in tools_data.normalized.items()
                            if name in extra_compression_tools
                        )
                    tool_choice_mode = "required"
                    _threshold_msg = (
                        "Context window is nearly full. "
                        "You must call `compress_context` now."
                    )
                    if log_steps == "full":
                        logger.info(
                            f"Context over threshold (no pending): {_threshold_msg}",
                            prefix=ICONS["summarize"],
                        )
                    await _msg_dispatcher.append_msgs(
                        [loop_user_notice(_threshold_msg)],
                    )
            else:
                # Schema constancy beats schema minimalism: tools stay visible
                # even while saturated on max_concurrent/max_total_calls —
                # a saturated call is refused at execution time instead (see
                # has_exceeded_concurrent_limit_for_tool / prune_over_quota_tool_calls),
                # so hitting the cap never changes what the model can see.
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
                ]
                # Keep compress_context out of eager gated turns so required
                # discovery/tool policies cannot be satisfied by compressing.
                # Forced over-threshold compression above is unchanged.
                if _compress_schema is not None and not _policy_eager:
                    visible_base_tools_schema.append(_compress_schema)

            # Inject the response-submission tool whenever response_format is
            # set — schema presence no longer depends on whether other tools
            # are in-flight (that used to mask it out and back in on every
            # pending<->idle transition, a prefix break each time).
            # This tool is semantically "end the current turn" (the
            # tool-call analogue of a bare text response). Calling it while
            # tools are still pending is refused at execution time instead —
            # the same schema-constant-but-execution-gated pattern already
            # used for concurrency/quota saturation and steer().
            #
            # Name varies by mode:
            #   persist=True  → "send_response"  (signals turn completion,
            #                    loop continues waiting for next interjection)
            #   persist=False → "final_response"  (terminates the loop)
            _response_tool_name = "send_response" if persist else "final_response"
            # "Ready" now means "present in the schema" (i.e. response_format
            # is configured and injection succeeded) — not "safe to call right
            # now"; whether it's actually safe is enforced by the pending-tools
            # refusal in the execution branch below, not by schema presence.
            _structured_response_tool_ready = False

            if _rf_norm is not None:
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
                    _answer_schema = _rf_norm.answer_json_schema

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": DEFAULT_TOOL_SCHEMA_STRICT,
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
                    _structured_response_tool_ready = True
                except Exception as _injection_exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to inject {_response_tool_name} tool: {_injection_exc!r}",
                    )

            # Only force tool use for structured output once the response tool
            # is actually available. Forcing required without final_response /
            # send_response creates an inescapable tool-call loop.
            if _structured_response_tool_ready and tool_choice_mode != "required":
                tool_choice_mode = "required"

            # Inject multi-handle `final_response` tool when coordinator is present.
            # This tool requires request_id to specify which request is being answered.
            # Unlike response_format mode, this is always available (tools may be shared).
            if multi_handle_coordinator is not None:
                visible_base_tools_schema.append(
                    {
                        "type": "function",
                        "strict": DEFAULT_TOOL_SCHEMA_STRICT,
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
                        "strict": DEFAULT_TOOL_SCHEMA_STRICT,
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
            logger.debug(f"[setup +{_setup_elapsed()}] yielding (asyncio.sleep(0))")
            await asyncio.sleep(0)
            logger.debug(f"[setup +{_setup_elapsed()}] resumed after yield")

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

            # Register callback to refresh capability bookkeeping (is_interjectable,
            # clarification queue wiring, live-ask closures) when a handle is
            # adopted mid-loop. No outer-visible tools are minted here anymore —
            # steer()/wait/ask_about_completed_tool are already static.
            def _refresh_helpers_for_task(task: asyncio.Task) -> None:
                with suppress(Exception):
                    dynamic_tool_factory._refresh_task_capabilities(task)

            tools_data._on_handle_adopted = _refresh_helpers_for_task

            # NOTE: `wait` is no longer hidden from the schema while a
            # clarification is pending — the interlock moved to execution
            # time (see the `lname_cf == "wait"` branch below), so `wait`
            # stays present and byte-stable every turn.

            # make sure every pending call already has a *tool* reply ──
            #  (a placeholder) before we let the assistant speak again.
            logger.debug(f"[setup +{_setup_elapsed()}] ensure_placeholders start")
            await ensure_placeholders_for_pending(
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
                time_ctx=time_ctx,
            )
            logger.debug(f"[setup +{_setup_elapsed()}] ensure_placeholders done")

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
            logger.debug(
                f"[setup +{_setup_elapsed()}] ready for LLM call (step={runtime_state.step_index}, {len(tmp_tools)} tools)",
            )
            if log_steps:
                logger.begin_thinking()

            await to_event_bus(
                {"role": "assistant", "_thinking_in_flight": True},
                cfg,
            )

            # Set only by patient mode below, to keep hold of the assistant
            # message this step produced.
            _patient_asst_msg: Optional[dict] = None

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
                    _gen_kwargs["parallel_tool_calls"] = max_parallel_tool_calls > 1
                elif _policy_eager:
                    # Discovery-first (and other eager gates) expose multiple
                    # required tools that must be callable in one assistant turn.
                    _gen_kwargs["parallel_tool_calls"] = True

                # The prompt this dispatch sends is about to be snapshotted
                # from the current transcript, so it provably contains every
                # result ingested so far — the obligation deferred_llm_turn
                # exists to enforce is satisfied by this dispatch alone.
                # Clearing here, not when the step completes, is what keeps
                # a result that lands *during* this same dispatch's flight
                # correctly deferred to the turn after it: the set sites run
                # after this point, so they still land after the clear.
                deferred_llm_turn = False

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

                if log_steps:
                    logger.emit_thinking_fallback()

                # Helper cleanup: cancel auxiliary waiters only.
                # NOTE: llm_task is deliberately NOT cancelled here. Each branch
                # below decides whether to cancel the LLM based on context:
                # - Tool finished → cancel LLM (needs new context), unless
                #   ``interrupt_llm_on_tool_completion`` is False
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
                if done & pending_snapshot:
                    logger.debug(
                        f"⏱️ [ToolLoop] tool(s) finished during LLM race: "
                        f"{len(done & pending_snapshot)} completed",
                    )
                    if not interrupt_llm_on_tool_completion:
                        # Patient mode: the reasoning step already sent its
                        # prompt to the provider, which bills it whether or not
                        # the answer is collected, so discarding it to re-ask
                        # with the tool result costs a whole step and buys only
                        # latency. Let it finish and be used instead.
                        #
                        # The results still have to be ingested here: leaving the
                        # task pending makes section F read it as work in flight,
                        # and ``cancel_pending_tasks`` drops it without
                        # processing, losing the very result this branch fired
                        # for. ``deferred_llm_turn`` then guarantees a further
                        # turn, so the model always sees these results before it
                        # can conclude; it reasons one step behind, never without.
                        #
                        # Order matters. The step is awaited first so its own
                        # assistant message can be captured, because ingesting a
                        # result whose placeholder is no longer at the tail
                        # appends a synthetic assistant/tool status pair, and the
                        # loop would otherwise mistake that pair's tool message
                        # for this step's turn.
                        deferred_llm_turn = True
                        await asyncio.gather(llm_task, return_exceptions=True)
                        _patient_asst_msg = client.messages[-1]
                        completed_snapshot = {
                            task for task in pending_snapshot if task.done()
                        }
                        for task in _sort_completed_tasks_by_call_id(
                            completed_snapshot,
                            tools_data,
                        ):
                            await tools_data.process_completed_task(
                                task=task,
                                consecutive_failures=consecutive_failures,
                                outer_handle_container=outer_handle_container,
                                assistant_meta=assistant_meta,
                                msg_dispatcher=_msg_dispatcher,
                            )
                    else:
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
                    # Cached-replay determinism: a read-only cache miss while
                    # tools are still in flight means the live run never
                    # consumed this step — a tool completion (or steering
                    # event) superseded it mid-call and the loop re-issued the
                    # turn with updated context, so no entry was ever
                    # recorded. Mirror that outcome here: discard the step and
                    # fall back to the tool-wait block, which grants a fresh
                    # turn once the superseding event lands. With nothing in
                    # flight the miss is genuinely fatal and propagates as
                    # before.
                    if _is_cache_miss_error(llm_task.exception()) and (
                        tools_data.pending
                    ):
                        llm_turn_required = False
                        continue
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

                _full_completion = llm_task.result()

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
                        _gen_kwargs["parallel_tool_calls"] = max_parallel_tool_calls > 1
                    elif _policy_eager:
                        _gen_kwargs["parallel_tool_calls"] = True

                    # See the matching comment at the interrupt-mode dispatch
                    # above: clearing here (not at step completion) means the
                    # prompt this dispatch is about to snapshot provably
                    # contains everything ingested so far.
                    deferred_llm_turn = False

                    _full_completion = await generate_with_preprocess(
                        client,
                        _apply_reasoning_model_compat(_gen_kwargs, tool_choice_mode),
                        **_gen_kwargs,
                    )
                    if log_steps:
                        logger.emit_thinking_fallback()
                except Exception as e:
                    raise Exception(
                        f"LLM call failed: {type(e).__name__}: {e}",
                    ) from e

            # Normally the step's assistant message is the tail. Patient mode
            # ingests tool results after it lands, which can append a synthetic
            # status pair on top, so it captures the message itself.
            msg = (
                _patient_asst_msg
                if _patient_asst_msg is not None
                else client.messages[-1]
            )
            await to_event_bus(msg, cfg)

            # Update context threshold from the LLM response usage data.
            if enable_compression:
                with suppress(Exception):
                    _usage = getattr(_full_completion, "usage", None)
                    if (
                        _usage
                        and getattr(_usage, "prompt_tokens", None)
                        and _max_input_tokens
                    ):
                        _over_threshold = context_over_threshold(
                            _usage.prompt_tokens,
                            0.7,
                            _max_input_tokens,
                        )

            # LLM responded - reset the activity-based timeout. The timeout is
            # designed to catch hung tools, not slow LLM inference. LLM providers
            # have their own timeout mechanisms; our timeout only guards against
            # user-defined tools that may hang indefinitely.
            timer.reset()

            if log_steps:
                with suppress(Exception):
                    from .utils import format_llm_response_for_log

                    logger.info(
                        format_llm_response_for_log(msg),
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
            runtime_state.step_index += 1

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
            _persist_response_emitted = False
            _persist_response_content = None  # captured by send_response for surfacing

            if msg["tool_calls"]:
                # Both mutations below edit msg["tool_calls"] in place — safe
                # only while msg is still mutable (an edit below the sent
                # watermark would mutate already-dispatched bytes). msg is
                # this turn's own freshly-generated message (index ==
                # watermark, nothing dispatched it yet), so this is expected
                # to always hold; checked explicitly, up front, so the
                # invariant is stated rather than accidental and doesn't
                # depend on which mutation happens to run first.
                if not is_mutable(client, msg):
                    logger.error(
                        "persist-mode tool_calls pruning: msg is already "
                        "below the sent watermark; an in-place edit would "
                        "mutate already-dispatched bytes.",
                        prefix="🚨",
                    )
                    raise ValueError(
                        "persist-mode tool_calls pruning: msg is already "
                        "below the sent watermark; an in-place edit would "
                        "mutate already-dispatched bytes.",
                    )

                # ── De-duplicate tool calls (optional) ────────────────────────
                # Runs before quota pruning (restored original order): quota
                # accounting should count unique calls, not raw duplicate
                # occurrences — a tool called identically 3x against a
                # max_total_calls=2 limit should spend 1 unit of quota, not 3.
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
                    sys_notice = loop_user_notice(
                        "System notification: The tool calls in your last response "
                        "were blocked due to quota limits. Please modify your plan "
                        "or conclude.",
                    )
                    await _msg_dispatcher.append_msgs([sys_notice])

                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    runtime_state.called_tools.append(name)

                    # Parse arguments - handle both string and dict formats.
                    #
                    # A model can emit arguments that are not valid JSON — most
                    # often truncated, because generation degenerated and ran to
                    # the output-token cap mid-object. That is a recoverable
                    # event for this one call, so surface it back to the model
                    # the same way an unavailable tool is (below) instead of
                    # letting one bad call abort the whole turn. Repetition is
                    # what ends the loop, via the refusal tally.
                    _raw_args = call["function"]["arguments"]
                    if isinstance(_raw_args, str):
                        try:
                            args = json.loads(_raw_args)
                        except ValueError as exc:
                            logger.error(
                                "Malformed tool-call arguments for %s (%d chars): %s",
                                name,
                                len(_raw_args),
                                exc,
                            )
                            refusal = (
                                f"⚠️ Error: the arguments for '{name}' were not "
                                f"valid JSON ({exc}). They may have been cut off "
                                "mid-object. Re-issue the call with complete, "
                                "well-formed JSON arguments, keeping each value "
                                "in the type the target expects."
                            )
                            consecutive_failures.note_refusal(
                                tool_name=name,
                                args=_raw_args,
                                message=refusal,
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                create_tool_call_message(
                                    name=name,
                                    call_id=call["id"],
                                    content=refusal,
                                ),
                                client,
                                _msg_dispatcher,
                            )
                            stop_reason = consecutive_failures.stop_reason()
                            if stop_reason:
                                raise RuntimeError(stop_reason)
                            continue
                    else:
                        args = _raw_args if isinstance(_raw_args, dict) else {}

                    # Special-case: handle response-submission tool
                    # (send_response in persist mode, final_response otherwise)
                    _is_response_tool = (
                        name in ("final_response", "send_response")
                        and _rf_norm is not None
                    )
                    if _is_response_tool:
                        if tools_data.pending:
                            # Execution-time refusal (schema presence is now
                            # unconditional — see the injection comment above).
                            # Name the exits explicitly: this fires under
                            # tool_choice="required" (has_pending_tools forces
                            # it), so a refusal with no way out would just be
                            # the discovery-gate retry-loop shape again.
                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=(
                                    f"⚠️ Cannot call '{name}': "
                                    f"{len(tools_data.pending)} tool call(s) still "
                                    "running. Call `wait` to let them finish, or "
                                    'steer(call_id=<id>, action="stop") one of '
                                    "them if it's no longer needed — do not retry "
                                    f"'{name}' until nothing is pending."
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
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            # Validate payload against the normalized schema /
                            # Pydantic model (JSON Schema dicts included).
                            validated_payload = _rf_norm.validate(payload)
                            if isinstance(validated_payload, BaseModel):
                                payload_for_return = validated_payload.model_dump(
                                    mode="json",
                                )
                            else:
                                payload_for_return = validated_payload

                            tool_msg = create_tool_call_message(
                                name=name,
                                call_id=call["id"],
                                content=_dumps(payload_for_return, indent=4),
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
                                _persist_response_content = json.dumps(
                                    payload_for_return,
                                )
                                break  # exit the for-loop over tool_calls
                            return json.dumps(payload_for_return)
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
                        and _rf_norm is None
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

                    # ── Special-case: compress_context ────────────────────
                    if name == "compress_context":
                        tool_msg = create_tool_call_message(
                            name=name,
                            call_id=call["id"],
                            content=(
                                "Compression initiated. Ending current loop "
                                "to restart with compressed context."
                            ),
                        )
                        await insert_tool_message_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        return _COMPRESSION_SIGNAL

                    # ── Special-case dynamic helpers ──────────────────────
                    # • wait  → acknowledge, list running tasks, no scheduling
                    # • steer → structured-args dispatch (stop/interject/pause/
                    #           resume/clarify/call/ask), see below
                    # Normalise tool-call name defensively
                    lname = str(name or "").strip()
                    lname_cf = lname.casefold()

                    if lname_cf == "wait" and any(
                        getattr(_inf, "waiting_for_clarification", False)
                        for _inf in tools_data.info.values()
                    ):
                        # Wait interlock (execution-time, per the stable-schema
                        # design): `wait` is always in the schema now, so the
                        # deadlock guard that used to hide it from the schema
                        # while a clarification is pending moves here instead.
                        _pending_clar_ids = [
                            _inf.call_id
                            for _inf in tools_data.info.values()
                            if getattr(_inf, "waiting_for_clarification", False)
                        ]
                        tool_msg = create_tool_call_message(
                            name="wait",
                            call_id=call["id"],
                            content=(
                                "⚠️ Refused: a clarification is pending on "
                                f"{_pending_clar_ids} — answer it via "
                                'steer(call_id=<id>, action="clarify", payload=<answer>) '
                                "before waiting."
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

                                await _prune_wait(
                                    msg,
                                    call["id"],
                                    client=client,
                                    assistant_meta=assistant_meta,
                                    msg_dispatcher=_msg_dispatcher,
                                )

                            # The assistant message containing this wait() was
                            # already published to EventBus before we could
                            # inspect it.  Emit a matching tool result so the
                            # frontend can resolve the pending tool-call row.
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        "wait",
                                        call["id"],
                                        "",
                                    ),
                                    cfg,
                                )

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

                    elif lname_cf == "steer":
                        # ── Unified steering dispatcher: structured-args routing
                        # keyed on (call_id, action) instead of name prefixes. ──
                        # `args` was already parsed (with malformed-JSON refusal
                        # already handled) earlier in this loop iteration — reuse
                        # it directly rather than re-parsing raw arguments here.
                        _target_call_id = args.get("call_id")
                        _action = str(args.get("action") or "").strip().lower()
                        _payload = args.get("payload")
                        _method = args.get("method")

                        async def _steer_reply(content: str) -> None:
                            _tm = create_tool_call_message(
                                name="steer",
                                call_id=call["id"],
                                content=content,
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                _tm,
                                client,
                                _msg_dispatcher,
                            )

                        if not isinstance(_target_call_id, str) or not _target_call_id:
                            await _steer_reply(
                                "⚠️ steer() requires a string `call_id` identifying "
                                "which call to steer.",
                            )
                            continue

                        tgt_task, tgt_info = tools_data.resolve_call_id(
                            _target_call_id,
                        )

                        if tgt_task is None or tgt_info is None:
                            # Not live — distinguish "already completed" from
                            # "never existed" so the model can self-correct.
                            _completed_name = tools_data._completed_tool_names.get(
                                _target_call_id,
                            )
                            if _completed_name is not None:
                                if _action == "ask":
                                    _err = (
                                        f"call_id={_target_call_id!r} "
                                        f"({_completed_name}) has already completed "
                                        "and is no longer live. Use "
                                        f'ask_about_completed_tool(tool_id="{_target_call_id}", '
                                        "question=...) instead."
                                    )
                                else:
                                    _err = (
                                        f"Cannot {_action or '<missing action>'} "
                                        f"call_id={_target_call_id!r} ({_completed_name}): "
                                        "it has already completed."
                                    )
                            else:
                                _err = (
                                    f"No live call found for call_id={_target_call_id!r}. "
                                    "It may never have existed, or is mistyped — check "
                                    "the call_id shown on the original tool call or on its "
                                    "[steerable ...]/[progress ...]/[clarification ...] "
                                    "tail messages."
                                )
                            await _steer_reply(f"⚠️ {_err}")
                            continue

                        _handle = tgt_info.handle
                        _orig_fn = tgt_info.name
                        _orig_arg_json = tgt_info.call_dict["function"]["arguments"]
                        _pretty_name = (
                            f"steer:{_action or '?'} {_orig_fn}({_orig_arg_json})"
                        )

                        if _action == "stop":
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "stop",
                                    {"reason": _payload} if _payload else {},
                                    tgt_info,
                                )
                            if not tgt_task.done():
                                tgt_task.cancel()
                            tools_data.pop_task(tgt_task)
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f"The call [{_target_call_id}] has been stopped successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        _orig_fn,
                                        _target_call_id,
                                        json.dumps({"status": "stopped"}),
                                    ),
                                    cfg,
                                )
                            continue

                        elif _action == "interject":
                            if not tgt_info.is_interjectable:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) does "
                                    "not accept interjections.",
                                )
                                continue

                            # Restore continuation-context propagation the old
                            # minted interject_<fn>_<id> tool provided: forward
                            # _parent_chat_context_cont when the target's own
                            # interject() accepts it and it originally opted in
                            # to context. Reuses steer's include_parent_context
                            # field as the same opt-out the old per-call tool's
                            # include_parent_chat_context_cont control was.
                            _interject_accepts_ctx_cont = False
                            if _handle is not None and hasattr(_handle, "interject"):
                                with suppress(Exception):
                                    _ij_sig = inspect.signature(_handle.interject)
                                    _ij_has_varkw = any(
                                        p.kind == inspect.Parameter.VAR_KEYWORD
                                        for p in _ij_sig.parameters.values()
                                    )
                                    _interject_accepts_ctx_cont = (
                                        "_parent_chat_context_cont"
                                        in _ij_sig.parameters
                                        or _ij_has_varkw
                                    )

                            _interject_extra_kwargs, _ = compute_context_injection(
                                args={
                                    "include_parent_chat_context_cont": args.get(
                                        "include_parent_context",
                                        True,
                                    ),
                                },
                                propagate_chat_context=propagate_chat_context,
                                context_state=context_state,
                                client_messages=client.messages,
                                call_id=f"interject_{_target_call_id}_{call['id']}",
                                accepts_parent_ctx=False,
                                accepts_parent_ctx_cont=_interject_accepts_ctx_cont,
                                target_context_opted_in=tgt_info.context_opted_in,
                                is_continuation_only=True,
                            )
                            _interject_payload: Dict[str, Any] = {"content": _payload}
                            if "_parent_chat_context_cont" in _interject_extra_kwargs:
                                _interject_payload["_parent_chat_context_cont"] = (
                                    _interject_extra_kwargs["_parent_chat_context_cont"]
                                )

                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "interject",
                                    _interject_payload,
                                    tgt_info,
                                )
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f'Guidance "{_payload}" forwarded to the running tool.',
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        elif _action in ("pause", "resume"):
                            _cap = (
                                _handle is not None
                                and hasattr(
                                    _handle,
                                    "pause" if _action == "pause" else "resume",
                                )
                            ) or (tgt_info.pause_event is not None)
                            if not _cap:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) cannot "
                                    f"be {_action}d.",
                                )
                                continue
                            _paused_state = (
                                get_handle_paused_state(_handle)
                                if _handle is not None
                                else None
                            )
                            if (
                                _paused_state is None
                                and tgt_info.pause_event is not None
                                and hasattr(tgt_info.pause_event, "is_set")
                            ):
                                with suppress(Exception):
                                    _paused_state = not tgt_info.pause_event.is_set()
                            if _action == "pause" and _paused_state is True:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) is "
                                    "already paused.",
                                )
                                continue
                            if _action == "resume" and _paused_state is not True:
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) is not "
                                    "currently paused.",
                                )
                                continue
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    _action,
                                    {},
                                    tgt_info,
                                )
                            _past = "paused" if _action == "pause" else "resumed"
                            tool_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=f"The call [{_target_call_id}] has been {_past} successfully.",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            with suppress(Exception):
                                await to_event_bus(
                                    create_tool_call_message(
                                        _orig_fn,
                                        _target_call_id,
                                        json.dumps({"status": _past}),
                                    ),
                                    cfg,
                                )
                            continue

                        elif _action == "clarify":
                            if tgt_info.clar_up_queue is None or not getattr(
                                tgt_info,
                                "waiting_for_clarification",
                                False,
                            ):
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) has no "
                                    "pending clarification to answer right now.",
                                )
                                continue
                            with suppress(Exception):
                                await _dispatch_steering_to_child(
                                    "clarify",
                                    {"answer": _payload},
                                    tgt_info,
                                )
                                tgt_info.waiting_for_clarification = False
                            tool_reply_msg = create_tool_call_message(
                                name=_pretty_name,
                                call_id=call["id"],
                                content=(
                                    f"Clarification answer sent upstream: {_payload!r}\n"
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
                            # Store the reply so the tool's eventual final result
                            # lands here (not the tool_reply_msg pending stub, and
                            # not the [clarification <call_id>] tail message that
                            # carried the question — see record_clarification).
                            tgt_info.clarify_placeholder = tool_reply_msg
                            continue

                        elif _action == "ask":
                            if _handle is None or not hasattr(_handle, "ask"):
                                await _steer_reply(
                                    f"⚠️ call_id={_target_call_id!r} ({_orig_fn}) has no "
                                    "ask capability.",
                                )
                                continue

                            _ask_extra_kwargs, _ask_ctx_opted_in = (
                                compute_context_injection(
                                    args={
                                        "include_parent_chat_context": args.get(
                                            "include_parent_context",
                                            False,
                                        ),
                                    },
                                    propagate_chat_context=propagate_chat_context,
                                    context_state=context_state,
                                    client_messages=client.messages,
                                    call_id=f"ask_{_target_call_id}_{call['id']}",
                                    accepts_parent_ctx=True,
                                    accepts_parent_ctx_cont=False,
                                    is_continuation_only=False,
                                )
                            )
                            _ask_kwargs: Dict[str, Any] = {"question": _payload}
                            if "_parent_chat_context" in _ask_extra_kwargs:
                                _ask_kwargs["_parent_chat_context"] = _ask_extra_kwargs[
                                    "_parent_chat_context"
                                ]

                            async def _do_ask(_h=_handle, _kw=_ask_kwargs):
                                return await forward_handle_call(
                                    _h,
                                    "ask",
                                    _kw,
                                    fallback_positional_keys=["question"],
                                )

                            _steer_call_dict = {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": "steer",
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                            _t = asyncio.create_task(
                                _do_ask(),
                                name="ToolCall_steer_ask",
                            )
                            tools_data.save_task(
                                _t,
                                ToolCallMetadata(
                                    name=f"{_orig_fn}.ask",
                                    call_id=call["id"],
                                    assistant_msg=msg,
                                    call_dict=_steer_call_dict,
                                    call_idx=idx,
                                    is_interjectable=False,
                                    is_dynamic=True,
                                    chat_context=_ask_extra_kwargs.get(
                                        "_parent_chat_context",
                                    ),
                                    pause_event=None,
                                    tool_schema={
                                        "type": "function",
                                        "function": {"name": "steer"},
                                    },
                                    llm_arguments=_ask_kwargs,
                                    raw_arguments_json=(_payload or ""),
                                    context_opted_in=_ask_ctx_opted_in,
                                ),
                            )
                            continue

                        elif _action == "call":
                            if not _method:
                                await _steer_reply(
                                    '⚠️ action="call" requires a `method` name.',
                                )
                                continue
                            _custom_methods = (
                                DynamicToolFactory._discover_custom_public_methods(
                                    _handle,
                                )
                                if _handle is not None
                                else {}
                            )
                            if _method not in _custom_methods:
                                await _steer_reply(
                                    f"⚠️ No custom method {_method!r} on "
                                    f"call_id={_target_call_id!r} ({_orig_fn}). "
                                    f"Available: {sorted(_custom_methods)}",
                                )
                                continue

                            _bound = _custom_methods[_method]
                            try:
                                _parsed_payload = (
                                    json.loads(_payload) if _payload else {}
                                )
                                if not isinstance(_parsed_payload, dict):
                                    raise ValueError(
                                        "payload must be a JSON object string",
                                    )
                                inspect.signature(_bound).bind(**_parsed_payload)
                            except Exception as _val_exc:
                                await _steer_reply(
                                    f"⚠️ Invalid payload for method={_method!r}: "
                                    f"{_val_exc}. Expected a JSON object string "
                                    f"matching signature {inspect.signature(_bound)}.",
                                )
                                continue

                            _write_only = set(
                                getattr(_handle, "write_only_methods", None) or [],
                            ) | set(
                                getattr(_handle, "write_only_tools", None) or [],
                            )

                            async def _invoke_custom(
                                _m=_method,
                                _h=_handle,
                                _kw=_parsed_payload,
                            ):
                                return await forward_handle_call(_h, _m, _kw)

                            if _method in _write_only:
                                tool_msg = create_tool_call_message(
                                    name=_pretty_name,
                                    call_id=call["id"],
                                    content=(
                                        f"Operation {_method!r} acknowledged and "
                                        "forwarded to the running tool."
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
                                    asyncio.create_task(
                                        _invoke_custom(),
                                        name=f"ToolCall_steer_call_{_method}",
                                    )
                                continue

                            _steer_call_dict = {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": "steer",
                                    "arguments": call["function"]["arguments"],
                                },
                            }
                            _t = asyncio.create_task(
                                _invoke_custom(),
                                name=f"ToolCall_steer_call_{_method}",
                            )
                            tools_data.save_task(
                                _t,
                                ToolCallMetadata(
                                    name=f"{_orig_fn}.{_method}",
                                    call_id=call["id"],
                                    assistant_msg=msg,
                                    call_dict=_steer_call_dict,
                                    call_idx=idx,
                                    is_interjectable=False,
                                    is_dynamic=True,
                                    chat_context=None,
                                    pause_event=None,
                                    tool_schema={
                                        "type": "function",
                                        "function": {"name": "steer"},
                                    },
                                    llm_arguments=_parsed_payload,
                                    raw_arguments_json=(_payload or "{}"),
                                    context_opted_in=False,
                                ),
                            )
                            continue

                        else:
                            await _steer_reply(
                                f"⚠️ Unknown action {_action!r}. Valid actions: "
                                "stop, interject, pause, resume, clarify, call, ask.",
                            )
                            continue

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
                                "already reached — at capacity. Use `wait` for an "
                                "existing call to finish, or "
                                'steer(call_id=<id>, action="stop") one before retrying.'
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

                    elif lname_cf == "ask_about_completed_tool":
                        # ── Frozen-docstring dispatcher for completed tools ──
                        # Ids arrive via appended "[askable <call_id>]" tail
                        # messages (ToolsData.record_tool_completed_askable)
                        # instead of a live listing baked into the docstring.
                        _tool_id = (
                            args.get("tool_id") if isinstance(args, dict) else None
                        )
                        _question = (
                            args.get("question") if isinstance(args, dict) else None
                        )

                        _entry = (
                            tools_data._completed_askable_tools.get(_tool_id)
                            if isinstance(_tool_id, str)
                            else None
                        )
                        if _entry is None:
                            _completed_name = (
                                tools_data._completed_tool_names.get(_tool_id)
                                if isinstance(_tool_id, str)
                                else None
                            )
                            if _completed_name is not None:
                                _err = (
                                    f"Cannot ask about tool_id={_tool_id!r} "
                                    f"({_completed_name}). This tool completed "
                                    "successfully but was not steerable — it "
                                    "executed as a direct function call with no "
                                    "inner reasoning trajectory to inspect. Its "
                                    "result is already visible in the outer "
                                    "transcript above."
                                )
                            else:
                                _available = list(
                                    tools_data._completed_askable_tools.keys(),
                                )
                                _err = (
                                    f"No tool found with tool_id={_tool_id!r}. "
                                    "This ID does not match any completed tool "
                                    "call. Available tool_ids for retrospective "
                                    f"inspection: {_available}"
                                )
                            tool_msg = create_tool_call_message(
                                name="ask_about_completed_tool",
                                call_id=call["id"],
                                content=f"⚠️ {_err}",
                            )
                            await insert_tool_message_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                        _completed_handle = _entry.get("handle")
                        _aact_extra_kwargs, _aact_ctx_opted_in = (
                            compute_context_injection(
                                args={},
                                propagate_chat_context=propagate_chat_context,
                                context_state=context_state,
                                client_messages=client.messages,
                                call_id=f"ask_{_tool_id}_{call['id']}",
                                accepts_parent_ctx=True,
                                accepts_parent_ctx_cont=False,
                                is_continuation_only=False,
                            )
                        )
                        _aact_kwargs: Dict[str, Any] = {"question": _question}
                        if "_parent_chat_context" in _aact_extra_kwargs:
                            _aact_kwargs["_parent_chat_context"] = _aact_extra_kwargs[
                                "_parent_chat_context"
                            ]

                        async def _do_completed_ask(
                            _h=_completed_handle,
                            _kw=_aact_kwargs,
                        ):
                            return await forward_handle_call(
                                _h,
                                "ask",
                                _kw,
                                fallback_positional_keys=["question"],
                            )

                        _aact_call_dict = {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": "ask_about_completed_tool",
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        _t = asyncio.create_task(
                            _do_completed_ask(),
                            name="ToolCall_ask_about_completed_tool",
                        )
                        tools_data.save_task(
                            _t,
                            ToolCallMetadata(
                                name=f"{_entry.get('name', '?')}.ask",
                                call_id=call["id"],
                                assistant_msg=msg,
                                call_dict=_aact_call_dict,
                                call_idx=idx,
                                is_interjectable=False,
                                is_dynamic=True,
                                chat_context=_aact_extra_kwargs.get(
                                    "_parent_chat_context",
                                ),
                                pause_event=None,
                                tool_schema={
                                    "type": "function",
                                    "function": {"name": "ask_about_completed_tool"},
                                },
                                llm_arguments=_aact_kwargs,
                                raw_arguments_json=call["function"]["arguments"],
                                context_opted_in=_aact_ctx_opted_in,
                            ),
                        )
                        continue

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
                        msg_dispatcher=_msg_dispatcher,
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

                    # Eager tool policies: if gates are still unsatisfied after
                    # the calls just scheduled, grant another LLM turn now
                    # (overlapping in-flight tools) instead of waiting for
                    # results.  Re-evaluate with the updated called_tools so
                    # eagerness ends as soon as the policy stops requesting it.
                    # Only re-invoke when this turn was already eager — non-eager
                    # policies must not get an extra same-step callback.
                    if tool_policy is not None and _policy_eager:
                        try:
                            _eager_snapshot = {
                                n: s.fn for n, s in tools_data.normalized.items()
                            }
                            if _policy_accepts_history:
                                _eager_result = tool_policy(
                                    runtime_state.step_index,
                                    _eager_snapshot,
                                    list(runtime_state.called_tools),
                                )
                            else:
                                _eager_result = tool_policy(
                                    runtime_state.step_index,
                                    _eager_snapshot,
                                )
                            _, _, _still_eager = _parse_tool_policy_result(
                                _eager_result,
                            )
                            if _still_eager:
                                llm_turn_required = True
                        except Exception as _eager_exc:
                            logger.error(
                                f"tool_policy eager re-check failed: {_eager_exc!r}",
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

            final_content = extract_substantive_text(msg["content"])

            # An empty/null/whitespace-only terminal turn must never override
            # a substantive answer already sitting in the transcript — a
            # model that has nothing left to add after answering can still
            # return empty content on a later turn, and that must not erase
            # the answer. Multi-handle and the plain return read
            # final_content after this point, so resolving it once here
            # covers both. Persist mode is exempt: it never finalizes here —
            # an empty turn surfaces nothing and re-enters the persist wait,
            # and with response_format the turn's answer is the
            # response-tool payload rather than text content, so the
            # nudge/loud-fail below would inject spurious turns and then
            # terminate a loop that only an explicit stop may end.
            if final_content is None and not persist:
                _substantive_content = None
                for _hist_msg in reversed(client.messages):
                    _hist_role = _hist_msg.get("role")
                    if _hist_role == "user" and not is_loop_authored_message(_hist_msg):
                        # A genuine user turn boundary (not a loop-authored
                        # status message) — don't reach past it into an
                        # earlier request/interjection cycle for an answer
                        # that belongs to a different question.
                        break
                    if _hist_role != "assistant":
                        continue
                    _hist_content = extract_substantive_text(_hist_msg.get("content"))
                    if _hist_content is not None:
                        _substantive_content = _hist_content
                        break

                if _substantive_content is not None:
                    final_content = _substantive_content
                elif _empty_final_answer_retries < _MAX_EMPTY_FINAL_ANSWER_RETRIES:
                    # No substantive answer exists anywhere in this cycle
                    # either — give the model a bounded number of chances to
                    # produce one before giving up loudly. Appended at the
                    # tail; nothing already dispatched is touched. Marked
                    # loop-authored so it can never masquerade as a genuine
                    # user turn boundary on the retry pass above.
                    _empty_final_answer_retries += 1
                    await _msg_dispatcher.append_msgs(
                        [
                            loop_user_notice(
                                "Produce your final answer as text.",
                                _nudge_msg=True,
                            ),
                        ],
                    )
                    continue
                else:
                    # Retries exhausted and nothing substantive was ever
                    # produced — fail loudly rather than return an empty
                    # result silently.
                    notice = {
                        "role": "assistant",
                        "content": (
                            "No final answer was produced: the model returned "
                            "empty content after "
                            f"{_MAX_EMPTY_FINAL_ANSWER_RETRIES} nudge attempt(s), "
                            "with no substantive answer anywhere in this "
                            "conversation."
                        ),
                    }
                    await _msg_dispatcher.append_msgs([notice])
                    logger.error(
                        "Empty final answer after "
                        f"{_MAX_EMPTY_FINAL_ANSWER_RETRIES} nudge attempt(s); no "
                        "substantive assistant content exists in this conversation.",
                        prefix=ICONS["llm_error"],
                    )
                    return notice["content"]

            # ── multi-handle mode: check if all requests are done ──
            if multi_handle_coordinator is not None:
                if multi_handle_coordinator.should_terminate():
                    # All requests completed/cancelled and persist=False
                    logger.info(
                        "Multi-handle mode: all requests completed, terminating loop.",
                        prefix=ICONS["completed"],
                    )
                    multi_handle_coordinator.close()
                    # final_content is resolved above: the last assistant
                    # content, or a substantive earlier answer if this turn's
                    # own content was empty.
                    return final_content
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
                    and not _suppress_persist_response
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

                # A parked turn's chain of thought is never consulted again:
                # the next dispatch starts from a fresh user interjection, so
                # provider reasoning payloads (encrypted blobs, reasoning
                # summaries) on every completed assistant message are pure
                # re-billed bulk from here on. Shed them now rather than
                # waiting for a storage review to cover the span — reviews
                # lag turns, and the lag is paid on every call in between.
                try:
                    _shed = 0
                    for _m in client.messages or []:
                        if isinstance(_m, dict) and _m.get("role") == "assistant":
                            _shed += strip_reasoning_payloads(_m)
                    if _shed:
                        _rebaseline_watermark_hash(client)
                except Exception:
                    pass

                logger.info(
                    "Persist mode: waiting for next interjection...",
                    prefix=ICONS["pause"],
                )
                try:
                    from ...events.manager_event_logging import (
                        publish_persist_session_phase,
                    )

                    await publish_persist_session_phase(_outer, "awaiting_input")
                except Exception:
                    pass
                while True:
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
                    for p in pending:
                        p.cancel()
                        await asyncio.gather(p, return_exceptions=True)

                    if cancel_event.is_set():
                        raise asyncio.CancelledError

                    if interject_waiter not in done:
                        continue

                    interjection = interject_waiter.result()

                    # Transcript-note sentinels are transcript-only: append the
                    # loop-authored note and stay in persist wait. The model
                    # reads it on its next granted turn.
                    if (
                        isinstance(interjection, dict)
                        and "_transcript_note" in interjection
                    ):
                        try:
                            _note = str(
                                (interjection.get("_transcript_note") or {}).get(
                                    "text",
                                )
                                or "",
                            )
                            if _note:
                                await _msg_dispatcher.append_msgs(
                                    [loop_user_notice(_note)],
                                )
                        except Exception:
                            pass
                        continue

                    # Transcript-compaction sentinels: the covered turns were
                    # consolidated by a storage review; shed their raw tool
                    # payloads and stay in persist wait.
                    if (
                        isinstance(interjection, dict)
                        and "_compact_transcript" in interjection
                    ):
                        try:
                            _n = int(
                                (interjection.get("_compact_transcript") or {}).get(
                                    "reviewed_messages",
                                )
                                or 0,
                            )
                            if _n > 0:
                                compact_reviewed_messages(client, _n)
                        except Exception:
                            pass
                        continue

                    # Mirror sentinels are transcript-only (no user message).
                    # Process them in-place and stay in persist wait — resuming
                    # the full loop would trigger an LLM call with a trailing
                    # assistant message, which strict models reject.
                    if isinstance(interjection, dict) and "_mirror" in interjection:
                        try:
                            _ms = interjection.get("_mirror") or {}
                            _m = _ms.get("method")
                            _kw = _ms.get("kwargs") or {}
                            if isinstance(_m, str) and _m:
                                merged = dict(_kw if isinstance(_kw, dict) else {})
                                for _key in ("_custom", "_aliases", "_fallback"):
                                    if _key in _ms:
                                        merged[_key] = _ms[_key]
                                await _synthesize_mirrored_helper_calls(_m, merged)
                        except Exception:
                            pass
                        continue

                    # Real interjection — put it back for normal processing
                    try:
                        await interject_queue.put(interjection)
                        logger.info(
                            "Persist mode: interjection received, resuming loop",
                            prefix=ICONS["resume"],
                        )
                        from ...events.manager_event_logging import (
                            publish_persist_session_phase,
                        )

                        await publish_persist_session_phase(_outer, "resumed")
                    except Exception:
                        pass
                    break

                # Reset timer for the new "turn"
                timer.reset()
                continue  # Back to top of loop to process the interjection

            # final_content was already resolved to non-empty content (or the
            # function returned earlier with a loud error) above.
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
