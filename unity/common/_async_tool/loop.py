import asyncio
import unify
import json
import inspect
import copy

from typing import Dict, Union, Callable, Tuple, Any, Set, Optional
from contextlib import suppress
from pydantic import BaseModel

from ...constants import LOGGER
from ..tool_spec import ToolSpec, normalise_tools
from .utils import maybe_await
from .event_bus_util import to_event_bus
from .messages import (
    find_unreplied_assistant_entries,
    chat_context_repr,
    generate_with_preprocess,
)
from .message_dispatcher import LoopMessageDispatcher
from .tools_utils import (
    ToolCallMetadata,
    create_tool_call_message,
)
from .images import (
    set_live_images_context,
    reset_live_images_context,
    align_images_for as _align_images_for,
    build_live_image_tools,
    refresh_overview_doc_if_present,
    append_source_scoped_images_with_text,
)
from ..llm_helpers import method_to_schema, _dumps
from .loop_config import (
    LoopConfig,
    TOOL_LOOP_LINEAGE,
)
from .timeout_timer import TimeoutTimer
from .messages import (
    insert_tool_message_after_assistant,
    ensure_placeholders_for_pending,
    propagate_stop_once,
    forward_handle_call,
    schedule_missing_for_message,
    build_helper_ack_content,
)
from .tools_data import ToolsData
from .dynamic_tools_factory import DynamicToolFactory
from . import semantic_cache as sc


class LoopLogger:
    def __init__(self, cfg: LoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps

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
    client: unify.AsyncUnify,
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
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    semantic_cache: Optional[bool] = False,
    images: Optional[dict[str, Any]] = None,
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
    client : ``unify.AsyncUnify``
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
        docstring – these are automatically converted to an OpenAI *tool
        schema* via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str | dict]``
        Thread-safe channel through which the *outer* application can push
        additional user turns at any time (e.g. the human changes their
        mind mid-generation). When a dict is provided it should follow the
        shape {"message": str, "parent_chat_context_continuted": list[dict]}.

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
        ``propagate_chat_context`` is on, the helper
        :pyfunc:`_chat_context_repr` merges this with the current
        ``client.messages`` and forwards the result downward.

    log_steps : ``bool | str``, default ``True``
        Controls verbosity of step logging to ``LOGGER``:
          • ``False`` – no logging
          • ``True``  – log everything except system messages
          • ``"full"`` – log everything including system messages

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """
    # unique id / lineage
    cfg = LoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    # Expose the resolved human-friendly label (with 4-hex suffix) to the outer handle
    # so that any steering logs (stop/pause/resume/interject/ask) include the same suffix.
    with suppress(Exception):
        if outer_handle_container and outer_handle_container[0] is not None:
            setattr(outer_handle_container[0], "_log_label", cfg.label)
    logger = LoopLogger(cfg, log_steps)
    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)
    _img_token = None
    _imglog_token = None
    # If live images are provided, set the registry for this loop's scope
    try:
        if images:
            _img_token, _imglog_token = set_live_images_context(images, message)
    except Exception:
        _img_token = None
        _imglog_token = None

    # normalise optional graceful stop event
    stop_event = stop_event or asyncio.Event()

    _initial_user_message = copy.deepcopy(message)

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # `set_response_format` still happens at the end of the loop.
    if response_format is not None:
        try:
            _schema = _check_valid_response_format(response_format)
            _hint = (
                "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                + json.dumps(_schema, indent=2)
            )

            client.set_system_message((client.system_message or "") + _hint)
        except Exception as _exc:  # noqa: BLE001
            logger.error(f"response_format hint failed: {_exc!r}")

    # ── runtime guards ────────────────────────────────────────────────────
    # rolling timeout ----------------------------------------------------
    timer: TimeoutTimer = TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
    )
    _msg_dispatcher = LoopMessageDispatcher(client, cfg, timer)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context:
                logger.info(
                    f"Parent Context: {json.dumps(parent_chat_context, indent=4)}\n",
                    prefix="⬇️",
                )
            logger.info(f"System Message: {client.system_message}\n", prefix="📋")
        logger.info(f"User Message: {message}\n", prefix="🧑‍💻")

    # ── 0-a. Inject **system** header with broader context ───────────────────
    #
    # When a parent context is supplied we prepend a single synthetic system
    # message that *summarises* it.  This offers the LLM immediate awareness
    # of the wider conversation without having to scroll the nested JSON.
    # The special marker ``_ctx_header=True`` lets us later strip it when
    # propagating context further down (avoids duplication).
    # -----------------------------------------------------------------------

    if parent_chat_context:
        sys_msg = {
            "role": "system",
            "_ctx_header": True,
            "content": (
                "Broader context (read-only):\n"
                f"{json.dumps(parent_chat_context, indent=2)}\n\n"
                "Resolve the *next* user request in light of this."
            ),
        }
        await _msg_dispatcher.append_msgs([sys_msg])

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

    # ── Live image helpers (optional) ─────────────────────────────────────────
    # When a mapping of span→ImageHandle is supplied, expose three helper tools:
    #   • live_images_overview() – docstring lists images, spans, substrings, captions
    #   • ask_image(image_id, question) – returns a nested handle for image Q&A
    #   • attach_image_raw(image_id, note=None) – attaches the image as vision context
    # The docstring for `live_images_overview` is visible every turn; calling it is
    # not required and it's a cheap no-op.

    # Use shared text extraction helper from tools_utils instead of local duplicate

    # Build live image helpers only when images were supplied and non-empty
    live_image_tools: Dict[str, Callable] = {}
    if images:
        live_image_tools = build_live_image_tools(
            reference_message=message,
            images=images,
            append_user_messages=_msg_dispatcher.append_msgs,
        )

    # ── Helper to prepare arg-scoped images mapping for inner tool calls ─────
    align_images_for = _align_images_for

    # Only add align_images_for when images are actually provided
    general_helpers = {"align_images_for": align_images_for} if images else {}

    # Merge helpers (if any) with base tools before normalisation
    tools = {**tools, **(live_image_tools or {}), **general_helpers}

    # Initialise loop state early so preflight backfill can schedule tasks
    tools_data: ToolsData = ToolsData(tools, client=client, logger=logger)
    semantic_closest_match = None
    last_valid_user_history = []
    if semantic_cache:
        if semantic_closest_match := sc.search_semantic_cache(message):
            msgs = await sc.get_dummy_tool(semantic_closest_match, tools_data)
            if log_steps == "full":
                logger.info(
                    f"Semantic cache hit ({semantic_closest_match.closest_user_message}): {json.dumps(msgs[1]['content'], indent=2)}",
                    prefix="🔍",
                )
            client.append_messages(msgs)
            client.set_system_message(
                (client.system_message or "") + sc.get_system_msg_hint(),
            )
            tools_data.normalized["semantic_search"] = ToolSpec(
                fn=sc.semantic_search_placeholder,
            )
        else:
            if log_steps == "full":
                logger.info(
                    "Semantic cache miss, no entry for the user message",
                    prefix="🔍",
                )

    consecutive_failures = _LoopToolFailureTracker(max_consecutive_failures)
    assistant_meta: Dict[int, Dict[str, Any]] = {}
    step_index: int = 0  # per assistant turn
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

    # Ensure we forward stop to nested handles at most once, even if multiple
    # branches detect cancellation/stop around the same time.
    _stop_forwarded_once: bool = False

    # Preflight repair: backfill any pre-existing assistant tool_calls without replies
    with suppress(Exception):
        unreplied = find_unreplied_assistant_entries(client)
        if unreplied:
            # backfill for all such assistant messages (oldest → newest)
            for entry in unreplied:
                amsg = entry["assistant_msg"]
                # Before scheduling, drop any over-quota tool calls in this message
                tools_data.prune_over_quota_tool_calls(amsg)
                missing_ids = set(entry["missing"])
                await schedule_missing_for_message(
                    amsg,
                    missing_ids,
                    tools_data=tools_data,
                    parent_chat_context=parent_chat_context,
                    propagate_chat_context=propagate_chat_context,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )

    # ── initial **user** message (single-message path)
    if seeded_batch is None:
        if isinstance(message, dict):
            initial_user_msg = message
        else:
            initial_user_msg = {"role": "user", "content": message}
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
            logger.info(f"Early exit – {reason}", prefix="⏹️")
        return notice["content"]

    # ── small local helpers to dedupe repeated logic ─────────────────────────
    def _pretty(tool_name: str, payload: Any) -> str:
        return ToolsData._pretty_tool_payload(tool_name, payload)

    async def _handle_clarification(
        src_task: asyncio.Task,
        question_payload: Any,
    ) -> None:
        images_from_child = None
        question_text = ""
        try:
            if isinstance(question_payload, dict):
                images_from_child = question_payload.get("images")
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

        # Append any images sent alongside the clarification request
        with suppress(Exception):
            append_source_scoped_images_with_text(
                images_from_child,
                "clar_request",
                question_text,
            )

    async def _handle_notification(src_task: asyncio.Task, payload: Any) -> None:
        call_id = tools_data.info[src_task].call_id
        tool_name = tools_data.info[src_task].name

        pretty = ToolsData._pretty_tool_payload(tool_name, payload)

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

        # Append images provided with the notification payload
        with suppress(Exception):
            images_from_child = (
                payload.get("images") if isinstance(payload, dict) else None
            )
            try:
                base_text = (
                    payload.get("message")
                    if isinstance(payload, dict)
                    else str(payload)
                )
            except Exception:
                base_text = ""
            append_source_scoped_images_with_text(
                images_from_child,
                "notification",
                base_text,
            )

    # Set to *True* whenever the loop must grant the LLM an immediate turn
    # before waiting again (user interjection, clarification answer, etc.).
    llm_turn_required = False

    # Loop returns immediately upon the final assistant message (no persist mode)

    try:
        while True:
            # ── 0-Ø. Immediate handover for passthrough delegates ─────────────

            # ── 0-α-P. Global *pause* gate  ────────────────────────────
            # Keep handling tool completions & cancellation, but *never*
            # let the LLM speak while we're paused.
            if not pause_event.is_set():
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
                    for t in done & tools_data.pending:
                        await tools_data.process_completed_task(
                            task=t,
                            consecutive_failures=consecutive_failures,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            msg_dispatcher=_msg_dispatcher,
                        )
                    if cancel_event.is_set():
                        # Forward stop to any nested handles before aborting
                        with suppress(Exception):
                            _stop_forwarded_once = await propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
                        raise asyncio.CancelledError
                    # No graceful stop path
                    continue  # remain paused: do not allow the LLM to speak while paused
                else:
                    # nothing running – just idle until resumed or cancelled
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
                        with suppress(Exception):
                            _stop_forwarded_once = await propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
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
                            parent_chat_context=parent_chat_context,
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

                llm_turn_required = True
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

                # Support dict-style interjections carrying continued parent context
                if isinstance(extra, dict):
                    _msg_text = str(extra.get("message", "")).strip()
                    _ctx_cont = extra.get("parent_chat_context_continuted")
                    _incoming_images = extra.get("images")
                    with suppress(Exception):
                        _ctx_str = (
                            json.dumps(_ctx_cont, indent=2)
                            if _ctx_cont is not None
                            else None
                        )

                    sys_content = (
                        "The user *cannot* see *any* of the contents of this ongoing tool use chat context. "
                        "They have just interjected with the following message (in bold at the bottom). "
                        "From their perspective, the conversation thus far is as follows:\n"
                        "--\n"
                        + ("\n".join(history_lines))
                        + f"\nuser: **{_msg_text}**\n"
                        "--\n"
                        + (
                            "A continued parent chat context has been provided for this interjection.\n"
                            + (_ctx_str or "(unserializable)")
                            + "\n"
                            if _ctx_cont is not None
                            else ""
                        )
                        + "Please consider and incorporate *all* interjections in your final response to the user. "
                        + "Later interjections should always override earlier interjections if there are "
                        + "any conflicting comments/requests across the different interjections."
                    )
                else:
                    _msg_text = str(extra)
                    _incoming_images = None
                    sys_content = (
                        "The user *cannot* see *any* of the contents of this ongoing tool use chat context. "
                        "They have just interjected with the following message (in bold at the bottom). "
                        "From their perspective, the conversation thus far is as follows:\n"
                        "--\n"
                        + ("\n".join(history_lines))
                        + f"\nuser: **{_msg_text}**\n"
                        "--\n"
                        "Please consider and incorporate *all* interjections in your final response to the user. "
                        "Later interjections should always override earlier interjections if there are "
                        "any conflicting comments/requests across the different interjections."
                    )

                interjection_msg = {"role": "system", "content": sys_content}
                await _msg_dispatcher.append_msgs([interjection_msg])
                last_valid_user_history = history_lines + [f"user: {extra}"]

                # If images accompany this interjection, accept source-scoped keys and append
                with suppress(Exception):
                    append_source_scoped_images_with_text(
                        _incoming_images,
                        "interjection",
                        _msg_text,
                    )

                # Append this interjection to the user-visible history for future context
                with suppress(Exception):
                    if outer_handle:
                        outer_handle._user_visible_history.append(
                            {
                                "role": "user",
                                "content": (
                                    {
                                        "message": _msg_text,
                                        "parent_chat_context_continuted": extra.get(
                                            "parent_chat_context_continuted",
                                        ),
                                    }
                                    if isinstance(extra, dict)
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
                    with suppress(Exception):
                        _stop_forwarded_once = await propagate_stop_once(
                            tools_data.info,
                            _stop_forwarded_once,
                            "outer-loop cancelled",
                        )
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
                for task in done & tools_data.pending:  # finished tool(s)
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
                    content=(
                        "Still running… you can use any of the available helper tools "
                        "to interact with this tool call while it is in progress."
                    ),
                    tools_data=tools_data,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )
                continue  # still waiting for other tool tasks

            # ── No passthrough delegate; outer loop continues scheduling ──────

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
                try:
                    tool_choice_mode, filtered = tool_policy(
                        step_index,
                        {n: s.fn for n, s in tools_data.normalized.items()},
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", {
                        n: s.fn for n, s in tools_data.normalized.items()
                    }
                policy_tools_norm = normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.normalized

            # Force tool usage when a response_format is required so the model
            # must submit the final JSON via the strongly-typed `final_answer` tool.
            # This preserves flexible tool use while guaranteeing typed completion.
            if response_format is not None and tool_choice_mode != "required":
                tool_choice_mode = "required"

            # Refresh live-images overview with the latest appended images
            try:
                if images:
                    refresh_overview_doc_if_present(tools_data.normalized)
            except Exception:
                pass

            visible_base_tools_schema = [
                method_to_schema(spec.fn, name)
                for name, spec in policy_tools_norm.items()
                if tools_data.concurrency_ok(name) and tools_data.quota_ok(name)
            ]

            # Inject `final_answer` tool automatically whenever a `response_format` is
            # supplied. The tool accepts a single `answer` argument whose schema matches
            # the provided Pydantic model.
            if response_format is not None:
                try:
                    _answer_schema = _check_valid_response_format(response_format)

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": True,
                            "function": {
                                "name": "final_answer",
                                "description": (
                                    "Submit your final answer in the required JSON format. "
                                    "Calling this tool marks the conversation as complete."
                                ),
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
                        f"Failed to inject final_answer tool: {_injection_exc!r}",
                    )

            dynamic_tool_factory = DynamicToolFactory(tools_data)
            dynamic_tool_factory.generate()
            dynamic_tools = dynamic_tool_factory.dynamic_tools

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
                content=(
                    "Still running… you can use any of the available helper tools "
                    "to interact with this tool call while it is in progress."
                ),
                tools_data=tools_data,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
            )

            # Merge helpers into the visible toolkit for the upcoming LLM step
            tmp_tools = visible_base_tools_schema + [
                method_to_schema(
                    fn,
                    include_class_name=include_class_in_dynamic_tool_names,
                )
                for fn in dynamic_tools.values()
            ]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                logger.info(f"LLM thinking…", prefix="🔄")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                llm_task = asyncio.create_task(
                    generate_with_preprocess(client, preprocess_msgs, **_gen_kwargs),
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

                # helper cleanup
                for tsk in (
                    llm_task,
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
                    for task in done & pending_snapshot:
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
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    await interject_queue.put(interject_w.result())
                    continue  # top of loop

                # 2️⃣ cancellation requested
                if cancel_waiter in done:
                    # Only escalate when the cancellation flag is actually set.
                    if cancel_event.is_set():
                        if not llm_task.done():
                            llm_task.cancel()
                            await asyncio.gather(llm_task, return_exceptions=True)
                        raise asyncio.CancelledError

                # 3️⃣ LLM finished normally
                if llm_task.exception():
                    try:
                        llm_task.result()
                    except Exception as e:
                        raise Exception(
                            f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
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
                    }
                    if max_parallel_tool_calls is not None:
                        _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                    await generate_with_preprocess(
                        client,
                        preprocess_msgs,
                        **_gen_kwargs,
                    )
                except Exception:
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                    )

            msg = client.messages[-1]
            await to_event_bus(msg, cfg)

            if log_steps:
                with suppress(Exception):
                    logger.info(f"{json.dumps(msg, indent=4)}\n", prefix="🤖")

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
                    seen: Set[tuple[str, str]] = set()
                    unique_calls: list = []
                    for call in msg["tool_calls"]:
                        sig = (call["function"]["name"], call["function"]["arguments"])
                        if sig not in seen:
                            seen.add(sig)
                            unique_calls.append(call)
                    if len(unique_calls) != len(msg["tool_calls"]):
                        # mutate in-place so history never contains duplicates
                        msg["tool_calls"] = unique_calls

                # Always ensure over-quota tool calls are removed regardless of
                # deduplication settings, before any scheduling occurs.
                tools_data.prune_over_quota_tool_calls(msg)
                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    # Special-case: handle synthetic `final_answer` tool
                    if name == "final_answer" and response_format is not None:
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            # Validate payload with the provided Pydantic model.
                            response_format.model_validate(payload)

                            tool_msg = create_tool_call_message(
                                name="final_answer",
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

                            return json.dumps(payload)
                        except Exception as _exc:
                            tool_msg = create_tool_call_message(
                                name="final_answer",
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

                    # ── Special-case dynamic helpers ──────────────────────
                    # • wait        → acknowledge, list running tasks, no scheduling
                    # • cancel_*    → cancel underlying task & purge metadata
                    if name == "wait":
                        # Log the no-op and prune it from the transcript to avoid clutter.
                        try:
                            logger.info(
                                "Assistant chose `wait` – no-op; not persisting to transcript.",
                                prefix="🕒",
                            )
                        except Exception:
                            pass

                        # Prune the `wait` tool call using a shared helper
                        with suppress(Exception):
                            from .messages import prune_wait_tool_call as _prune_wait

                            _prune_wait(msg, call["id"], client=client)

                        # After acknowledging a wait, do NOT grant an immediate LLM turn.
                        # The loop should now wait for any pending tools or interjections.
                        continue

                    if name.startswith("stop_") and not name.startswith(
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

                        # ── gracefully shut down any *nested* async-tool loop first ──────
                        if task_to_cancel:
                            nested_handle = tools_data.info[task_to_cancel].handle
                            if nested_handle is not None:
                                # public API call – propagates cancellation downwards
                                await forward_handle_call(
                                    nested_handle,
                                    "stop",
                                    payload,
                                    fallback_positional_keys=["reason"],
                                )

                        # ── then cancel the waiter coroutine itself ───────────────────────────
                        if task_to_cancel and not task_to_cancel.done():
                            task_to_cancel.cancel()
                        if task_to_cancel:
                            tools_data.pop_task(task_to_cancel)

                        # Record any images provided with the stop helper and capture reason text
                        with suppress(Exception):
                            try:
                                reason_txt = payload.get("reason")
                            except Exception:
                                reason_txt = ""
                            append_source_scoped_images_with_text(
                                payload.get("images"),
                                "stop",
                                reason_txt or "",
                            )

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
                        # Trigger an immediate LLM turn after helper action
                        llm_turn_required = True

                        continue  # nothing else to schedule

                    # ── _pause helper ────────────────────────────────────────────────
                    if name.startswith("pause_") and not name.startswith(
                        "_pause_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info.call_id
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

                        # Forward any extra kwargs to handle.pause if available
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                        if "payload" not in locals():
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].handle
                            ev = tools_data.info[tgt_task].pause_event
                            if h is not None and hasattr(h, "pause"):
                                await forward_handle_call(h, "pause", payload)
                            elif ev is not None:
                                ev.clear()

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
                        # Trigger an immediate LLM turn after helper action
                        llm_turn_required = True
                        continue  # helper handled, move on

                    # ── _resume helper ───────────────────────────────────────────────
                    if name.startswith("resume_") and not name.startswith(
                        "_resume_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info.call_id
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

                        # Forward any extra kwargs to handle.resume if available
                        with suppress(Exception):
                            payload = json.loads(call["function"]["arguments"]) or {}
                        if "payload" not in locals():
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].handle
                            ev = tools_data.info[tgt_task].pause_event
                            if h is not None and hasattr(h, "resume"):
                                await forward_handle_call(h, "resume", payload)
                            elif ev is not None:
                                ev.set()

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
                        # Trigger an immediate LLM turn after helper action
                        llm_turn_required = True
                        continue  # helper handled

                    if name.startswith("clarify_"):
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

                        # Find clarification channel by matching call-id suffix
                        _clar_key = next(
                            (
                                k
                                for k in tools_data.clarification_channels.keys()
                                if k.endswith(call_id_suffix)
                            ),
                            None,
                        )
                        if _clar_key is not None:
                            await tools_data.clarification_channels[_clar_key][1].put(
                                ans,
                            )  # down-queue
                            # ✔️ the tool is un-blocked – start watching it again
                            for _t, _inf in tools_data.info.items():
                                if str(_inf.call_id).endswith(
                                    call_id_suffix,
                                ):
                                    _inf.waiting_for_clarification = False
                                    break

                        # Record any images provided with the clarification answer
                        with suppress(Exception):
                            append_source_scoped_images_with_text(
                                args.get("images") if isinstance(args, dict) else None,
                                "clar_answer",
                                ans,
                            )
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
                        if tgt_task is not None:
                            tools_data.info[tgt_task].clarify_placeholder = (
                                tool_reply_msg
                            )
                        # Trigger an immediate LLM turn after helper action
                        llm_turn_required = True
                        continue

                    if name.startswith("interject_"):
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

                        # ― push guidance onto the private queue or forward to handle with full kwargs -------------
                        if tgt_task:
                            iq = tools_data.info[tgt_task].interject_queue
                            h = tools_data.info[tgt_task].handle

                            if iq is not None:
                                await iq.put(new_text)
                            elif h is not None and hasattr(h, "interject"):
                                await forward_handle_call(
                                    h,
                                    "interject",
                                    payload,
                                    fallback_positional_keys=["content", "message"],
                                )

                        # Record any images provided with the interjection helper
                        with suppress(Exception):
                            append_source_scoped_images_with_text(
                                payload.get("images"),
                                "interjection",
                                new_text,
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
                        # Trigger an immediate LLM turn after helper action
                        llm_turn_required = True

                        continue  # nothing else to schedule

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
                        fn = dynamic_tools[name]

                        # ── build **extra** kwargs (chat context + queue) for dynamic helper ──
                        extra_kwargs: dict = {}
                        if propagate_chat_context:
                            cur_msgs = [
                                m for m in client.messages if not m.get("_ctx_header")
                            ]
                            ctx_repr = chat_context_repr(parent_chat_context, cur_msgs)
                            extra_kwargs["parent_chat_context"] = ctx_repr

                        sig = inspect.signature(fn)
                        params = sig.parameters
                        has_varkw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
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
                            chat_context=extra_kwargs.get("parent_chat_context"),
                            pause_event=None,
                            # Debug helpers for failure logging
                            tool_schema=method_to_schema(
                                fn,
                                include_class_name=include_class_in_dynamic_tool_names,
                            ),
                            llm_arguments=allowed_call_args,
                            raw_arguments_json=call["function"]["arguments"],
                        )
                        tools_data.save_task(
                            coro=t,
                            metadata=metadata,
                        )
                    else:
                        # Use shared helper for base tools
                        await tools_data.schedule_base_tool_call(
                            msg,
                            name=name,
                            args_json=call["function"]["arguments"],
                            call_id=call["id"],
                            call_idx=idx,
                            parent_chat_context=parent_chat_context,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                        )

                # metadata for orderly insertion
                assistant_meta[id(msg)] = {
                    "results_count": 0,
                }

                # Immediately insert placeholder tool replies for every newly scheduled call
                #  to satisfy API ordering even if a user interjection arrives instantly.
                try:
                    await ensure_placeholders_for_pending(
                        assistant_msg=msg,
                        content="Pending… tool call accepted. Working on it.",
                        tools_data=tools_data,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                except Exception as _ph_exc:
                    logger.error(
                        f"Failed to insert immediate placeholders: {_ph_exc!r}",
                    )

                continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if tools_data.pending:  # still running
                continue  # wait for completions, then prompt LLM

            # ── timeout guard (final turn) ──────────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            final_answer = msg["content"]

            return final_answer  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
        # NOTE: Caller (or parent task) requested cancellation.  We propagate
        # the signal to *all* running tool tasks first so each can release
        # resources cleanly.  Only after every task has finished/aborted do
        # we re-raise the same `CancelledError`, preserving expected asyncio
        # semantics for upstream callers.
        with suppress(Exception):
            _stop_forwarded_once = await propagate_stop_once(
                tools_data.info,
                _stop_forwarded_once,
                "outer-loop cancelled",
            )
        await tools_data.cancel_pending_tasks()
        raise
    finally:
        with suppress(Exception):
            TOOL_LOOP_LINEAGE.reset(_token)
        reset_live_images_context(_img_token, _imglog_token)

        if semantic_cache:
            sc.save_semantic_cache(
                _initial_user_message,
                last_valid_user_history,
                client.messages,
                previous_tool_trajectory=(
                    semantic_closest_match.tool_trajectory
                    if semantic_closest_match
                    else None
                ),
            )
