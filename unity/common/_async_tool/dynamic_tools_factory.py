from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from contextlib import suppress
from .tools_data import ToolsData
from .messages import forward_handle_call
from .tools_utils import ToolCallMetadata
from .tools_utils import (
    append_source_scoped_images,
    default_source_label,
)
from .utils import maybe_await


class DynamicToolFactory:
    _MANAGEMENT_METHOD_NAMES: set[str] = {
        "interject",
        "pause",
        "resume",
        "stop",
        "done",
        "result",
    }

    @dataclass
    class _ToolContext:
        fn_name: str
        arg_repr: str
        call_id: str
        safe_call_id: str

    def __init__(self, tools_data: ToolsData):
        self.dynamic_tools = {}
        self.tools_data = tools_data

    # Shared steering helpers – reduce duplication across dynamic helper tools
    @staticmethod
    def _adopt_signature_and_annotations(from_callable, to_wrapper) -> None:
        """Copy signature and annotations (excluding 'self') from from_callable to to_wrapper."""
        try:
            to_wrapper.__signature__ = inspect.signature(from_callable)
            try:
                ann = dict(getattr(from_callable, "__annotations__", {}))
                ann.pop("self", None)
                to_wrapper.__annotations__ = ann
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _discover_custom_public_methods(handle) -> dict[str, Callable]:
        """
        Return a mapping ``name → bound_method`` of *public* callables on *handle*:
            • name does **not** start with ``_``  _and_
            • name is not one of the management helpers above.
        """
        methods: dict[str, Callable] = {}
        for name, attr in inspect.getmembers(handle):
            if (
                name.startswith("_")
                or name in DynamicToolFactory._MANAGEMENT_METHOD_NAMES
                or not callable(attr)
            ):
                continue
            # Bind the method to *handle* (important for late-added attributes).
            try:
                bound = handle.__getattribute__(name)
            except Exception:
                # Attribute access raised – treat as non-callable.
                continue

            methods[name] = bound
        return methods

    # helper: register a freshly-minted coroutine as a *temporary* tool
    def _register_tool(
        self,
        func_name: str,
        fallback_doc: str,
        fn: Callable,
    ) -> None:
        # prefer the function's own docstring if it exists, else fall back
        existing = inspect.getdoc(fn)
        fn.__doc__ = existing.strip() if existing else fallback_doc
        fn.__name__ = func_name[:64]
        fn.__qualname__ = func_name[:64]
        self.dynamic_tools[func_name.lstrip("_")] = fn

    def _create_wait_tool(self) -> None:
        """
        Expose a single global helper tool `wait` that performs a no-op.

        Purpose
        -------
        Use this when you do not want to take any new action at this time.
        Calling `wait` explicitly instructs the agent to keep waiting for
        any currently running tool calls to finish (or for an interjection
        to arrive) before deciding whether to act next. It does not start,
        stop, pause, resume, or modify any in-flight work.
        """

        async def _wait() -> Dict[str, str]:
            return {"status": "waiting"}

        self._register_tool(
            func_name="wait",
            fallback_doc=(
                "No-op: keep waiting on the currently running tool calls. "
                "Use this when you don't need to start/stop/pause/resume anything right now; "
                "decide what to do after the next tool completes or a new interjection arrives."
            ),
            fn=_wait,
        )

    def _create_stop_tool(
        self,
        tool_context: _ToolContext,
        task: asyncio.Task,
        handle: Any,
    ) -> None:
        doc = (
            f"Stop pending call {tool_context.fn_name}({tool_context.arg_repr}).\n\n"
            "Parameters\n"
            "----------\n"
            "reason : str | None\n"
            "    Optional human‑readable reason for stopping the running tool call.\n"
            "images : dict | None\n"
            "    Optional source‑scoped images mapping to append at the time of this command.\n"
            "    Keys use `<source>[start:end]`. Supported sources include: `this`, `user_message`, `interjectionN`,\n"
            "    `askN`, `clar_requestN`, `clar_answerN`, `notificationN`, `stopN`. Use `this[:]` to associate images\n"
            "    with the stop command itself. Values are image ids or live image handle objects.\n\n"
            "Returns\n"
            "-------\n"
            "Dict[str, str]\n"
            "    Status acknowledgement including the underlying call id.\n\n"
            "Notes\n"
            "-----\n"
            "- Images are appended to the live images log immediately and reflected in `live_images_overview`.\n"
            "- The stop request is forwarded to the underlying handle when available."
        )

        async def _stop(**_kw) -> Dict[str, str]:
            # Forward stop intent to the running handle with any extra kwargs
            if handle is not None and hasattr(handle, "stop"):
                await forward_handle_call(
                    handle,
                    "stop",
                    _kw,
                    fallback_positional_keys=["reason"],
                )
            # Append any provided images into the live registry/log
            try:
                append_source_scoped_images(
                    _kw.get("images"),
                    default_source_label("stop"),
                )
            except Exception:
                pass
            if not task.done():
                task.cancel()  # kill the waiter coroutine
            self.tools_data.pop_task(task)
            return {
                "status": "stopped",
                "call_id": tool_context.call_id,
                **{k: v for k, v in _kw.items() if k != "images"},
            }

        self._register_tool(
            func_name=f"stop_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_stop,
        )
        # Expose full argspec of handle.stop in the helper schema
        # Ensure helper schema mirrors underlying handle.stop parameters and adds `images`.
        with suppress(Exception):
            import inspect as _inspect

            params: list[_inspect.Parameter] = []
            annotations: dict = {"return": Dict[str, str]}
            if handle is not None and hasattr(handle, "stop"):
                sig = _inspect.signature(getattr(handle, "stop"))
                for name, p in sig.parameters.items():
                    if name == "self":
                        continue
                    params.append(
                        _inspect.Parameter(
                            name,
                            kind=_inspect.Parameter.KEYWORD_ONLY,
                            default=p.default,
                            annotation=p.annotation,
                        ),
                    )
                    if p.annotation is not _inspect._empty:  # type: ignore[attr-defined]
                        annotations[name] = p.annotation
            # Append images param if not already present
            if not any(p.name == "images" for p in params):
                params.append(
                    _inspect.Parameter(
                        "images",
                        kind=_inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[dict],
                    ),
                )
                annotations["images"] = Optional[dict]
            _stop.__signature__ = _inspect.Signature(
                parameters=params,
                return_annotation=Dict[str, str],
            )
            _stop.__annotations__ = annotations

    def _create_interject_tool(
        self,
        tool_context: _ToolContext,
        task_info: ToolCallMetadata,
        handle: Any,
    ) -> None:
        doc = (
            f"Inject additional instructions for {tool_context.fn_name}({tool_context.arg_repr}).\n\n"
            "Parameters\n"
            "----------\n"
            "content : str | None\n"
            "    Interjection text. When omitted, `message` may be used as a synonym.\n"
            "message : str | None\n"
            "    Synonym for `content`. If both are provided, `content` takes precedence.\n"
            "images : dict | None\n"
            "    Optional source‑scoped images mapping to append at the time of this interjection.\n"
            "    Keys use `<source>[start:end]`. Supported sources include: `this`, `user_message`, `interjectionN`,\n"
            "    `askN`, `clar_requestN`, `clar_answerN`, `notificationN`, `stopN`. Use `this[:]` to associate images with\n"
            "    the interjection text itself. Values are image ids or live image handle objects.\n\n"
            "Returns\n"
            "-------\n"
            "Dict[str, str]\n"
            "    Status acknowledgement including the underlying call id.\n"
        )

        if handle is not None:

            async def _interject(**_kw) -> Dict[str, str]:
                # nested async-tool loop: delegate to its public API with full argspec
                with suppress(Exception):
                    await forward_handle_call(
                        handle,
                        "interject",
                        _kw,
                        fallback_positional_keys=["content", "message"],
                    )
                # Append any provided images into the live registry/log
                try:
                    append_source_scoped_images(
                        _kw.get("images"),
                        default_source_label("interjection"),
                    )
                except Exception:
                    pass
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **{k: v for k, v in _kw.items() if k != "images"},
                }

            # Expose the downstream handle's signature to the LLM
            with suppress(Exception):
                import inspect as _inspect

                params: list[_inspect.Parameter] = []
                annotations: dict = {"return": Dict[str, str]}
                if hasattr(handle, "interject"):
                    sig = _inspect.signature(getattr(handle, "interject"))
                    for name, p in sig.parameters.items():
                        if name == "self":
                            continue
                        params.append(
                            _inspect.Parameter(
                                name,
                                kind=_inspect.Parameter.KEYWORD_ONLY,
                                default=p.default,
                                annotation=p.annotation,
                            ),
                        )
                        if p.annotation is not _inspect._empty:  # type: ignore[attr-defined]
                            annotations[name] = p.annotation
                # Ensure common aliases are visible in schema if not present
                existing = {p.name for p in params}
                if "content" not in existing and "message" not in existing:
                    params.insert(
                        0,
                        _inspect.Parameter(
                            "content",
                            kind=_inspect.Parameter.KEYWORD_ONLY,
                            default=None,
                            annotation=Optional[str],
                        ),
                    )
                    annotations["content"] = Optional[str]
                # Append images param if not already present
                if not any(p.name == "images" for p in params):
                    params.append(
                        _inspect.Parameter(
                            "images",
                            kind=_inspect.Parameter.KEYWORD_ONLY,
                            default=None,
                            annotation=Optional[dict],
                        ),
                    )
                    annotations["images"] = Optional[dict]
                _interject.__signature__ = _inspect.Signature(
                    parameters=params,
                    return_annotation=Dict[str, str],
                )
                _interject.__annotations__ = annotations

        else:

            async def _interject(
                *,
                content: Optional[str] = None,
                message: Optional[str] = None,
                images: dict | None = None,
            ) -> Dict[str, str]:
                # regular tool: push onto its private queue
                actual = content if content is not None else (message or "")
                await task_info.interject_queue.put(actual)
                # Append any provided images into the live registry/log
                try:
                    append_source_scoped_images(
                        images,
                        default_source_label("interjection"),
                    )
                except Exception:
                    pass
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **({"content": actual} if actual else {}),
                }

        self._register_tool(
            func_name=f"interject_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_interject,
        )
        with suppress(Exception):
            import inspect as _inspect

            _interject.__annotations__ = {
                "content": Optional[str],
                "message": Optional[str],
                "images": Optional[dict],
                "return": Dict[str, str],
            }
            _interject.__signature__ = _inspect.Signature(
                parameters=[
                    _inspect.Parameter(
                        "content",
                        kind=_inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[str],
                    ),
                    _inspect.Parameter(
                        "message",
                        kind=_inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[str],
                    ),
                    _inspect.Parameter(
                        "images",
                        kind=_inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[dict],
                    ),
                ],
                return_annotation=Dict[str, str],
            )

    def _create_clarify_tool(
        self,
        tool_context: _ToolContext,
    ) -> None:
        doc = (
            f"Provide an answer to the clarification which was requested by the (currently pending) tool "
            f"{tool_context.fn_name}({tool_context.arg_repr}).\n\n"
            "Parameters\n"
            "----------\n"
            "answer : str\n"
            "    The answer text.\n"
            "images : dict | None\n"
            "    Optional source‑scoped images mapping to append alongside this answer.\n"
            "    Keys use `<source>[start:end]`. Supported sources include: `this`, `user_message`, `interjectionN`,\n"
            "    `askN`, `clar_requestN`, `clar_answerN`, `notificationN`, `stopN`. Use `this[:]` to associate images\n"
            "    with the answer text itself. Values are image ids or live image handle objects.\n\n"
            "Returns\n"
            "-------\n"
            "Dict[str, str]\n"
            "    Status acknowledgement including the underlying call id.\n"
        )

        async def _clarify(answer: str, images: dict | None = None) -> Dict[str, str]:  # type: ignore[valid-type]
            try:
                append_source_scoped_images(
                    images,
                    default_source_label("clar_answer"),
                )
            except Exception:
                pass
            return {
                "status": "clar_answer",
                "call_id": tool_context.call_id,
                "answer": answer,
            }

        self._register_tool(
            func_name=f"clarify_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_clarify,
        )

    def _create_pause_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
        pause_event: Optional[asyncio.Event],
    ) -> None:
        handle_available = handle is not None

        if handle_available and hasattr(handle, "pause"):

            async def _pause(**_kw) -> Dict[str, str]:
                with suppress(Exception):
                    await forward_handle_call(handle, "pause", _kw)
                return {"status": "paused", "call_id": tool_context.call_id, **_kw}

            # Reflect downstream signature/annotations
            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "pause"),
                    _pause,
                )

        else:

            async def _pause() -> Dict[str, str]:
                if handle_available and hasattr(handle, "pause"):
                    await maybe_await(handle.pause())
                elif pause_event is not None:
                    pause_event.clear()
                return {"status": "paused", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"pause_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=f"Pause the pending call {tool_context.fn_name}({tool_context.arg_repr}).",
            fn=_pause,
        )

    def _create_resume_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
        pause_event: Optional[asyncio.Event],
    ) -> None:
        doc = f"Resume the previously paused call {tool_context.fn_name}({tool_context.arg_repr})."

        handle_available = handle is not None

        if handle_available and hasattr(handle, "resume"):

            async def _resume(**_kw) -> Dict[str, str]:
                with suppress(Exception):
                    await forward_handle_call(handle, "resume", _kw)
                return {"status": "resumed", "call_id": tool_context.call_id, **_kw}

            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "resume"),
                    _resume,
                )

        else:

            async def _resume() -> Dict[str, str]:
                if handle_available and hasattr(handle, "resume"):
                    await maybe_await(handle.resume())
                elif pause_event is not None:
                    pause_event.set()
                return {"status": "resumed", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"resume_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_resume,
        )

    def _expose_public_methods(self, tool_context: _ToolContext, handle: Any):
        public_methods = self._discover_custom_public_methods(handle)

        # Identify write-only helpers declared by the handle
        write_only_set: set[str] = set()
        with suppress(Exception):
            wo = getattr(handle, "write_only_methods", None)
            if wo is not None:
                write_only_set |= set(wo)

        with suppress(Exception):
            wo2 = getattr(handle, "write_only_tools", None)
            if wo2 is not None:
                write_only_set |= set(wo2)

        for meth_name, bound in public_methods.items():
            # use the same name we're about to give fn.__name__
            func_name = (
                f"{meth_name}_{tool_context.fn_name}_{tool_context.safe_call_id}"
            )
            helper_key = func_name

            # Skip if we already generated one this turn (possible when
            # the loop revisits the same pending task).
            if helper_key in self.dynamic_tools:
                continue

            # Write-only helpers: fire-and-forget operations
            if meth_name in write_only_set:

                async def _invoke_handle_method(
                    _method_name=meth_name,
                    **_kw,
                ):
                    # Robust forwarding incl. kwargs normalisation and fallbacks
                    with suppress(Exception):
                        await forward_handle_call(
                            handle,
                            _method_name,
                            _kw,
                        )
                    # Write-only: no result propagation
                    return {"call_id": tool_context.call_id, "status": "ack"}

            else:

                async def _invoke_handle_method(
                    _method_name=meth_name,
                    **_kw,
                ):  # default args → capture current method name
                    """
                    Auto-generated wrapper that calls the corresponding
                    method on the live handle and **waits** for the return
                    value (sync or async).
                    """
                    # Use shared forwarding to support flexible args and fallbacks
                    res = await forward_handle_call(
                        handle,
                        _method_name,
                        _kw,
                    )
                    return {"call_id": tool_context.call_id, "result": res}

            # override the wrapper's signature to match the real method
            _invoke_handle_method.__signature__ = inspect.signature(bound)

            self._register_tool(
                func_name=func_name,
                fallback_doc=(
                    (
                        f"Perform `{meth_name}` on the running handle (id={tool_context.call_id}). "
                        "Fire-and-forget write-only operation; returns immediately."
                    )
                    if meth_name in write_only_set
                    else (
                        f"Invoke `{meth_name}` on the running handle (id={tool_context.call_id}). "
                        "Returns when that method finishes."
                    )
                ),
                fn=_invoke_handle_method,
            )
            # Mark write-only helpers so scheduling can acknowledge and avoid tracking
            if meth_name in write_only_set:
                with suppress(Exception):
                    self.dynamic_tools[helper_key].__write_only__ = True  # type: ignore[attr-defined]

    def _process_task(self, task: asyncio.Task):
        info = self.tools_data.info[task]
        handle = info.handle
        task_pause_event = info.pause_event
        handle_available = handle is not None

        # ── DYNAMIC capability refresh (handle may change) ─────
        if handle_available:
            # 1. interjection
            info.is_interjectable = hasattr(handle, "interject")

            # 2. clarification queues
            h_up_q = getattr(
                handle,
                "clarification_up_q",
                info.clar_up_queue,
            )
            h_dn_q = getattr(
                handle,
                "clarification_down_q",
                info.clar_down_queue,
            )

            if (h_up_q is not None) ^ (h_dn_q is not None):
                raise AttributeError(
                    f"Handle of call {info.call_id} now exposes only one "
                    "of clarification queues; both or neither required.",
                )

            # update bookkeeping & channel map
            prev_up_q = info.clar_up_queue
            if h_up_q is not prev_up_q:
                # remove old mapping if any
                self.tools_data.clarification_channels.pop(info.call_id, None)
                if h_up_q is not None:
                    self.tools_data.clarification_channels[info.call_id] = (
                        h_up_q,
                        h_dn_q,
                    )
            info.clar_up_queue = h_up_q
            info.clar_down_queue = h_dn_q

        _call_id: str = info.call_id
        # Create a sanitized version of the call_id for use in function names.
        _safe_call_id: str = _call_id.replace("-", "_").split("_")[-1]
        _fn_name: str = info.name
        _arg_json: str = info.call_dict["function"]["arguments"]
        try:
            _arg_dict = json.loads(_arg_json)
            _arg_repr = ", ".join(f"{k}={v!r}" for k, v in _arg_dict.items())
        except Exception:
            _arg_repr = _arg_json  # fallback: raw JSON string

        create_tool_ctx = self._ToolContext(
            fn_name=_fn_name,
            arg_repr=_arg_repr,
            call_id=_call_id,
            safe_call_id=_safe_call_id,
        )

        self._create_stop_tool(
            create_tool_ctx,
            task,
            handle,
        )

        if info.is_interjectable:
            self._create_interject_tool(
                create_tool_ctx,
                info,
                handle,
            )

        # Clarify helpers are exposed dynamically only when a call is actively awaiting
        # clarification and the transcript shows a user-facing clarification exchange
        # (request followed by a `request_clarification` tool reply) since the most
        # recent clarification event. This avoids premature answers and supports
        # multiple clarification cycles across a single long-running call.
        if info.clar_up_queue is not None and getattr(
            info,
            "waiting_for_clarification",
            False,
        ):
            # Additional gate: only expose clarify AFTER a request_clarification tool reply
            # is present in the transcript (post the recorded clarification request point).
            try:
                msgs = getattr(self.tools_data, "_client").messages  # transcript
            except Exception:
                msgs = []
            start_idx = getattr(info, "clar_request_index", 0) or 0
            saw_request_tool_reply = False
            try:
                for m in list(msgs)[start_idx:]:
                    if (
                        isinstance(m, dict)
                        and m.get("role") == "tool"
                        and m.get("name") == "request_clarification"
                    ):
                        saw_request_tool_reply = True
                        break
            except Exception:
                saw_request_tool_reply = False

            if saw_request_tool_reply:
                self._create_clarify_tool(create_tool_ctx)
            # else: still waiting; do not expose yet

        # Determine capability and current pause state; expose only one helper at a time
        cap_pause = (handle_available and hasattr(handle, "pause")) or (
            task_pause_event is not None
        )
        cap_resume = (handle_available and hasattr(handle, "resume")) or (
            task_pause_event is not None
        )

        paused_state = None
        try:
            # Prefer downstream handle's pause event if available
            pev = getattr(handle, "_pause_event", None) if handle_available else None
            if pev is not None and hasattr(pev, "is_set"):
                paused_state = not pev.is_set()  # running ⇢ set, paused ⇢ cleared
        except Exception:
            pass
        if (
            paused_state is None
            and task_pause_event is not None
            and hasattr(task_pause_event, "is_set")
        ):
            try:
                paused_state = not task_pause_event.is_set()
            except Exception:
                paused_state = None

        # Default to "running" when unknown → expose pause first
        if paused_state is True:
            if cap_resume:
                self._create_resume_tool(
                    create_tool_ctx,
                    handle,
                    task_pause_event,
                )
        else:
            if cap_pause:
                self._create_pause_tool(
                    create_tool_ctx,
                    handle,
                    task_pause_event,
                )

        # 7.  expose *all* other public methods of the handle
        if handle_available:
            self._expose_public_methods(create_tool_ctx, handle)

    def generate(self):
        for task in list(self.tools_data.pending):
            self._process_task(task)
        # Expose a single global `wait` helper when anything is in flight
        if self.tools_data.pending:
            self._create_wait_tool()
