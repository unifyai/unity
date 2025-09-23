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

    def _create_continue_tool(
        self,
        tool_context: _ToolContext,
    ) -> None:
        async def _continue() -> Dict[str, str]:
            return {"status": "continue", "call_id": tool_context.call_id}

        self._register_tool(
            func_name=f"continue_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=f"Continue waiting for {tool_context.fn_name}({tool_context.arg_repr}).",
            fn=_continue,
        )

    def _create_stop_tool(
        self,
        tool_context: _ToolContext,
        task: asyncio.Task,
        handle: Any,
    ) -> None:
        doc = (
            f"Stop pending call {tool_context.fn_name}({tool_context.arg_repr}). "
            "Accepts any arguments supported by the underlying handle's `stop` method (e.g. `reason`)."
        )

        async def _stop(
            **_kw,
        ) -> Dict[str, str]:
            # Forward stop intent to the running handle with any extra kwargs
            if handle is not None and hasattr(handle, "stop"):
                await forward_handle_call(
                    handle,
                    "stop",
                    _kw,
                    fallback_positional_keys=["reason"],
                )
            if not task.done():
                task.cancel()  # kill the waiter coroutine
            self.tools_data.pop_task(task)
            return {"status": "stopped", "call_id": tool_context.call_id, **_kw}

        self._register_tool(
            func_name=f"stop_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_stop,
        )
        # Expose full argspec of handle.stop in the helper schema
        with suppress(Exception):
            if handle is not None and hasattr(handle, "stop"):
                self._adopt_signature_and_annotations(getattr(handle, "stop"), _stop)

    def _create_interject_tool(
        self,
        tool_context: _ToolContext,
        task_info: ToolCallMetadata,
        handle: Any,
    ) -> None:
        doc = (
            f"Inject additional instructions for {tool_context.fn_name}({tool_context.arg_repr}). "
            "Accepts any arguments supported by the underlying handle's `interject` method (e.g. `content`)."
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
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **{k: v for k, v in _kw.items()},
                }

            # Expose the downstream handle's signature to the LLM
            with suppress(Exception):
                self._adopt_signature_and_annotations(
                    getattr(handle, "interject"),
                    _interject,
                )

        else:

            async def _interject(content: str) -> Dict[str, str]:
                # regular tool: push onto its private queue
                await task_info.interject_queue.put(content)
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    "content": content,
                }

        self._register_tool(
            func_name=f"interject_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_interject,
        )

    def _create_clarify_tool(
        self,
        tool_context: _ToolContext,
    ) -> None:
        doc = (
            f"Provide an answer to the clarification which was requested by the (currently pending) tool "
            f"{tool_context.fn_name}({tool_context.arg_repr}). Takes a single argument `answer`."
        )

        async def _clarify(answer: str) -> Dict[str, str]:  # type: ignore[valid-type]
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

        # ── honour handle.valid_tools, if present ──────────────
        if hasattr(handle, "valid_tools"):
            allowed: set[str] = set(getattr(handle, "valid_tools", []))
            public_methods = {
                name: bound for name, bound in public_methods.items() if name in allowed
            }

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

        if not info.waiting_for_clarification:
            self._create_continue_tool(create_tool_ctx)

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

        if info.clar_up_queue is not None:
            self._create_clarify_tool(create_tool_ctx)

        can_pause = (handle_available and hasattr(handle, "pause")) or task_pause_event
        if can_pause:
            self._create_pause_tool(
                create_tool_ctx,
                handle,
                task_pause_event,
            )

        can_resume = (
            handle_available and hasattr(handle, "resume")
        ) or task_pause_event
        if can_resume:
            self._create_resume_tool(
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
