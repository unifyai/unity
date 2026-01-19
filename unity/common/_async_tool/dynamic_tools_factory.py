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
from .images import (
    append_images_with_source,
)
from .utils import maybe_await
from unity.image_manager.types.image_refs import ImageRefs


class DynamicToolFactory:

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
        """Copy signature, annotations, and docstring from from_callable to to_wrapper.

        Notes
        -----
        - The 'self' parameter (if any) is stripped from both the signature and annotations.
        - If the source has a docstring, it is copied verbatim (stripped) onto the wrapper.
        - If the source method has no docstring, attempt to fall back to the first
          ancestor in the MRO that defines a docstring for a method with the same name.
        """
        try:
            src = getattr(from_callable, "__func__", from_callable)
            # Build a new signature that removes any leading 'self' parameter
            _sig = inspect.signature(src)
            try:
                _params = list(_sig.parameters.values())
            except Exception:
                _params = []
            try:
                _filtered_params = [p for p in _params if p.name != "self"]
            except Exception:
                _filtered_params = _params
            try:
                to_wrapper.__signature__ = inspect.Signature(
                    parameters=_filtered_params,
                    return_annotation=_sig.return_annotation,
                )
            except Exception:
                # Fallback: if building a filtered signature fails, at least set the original one
                to_wrapper.__signature__ = _sig
            try:
                ann = dict(getattr(src, "__annotations__", {}) or {})
                ann.pop("self", None)
                to_wrapper.__annotations__ = ann
            except Exception:
                pass
            try:
                doc = inspect.getdoc(src)
                if isinstance(doc, str) and doc.strip():
                    to_wrapper.__doc__ = doc.strip()
                else:
                    # Fallback: walk MRO to find a base-class method docstring
                    try:
                        name = getattr(src, "__name__", None) or getattr(
                            from_callable,
                            "__name__",
                            "",
                        )
                        owner_cls = getattr(
                            getattr(from_callable, "__self__", None),
                            "__class__",
                            None,
                        )
                        if isinstance(name, str) and name and owner_cls is not None:
                            for base in getattr(owner_cls, "__mro__", ())[1:]:
                                try:
                                    cand = getattr(base, name, None)
                                except Exception:
                                    cand = None
                                if cand is None:
                                    continue
                                fn_obj = getattr(cand, "__func__", cand)
                                base_doc = inspect.getdoc(fn_obj)
                                if isinstance(base_doc, str) and base_doc.strip():
                                    to_wrapper.__doc__ = base_doc.strip()
                                    break
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _ensure_kwonly_param(to_wrapper, name: str, annotation, default=None) -> None:
        """Ensure a keyword-only parameter exists on the wrapper's signature."""
        with suppress(Exception):
            import inspect as _inspect

            sig = _inspect.signature(to_wrapper)
            params = list(sig.parameters.values())
            if any(p.name == name for p in params):
                return
            params.append(
                _inspect.Parameter(
                    name,
                    kind=_inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                ),
            )
            to_wrapper.__signature__ = _inspect.Signature(
                parameters=params,
                return_annotation=sig.return_annotation,
            )
            anns = dict(getattr(to_wrapper, "__annotations__", {}))
            anns[name] = annotation
            to_wrapper.__annotations__ = anns

    @staticmethod
    def _discover_custom_public_methods(handle) -> dict[str, Callable]:
        """
        Return a mapping ``name → bound_method`` of *public* callables on *handle*:
            • name does **not** start with ``_``  _and_
            • name is not a core steering method defined on base async-tool loop handles
              (SteerableToolHandle, AsyncToolLoopHandle).
        """

        def _management_method_names_for_handle(_h) -> set[str]:
            names: set[str] = set()
            mro = getattr(getattr(_h, "__class__", object), "__mro__", ())
            for base in mro:
                bmod = getattr(base, "__module__", "")
                bname = getattr(base, "__name__", "")
                if bmod == "unity.common.async_tool_loop" and bname in (
                    "SteerableToolHandle",
                    "AsyncToolLoopHandle",
                ):
                    for n, member in inspect.getmembers(base, inspect.isroutine):
                        if not n.startswith("_"):
                            names.add(n)
            return names

        management_names = _management_method_names_for_handle(handle)
        methods: dict[str, Callable] = {}
        for name, attr in inspect.getmembers(handle):
            if name.startswith("_") or name in management_names or not callable(attr):
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
            "images : ImageRefs | None\n"
            "    Optional image references to append at the time of this command (same `ImageRefs` model as `start_async_tool_loop`).\n\n"
            "Returns\n"
            "-------\n"
            "Dict[str, str]\n"
            "    Status acknowledgement including the underlying call id.\n\n"
            "Notes\n"
            "-----\n"
            "- Images are appended to the live images log immediately.\n"
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
                append_images_with_source(_kw.get("images"))
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

        # Set fallback docstring first; _adopt_signature_and_annotations may override
        _stop.__doc__ = doc
        # Expose full argspec and docstring of handle.stop in the helper schema
        with suppress(Exception):
            if handle is not None and hasattr(handle, "stop"):
                self._adopt_signature_and_annotations(getattr(handle, "stop"), _stop)
        # Ensure images kw-only param
        self._ensure_kwonly_param(_stop, "images", Optional[ImageRefs], default=None)
        # Register after adopting signature/doc so factory falls back only when needed
        self._register_tool(
            func_name=f"stop_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
            fn=_stop,
        )

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
            "images : ImageRefs | None\n"
            "    Optional image references to append at the time of this interjection (same `ImageRefs` model as `start_async_tool_loop`).\n\n"
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
                    append_images_with_source(_kw.get("images"))
                except Exception:
                    pass
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **{k: v for k, v in _kw.items() if k != "images"},
                }

            # Set fallback docstring first; _adopt_signature_and_annotations may override
            _interject.__doc__ = doc
            # Expose the downstream handle's signature to the LLM and ensure common params
            with suppress(Exception):
                if hasattr(handle, "interject"):
                    self._adopt_signature_and_annotations(
                        getattr(handle, "interject"),
                        _interject,
                    )
            # Ensure `content` alias and `images` kw-only parameters exist
            self._ensure_kwonly_param(
                _interject,
                "content",
                Optional[str],
                default=None,
            )
            self._ensure_kwonly_param(
                _interject,
                "images",
                Optional[ImageRefs],
                default=None,
            )

        else:

            async def _interject(
                *,
                content: Optional[str] = None,
                message: Optional[str] = None,
                images: ImageRefs | None = None,
            ) -> Dict[str, str]:
                # regular tool: push onto its private queue
                actual = content if content is not None else (message or "")
                await task_info.interject_queue.put(actual)
                # Append any provided images into the live registry/log
                try:
                    append_images_with_source(images)
                except Exception:
                    pass
                return {
                    "status": "interjected",
                    "call_id": tool_context.call_id,
                    **({"content": actual} if actual else {}),
                }

            # Set fallback docstring (no handle to adopt from in this branch)
            _interject.__doc__ = doc

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
                "images": Optional[ImageRefs],
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
                        annotation=Optional[ImageRefs],
                    ),
                ],
                return_annotation=Dict[str, str],
            )

    def _create_ask_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
    ) -> None:
        """
        Expose a synthetic helper to invoke the handle's `ask` method for inspection.

        Behaviour
        ---------
        - For nested async handles, returns the downstream handle so the loop can adopt
          and await its result (consistent with base-tool behaviour).
        - Otherwise returns the direct answer value from the handle.
        """

        if handle is None or not hasattr(handle, "ask"):
            return

        async def _ask(**_kw):
            # Robust forwarding with support for positional fallback (question)
            return await forward_handle_call(
                handle,
                "ask",
                _kw,
                fallback_positional_keys=["question"],
            )

        # Set fallback docstring first; _adopt_signature_and_annotations may override
        _ask.__doc__ = (
            f"Ask a read-only question about the running call {tool_context.fn_name}({tool_context.arg_repr}).\n\n"
            "Returns either a nested handle (adopted by the loop) or a direct answer."
        )
        # Reflect downstream signature/annotations for clean tool schema
        with suppress(Exception):
            self._adopt_signature_and_annotations(getattr(handle, "ask"), _ask)

        self._register_tool(
            func_name=f"ask_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=_ask.__doc__,
            fn=_ask,
        )

    def _create_clarify_tool(
        self,
        tool_context: _ToolContext,
        handle: Any,
    ) -> None:
        doc = (
            f"Provide an answer to the clarification which was requested by the (currently pending) tool "
            f"{tool_context.fn_name}({tool_context.arg_repr}).\n\n"
            "Parameters\n"
            "----------\n"
            "answer : str\n"
            "    The answer text.\n"
            "images : ImageRefs | None\n"
            "    Optional image references to append alongside this answer (same `ImageRefs` model as `start_async_tool_loop`).\n\n"
            "Returns\n"
            "-------\n"
            "Dict[str, str]\n"
            "    Status acknowledgement including the underlying call id.\n"
        )

        async def _clarify(answer: str, images: ImageRefs | None = None) -> Dict[str, str]:  # type: ignore[valid-type]
            try:
                append_images_with_source(images)
            except Exception:
                pass
            return {
                "status": "clar_answer",
                "call_id": tool_context.call_id,
                "answer": answer,
            }

        # Set fallback docstring first; handle docstring may override below
        _clarify.__doc__ = doc
        # Prefer to propagate a class method docstring when available (e.g., handle.answer_clarification)
        with suppress(Exception):
            if handle is not None and hasattr(handle, "answer_clarification"):
                src = getattr(handle, "answer_clarification")
                src_doc = inspect.getdoc(getattr(src, "__func__", src))
                if isinstance(src_doc, str) and src_doc.strip():
                    _clarify.__doc__ = src_doc.strip()

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
        doc = f"Pause the pending call {tool_context.fn_name}({tool_context.arg_repr})."

        if handle_available and hasattr(handle, "pause"):

            async def _pause(**_kw) -> Dict[str, str]:
                with suppress(Exception):
                    await forward_handle_call(handle, "pause", _kw)
                return {"status": "paused", "call_id": tool_context.call_id, **_kw}

            # Set fallback docstring first; _adopt_signature_and_annotations may override
            _pause.__doc__ = doc
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

            # Set fallback docstring (no handle to adopt from in this branch)
            _pause.__doc__ = doc

        self._register_tool(
            func_name=f"pause_{tool_context.fn_name}_{tool_context.safe_call_id}",
            fallback_doc=doc,
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

            # Set fallback docstring first; _adopt_signature_and_annotations may override
            _resume.__doc__ = doc
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

            # Set fallback docstring (no handle to adopt from in this branch)
            _resume.__doc__ = doc

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

            # Override the wrapper's signature and annotations to match the real method
            _invoke_handle_method.__signature__ = inspect.signature(bound)
            # Also copy annotations so downstream schema generation preserves types (e.g., int)
            self._adopt_signature_and_annotations(bound, _invoke_handle_method)

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

        if info.clar_up_queue is not None:
            self._create_clarify_tool(create_tool_ctx, handle)

        # Synthetic `ask` helper for LLM-accessible inspection
        if handle_available:
            self._create_ask_tool(create_tool_ctx, handle)

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
