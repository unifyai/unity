from __future__ import annotations

"""scenario_builder.py

Generic, manager‑agnostic **ScenarioBuilder** – a tiny utility that lets
unit‑tests spin up realistic *imaginary* data sets *entirely through the
public tool surfaces of any Manager* (ContactManager, TaskScheduler,
KnowledgeManager, …).

Usage
-----
>>> builder = ScenarioBuilder(
...     description="Populate a small CRM with five contacts and two follow‑up tasks",
...     tools={
...         "ask":    contact_manager.ask,
...         "update": contact_manager.update,
...     },
... )
>>> await builder.create()     # returns the assistant's confirmation

Design goals
------------
* **No hidden back‑doors** – the LLM can *only* interact with the supplied
  tool dict, mimicking how a real caller would build state.
* **Model‑agnostic** – managers and their tools are injected at runtime; the
  system prompt is built dynamically from the live arg‑specs.
* **Deterministic** – the outer create() call resolves only when the inner
  tool loop signals completion.
"""

import inspect
import json
from datetime import datetime, timezone
from typing import Callable, Dict, Any, Optional

import unify
from unity.common.llm_helpers import start_async_tool_use_loop, SteerableToolHandle
from sandboxes.utils import await_with_interrupt

__all__ = ["ScenarioBuilder"]


class ScenarioBuilder:
    """High‑level helper that orchestrates a *self‑contained* tool loop.

    Parameters
    ----------
    description
        A **detailed** textual brief capturing what the synthetic scenario
        should look like (e.g. *"Create three overdue invoices and two paid
        ones for customer ABC Corp"*).
    tools
        Mapping ``name → callable`` – *exactly* the helpers to expose to the
        LLM.  They may come from one manager (e.g. ContactManager) or be a
        blend from several.
    model
        Identifier of the chat‑completion model to use (defaults to the same
        *o4-mini* model the concrete managers rely on).
    traced
        Mirror the manager behaviour – when *True* every public method call is
        wrapped by :pyfunc:`unify.traced`.
    stateful
        If *True* the underlying :class:`unify.AsyncUnify` instance keeps
        conversational state across multiple `create()` invocations – handy
        when building *related* scenarios incrementally.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        description: str,
        tools: Dict[str, Callable],
        endpoint: str = "gpt-5@openai",
        traced: bool = True,
        stateful: bool = True,
        enable_voice: bool = False,
        clarifications_enabled: bool = True,
    ) -> None:
        if not tools:
            raise ValueError("ScenarioBuilder requires at least one tool.")

        self._description = description.strip()
        # Ensure the tool‑dict uses **exactly** the given callables –
        # no accidental leakage of private helpers.
        self._tools: Dict[str, Callable] = dict(tools)

        self._client = unify.AsyncUnify(
            endpoint,
            cache=True,
            traced=traced,
            stateful=stateful,
            reasoning_effort="high",
            service_tier="priority",
        )
        self._enable_voice = enable_voice
        self._clarifications_enabled = clarifications_enabled
        # System prompt is rebuilt lazily in .create() so that the arg‑specs
        # reflect any monkey‑patched callables between calls.

    # ------------------------------------------------------------------ #
    #  Public entry‑point                                                #
    # ------------------------------------------------------------------ #

    async def create(
        self,
        *,
        parent_chat_context: Optional[list[dict]] = None,
        _return_reasoning_steps: bool = False,
    ) -> Any | tuple[Any, list[dict]]:
        """Run the *inner* tool loop until the scenario is fully built.

        Returns
        -------
        Any | tuple
            The assistant's final confirmation (usually a short summary such
            as *"✓ Created 5 contacts and scheduled 2 follow‑up calls"*).  If
            *_return_reasoning_steps* is *True* the call yields a
            ``(answer, messages)`` pair where *messages* is the invisible
            chain‑of‑thought.
        """

        # 1️⃣  Build & inject system prompt
        self._client.set_system_message(self._build_system_prompt())

        # 2️⃣  Build wrappers that add clarification handling (when supported)
        wrapped_tools: Dict[str, Callable] = {}
        for name, fn in self._tools.items():
            sig = inspect.signature(fn)

            async def _wrapped(*args, __fn: Callable = fn, __sig: inspect.Signature = sig, **kwargs):  # type: ignore[no-redef]
                # Inject clarification queues only when the tool supports them
                clar_up = None
                clar_down = None
                if (
                    self._clarifications_enabled
                    and "clarification_up_q" in __sig.parameters
                    and "clarification_down_q" in __sig.parameters
                ):
                    import asyncio as _asyncio  # local import

                    clar_up = _asyncio.Queue()
                    clar_down = _asyncio.Queue()
                    kwargs = {
                        **kwargs,
                        "clarification_up_q": clar_up,
                        "clarification_down_q": clar_down,
                    }

                # Filter out any loop-internal kwargs (e.g. pause_event, interject_queue)
                # that the underlying tool does not support. Preserve extras only if the
                # original callable accepts **kwargs.
                try:
                    has_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in __sig.parameters.values()
                    )
                    if not has_var_kw:
                        kwargs = {
                            k: v for k, v in kwargs.items() if k in __sig.parameters
                        }
                except Exception:
                    # Defensive: if signature inspection fails, pass through existing kwargs
                    pass

                ret = __fn(*args, **kwargs)
                # Await if coroutine
                if inspect.isawaitable(ret):
                    ret = await ret  # type: ignore[assignment]

                # If the tool returned a SteerableToolHandle, resolve it with our interactive helper
                if isinstance(ret, SteerableToolHandle):
                    return await await_with_interrupt(
                        ret,
                        enable_voice_steering=self._enable_voice,
                        clarification_up_q=clar_up,
                        clarification_down_q=clar_down,
                        clarifications_enabled=self._clarifications_enabled,
                    )
                return ret

            # Expose the original tool's signature and docstring so downstream
            # tooling (schema generation, kwarg injection) sees the correct API.
            try:
                _wrapped.__signature__ = sig  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                _wrapped.__doc__ = getattr(fn, "__doc__", "")
            except Exception:
                pass

            wrapped_tools[name] = _wrapped

        # 3️⃣  Fire up the generic tool‑loop – the **description itself** acts
        #     as the initial *user* turn. Enforce that the LLM *must* call
        #     at least one tool (index 0) so generators like TranscriptGenerator
        #     are guaranteed to receive data.
        handle = start_async_tool_use_loop(
            self._client,
            self._description,
            wrapped_tools,
            loop_id=f"{self.__class__.__name__}.{self.create.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if not _return_reasoning_steps:
            return await handle.result()

        answer = await handle.result()
        return answer, self._client.messages

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        """Compose a dynamic system‑prompt from the live *tools* mapping."""
        sig_json = json.dumps(
            {n: str(inspect.signature(fn)) for n, fn in self._tools.items()},
            indent=4,
        )

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return "\n".join(
            [
                "You are a **Scenario Construction Assistant**.",
                "Your ONLY goal is to populate an *imaginary yet self‑consistent* data set",
                "suitable for automated tests.  Operate **strictly through the tools provided** –",
                "never fabricate side‑effects outside them.",
                "",
                "Stop creating data once ALL key aspects of the scenario description are faithfully",
                "represented in the underlying store(s).  If a tool call fails you may retry or use",
                "other tools, but finish with a short natural‑language *confirmation* describing",
                "what was created or updated (include identifiers where helpful).",
                "",
                "Tools (name → argspec):",
                sig_json,
                "",
                f"Current UTC time: {now}.",
            ],
        )
