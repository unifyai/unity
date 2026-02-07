from __future__ import annotations

import asyncio
import json
import textwrap
from contextlib import suppress
from typing import Any, Optional

import unillm

from .llm_client import new_llm_client
from .async_tool_loop import AsyncToolLoopHandle
from pydantic import BaseModel, Field


def read_only_ask_mutation_exit_block() -> str:
    """
    Return a canonical, manager-agnostic prompt section instructing the model
    to exit immediately when a mutation request reaches a read-only ask/retrieve
    channel. This block is reused across managers to keep wording consistent.
    """
    return textwrap.dedent(
        """
        Early exit on mutation requests
        ------------------------------
        • If the incoming request asks to create, update, delete, merge, rename, reorder, move, ingest, transform, refactor, set a field, change values, or otherwise modify state, EXIT IMMEDIATELY.
        • Do not call any tools. Do not propose steps. Do not ask questions.
        • Return exactly ONE short sentence that:
          - clearly states this read‑only channel cannot make changes;
          - avoids naming specific mutation tools or methods;
          - may generically note that a separate mutation/write request is required;
          - may optionally add that you can answer questions about existing data only.
        """,
    ).strip()


class ReadOnlyAskGuardHandle(AsyncToolLoopHandle):
    """
    A reusable outer handle that runs a tiny, parallel LLM classification to detect
    mutation intent on read-only ask() calls. If mutation intent is detected, the
    outer loop is stopped immediately and the classifier's early response is used
    as the final result.

    Behaviour
    ---------
    - Launches a short-lived classification task on construction using only the
      initial user message text (no internal tool context).
    - If the classifier returns mutation_intent=True, we call stop() on the
      running loop and cache the early_response.
    - Overrides result() to return the cached early_response when the outer loop
      has been stopped early; otherwise, defers to the base implementation.
    """

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[dict | str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        pause_event: Optional[asyncio.Event] = None,
        client: "unillm.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[Any] = None,
        response_format: Optional[Any] = None,
    ):
        super().__init__(
            task=task,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            stop_event=stop_event,
            pause_event=pause_event,
            client=client,
            loop_id=loop_id,
            initial_user_message=initial_user_message,
            response_format=response_format,
        )

        self._early_result: Optional[str] = None
        self._cls_task: Optional[asyncio.Task] = None

        # Kick off classifier concurrently; do not block the main loop.
        try:
            self._cls_task = asyncio.create_task(self._classify_and_maybe_stop())
        except Exception:
            # Defensive: never let classification failure affect the main loop
            self._cls_task = None

        # Best-effort: when the main loop task completes, cancel classifier
        # if it's still running to avoid leaks.
        try:
            self._task.add_done_callback(lambda _t: self._cancel_classifier())
        except Exception:
            pass

    def _cancel_classifier(self) -> None:
        with suppress(Exception):
            if self._cls_task and not self._cls_task.done():
                self._cls_task.cancel()

    def _initial_text(self) -> str:
        # Normalise initial message into a plain text string.
        try:
            msg = getattr(self, "_user_visible_history", [])
            if msg:
                c = msg[0].get("content")
                if isinstance(c, dict):
                    return str(c.get("message", ""))
                return str(c or "")
        except Exception:
            pass
        try:
            raw = getattr(self, "_loop_id", "")
        except Exception:
            raw = ""
        return str(raw or "")

    async def _classify_and_maybe_stop(self) -> None:
        class _AskGuardSchema(BaseModel):
            mutation_intent: bool = Field(
                ...,
                description="Whether the user is asking for a state mutation",
            )
            early_response: str = Field(
                default="",
                description="A concise assistant reply to return immediately when mutation is detected",
            )

        # Build classification client
        cls_client = new_llm_client()

        # Derive a concise label so the LLM knows which manager/method this is
        label = "ask"
        with suppress(Exception):
            if getattr(self, "_log_label", None):
                # _log_label typically looks like "ContactManager.ask(x2ab)"
                base = str(getattr(self, "_log_label"))
                label = base.split("(")[0]
            elif getattr(self, "_loop_id", None):
                label = str(getattr(self, "_loop_id"))

        # Only the initial text by design
        user_text = self._initial_text()

        # Very concise instruction – return minimal JSON
        sys_msg = (
            "You classify whether a read-only ask() request actually tries to mutate state.\n"
            "Return a JSON object matching the required schema.\n"
            "If mutation_intent is false, early_response may be an empty string.\n"
            f"Manager/Method: {label}\n"
            "\n"
            "If mutation_intent is true, craft early_response as ONE short sentence that:\n"
            "- clearly states this ask channel is read-only and cannot make changes;\n"
            "- does not propose drafting, planning, or performing changes;\n"
            "- does not ask the user follow-up questions;\n"
            "- avoids naming specific mutation methods or tools (do not guess names like 'update');\n"
            "- may generically say a separate mutation request/write operation is required;\n"
            "- may optionally add that you can answer questions about existing data only.\n"
            "Prefer plain, declarative wording without offers (e.g., 'I can’t make changes from this read‑only ask channel.')."
        )
        cls_client.set_system_message(sys_msg)
        cls_client.set_response_format(_AskGuardSchema)

        payload = user_text

        async def _run():
            try:
                res = await asyncio.wait_for(
                    cls_client.generate(
                        return_full_completion=False,
                        stateful=False,
                        tools=None,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Classify this user text using the active response format schema.\n\n"
                                    + str(payload)
                                ),
                            },
                        ],
                    ),
                    timeout=60.0,
                )
                return str(res)
            except Exception as _exc:  # noqa: BLE001
                return None

        raw = await _run()
        if not raw:
            return

        # With set_response_format, the client should already return strict JSON
        # matching the schema; parse directly.
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return
        mut = bool(data.get("mutation_intent", False))
        early = str(data.get("early_response", ""))
        if not mut:
            return

        # Cache early result and stop the outer loop. Also record in user-visible history.
        self._early_result = early or ""
        with suppress(Exception):
            if self._early_result:
                self._append_user_visible_assistant(self._early_result)
                # Also append into the public chat transcript so observers see the final answer
                if self._client is not None:
                    try:
                        self._client.append_messages(
                            [
                                {
                                    "role": "assistant",
                                    "content": self._early_result,
                                },
                            ],
                        )
                    except Exception:
                        pass
        try:
            # Use public stop API; downstream loop will cancel promptly
            self.stop(reason="mutation intent detected in ask()")
        except Exception:
            pass

    async def result(self) -> str:  # type: ignore[override]
        # If a classifier is still running, give it a brief chance to finish so
        # we can return the early response deterministically in tests.
        try:
            if self._cls_task is not None and not self._cls_task.done():
                # Wait up to the classifier's own timeout to get a definitive answer
                await asyncio.wait_for(self._cls_task, timeout=60.0)
        except asyncio.CancelledError:
            # Classifier was cancelled (e.g., outer loop stopped) — ignore
            pass
        except Exception:
            # Any other classifier failure should never break result()
            pass

        # If we already have an early response, prefer it (the loop will be or
        # has been stopped by the classifier).
        if self._early_result is not None and self._early_result != "":
            return self._early_result

        res = await super().result()

        # If the outer loop ended with a stop and the classifier populated a
        # result right after, return it; else return the base result.
        if self._early_result is not None and self._early_result != "":
            return self._early_result
        return res
