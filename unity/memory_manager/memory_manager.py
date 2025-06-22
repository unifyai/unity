from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional

import unify

from ..common.llm_helpers import (
    methods_to_tool_dict,
    start_async_tool_use_loop,
    SteerableToolHandle,
)
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.base import BaseTranscriptManager
from ..transcript_manager.transcript_manager import TranscriptManager

from .base import BaseMemoryManager
from .prompt_builders import (
    build_update_contacts_prompt,
    build_update_bio_prompt,
    build_update_rolling_summary_prompt,
)


class MemoryManager(BaseMemoryManager):
    """Production implementation backed by *ContactManager* + *TranscriptManager*."""

    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        transcript_manager: Optional[BaseTranscriptManager] = None,
    ) -> None:
        self._contact_manager = contact_manager or ContactManager()
        self._transcript_manager = transcript_manager or TranscriptManager(
            contact_manager=self._contact_manager,
        )

    # ––– internal tool wrappers ––––––––––––––––––––––––––––––––––––––––––
    async def _tool_transcript_ask(self, text: str) -> str:
        """High‑level search over transcripts – returns plain answer text."""
        handle = await self._transcript_manager.ask(text)
        return await handle.result()

    async def _tool_contact_ask(self, text: str) -> str:
        """High‑level question about contacts – returns plain answer text."""
        handle = await self._contact_manager.ask(text)
        return await handle.result()

    async def _tool_contact_update(self, text: str) -> str:
        """Execute a *natural‑language* contact update via ContactManager.update."""
        handle = await self._contact_manager.update(text)
        return await handle.result()

    # Column‑specific wrappers (private) ----------------------------------
    async def _set_bio(self, *, contact_id: int, bio: str) -> str:
        outcome = self._contact_manager._update_contact(
            contact_id=contact_id,
            custom_fields={"bio": bio},
        )
        return json.dumps(outcome)

    async def _set_rolling_summary(
        self,
        *,
        contact_id: int,
        rolling_summary: str,
    ) -> str:
        outcome = self._contact_manager._update_contact(
            contact_id=contact_id,
            custom_fields={"rolling_summary": rolling_summary},
        )
        return json.dumps(outcome)

    # ––– public API –––––––––––––––––––––––––––––––––––––––––––––––––––––––
    async def update_contacts(
        self,
        transcript: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """Synchronise the Contacts table with a *fresh* transcript chunk."""

        # Build tools dict (high‑level only!)
        tools: Dict[str, Callable] = methods_to_tool_dict(
            self._tool_transcript_ask,
            self._tool_contact_ask,
            self._tool_contact_update,
            include_class_name=False,
        )

        # Optional clarification helper -----------------------------------
        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # LLM client -------------------------------------------------------
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_update_contacts_prompt(tools))

        # Kick off tool‑use loop ------------------------------------------
        prompt = (
            "Here is the *verbatim* transcript text you must process:\n\n" + transcript
        )
        handle = start_async_tool_use_loop(
            client,
            prompt,
            tools,
            loop_id=f"{self.__class__.__name__}.update_contacts",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """Refresh a single contact's **bio** field from new transcript text."""

        tools: Dict[str, Callable] = methods_to_tool_dict(
            self._tool_transcript_ask,
            self._tool_contact_ask,
            self._set_bio,
            include_class_name=False,
        )

        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_update_bio_prompt(tools))

        prompt_parts = [
            "Latest transcript chunk:",
            transcript,
            "\nCurrent stored bio (may be None):",  # show even if None
            str(latest_bio),
        ]
        prompt = "\n\n".join(prompt_parts)

        handle = start_async_tool_use_loop(
            client,
            prompt,
            tools,
            loop_id=f"{self.__class__.__name__}.update_contact_bio",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """Update the **rolling_summary** field for a single contact."""

        tools: Dict[str, Callable] = methods_to_tool_dict(
            self._tool_transcript_ask,
            self._tool_contact_ask,
            self._set_rolling_summary,
            include_class_name=False,
        )

        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_update_rolling_summary_prompt(tools))

        prompt_parts = [
            "Transcript excerpt (last ≈50 messages):",
            transcript,
            "\nCurrent rolling summary (may be None):",
            str(latest_rolling_summary),
        ]
        prompt = "\n\n".join(prompt_parts)

        handle = start_async_tool_use_loop(
            client,
            prompt,
            tools,
            loop_id=f"{self.__class__.__name__}.update_contact_rolling_summary",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle
