import os
import json
import asyncio
import functools
from typing import List, Dict, Optional, Union, Callable, Any

import unify
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from .types.message import Message
from .types.message_exchange_summary import MessageExchangeSummary
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from ..events.event_bus import EventBus, Event
from .prompt_builders import build_ask_prompt, build_summarize_prompt
from .base import BaseTranscriptManager


class TranscriptManager(BaseTranscriptManager):

    # Vector embedding column names
    _VEC_MSG = "content_emb"
    _VEC_SUM = "summary_emb"

    def __init__(
        self,
        event_bus: EventBus,
        *,
        traced: bool = True,
        contact_manager: Optional[BaseContactManager] = None,
    ) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        self._event_bus = event_bus
        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ContactManager(
                event_bus=event_bus,
                traced=traced,
            )

        self._tools = methods_to_tool_dict(
            self.summarize,
            self._contact_manager._search_contacts,
            self._search_messages,
            self._search_summaries,
            self._nearest_messages,
            include_class_name=False,
        )

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        event_bus.register_event_types(
            ["Messages", "MessageExchangeSummaries"],
        )
        self._transcripts_ctx = event_bus.ctxs["Messages"]
        self._summaries_ctx = event_bus.ctxs["MessageExchangeSummaries"]

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Question

    @functools.wraps(BaseTranscriptManager.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # ── 0.  Build the *live* tools-dict (may include clarification helper) ──
        tools = dict(self._tools)

        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """
                Query the user for more information about their question, and wait for the reply. Especially useful if their question feels incomplete, and more clarifying details would be useful. Please use this tool liberally if you're unsure, it's always better to ask than to do the wrong thing.
                """
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "TranscriptManager.ask was called without both "
                        "clarification queues but the model requested clarifications.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 1.  Build LLM client & inject dynamic system-prompt ───────────
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_ask_prompt(tools))

        # ── 2.  Launch the interactive tool-use loop ───────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        # ── 3.  Optionally wrap .result() to expose reasoning  ────────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

    # Summarize Exchange(s)

    @functools.wraps(BaseTranscriptManager.summarize, updated=())
    async def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> str:
        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]

        # ── 0.  Build LLM client ────────────────────────────────────────────
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_summarize_prompt(guidance))

        # ── 1.  Collect raw messages → JSON blob for the prompt ────────────
        msgs = self._search_messages(filter=f"exchange_id in {exchange_ids}")
        exchanges = {
            id: [m.content for m in msgs if m.exchange_id == id] for id in exchange_ids
        }
        latest_timestamp = max(m.timestamp for m in msgs)
        exchanges_json = json.dumps(exchanges, indent=2)

        # ── 2.  Optional request_clarification helper tool ─────────────────
        tools: dict[str, Callable] = {}
        if clarification_up_q is not None or clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                """Query the user for more information, and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 3.  Kick off the interactive loop (even if no tools) ───────────
        from unity.common.llm_helpers import start_async_tool_use_loop

        prompt = (
            f"Here are the raw messages for exchange_id(s) {exchange_ids}:\n"
            f"{exchanges_json}\n\nPlease produce a concise summary."
            + (f"\n\nAdditional guidance:\n{guidance}" if guidance else "")
        )

        handle = start_async_tool_use_loop(
            client,
            prompt,
            tools,
            parent_chat_context=parent_chat_context,
        )
        summary: str = await handle.result()
        await self._event_bus.publish(
            Event(
                type="MessageExchangeSummaries",
                timestamp=latest_timestamp,
                payload=MessageExchangeSummary(
                    summary=summary,
                    exchange_ids=exchange_ids,
                ),
            ),
        )
        return summary

    # Private #
    # --------#
    def _nearest_messages(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[Message]:
        """
        Return the *k* transcript messages whose **content** embedding is
        *closest* to the embedding of **text** (cosine similarity).

        Parameters
        ----------
        text : str
            Free-form query text to embed.
        k : int, default ``10``
            Number of neighbours to return.

        Returns
        -------
        list[Message]
            Messages sorted by **ascending** cosine distance (best match first).
        """
        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            sorting={
                f"cosine({self._VEC_MSG}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [Message(**lg.entries) for lg in logs]

    def _nearest_summaries(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[MessageExchangeSummary]:
        """
        Retrieve the *k* stored summaries whose **summary text** embedding is
        closest to the embedding of **text**.

        Parameters
        ----------
        text : str
            Query text.
        k : int, default ``10``
            How many nearest summaries to return.

        Returns
        -------
        list[MessageExchangeSummary]
            Summaries ordered by similarity (*lowest* cosine distance first).
        """

        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._summaries_ctx,
            sorting={
                f"cosine({self._VEC_SUM}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [MessageExchangeSummary(**lg.entries) for lg in logs]

    def _search_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Message]:
        """
        Fetch **raw transcript messages** matching an arbitrary Python
        boolean *filter*.

        Parameters
        ----------
        filter : str | None, default ``None``
            Expression evaluated against each :class:`Message`
            (e.g. ``"medium == 'email' and 'urgent' in content"``).
            ``None`` selects *all* messages.
        offset : int, default ``0``
            Zero-based index of the first result.
        limit : int, default ``100``
            Maximum number of messages to return.

        Returns
        -------
        list[Message]
            Matching messages in creation order.
        """
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Message(**lg.entries) for lg in logs]

    def _search_summaries(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[MessageExchangeSummary]:
        """
        Retrieve persisted **exchange summaries** selected by an arbitrary
        Python boolean *filter*.

        Parameters
        ----------
        filter : str | None, default ``None``
            Expression evaluated against each
            :class:`~MessageExchangeSummary`
            (e.g. ``"5 in exchange_ids and 'deadline' in summary"``).
        offset : int, default ``0``
            Start index for pagination.
        limit : int, default ``100``
            Maximum number of summaries to return.

        Returns
        -------
        list[MessageExchangeSummary]
            Summaries satisfying the filter in creation order.
        """
        logs = unify.get_logs(
            context=self._summaries_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [MessageExchangeSummary(**lg.entries) for lg in logs]
