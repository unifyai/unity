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
from ..common.model_to_fields import model_to_fields
from .types.message_exchange_summary import MessageExchangeSummary
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from .prompt_builders import build_ask_prompt, build_summarize_prompt
from .base import BaseTranscriptManager


class TranscriptManager(BaseTranscriptManager):

    # Vector embedding column names
    _MSG_EMB = "_content_emb"
    _SUM_EMB = "_summary_emb"

    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
    ) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ContactManager()

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

        if read_ctx:
            self._messages_ctx = f"{read_ctx}/Messages"
            self._summaries_ctx = f"{read_ctx}/MessageExchangeSummaries"
        else:
            self._messages_ctx = "Contacts"
            self._summaries_ctx = "MessageExchangeSummaries"
        ctxs = unify.get_contexts()
        if self._messages_ctx not in ctxs:
            unify.create_context(
                self._messages_ctx,
                unique_id_column=True,
                unique_id_name="message_id",
                description="List of *all* timestamped messages sent between *all* contacts across *all* mediums.",
            )
            fields = model_to_fields(Message)
            unify.create_fields(
                fields,
                context=self._messages_ctx,
            )
        if self._summaries_ctx not in ctxs:
            unify.create_context(
                self._summaries_ctx,
                unique_id_column=True,
                unique_id_name="summary_id",
                description="List of all message exchange summaries, with each summary covering a fixed number of exchanges.",
            )
            fields = model_to_fields(MessageExchangeSummary)
            unify.create_fields(
                fields,
                context=self._summaries_ctx,
            )

        # ── Async logging (mirrors EventBus) ────────────────────────────────
        # Using a dedicated logger means log_create() returns immediately,
        # leaving the actual network I/O to an internal worker thread.
        self._logger = unify.AsyncLoggerManager()

    # Public #
    # -------#

    # English-Text Question

    @functools.wraps(BaseTranscriptManager.ask, updated=())
    async def ask(
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
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
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
        from_exchanges: Optional[Union[int, List[int]]] = None,
        from_messages: Optional[Union[int, List[int]]] = None,
        omit_messages: Optional[List[int]] = None,
        guidance: Optional[str] = None,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> SteerableToolHandle:
        # -- 0.  Validate & canonicalise ------------------------------------
        if from_exchanges is None and from_messages is None:
            raise ValueError(
                "Either 'from_exchanges' or 'from_messages' must be provided.",
            )

        if isinstance(from_exchanges, int):
            from_exchanges = [from_exchanges]
        if isinstance(from_messages, int):
            from_messages = [from_messages]
        from_exchanges = list(from_exchanges or [])
        from_messages = list(from_messages or [])
        omit_messages = set(omit_messages or [])

        # ── 1.  Build LLM client ────────────────────────────────────────────
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(build_summarize_prompt(guidance))

        # ── 2.  Collect raw messages – single back-end filter ---------------
        inc_clauses: list[str] = []
        if from_exchanges:
            inc_clauses.append(f"exchange_id in {from_exchanges}")
        if from_messages:
            inc_clauses.append(f"message_id in {from_messages}")

        include_expr = " or ".join(f"({c})" for c in inc_clauses)
        filter_expr = include_expr
        if omit_messages:
            filter_expr = (
                f"({include_expr}) and (message_id not in {list(omit_messages)})"
            )

        msgs: list[Message] = self._search_messages(filter=filter_expr, limit=None)

        #  Group by exchange so the LLM can see conversation structure
        exchanges: dict[int, list[str]] = {}
        for m in msgs:
            exchanges.setdefault(m.exchange_id, []).append(m.content)

        message_ids_sorted = sorted(m.message_id for m in msgs)
        exchanges_json = json.dumps(exchanges, indent=2)

        # ── 3.  Optional request_clarification helper tool ─────────────────
        tools: dict[str, Callable] = {}
        if (
            clarification_up_q is not None
            and clarification_down_q is not None
            and guidance is not None
        ):

            # clarification capability only added if explicit guidance is given.
            async def request_clarification(question: str) -> str:
                """Query the user for more information, and wait for the reply."""
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError("Clarification queues missing")
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        # ── 4.  Kick off the interactive loop (even if no tools) ───────────
        from unity.common.llm_helpers import start_async_tool_use_loop

        prompt = (
            "Here are the raw messages selected for this summary "
            f"(grouped by exchange_id):\n{exchanges_json}\n\nPlease "
            "produce a concise cross-exchange summary."
            + (f"\n\nAdditional guidance:\n{guidance}" if guidance else "")
        )

        handle = start_async_tool_use_loop(
            client,
            prompt,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.summarize.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: (
                ("required", _) if (tools and i < 1) else ("auto", _)
            ),
        )

        # Wrap the original result to log the summary when it completes
        original_result = handle.result

        async def wrapped_result():
            summary = await original_result()
            ex_ids_for_log = sorted(exchanges.keys())
            unify.log(
                context=self._summaries_ctx,
                exchange_ids=ex_ids_for_log,
                message_ids=message_ids_sorted,
                summary=summary,
                new=True,
                mutable=True,
            )
            return summary

        handle.result = wrapped_result

        return handle

    # Helpers #
    # --------#
    def log_message(self, message: Union[Dict, Message]) -> None:
        """
        Insert messages into the backing store.

        Parameters
        ----------
        message : dict | Message
            Either a dictionary whose keys conform to the
            :class:`unity.transcript_manager.types.message.Message` schema
            (``medium``, ``sender_id``, ``receiver_id``, ``timestamp``,
            ``content``, ``exchange_id`` …), or a Message object.
        """
        # Fast-return if nothing to log --------------------------------------
        if not message:
            return

        if isinstance(message, Message):
            message = message.to_post_json()

        self._logger.log_create(
            project=unify.active_project(),
            context=self._messages_ctx,
            params={},
            entries=message,
        )

    def join_published(self):
        self._logger.join()

    # Tools #
    # ------#
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
        ensure_vector_column(self._messages_ctx, self._MSG_EMB, "content")
        logs = unify.get_logs(
            context=self._messages_ctx,
            sorting={
                f"cosine({self._MSG_EMB}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._messages_ctx).keys()
                if k.endswith("_emb")
            ],
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

        ensure_vector_column(self._summaries_ctx, self._SUM_EMB, "summary")
        logs = unify.get_logs(
            context=self._summaries_ctx,
            sorting={
                f"cosine({self._SUM_EMB}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._summaries_ctx).keys()
                if k.endswith("_emb")
            ],
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
            context=self._messages_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            sorting={"timestamp": "descending"},
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._messages_ctx).keys()
                if k.endswith("_emb")
            ],
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
            sorting={"timestamp": "descending"},
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._summaries_ctx).keys()
                if k.endswith("_emb")
            ],
        )
        return [MessageExchangeSummary(**lg.entries) for lg in logs]
