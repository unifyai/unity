import os
import json
import asyncio
import functools
from typing import List, Dict, Optional, Union

import unify
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from .types.message import Message
from ..common.model_to_fields import model_to_fields
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from .prompt_builders import build_ask_prompt
from .base import BaseTranscriptManager


class TranscriptManager(BaseTranscriptManager):

    # Vector embedding column names
    _MSG_EMB = "_content_emb"

    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ContactManager()

        self._tools = methods_to_tool_dict(
            self._contact_manager._search_contacts,
            self._search_messages,
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
        else:
            self._messages_ctx = "Contacts"
        ctxs = unify.get_contexts()
        if self._messages_ctx not in ctxs:
            unify.create_context(
                self._messages_ctx,
                unique_column_ids="message_id",
                description="List of *all* timestamped messages sent between *all* contacts across *all* mediums.",
            )
            fields = model_to_fields(Message)
            unify.create_fields(
                fields,
                context=self._messages_ctx,
            )

        # ── Async logging (mirrors EventBus) ────────────────────────────────
        # Using a dedicated logger means log_create() returns immediately,
        # leaving the actual network I/O to an internal worker thread.
        self._logger = unify.AsyncLoggerManager()
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

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
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> SteerableToolHandle:
        # ── 0.  Build the *live* tools-dict (may include clarification helper) ──
        tools = dict(self._tools)

        # ── 0b.  Create a call-ID & log the incoming request ────────────────
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "TranscriptManager",
            "ask",
            phase="incoming",
            question=text,
        )

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
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(tools, include_activity=include_activity),
        )

        # ── 2.  Launch the interactive tool-use loop ───────────────────────
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        # ── 3.  Wrap with logging (outgoing, pause, …)  ─────────────────────
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "TranscriptManager",
            "ask",
        )

        # ── 4.  Optional reasoning exposure  ───────────────────────────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    async def summarize(self, *args, **kwargs):
        """Deprecated: summarize functionality removed."""
        raise NotImplementedError(
            "Summarize functionality has been removed from TranscriptManager.",
        )

    # Helpers #
    # --------#
    def log_messages(
        self,
        messages: Union[Union[Dict, Message], List[Union[Dict, Message]]],
    ) -> None:
        """
        Insert one or more messages into the backing store.

        Parameters
        ----------
        messages : dict | Message | list[dict | Message]
            One or more messages to log. Each message can be either:
            - A dictionary whose keys conform to the
              :class:`unity.transcript_manager.types.message.Message` schema
              (``medium``, ``sender_id``, ``receiver_id``, ``timestamp``,
              ``content``, ``exchange_id`` …)
            - A Message object
            - A list containing any combination of the above
        """
        # Fast-return if nothing to log --------------------------------------
        if not messages:
            return

        if not isinstance(messages, list):
            messages = [messages]

        msg_entries = [
            msg.to_post_json() if isinstance(msg, Message) else msg for msg in messages
        ]
        messages = [
            msg if isinstance(msg, Message) else Message(**msg) for msg in messages
        ]

        from ..events.event_bus import EVENT_BUS, Event  # local import to avoid cycles

        async def _publish_message(msg: Message) -> None:
            try:
                await EVENT_BUS.publish(
                    Event(
                        type="Message",
                        timestamp=msg.timestamp,
                        payload=msg,
                    ),
                )
            except Exception:
                # Defensive – never propagate EventBus issues to caller
                pass

        for entries, msg in zip(msg_entries, messages):
            self._logger.log_create(
                project=unify.active_project(),
                context=self._messages_ctx,
                params={},
                entries=entries,
            )

            try:
                # If we're already inside an event-loop schedule the coroutine there …
                loop = asyncio.get_running_loop()
                loop.create_task(_publish_message(msg))
            except RuntimeError:
                # … otherwise create a *temporary* loop to run it synchronously so the
                # event doesn't get lost in purely synchronous contexts (e.g. tests).
                asyncio.run(_publish_message(msg))

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

    # _nearest_summaries removed – summary functionality deprecated.
    def _nearest_summaries(self, *args, **kwargs):
        raise NotImplementedError("Summary functionality removed.")

    # ------------------------------------------------------------------ #
    #  Reset helper (sandbox)                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def reset() -> None:
        """
        Delete the `Transcripts` contexts (and any namespaced variants) for the
        *current* Unify project so that a clean slate is created when a
        new TranscriptManager is instantiated.
        """
        import unify

        targets = [
            ctx
            for ctx in list(unify.get_contexts())
            if ctx == "Transcripts" or ctx.endswith("/Transcripts")
        ]
        for ctx in targets:
            try:
                unify.delete_context(ctx)
            except Exception:
                pass

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

    # _search_summaries removed – summary functionality deprecated.
    def _search_summaries(self, *args, **kwargs):
        raise NotImplementedError("Summary functionality removed.")
