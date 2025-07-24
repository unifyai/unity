import os
import json
import asyncio
import functools
from typing import List, Dict, Optional, Union, Any

import unify
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from .types.message import Message

# New: allow Contact objects to appear in messages
from ..contact_manager.types.contact import Contact
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
    _LOGGER = unify.AsyncLoggerManager(name="TranscriptManager", num_consumers=16)

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
            self._transcripts_ctx = f"{read_ctx}/Transcripts"
        else:
            self._transcripts_ctx = "Transcripts"
        ctxs = unify.get_contexts()
        if self._transcripts_ctx not in ctxs:
            unify.create_context(
                self._transcripts_ctx,
                unique_column_ids="message_id",
                description="List of *all* timestamped messages sent between *all* contacts across *all* mediums.",
            )
            fields = model_to_fields(Message)
            unify.create_fields(
                fields,
                context=self._transcripts_ctx,
            )

        # ── Async logging (mirrors EventBus) ────────────────────────────────
        # Using a dedicated logger means log_create() returns immediately,
        # leaving the actual network I/O to an internal worker thread.
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

    @classmethod
    def _get_logger(cls) -> unify.AsyncLoggerManager:
        return cls._LOGGER

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
            preprocess_msgs=self._inject_broader_context,
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
        messages: Union[
            Union[Dict[str, Any], Message],
            List[Union[Dict[str, Any], Message]],
        ],
        synchronous: bool = False,
    ) -> None:
        """
        Insert one or more messages into the backing store.

        This enhanced variant additionally accepts **Contact** objects in place
        of numeric ``sender_id`` / ``receiver_ids``.  When such a Contact has
        its ``contact_id`` set to the sentinel ``-1`` (meaning *not yet
        persisted*) the contact is **created on-the-fly** via
        :pyfunc:`ContactManager._create_contact` before the message is logged.

        Parameters
        ----------
        messages : dict | Message | list[dict | Message]
            One or more messages to log. Each message can be either:
            - A dictionary following the
              :class:`unity.transcript_manager.types.message.Message` schema –
              where ``sender_id`` / ``receiver_ids`` may contain ``Contact``
              objects instead of ints.
            - A :class:`~unity.transcript_manager.types.message.Message` instance
              whose *id* fields may likewise contain ``Contact`` objects.
            - A list with any combination of the above.
        synchronous : bool, default=False
            If True, messages will be logged in order synchronously. If False,
            messages may be logged asynchronously in any order.
        """

        # ── 0. Early-exit on empty input ────────────────────────────────────
        if not messages:
            return

        if not isinstance(messages, list):
            messages = [messages]

        # ── 1. Helper to ensure we have a numeric contact-id ───────────────
        # Derive the built-in (canonical) Contact fields *dynamically* from the
        # `Contact` model itself – this avoids hard-coding and ensures there is
        # exactly one source of truth across the code-base.
        built_in_fields = set(Contact.model_fields.keys())

        contact_cache: Dict[int, int] = {}

        def _ensure_contact_id(c: Union[int, Contact]) -> int:
            """Return an existing or newly-created **contact_id** for *c*."""

            # Fast-path: already an int → nothing to do
            if not isinstance(c, Contact):
                if c is None:
                    raise ValueError(
                        "sender_id / receiver_ids cannot be None – either provide an int or a Contact instance.",
                    )
                return int(c)

            # If the Contact already had a valid id – reuse it
            if c.contact_id is not None and c.contact_id != -1:
                return int(c.contact_id)

            # Deduplicate identical Contact objects within the same call
            obj_key = id(c)
            if obj_key in contact_cache:
                return contact_cache[obj_key]

            # Build kwargs for _create_contact using *non-None* built-in fields
            # detected directly from the `Contact` schema instead of hard-coding
            # the field names.  This ensures any future additions to the
            # Contact model automatically propagate here.

            full_data = c.model_dump(exclude_none=True)  # include only defined fields

            # Separate canonical Contact fields from any custom extras
            create_kwargs: Dict[str, Any] = {
                k: v
                for k, v in full_data.items()
                if k in built_in_fields and k != "contact_id"
            }

            # Capture any extra / custom fields present on the Contact
            custom_fields = {
                k: v
                for k, v in c.model_dump().items()
                if k not in built_in_fields and v is not None
            }
            if custom_fields:
                create_kwargs["custom_fields"] = custom_fields

            # Synchronously create the new contact
            outcome = self._contact_manager._create_contact(**create_kwargs)
            try:
                new_cid = int(outcome["details"]["contact_id"])
            except Exception:
                # Fall back to best-effort id extraction / raise
                raise RuntimeError(
                    "Failed to extract contact_id from ContactManager outcome: "
                    f"{outcome}",
                )

            # Update cache and the original Contact instance for consistency
            contact_cache[obj_key] = new_cid
            try:
                c.contact_id = new_cid  # type: ignore[attr-defined]
            except Exception:
                pass  # read-only / frozen instance – safe to ignore

            return new_cid

        # ── 2. Normalise each input payload into Message objects ───────────
        normalised_messages: List[Message] = []
        for raw in messages:
            # Convert to dict early so we can mutate fields easily
            if isinstance(raw, Message):
                payload: Dict[str, Any] = raw.model_dump(mode="python")
            else:  # assume mapping
                payload = dict(raw)

            # Ensure required keys exist
            if "receiver_ids" not in payload:
                raise ValueError("Each message must include 'receiver_ids'.")

            # Replace any Contact objects with their numeric ids
            payload["sender_id"] = _ensure_contact_id(payload.get("sender_id"))
            payload["receiver_ids"] = [
                _ensure_contact_id(r) for r in payload.get("receiver_ids", [])
            ]

            # Re-instantiate Message model for validation
            normalised_messages.append(Message(**payload))

        # ── 3. Dump POST-ready JSON for each message ──────────────────────
        msg_entries = [m.to_post_json() for m in normalised_messages]

        # ── 4. Persist messages and publish EventBus notifications ───────
        from ..events.event_bus import EVENT_BUS, Event  # local import to avoid cycles

        async def _publish_message(msg: Message) -> None:
            try:
                await EVENT_BUS.publish(
                    Event(
                        type="Message",
                        timestamp=msg.timestamp,
                        payload=msg,
                    ),
                    blocking=synchronous,
                )
            except Exception:
                # Defensive – never propagate EventBus issues to caller
                pass

        for entries, msg in zip(msg_entries, normalised_messages):
            # Ensure correct creation order by performing contact creation *before*
            # the logger call (already satisfied above).  Now we can log safely.
            if synchronous:
                unify.log(
                    project=unify.active_project(),
                    context=self._transcripts_ctx,
                    **entries,
                    params={},
                )
            else:
                self._get_logger().log_create(
                    project=unify.active_project(),
                    context=self._transcripts_ctx,
                    params={},
                    entries=entries,
                )

            try:
                # If we're inside an event-loop schedule the coroutine there …
                loop = asyncio.get_running_loop()
                loop.create_task(_publish_message(msg))
            except RuntimeError:
                # … otherwise create a *temporary* loop so the event isn't lost.
                asyncio.run(_publish_message(msg))

    def join_published(self):
        self._get_logger().join()

    # ------------------------------------------------------------------ #
    #  Shared helper – convert event/message payloads to plain-text
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        *,
        contact_manager: Optional["ContactManager"] = None,
    ) -> str:
        """Return a plain-text transcript (`Full Name: content`) for *messages*.

        Accepts two input shapes:

        1. Raw EventBus events::

               {"kind": "message", "data": {"sender_id": 3, "content": "Hi"}}

        2. Simplified sandbox dicts::

               {"sender": "Daniel Lenton", "content": "Hi"}

        An optional *contact_manager* can be supplied; otherwise a fresh
        `ContactManager` is constructed lazily.  Numeric sender_ids are
        resolved to *full* names (first + surname when available).
        """

        # Local import avoids widening module dependencies at import-time
        from unity.contact_manager.contact_manager import (
            ContactManager as _CM,
        )  # noqa: WPS433

        cm = contact_manager or _CM()

        name_cache: dict[int, str] = {}

        def _name_for_cid(cid: int) -> str:  # noqa: D401 – helper
            if cid in name_cache:
                return name_cache[cid]
            try:
                recs = cm._search_contacts(filter=f"contact_id == {cid}", limit=1)
                if recs:
                    rec = recs[0]
                    full = " ".join(
                        p for p in [rec.first_name, rec.surname] if p
                    ).strip()
                    if not full:
                        full = (rec.first_name or "").strip()
                    if full:
                        name_cache[cid] = full
                        return full
            except Exception:
                pass
            name_cache[cid] = str(cid)
            return name_cache[cid]

        lines: list[str] = []
        for itm in messages:
            # Shape 1 – live EventBus message
            if "kind" in itm:
                if itm.get("kind") != "message":
                    continue
                data = itm.get("data", {})
                sender_val = data.get("sender_id")
                content_val = data.get("content", "")
                if sender_val is None:
                    continue
                sender_name = _name_for_cid(int(sender_val))
            else:  # Shape 2 – sandbox simplified dict
                sender_name = str(itm.get("sender"))
                content_val = str(itm.get("content", ""))

            lines.append(f"{sender_name}: {content_val}")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # Broader context helper
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace the `{broader_context}` placeholder inside *system* messages
        with a fresh snapshot pulled from `MemoryManager` just before the LLM call."""

        import copy

        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local import to avoid cycles

        patched = copy.deepcopy(msgs)

        try:
            broader_ctx = MemoryManager().get_rolling_activity()
        except Exception:
            broader_ctx = ""

        for m in patched:
            if m.get("role") == "system" and "{broader_context}" in (
                m.get("content") or ""
            ):
                m["content"] = m["content"].replace("{broader_context}", broader_ctx)

        return patched

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
        ensure_vector_column(self._transcripts_ctx, self._MSG_EMB, "content")
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            sorting={
                f"cosine({self._MSG_EMB}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._transcripts_ctx).keys()
                if k.endswith("_emb")
            ],
        )
        return [Message(**lg.entries) for lg in logs]

    # _nearest_summaries removed – summary functionality deprecated.
    def _nearest_summaries(self, *args, **kwargs):
        raise NotImplementedError("Summary functionality removed.")

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
            sorting={"timestamp": "descending"},
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._transcripts_ctx).keys()
                if k.endswith("_emb")
            ],
        )
        return [Message(**lg.entries) for lg in logs]

    # _search_summaries removed – summary functionality deprecated.
    def _search_summaries(self, *args, **kwargs):
        raise NotImplementedError("Summary functionality removed.")
