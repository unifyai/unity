from __future__ import annotations

import os
import json
import asyncio
import functools
from typing import List, Dict, Optional, Union, Any

import unify
import requests
from ..common.embed_utils import ensure_vector_column
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
from ..helpers import _handle_exceptions
from ..common.semantic_search import (
    is_plain_identifier,
    ensure_vector_for_source,
    fetch_top_k_by_terms,
)


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
            self._contact_manager._filter_contacts,
            self._filter_messages,
            self._search_messages,
            include_class_name=False,
        )

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
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
                recs = cm._filter_contacts(filter=f"contact_id == {cid}", limit=1)
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
            broader_ctx = MemoryManager.get_rolling_activity()
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
    def _search_messages(
        self,
        *,
        references: Dict[str, str],
        k: int = 10,
    ) -> List[Message]:
        """
        Search transcript messages by minimising the sum of cosine distances to multiple reference texts.

        The references map allows mixing message-side expressions (e.g. "content") with
        sender contact-side expressions (e.g. "bio"). Each key is either a plain column name
        or a full expression using Unify's expression language, with field references in braces
        (e.g. "str({first_name}) + ' ' + str({bio})").

        Example:
            references={"content": "let's meet up soon", "bio": "accountant"}

        Parameters
        ----------
        references : Dict[str, str]
            Mapping from a source expression → reference text. Expressions may refer to
            message fields (e.g. content) or contact fields of the sender (e.g. bio).
        k : int, default 10
            Number of closest messages to return.

        Returns
        -------
        List[Message]
            Messages sorted by ascending summed cosine distance (best match first).
        """
        assert (
            isinstance(references, dict) and len(references) > 0
        ), "references must be a non-empty dict"

        # Field name sets to classify expressions as message-side vs contact-side
        msg_fields = set(Message.model_fields.keys())
        contact_fields = set(Contact.model_fields.keys())

        def _extract_placeholders(expr: str) -> list[str]:
            import re as _re

            return _re.findall(r"\{\s*([a-zA-Z_][\w]*)\s*\}", expr)

        # Ensure/embed columns and gather terms
        msg_embed_columns: list[tuple[str, str]] = []
        contact_embed_columns: list[tuple[str, str]] = []

        # For deterministic naming of derived columns and the join context
        import hashlib

        canonical = "|".join(f"{k}=>{references[k]}" for k in sorted(references.keys()))
        query_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

        # 1) Prepare message-side vector columns in transcripts context
        for source_expr, ref_text in references.items():
            placeholders = (
                _extract_placeholders(source_expr)
                if not is_plain_identifier(source_expr)
                else []
            )
            is_message_side = False
            if is_plain_identifier(source_expr):
                is_message_side = source_expr in msg_fields
            else:
                # If any placeholder matches a message field, treat as message-side
                is_message_side = any(ph in msg_fields for ph in placeholders)

            if is_message_side:
                embed_column_name = ensure_vector_for_source(
                    self._transcripts_ctx,
                    source_expr,
                )
                msg_embed_columns.append((embed_column_name, ref_text))

        # 2) Prepare contact-side vector columns in contacts context via ContactManager helper
        for source_expr, ref_text in references.items():
            placeholders = (
                _extract_placeholders(source_expr)
                if not is_plain_identifier(source_expr)
                else []
            )
            is_contact_side = False
            if is_plain_identifier(source_expr):
                is_contact_side = (source_expr in contact_fields) and (
                    source_expr not in msg_fields
                )
            else:
                # If all placeholders are contact fields (or there are placeholders and none are message fields), treat as contact-side
                is_contact_side = (
                    (len(placeholders) > 0)
                    and all(ph in contact_fields for ph in placeholders)
                    and not any(ph in msg_fields for ph in placeholders)
                )

            if is_contact_side:
                embed_column_name = ensure_vector_for_source(
                    self._contact_manager._ctx,
                    source_expr,
                )
                contact_embed_columns.append((embed_column_name, ref_text))

        # 3) If there are no contact-side terms, we can compute directly in transcripts context (no join)
        if not contact_embed_columns:
            # Ensure at least one message-side term exists; otherwise default to content
            if not msg_embed_columns:
                ensure_vector_column(self._transcripts_ctx, self._MSG_EMB, "content")
                msg_embed_columns = [(self._MSG_EMB, next(iter(references.values())))]

            rows = fetch_top_k_by_terms(
                self._transcripts_ctx,
                msg_embed_columns,
                k=k,
            )
            return [Message(**lg) for lg in rows]

        # 4) Otherwise, create a temporary joined context between transcripts and contacts
        left_ctx = self._transcripts_ctx
        right_ctx = self._contact_manager._ctx  # reuse the active Contacts table

        # Select columns to carry into the joined context
        select: Dict[str, str] = {
            f"{left_ctx}.message_id": "message_id",
        }
        for embed_col, _ in msg_embed_columns:
            select[f"{left_ctx}.{embed_col}"] = embed_col
        for embed_col, _ in contact_embed_columns:
            select[f"{right_ctx}.{embed_col}"] = embed_col

        # Create a deterministic destination context name
        join_ctx = f"{left_ctx}__sender_join__{query_hash}"

        # Fire the join request
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/join"
        headers = {
            "Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "project": unify.active_project(),
            "pair_of_args": (
                {"context": left_ctx},
                {"context": right_ctx},
            ),
            "join_expr": f"{left_ctx}.sender_id == {right_ctx}.contact_id",
            "mode": "inner",
            "new_context": join_ctx,
            "columns": select,
        }
        resp = requests.request("POST", url, json=payload, headers=headers)
        _handle_exceptions(resp)

        # Rank by summed cosine across all included embed columns
        joined_rows = fetch_top_k_by_terms(
            join_ctx,
            msg_embed_columns + contact_embed_columns,
            k=k,
        )

        # Query top-k original Messages by message_id
        results: List[Message] = []
        for row in joined_rows:
            mid = row.get("message_id")
            if mid is None:
                continue
            rows = unify.get_logs(
                context=left_ctx,
                filter=f"message_id == {int(mid)}",
                limit=1,
            )
            if rows:
                results.append(Message(**rows[0].entries))

        return results

    def _filter_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Message]:
        """
        Fetch **raw transcript messages** matching an arbitrary Python
        boolean *filter*.

        Do *not* use this tool when searching for messages based on semantic content.
        Trying to get an exact match on substrings (especially with multiple words)
        is very brittle, and likely to return no matches. The `search_messages` tool is
        *much* more robust and accurate when searching for semantic content in messages.

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

    def _update_contact_id(
        self,
        *,
        original_contact_id: int,
        new_contact_id: int,
    ) -> Dict[str, Any]:
        """Replace **all** occurrences of *original_contact_id* with *new_contact_id*
        across every transcript message.

        The substitution is applied to both the ``sender_id`` **and** every entry
        inside ``receiver_ids``.  The update is *in-place* – no new rows are
        created.

        Parameters
        ----------
        original_contact_id : int
            The contact identifier to be replaced.
        new_contact_id : int
            The replacement contact identifier.

        Returns
        -------
        dict
            ToolOutcome-style payload summarising how many messages were
            updated.
        """
        if original_contact_id == new_contact_id:
            raise ValueError("original_contact_id and new_contact_id must differ.")

        total_updates = 0

        # ── 1.  Bulk update all *sender_id* occurrences ────────────────────
        sender_log_ids = unify.get_logs(
            context=self._transcripts_ctx,
            filter=f"sender_id == {original_contact_id}",
            return_ids_only=True,
        )
        if sender_log_ids:
            unify.update_logs(
                logs=sender_log_ids,
                context=self._transcripts_ctx,
                entries={"sender_id": new_contact_id},
                overwrite=True,
            )
            total_updates += len(sender_log_ids)

        # ── 2.  Update all *receiver_ids* lists containing the old id ──────
        receiver_logs = unify.get_logs(
            context=self._transcripts_ctx,
            filter=f"{original_contact_id} in receiver_ids",
            return_ids_only=False,
        )
        for lg in receiver_logs:
            rids = lg.entries.get("receiver_ids", [])
            if not isinstance(rids, list):  # defensive – should always be list
                continue

            updated_rids = [
                (new_contact_id if rid == original_contact_id else rid) for rid in rids
            ]
            # Optional: remove duplicates while preserving order
            seen: set[int] = set()
            deduped_rids: list[int] = []
            for rid in updated_rids:
                if rid not in seen:
                    seen.add(rid)
                    deduped_rids.append(rid)

            # Only write when the list actually changed
            if deduped_rids != rids:
                unify.update_logs(
                    logs=lg.id if hasattr(lg, "id") else lg,
                    context=self._transcripts_ctx,
                    entries={"receiver_ids": deduped_rids},
                    overwrite=True,
                )
                total_updates += 1

        return {
            "outcome": "contact ids updated",
            "details": {
                "old_contact_id": original_contact_id,
                "new_contact_id": new_contact_id,
                "updated_messages": total_updates,
            },
        }
