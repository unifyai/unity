from __future__ import annotations

import os
import json
import asyncio
import functools
from typing import List, Dict, Optional, Union, Any

import unify
import requests
from ..common.embed_utils import ensure_vector_column, list_private_fields
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
    fetch_top_k_by_terms_with_score,
    fetch_scores_for_ids,
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
            "gpt-5->o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools,
                num_messages=self._num_messages(),
                transcript_columns=self._list_columns(),
                contact_columns=self._contact_manager._list_columns(),
                include_activity=include_activity,
            ),
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

            # Merge any extra / custom fields directly into the creation kwargs
            for k, v in c.model_dump().items():
                if k not in built_in_fields and v is not None:
                    create_kwargs[k] = v

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
        """Return a plain-text transcript (``Full Name: content``) for ``messages``.

        Accepts two input shapes:

        1. Raw EventBus events::

               {"kind": "message", "data": {"sender_id": 3, "content": "Hi"}}

        2. Simplified sandbox dicts::

               {"sender": "Daniel Lenton", "content": "Hi"}

        An optional ``contact_manager`` can be supplied; otherwise a fresh
        ``ContactManager`` is constructed lazily. Numeric ``sender_id`` values are
        resolved to full names (first + surname when available).

        Parameters
        ----------
        messages : list[dict]
            The list of message-like dictionaries to convert.
        contact_manager : ContactManager | None, optional
            Manager used to resolve numeric ``sender_id`` values to names. If not
            provided, a new ``ContactManager`` instance is constructed lazily.

        Returns
        -------
        str
            The plain-text transcript with one line per message in the format
            ``Full Name: content``.
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
        """Replace the ``{broader_context}`` placeholder inside system messages with a fresh snapshot.

        The snapshot is pulled from ``MemoryManager`` just before the LLM call.

        Parameters
        ----------
        msgs : list[dict]
            The chat messages to preprocess.

        Returns
        -------
        list[dict]
            A deep-copied list of messages where system prompts have the
            ``{broader_context}`` placeholder replaced with the current rolling
            activity snapshot.
        """

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
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Message]:
        """
        Semantic search across transcript messages using one or more reference texts, ranked by the summed cosine similarity across all provided terms.

        Two tables and how they are used
        --------------------------------
        - Transcripts table (Message schema): fields like `content`, `medium`, `timestamp`, `sender_id`, `receiver_ids`.
        - Contacts table (Contact schema): fields describing the sender, e.g., `bio`, `first_name`, `surname`, plus any custom contact columns.

        Provide a mapping of source expressions to reference texts. Each source expression can target either side:
        - Message-side fields (columns in the `Message` schema), e.g. "content" or a derived expression like "str({content}).lower()".
        - Contact-side fields scoped to either the sender or the receivers, by prefixing with either "sender_" or "receiver_" (e.g., "sender_bio", "receiver_bio").
          Backwards-compat: unprefixed contact fields (e.g., "bio") are treated as sender-side.

        The function automatically ensures embedding columns exist for every source expression and then ranks messages by the sum of cosine similarities between each term's embedding and its reference text embedding. If at least one sender-side contact term is present, a temporary join between Transcripts (messages) and Contacts (senders) is performed on `sender_id == contact_id` to compute the combined ranking. When receiver-side terms are present, each message is scored using the minimum (best) cosine distance among all of its receivers against the receiver-side terms, and this value is added to the sender/message score for final ranking.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of `source_expr → reference_text` that defines the semantic query.
            - source_expr: Either a plain identifier naming a field on `Message` (message-side) or `Contact` (contact-side), or a full Unify expression that can reference fields using `{field_name}` placeholders.
              Examples:
                - Message-side (plain): "content"
                - Message-side (derived): "str({content}).lower()"
                - Contact-side (plain): "bio", "first_name", "surname"
                - Contact-side (derived): "str({first_name}) + ' ' + str({bio})"
            - reference_text: The free-form text to embed and compare against each row’s source embedding for this term.
            Notes:
            - When an expression is not a plain identifier, any `{...}` placeholders must reference valid fields on the selected side (message vs contact). Mixed-side expressions are not allowed; if placeholders include any message fields, the term is treated as message-side; if placeholders include only contact fields, the term is contact-side.
            - If you supply only contact-side terms, a join with the contacts table is performed and the top-k messages are returned based on their senders' similarity to the provided references.
            - The embeddings model and derived columns are managed automatically.
        k : int, default 10
            Maximum number of closest results to return. Must be a positive integer (k ≥ 1). Larger values may increase latency.

        Returns
        -------
        List[Message]
            Up to `k` messages sorted by best match first (highest summed cosine similarity / lowest summed distance). Each element is a validated `Message` model from the original transcripts context. Private embedding columns (those ending with `_emb`) are not included in the returned models.

        Behaviour and Details
        ---------------------
        - Term classification:
          • Plain identifiers are classified as message-side if they are valid `Message` fields, otherwise as contact-side if they are valid `Contact` fields.
          • Derived expressions are classified by their placeholders: any placeholder that matches a `Message` field makes the term message-side; if placeholders exist and all match `Contact` fields (and none match message fields), the term is contact-side.
        - Ranking:
          • Single term: messages ranked by cosine similarity to that reference.
          • Multiple terms: messages ranked by the sum of per-term cosine similarities, favouring rows that are jointly similar across all terms.
        - Join semantics:
          • When at least one sender contact-side term exists, the method creates a temporary joined context between Transcripts and Contacts on `sender_id == contact_id`.
          • Receiver contact-side terms do not materialize a join per receiver. Instead, receivers are scored in the Contacts table and each message aggregates its receivers by the minimum distance.
        - Column management:
          • For plain identifiers, the function embeds the referenced column directly.
          • For derived expressions, a stable derived source column is created (if needed) and then embedded.

        Examples
        --------
        - Message content only:
            references = {"content": "let's meet up soon"}
        - Combine message content with sender bio:
            references = {"content": "contract renewal", "bio": "procurement manager"}
        - Derived contact expression (full name) + message content:
            references = {"str({first_name}) + ' ' + str({surname})": "Jane Doe", "content": "invoice"}

        Notes
        -----
        - This tool considers the sender contact only. If you need to factor in receivers, perform a separate search and then filter/merge as needed.
        - Avoid quoting issues in expressions; use single quotes inside expressions where necessary. The API will create any required derived columns automatically.
        - For exact, column-wise filtering (e.g., by `medium` or `sender_id`), prefer `_filter_messages` instead of this semantic search; `_filter_messages` cannot reference Contact fields.
        """
        # Default behaviour: when references is None/empty, skip semantic search and
        # return the most recent messages directly from transcripts context.
        if not references:
            logs = unify.get_logs(
                context=self._transcripts_ctx,
                limit=k,
                exclude_fields=list_private_fields(self._transcripts_ctx),
            )
            return [Message(**lg.entries) for lg in logs]

        # Field name sets to classify expressions as message-side vs contact-side
        msg_fields = set(Message.model_fields.keys())
        contact_fields = set(Contact.model_fields.keys())

        def _extract_placeholders(expr: str) -> list[str]:
            import re as _re

            return _re.findall(r"\{\s*([a-zA-Z_][\w]*)\s*\}", expr)

        # Ensure/embed columns and gather terms
        msg_embed_columns: list[tuple[str, str]] = []
        sender_contact_embed_columns: list[tuple[str, str]] = []
        receiver_contact_embed_columns: list[tuple[str, str]] = []

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
                # Only treat as message-side if it's a Message field and not a prefixed contact key
                is_message_side = (
                    source_expr in msg_fields
                    and not source_expr.startswith("sender_")
                    and not source_expr.startswith("receiver_")
                )
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
            # Determine contact role and base expression/key
            role: Optional[str] = None
            base_expr = source_expr
            if is_plain_identifier(source_expr):
                if source_expr.startswith("sender_"):
                    role = "sender"
                    base_expr = source_expr[len("sender_") :]
                elif source_expr.startswith("receiver_"):
                    role = "receiver"
                    base_expr = source_expr[len("receiver_") :]
                elif (source_expr in contact_fields) and (
                    source_expr not in msg_fields
                ):
                    # Backward-compat: unprefixed contact field → sender
                    role = "sender"
                    base_expr = source_expr
            else:
                # Derived expressions for contacts are only supported if placeholders
                # are exclusively contact fields; treat as sender-side unless explicitly
                # prefixed (we do not support derived receiver_* expressions for now).
                if (
                    (len(placeholders) > 0)
                    and all(ph in contact_fields for ph in placeholders)
                    and not any(ph in msg_fields for ph in placeholders)
                ):
                    if base_expr.startswith("sender_"):
                        role = "sender"
                        base_expr = base_expr[len("sender_") :]
                    elif base_expr.startswith("receiver_"):
                        role = "receiver"  # best-effort; see note above
                        base_expr = base_expr[len("receiver_") :]
                    else:
                        role = "sender"

            if role is not None:
                embed_column_name = ensure_vector_for_source(
                    self._contact_manager._ctx,
                    base_expr,
                )
                if role == "sender":
                    sender_contact_embed_columns.append((embed_column_name, ref_text))
                else:
                    receiver_contact_embed_columns.append((embed_column_name, ref_text))

        # 3) If there are no contact-side terms (sender/receiver), compute directly in transcripts context (no join)
        if not sender_contact_embed_columns and not receiver_contact_embed_columns:
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

        # 4) Build sender-join context and compute base scores (message + sender)
        left_ctx = self._transcripts_ctx
        right_ctx = self._contact_manager._ctx  # Contacts table

        sender_join_ctx = f"{left_ctx}__sender_join__{query_hash}"

        # Temporary backend workaround: join and recompute selected vectors
        self._TEMP_join_on_raw_and_recompute_vectors(
            left_ctx=left_ctx,
            right_ctx=right_ctx,
            join_ctx=sender_join_ctx,
            msg_embed_columns=msg_embed_columns,
            contact_embed_columns=sender_contact_embed_columns,
        )

        # Base terms for sender join ranking
        base_terms = list(msg_embed_columns) + list(sender_contact_embed_columns)

        # If there are no base terms (only receiver terms supplied), select candidates by top receiver contacts
        candidate_rows: list[dict]
        candidate_score_key = ""
        if base_terms:
            # Oversample to allow receiver score to influence final ranking
            oversample = max(k * 5, 50)
            candidate_rows, candidate_score_key = fetch_top_k_by_terms_with_score(
                sender_join_ctx,
                base_terms,
                k=oversample,
            )
        else:
            # Receiver-only search: rank contacts by receiver terms, then gather messages that include
            # those contacts as receivers; compute per-message min receiver score; finally return top-k.
            top_contacts_limit = max(k * 10, 200)
            top_contact_rows, recv_score_key = fetch_top_k_by_terms_with_score(
                right_ctx,
                receiver_contact_embed_columns,
                k=top_contacts_limit,
            )
            # Accumulate candidate messages keyed by message_id with their provisional score (min over receivers)
            msg_to_score: dict[int, float] = {}
            seen_msg_ids: set[int] = set()
            for contact_row in top_contact_rows:
                cid = contact_row.get("contact_id")
                if cid is None:
                    continue
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                try:
                    c_score = float(contact_row.get(recv_score_key, 0))
                except Exception:
                    c_score = 0.0
                # Fetch messages where this contact is in receiver_ids
                msgs = unify.get_logs(
                    context=left_ctx,
                    filter=f"{cid_int} in receiver_ids",
                    limit=max(k * 5, 100),
                    exclude_fields=list_private_fields(left_ctx),
                )
                for m in msgs:
                    mid = m.entries.get("message_id")
                    if mid is None:
                        continue
                    try:
                        mid_int = int(mid)
                    except Exception:
                        continue
                    prev = msg_to_score.get(mid_int)
                    if (prev is None) or (c_score < prev):
                        msg_to_score[mid_int] = c_score
                        seen_msg_ids.add(mid_int)
                if len(seen_msg_ids) >= k * 5:
                    break
            # Turn into candidate rows with only message_id and receiver_ids; we'll refine later
            candidate_rows = []
            if msg_to_score:
                # Fetch these messages to populate receiver_ids for the next step
                for mid in list(msg_to_score.keys()):
                    rows = unify.get_logs(
                        context=left_ctx,
                        filter=f"message_id == {int(mid)}",
                        limit=1,
                        exclude_fields=list_private_fields(left_ctx),
                    )
                    if rows:
                        row = dict(rows[0].entries)
                        # Inject a synthetic base score column name for uniformity in later code
                        row["_receiver_only_base"] = 0.0
                        candidate_rows.append(row)

        # Fast path: no receiver terms → return top-k by base ranking
        if not receiver_contact_embed_columns:
            # Map back to original messages by message_id
            results: List[Message] = []
            taken = 0
            for row in candidate_rows:
                if taken >= k:
                    break
                mid = row.get("message_id")
                if mid is None:
                    continue
                rows = unify.get_logs(
                    context=left_ctx,
                    filter=f"message_id == {int(mid)}",
                    limit=1,
                    exclude_fields=list_private_fields(left_ctx),
                )
                if rows:
                    results.append(Message(**rows[0].entries))
                    taken += 1
            return results

        # 5) Receiver terms present → compute per-contact receiver scores and combine per message (min over receivers)
        # Collect unique receiver ids across candidates
        receiver_id_set: set[int] = set()
        for row in candidate_rows:
            rids = row.get("receiver_ids", [])
            if isinstance(rids, list):
                for rid in rids:
                    try:
                        receiver_id_set.add(int(rid))
                    except Exception:
                        continue

        # Fetch scores for those receiver contacts
        receiver_scores_map, receiver_score_key = fetch_scores_for_ids(
            right_ctx,
            receiver_contact_embed_columns,
            id_field="contact_id",
            ids=sorted(receiver_id_set),
        )

        # Combine scores per message_id
        combined: list[tuple[int, float]] = []
        for row in candidate_rows:
            mid = row.get("message_id")
            if mid is None:
                continue
            # Base score (if available); when only receiver terms were provided, treat base score as 0
            base_score = 0.0
            if candidate_score_key and (candidate_score_key in row):
                try:
                    base_score = float(row.get(candidate_score_key, 0))
                except Exception:
                    base_score = 0.0

            # Min receiver score across this message's receivers
            min_recv = 2.0
            rids = row.get("receiver_ids", [])
            if isinstance(rids, list) and rids:
                for rid in rids:
                    try:
                        rv = receiver_scores_map.get(int(rid))
                        if rv is not None:
                            if rv < min_recv:
                                min_recv = rv
                    except Exception:
                        continue
            # If there are no receivers or no scores, keep min_recv at worst-case 2.0
            combined.append((int(mid), base_score + min_recv))

        # Sort by combined score ascending and take top-k message_ids
        combined.sort(key=lambda t: t[1])
        top_ids = [mid for mid, _ in combined[:k]]

        # Fetch and return original message rows
        results: List[Message] = []
        for mid in top_ids:
            rows = unify.get_logs(
                context=left_ctx,
                filter=f"message_id == {int(mid)}",
                limit=1,
                exclude_fields=list_private_fields(left_ctx),
            )
            if rows:
                results.append(Message(**rows[0].entries))

        return results

    # ────────────────────────────────────────────────────────────────────
    # TEMPORARY HACK – remove after backend join bug is fixed
    # ────────────────────────────────────────────────────────────────────
    def _TEMP_join_on_raw_and_recompute_vectors(
        self,
        *,
        left_ctx: str,
        right_ctx: str,
        join_ctx: str,
        msg_embed_columns: list[tuple[str, str]],
        contact_embed_columns: list[tuple[str, str]],
    ) -> None:
        """Workaround for backend not copying derived *_emb columns on join.

        Strategy:
        - Select embedding columns requested by the caller as usual.
        - Additionally select raw text sources ("content" from transcripts,
          "bio" from contacts) when their corresponding *_emb columns are
          involved in the search.
        - Perform the join into ``join_ctx``.
        - Recreate the *_emb derived columns inside ``join_ctx`` from the raw
          text columns so downstream ranking works.
        """

        # Determine whether to bring raw source columns into the join
        need_msg_content = any(col == "_content_emb" for col, _ in msg_embed_columns)
        need_contact_bio = any(col == "_bio_emb" for col, _ in contact_embed_columns)

        # Build the column selection for the join. Intentionally SKIP copying
        # the buggy *_emb columns that we intend to recompute in the joined
        # context; otherwise the backend will create empty columns and our
        # ensure helper would early‑exit.
        select: Dict[str, str] = {
            f"{left_ctx}.message_id": "message_id",
            f"{left_ctx}.receiver_ids": "receiver_ids",
        }
        for embed_col, _ in msg_embed_columns:
            if not (need_msg_content and embed_col == "_content_emb"):
                select[f"{left_ctx}.{embed_col}"] = embed_col
        for embed_col, _ in contact_embed_columns:
            if not (need_contact_bio and embed_col == "_bio_emb"):
                select[f"{right_ctx}.{embed_col}"] = embed_col

        # Add raw sources if needed (hack-specific)
        if need_msg_content:
            select[f"{left_ctx}.content"] = "content"
        if need_contact_bio:
            select[f"{right_ctx}.bio"] = "bio"

        # Execute the join
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

        # Recompute embeddings inside the joined context
        if need_msg_content:
            ensure_vector_column(join_ctx, "_content_emb", "content")
        if need_contact_bio:
            ensure_vector_column(join_ctx, "_bio_emb", "bio")

    def _filter_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Message]:
        """
        Filter transcript messages using an exact column-wise boolean expression evaluated per row.

        Use this tool for precise filters on structured fields (ids, mediums, equality checks, simple membership). For fuzzy substring or semantic matching across free-text columns, prefer `_search_messages`.

        Parameters
        ----------
        filter : str | None, default None
            A Python-like boolean expression evaluated with `Message` columns in scope for each row. Examples:
            - "medium == 'email' and sender_id == 3"
            - "'urgent' in content and medium != 'sms'"
            - "timestamp >= '2024-01-01T00:00:00' and timestamp < '2024-02-01T00:00:00'" (if your backend supports datetime comparisons)
            When `None`, all messages are returned (subject to `offset`/`limit`).
            Notes:
            - String comparisons are case-sensitive unless you explicitly normalize (e.g., `content.lower().contains('foo')` if supported by your Unify backend).
            - Only `Message` fields are available here. Contact fields are not in scope; to filter by sender attributes, either precompute columns or combine with results from `_search_messages`.
        offset : int, default 0
            Zero-based index of the first row to include. Must be non-negative. Use for pagination together with `limit`.
        limit : int, default 100
            Maximum number of rows to return. Must be a positive integer. Larger values may increase latency.

        Returns
        -------
        List[Message]
            Matching messages as validated `Message` models. Results are sorted by `timestamp` in descending order. Any private embedding columns (those ending with `_emb`) are excluded from the payload to keep responses compact.

        Guidance
        --------
        - Prefer equality or explicit range filters for reliability. Substring checks on large free-text columns can be brittle; consider `_search_messages` for robust semantic queries.
        - Quote strings with single quotes inside the filter expression to avoid escaping issues.
        - If you need deterministic pagination, keep your filter stable and page using consistent `offset`/`limit` values.
        """
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            sorting={"timestamp": "descending"},
            exclude_fields=list_private_fields(self._transcripts_ctx),
        )
        return [Message(**lg.entries) for lg in logs]

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

    # ────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling ContactManager)
    # ────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the transcripts table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their types.
        """
        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={self._transcripts_ctx}"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, str] | list[str]:
        """
        Return the list of available columns in the transcripts table, optionally with types.

        Parameters
        ----------
        include_types : bool, default True
            Controls the shape of the returned value:
            - When True: returns a mapping {column_name: column_type}.
            - When False: returns a list of column names.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _num_messages(self) -> int:
        """Return the total number of messages in transcripts."""
        ret = unify.get_logs_metric(
            metric="count",
            key="message_id",
            context=self._transcripts_ctx,
        )
        if ret is None:
            return 0
        return int(ret)
