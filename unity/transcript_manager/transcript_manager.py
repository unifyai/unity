from __future__ import annotations

import asyncio
import functools
from typing import List, Dict, Optional, Union, Any, Callable, Literal

import unify
from ..contact_manager.base import BaseContactManager
from ..contact_manager.contact_manager import ContactManager
from .types.message import Message, UNASSIGNED

# New: allow Contact objects to appear in messages
from ..contact_manager.types.contact import Contact
from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
)
from ..common.llm_client import new_llm_client
from ..common.clarification_tools import add_clarification_tool_with_events
from ..common.llm_policies import require_first
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.manager_event_logging import (
    log_manager_call,
)
from .prompt_builders import build_ask_prompt
from .base import BaseTranscriptManager
from ..common.tool_spec import read_only, manager_tool
from ..constants import is_semantic_cache_enabled
from ..constants import is_readonly_ask_guard_enabled
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
    list_columns as _storage_list_columns,
    num_messages as _storage_num_messages,
    clear as _storage_clear,
    ensure_exchanges_records as _storage_ensure_exchanges,
)
from .search import (
    search_messages as _search_messages_impl,
    filter_messages as _filter_messages_impl,
    format_contacts_and_messages as _format_contacts_and_messages_impl,
)
from .images import (
    ensure_image_manager as _ensure_image_manager,
    get_images_for_message as _get_images_for_message_impl,
    ask_image as _ask_image_impl,
    attach_image_to_context as _attach_image_to_context_impl,
    attach_message_images_to_context as _attach_message_images_to_context_impl,
)


class TranscriptManager(BaseTranscriptManager):
    # ──────────────────────────────────────────────────────────────────────
    #  Class-level constants / configuration
    # ──────────────────────────────────────────────────────────────────────
    _LOGGER = unify.AsyncLoggerManager(name="TranscriptManager", num_consumers=16)

    # Vector embedding column names
    _MSG_EMB = "_content_emb"

    # ──────────────────────────────────────────────────────────────────────
    #  Construction & tool registration
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        contact_manager: Optional[BaseContactManager] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        super().__init__()

        if contact_manager is not None:
            self._contact_manager = contact_manager
        else:
            self._contact_manager = ContactManager()

        # Tools exposed to the LLM. We wrap message-search/filter so that the
        # tool returns a compact string containing a single JSON table of all
        # participant contacts followed by the list of messages, avoiding
        # repeating long bios per message. Direct method calls (e.g., tests)
        # retain their original return types for backward-compat.
        @functools.wraps(self._filter_messages, updated=())
        @read_only
        def _filter_messages(
            *,
            filter: Optional[str] = None,
            offset: int = 0,
            limit: int = 100,
        ) -> Dict[str, Any]:  # type: ignore[override]
            return self._filter_messages(filter=filter, offset=offset, limit=limit)  # type: ignore[misc]

        @functools.wraps(self._search_messages, updated=())
        @read_only
        def _search_messages(
            *,
            references: Optional[Dict[str, str]] = None,
            k: int = 10,
        ) -> Dict[str, Any]:  # type: ignore[override]
            return self._search_messages(references=references, k=k)  # type: ignore[misc]

        ask_tools = {
            **methods_to_tool_dict(
                self._contact_manager.ask,
                include_class_name=True,
            ),
            **methods_to_tool_dict(
                _filter_messages,
                _search_messages,
                include_class_name=False,
            ),
        }

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
        if read_ctx:
            self._exchanges_ctx = f"{read_ctx}/Exchanges"
        else:
            self._exchanges_ctx = "Exchanges"

        # Image support: lazy-safe image manager and image-aware tools
        _ensure_image_manager(self)
        ask_tools.update(
            methods_to_tool_dict(
                self._get_images_for_message,
                self._ask_image,
                self._attach_image_to_context,
                self._attach_message_images_to_context,
                include_class_name=False,
            ),
        )
        self.add_tools("ask", ask_tools)

        # ── Async logging (mirrors EventBus) ────────────────────────────────
        # Using a dedicated logger means log_create() returns immediately,
        # leaving the actual network I/O to an internal worker thread.
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Provision storage (contexts, fields, columns)
        self._provision_storage()

    # ──────────────────────────────────────────────────────────────────────
    #  Public API (English-only entrypoints for the LLM)
    # ──────────────────────────────────────────────────────────────────────
    # English-Text Question
    @functools.wraps(BaseTranscriptManager.ask, updated=())
    @manager_tool
    @log_manager_call("TranscriptManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        tool_policy: Union[
            Literal["default"],
            Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]],
            None,
        ] = "default",
        _call_id: Optional[str] = None,
        images: Optional[
            "ImageRefs" | list["RawImageRef" | "AnnotatedImageRef"]
        ] = None,
    ) -> SteerableToolHandle:
        # ── 0.  Build the *live* tools-dict (may include clarification helper) ──
        tools = dict(self.get_tools("ask"))

        if _clarification_up_q is not None and _clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                _clarification_up_q,
                _clarification_down_q,
                manager="TranscriptManager",
                method="ask",
                call_id=_call_id,
            )

        # ── 1.  Build LLM client & inject dynamic system-prompt ───────────
        client = new_llm_client()
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools,
                num_messages=_storage_num_messages(self),
                transcript_columns=_storage_list_columns(self),
                contact_columns=self._contact_manager._list_columns(),
                include_activity=include_activity,
            ),
        )

        # Decide effective tool policy (default requires search_messages first),
        # with special handling when images are present to encourage image-aware tools.
        if images:
            effective_tool_policy = self._ask_tool_policy_with_images
            use_semantic_cache = None
        else:
            if tool_policy == "default":
                effective_tool_policy = require_first("search_messages")
            else:
                effective_tool_policy = tool_policy
            use_semantic_cache = "both" if is_semantic_cache_enabled() else None
            # When semantic cache read is enabled, use "auto" tool policy to allow the LLM to return without calling any tools
            effective_tool_policy = (
                None
                if use_semantic_cache in ("read", "both")
                else effective_tool_policy
            )

        # ── 2.  Launch the interactive tool-use loop ───────────────────────
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            preprocess_msgs=inject_broader_context,
            tool_policy=effective_tool_policy,
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.{self.ask.__name__}",
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
            images=images,
        )

        # ── 4.  Optional reasoning exposure  ───────────────────────────────
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseTranscriptManager.clear, updated=())
    def clear(self) -> None:

        _storage_clear(self)

    # (Optional) Public programmatic helpers (non-LLM)
    async def summarize(self, *args, **kwargs):
        """Deprecated: summarize functionality removed."""
        raise NotImplementedError(
            "Summarize functionality has been removed from TranscriptManager.",
        )

    def log_messages(
        self,
        messages: Union[
            Union[Dict[str, Any], Message],
            List[Union[Dict[str, Any], Message]],
        ],
        synchronous: bool = False,
    ) -> List[Message]:
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

        Returns
        -------
        list[Message]
            The created messages as validated ``Message`` models, populated
            with assigned identifiers (e.g., ``message_id`` and
            ``exchange_id``) exactly as returned by the storage backend. The
            shape mirrors ``_filter_messages`` (list of ``Message``).
        """

        # ── 0. Early-exit on empty input ────────────────────────────────────
        if not messages:
            return []

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
        per_message_metadata: List[Optional[Dict[str, Any]]] = []
        for raw in messages:
            # Convert to dict early so we can mutate fields easily
            if isinstance(raw, Message):
                payload: Dict[str, Any] = raw.model_dump(mode="python")
                meta_val = None
            else:  # assume mapping
                payload = dict(raw)
                # Extract optional private metadata without letting it leak into the model
                meta_val = payload.pop("_metadata", None)

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
            per_message_metadata.append(meta_val)

        # ── 3. Dump POST-ready JSON for each message ──────────────────────
        msg_entries = [m.to_post_json() for m in normalised_messages]

        # Attach metadata payloads to corresponding entries (column ensured in __init__)
        if any(pm is not None for pm in per_message_metadata):
            for idx, meta_val in enumerate(per_message_metadata):
                if meta_val is not None:
                    try:
                        msg_entries[idx]["_metadata"] = meta_val
                    except Exception:
                        pass

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

        created_messages: List[Message] = []

        for entries, _orig_msg in zip(msg_entries, normalised_messages):
            # Ensure correct creation order by performing contact creation *before*
            # the logger call (already satisfied above).  Now we can log safely.
            log = unify.log(
                context=self._transcripts_ctx,
                **entries,
                new=True,
                mutable=True,
                params={},
            )

            # Build a Message from the POST response; if ids look unassigned,
            # perform a one-off read to retrieve the assigned values.
            # TODO: Remove this GET fallback once the backend echoes auto-assigned
            # exchange_id on POST responses consistently. message_id already echoes reliably.
            try:
                persisted_payload = {
                    k: log.entries.get(k) for k in Message.model_fields.keys()
                }
                # Only refetch when exchange_id is missing/unassigned, since message_id
                # is already returned by the POST in current backends.
                need_refetch = False
                try:
                    xid_val = persisted_payload.get("exchange_id")
                    if xid_val is None or int(xid_val) <= -1:
                        need_refetch = True
                except Exception:
                    need_refetch = True

                if need_refetch:
                    try:
                        ts = entries.get("timestamp")
                        snd = entries.get("sender_id")
                        med = entries.get("medium")
                        flt = f"timestamp == '{ts}' and sender_id == {snd} and medium == '{med}'"
                        rows = unify.get_logs(
                            context=self._transcripts_ctx,
                            filter=flt,
                            limit=1,
                            from_fields=list(Message.model_fields.keys()),
                            sorting={"timestamp": "descending"},
                        )
                        if rows:
                            persisted_payload = dict(rows[0].entries)
                    except Exception:
                        pass

                # Remove any None values for id fields so the validator can apply sentinel if needed
                if persisted_payload.get("message_id") is None:
                    persisted_payload.pop("message_id", None)
                if persisted_payload.get("exchange_id") is None:
                    persisted_payload.pop("exchange_id", None)

                created_msg = Message(**persisted_payload)
                created_messages.append(created_msg)
            except Exception:
                # Fallback to constructing from the original request shape, omitting id keys
                fallback_payload = {
                    k: entries.get(k) for k in Message.model_fields.keys()
                }
                if fallback_payload.get("message_id") is None:
                    fallback_payload.pop("message_id", None)
                if fallback_payload.get("exchange_id") is None:
                    fallback_payload.pop("exchange_id", None)
                created_msg = Message(**fallback_payload)
                created_messages.append(created_msg)

            try:
                # If we're inside an event-loop schedule the coroutine there …
                loop = asyncio.get_running_loop()
                loop.create_task(_publish_message(created_msg))
            except RuntimeError:
                # … otherwise create a *temporary* loop so the event isn't lost.
                asyncio.run(_publish_message(created_msg))

        # ── 5. Ensure Exchanges rows exist for any newly seen exchange_ids ──
        try:
            eids: set[int] = set()
            eid_to_medium: Dict[int, str] = {}
            for m in created_messages:
                try:
                    if getattr(m, "exchange_id", UNASSIGNED) is not None:
                        exid = int(getattr(m, "exchange_id", UNASSIGNED))
                        if exid != UNASSIGNED and exid >= 0:
                            eids.add(exid)
                            if exid not in eid_to_medium:
                                try:
                                    eid_to_medium[exid] = str(getattr(m, "medium"))
                                except Exception:
                                    pass
                except Exception:
                    continue
            if eids:
                self._ensure_exchanges_records(eids, eid_to_medium=eid_to_medium)
        except Exception:
            # Non-fatal: do not break message logging if exchanges upsert fails
            pass

        return created_messages

    def join_published(self):
        self._get_logger().join()

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
                recs = cm.filter_contacts(filter=f"contact_id == {cid}", limit=1)
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

    # ──────────────────────────────────────────────────────────────────────
    #  Private tools (LLM-exposed to tool loops)
    #    – these are the underscore-prefixed methods you pass into add_tools
    # ──────────────────────────────────────────────────────────────────────
    def _search_messages(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        return _search_messages_impl(self, references=references, k=k)

    def _filter_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int | None = 100,
    ) -> Dict[str, Any]:
        return _filter_messages_impl(self, filter=filter, offset=offset, limit=limit)

    def update_contact_id(
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

    # ──────────────────────────────────────────────────────────────────────
    #  Image tools
    # ──────────────────────────────────────────────────────────────────────
    @read_only
    def _get_images_for_message(self, *, message_id: int) -> List[Dict[str, Any]]:
        """Return image metadata (no raw data) for images referenced by a message.

        Output schema (list of objects):
        - image_id: int
        - caption: str | None
        - timestamp: str (ISO8601)
        - annotation: str  → freeform explanation of how the image relates to the text
        """
        return _get_images_for_message_impl(self, message_id=message_id)

    @read_only
    async def _ask_image(self, *, image_id: int, question: str) -> str:
        """Ask a one‑off question about a specific stored image.

        This helper mirrors the behaviour of :pyfunc:`ImageHandle.ask` but is
        exposed as a TranscriptManager tool that requires an explicit
        ``image_id``. It sends the underlying image to a vision‑capable model as
        an image block and returns a textual answer only.

        Parameters
        ----------
        image_id : int
            Identifier of the image to analyse. If the image's ``data`` is a
            Google Cloud Storage URL, a short‑lived signed URL is generated to
            grant the model access; otherwise the stored base64 is converted to
            a ``data:image/...;base64,`` URL.
        question : str
            Natural‑language question to ask about the image.

        Returns
        -------
        str
            Text answer from the vision model. This call does not persist the
            visual context for follow‑up turns; prefer
            ``attach_image_to_context``/``attach_message_images_to_context``
            when subsequent steps should keep seeing the image(s).
        """
        return await _ask_image_impl(self, image_id=image_id, question=question)

    def _attach_image_to_context(
        self,
        *,
        image_id: int,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach a single image (by id) as raw base64 for persistent context.

        Loads the image bytes for ``image_id`` and returns a payload suitable
        for inclusion as an image block in the current tool‑use loop. Behaviour
        aligns with :pyfunc:`ImageHandle.raw` for source resolution:
        - If ``data`` is a GCS URL (``gs://`` or
          ``https://storage.googleapis.com/...``), the bytes are downloaded
          (raising if not accessible).
        - Otherwise, ``data`` is expected to be base64 and is decoded to bytes.

        Parameters
        ----------
        image_id : int
            Identifier of the image to attach.
        note : str | None
            Optional human‑readable note describing why the image is attached.

        Returns
        -------
        dict
            A payload of the form:
            {"note": str, "image": base64_string}
            where ``image`` is the raw bytes of the image encoded as base64
            (PNG or JPEG). Downstream should render this as an image block.
        """
        return _attach_image_to_context_impl(self, image_id=image_id, note=note)

    def _attach_message_images_to_context(
        self,
        *,
        message_id: int,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Attach multiple images referenced by a message to the loop context.

        Characteristics
        ---------------
        - Batches attachment of several images linked via the message's image references.
        - Returns metadata (including optional annotations) alongside the base64 for each image.
        - Useful for multi‑image tasks where the loop should retain visual context.

        Parameters
        ----------
        limit : int
            Cap on how many images are attached (order preserved by reference order).

        Returns
        -------
        dict
            { "attached_count": int, "images": [ { "meta": {...}, "image": base64 }, ... ] }
            Each ``meta`` includes ``image_id``, ``caption``, ``timestamp``, and optional ``annotation``.
        """
        return _attach_message_images_to_context_impl(
            self,
            message_id=message_id,
            limit=limit,
        )

    # (Span substring helper removed – images now aligned via freeform annotations)

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers (not exposed as tools)
    # ──────────────────────────────────────────────────────────────────────
    # Column and metrics helpers (paralleling ContactManager)
    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the transcripts table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their types.
        """
        return _storage_get_columns(self)

    def _list_columns(
        self,
        *,
        include_types: bool = True,
        include_private: bool = False,
    ) -> Dict[str, str] | list[str]:
        """
        Return the list of available columns in the transcripts table, optionally with types.

        Parameters
        ----------
        include_types : bool, default True
            Controls the shape of the returned value:
            - When True: returns a mapping {column_name: column_type}.
            - When False: returns a list of column names.
        include_private : bool, default False
            When False, private/internal columns (those whose names start with "_")
            are omitted from the result to reduce payload size and avoid exposing
            vector/derived fields. Set to True to return all columns.
        """
        cols = _storage_list_columns(
            self,
            include_types=include_types,
            include_private=include_private,
        )
        return cols

    def _num_messages(self) -> int:
        """Return the total number of messages in transcripts."""
        return _storage_num_messages(self)

    # Internal provisioning helper
    def _provision_storage(self) -> None:
        _storage_provision(self)

    # Exchanges helper
    def _ensure_exchanges_records(
        self,
        exchange_ids: set[int],
        *,
        eid_to_medium: Optional[Dict[int, str]] = None,
    ) -> None:
        _storage_ensure_exchanges(self, exchange_ids, eid_to_medium=eid_to_medium)

    # Formatting helper: single contacts table + messages
    def _format_contacts_and_messages(self, messages: List[Message]) -> Dict[str, Any]:
        return _format_contacts_and_messages_impl(self, messages)

    # Misc small utilities (kept last)
    @classmethod
    def _get_logger(cls) -> unify.AsyncLoggerManager:
        return cls._LOGGER

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        # Deprecated: use common.llm_policies.require_first("search_messages") instead.
        return require_first("search_messages")(step_index, current_tools)

    @staticmethod
    def _ask_tool_policy_with_images(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """On step 0, require one of search_messages/ask_image/attach_image_raw; auto thereafter.

        Encourages the model to either begin with a semantic query over transcripts
        or explicitly use the image helpers when visual context is supplied.
        """
        if step_index < 1:
            allowed_first_turn: Dict[str, Any] = {}
            for name in ("search_messages", "ask_image", "attach_image_raw"):
                if name in current_tools:
                    allowed_first_turn[name] = current_tools[name]
            if allowed_first_turn:
                return ("required", allowed_first_turn)
        return ("auto", current_tools)
