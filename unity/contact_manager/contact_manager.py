from typing import List, Dict, Optional, Callable, Any, Tuple, Type, Union
from pydantic import BaseModel
import asyncio
import functools
import re
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..knowledge_manager.types import ColumnType
from ..common.embed_utils import ensure_vector_column
from ..common.tool_outcome import ToolOutcome
from ..common.tool_spec import read_only, manager_tool, ToolSpec
from ..common.metrics_utils import reduce_logs

import unify
from .types.contact import Contact, ContactMembership, HydratedContact
from .base import BaseContactManager
from ..common.context_registry import ContextRegistry, TableContext
from ..common.data_store import DataStore
from ..common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..common.model_to_fields import model_to_fields
from ..events.manager_event_logging import log_manager_call
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..common.llm_client import new_llm_client
from ..events.event_bus import EVENT_BUS, Event
from ..blacklist_manager.blacklist_manager import BlackListManager
from ..conversation_manager.cm_types import Medium

# Module delegations (split helpers)
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
)
from .system_contacts import (
    _ensure_columns_exist as _sys_ensure_columns_exist,
    _overlay_contact_id_for_relationship as _sys_overlay_contact_id_for_relationship,
    provision_assistant_contact as _sys_provision_assistant_contact,
    provision_user_contact as _sys_provision_user_contact,
    provision_org_member_contacts as _sys_provision_org_member_contacts,
    provision_system_overlays as _sys_provision_system_overlays,
)
from .settings import (
    CONTACT_MEMBERSHIP_TABLE,
    RELATIONSHIP_BOSS,
    RELATIONSHIP_SELF,
)
from .custom_columns import (
    create_custom_column as _cc_create,
    delete_custom_column as _cc_delete,
)
from .ops import (
    create_contact as _op_create,
    update_contact as _op_update,
    delete_contact as _op_delete,
    merge_contacts as _op_merge,
)
from .search import (
    filter_contacts as _srch_filter,
    search_contacts as _srch_search,
)


class ContactManager(BaseContactManager):
    class Config:
        required_contexts = [
            TableContext(
                name="Contacts",
                description="List of contacts, with all contact details stored.",
                fields=model_to_fields(Contact),
                unique_keys={"contact_id": "int"},
                auto_counting={"contact_id": None},
            ),
            # ContactMembership is the per-body overlay onto the shared
            # Contacts table. It carries the fields that are legitimately
            # body-local (``relationship``, ``should_respond``,
            # ``response_policy``, ``can_edit``) and references the shared
            # row by ``contact_id``. ContextRegistry routes the FK to the
            # correct root automatically — Hive-shared Contacts for Hive
            # members, per-body Contacts for solo bodies — via the
            # target-table-aware FK resolver.
            TableContext(
                name=CONTACT_MEMBERSHIP_TABLE,
                description=(
                    "Per-body overlay describing this body's relationship "
                    "to each shared contact and its response policy."
                ),
                fields=model_to_fields(ContactMembership),
                unique_keys={"contact_id": "int"},
                auto_counting=None,
                foreign_keys=[
                    {
                        "name": "contact_id",
                        "references": "Contacts.contact_id",
                        "on_delete": "SET NULL",
                    },
                ],
            ),
        ]

    # ──────────────────────────────────────────────────────────────────────
    #  Class-level constants / configuration
    # ──────────────────────────────────────────────────────────────────────

    DEFAULT_RESPONSE_POLICY: str = (
        "Please engage politely, helpfully, and respectfully, but you do not need to take orders from them. "
        "Please also do not share **any** sensitive or personal information with them about any other person, "
        "company or policy at all."
    )

    USER_MANAGER_RESPONSE_POLICY: str = (
        "Your immediate manager, please do whatever they ask you to do within reason, and do *not* withhold any "
        "information from them."
    )

    # Response policy for contacts created from unknown inbound messages.
    # Used by CommsManager when creating contacts for unknown senders.
    UNKNOWN_INBOUND_RESPONSE_POLICY: str = (
        "This contact was automatically created from an unknown inbound message. "
        "Do NOT respond to this contact yet. Use your judgement to decide the best course of action: "
        "you may inform your boss about this new contact and ask for guidance, or if this appears to be "
        "spam or unwanted contact, you may choose to blacklist them via the Actor. If your boss confirms "
        "this is a legitimate contact, you should update their details (name, etc.) and set should_respond=True."
    )

    # ──────────────────────────────────────────────────────────────────────
    #  Construction & tool registration
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, *, rolling_summary_in_prompts: bool = True) -> None:
        """
        Responsible for managing the list of contact details stored upstream.

        Parameters
        ----------
        rolling_summary_in_prompts : bool, default ``True``
            Whether to include the rolling activity summary in prompts by default.
        """
        super().__init__()
        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "Contacts")
        # Per-body overlay context. Lives under ``{user}/{assistant}/``
        # for both Hive members and solo bodies — only the FK's *target*
        # (Contacts) routes differently.
        self._membership_ctx = ContextRegistry.get_context(
            self,
            CONTACT_MEMBERSHIP_TABLE,
        )

        # Local DataStore mirror (write-through only; never read from it)
        self._data_store = DataStore.for_context(self._ctx, key_fields=("contact_id",))

        # ── immutable built-in columns ───────────────────────────────────
        # Derive the required/built-in columns directly from the Contact model so
        # that there is a single source-of-truth for field names across the
        # code-base.  Any future change to the Contact schema will
        # automatically propagate here.
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Contact.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        # Track observed/created custom columns in-process so immediate reads
        # right after creation include the new columns without requiring a
        # round-trip schema refresh.
        self._known_custom_fields: set[str] = set()

        # ── public tool dictionaries ─────────────────────────────────────
        # ask-side tools are read-only, so they never change
        ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(fn=self._list_columns, display_label="Listing contact fields"),
                ToolSpec(fn=self.filter_contacts, display_label="Filtering contacts"),
                ToolSpec(fn=self._search_contacts, display_label="Searching contacts"),
                ToolSpec(fn=self._reduce, display_label="Summarising contact data"),
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)

        # update-side tools can read and write
        update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(fn=self.ask, display_label="Querying contact book"),
                ToolSpec(
                    fn=self._create_contact,
                    display_label="Creating a new contact",
                ),
                ToolSpec(fn=self.update_contact, display_label="Updating a contact"),
                ToolSpec(fn=self._delete_contact, display_label="Deleting a contact"),
                ToolSpec(
                    fn=self._create_custom_column,
                    display_label="Adding a custom contact field",
                ),
                ToolSpec(
                    fn=self._delete_custom_column,
                    display_label="Removing a custom contact field",
                ),
                ToolSpec(
                    fn=self._merge_contacts,
                    display_label="Merging duplicate contacts",
                ),
                ToolSpec(
                    fn=self._move_to_blacklist,
                    display_label="Blocking a contact",
                ),
                include_class_name=False,
            ),
        }
        self.add_tools("update", update_tools)

        # rolling activity inclusion flag
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # (No per-instance shorthand registry; dynamic aliases are registered
        #  directly on the Contact class for minimal plumbing.)

        # Ensure context/schema and prefill known custom fields
        self._provision_storage()

        # Provision this body's system contacts (shared rows + "self"
        # / "boss" ContactMembership overlays) and org-member fan-out.
        self._provision_system_overlays()

    # ──────────────────────────────────────────────────────────────────────
    #  Public API (English-only entrypoints for the LLM)
    # ──────────────────────────────────────────────────────────────────────
    @functools.wraps(BaseContactManager.ask, updated=())
    @manager_tool
    @log_manager_call(
        "ContactManager",
        "ask",
        payload_key="question",
        display_label="Checking contact book",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        # Build a *live* tools-dict so the prompt never hard-codes
        # either the number of tools or their names/argspecs.
        tools = dict(self.get_tools("ask"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "ContactManager",
                                "method": "ask",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "ContactManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        _ask_prompt = build_ask_prompt(
            tools=tools,
            num_contacts=self._num_contacts(),
            columns=self._list_columns(),
            include_activity=include_activity,
        ).to_list()
        client.set_system_message(_ask_prompt)

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseContactManager.update, updated=())
    @log_manager_call(
        "ContactManager",
        "update",
        payload_key="request",
        display_label="Updating contact book",
    )
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        tools = dict(self.get_tools("update"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "ContactManager",
                                "method": "update",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "ContactManager",
                                "method": "update",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        _upd_prompt = build_update_prompt(
            tools,
            num_contacts=self._num_contacts(),
            columns=self._list_columns(),
            include_activity=include_activity,
        ).to_list()
        client.set_system_message(_upd_prompt)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseContactManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Clear local cache and custom-field state so subsequent reads/writes
        # operate against a clean slate
        try:
            self._data_store.clear()
        except Exception:
            pass

        # No per-instance custom field state to reset

        # Ensure the schema exists again via shared provisioning helper
        ContextRegistry.refresh(self, "Contacts")

        self._provision_storage()

        # Verify the context is visible before attempting reads
        try:
            import time as _time  # local import to avoid polluting module namespace

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

        # Re-seed this body's system contacts and their self/boss overlays.
        self._provision_system_overlays()

    # (Optional) Public programmatic helpers (non-LLM)
    def materialize_membership_if_missing(
        self,
        contact_id: int,
        *,
        relationship: Optional[str] = None,
        should_respond: Optional[bool] = None,
        response_policy: Optional[str] = None,
    ) -> bool:
        """Create this body's ContactMembership overlay when absent.

        Idempotent: if a :class:`ContactMembership` row for
        *contact_id* already exists in
        ``{user}/{assistant}/ContactMembership``, this method leaves it
        untouched and returns ``False``. A missing row is inserted with
        the supplied fields (unspecified fields default to the overlay
        model's defaults) and returns ``True``.

        Used at interaction boundaries (inbound message fan-in,
        LLM tool calls that reference a contact) to backfill the
        overlay lazily — we don't eagerly mint overlays for contacts
        this body has never interacted with.

        Parameters
        ----------
        contact_id :
            The shared Contacts.contact_id the overlay describes.
        relationship, should_respond, response_policy :
            Seed values for the overlay row when it is newly minted.
            Passed through to :func:`_write_membership_overlay`.
        """
        from .ops import _write_membership_overlay

        try:
            existing = unify.get_logs(
                context=self._membership_ctx,
                filter=f"contact_id == {int(contact_id)}",
                limit=1,
                return_ids_only=True,
            )
        except Exception:
            existing = []
        if existing:
            return False

        _write_membership_overlay(
            self,
            contact_id=int(contact_id),
            relationship=relationship,
            should_respond=should_respond,
            response_policy=response_policy,
        )
        return True

    def _hydrate(self, contact_id: int) -> Optional[HydratedContact]:
        """Compose the shared Contact row with this body's overlay.

        The shared row is read through :meth:`get_contact_info` which is
        DataStore-backed; the per-body :class:`ContactMembership`
        overlay is fetched from ``{user}/{assistant}/ContactMembership``
        with a single bounded ``get_logs`` call. No materialization
        happens here — a missing overlay returns a
        :class:`HydratedContact` whose ``membership`` is ``None`` and
        callers handle the fallback locally (the routing code prefers
        the overlay signal when present and falls back to ``is_system``
        / shared-row signals otherwise).

        Returns
        -------
        Optional[HydratedContact]
            ``None`` when no shared Contact row exists for *contact_id*,
            otherwise the composite view.
        """
        cid = int(contact_id)
        info = self.get_contact_info(cid)
        row = info.get(cid)
        if not row:
            return None
        try:
            shared = Contact.model_validate(row)
        except Exception:
            # Defensive: if the shared row fails model validation for
            # any reason (e.g. transient schema drift), fall back to a
            # best-effort construction so downstream routing can still
            # observe the known fields.
            shared = Contact.model_construct(**row)

        membership: Optional[ContactMembership] = None
        try:
            overlay_rows = unify.get_logs(
                context=self._membership_ctx,
                filter=f"contact_id == {cid}",
                limit=1,
            )
            if overlay_rows:
                entries = overlay_rows[0].entries or {}
                try:
                    membership = ContactMembership.model_validate(entries)
                except Exception:
                    membership = ContactMembership.model_construct(**entries)
        except Exception:
            # Backend read failure must not break the shared-row path.
            # Callers that need membership will observe `None` and fall
            # back to is_system + explicit relationship lookups.
            membership = None

        return HydratedContact(shared=shared, membership=membership)

    def should_respond_to(self, contact_id: int) -> bool:
        """Return whether this body may proactively message *contact_id*.

        Reads the per-body :class:`ContactMembership` overlay via
        :meth:`_hydrate`. See the base class for the full contract.
        """
        hydrated = self._hydrate(contact_id)
        if hydrated is None or hydrated.membership is None:
            return True
        return bool(hydrated.membership.should_respond)

    def get_contact_info(
        self,
        contact_id: Union[int, List[int]],
        fields: Optional[Union[str, List[str]]] = None,
        search_local_storage: bool = True,
        include_membership: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Return a mapping of requested fields for a single contact.

        Behaviour
        ---------
        - When search_local_storage is True, look in the local DataStore first.
          If not found, fall back to a backend read and resync the DataStore.
        - When fields is None or "all", include all allowed fields.
        - Vector/private fields are never loaded.
        - When ``include_membership=True``, compose this body's
          :class:`ContactMembership` overlay into the returned dict
          under the overlay's own keys (``relationship``,
          ``should_respond``, ``response_policy``, ``can_edit``). Keys
          missing from the overlay are omitted; no defaults are
          synthesized so callers can tell "overlay not materialized"
          apart from "overlay says should_respond=True". Setting this
          flag to ``True`` issues one ``get_logs`` against the overlay
          context per returned contact; leave it ``False`` (the default)
          on hot paths that only need shared fields.

        Returns
        -------
        dict[int, dict]
            Mapping of contact_id → selected field→value. Missing ids are omitted.
        """
        allowed = set(self._allowed_fields())

        # Normalise requested fields
        if fields is None or (isinstance(fields, str) and fields.lower() == "all"):
            requested: List[str] = list(allowed)
        elif isinstance(fields, str):
            requested = [fields]
        else:
            requested = list(fields or [])

        # Intersect with allowed set to avoid accidental vector/private columns
        requested = [f for f in requested if f in allowed]
        if not requested:
            requested = list(allowed)

        # Normalise ids list
        if isinstance(contact_id, list):
            ids: List[int] = [int(x) for x in contact_id]
        else:
            ids = [int(contact_id)]

        results: Dict[int, Dict[str, Any]] = {}
        misses: List[int] = []

        # 1) Try local cache
        if search_local_storage:
            for cid in ids:
                try:
                    row = self._data_store[cid]
                    results[cid] = {k: v for k, v in row.items() if k in requested}
                except KeyError:
                    misses.append(cid)
        else:
            misses = list(ids)

        # 2) Backend read for misses (allowed-field superset); write-through to cache
        if misses:
            if len(misses) == 1:
                filt = f"contact_id == {misses[0]}"
            else:
                filt = f"contact_id in [{', '.join(str(x) for x in misses)}]"
            rows = unify.get_logs(
                context=self._ctx,
                filter=filt,
                limit=len(misses),
                from_fields=list(allowed),
            )
            for lg in rows:
                try:
                    backend_row = lg.entries
                    cid_val = int(backend_row.get("contact_id"))
                except Exception:
                    continue
                try:
                    self._data_store.put(backend_row)
                except Exception:
                    pass
                results[cid_val] = {k: backend_row.get(k) for k in requested}

        # Optional: layer the per-body ContactMembership overlay on top.
        # Kept out of the hot path (default False) so existing callers
        # that only read shared fields don't pay the extra backend hop.
        if include_membership and results:
            cids = list(results.keys())
            if len(cids) == 1:
                overlay_filter = f"contact_id == {cids[0]}"
            else:
                overlay_filter = f"contact_id in [{', '.join(str(c) for c in cids)}]"
            try:
                overlay_logs = unify.get_logs(
                    context=self._membership_ctx,
                    filter=overlay_filter,
                    limit=len(cids),
                )
            except Exception:
                overlay_logs = []

            overlay_keys = tuple(ContactMembership.model_fields.keys())
            for lg in overlay_logs:
                try:
                    entries = lg.entries or {}
                    cid_val = int(entries.get("contact_id"))
                except Exception:
                    continue
                if cid_val not in results:
                    continue
                for key in overlay_keys:
                    if key == "contact_id":
                        continue
                    if key in entries:
                        results[cid_val][key] = entries[key]

        return results

    # ──────────────────────────────────────────────────────────────────────
    #  Private tools (LLM-exposed to tool loops)
    #    – these are the underscore-prefixed methods you pass into add_tools
    # ──────────────────────────────────────────────────────────────────────
    # Read-only tools
    @read_only
    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """
        Return the list of available columns in the contacts table, optionally with types.

        Parameters
        ----------
        include_types : bool, default True
            Controls the shape of the returned value:
            - When True: returns a mapping ``{column_name: column_type}`` where
              ``column_type`` is a string label used by Unify (e.g. ``"str"``,
              ``"int"``, ``"bool"``, ``"list"``, ``"dict"``, ``"datetime"``).
            - When False: returns a ``set`` of column names (types omitted). This is
              useful to check for presence/absence without caring about data types.

        Returns
        -------
        Dict[str, Any] | List[str]
            - If ``include_types=True``: ``dict`` mapping column names to their types.
            - If ``include_types=False``: ``list`` of column names.

        Notes
        -----
        - Columns that store embeddings (those whose names end with ``"_emb"``)
          may exist in the backend but are not filtered out here; consumers that
          don't want to see private vector columns should filter them out
          themselves.
        - Column names follow snake_case. Built‑in columns are derived directly from
          the Pydantic ``Contact`` model and are immutable.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    @read_only
    def _reduce(
        self,
        *,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Compute basic reduction metrics over the Contacts table.

        Parameters
        ----------
        metric : str
            Reduction metric to compute. Supported values (case-insensitive) are
            ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
            ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
        keys : str | list[str]
            One or more numeric contact fields to aggregate, for example
            ``\"contact_id\"`` or numeric custom columns. A single column name
            returns a scalar; a list of column names computes the metric
            independently per key and returns a ``{key -> value}`` mapping.
        filter : str | dict[str, str] | None, default None
            Optional row-level filter expression(s) in the same Python syntax as
            :py:meth:`filter_contacts`. When a string, the expression is applied
            uniformly; when a dict, each key maps to its own filter expression.
        group_by : str | list[str] | None, default None
            Optional contact field(s) to group by, for example ``\"should_respond\"``
            or a segmenting custom column. Use a single column name for one
            grouping level, or a list such as ``[\"should_respond\", \"contact_id\"]``
            to group hierarchically in that order. When provided, the result
            becomes a nested mapping keyed by group values, mirroring
            :func:`unify.get_logs_metric` behaviour.

        Returns
        -------
        Any
            Metric value(s) computed over the Contacts context:

            * Single key, no grouping  → scalar (float/int/str/bool).
            * Multiple keys, no grouping → ``dict[key -> scalar]``.
            * With grouping             → nested ``dict`` keyed by group values.
        """
        return reduce_logs(
            context=self._ctx,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    @read_only
    def filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Filter contacts using a boolean Python expression evaluated per row.

        For exact, equality, inequality, membership checks and column-wise filtering (e.g. id or equality checks).

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope. Examples:
            - ``"first_name == 'John' and surname == 'Doe'"``
            - ``"is_system != True"``
            - ``"email_address.endswith('@company.com')"``
            When ``None``, returns all contacts. String comparisons are case‑sensitive unless
            your expression applies a case‑normalisation.
        offset : int, default 0
            Zero‑based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[Contact]
            Matching contacts as Pydantic ``Contact`` models in creation order.

        Notes
        -----
        - Be careful with quoting inside the expression. Use single quotes to delimit string
          literals inside the filter string.
        - This tool is brittle for substring searches across text; prefer ``_search_contacts``
          for that purpose.
        """
        # Prefer a single backend call that whitelists the built‑in columns to
        # keep payloads small without a prior fields introspection request.
        # Fast-path: tighten the requested limit when the filter guarantees
        # at most a single match (unique equality) or a bounded small list.
        eff_limit = limit
        if isinstance(filter, str):
            # contact_id == <int>
            if re.fullmatch(r"\s*contact_id\s*==\s*\d+\s*", filter):
                eff_limit = min(eff_limit, 1)
            else:
                # Equality on unique fields → at most one row
                unique_eq_patterns = (
                    r"\s*email_address\s*==\s*(['\"])\S.*?\1\s*",
                    r"\s*phone_number\s*==\s*(['\"])\S.*?\1\s*",
                    r"\s*whatsapp_number\s*==\s*(['\"])\S.*?\1\s*",
                    r"\s*discord_id\s*==\s*(['\"])\S.*?\1\s*",
                )
                if any(re.fullmatch(p, filter) for p in unique_eq_patterns):
                    eff_limit = min(eff_limit, 1)
                else:
                    # contact_id in [a, b, c] → cap at list length
                    m = re.fullmatch(
                        r"\s*contact_id\s*in\s*\[\s*([0-9,\s]+)\s*\]\s*",
                        filter,
                    )
                    if m:
                        count_ids = len(re.findall(r"\d+", m.group(1)))
                        if count_ids > 0:
                            eff_limit = min(eff_limit, count_ids)

        return _srch_filter(self, filter=filter, offset=offset, limit=limit)

    @read_only
    def _search_contacts(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Semantic search over contacts using one or more reference texts.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of ``source_expr → reference_text`` terms that define the search space.
            - ``source_expr`` can be either a simple column name (e.g. ``"bio"``,
              ``"first_name"``) or a full Unify derived‑expression (e.g.
              ``"str({first_name}) + ' ' + str({surname})"``). For expressions, a stable
              derived source column is created automatically if needed.
            - ``reference_text`` is free‑form text which will be embedded using the
              configured embedding model.
            When ``None`` or an empty dict, semantic search is skipped and the most recent
            contacts are returned using backfill-only logic.
        k : int, default 10
            Maximum number of contacts to return. Must be a positive integer. Must be <= 1000.

        Returns
        -------
        List[Contact]
            Up to ``k`` Pydantic ``Contact`` models. When semantic references are provided,
            results are sorted by similarity (ascending cosine distance). When references
            are omitted/empty, returns the most recent contacts. System contacts
            (those whose ContactMembership overlay carries
            ``relationship == "self"`` or ``"boss"``, or whose shared
            row is marked ``is_system=True``) are excluded.

        Notes
        -----
        - When a single term is provided, results are ranked by ``cosine(column_emb, ref)``.
        - When multiple terms are provided, results are ranked by the sum of cosines across
          all terms to favour contacts similar across several fields.
        """
        return _srch_search(self, references=references, k=k)

    # Mutation tools
    def _create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        bio: Optional[str] = None,
        job_title: Optional[str] = None,
        timezone: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        should_respond: bool = True,
        response_policy: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolOutcome:
        """
        Create and persist a new contact.

        Parameters
        ----------
        first_name : str | None
            Given name. Validation guidance: should start with a capital letter; allowed
            characters are letters, spaces, periods, and hyphens. Optional.
        surname : str | None
            Family name (stored in the ``surname`` column). Same validation guidance as
            ``first_name``. Optional.
        email_address : str | None
            Email address. Must contain exactly one ``@`` with characters on both sides
            (basic validation). Must be unique across all contacts.
        phone_number : str | None
            Phone number. May start with ``+`` (only if explicitly provided by the user),
            otherwise digits only. Must be unique.
        bio : str | None
            Free‑form notes or description about the contact. Optional.
        job_title : str | None
            Free‑text job title / specialization (e.g. "Growth marketing",
            "QA engineer"). On the contact whose ContactMembership
            overlay has ``relationship == "self"`` this mirrors the
            assistant's job title from the backend and is surfaced to
            the LLM via the broader-context prompt. Optional.
        timezone : str | None
            IANA Timezone identifier (e.g. "America/New_York"). Optional.
        rolling_summary : str | None
            Internal running summary of recent activity for this contact. Optional.
        should_respond : bool, default True
            Whether the assistant should reply to this contact by default when
            communicating in user‑facing experiences.
        response_policy : str | None
            Optional policy text that qualifies how the assistant should respond to this
            contact. When omitted, a safe default policy is automatically applied.
        Additional keyword arguments
        ----------------------------
        Any additional top‑level keyword arguments are treated as values for existing
        custom columns.
        - Keys must be existing column names (snake_case) that are not part of the
          built‑in ``Contact`` schema. Create new columns first via
          ``_create_custom_column``.
        - Values are stored as‑is. Choose appropriate types when creating the column
          (e.g. ``str``, ``int``, ``bool``, ``list``, ``dict``).
        - Do not include a key literally named ``"kwargs"``. Pass custom fields
          as top‑level keys instead.

        Returns
        -------
        ToolOutcome
            A standard outcome dict: ``{"outcome": "contact created successfully", "details": {"contact_id": <int>}}``.

        Raises
        ------
        AssertionError
            - If all provided fields are ``None`` (at least one field is required).
            - If any uniqueness constraint is violated (duplicate ``email_address``
              or ``phone_number``).

        Behaviour and Edge Cases
        ------------------------
        - Unify auto-assigns ``contact_id`` via the auto-counting column on
          the shared ``Contacts`` table; no id is reserved for any particular
          role. System contacts are identified by their ``ContactMembership``
          overlay relationship (``"self"`` / ``"boss"``) rather than their id.
        - ``response_policy`` defaults to a conservative policy that avoids sharing sensitive
          information when not explicitly provided.
        - Unspecified fields remain ``None`` and can be populated later via ``update_contact``.
        - For custom columns, ensure the column exists beforehand via ``_create_custom_column``;
          otherwise the request will fail server‑side.
        """
        return _op_create(
            self,
            first_name=first_name,
            surname=surname,
            email_address=email_address,
            phone_number=phone_number,
            bio=bio,
            job_title=job_title,
            timezone=timezone,
            rolling_summary=rolling_summary,
            should_respond=should_respond,
            response_policy=response_policy,
            **kwargs,
        )

    def update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        job_title: Optional[str] = None,
        timezone: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        should_respond: Optional[bool] = None,
        response_policy: Optional[str] = None,
        _log_id: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolOutcome:
        """
        Update one or more fields of an existing contact.

        Parameters
        ----------
        contact_id : int
            The numeric identifier of the contact to modify. Must refer to exactly one
            existing contact.
        first_name : str | None
            New given name. Same validation guidance as in ``_create_contact``. Omit (leave
            as ``None``) to keep unchanged.
        surname : str | None
            New family name (stored as ``surname``). Same guidance as ``first_name``. Omit
            to keep unchanged.
        email_address : str | None
            New email address. Must be unique across all contacts and contain one ``@``.
        phone_number : str | None
            New phone number. Digits only unless explicitly provided with leading ``+``.
            Must be unique.
        whatsapp_number : str | None
            New WhatsApp number. Digits only unless explicitly provided with leading ``+``.
        bio : str | None
            Free‑form notes/description.
        job_title : str | None
            Free‑text job title / specialization. See ``_create_contact``.
        timezone : str | None
            IANA Timezone identifier.
        rolling_summary : str | None
            Updated rolling activity summary (internal).
        should_respond : bool | None
            Whether the assistant should reply to this contact by default. Omit to leave
            unchanged.
        response_policy : str | None
            Override the contact‑specific response policy. Omit to leave unchanged.
        Additional keyword arguments
        ----------------------------
        Any additional top‑level keyword arguments are treated as updates for existing
        custom columns. Keys must be existing column names (snake_case) that are not part of
        the built‑in ``Contact`` schema. Any key with a ``None`` value is ignored.
        Do not include a key literally named ``"kwargs"``; pass custom fields directly at
        the top level.

        Returns
        -------
        ToolOutcome
            A standard outcome dict: ``{"outcome": "contact updated", "details": {"contact_id": <int>}}``.

        Raises
        ------
        ValueError
            - If no updatable field is provided (all parameters ``None`` except ``contact_id``).
            - If ``contact_id`` does not exist or resolves to multiple records (data integrity issue).
            - If updating to a value that violates uniqueness constraints (duplicate email/phone).

        Notes
        -----
        - Fields not supplied remain unchanged.
        - This operation overwrites the stored values for the selected fields.
        - ``contact_id`` itself cannot be changed.
        """

        return _op_update(
            self,
            contact_id=contact_id,
            first_name=first_name,
            surname=surname,
            email_address=email_address,
            phone_number=phone_number,
            whatsapp_number=whatsapp_number,
            bio=bio,
            job_title=job_title,
            timezone=timezone,
            rolling_summary=rolling_summary,
            should_respond=should_respond,
            response_policy=response_policy,
            _log_id=_log_id,
            **kwargs,
        )

    def _delete_contact(
        self,
        *,
        contact_id: int,
        _log_id: Optional[int] = None,
    ) -> ToolOutcome:
        """
        Permanently delete a contact.

        Parameters
        ----------
        contact_id : int
            The identifier of the contact to remove. Must refer to a non‑system contact.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "contact deleted", "details": {"contact_id": <int>}}``.

        Raises
        ------
        RuntimeError
            If attempting to delete a contact whose ContactMembership
            overlay carries ``relationship == "self"`` or ``"boss"``,
            or a shared row marked ``is_system=True`` when no overlay
            has been minted yet.
        ValueError
            If the contact does not exist, or if multiple records share the same ``contact_id``
            (indicates data integrity issues).

        Notes
        -----
        - This operation cannot be undone. Consider ``_merge_contacts`` to consolidate records
          without losing history.
        """
        return _op_delete(self, contact_id=contact_id, _log_id=_log_id)

    def _merge_contacts(
        self,
        *,
        contact_id_1: int,
        contact_id_2: int,
        overrides: Optional[Dict[str, int]] = None,
    ) -> ToolOutcome:
        """
        Merge two contacts into a single consolidated record.

        Overview
        --------
        This operation reads both source contacts, computes a per‑column winner, updates
        the kept record with the consolidated values, deletes the other record, and then
        rewrites transcript references so message histories remain consistent.

        Parameters
        ----------
        contact_id_1 : int
            Identifier of the first source contact.
        contact_id_2 : int
            Identifier of the second source contact. Must be different from ``contact_id_1``.
        overrides : Dict[str, int], optional
            A map indicating which source wins for each column. Keys are column names
            (built‑in or custom). Values must be either ``1`` or ``2`` where:
            - ``1`` → take the value from ``contact_id_1``
            - ``2`` → take the value from ``contact_id_2``

            If not provided, the first non‑``None`` value in the order ``contact_id_1`` → ``contact_id_2`` is used for each column.
            The special key ``"contact_id"`` can be provided to explicitly choose which id to keep; the other contact will be deleted.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "contacts merged successfully", "details": {"kept_contact_id": <int>, "deleted_contact_id": <int>}}``.

        Raises
        ------
        ValueError
            - If the two ids are identical.
            - If either contact cannot be found.
            - If any value in ``overrides`` is not ``1`` or ``2``.
        RuntimeError
            If the merge would delete a system contact (the loser's
            ContactMembership overlay carries
            ``relationship == "self"`` or ``"boss"``). Flip
            ``"contact_id"`` in ``overrides`` to keep the system row
            instead.

        Notes
        -----
        - After the merge, transcript messages that referenced the deleted contact will have
          their ``contact_id`` updated to the kept id for consistency.
        - Custom fields are applied via ``update_contact``; built‑in fields are applied
          directly as arguments.
        """

        return _op_merge(
            self,
            contact_id_1=contact_id_1,
            contact_id_2=contact_id_2,
            overrides=overrides,
        )

    def _move_to_blacklist(
        self,
        *,
        contact_id: int,
        reason: str,
    ) -> ToolOutcome:
        """
        Add all non-empty contact details for the specified contact to the blacklist.

        For each available detail:
        - email_address → one blacklist entry with ``medium == email``.
        - phone_number → two entries with ``medium == sms_message`` and ``phone_call``.

        The blacklist reason is standardised as a concise summary of the contact followed by the cause:
        - ``"{first_name}, {surname}, {bio}, moved to blacklist due to {reason}"`` with missing parts omitted and no stray commas.

        Additionally, this tool deletes the contact from the Contacts table once the blacklist entries
        have been created. When no details exist to blacklist, the contact is still deleted as part of
        the move operation.

        Returns
        -------
        ToolOutcome
            ``{"outcome": "contact details moved to blacklist", "details": {"contact_id": <int>, "blacklist_ids": [<int>, ...]}}``.

        Raises
        ------
        ValueError
            If the contact cannot be found.
        """
        # Fetch the contact row (public fields only)
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {int(contact_id)}",
            limit=1,
            from_fields=self._allowed_fields(),
        )
        if not rows:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to move to blacklist.",
            )
        ent = rows[0].entries

        first = (ent.get("first_name") or "").strip()
        last = (ent.get("surname") or "").strip()
        bio = (ent.get("bio") or "").strip()
        parts = [p for p in (first, last, bio) if p]
        head = ", ".join(parts)
        suffix = f"moved to blacklist due to {reason}"
        bl_reason = f"{head}, {suffix}" if head else suffix

        # Build detail → media pairs
        detail_media: list[tuple[str, Medium]] = []
        email = (ent.get("email_address") or "").strip()
        if email:
            detail_media.append((email, Medium.EMAIL))
        phone = (ent.get("phone_number") or "").strip()
        if phone:
            detail_media.append((phone, Medium.SMS_MESSAGE))
            detail_media.append((phone, Medium.PHONE_CALL))

        if not detail_media:
            # Even when no details exist, delete the contact as part of the move
            try:
                _op_delete(self, contact_id=contact_id, _log_id=None)
            except Exception:
                # Best-effort delete; surface original outcome regardless
                pass
            return {
                "outcome": "no contact details to blacklist",
                "details": {"contact_id": int(contact_id), "blacklist_ids": []},
            }

        blm = BlackListManager()
        created_ids: list[int] = []

        # Best-effort de-duplication per (medium, contact_detail)
        for detail, med in detail_media:
            existing = blm.filter_blacklist(
                filter=f"medium == '{med.value}' and contact_detail == '{detail}'",
                limit=1,
            )["entries"]
            if existing:
                # Skip creating duplicates
                try:
                    created_ids.append(int(existing[0].blacklist_id))
                except Exception:
                    pass
                continue

            res = blm.create_blacklist_entry(
                medium=med,
                contact_detail=detail,
                reason=bl_reason,
            )
            try:
                created_ids.append(int(res["details"]["blacklist_id"]))
            except Exception:
                pass

        # Finally, delete the original contact
        try:
            _op_delete(self, contact_id=contact_id, _log_id=None)
        except Exception:
            pass

        return {
            "outcome": "contact details moved to blacklist",
            "details": {"contact_id": int(contact_id), "blacklist_ids": created_ids},
        }

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: ColumnType | str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Create a new custom column on the contacts table.

        Parameters
        ----------
        column_name : str
            The name of the new custom column. Must be a valid snake_case name.
        column_type : ColumnType | str
            The type of the new custom column.
        column_description : str | None
            The description of the new custom column.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the name and type of the new custom column.
        """
        return _cc_create(
            self,
            column_name=column_name,
            column_type=column_type,
            column_description=column_description,
        )

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        """
        Delete a custom column from the contacts table.

        Parameters
        ----------
        column_name : str
            The name of the custom column to delete.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the name.
        """
        return _cc_delete(self, column_name=column_name)

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers (not exposed as tools)
    # ──────────────────────────────────────────────────────────────────────
    # Storage / provisioning
    def warm_embeddings(self) -> None:
        try:
            ensure_vector_column(
                self._ctx,
                embed_column="_bio_emb",
                source_column="bio",
            )
        except Exception:
            pass

    def _provision_storage(self) -> None:
        """Ensure Contacts context, schema, and local view exist (delegated)."""
        _storage_provision(self)

    def _num_contacts(
        self,
    ) -> int:
        """
        Get the total number of contacts stored in the contacts table.

        Returns
        -------
        int
            The total number of contacts.
        """
        ret = unify.get_logs_metric(
            metric="count",
            key="contact_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    def _get_columns(self) -> Dict[str, str]:
        return _storage_get_columns(self)

    # System contact sync
    def _ensure_columns_exist(self, extra_fields: Dict[str, Any]) -> None:
        _sys_ensure_columns_exist(self, extra_fields)

    def _provision_system_overlays(self) -> None:
        """Provision this body's system contacts and their overlays.

        The contact rows representing the assistant ("self") and the
        primary user ("boss") are seeded in the shared ``Contacts``
        table, then the corresponding :class:`ContactMembership`
        overlays are materialized eagerly so every reader on this body
        can rely on the overlay as the source of truth for
        ``relationship`` / ``should_respond`` / ``response_policy``.

        Shared rows are located via the body's ``ContactMembership``
        overlay (``relationship == "self" | "boss"``); bodies with no
        overlay yet fall through to the create paths inside the
        provision helpers.

        Org-member fan-out runs last and returns early outside of
        organization-scoped API key sessions.
        """
        assistant_log = self._shared_row_for_relationship(RELATIONSHIP_SELF)
        user_log = self._shared_row_for_relationship(RELATIONSHIP_BOSS)
        _sys_provision_assistant_contact(self, assistant_log)
        _sys_provision_user_contact(self, user_log)

        _sys_provision_system_overlays(self)

        _sys_provision_org_member_contacts(self)

    def _shared_row_for_relationship(self, relationship: str):
        """Return the shared ``Contacts`` row for this body's ``relationship`` overlay.

        Resolves the body's ``ContactMembership`` row for *relationship*
        (``"self"`` or ``"boss"``), then loads the corresponding row from
        the shared ``Contacts`` table. Returns ``None`` when either lookup
        misses — the body has no overlay yet or the shared row was wiped.
        """
        contact_id = _sys_overlay_contact_id_for_relationship(self, relationship)
        if contact_id is None:
            return None
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"contact_id == {int(contact_id)}",
                limit=1,
            )
        except Exception:
            return None
        return rows[0] if rows else None

    # Validation / sanitization
    def _allowed_fields(self) -> list[str]:
        """Return the list of columns safe to fetch (exclude private/vector)."""
        cols = self._get_columns()
        # Exclude private (leading underscore) and vector columns ("*_emb")
        allowed = [
            name
            for name in cols.keys()
            if not str(name).startswith("_") and not str(name).endswith("_emb")
        ]
        # Ensure all built-ins are present even if schema drifted
        for b in self._BUILTIN_FIELDS:
            if b not in allowed:
                allowed.append(b)
        # Include any custom fields created/observed during this manager's lifetime
        try:
            for k in getattr(self, "_known_custom_fields", set()):
                if k not in allowed:
                    allowed.append(k)
        except Exception:
            pass
        return allowed

    # Misc small utilities (kept last)
    # Deprecated: client construction is centralized in unity.common.llm_client.new_llm_client
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_contacts on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_ASK_TOOL_IS_SEARCH
            and step_index < 1
            and "search_contacts" in current_tools
        ):
            return (
                "required",
                {"search_contacts": current_tools["search_contacts"]},
            )
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require ask on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)
