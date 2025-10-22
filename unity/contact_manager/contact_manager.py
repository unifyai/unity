from typing import List, Dict, Optional, Callable, Any, Tuple, Union
import asyncio
import json
import functools
import re
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..knowledge_manager.types import ColumnType
from ..common.tool_outcome import ToolOutcome
from ..common.tool_spec import read_only, manager_tool

import unify
from .types.contact import Contact
from .base import BaseContactManager
from ..common.data_store import DataStore
from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.manager_event_logging import log_manager_call
from ..constants import is_semantic_cache_enabled
from ..constants import is_readonly_ask_guard_enabled
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..common.llm_client import new_llm_client
from ..common.clarification_tools import add_clarification_tool_with_events

# Module delegations (split helpers)
from .storage import (
    provision_storage as _storage_provision,
    get_columns as _storage_get_columns,
)
from .system_contacts import (
    _ensure_columns_exist as _sys_ensure_columns_exist,
    sync_assistant_contact as _sys_sync_assistant_contact,
    sync_user_contact as _sys_sync_user_contact,
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
    # ──────────────────────────────────────────────────────────────────────
    #  Class-level constants for response policy
    # ──────────────────────────────────────────────────────────────────────

    DEFAULT_RESPONSE_POLICY: str = (
        "Please engage politely, helpfully, and respectfully, but you do not need to take orders from them. "
        "Please also do not share **any** sensitive or personal information with them about any other person, "
        "company or policy at all."
    )

    USER_MANAGER_RESPONSE_POLICY: str = (
        "Your immediate manager, please do whatever they ask you to do within reason, and do *not* withhold any "
        "information from them"
    )

    def __init__(self, *, rolling_summary_in_prompts: bool = True) -> None:
        """
        Responsible for managing the list of contact details stored upstream.

        Parameters
        ----------
        batched : bool, default ``False``
            • ``False`` – expose the original *atomic* tools\
            • ``True``  – expose only the new *batched* variants
        """
        super().__init__()

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
        self._ctx = f"{read_ctx}/Contacts"

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
                self._list_columns,
                self._filter_contacts,
                self._search_contacts,
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)

        # update-side tools are can read and write
        update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.ask,
                self._create_contact,
                self._update_contact,
                self._delete_contact,
                self._create_custom_column,
                self._delete_custom_column,
                self._merge_contacts,
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

        # ── ensure an assistant contact with id 0 exists and is up-to-date ──
        self._sync_assistant_contact()
        # ── ensure a default *user* contact with id 1 exists and is up-to-date ──
        self._sync_user_contact()

    # ──────────────────────────────────────────────────────────────────────
    #  Assistant syncing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_assistant_info(self) -> List[Dict[str, Any]]:
        """Return the list of assistants configured for the current account.

        The API is expected to return a JSON object with an ``info`` key
        containing the assistant records.  If the request fails, an
        exception is raised via ``_handle_exceptions`` so callers do not
        silently proceed with incomplete data.

        Returns
        -------
        List[Dict[str, Any]]
            The list of assistants for the current account.
        """
        return unify.list_assistants()

    def _ensure_columns_exist(self, extra_fields: Dict[str, Any]) -> None:
        _sys_ensure_columns_exist(self, extra_fields)

    def _sync_assistant_contact(self) -> None:
        _sys_sync_assistant_contact(self)

    # ------------------------------------------------------------------
    #  Default *user* contact helpers (contact_id == 1)
    # ------------------------------------------------------------------
    def _fetch_user_info(self) -> Dict[str, Any]:
        """Return basic information for the authenticated human user (contact_id == 1).

        Attempts to fetch the real details from the backend endpoint
        ``/user/basic-info``.  On *any* failure (network, authentication,
        unexpected payload, etc.) the function falls back to a dummy
        placeholder user so that offline test-suites continue to operate
        unchanged.

        Returns
        -------
        Dict[str, Any]
            Basic user information mapping.
        """

        user_info: Dict[str, Any] = {}

        data: Any = unify.get_user_basic_info()
        # Map API payload → expected field names
        mapped: Dict[str, Any] = {
            "first_name": data.get("first"),
            "last_name": data.get("last"),
            "email": data.get("email"),
        }

        # Filter out *None* values so downstream logic does not
        # inadvertently overwrite existing data with nulls.
        user_info.update({k: v for k, v in mapped.items() if v is not None})

        from .. import ASSISTANT

        if ASSISTANT is not None:
            phone = ASSISTANT.get("user_phone")
            whatsapp = ASSISTANT.get("user_whatsapp_number")
            mapped_extra: Dict[str, Any] = {
                "phone_number": phone,
                "whatsapp_number": whatsapp,
            }
            user_info.update(
                {k: v for k, v in mapped_extra.items() if v is not None},
            )

        # If we managed to retrieve *any* real data, return it.
        if user_info:
            return user_info

        # ── fallback: dummy user ──────────────────────────────────────────
        return {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@email.com",
        }

    def _sync_user_contact(self) -> None:
        _sys_sync_user_contact(self)

    # ──────────────────────────────────────────────────────────────────────
    #  Column helpers (single-table version of KnowledgeManager's helpers)
    # ──────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        return _storage_get_columns(self)

    # Apply timing to tool methods
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
          themselves (other tools in this class exclude them where appropriate).
        - Column names follow snake_case. Built‑in columns are derived directly from
          the Pydantic ``Contact`` model and are immutable.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: ColumnType | str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        return _cc_create(
            self,
            column_name=column_name,
            column_type=column_type,
            column_description=column_description,
        )

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        return _cc_delete(self, column_name=column_name)

    # Public #
    # -------#
    @functools.wraps(BaseContactManager.ask, updated=())
    @manager_tool
    @log_manager_call("ContactManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        images: Optional[Dict[str, Any]] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        # Build a *live* tools-dict so the prompt never hard-codes
        # either the number of tools or their names/argspecs.
        tools = dict(self.get_tools("ask"))
        if clarification_up_q is not None and clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                clarification_up_q,
                clarification_down_q,
                manager="ContactManager",
                method="ask",
                call_id=_call_id,
            )

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
        )
        client.set_system_message(_ask_prompt)

        use_semantic_cache = (
            ("both" if is_semantic_cache_enabled() else None) if not images else None
        )
        if use_semantic_cache in ("read", "both"):
            # When semantic cache is enabled, use "auto" tool policy to allow the LLM to return without calling any tools
            tool_policy_fn = None
        elif images:
            tool_policy_fn = self._ask_tool_policy_with_images
        else:
            tool_policy_fn = self._default_ask_tool_policy

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=tool_policy_fn,
            preprocess_msgs=inject_broader_context,
            semantic_cache=use_semantic_cache,
            semantic_cache_namespace=f"{self.__class__.__name__}.{self.ask.__name__}",
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
            images=images,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseContactManager.update, updated=())
    @log_manager_call("ContactManager", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        images: Optional[Dict[str, Any]] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        tools = dict(self.get_tools("update"))
        if clarification_up_q is not None and clarification_down_q is not None:
            add_clarification_tool_with_events(
                tools,
                clarification_up_q,
                clarification_down_q,
                manager="ContactManager",
                method="update",
                call_id=_call_id,
            )

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
        )
        client.set_system_message(_upd_prompt)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            preprocess_msgs=inject_broader_context,
            images=images,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # Helpers #
    # --------#

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

    # Public non-tool method
    def get_contact_info(
        self,
        contact_id: Union[int, List[int]],
        fields: Optional[Union[str, List[str]]] = None,
        search_local_storage: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Return a mapping of requested fields for a single contact.

        Behaviour
        ---------
        - When search_local_storage is True, look in the local DataStore first.
          If not found, fall back to a backend read and resync the DataStore.
        - When fields is None or "all", include all allowed fields.
        - Vector/private fields are never loaded.

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

        return results

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

    @functools.wraps(BaseContactManager.clear, updated=())
    def clear(self) -> None:
        try:
            # Drop the entire contacts table for this active assistant context
            unify.delete_context(self._ctx)
        except Exception:
            # Proceed even if deletion fails (context may already be absent)
            pass

        # Clear local cache and custom-field state so subsequent reads/writes
        # operate against a clean slate
        try:
            self._data_store.clear()
        except Exception:
            pass

        # No per-instance custom field state to reset

        # Ensure the schema exists again via shared provisioning helper
        try:
            # Remove any previous ensure memo and force re-provisioning
            from ..common.context_store import TableStore as _TS  # local import

            try:
                _TS._ENSURED.discard((unify.active_project(), self._ctx))
            except Exception:
                pass
        except Exception:
            pass

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

        # Recreate assistant and default user contacts (id 0 and 1)
        self._sync_assistant_contact()
        self._sync_user_contact()

    # Private #
    # --------#

    def _provision_storage(self) -> None:
        """Ensure Contacts context, schema, and local view exist (delegated)."""
        _storage_provision(self)

    # Helper to derive a unique shorthand for a given custom column name.
    def _derive_shorthand(self, column_name: str) -> str:
        from .types.contact import Contact as _C

        parts = [p for p in str(column_name).split("_") if p]
        base = "".join(p[:2] for p in parts) or str(column_name)[:3]
        base = re.sub(r"[^a-z0-9_]", "", base.lower())
        if not base or not re.match(r"^[a-z]", base):
            base = ("c_" + base) if base else "c"
        used = set(_C.shorthand_map().values())
        used.update(getattr(self, "_custom_shorthand", {}).values())
        cand = base
        idx = 1
        while cand in used:
            cand = f"{base}{idx}"
            idx += 1
        return cand

    def _sanitize_custom_columns(
        self,
        custom_columns: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a filtered copy of custom columns safe for JSON logging.

        - Drops internal control keys injected by the async tool loop
          (pause/interject/clarification/context channels).
        - Drops any values that are not JSON-serialisable.
        """
        internal_keys = {
            "parent_chat_context",
            "interject_queue",
            "pause_event",
            "clarification_up_q",
            "clarification_down_q",
            "kwargs",
            "_log_id",
        }
        safe: Dict[str, Any] = {}
        for key, value in (custom_columns or {}).items():
            if key in internal_keys:
                continue
            try:
                json.dumps(value)
            except Exception:
                # Skip non-serialisable values (e.g. asyncio.Event, queues, etc.)
                continue
            safe[key] = value
        return safe

    def _create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: bool = False,
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
        whatsapp_number : str | None
            WhatsApp number. Same formatting guidance as ``phone_number``. Must be unique.
        bio : str | None
            Free‑form notes or description about the contact. Optional.
        rolling_summary : str | None
            Internal running summary of recent activity for this contact. Optional.
        respond_to : bool, default False
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
            - If any uniqueness constraint is violated (duplicate ``email_address``,
              ``phone_number``, or ``whatsapp_number``).

        Behaviour and Edge Cases
        ------------------------
        - If this is the very first contact in the table, the record is inserted immediately
          and Unify will assign ``contact_id == 0`` (reserved for the assistant account).
          Subsequent creations will receive the next available id.
        - ``response_policy`` defaults to a conservative policy that avoids sharing sensitive
          information when not explicitly provided.
        - Unspecified fields remain ``None`` and can be populated later via ``_update_contact``.
        - For custom columns, ensure the column exists beforehand via ``_create_custom_column``;
          otherwise the request will fail server‑side.
        """
        return _op_create(
            self,
            first_name=first_name,
            surname=surname,
            email_address=email_address,
            phone_number=phone_number,
            whatsapp_number=whatsapp_number,
            bio=bio,
            rolling_summary=rolling_summary,
            respond_to=respond_to,
            response_policy=response_policy,
            **kwargs,
        )

    def _update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: Optional[bool] = None,
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
            New WhatsApp number. Same formatting and uniqueness rules as ``phone_number``.
        bio : str | None
            Free‑form notes/description.
        rolling_summary : str | None
            Updated rolling activity summary (internal).
        respond_to : bool | None
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
            - If updating to a value that violates uniqueness constraints (duplicate email/phone/WhatsApp).

        Notes
        -----
        - Fields not supplied remain unchanged.
        - This operation overwrites the stored values for the selected fields.
        - ``contact_id`` itself cannot be changed here; use ``_merge_contacts`` if you need
          to consolidate records and choose which id to keep.
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
            rolling_summary=rolling_summary,
            respond_to=respond_to,
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
            If attempting to delete reserved system contacts: ``0`` (assistant) or ``1`` (default user).
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
            If the merge would delete a protected system contact (ids ``0`` or ``1``).

        Notes
        -----
        - Private vector columns (names ending with ``"_emb"``) are ignored during merge.
        - After the merge, transcript messages that referenced the deleted contact will have
          their ``contact_id`` updated to the kept id for consistency.
        - Custom fields are applied via ``_update_contact``; built‑in fields are applied
          directly as arguments.
        """

        return _op_merge(
            self,
            contact_id_1=contact_id_1,
            contact_id_2=contact_id_2,
            overrides=overrides,
        )

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
            Maximum number of contacts to return. Must be a positive integer.

        Returns
        -------
        List[Contact]
            Up to ``k`` Pydantic ``Contact`` models. When semantic references are provided,
            results are sorted by similarity (ascending cosine distance). When references
            are omitted/empty, returns the most recent contacts. System contacts (ids ``0``
            and ``1``) are excluded.

        Notes
        -----
        - When a single term is provided, results are ranked by ``cosine(column_emb, ref)``.
        - When multiple terms are provided, results are ranked by the sum of cosines across
          all terms to favour contacts similar across several fields.
        - Embedding columns (``*_emb``) are excluded from the returned models to keep payloads
          compact.
        """
        return _srch_search(self, references=references, k=k)

    @read_only
    def _filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Filter contacts using a boolean Python expression evaluated per row.

        Prefer this for exact, column‑wise filtering (e.g. id or equality checks). For
        fuzzy or semantic matches across free‑text columns, use ``_search_contacts``.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope. Examples:
            - ``"first_name == 'John' and surname == 'Doe'"``
            - ``"contact_id != 0 and contact_id != 1"``
            - ``"email_address.endswith('@company.com')"``
            When ``None``, returns all contacts. String comparisons are case‑sensitive unless
            your expression applies a case‑normalisation.
        offset : int, default 0
            Zero‑based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return.

        Returns
        -------
        List[Contact]
            Matching contacts as Pydantic ``Contact`` models in creation order. Embedding
            columns (``*_emb``) are excluded from the payload to keep responses small.

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

    # ------------------------------------------------------------------ #
    #  Small internal helpers (LLM client + tool policies)               #
    # ------------------------------------------------------------------ #

    # Deprecated: client construction is centralized in unity.common.llm_client.new_llm_client

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require search_contacts on the first step; auto thereafter."""
        if step_index < 1 and "search_contacts" in current_tools:
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
        """Require ask on the first step; auto thereafter."""
        if step_index < 1 and "ask" in current_tools:
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    @staticmethod
    def _ask_tool_policy_with_images(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """On step 0, require one of search_contacts/ask_image/attach_image_raw; auto thereafter.

        This ensures the model begins by either running a semantic query, asking
        a provided image a question, or attaching image context; subsequent steps
        can proceed freely.
        """
        if step_index < 1:
            allowed_first_turn: Dict[str, Any] = {}
            for name in ("search_contacts", "ask_image", "attach_image_raw"):
                if name in current_tools:
                    allowed_first_turn[name] = current_tools[name]
            if allowed_first_turn:
                return ("required", allowed_first_turn)
        return ("auto", current_tools)

    @staticmethod
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace the ``{broader_context}`` placeholder in *system* messages.

        The helper is fed into ``start_async_tool_loop`` via the
        ``preprocess_msgs`` parameter so that **every** LLM invocation sees a
        *fresh* broader-context snippet pulled from ``MemoryManager`` just
        before the request is dispatched.

        Parameters
        ----------
        msgs : list[dict]
            Messages to preprocess.

        Returns
        -------
        list[dict]
            Messages with the broader context injected into system prompts.
        """

        import copy

        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local to avoid cycles

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
