from typing import List, Dict, Optional, Callable, Any, Tuple
import asyncio
import requests
import json
import functools
import os
import re
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.embed_utils import ensure_vector_column
from ..knowledge_manager.types import ColumnType
from ..helpers import _handle_exceptions
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore

import unify
from .types.contact import Contact
from .base import BaseContactManager
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
    inject_broader_context,
    make_request_clarification_tool,
    TOOL_LOOP_LINEAGE,
)
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_call
import asyncio
from ..common.semantic_search import (
    fetch_top_k_by_references,
    backfill_rows,
)

# ------------------------------------------------------------------ #
#  Optional per-tool runtime logging                                  #
# ------------------------------------------------------------------ #
import time


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw_l = str(raw).strip().lower()
    return raw_l in {"1", "true", "yes", "on"}


def _timing_enabled() -> bool:
    # Per-manager override → global fallback
    return _env_truthy(
        "CONTACT_MANAGER_TOOL_TIMING",
        _env_truthy("TOOL_TIMING", False),
    )


def _timing_print_enabled() -> bool:
    return _env_truthy(
        "CONTACT_MANAGER_TOOL_TIMING_PRINT",
        _env_truthy("TOOL_TIMING_PRINT", False),
    )


def _log_tool_runtime(func):
    """Decorator to measure and optionally publish per-tool runtimes.

    When CONTACT_MANAGER_TOOL_TIMING is truthy, publishes a ManagerTool event via
    EVENT_BUS containing the tool name, category (ask/update/direct) and duration_ms.
    Printing can be enabled with CONTACT_MANAGER_TOOL_TIMING_PRINT.
    """

    @functools.wraps(func, updated=())
    def _wrapper(self: "ContactManager", *args, **kwargs):
        start = time.perf_counter()
        try:
            return func(self, *args, **kwargs)
        finally:
            try:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            except Exception:
                elapsed_ms = -1.0

            if _timing_print_enabled():
                try:
                    print(f"ContactManager.{func.__name__} took {elapsed_ms:.2f} ms")
                except Exception:
                    pass

            if not _timing_enabled():
                return

            # Determine category best-effort at runtime
            try:
                if (
                    isinstance(getattr(self, "_ask_tools", None), dict)
                    and func.__name__ in self._ask_tools
                ):
                    category = "ask"
                elif (
                    isinstance(getattr(self, "_update_tools", None), dict)
                    and func.__name__ in self._update_tools
                ):
                    category = "update"
                else:
                    category = "direct"
            except Exception:
                category = "direct"

            # Publish as a lightweight event if the bus is ready and a loop is running
            try:
                evt = Event(
                    type="ManagerTool",
                    payload={
                        "manager": "ContactManager",
                        "tool": func.__name__,
                        "category": category,
                        "duration_ms": float(elapsed_ms),
                    },
                )
                # Only publish when EVENT_BUS is initialised and an event loop exists
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and EVENT_BUS:
                    asyncio.create_task(EVENT_BUS.publish(evt))
            except Exception:
                # Swallow any timing/logging issues – never affect tool behaviour
                pass

    return _wrapper


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
        # Ensure context/fields exist deterministically (idempotent)
        self._store = TableStore(
            self._ctx,
            unique_keys={"contact_id": "int"},
            auto_counting={"contact_id": None},
            description="List of contacts, with all contact details stored.",
            fields=model_to_fields(Contact),
        )
        self._store.ensure_context()

        # ── immutable built-in columns ───────────────────────────────────
        # Derive the required/built-in columns directly from the Contact model so
        # that there is a single source-of-truth for field names across the
        # code-base.  Any future change to the Contact schema will
        # automatically propagate here.
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Contact.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        # ── public tool dictionaries ─────────────────────────────────────
        # ask-side tools are read-only, so they never change
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._list_columns,
                self._filter_contacts,
                self._search_contacts,
                include_class_name=False,
            ),
        }

        # update-side tools are can read and write
        self._update_tools: Dict[str, Callable] = {
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

        # rolling activity inclusion flag
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Track custom columns observed/created during this manager's lifetime so we can
        # whitelist fields in a single read without a separate fields-introspection call.
        # This avoids redundant backend calls inside tools like `_filter_contacts` while
        # still returning custom fields commonly used right after creation/update.
        self._known_custom_fields: set[str] = set()

        # Prefill known custom fields once at construction to include any preexisting
        # non-private columns without an extra lookup per tool call.
        try:
            existing_cols = self._get_columns()
            for col in existing_cols:
                if col not in self._REQUIRED_COLUMNS and not str(col).startswith("_"):
                    self._known_custom_fields.add(col)
        except Exception:
            # Best-effort only; tools fall back safely
            pass

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
        url = f"{os.environ['UNIFY_BASE_URL']}/assistant?"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        data = response.json()
        return data.get("info", []) if isinstance(data, dict) else []

    def _ensure_columns_exist(self, extra_fields: Dict[str, Any]) -> None:
        """Create custom columns for *extra_fields* that are not yet present.

        Only simple string columns are created for now – if richer typing is
        required in future we can extend the heuristics.

        Parameters
        ----------
        extra_fields : Dict[str, Any]
            Extra fields that are not yet present.
        """
        existing_cols = self._get_columns()
        for col in extra_fields:
            if col in self._REQUIRED_COLUMNS or col in existing_cols:
                continue
            try:
                # Default to string type for new assistant metadata columns
                self._create_custom_column(
                    column_name=col,
                    column_type=ColumnType.str,
                )
            except Exception:
                # Column may have been created concurrently – ignore
                pass

    def _sync_assistant_contact(self) -> None:
        """Ensure the *current* assistant (id == 0 in this context) exists and is correct.

        Notes
        -----
        The assistant record is selected using the following precedence:

        1. The globally initialised ``unity.ASSISTANT`` object – this is set
           by :pyfunc:`unity.init` *after* validating that the requested
           ``assistant_id`` exists via the Unify API.
        2. Fallback to the *assistant_index* implied by the active context
           (i.e. ``unify.get_active_context()['read']``) when the global
           variable is ``None``.
        3. If neither method yields a record or the API returns an empty list
           (offline tests), a dummy placeholder assistant is created.
        """

        from .. import ASSISTANT as _GLOBAL_ASSISTANT  # local import to avoid cycles

        assistants = self._fetch_assistant_info()

        # 1) Prefer the assistant provided by unity.init
        if _GLOBAL_ASSISTANT is not None:
            selected = _GLOBAL_ASSISTANT

        # 2) Otherwise map the active context (if numeric) onto the list index
        else:
            ctxs = unify.get_active_context()
            read_ctx = ctxs.get("read")
            try:
                idx = int(read_ctx) if read_ctx is not None else 0
            except (TypeError, ValueError):
                idx = 0

            selected = assistants[idx] if idx < len(assistants) else None

        # 3) No assistant found – will create a dummy record

        # ------------------------------------------------------------------
        # Build the canonical assistant record (real or dummy)
        # ------------------------------------------------------------------

        if selected is not None:
            a = selected
            # Start with a dictionary that contains *all* builtin fields (except
            # contact_id) set to None so we never forget to initialise a field if
            # the Contact schema evolves.
            base_fields = {
                fld: None for fld in self._BUILTIN_FIELDS if fld != "contact_id"
            }
            base_fields["respond_to"] = True
            base_fields["response_policy"] = ""

            # Map assistant API payload → Contact fields.  We still spell the
            # Contact field names exactly *once* here, centralising the mapping
            # logic in a single place.
            base_fields.update(
                {
                    "first_name": a.get("first_name"),
                    "surname": a.get("surname"),
                    "email_address": a.get("email"),
                    "phone_number": a.get("phone"),
                    "whatsapp_number": a.get("phone"),
                    "bio": a.get("about"),
                    "rolling_summary": None,
                },
            )
            # Everything else is stored verbatim as custom fields
            mapped_keys = {"first_name", "surname", "email", "phone", "about"}
        else:
            # Dummy assistant when account has no assistants configured – again
            # start with all builtin fields set to None and then populate the
            # known ones so that we never miss a schema update.
            base_fields = {
                fld: None for fld in self._BUILTIN_FIELDS if fld != "contact_id"
            }
            base_fields["respond_to"] = True
            base_fields["response_policy"] = ""
            base_fields.update(
                {
                    "first_name": "Unify",
                    "surname": "Assistant",
                    "email_address": "unify.assistant@unify.ai",
                    "phone_number": "+10000000000",
                    "whatsapp_number": "+10000000000",
                    "bio": "Your helpful Unify AI assistant.",
                    "rolling_summary": None,
                },
            )

        # ------------------------------------------------------------------
        # Retrieve contact_id == 0 (if any) and decide whether to create/update
        # ------------------------------------------------------------------
        existing_logs = unify.get_logs(
            context=self._ctx,
            filter="contact_id == 0",
            limit=1,
        )

        # If the assistant contact already exists **leave it untouched** – never
        # overwrite any backend-curated fields on initialisation.
        if existing_logs:
            return  # Contact is present – nothing to sync.

        if not existing_logs:
            # Either the table is empty or contact_id 0 was never created.
            # Use the standard helper which will assign contact_id == 0 when
            # inserting the first contact into an empty table.  If the table
            # already had contacts, fall back to a direct log with explicit id.
            if not unify.get_logs(context=self._ctx):
                self._create_contact(**base_fields)
            else:
                # Direct log insertion with explicit contact_id 0
                unify.log(
                    context=self._ctx,
                    contact_id=0,
                    **base_fields,
                    new=True,
                    mutable=True,
                )
            return  # nothing further to do

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

        url = f"{os.environ['UNIFY_BASE_URL']}/user/basic-info"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)

        # Raise for HTTP errors so the except-block handles them uniformly
        _handle_exceptions(response)

        data: Any = response.json()
        if isinstance(data, dict):
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
        """Ensure the default *user* (id == 1) contact exists and is correct."""
        user_info = self._fetch_user_info()

        # Build defaults for *all* built-in fields (except the primary key)
        base_fields: Dict[str, Any] = {
            fld: None
            for fld in self._BUILTIN_FIELDS
            if fld not in {"contact_id", "bio", "rolling_summary"}
        }
        # Default for user: respond_to True so assistant replies
        base_fields["respond_to"] = True

        # Merge in the real user metadata that we discovered.  Crucially we
        # *omit* the ``bio`` and ``rolling_summary`` keys so any manually
        # curated text is **preserved** rather than being reset to ``None``
        # every time a new ContactManager instance is created.
        base_fields.update(
            {
                "first_name": user_info.get("first_name"),
                # Map provided *last_name* → Contact.surname
                "surname": user_info.get("last_name"),
                "email_address": user_info.get("email"),
                "phone_number": user_info.get("phone_number"),
                "whatsapp_number": user_info.get("whatsapp_number"),
                "response_policy": self.USER_MANAGER_RESPONSE_POLICY,
            },
        )

        # Ensure any additional fields (not part of the built-ins) exist as
        # custom columns so future API expansions are tolerated.
        extra_fields = {
            k: v
            for k, v in user_info.items()
            if k
            not in {
                "first_name",
                "last_name",
                "email",
                "phone_number",
                "whatsapp_number",
            }
        }
        if extra_fields:
            self._ensure_columns_exist(extra_fields)

        # ------------------------------------------------------------------
        # Create or update contact_id == 1
        # ------------------------------------------------------------------
        existing_logs = unify.get_logs(
            context=self._ctx,
            filter="contact_id == 1",
            limit=1,
        )

        # If the user contact already exists **leave it untouched** – protect all
        # backend data from accidental resets during ContactManager construction.
        if existing_logs:
            return  # Contact is present – nothing to sync.

        if not existing_logs:
            # No user contact yet → create it.  We *do not* supply a
            # `contact_id` so that Unify allocates the next auto-incremented
            # value.  Provided the assistant was inserted first this will be
            # **1** as desired.  If the table was initially empty the
            # assistant sync ensures id 0 is reserved before we reach here.

            self._create_contact(
                **{k: v for k, v in base_fields.items() if v is not None},
            )
            return  # done

    # ──────────────────────────────────────────────────────────────────────
    #  Column helpers (single-table version of KnowledgeManager's helpers)
    # ──────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """Return {column_name: column_type} for the contacts table."""
        return self._store.get_columns()

    # Apply timing to tool methods
    @_log_tool_runtime
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

    @_log_tool_runtime
    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: ColumnType | str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Create a new optional (mutable) column on the contacts table.

        Parameters
        ----------
        column_name : str
            The exact column key to create. Requirements:
            - Must be snake_case (letters, digits, and underscores; starts with a letter).
            - Must not collide with any built‑in (required) columns of the ``Contact`` schema.
            - Must not already exist in the table.
        column_type : ColumnType | str
            Logical type for the column. Accepts either the enum ``ColumnType`` or one of
            its string values. Common values include: ``"str"``, ``"int"``, ``"float"``,
            ``"bool"``, ``"list"``, ``"dict"``, ``"datetime"``, ``"date"``, ``"time"``.
            Choose the type that best matches the data you intend to store.
        column_description : Optional[str], default None
            Optional human‑readable description to help other users understand the column.

        Returns
        -------
        Dict[str, str]
            The Unify API response payload acknowledging column creation.

        Raises
        ------
        AssertionError
            If ``column_name`` is one of the built‑in/required columns and therefore
            cannot be (re)created.
        ValueError
            If a column with the same name already exists.

        Usage Guidance
        --------------
        - Prefer concise names that describe the field's purpose (e.g. ``"linkedin_url"``).
        - If you need to store vectors/embeddings, use the dedicated vector helpers instead;
          this method creates standard mutable columns.
        - After creating the column you can write values via ``_create_contact`` or
          ``_update_contact`` using the same key.
        """
        assert (
            column_name not in self._REQUIRED_COLUMNS
        ), f"'{column_name}' is a required column and cannot be recreated."

        # Fast local validation to avoid unnecessary network round-trips
        # Enforce simple snake_case starting with a letter
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column_name):
            raise ValueError(
                "column_name must be snake_case: start with a letter, then letters/digits/underscores",
            )

        # Avoid a pre-flight GET to check for existence; rely on our singleton's
        # private state which is kept in sync at construction and on create/delete.
        # This prevents an extra blocking backend read on every create call.
        if (
            getattr(self, "_known_custom_fields", None)
            and column_name in self._known_custom_fields
        ):
            raise ValueError(f"Column '{column_name}' already exists.")

        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        column_info = {
            "type": str(column_type),
            "mutable": True,
        }
        if column_description is not None:
            column_info["description"] = column_description
        json_input = {
            "project": proj,
            "context": self._ctx,
            "fields": {
                column_name: column_info,
            },
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        # Remember the new column for subsequent reads within this manager instance
        try:
            self._known_custom_fields.add(column_name)
        except Exception:
            pass
        return response.json()

    @_log_tool_runtime
    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        """
        Delete a previously created custom column from the contacts table.

        Parameters
        ----------
        column_name : str
            The exact name of the column to remove. Must be a non‑required (custom)
            column that currently exists. Snake_case is expected.

        Returns
        -------
        Dict[str, str]
            The Unify API response payload acknowledging deletion.

        Raises
        ------
        ValueError
            - If ``column_name`` is a built‑in/required column (protected against deletion).
            - If the column does not exist.

        Notes
        -----
        - Deletion is performed with ``delete_empty_logs=True`` to clean up empty records
          if applicable.
        - Removing a column permanently drops its values from all contacts. This action
          cannot be undone.
        """
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(f"Cannot delete required column '{column_name}'.")

        # Avoid a pre-read of fields; attempt deletion directly via the
        # dedicated field-deletion endpoint which removes both the field
        # definition and all associated entries in a single backend call.
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        json_input = {
            "project": unify.active_project(),
            "context": self._ctx,
            "fields": [column_name],
        }
        response = requests.request("DELETE", url, json=json_input, headers=headers)
        _handle_exceptions(response)

        payload: Dict[str, Any] = {}
        try:
            payload = response.json()
        except Exception:
            payload = {}

        # If the backend returns the list of deleted fields and our target
        # isn't included, treat it as a non-existent column for parity with
        # previous semantics.
        deleted_fields = None
        if isinstance(payload, dict):
            deleted_fields = payload.get("deleted_fields")
        if isinstance(deleted_fields, list) and column_name not in deleted_fields:
            raise ValueError(f"Column '{column_name}' does not exist.")

        # Fallback for environments where DELETE /logs/fields is not implemented
        # (e.g., test stubs). When no structured confirmation is present, issue a
        # single idempotent deletion via the generic logs endpoint which will drop
        # the field values and clean up the field definition.
        if not isinstance(deleted_fields, list):
            fallback_url = f"{os.environ['UNIFY_BASE_URL']}/logs?delete_empty_logs=True"
            fallback_body = {
                "project": unify.active_project(),
                "context": self._ctx,
                "ids_and_fields": [[None, column_name]],
                "source_type": "all",
            }
            fb_resp = requests.request(
                "DELETE",
                fallback_url,
                json=fallback_body,
                headers=headers,
            )
            _handle_exceptions(fb_resp)
            try:
                payload = fb_resp.json()
            except Exception:
                pass

        # Update local view of known custom columns on success
        try:
            if column_name in getattr(self, "_known_custom_fields", set()):
                self._known_custom_fields.discard(column_name)
        except Exception:
            pass

        return payload

    # ------------------------------------------------------------------ #
    #  Vector-search helpers                                             #
    # ------------------------------------------------------------------ #

    def _ensure_table_vector(
        self,
        *,
        column: str,
        source_expr: str,
    ) -> None:
        """
        Ensure that an embedding column exists for the provided source expression.

        Parameters
        ----------
        column : str
            The (private) vector column name (e.g. "_notes_emb"). Must end with
            the suffix "_emb". The corresponding source column name will be
            derived by stripping the suffix.
        source_expr : str
            A Unify expression string that produces the source text to embed.
            This may be either:
            - a plain column name like "bio" (treated as an existing column), or
            - a full expression using Unify's expression language, e.g.
              "str({first_name}) + ' ' + str({surname})".

        Notes
        -----
        When a plain column name is provided, the function will reference that
        column directly. When a full expression is provided, a derived source
        column will be created (if needed) using the name obtained by removing
        the trailing "_emb" from the provided embedding column key.
        """
        # Derive a stable source column key from the embedding column name.
        source_column_name = column[:-4] if column.endswith("_emb") else f"{column}_src"

        # Heuristic: treat simple identifiers (no braces or ops) as direct columns
        is_plain_identifier = (
            "{" not in source_expr
            and "}" not in source_expr
            and any(c.isalpha() for c in source_expr)
        )

        if is_plain_identifier:
            # Use the provided identifier as the source column directly
            ensure_vector_column(
                self._ctx,
                embed_column=column,
                source_column=source_expr,
                derived_expr=None,
            )
        else:
            # Treat the input as a full expression that defines/derives the source
            ensure_vector_column(
                self._ctx,
                embed_column=column,
                source_column=source_column_name,
                derived_expr=source_expr,
            )

    # Public #
    # -------#
    @functools.wraps(BaseContactManager.ask, updated=())
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
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        # Build a *live* tools-dict so the prompt never hard-codes
        # either the number of tools or their names/argspecs.
        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
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

            async def _on_answer(ans: str):
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

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools=tools,
                num_contacts=self._num_contacts(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            preprocess_msgs=inject_broader_context,
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
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
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

            async def _on_answer(ans: str):
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

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(
                tools,
                num_contacts=self._num_contacts(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            preprocess_msgs=inject_broader_context,
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

    # Private #
    # --------#

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

    @_log_tool_runtime
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
        # remove and unpack kwargs from kwargs, if passed by mistaken LLM
        if "kwargs" in kwargs:
            kwargs = {**kwargs, **kwargs.pop("kwargs")}

        # Build the contact dictionary directly from the arguments
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
            "bio": bio,
            "rolling_summary": rolling_summary,
            "respond_to": respond_to,
            "response_policy": response_policy,
        }

        # Apply default response policy if none provided
        if contact_details["response_policy"] is None:
            contact_details["response_policy"] = self.DEFAULT_RESPONSE_POLICY

        # Merge any custom columns provided by the caller (sanitised first).
        if kwargs:
            safe_custom = self._sanitize_custom_columns(kwargs)
            contact_details.update(safe_custom)
            # Track keys so subsequent reads in this instance can whitelist them
            try:
                for k in safe_custom.keys():
                    if k not in self._BUILTIN_FIELDS:
                        self._known_custom_fields.add(k)
            except Exception:
                pass

        assert any(
            v is not None for v in contact_details.values()
        ), "At least one contact detail must be provided."

        # Verify uniqueness for contact fields that should be unique (emails,
        # phone numbers, etc.).  We use a simple heuristic to consider any
        # field ending in *_address or *_number as unique.
        unique_fields = {
            f
            for f in Contact.model_fields
            if f.endswith("_address") or f.endswith("_number")
        }

        # Perform a single existence check across all provided unique fields
        provided_unique_constraints = [
            f"{key} == {value!r}"
            for key, value in contact_details.items()
            if key in unique_fields and value is not None
        ]

        if provided_unique_constraints:
            or_expr = " or ".join(provided_unique_constraints)
            dupes = unify.get_logs(
                context=self._ctx,
                filter=or_expr,
                limit=1,
                return_ids_only=True,
            )
            assert (
                len(dupes) == 0
            ), "Invalid, contact with a provided unique field already exists."

        # Create the new contact
        log = unify.log(
            context=self._ctx,
            **contact_details,
            new=True,
            mutable=True,
        )
        return {
            "outcome": "contact created successfully",
            "details": {"contact_id": log.entries["contact_id"]},
        }

    @_log_tool_runtime
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

        # remove and unpack kwargs from kwargs, if passed by mistaken LLM
        if "kwargs" in kwargs:
            kwargs = {**kwargs, **kwargs.pop("kwargs")}

        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
            "bio": bio,
            "rolling_summary": rolling_summary,
            "respond_to": respond_to,
            "response_policy": response_policy,
        }
        # Merge any additional custom columns (sanitised first)
        if kwargs:
            safe_custom = self._sanitize_custom_columns(kwargs)
            contact_details.update(safe_custom)
            try:
                for k in safe_custom.keys():
                    if k not in self._BUILTIN_FIELDS:
                        self._known_custom_fields.add(k)
            except Exception:
                pass

        # Collapse updates into a single dict so we only perform one write op
        updates_dict = {k: v for k, v in contact_details.items() if v is not None}
        if not updates_dict:
            raise ValueError(
                "At least one contact detail must be provided for an update.",
            )

        unique_fields = {
            f
            for f in Contact.model_fields
            if f.endswith("_address") or f.endswith("_number")
        }

        # Perform a single existence check across all provided unique fields
        provided_unique_constraints = [
            f"{key} == {value!r}"
            for key, value in contact_details.items()
            if key in unique_fields and value is not None
        ]
        if provided_unique_constraints:
            or_expr = " or ".join(provided_unique_constraints)
            dupes = unify.get_logs(
                context=self._ctx,
                filter=f"({or_expr}) and contact_id != {contact_id}",
                limit=1,
                return_ids_only=True,
            )
            if dupes:
                raise ValueError(
                    "Another contact already exists with one of the provided unique fields.",
                )

        # Find the specific log entry to update
        target_ids = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
            return_ids_only=True,
        )
        if not target_ids:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to update.",
            )
        if len(target_ids) > 1:
            raise ValueError(
                f"Multiple contacts found with contact_id {contact_id}. Data integrity issue.",
            )

        log_to_update_id = target_ids[0]

        unify.update_logs(
            logs=[log_to_update_id],
            context=self._ctx,
            entries=updates_dict,
            overwrite=True,
        )
        return {
            "outcome": "contact updated",
            "details": {"contact_id": contact_id},
        }

    @_log_tool_runtime
    def _delete_contact(
        self,
        *,
        contact_id: int,
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
        # Protect special contacts (assistant/user) from accidental deletion
        if contact_id in (0, 1):
            raise RuntimeError("Cannot delete system contacts with id 0 or 1.")

        # Minimise backend scan work while preserving duplicate detection by
        # capping the lookup to at most two rows.
        log_ids = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
            limit=2,
            return_ids_only=True,
        )
        if not log_ids:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to delete.",
            )
        if len(log_ids) > 1:
            raise RuntimeError(
                f"Multiple contacts found with contact_id {contact_id}. Data integrity issue.",
            )

        # Pass a single integer id to avoid wrapping in a list
        unify.delete_logs(
            context=self._ctx,
            logs=log_ids[0],
        )
        return {
            "outcome": "contact deleted",
            "details": {"contact_id": contact_id},
        }

    @_log_tool_runtime
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

        if contact_id_1 == contact_id_2:
            raise ValueError("contact_id_1 and contact_id_2 must be distinct.")

        if overrides is not None:
            if any(v not in (1, 2) for v in overrides.values()):
                raise ValueError(
                    "Override values must be 1 or 2, referring to the corresponding contact id argument.",
                )
        else:
            overrides = {}

        # Retrieve both contacts
        def _fetch(cid: int):
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"contact_id == {cid}",
                limit=1,
            )
            if not logs:
                raise ValueError(f"No contact found with contact_id {cid}.")
            return logs[0]

        log1 = _fetch(contact_id_1)
        log2 = _fetch(contact_id_2)

        entries1 = log1.entries
        entries2 = log2.entries

        # Decide which contact id to keep
        keep_id = contact_id_1 if overrides.get("contact_id", 1) == 1 else contact_id_2
        delete_id = contact_id_2 if keep_id == contact_id_1 else contact_id_1

        # Protect system contacts from deletion
        if delete_id in (0, 1):
            raise RuntimeError(
                "Cannot delete system contacts with id 0 or 1 during merge.",
            )

        # Build the consolidated field map (skip contact_id – handled separately)
        consolidated: Dict[str, Any] = {}

        all_cols = set(self._get_columns())
        all_cols.discard("contact_id")

        for col in all_cols:
            # Ignore private vector columns ("*_emb")
            if col.endswith("_emb"):
                continue

            if col in overrides:
                source = overrides[col]
                value = entries1.get(col) if source == 1 else entries2.get(col)
            else:
                value = entries1.get(col)
                if value is None:
                    value = entries2.get(col)

            if value is not None:
                consolidated[col] = value

        # Split consolidated fields into built-in vs custom for _update_contact
        builtin_updates = {
            k: v for k, v in consolidated.items() if k in self._BUILTIN_FIELDS
        }
        custom_updates = {
            k: v for k, v in consolidated.items() if k not in self._BUILTIN_FIELDS
        }

        # Apply updates to the kept contact
        if builtin_updates or custom_updates:
            self._update_contact(
                contact_id=keep_id,
                **{
                    k: builtin_updates.get(k)
                    for k in self._BUILTIN_FIELDS
                    if k in builtin_updates
                },
                **(custom_updates or {}),
            )

        # Delete the other contact
        self._delete_contact(contact_id=delete_id)

        # Keep transcript history consistent by rewriting old ids
        from unity.transcript_manager.transcript_manager import (
            TranscriptManager,
        )  # noqa: WPS433

        tm = TranscriptManager(contact_manager=self)
        tm._update_contact_id(
            original_contact_id=delete_id,
            new_contact_id=keep_id,
        )

        return {
            "outcome": "contacts merged successfully",
            "details": {
                "kept_contact_id": keep_id,
                "deleted_contact_id": delete_id,
            },
        }

    @_log_tool_runtime
    def _search_contacts(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Contact]:
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
        # Restrict payloads to built‑in + known custom columns to avoid an
        # upfront fields lookup and reduce transfer size. Semantic sorting uses
        # private columns server‑side and does not require them in the payload.
        allowed_fields = list(self._BUILTIN_FIELDS)
        if getattr(self, "_known_custom_fields", None):
            allowed_fields.extend(sorted(self._known_custom_fields))

        rows = fetch_top_k_by_references(
            self._ctx,
            references,
            k=k,
            allowed_fields=allowed_fields,
            row_filter=None,
        )
        filled = backfill_rows(
            self._ctx,
            rows,
            k,
            unique_id_field="contact_id",
            allowed_fields=allowed_fields,
        )
        return [Contact(**r) for r in filled]

    @_log_tool_runtime
    def _filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
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
        # If the client does not support `from_fields`, fall back to excluding
        # private fields using a lightweight introspection.
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

        try:
            # Read built‑ins plus any custom columns we observed in this instance
            from_fields = list(self._BUILTIN_FIELDS)
            if getattr(self, "_known_custom_fields", None):
                from_fields.extend(sorted(self._known_custom_fields))
            logs = unify.get_logs(
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=eff_limit,
                from_fields=from_fields,
            )
        except TypeError:
            # Older client without from_fields support → avoid an extra
            # get_fields call and fetch once with the tightened limit.
            logs = unify.get_logs(
                context=self._ctx,
                filter=filter,
                offset=offset,
                limit=eff_limit,
            )
        return [Contact(**lg.entries) for lg in logs]

    # ------------------------------------------------------------------ #
    #  Small internal helpers (LLM client + tool policies)               #
    # ------------------------------------------------------------------ #

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        """Construct a configured AsyncUnify client for the given model."""
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )

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
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace the ``{broader_context}`` placeholder in *system* messages.

        The helper is fed into ``start_async_tool_use_loop`` via the
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
