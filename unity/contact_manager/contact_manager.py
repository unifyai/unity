from typing import List, Dict, Optional, Callable, Any, Tuple
import asyncio
import requests
import json
import functools
import os
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..knowledge_manager.types import ColumnType
from ..helpers import _handle_exceptions
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields

import unify
from .types.contact import Contact
from .base import BaseContactManager
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
import asyncio


class ContactManager(BaseContactManager):
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
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        if read_ctx:
            self._ctx = f"{read_ctx}/Contacts"
        else:
            self._ctx = "Contacts"
        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_column_ids="contact_id",
                description="List of contacts, with all contact details stored.",
            )
            fields = model_to_fields(Contact)
            unify.create_fields(
                fields,
                context=self._ctx,
            )

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
                self._search_contacts,
                self._nearest_contacts,
                include_class_name=False,
            ),
        }

        # update-side tools are can read and write
        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._list_columns,
                self._search_contacts,
                self._nearest_contacts,
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
            base_fields["respond_to"] = False

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
            base_fields["respond_to"] = False
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
        """
        Return {column_name: column_type} for the contacts table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their types.
        """
        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={self._ctx}"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

    def _list_columns(self, *, include_types: bool = True) -> Dict[str, Any]:
        """
        List current columns; with types if include_types.

        Parameters
        ----------
        include_types : bool, default True
            Whether to include column types in output.

        Returns
        -------
        Dict[str, Any]
            Dictionary of columns, with types if requested.
        """
        cols = self._get_columns()
        return cols if include_types else set(cols)

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: ColumnType | str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Add a new optional column to the contacts table.

        Parameters
        ----------
        column_name : str
            The name of the column to create (which MUST be snake case).
        column_type : ColumnType | str
            The type of the column to create.
        column_description : Optional[str], default None
            Optional description of the column's purpose.

        Returns
        -------
        Dict[str, str]
            Dictionary containing the response from the Unify API.

        Raises
        ------
        AssertionError
            If column_name is a required column.
        ValueError
            If column already exists.
        """
        assert (
            column_name not in self._REQUIRED_COLUMNS
        ), f"'{column_name}' is a required column and cannot be recreated."

        if column_name in self._get_columns():
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
        return response.json()

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        """
        Remove a custom column. Built-in columns are protected.

        Parameters
        ----------
        column_name : str
            The name of the column to delete (which MUST be snake case).

        Returns
        -------
        Dict[str, str]
            Dictionary containing the response from the Unify API.

        Raises
        ------
        ValueError
            If column_name is a required column or does not exist.
        """
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(f"Cannot delete required column '{column_name}'.")

        if column_name not in self._get_columns():
            raise ValueError(f"Column '{column_name}' does not exist.")

        url = f"{os.environ['UNIFY_BASE_URL']}/logs?delete_empty_logs=True"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        json_input = {
            "project": unify.active_project(),
            "context": self._ctx,
            "ids_and_fields": [[None, column_name]],
            "source_type": "all",
        }
        response = requests.request("DELETE", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    # ------------------------------------------------------------------ #
    #  Vector-search helpers                                             #
    # ------------------------------------------------------------------ #

    def _ensure_table_vector(self, *, column: str, source: str) -> None:
        """
        Ensure that column exists as a vector-embedding derived from source.

        Parameters
        ----------
        column : str
            The (private) vector column name (e.g. "_notes_emb").
        source : str
            The source column name (e.g. "notes").
        """
        ensure_vector_column(
            self._ctx,  # contacts live in a single context
            embed_column=column,
            source_column=source,
        )

    # Public #
    # -------#
    @functools.wraps(BaseContactManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> SteerableToolHandle:
        # ── generate 1 call-id & log *incoming* request ─────────────────
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "ContactManager",
            "ask",
            phase="incoming",
            question=text,
        )

        client = unify.AsyncUnify(
            "o4-mini@openai",  # Consider making model configurable
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # Build a *live* tools-dict so the prompt never hard-codes
        # either the number of tools or their names/argspecs.
        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "Clarification queues not properly initialized for ask.",
                    )
                # 🔔 clarification requested
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=call_id,
                        payload={
                            "manager": "ContactManager",
                            "method": "ask",
                            "action": "clarification_request",
                            "question": question,
                        },
                    ),
                )
                await clarification_up_q.put(question)
                answer = await clarification_down_q.get()

                # 🔔 clarification answered
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=call_id,
                        payload={
                            "manager": "ContactManager",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": answer,
                        },
                    ),
                )
                return answer

            tools["request_clarification"] = request_clarification

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
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
            preprocess_msgs=self._inject_broader_context,
        )

        # wrap the raw handle so *every* public method logs an event
        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "ContactManager",
            "ask",
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseContactManager.update, updated=())
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> SteerableToolHandle:
        # ── event: incoming update request ──────────────────────────────
        call_id = new_call_id()
        await publish_manager_method_event(
            call_id,
            "ContactManager",
            "update",
            phase="incoming",
            request=text,
        )

        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "Clarification queues not properly initialized for update.",
                    )
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=call_id,
                        payload={
                            "manager": "ContactManager",
                            "method": "update",
                            "action": "clarification_request",
                            "question": question,
                        },
                    ),
                )
                await clarification_up_q.put(question)
                answer = await clarification_down_q.get()
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=call_id,
                        payload={
                            "manager": "ContactManager",
                            "method": "update",
                            "action": "clarification_answer",
                            "answer": answer,
                        },
                    ),
                )
                return answer

            tools["request_clarification"] = request_clarification

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(tools, include_activity=include_activity),
        )
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
            preprocess_msgs=self._inject_broader_context,
        )

        handle = wrap_handle_with_logging(
            handle,
            call_id,
            "ContactManager",
            "update",
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
        """
        ret = unify.get_logs_metric(
            metric="count",
            key="contact_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return ret

    # Private #
    # --------#

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
        custom_fields: Optional[Dict[str, ColumnType]] = None,
    ) -> ToolOutcome:
        """
        Persist a new contact record.

        Parameters
        ----------
        first_name : str | None
            Contact's first name. Must start with a capital letter and can only contain
            letters, spaces, periods and hyphens. May be None.
        surname : str | None
            Contact's surname/family name. Must start with a capital letter and can only
            contain letters, spaces, periods and hyphens. May be None.
        email_address : str | None
            Contact's email address. Must contain exactly one @ symbol with characters
            on either side. Must not clash with an existing record.
        phone_number : str | None
            Contact's phone number. Can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits. Must be unique.
        whatsapp_number : str | None
            Contact's WhatsApp number. Can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits. Must be unique.
        bio : str | None
            A free-form text description of the contact. Can contain any additional notes
            or information about the contact. May be None.
        custom_fields : Dict[str, ColumnType] | None
            Additional contact information as key-value pairs, where keys are string column
            names and values are of type ColumnType. Can include any other relevant
            information about the contact. May be None.

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

        Raises
        ------
        AssertionError
            If all fields are None or if any uniqueness constraint
            (email / phone / WhatsApp) is violated.
        """

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
        }

        # Merge any custom fields provided by the caller
        if custom_fields:
            contact_details.update(custom_fields)

        assert any(
            v is not None for v in contact_details.values()
        ), "At least one contact detail must be provided."

        # If it's the first contact, create immediately
        if not unify.get_logs(context=self._ctx):
            unify.log(
                context=self._ctx,
                **contact_details,
                new=True,
                mutable=True,
            )
            return {
                "outcome": "contact created successfully",
                "details": {"contact_id": 0},
            }

        # Verify uniqueness for contact fields that should be unique (emails,
        # phone numbers, etc.).  We use a simple heuristic to consider any
        # field ending in *_address or *_number as unique.
        unique_fields = {
            f
            for f in Contact.model_fields
            if f.endswith("_address") or f.endswith("_number")
        }

        for key, value in contact_details.items():
            if key not in unique_fields or value is None:
                continue
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == {value!r}",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

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
        custom_fields: Optional[Dict[str, ColumnType]] = None,
    ) -> ToolOutcome:
        """
        Modify selected (not None) fields of an existing contact.

        Parameters
        ----------
        contact_id : int
            Target record's unique identifier.
        first_name : str | None
            Contact's first name - must start with a capital letter and can only contain
            letters, spaces, periods and hyphens.
        surname : str | None
            Contact's surname/family name - must start with a capital letter and can only
            contain letters, spaces, periods and hyphens.
        email_address : str | None
            Contact's email address - must contain exactly one @ symbol with characters
            on either side.
        phone_number : str | None
            Contact's phone number - can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits.
        whatsapp_number : str | None
            Contact's WhatsApp number - can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits.
        bio : str | None
            A free-form text description or notes about the contact.
        custom_fields : Dict[str, ColumnType] | None
            Additional contact information as key-value pairs, where keys are string column
            names (which MUST be snake case) and values are of type ColumnType.
            Can include any other relevant information about the contact. May be None.

        Returns
        -------
        ToolOutcome
            Tool outcome with any extra relevant details.

        Raises
        ------
        ValueError
            When no updatable field is provided, when contact_id does not exist,
            or when the new email/phone/WhatsApp value duplicates another record.
        """
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
            "bio": bio,
            "rolling_summary": rolling_summary,
            "respond_to": respond_to,
        }

        if custom_fields:
            contact_details.update(custom_fields)

        updates_to_apply = [{k: v} for k, v in contact_details.items() if v is not None]
        if not updates_to_apply:
            raise ValueError(
                "At least one contact detail must be provided for an update.",
            )

        unique_fields = {
            f
            for f in Contact.model_fields
            if f.endswith("_address") or f.endswith("_number")
        }

        for key, value in contact_details.items():
            if key in unique_fields and value is not None:
                logs = unify.get_logs(
                    context=self._ctx,
                    filter=f"{key} == {value!r} and contact_id != {contact_id}",
                )
                if logs:
                    raise ValueError(
                        f"Another contact with {key} '{value}' already exists.",
                    )

        # Find the specific log entry to update
        target_logs = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
        )
        if not target_logs:
            raise ValueError(
                f"No contact found with contact_id {contact_id} to update.",
            )
        if len(target_logs) > 1:
            raise ValueError(
                f"Multiple contacts found with contact_id {contact_id}. Data integrity issue.",
            )

        log_to_update_id = target_logs[0].id  # Get the actual Unify log ID

        unify.update_logs(
            logs=[log_to_update_id] * len(updates_to_apply),
            context=self._ctx,
            entries=updates_to_apply,
            overwrite=True,
        )
        return {
            "outcome": "contact updated",
            "details": {"contact_id": contact_id},
        }

    def _delete_contact(
        self,
        *,
        contact_id: int,
    ) -> ToolOutcome:
        """
        Permanently **remove** a contact from storage.

        Parameters
        ----------
        contact_id : int
            Identifier of the contact to delete.

        Returns
        -------
        ToolOutcome
            Confirmation of deletion with the contact_id.
        """
        # Protect special contacts (assistant/user) from accidental deletion
        if contact_id in (0, 1):
            raise RuntimeError("Cannot delete system contacts with id 0 or 1.")

        log_ids = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
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

        unify.delete_logs(
            context=self._ctx,
            logs=log_ids,
        )
        return {
            "outcome": "contact deleted",
            "details": {"contact_id": contact_id},
        }

    def _merge_contacts(
        self,
        *,
        contact_id_1: int,
        contact_id_2: int,
        overrides: Dict[str, int],
    ) -> ToolOutcome:
        """
        Merge exactly two existing contacts into **one** consolidated record.

        The caller must provide a per-column *overrides* map indicating which of
        the two source contacts wins for that column.  The map values **must**
        be either ``1`` (take the value from *contact_id_1*) or ``2`` (take the
        value from *contact_id_2*).  Any column absent from *overrides* keeps
        the first non-``None`` value when scanned in the order
        ``contact_id_1`` → ``contact_id_2``.

        The *contact_id* itself can be overridden.  The resulting record keeps
        whichever id is chosen while the *other* contact is permanently
        deleted.  System contacts with ids 0 and 1 are **protected** and cannot
        be deleted.

        Parameters
        ----------
        contact_id_1 : int
            Identifier of the **first** source contact.
        contact_id_2 : int
            Identifier of the **second** source contact.
        overrides : Dict[str, int]
            Mapping ``column_name → 1 | 2`` picking the winner for that field.

        Returns
        -------
        ToolOutcome
            Confirmation payload indicating the kept/deleted contact ids.
        """

        if contact_id_1 == contact_id_2:
            raise ValueError("contact_id_1 and contact_id_2 must be distinct.")

        if any(v not in (1, 2) for v in overrides.values()):
            raise ValueError(
                "Override values must be 1 or 2, referring to the corresponding contact id argument.",
            )

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
                custom_fields=custom_updates or None,
            )

        # Delete the other contact
        self._delete_contact(contact_id=delete_id)

        # ──────────────────────────────────────────────────────────────
        # Keep transcript history consistent by rewriting old ids
        # ──────────────────────────────────────────────────────────────
        # Local import to prevent heavy top-level dependency and possible
        # circular-import issues at module load time.
        from unity.transcript_manager.transcript_manager import (
            TranscriptManager,
        )  # noqa: WPS433

        # Re-use *this* ContactManager instance to avoid creating a second
        # one inside TranscriptManager which would trigger another round of
        # context/column checks.
        tm = TranscriptManager(contact_manager=self)
        # Update all sender/receiver occurrences of the deleted id so that
        # future transcript queries remain consistent with the merged
        # contact record.
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

    def _nearest_contacts(
        self,
        *,
        column: str,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic nearest-neighbour search over the source column.

        Parameters
        ----------
        column : str
            Name of the text column to embed (any default or custom column).
        text : str
            Query text.
        k : int, default 5
            Number of closest rows to return.

        Returns
        -------
        List[Dict[str, Any]]
            Rows sorted by ascending cosine distance.
        """
        vec_col = f"_{column}_emb"
        self._ensure_table_vector(column=vec_col, source=column)
        logs = unify.get_logs(
            context=self._ctx,
            sorting={
                f"cosine({vec_col}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._ctx).keys()
                if k.endswith("_emb")
            ],
        )
        return [Contact(**lg.entries) for lg in logs]

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
        """
        Retrieve one or many contacts matching an arbitrary Python expression.

        Parameters
        ----------
        filter : str | None, default None
            A boolean Python expression evaluated against each contact
            (e.g. "first_name == 'John' and surname == 'Doe'"). None
            returns all records.
        offset : int, default 0
            Index of the first result to return (0-based).
        limit : int, default 100
            Maximum number of records to return.

        Returns
        -------
        List[Contact]
            A list of Pydantic Contact models in creation order.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            exclude_fields=[
                k
                for k in unify.get_fields(context=self._ctx).keys()
                if k.endswith("_emb")
            ],
        )
        return [Contact(**lg.entries) for lg in logs]

    @staticmethod
    def _inject_broader_context(msgs: list[dict]) -> list[dict]:
        """Replace the ``{broader_context}`` placeholder in *system* messages.

        The helper is fed into ``start_async_tool_use_loop`` via the
        ``preprocess_msgs`` parameter so that **every** LLM invocation sees a
        *fresh* broader-context snippet pulled from ``MemoryManager`` just
        before the request is dispatched.
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
