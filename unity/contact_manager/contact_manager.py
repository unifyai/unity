from typing import List, Dict, Optional, Callable, Any
import asyncio
import requests
import json
import functools
import os
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.embed_utils import EMBED_MODEL, ensure_vector_column
from ..knowledge_manager.types import ColumnType
from ..helpers import _handle_exceptions

import unify
from .types.contact import Contact
from .base import BaseContactManager
from ..events.event_bus import EventBus
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)


class ContactManager(BaseContactManager):

    def __init__(self, event_bus: EventBus, *, traced: bool = True) -> None:
        """
        Responsible managing the list of contact details stored upstream.
        """
        self._event_bus = event_bus

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        event_bus.register_event_types(["Contacts"])
        self._ctx = event_bus.ctxs["Contacts"]

        # ── immutable built-in columns ───────────────────────────────────
        self._REQUIRED_COLUMNS: set[str] = {
            "contact_id",
            "first_name",
            "surname",
            "email_address",
            "phone_number",
            "whatsapp_number",
            "description",
        }

        # ── schema-management internal tools (mirrors KnowledgeManager) ──
        self._schema_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._create_custom_column,
            self._delete_custom_column,
            self._list_columns,
            include_class_name=False,
        )

        # ── public tool dictionaries ─────────────────────────────────────
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._search_contacts,
                self._nearest_column,
                include_class_name=False,
            ),
            **self._schema_tools,
        }

        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._create_contact,
                self._update_contact,
                self._search_contacts,
                self._nearest_column,
                include_class_name=False,
            ),
            **self._schema_tools,
        }
        # Add tracing
        if traced:
            self = unify.traced(self)

    # ──────────────────────────────────────────────────────────────────────
    #  Column helpers (single-table version of KnowledgeManager's helpers)
    # ──────────────────────────────────────────────────────────────────────

    def _get_columns(self) -> Dict[str, str]:
        """Return {column_name: column_type} for the contacts table."""
        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields?project={proj}&context={self._ctx}"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        ret = response.json()
        return {k: v["data_type"] for k, v in ret.items()}

    def _list_columns(self, *, include_types: bool = True) -> Dict[str, Any]:
        """List current columns; with types if *include_types*."""
        cols = self._get_columns()
        return {"columns": cols} if include_types else set(cols)

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: ColumnType | str,
    ) -> Dict[str, str]:
        """
        **Add** a new optional column to the contacts table.
        """
        assert (
            column_name not in self._REQUIRED_COLUMNS
        ), f"'{column_name}' is a required column and cannot be recreated."

        if column_name in self._get_columns():
            raise ValueError(f"Column '{column_name}' already exists.")

        proj = unify.active_project()
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/fields"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        json_input = {
            "project": proj,
            "context": self._ctx,
            "fields": {column_name: str(column_type)},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(response)
        return response.json()

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        """
        **Remove** a custom column**. Built-in columns are protected.
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
        Ensure that *column* (e.g. ``"notes_vec"``) exists as a vector-embedding
        derived from *source* (e.g. ``"notes"``).  If it doesn't, create it.
        """
        ensure_vector_column(
            self._ctx,  # contacts live in a single context
            embed_column=column,
            source_column=source,
        )

    def _nearest_column(
        self,
        *,
        source: str,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        **Semantic nearest-neighbour search** over the *source* column.

        Parameters
        ----------
        source : str
            Name of the text column to embed (any default or custom column).
        text : str
            Query text.
        k : int, default ``5``
            Number of closest rows to return.

        Returns
        -------
        list[dict]
            Rows sorted by ascending cosine distance.
        """
        vec_col = f"{source}_vec"
        self._ensure_table_vector(column=vec_col, source=source)
        logs = unify.get_logs(
            context=self._ctx,
            sorting={
                f"cosine({vec_col}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [lg.entries for lg in logs]

    # Public #
    # -------#
    @functools.wraps(BaseContactManager.ask, updated=())
    def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
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
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client.set_system_message(build_ask_prompt(tools))
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseContactManager.update, updated=())
    def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
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
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        client.set_system_message(build_update_prompt(tools))
        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

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
        description: Optional[str] = None,
        custom_fields: Optional[Dict[str, ColumnType]] = None,
    ) -> int:
        """
        Persist a **new** contact record.

        Parameters
        ----------
        first_name : str | None
            Contact's first name. Must start with a capital letter and can only contain
            letters, spaces, periods and hyphens. May be *None*.
        surname : str | None
            Contact's surname/family name. Must start with a capital letter and can only
            contain letters, spaces, periods and hyphens. May be *None*.
        email_address : str | None
            Contact's email address. Must contain exactly one @ symbol with characters
            on either side. Must not clash with an existing record.
        phone_number : str | None
            Contact's phone number. Can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits. Must be unique.
        whatsapp_number : str | None
            Contact's WhatsApp number. Can optionally start with '+' (only if explicitly
            mentioned by the user), but must otherwise contain only digits. Must be unique.
        description : str | None
            A free-form text description of the contact. Can contain any additional notes
            or information about the contact. May be *None*.
        custom_fields : Dict[str, ColumnType] | None
            Additional contact information as key-value pairs, where keys are string column
            names and values are of type ColumnType. Can include any other relevant
            information about the contact. May be *None*.

        Returns
        -------
        int
            The **integer** ``contact_id`` of the newly created record.

        Raises
        ------
        AssertionError
            If *all* fields are ``None`` **or** if any uniqueness constraint
            (email / phone / WhatsApp) is violated.
        """

        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
            "description": description,
            **(custom_fields if custom_fields is not None else {}),
        }
        assert any(
            contact_details.values(),
        ), "At least one contact detail must be provided."

        # If it's the first contact, create immediately
        if not unify.get_logs(context=self._ctx):
            unify.log(
                context=self._ctx,
                **contact_details,
                contact_id=0,
                new=True,
                mutable=True,
            )
            return 0

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # ToDo: filter only for contact_id once supported in the Python utility function
        logs = unify.get_logs(
            context=self._ctx,
        )
        largest_id = max([lg.entries["contact_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new contact
        unify.log(
            context=self._ctx,
            **contact_details,
            contact_id=this_id,
            new=True,
            mutable=True,
        )
        return this_id

    def _update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        description: Optional[str] = None,
        custom_fields: Optional[Dict[str, ColumnType]] = None,
    ) -> int:
        """
        Modify **selected** (not `None`) fields of an existing contact.

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
            Contact's phone number - can optionally start with '+' (only if *explicitly*
            mentioned by the user), but must otherwise contain only digits.
        whatsapp_number : str | None
            Contact's WhatsApp number - can optionally start with '+' (only if *explicitly*
            mentioned by the user), but must otherwise contain only digits.
        description : str | None
            A free-form text description or notes about the contact.
        custom_fields : Dict[str, ColumnType] | None
            Additional contact information as key-value pairs, where keys are string column
            names and values are of type ColumnType. Can include any other relevant
            information about the contact. May be *None*.

        Returns
        -------
        int
            The contact's *unchanged* ``contact_id`` on success.

        Raises
        ------
        ValueError
            • When *no* updatable field is provided.
            • When *contact_id* does not exist.
            • When the new email / phone / WhatsApp value duplicates another
              record.
        """
        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
            "description": description,
            **(custom_fields if custom_fields is not None else {}),
        }
        updates_to_apply = [{k: v} for k, v in contact_details.items() if v is not None]
        if not updates_to_apply:
            raise ValueError(
                "At least one contact detail must be provided for an update.",
            )

        for key, value in contact_details.items():
            if (
                key in ["email_address", "phone_number", "whatsapp_number"]
                and value is not None
            ):
                logs = unify.get_logs(
                    context=self._ctx,
                    filter=f"{key} == '{value}' and contact_id != {contact_id}",
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
        return contact_id

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
        """
        Retrieve **one or many** contacts matching an arbitrary Python
        expression.

        Parameters
        ----------
        filter : str | None, default ``None``
            A boolean Python expression evaluated against each contact
            (e.g. ``"first_name == 'John' and surname == 'Doe'"``).  *None*
            returns **all** records.
        offset : int, default ``0``
            Index of the first result to return (0-based).
        limit : int, default ``100``
            Maximum number of records to return.

        Returns
        -------
        list[Contact]
            A list of Pydantic :class:`Contact` models in creation order.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]
