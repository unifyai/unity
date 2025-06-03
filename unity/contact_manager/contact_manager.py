from typing import List, Dict, Optional, Callable, Any
import asyncio
import functools
import json
import os

import unify
from .types.contact import Contact
from .base import BaseContactManager
from ..events.event_bus import EventBus
from ..common.llm_helpers import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    methods_to_tool_dict,
)
from .sys_msgs import ASK_CONTACTS, UPDATE_CONTACTS


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

        # Define tools for ask and update methods
        self._ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._search_contacts,
            include_class_name=False,
        )
        self._update_tools: Dict[str, Callable] = methods_to_tool_dict(
            self._create_contact,
            self._update_contact,
            self._search_contacts,
            include_class_name=False,
        )
        # Add tracing
        if traced:
            self = unify.traced(self)

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
        client.set_system_message(ASK_CONTACTS)

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
        client.set_system_message(UPDATE_CONTACTS)

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
    ) -> int:
        """
        Creates a new contact with the following contact details, as available.

        Args:
            first_name (str): The first name of the contact.
            surname (str): The surname of the contact.
            email_address (str): The email address of the contact.
            phone_number (str): The phone number of the contact.
            whatsapp_number (str): The WhatsApp number of the contact.
        Returns:
            int: The contact_id of the newly created contact.
        """

        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
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
    ) -> int:
        """
        Update the contact details of a contact.

        Args:
            contact_id (int): The id of the contact to update.
            first_name (Optional[str]): The first name of the contact.
            surname (Optional[str]): The surname of the contact.
            email_address (Optional[str]): The email address of the contact.
            phone_number (Optional[str]): The phone number of the contact.
            whatsapp_number (Optional[str]): The WhatsApp number of the contact.

        Returns:
            int: The contact_id of the updated contact.
        """
        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
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
        Retrieve contact details, based on flexible filtering for first name, surname, email address, WhatsApp number, phone number, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the contacts.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of contacts to retrieve.

        Returns:
            List[Contact]: A list of contacts.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]
