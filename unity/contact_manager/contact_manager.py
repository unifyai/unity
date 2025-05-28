from typing import List, Dict, Optional

import unify
from .types.contact import Contact
from ..events.event_bus import EventBus


class ContactManager:

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

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    def create_contact(
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
            int: The id of the newly created contact.
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
            return unify.log(
                context=self._ctx,
                **contact_details,
                contact_id=0,
                new=True,
            ).id

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
        return unify.log(
            context=self._ctx,
            **contact_details,
            contact_id=this_id,
            new=True,
        ).id

    def update_contact(
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
            int: The id of the updated contact.
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

        # get log id
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert len(logs) == 1
        log: unify.Log = logs[0]
        log.update_entries(
            **contact_details,
            contact_id=contact_id,
        )

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Retrieve contact details, based on flexible filtering for first name, surname, email address, WhatsApp number, phone number, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the contacts.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of contacts to retrieve.

        Returns:
            List[Dict[str, str]]: A list of contacts.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]
