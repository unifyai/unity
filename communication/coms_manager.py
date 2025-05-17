import threading
import functools
from typing import Optional
from datetime import datetime, UTC
from communication.sys_msgs import COMS_MANAGER
from communication.types.message import Message
from communication.types.contact import Contact
from communication.transcript_manager.transcript_manager import TranscriptManager
from knowledge_manager.knowledge_manager import KnowledgeManager
from common.llm_helpers import tool_use_loop

import unify


class ComsManager(threading.Thread):

    def __init__(self, daemon: bool = True):
        """
        Responsible for *sending*, *summarizing* and *searching through* all communcation channels exposed to the assistant.
        """
        super().__init__(daemon=daemon)

        # Managers
        self._transcript_manager = TranscriptManager()
        self._knowledge_manager = KnowledgeManager()

        # Tools
        search_tools = {
            self._transcript_manager.ask.__name__: self._transcript_manager.ask,
            self._knowledge_manager.store.__name__: self._transcript_manager.store,
            self._knowledge_manager.retrieve.__name__: self._knowledge_manager.retrieve,
        }
        outbound_text_tools = {
            self._send_email.__name__: self._send_email,
            self._send_sms_message.__name__: self._send_sms_message,
            self._send_whatsapp_message.__name__: self._send_whatsapp_message,
        }
        outbound_call_tools = {
            self._make_phone_call.__name__: self._make_phone_call,
            self._make_whatsapp_call.__name__: self._make_whatsapp_call,
        }
        self._tools = {
            **search_tools,
            **outbound_text_tools,
            **outbound_call_tools,
        }

        # Agent
        self._client = unify.Unify("o4-mini@openai", cache=True)
        # ToDo: implement this system message
        self._client.set_system_message(COMS_MANAGER)

        # Moving window context: initially (up to) the most recent 50 messages across ALL contacts + ALL mediums
        self._client.set_messages(
            self._transcript_manager._search_messages(limit=50),
        )

    # Helpers #
    # --------#

    def _shift_context_window(self, fn: callable) -> callable:

        @functools.wraps
        def _wrapped(*a, **kw):
            ret = fn(*a, **kw)
            # ToDo: make this message state resetting more efficient. However, we can't simply apply a moving window
            #  on the self._client.messages because this would include all of the tool call decision making,
            #  which we don't want to be preserved in the ongoing message history (otherwise it would become very bloated)
            self._client.set_messages(
                self._transcript_manager._search_messages(limit=50),
            )
            return ret

        return _wrapped

    def _maybe_create_new_contact(
        self,
        *,
        medium: str,
        contact_detail: str,
        **other_details,
    ) -> Contact:
        search_res = self._transcript_manager._search_contacts(
            f"{medium} == '{contact_detail}'",
        )
        if not search_res:
            return self._transcript_manager.create_contact(
                **{f"{medium}": contact_detail},
                **other_details,
            )
        assert (
            len(search_res) == 1
        ), f"Only **one** contact should have {medium}: {contact_detail}"
        return search_res[0]

    def _maybe_get_first_name(self, *, contact_detail: str, contact_id: int) -> str:
        contacts = self._transcript_manager._search_contacts(
            f"contact_id == {contact_id}",
        )
        assert (
            len(contacts) == 1
        ), f"Only **one** contact should have contact_id: {contact_id}"
        contact = contacts[0]
        if contact.first_name:
            return contact.first_name
        return contact_detail

    # Text #
    # -----#

    # Inbound

    @_shift_context_window
    def _receive_email(
        self,
        *,
        sender: str,
        content: str,
        return_reasoning_steps: bool = False,
    ) -> int:
        """
        Receive an email from the given sender with the given content, and perform the appropriate action.

        Args:
            sender (str): The email address of the sender.
            content (str): The content of the email.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.
        Returns:
            str: A summary of the action (or inaction) taken in response to the inbound email.
        """
        contact = self._maybe_create_new_contact(medium="email", contact_detail=sender)
        sender = self._maybe_get_first_name(
            contact_id=contact.contact_id,
            contact_detail=sender,
        )
        ret = tool_use_loop(self._client, content, self._tools, name=sender)
        if return_reasoning_steps:
            return ret, self._client.messages
        return ret

    @_shift_context_window
    def _receive_sms_message(
        self,
        *,
        sender: str,
        content: str,
        return_reasoning_steps: bool = False,
    ) -> int:
        """
        Receive an sms message from the given sender with the given content, and perform the appropriate action.

        Args:
            sender (str): The phone number of the sender.
            content (str): The content of the sms message.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.
        Returns:
            str: A summary of the action (or inaction) taken in response to the inbound sms message.
        """
        contact = self._maybe_create_new_contact(
            medium="sms_message",
            contact_detail=sender,
        )
        sender = self._maybe_get_first_name(
            contact_id=contact.contact_id,
            contact_detail=sender,
        )
        ret = tool_use_loop(self._client, content, self._tools, name=sender)
        if return_reasoning_steps:
            return ret, self._client.messages
        return ret

    @_shift_context_window
    def _receive_whatsapp_message(
        self,
        *,
        sender: str,
        content: str,
        return_reasoning_steps: bool = False,
    ) -> int:
        """
        Receive an WhatsApp message from the given sender with the given content, and perform the appropriate action.

        Args:
            sender (str): The phone number of the sender.
            content (str): The content of the WhatsApp message.
            return_reasoning_steps (bool): Whether to return the reasoning steps for the update request.
        Returns:
            str: A summary of the action (or inaction) taken in response to the inbound WhatsApp message.
        """
        contact = self._maybe_create_new_contact(
            medium="whatsapp_message",
            contact_detail=sender,
        )
        sender = self._maybe_get_first_name(
            contact_id=contact.contact_id,
            contact_detail=sender,
        )
        ret = tool_use_loop(self._client, content, self._tools, name=sender)
        if return_reasoning_steps:
            return ret, self._client.messages
        return ret

    # Outbound

    def _send_email(
        self,
        *,
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
    ) -> int:
        """
        Send an email to the given receiver with the given content.

        Args:
            receiver (str): The email address of the receiver.
            content (str): The content of the email.
            exchange_id (int): The email thread to respond to, otherwise a new thread is started.
        Returns:
            str: The id of the email.
        """
        contact = self._maybe_create_new_contact(
            medium="email",
            contact_detail=receiver,
        )
        receiver = self._maybe_get_first_name(
            contact_id=contact.contact_id,
            contact_detail=receiver,
        )
        self._transcript_manager.log_messages(
            [
                Message(
                    medium="email",
                    sender_id=0,  # self is always 0
                    receiver_id=contact.contact_id,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=content,
                    exchange_id=exchange_id,
                ),
            ],
        )
        raise NotImplemented

    def _send_sms_message(
        self,
        *,
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
    ):
        """
        Send a text message to the given receiver with the given content.

        Args:
            receiver (str): The phone number of the receiver.
            content (str): The content of the text message.
            exchange_id (int): The sms message thread to respond to, otherwise a new chat is started.
        Returns:
            str: The id of the text message.
        """
        raise NotImplemented

    def _send_whatsapp_message(
        self,
        *,
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
    ):
        """
        Send a WhatsApp message to the given receiver with the given content.

        Args:
            receiver (str): The WhatsApp number of the receiver.
            content (str): The content of the WhatsApp message.
            exchange_id (int): The WhatsApp group thread to respond to, otherwise a new chat is started.
        Returns:
            str: The id of the WhatsApp message.
        """
        raise NotImplemented

    # Voice Calls #
    # ------------#

    # Inbound

    def _receive_phone_call(
        self,
        *,
        caller: str,
    ):
        """
        Accept phone call from the caller.

        Args:
            caller (str): The phone number of the caller.

        Returns:
            The live handle of the phone call, for another agent to take control of.
        """
        raise NotImplemented

    def _receive_whatsapp_call(
        self,
        *,
        caller: str,
    ):
        """
        Accept WhatsApp call from the caller.

        Args:
            caller (str): The WhatsApp number of the caller.

        Returns:
            The live handle of the WhatsApp call, for another agent to take control of.
        """
        raise NotImplemented

    # Outbound

    def _make_phone_call(
        self,
        *,
        receiver: str,
    ):
        """
        Phone call the given receiver at the given date and time.

        Args:
            receiver (str): The phone number of the receiver.

        Returns:
            The live handle of the phone call, for another agent to take control of.
        """
        raise NotImplemented

    def _make_whatsapp_call(
        self,
        *,
        receiver: str,
    ):
        """
        WhatsApp call the given receiver at the given date and time.

        Args:
            receiver (str): The WhatsApp number of the receiver.

        Returns:
            The live handle of the WhatsApp call, for another agent to take control of.
        """
        raise NotImplemented
