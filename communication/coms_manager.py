import threading
from datetime import datetime
from typing import List, Dict, Optional
from communication.transcript_manager.transcript_manager import TranscriptManager


class ComsManager(threading.Thread):

    def __init__(self):
        """
        Responsible for *sending*, *summarizing* and *searching through* all communcation channels exposed to the assistant.
        """
        self._transcript_manager = TranscriptManager()
        raise NotImplemented

    # Private #
    # --------#

    # Search Tools

    def _probe_transcripts(
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Retrieve messages from the transcript history, based on flexible filtering for a specific sender, group of senders, receiver, group of receivers, medium, set of mediums, timestamp range, message length, messages containing a phrase, not containing a phrase, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the messages.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of messages to retrieve.

        Returns:
            List[Dict[str, str]]: A list of messages.
        """
        raise NotImplemented

    def _probe_knowledge(
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Retrieve summaries from the transcript history, based on flexible filtering for a specific exchange id, group of exchange ids, medium, set of mediums, timestamp range, summary length, summaries containing a phrase, not containing a phrase, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the summaries.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of summaries to retrieve.

        Returns:
            List[Dict[str, str]]: A list of exchange summaries.
        """
        raise NotImplemented

    # Outbound Text

    def _send_email(
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
        scheduled: Optional[datetime] = None,
    ) -> int:
        """
        Send an email to the given receiver with the given content.

        Args:
            receiver (str): The email address of the receiver.
            content (str): The content of the email.
            exchange_id (Optional[int]): The id of the email thread to respond to.
            scheduled (Optional[datetime]): The date and time to schedule the email to be sent.
        Returns:
            str: The id of the email.
        """
        raise NotImplemented

    def _send_text(
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
        scheduled: Optional[datetime] = None,
    ):
        """
        Send a text message to the given receiver with the given content.

        Args:
            receiver (str): The phone number of the receiver.
            content (str): The content of the text message.
            exchange_id (Optional[int]): The id of the text exchange to respond to. Relevant if the user is in multiple group chats, for example.
            scheduled (Optional[datetime]): The date and time to schedule the text message to be sent.

        Returns:
            str: The id of the text message.
        """
        raise NotImplemented

    def _send_whatsapp_message(
        receiver: str,
        content: str,
        exchange_id: Optional[int] = None,
        scheduled: Optional[datetime] = None,
    ):
        """
        Send a WhatsApp message to the given receiver with the given content.

        Args:
            receiver (str): The WhatsApp number of the receiver.
            content (str): The content of the WhatsApp message.
            exchange_id (Optional[int]): The id of the exchange to respond to. Relevant if the user is in multiple group chats, for example.
            scheduled (Optional[datetime]): The date and time to schedule the WhatsApp message to be sent.

        Returns:
            str: The id of the WhatsApp message.
        """
        raise NotImplemented

    # Outbound Calls

    def _make_call(receiver: str, scheduled: Optional[datetime] = None):
        """
        Phone call the given receiver at the given date and time.

        Args:
            receiver (str): The phone number of the receiver.
            scheduled (Optional[datetime]): The date and time to schedule the phone call to be made.

        Returns:
            The live handle of the phone call, for another agent to take control of.
        """
        raise NotImplemented

    def _make_whatsapp_call(receiver: str, scheduled: Optional[datetime] = None):
        """
        WhatsApp call the given receiver at the given date and time.

        Args:
            receiver (str): The WhatsApp number of the receiver.
            scheduled (Optional[datetime]): The date and time to schedule the WhatsApp call to be made.

        Returns:
            The live handle of the WhatsApp call, for another agent to take control of.
        """
        raise NotImplemented

    def _receive_call(receiver: str):
        """
        Accept phone call from the receiver..

        Args:
            receiver (str): The phone number of the calling receiver.

        Returns:
            The live handle of the phone call, for another agent to take control of.
        """
        raise NotImplemented

    def _receive_whatsapp_call(receiver: str):
        """
        Accept WhatsApp call from the receiver..

        Args:
            receiver (str): The WhatsApp number of the calling receiver.

        Returns:
            The live handle of the WhatsApp call, for another agent to take control of.
        """
        raise NotImplemented
