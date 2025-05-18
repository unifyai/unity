import json
import threading
from typing import List, Dict, Any, Optional, Union

import unify
from common.embed_utils import EMBED_MODEL, ensure_vector_column
from communication.types.contact import Contact
from communication.types.message import Message
from communication.types.summary import Summary
from common.llm_helpers import async_tool_use_loop


class TranscriptManager(threading.Thread):

    # Vector embedding column names
    _VEC_MSG = "content_emb"
    _VEC_SUM = "summary_emb"

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        super().__init__(daemon=daemon)
        self._tools = {
            self.summarize.__name__: self.summarize,
            self._search_contacts.__name__: self._search_contacts,
            self._search_messages.__name__: self._search_messages,
            self._search_summaries.__name__: self._search_summaries,
            self._nearest_messages.__name__: self._nearest_messages,
        }

    # Public #
    # -------#

    # English-Text Question

    async def ask(self, text: str, *, return_reasoning_steps: bool = False) -> Any:
        """
        Ask any question as a text command, and use the tools available (the private methods of this class) to perform the action.

        Args:
            text (str): The text-based question to answer.

            return_reasoning_steps (bool): Whether to return the reasoning steps for the question.

        Returns:
            Any: The answer to the question.
        """
        from communication.transcript_manager.sys_msgs import ANSWER

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(ANSWER)
        ans = await async_tool_use_loop(client, text, self._tools)
        if return_reasoning_steps:
            return ans, client.messages
        return ans

    # Summarize Exchange(s)

    def summarize(
        self,
        *,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
    ) -> str:
        """
        Summarize the email thread, phone call, or a time-clustered text exchange, save the summary in the backend, and also return it.

        Args:
            exchange_ids (int): The ids of the exchanges to summarize.
            guidance (Optional[str]): Optional guidance for the summarization.

        Returns:
            str: The summary of the exchanges.
        """
        from communication.transcript_manager.sys_msgs import SUMMARIZE

        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]
        client = unify.Unify("o4-mini@openai", cache=True)
        client.set_system_message(
            SUMMARIZE.replace("{guidance}", f"\n{guidance}\n" if guidance else ""),
        )
        msgs = self._search_messages(filter=f"exchange_id in {exchange_ids}")
        exchanges = {
            id: [msg.content for msg in msgs if msg.exchange_id == id]
            for id in exchange_ids
        }
        summary = client.generate(json.dumps(exchanges, indent=4))
        unify.log(
            context="TranscriptSummaries",
            exchange_ids=exchange_ids,
            summary=summary,
        )
        return summary

    def log_messages(self, messages: List[Message]):
        """
        Log messages onto the platform.
        """
        return unify.create_logs(
            context="Transcripts",
            entries=[msg.model_dump() for msg in messages],
        )

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

        # If it's the fist contact, create immediately
        if "Contacts" not in unify.get_contexts():
            return unify.log(
                context="Contacts",
                **contact_details,
                contact_id=0,
                new=True,
            ).id

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context="Contacts",
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # ToDo: filter only for contact_id once supported in the Python utility function
        logs = unify.get_logs(
            context="Contacts",
        )
        largest_id = max([lg.entries["contact_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new contact
        return unify.log(
            context="Contacts",
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
                context="Contacts",
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # get log id
        logs = unify.get_logs(context="Contacts", filter=f"contact_id == {contact_id}")
        assert len(logs) == 1
        log: unify.Log = logs[0]
        log.update_entries(
            **contact_details,
            contact_id=contact_id,
        )

    # Private #
    # --------#
    def _nearest_messages(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[Message]:
        """
        Find messages semantically similar to the provided text using vector embeddings.

        Args:
            text (str): The text to find similar messages to.
            k (int): The number of similar messages to return.

        Returns:
            List[Message]: A list of messages semantically similar to the provided text.
        """
        ensure_vector_column("Transcripts", self._VEC_MSG, "content")
        logs = unify.get_logs(
            context="Transcripts",
            sorting={
                f"cosine({self._VEC_MSG}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [Message(**lg.entries) for lg in logs]

    def _nearest_summaries(
        self,
        *,
        text: str,
        k: int = 10,
    ) -> List[Summary]:
        """
        Find summaries semantically similar to the provided text using vector embeddings.

        Args:
            text (str): The text to find similar summaries to.
            k (int): The number of similar summaries to return.

        Returns:
            List[Summary]: A list of summaries semantically similar to the provided text.
        """

        ensure_vector_column("Transcripts", self._VEC_MSG, "content")
        logs = unify.get_logs(
            context="TranscriptSummaries",
            sorting={
                f"cosine({self._VEC_SUM}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [Summary(**lg.entries) for lg in logs]

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
            context="Contacts",
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]

    def _search_messages(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
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
        logs = unify.get_logs(
            context="Transcripts",
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Message(**lg.entries) for lg in logs]

    def _search_summaries(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
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
        logs = unify.get_logs(
            context="TranscriptSummaries",
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Summary(**lg.entries) for lg in logs]
