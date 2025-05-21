import json
from typing import List, Dict, Any, Optional, Union

import unify
from ...common.embed_utils import EMBED_MODEL, ensure_vector_column
from ...communication.types.contact import Contact
from ...communication.types.message import Message
from ..types.message_exchange_summary import MessageExchangeSummary
from ...common.llm_helpers import start_async_tool_use_loop, AsyncToolLoopHandle
from ...events.event_bus import EventBus, Event


class TranscriptManager:

    # Vector embedding column names
    _VEC_MSG = "content_emb"
    _VEC_SUM = "summary_emb"

    def __init__(self, event_bus: EventBus, *, traced: bool = True) -> None:
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        self._event_bus = event_bus

        self._tools = {
            self.summarize.__name__: self.summarize,
            self._search_contacts.__name__: self._search_contacts,
            self._search_messages.__name__: self._search_messages,
            self._search_summaries.__name__: self._search_summaries,
            self._nearest_messages.__name__: self._nearest_messages,
        }

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        self._contacts_ctx = f"{read_ctx}/Contacts" if read_ctx else "Contacts"
        self._transcripts_ctx = self._event_bus.ctxs["Messages"]
        self._summaries_ctx = self._event_bus.ctxs["MessageExchangeSummaries"]
        if self._contacts_ctx not in unify.get_contexts():
            unify.create_context(self._contacts_ctx)

        # Add tracing
        if traced:
            self = unify.traced(self)

    # Public #
    # -------#

    # English-Text Question

    def ask(
        self, text: str, *, return_reasoning_steps: bool = False
    ) -> "AsyncToolLoopHandle":
        """
        Ask any question as a text command, and use the tools available (the private methods of this class) to perform the action.

        Args:
            text (str): The text-based question to answer.
            return_reasoning_steps (bool): Whether to return the reasoning steps along with the answer.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation that supports:
                - await handle.result(): Get the final answer when ready
                - await handle.interject(message): Add a new user message mid-conversation
                - handle.stop(): Gracefully terminate the conversation

        Usage:
            # Synchronous call that returns a handle immediately:
            handle = transcript_manager.ask("Find recent emails from John")

            # To get the final answer (must be awaited):
            answer = await handle.result()

            # If return_reasoning_steps=True:
            handle = transcript_manager.ask("Find emails from John", return_reasoning_steps=True)
            answer, reasoning_steps = await handle.result()

            # To add clarification mid-conversation:
            await handle.interject("I meant John Smith specifically")

            # To stop the conversation early:
            handle.stop()
        """
        from unity.communication.transcript_manager.sys_msgs import ANSWER

        client = unify.AsyncUnify("o4-mini@openai", cache=True)
        client.set_system_message(ANSWER)
        handle = start_async_tool_use_loop(client, text, self._tools)
        if return_reasoning_steps:
            # Wrap the handle.result() to return both answer and reasoning steps
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result

        return handle

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
        from unity.communication.transcript_manager.sys_msgs import SUMMARIZE

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
        latest_timestamp = max([msg.timestamp for msg in msgs]).isoformat()
        summary = client.generate(json.dumps(exchanges, indent=4))
        self._event_bus.publish(
            Event(
                type="message_exchange_summary",
                timestamp=latest_timestamp,
                payload=summary,
            )
        )
        return summary

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
        if not unify.get_logs(context=self._contacts_ctx):
            return unify.log(
                context=self._contacts_ctx,
                **contact_details,
                contact_id=0,
                new=True,
            ).id

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._contacts_ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # ToDo: filter only for contact_id once supported in the Python utility function
        logs = unify.get_logs(
            context=self._contacts_ctx,
        )
        largest_id = max([lg.entries["contact_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new contact
        return unify.log(
            context=self._contacts_ctx,
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
                context=self._contacts_ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # get log id
        logs = unify.get_logs(
            context=self._contacts_ctx,
            filter=f"contact_id == {contact_id}",
        )
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
        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._transcripts_ctx,
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
    ) -> List[MessageExchangeSummary]:
        """
        Find summaries semantically similar to the provided text using vector embeddings.

        Args:
            text (str): The text to find similar summaries to.
            k (int): The number of similar summaries to return.

        Returns:
            List[MessageExchangeSummary]: A list of summaries semantically similar to the provided text.
        """

        ensure_vector_column(self._transcripts_ctx, self._VEC_MSG, "content")
        logs = unify.get_logs(
            context=self._summaries_ctx,
            sorting={
                f"cosine({self._VEC_SUM}, embed('{text}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
        )
        return [MessageExchangeSummary(**lg.entries) for lg in logs]

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
            context=self._contacts_ctx,
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
            context=self._transcripts_ctx,
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
            context=self._summaries_ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [MessageExchangeSummary(**lg.entries) for lg in logs]
