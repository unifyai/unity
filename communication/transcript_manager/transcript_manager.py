import json
import threading
from typing import List, Dict, Any, Optional, Union

import unify
from communication.message import Message
from communication.summary import Summary
from communication.transcript_manager.sys_msgs import SUMMARIZE


class TranscriptManager(threading.Thread):
    """
    Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
    """

    # Public #
    # -------#

    # English-Text Question

    def ask(self, text: str) -> Any:
        """
        Ask any question as a text command, and use the tools available (the private methods of this class) to perform the action.

        Args:
            text (str): The text-based question to answer.

        Returns:
            Any: The answer to the question.
        """
        raise NotImplemented

    # Summarize Exchange(s)

    def summarize(
        self,
        exchange_ids: Union[int, List[int]],
        guidance: Optional[str] = None,
    ) -> str:
        """
        Summarize the email thread, phone call, or a time-clustered text exchange, save the summary in the backend, and also oreturn it.

        Args:
            exchange_ids (int): The ids of the exchanges to summarize.
            guidance (Optional[str]): Optional guidance for the summarization.

        Returns:
            str: The summary of the exchanges.
        """
        if not isinstance(exchange_ids, list):
            exchange_ids = [exchange_ids]
        client = unify.Unify("gpt-4o@openai")
        client.set_system_message(
            SUMMARIZE.replace("{guidance}", f"\n{guidance}\n" if guidance else ""),
        )
        msgs = self._get_messages(filter=f"exchange_id in {exchange_ids}")
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

    # Private #
    # --------#

    def _get_messages(
        self,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
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

    def _get_summaries(
        self,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
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
