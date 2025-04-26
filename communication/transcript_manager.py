import threading
from typing import List, Dict, Any, Optional


class TranscriptManager(threading.Thread):

    def __init__(self):
        """
        Responsible for *searching through* the full transcripts across all communcation channels exposed to the assistant.
        """
        raise NotImplemented

    # Public #
    # -------#

    # English-Text Question

    def ask(text: str) -> Any:
        """
        Ask any question as a text command, and use the tools available (the private methods of this class) to perform the action.

        Args:
            text (str): The text-based question to answer.

        Returns:
            Any: The answer to the question.
        """
        raise NotImplemented

    # Summarize Exchange(s)

    def summarize(exchange_ids: int, guidance: Optional[str] = None) -> str:
        """
        Summarize the email thread, phone call, or a time-clustered text exchange, save the summary in the backend, and also oreturn it.

        Args:
            exchange_ids (int): The ids of the exchanges to summarize.
            guidance (Optional[str]): Optional guidance for the summarization.

        Returns:
            str: The summary of the exchanges.
        """
        raise NotImplemented

    # Private #
    # --------#

    def _get_messages(
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

    def _get_summaries(
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
