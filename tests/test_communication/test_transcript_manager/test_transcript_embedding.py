import pytest

from communication.transcript_manager.transcript_manager import TranscriptManager
from communication.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@_handle_project
def test_transcript_embedding_semantic_search():
    """
    Test the transcript manager's ability to perform semantic search via nearest message retrieval.
    """
    # Create the TranscriptManager instance
    tm = TranscriptManager(name="EmbeddingTest")

    # Create a few test messages
    msg1 = Message(
        role=Role.USER,
        content="Can you help me with my banking questions? I'm looking to set up a new account.",
    )
    msg2 = Message(
        role=Role.ASSISTANT,
        content="I'd be happy to help with your banking needs! What type of account would you like to set up? Checking, savings, or investment?",
    )
    msg3 = Message(
        role=Role.USER,
        content="I'm interested in learning about Python programming, especially data science applications.",
    )

    # Add them to the transcript
    tm.add_message(msg1)
    tm.add_message(msg2)
    tm.add_message(msg3)

    # Give time for embedding to be created
    import time

    time.sleep(1)  # Allow time for the embeddings to be calculated asynchronously

    # Use semantic search to find the nearest messages to the query
    nearest = tm._nearest_messages(text="banking and budgeting", k=2)

    # Validate results
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # Messages should be sorted by relevance
    assert nearest[0].content == msg2.content
    assert nearest[1].content == msg1.content

    # Test with a higher k value to get all messages
    all_nearest = tm._nearest_messages(text="banking and budgeting", k=10)
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted
