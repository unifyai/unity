import random
from datetime import datetime

from communication.transcript_manager.transcript_manager import TranscriptManager
from communication.types.message import VALID_MEDIA, Message
from tests.helpers import _handle_project


@_handle_project
def test_transcript_embedding_semantic_search():
    # Initialize and start the TranscriptManager thread
    tm = TranscriptManager()
    tm.start()

    # Insert semantically related messages without the keywords 'banking' or 'budgetting'
    msg1 = Message(
        medium=random.choice(VALID_MEDIA),
        sender_id=0,
        receiver_id=1,
        timestamp=datetime.now().isoformat(),
        content="I need to manage my finances, savings and expenses.",
        exchange_id=0,
    )
    msg2 = Message(
        medium=random.choice(VALID_MEDIA),
        sender_id=0,
        receiver_id=1,
        timestamp=datetime.now().isoformat(),
        content="Planning for future expenses and money management is important.",
        exchange_id=1,
    )
    msg3 = Message(
        medium=random.choice(VALID_MEDIA),
        sender_id=0,
        receiver_id=1,
        timestamp=datetime.now().isoformat(),
        content="I'm thinking about my retirement plan and investments.",
        exchange_id=2,
    )
    tm.log_messages([msg1, msg2, msg3])

    # Ensure that a lexical search for the word 'banking' returns no results
    lexical_results = tm._search_messages(filter="'banking' in content")
    assert lexical_results == []

    # Use semantic search to find the nearest messages to the query
    nearest = tm._nearest_messages(text="banking and budgeting", k=2)

    # Verify the result length and type
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # Verify that the messages are returned in ascending order of distance
    assert nearest[0].content == msg2.content
    assert nearest[1].content == msg1.content

    # Test k-limit behavior
    all_nearest = tm._nearest_messages(text="banking and budgeting", k=10)
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted
