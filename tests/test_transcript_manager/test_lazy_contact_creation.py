import pytest
from datetime import datetime, UTC


from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_lazy_contact_creation_creates_contacts_and_logs_message():
    """Ensure that providing Contact objects lazily creates contacts and logs message with correct IDs."""
    tm = TranscriptManager()

    # Baseline contact count
    initial_contacts = tm._contact_manager._filter_contacts()
    initial_count = len(initial_contacts)

    # Define new (non-persisted) contacts
    sender = Contact(first_name="Lazy", email_address="lazy@example.com")
    receiver = Contact(phone_number="+15551234567")

    unique_content = "Lazy contact creation message – unique content"

    msg_dict = {
        "medium": "email",
        "sender_id": sender,
        "receiver_ids": [receiver],
        "timestamp": datetime.now(UTC),
        "content": unique_content,
        "exchange_id": 98765,
    }

    # Log and flush
    tm.log_messages(msg_dict)
    tm.join_published()

    # New contacts must have been created
    all_contacts = tm._contact_manager._filter_contacts()
    assert (
        len(all_contacts) == initial_count + 2
    ), "Exactly two new contacts should be created."

    # Fetch the stored message
    stored_msgs = tm._filter_messages(filter="exchange_id == 98765")
    assert len(stored_msgs) == 1, "Message should have been logged exactly once."

    stored = stored_msgs[0]
    # sender_id / receiver_ids should now be concrete ints coming from backend
    assert isinstance(stored.sender_id, int) and stored.sender_id != -1
    assert all(isinstance(cid, int) and cid != -1 for cid in stored.receiver_ids)
    assert stored.content == unique_content

    # Verify the newly created contact details match the originals
    sender_contact = tm._contact_manager._filter_contacts(
        filter=f"contact_id == {stored.sender_id}",
    )[0]
    receiver_contact = tm._contact_manager._filter_contacts(
        filter=f"contact_id == {stored.receiver_ids[0]}",
    )[0]

    assert sender_contact.email_address == "lazy@example.com"
    assert receiver_contact.phone_number == "+15551234567"
