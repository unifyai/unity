"""
Foreign Key Integration Tests (Cross-Manager)

Coverage
========
Phase 4: Integration testing across all managers with FK relationships

✓ Contact deletion cascade to transcripts
✓ Image deletion cascade to transcripts and guidance
✓ Function-Guidance bidirectional FK consistency
✓ Function deletion cascade to tasks and guidance
✓ Complex multi-manager workflows
✓ Circular reference handling (Functions ↔ Guidance)
✓ Bulk operations with FK constraints
✓ Absolute cross-root FK references (overlay → Hive-shared parent)
"""

from __future__ import annotations

import pytest
import unify
from datetime import datetime
from tests.helpers import _handle_project
from unity.common.context_registry import ContextRegistry, TableContext
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager

# --------------------------------------------------------------------------- #
#  Integration: Contact Deletion Cascade                                      #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_delete_contact_transcripts_fk():
    """
    Test contact deletion effects on transcripts:
    - sender_id: SET NULL (message survives with null sender)
    - receiver_ids: SET NULL (contact removed from arrays)
    """
    cm = ContactManager()
    tm = TranscriptManager()

    # Create 3 contacts
    cm._create_contact(
        first_name="Alice",
        email_address="alice@test.com",
        phone_number="1111111111",
    )
    cm._create_contact(
        first_name="Bob",
        email_address="bob@test.com",
        phone_number="2222222222",
    )
    cm._create_contact(
        first_name="Carol",
        email_address="carol@test.com",
        phone_number="3333333333",
    )

    contacts = unify.get_logs(context=cm._ctx, from_fields=["contact_id", "first_name"])
    contact_map = {
        c.entries["first_name"]: int(c.entries["contact_id"]) for c in contacts
    }

    # Alice sends to Bob
    tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": contact_map["Alice"],
            "receiver_ids": [contact_map["Bob"]],
            "content": "Alice to Bob",
            "timestamp": datetime.now(),
        },
    )

    # Bob sends to Alice and Carol
    tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": contact_map["Bob"],
            "receiver_ids": [contact_map["Alice"], contact_map["Carol"]],
            "content": "Bob to Alice and Carol",
            "timestamp": datetime.now(),
        },
    )

    # Carol sends to Alice and Bob
    tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": contact_map["Carol"],
            "receiver_ids": [contact_map["Alice"], contact_map["Bob"]],
            "content": "Carol to Alice and Bob",
            "timestamp": datetime.now(),
        },
    )

    # Verify 3 messages
    messages = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["message_id", "sender_id", "receiver_ids"],
    )
    assert len(messages) == 3

    # Delete Alice (SET NULL on sender_id, SET NULL on receiver_ids)
    cm._delete_contact(contact_id=contact_map["Alice"])

    # Verify all 3 messages still exist (SET NULL on sender_id preserves messages)
    # Alice's message survives with null sender_id
    # Bob's and Carol's messages survive with Alice replaced by None in receiver_ids (in-place)
    messages_after = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["message_id", "sender_id", "receiver_ids", "content"],
    )
    assert len(messages_after) == 3  # All messages survive

    # Alice's message should have null sender_id but preserved content
    alice_msg = next(
        (m for m in messages_after if m.entries["content"] == "Alice to Bob"),
        None,
    )
    assert alice_msg is not None
    assert alice_msg.entries.get("sender_id") is None  # SET NULL
    assert contact_map["Bob"] in alice_msg.entries["receiver_ids"]

    # Bob's message: Alice replaced with None in receiver_ids (in-place SET NULL)
    bob_msg = next(
        m for m in messages_after if m.entries["sender_id"] == contact_map["Bob"]
    )
    bob_receivers = bob_msg.entries["receiver_ids"]
    assert len(bob_receivers) == 2  # Array length unchanged
    assert None in bob_receivers  # Alice replaced with None
    assert contact_map["Carol"] in bob_receivers  # Carol preserved

    # Carol's message: Alice replaced with None in receiver_ids (in-place SET NULL)
    carol_msg = next(
        m for m in messages_after if m.entries["sender_id"] == contact_map["Carol"]
    )
    carol_receivers = carol_msg.entries["receiver_ids"]
    assert len(carol_receivers) == 2  # Array length unchanged
    assert None in carol_receivers  # Alice replaced with None
    assert contact_map["Bob"] in carol_receivers  # Bob preserved


# --------------------------------------------------------------------------- #
#  Integration: Image Deletion SET NULL                                        #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_delete_image_nullifies_refs():
    """
    Test that deleting an image sets image_id to None (in-place) in both
    transcript messages and guidance entries that reference it.

    FK Policies:
    - Transcripts.images[*].raw_image_ref.image_id → Images.image_id: SET NULL
    - Guidance.images[*].raw_image_ref.image_id → Images.image_id: SET NULL
    """
    from unity.contact_manager.contact_manager import ContactManager
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.guidance_manager.guidance_manager import GuidanceManager
    from unity.image_manager.image_manager import ImageManager

    cm = ContactManager()
    tm = TranscriptManager()
    gm = GuidanceManager()
    im = ImageManager()

    # Helper for creating valid test images
    def _make_test_image_b64(
        size: int = 32,
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> str:
        from unity.image_manager.utils import make_solid_png_base64

        return make_solid_png_base64(size, size, color)

    # Create contacts
    cm._create_contact(
        first_name="Alice",
        email_address="alice@test.com",
        phone_number="1111111111",
    )
    cm._create_contact(
        first_name="Bob",
        email_address="bob@test.com",
        phone_number="2222222222",
    )
    contacts = unify.get_logs(context=cm._ctx, from_fields=["contact_id", "first_name"])
    alice_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Alice"
    )
    bob_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Bob"
    )

    # Store shared image
    img_ids = im.add_images(
        [{"data": _make_test_image_b64(color=(255, 0, 0)), "caption": "Shared"}],
        synchronous=True,
    )
    assert img_ids[0] is not None, "Image creation failed"
    shared_img_id = img_ids[0]

    # Use image in transcript
    tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": alice_id,
            "receiver_ids": [bob_id],
            "content": "Check this screenshot",
            "timestamp": datetime.now(),
            "images": [
                {
                    "raw_image_ref": {"image_id": shared_img_id},
                    "annotation": "Screenshot",
                },
            ],
        },
    )

    # Use same image in guidance
    gm.add_guidance(
        title="Setup Guide",
        content="Follow these steps",
        images=[
            {
                "raw_image_ref": {"image_id": shared_img_id},
                "annotation": "Setup screenshot",
            },
        ],
    )

    # Verify image in both
    messages = unify.get_logs(context=tm._transcripts_ctx, from_fields=["images"])
    assert (
        messages[0].entries["images"][0]["raw_image_ref"]["image_id"] == shared_img_id
    )

    guidance = unify.get_logs(context=gm._ctx, from_fields=["images"])
    assert (
        guidance[0].entries["images"][0]["raw_image_ref"]["image_id"] == shared_img_id
    )

    # Delete the shared image
    img_logs = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {shared_img_id}",
        return_ids_only=True,
    )
    assert img_logs, "Image not found"
    unify.delete_logs(context=im._ctx, logs=img_logs[0])

    # Verify image_id replaced with None in both transcript and guidance (in-place SET NULL)
    messages_after = unify.get_logs(context=tm._transcripts_ctx, from_fields=["images"])
    msg_images = messages_after[0].entries.get("images", [])
    assert len(msg_images) == 1  # Array length unchanged
    assert msg_images[0]["raw_image_ref"]["image_id"] is None  # image_id set to None
    assert msg_images[0]["annotation"] == "Screenshot"  # Annotation preserved

    guidance_after = unify.get_logs(context=gm._ctx, from_fields=["images"])
    guid_images = guidance_after[0].entries.get("images", [])
    assert len(guid_images) == 1  # Array length unchanged
    assert guid_images[0]["raw_image_ref"]["image_id"] is None  # image_id set to None
    assert guid_images[0]["annotation"] == "Setup screenshot"  # Annotation preserved


# --------------------------------------------------------------------------- #
#  Integration: Function-Guidance Bidirectional FK (CASCADE)                   #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_function_guidance_bidirectional_cascade():
    """
    Test bidirectional CASCADE FK relationship between Functions and Guidance.

    FK Policies:
    - Functions.guidance_ids[*] → Guidance.guidance_id: CASCADE (remove from array)
    - Guidance.function_ids[*] → Functions.function_id: CASCADE (remove from array)
    """
    from unity.function_manager.function_manager import FunctionManager
    from unity.guidance_manager.guidance_manager import GuidanceManager

    fm = FunctionManager()
    gm = GuidanceManager()

    # Create guidance entries first
    gm.add_guidance(title="Setup Guide", content="Setup instructions")
    gm.add_guidance(title="Usage Guide", content="Usage instructions")

    guidance_list = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "title"],
    )
    g_map = {g.entries["title"]: int(g.entries["guidance_id"]) for g in guidance_list}

    # Create function
    src = "def setup_system():\n    return 'setup'\n"
    fm.add_functions(implementations=src)

    # Get function ID
    func_logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter="name == 'setup_system'",
        return_ids_only=True,
    )
    assert func_logs, "Function not created"

    # Update function to reference both guidance entries
    unify.update_logs(
        context=fm._compositional_ctx,
        logs=func_logs[0],
        entries={"guidance_ids": [g_map["Setup Guide"], g_map["Usage Guide"]]},
        overwrite=True,
    )

    # Get function ID
    funcs = unify.get_logs(
        context=fm._compositional_ctx,
        from_fields=["function_id", "guidance_ids"],
    )
    func_id = int(funcs[0].entries["function_id"])

    # Update guidance entries to reference the function (using unify.update_logs)
    for title in ["Setup Guide", "Usage Guide"]:
        guid_logs = unify.get_logs(
            context=gm._ctx,
            filter=f"title == '{title}'",
            return_ids_only=True,
        )
        unify.update_logs(
            context=gm._ctx,
            logs=guid_logs[0],
            entries={"function_ids": [func_id]},
            overwrite=True,
        )

    # Verify bidirectional linkage
    # Function → Guidance
    func_data = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"function_id == {func_id}",
        from_fields=["guidance_ids"],
    )
    assert sorted(func_data[0].entries["guidance_ids"]) == sorted(
        [g_map["Setup Guide"], g_map["Usage Guide"]],
    )

    # Guidance → Function
    guidance_after = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "function_ids"],
    )
    for g in guidance_after:
        assert func_id in g.entries["function_ids"]

    # Delete one guidance entry (CASCADE should remove from function.guidance_ids array)
    gm.delete_guidance(guidance_id=g_map["Setup Guide"])

    # Verify removed from function's guidance_ids array (CASCADE behavior)
    func_after = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"function_id == {func_id}",
        from_fields=["guidance_ids"],
    )
    remaining = func_after[0].entries.get("guidance_ids", [])
    assert g_map["Usage Guide"] in remaining
    assert g_map["Setup Guide"] not in remaining  # CASCADE removed it

    # Delete function (CASCADE should remove from guidance.function_ids array)
    fm.delete_function(function_id=func_id)

    # Verify removed from guidance's function_ids array (CASCADE behavior)
    usage_guidance = unify.get_logs(
        context=gm._ctx,
        filter=f"guidance_id == {g_map['Usage Guide']}",
        from_fields=["function_ids"],
    )
    assert func_id not in usage_guidance[0].entries.get(
        "function_ids",
        [],
    )  # CASCADE removed it


# --------------------------------------------------------------------------- #
#  Integration: Function Deletion Effects on Tasks and Guidance               #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_delete_function_cascades_tasks_guidance():
    """
    Test that deleting a function:
    - Sets task.entrypoint to null (SET NULL)
    - Removes function_id from guidance.function_ids array (CASCADE)

    FK Policies:
    - Tasks.entrypoint → Functions.function_id: SET NULL
    - Guidance.function_ids[*] → Functions.function_id: CASCADE
    """
    from unity.function_manager.function_manager import FunctionManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.guidance_manager.guidance_manager import GuidanceManager

    fm = FunctionManager()
    ts = TaskScheduler()
    gm = GuidanceManager()

    # Create function
    src = "def worker():\n    return 'work'\n"
    fm.add_functions(implementations=src)

    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create task using this function
    result = ts._create_task(
        name="Work Task",
        description="Task using function",
        entrypoint=func_id,
    )
    task_id = result["details"]["task_id"]

    # Create guidance referencing this function
    gm.add_guidance(
        title="Function Guide",
        content="How to use worker()",
        function_ids=[func_id],
    )

    # Verify references
    task = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {task_id}",
        from_fields=["entrypoint"],
    )
    assert task[0].entries["entrypoint"] == func_id

    guidance = unify.get_logs(context=gm._ctx, from_fields=["function_ids"])
    assert func_id in guidance[0].entries["function_ids"]

    # Delete the function
    fm.delete_function(function_id=func_id)

    # Verify task survives with null entrypoint (SET NULL behavior)
    task_after = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {task_id}",
        from_fields=["task_id", "entrypoint"],
    )
    assert len(task_after) == 1  # Task still exists
    assert task_after[0].entries.get("entrypoint") is None  # SET NULL

    # Verify function_id removed from guidance array (CASCADE behavior)
    guidance_after = unify.get_logs(context=gm._ctx, from_fields=["function_ids"])
    assert func_id not in guidance_after[0].entries.get(
        "function_ids",
        [],
    )  # CASCADE removed it


# --------------------------------------------------------------------------- #
#  Integration: Complex Multi-Manager Workflow                                #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_complex_fk_workflow():
    """
    Test complex workflow involving all managers with FK relationships:
    - Contacts, Images, Functions, Guidance, Tasks, Transcripts

    FK Policies:
    - Transcripts.sender_id → Contacts: SET NULL
    - Transcripts.receiver_ids[*] → Contacts: SET NULL
    - Transcripts.images[*].raw_image_ref.image_id → Images: SET NULL
    - Guidance.images[*].raw_image_ref.image_id → Images: SET NULL
    - Guidance.function_ids[*] → Functions: CASCADE
    - Tasks.entrypoint → Functions: SET NULL
    """
    from unity.contact_manager.contact_manager import ContactManager
    from unity.image_manager.image_manager import ImageManager
    from unity.function_manager.function_manager import FunctionManager
    from unity.guidance_manager.guidance_manager import GuidanceManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.transcript_manager.transcript_manager import TranscriptManager

    cm = ContactManager()
    im = ImageManager()
    fm = FunctionManager()
    gm = GuidanceManager()
    ts = TaskScheduler()
    tm = TranscriptManager()

    # Helper for creating valid test images
    def _make_test_image_b64(
        size: int = 32,
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> str:
        from unity.image_manager.utils import make_solid_png_base64

        return make_solid_png_base64(size, size, color)

    # Step 1: Create contacts
    cm._create_contact(
        first_name="Alice",
        email_address="alice@test.com",
        phone_number="1111111111",
    )
    cm._create_contact(
        first_name="Bob",
        email_address="bob@test.com",
        phone_number="2222222222",
    )
    contacts = unify.get_logs(context=cm._ctx, from_fields=["contact_id", "first_name"])
    alice_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Alice"
    )
    bob_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Bob"
    )

    # Step 2: Store images
    img_ids = im.add_images(
        [
            {
                "data": _make_test_image_b64(color=(255, 0, 0)),
                "caption": "Screenshot 1",
            },
            {
                "data": _make_test_image_b64(color=(0, 0, 255)),
                "caption": "Screenshot 2",
            },
        ],
        synchronous=True,
    )
    assert img_ids[0] is not None and img_ids[1] is not None, "Image creation failed"
    img1_id = img_ids[0]
    img2_id = img_ids[1]

    # Step 3: Create function
    src = "def process():\n    return 'processed'\n"
    fm.add_functions(implementations=src)
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Step 4: Create guidance with images and function reference
    gm.add_guidance(
        title="Processing Guide",
        content="How to process data",
        images=[{"raw_image_ref": {"image_id": img1_id}, "annotation": "Setup"}],
        function_ids=[func_id],
    )
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    guidance_id = int(guidance_list[0].entries["guidance_id"])

    # Step 5: Create task with function entrypoint
    result = ts._create_task(
        name="Process Task",
        description="Processing task",
        entrypoint=func_id,
    )
    task_id = result["details"]["task_id"]

    # Step 6: Log message with image
    tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": alice_id,
            "receiver_ids": [bob_id],
            "content": "Check the guide",
            "timestamp": datetime.now(),
            "images": [
                {"raw_image_ref": {"image_id": img2_id}, "annotation": "Reference"},
            ],
        },
    )

    # Verify all relationships established
    messages = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["sender_id", "receiver_ids", "images"],
    )
    assert len(messages) == 1
    assert messages[0].entries["sender_id"] == alice_id
    assert bob_id in messages[0].entries["receiver_ids"]
    assert messages[0].entries["images"][0]["raw_image_ref"]["image_id"] == img2_id

    # Step 7: Delete img1 (used in guidance) - SET NULL behavior
    img1_logs = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {img1_id}",
        return_ids_only=True,
    )
    assert img1_logs, "Image not found"
    unify.delete_logs(context=im._ctx, logs=img1_logs[0])

    guidance_check = unify.get_logs(
        context=gm._ctx,
        filter=f"guidance_id == {guidance_id}",
        from_fields=["images"],
    )
    guid_imgs = guidance_check[0].entries.get("images", [])
    assert len(guid_imgs) == 1  # Array length unchanged (in-place SET NULL)
    assert (
        guid_imgs[0]["raw_image_ref"]["image_id"] is None
    )  # image_id replaced with None

    # Step 8: Delete function (used in task and guidance)
    fm.delete_function(function_id=func_id)

    # Task: SET NULL behavior (entrypoint becomes null)
    task_check = unify.get_logs(
        context=ts._ctx,
        filter=f"task_id == {task_id}",
        from_fields=[
            "task_id",
            "entrypoint",
        ],  # Include task_id to avoid NULL-only field omission
    )
    assert task_check[0].entries.get("entrypoint") is None  # SET NULL

    # Guidance: CASCADE behavior (function_id removed from array)
    guidance_check2 = unify.get_logs(
        context=gm._ctx,
        filter=f"guidance_id == {guidance_id}",
        from_fields=["function_ids"],
    )
    assert func_id not in guidance_check2[0].entries.get(
        "function_ids",
        [],
    )  # CASCADE removed it

    # Step 9: Delete Alice - SET NULL behavior (message survives with null sender)
    cm._delete_contact(contact_id=alice_id)

    messages_check = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["message_id", "sender_id", "receiver_ids"],
    )
    assert len(messages_check) == 1  # Message survives (SET NULL)
    assert (
        messages_check[0].entries.get("sender_id") is None
    )  # SET NULL anonymized sender


# --------------------------------------------------------------------------- #
#  Integration: Bulk Operations with FK Constraints                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_bulk_delete_preserves_fk_integrity():
    """
    Test bulk deletion operations with FK SET NULL behavior.
    Deleting senders should preserve messages with null sender_id.

    FK Policies:
    - Transcripts.sender_id → Contacts: SET NULL
    - Transcripts.receiver_ids[*] → Contacts: SET NULL (in-place replacement)
    """
    cm = ContactManager()
    tm = TranscriptManager()

    # Create 10 contacts (excluding system contacts 0 and 1)
    # Use letters instead of digits in names to match Contact.first_name pattern
    for i in range(10):
        cm._create_contact(
            first_name=f"TestUser{chr(65 + i)}",  # TestUserA, TestUserB, ..., TestUserJ
            email_address=f"user{i}@test.com",
            phone_number=f"{i:010d}",
        )

    contacts = unify.get_logs(
        context=cm._ctx,
        filter="contact_id > 1",
        from_fields=["contact_id"],
    )
    contact_ids = [int(c.entries["contact_id"]) for c in contacts]
    assert len(contact_ids) == 10

    # Create messages between all contacts (mesh network)
    for sender_id in contact_ids[:5]:
        for receiver_id in contact_ids[5:]:
            tm.log_first_message_in_new_exchange(
                {
                    "medium": "sms_message",
                    "sender_id": sender_id,
                    "receiver_ids": [receiver_id],
                    "content": f"Message from {sender_id} to {receiver_id}",
                    "timestamp": datetime.now(),
                },
            )

    # Should have 5*5 = 25 messages
    messages = unify.get_logs(context=tm._transcripts_ctx, from_fields=["message_id"])
    assert len(messages) == 25

    # Bulk delete first 5 contacts (senders)
    for cid in contact_ids[:5]:
        cm._delete_contact(contact_id=cid)

    # All messages should survive with SET NULL on sender_id
    messages_after = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["message_id", "sender_id", "receiver_ids"],
    )
    assert len(messages_after) == 25  # Messages survive

    # All messages should have null sender_id (SET NULL behavior)
    assert all(m.entries.get("sender_id") is None for m in messages_after)

    # Receiver_ids should still contain valid contacts (not deleted)
    for msg in messages_after:
        receivers = msg.entries.get("receiver_ids", [])
        assert len(receivers) > 0  # Receivers preserved
        assert all(
            rid in contact_ids[5:] for rid in receivers
        )  # Only last 5 contacts remain


@_handle_project
@pytest.mark.integration
def test_circular_fk_deletion_safety():
    """
    Test that circular FK references (Functions ↔ Guidance) don't cause
    infinite loops on deletion, even though both use CASCADE.

    FK Policies:
    - Functions.guidance_ids[*] → Guidance: CASCADE
    - Guidance.function_ids[*] → Functions: CASCADE

    The backend should handle circular CASCADE correctly without infinite loops.
    """
    from unity.function_manager.function_manager import FunctionManager
    from unity.guidance_manager.guidance_manager import GuidanceManager

    fm = FunctionManager()
    gm = GuidanceManager()

    # Create function
    src = "def circular():\n    return 'loop'\n"
    fm.add_functions(implementations=src)
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create guidance referencing function
    gm.add_guidance(title="Circular Guide", content="Guide", function_ids=[func_id])
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    guidance_id = int(guidance_list[0].entries["guidance_id"])

    # Update function to reference guidance (circular reference)
    func_logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"function_id == {func_id}",
        return_ids_only=True,
    )
    unify.update_logs(
        context=fm._compositional_ctx,
        logs=func_logs[0],
        entries={"guidance_ids": [guidance_id]},
        overwrite=True,
    )

    # Verify circular reference exists
    func_data = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"function_id == {func_id}",
        from_fields=["guidance_ids"],
    )
    assert guidance_id in func_data[0].entries["guidance_ids"]

    guidance_data = unify.get_logs(
        context=gm._ctx,
        filter=f"guidance_id == {guidance_id}",
        from_fields=["function_ids"],
    )
    assert func_id in guidance_data[0].entries["function_ids"]

    # Delete function (should not cause infinite loop, CASCADE removes from guidance array)
    fm.delete_function(function_id=func_id)

    # Verify guidance survives with function_id removed (CASCADE behavior)
    guidance_after = unify.get_logs(
        context=gm._ctx,
        filter=f"guidance_id == {guidance_id}",
        from_fields=["function_ids"],
    )
    assert len(guidance_after) == 1  # Guidance still exists
    assert func_id not in guidance_after[0].entries.get(
        "function_ids",
        [],
    )  # CASCADE removed it


@_handle_project
@pytest.mark.integration
def test_delete_exchange_cascades_messages():
    """
    Test that deleting an exchange cascades to delete all its messages.

    FK Policy:
    - Transcripts.exchange_id → Exchanges.exchange_id: CASCADE
    """
    cm = ContactManager()
    tm = TranscriptManager()

    # Create contacts
    cm._create_contact(
        first_name="Alice",
        email_address="alice@test.com",
        phone_number="1111111111",
    )
    cm._create_contact(
        first_name="Bob",
        email_address="bob@test.com",
        phone_number="2222222222",
    )
    contacts = unify.get_logs(context=cm._ctx, from_fields=["contact_id", "first_name"])
    alice_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Alice"
    )
    bob_id = next(
        int(c.entries["contact_id"])
        for c in contacts
        if c.entries["first_name"] == "Bob"
    )

    # Log first message (creates exchange)
    exchange_id, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": alice_id,
            "receiver_ids": [bob_id],
            "content": "Message 1",
            "timestamp": datetime.now(),
        },
    )

    # Log more messages in same exchange
    for i in range(2, 6):
        tm.log_messages(
            {
                "medium": "sms_message",
                "sender_id": bob_id if i % 2 == 0 else alice_id,
                "receiver_ids": [alice_id if i % 2 == 0 else bob_id],
                "content": f"Message {i}",
                "timestamp": datetime.now(),
                "exchange_id": exchange_id,
            },
        )

    # Verify 5 messages in exchange
    messages_in_exchange = unify.get_logs(
        context=tm._transcripts_ctx,
        filter=f"exchange_id == {exchange_id}",
        from_fields=["message_id"],
    )
    assert len(messages_in_exchange) == 5

    # Delete the exchange (CASCADE should delete all messages)
    exchange_logs = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exchange_id}",
        return_ids_only=True,
    )
    assert exchange_logs, "Exchange not found"
    unify.delete_logs(context=tm._exchanges_ctx, logs=exchange_logs[0])

    # Verify all messages deleted (CASCADE behavior)
    messages_after = unify.get_logs(
        context=tm._transcripts_ctx,
        from_fields=["message_id"],
    )
    assert len(messages_after) == 0  # All messages CASCADE deleted


# --------------------------------------------------------------------------- #
#  Integration: Cross-root FK references (overlay → Hive-shared parent)       #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.integration
def test_absolute_cross_root_fk_resolves_to_hive_target():
    """Per-body overlays can point at a Hive-shared parent via absolute FKs.

    An overlay that stores per-body state alongside Hive-shared rows
    spells the FK target root explicitly
    (``Hives/{hive_id}/Contacts.contact_id``). The FK rewriter in
    :meth:`ContextRegistry._get_contexts_for_manager` must leave those
    references untouched — otherwise it would prefix them with the
    overlay's own per-body base and the FK would never resolve.

    The test creates a Hive-shared parent, a per-body overlay whose FK
    targets the parent's absolute path, and asserts that deleting a parent
    row propagates through the FK (SET NULL) instead of silently
    orphaning the overlay row — proof the FK landed where we asked it to.
    """
    base = ContextRegistry.base_for("TestOverlay")
    assert base, "test setup did not provide a base context"

    hive_parent = f"{base}/Hives/99/TestTarget"
    unify.create_context(
        hive_parent,
        description="Hive-shared parent for the overlay FK integration test.",
        auto_counting={"id": None},
    )
    unify.create_fields({"id": "int", "label": "str"}, context=hive_parent)

    class _OverlayManager:
        class Config:
            required_contexts = [
                TableContext(
                    name="TestOverlay",
                    description="Per-body overlay pointing at a Hive-shared parent.",
                    fields={"overlay_id": "int", "target_id": "int"},
                    auto_counting={"overlay_id": None},
                    foreign_keys=[
                        {
                            "column": "target_id",
                            "references": f"{base}/Hives/99/TestTarget.id",
                            "on_delete": "SET NULL",
                        },
                    ],
                ),
            ]

    contexts = ContextRegistry._get_contexts_for_manager(_OverlayManager)
    entry = contexts["TestOverlay"]

    assert entry["resolved_name"] == f"{base}/TestOverlay"
    fk = entry["resolved_foreign_keys"][0]
    assert fk["references"] == f"{base}/Hives/99/TestTarget.id"

    ContextRegistry._create_context_wrapper("TestOverlayManager", entry)

    unify.log(context=hive_parent, entries={"label": "parent"})
    parent_row = unify.get_logs(
        context=hive_parent,
        filter="label == 'parent'",
        from_fields=["id"],
    )
    parent_id = int(parent_row[0].entries["id"])

    overlay_ctx = entry["resolved_name"]
    unify.log(context=overlay_ctx, entries={"target_id": parent_id})
    overlay_row = unify.get_logs(
        context=overlay_ctx,
        from_fields=["overlay_id", "target_id"],
    )
    assert overlay_row[0].entries["target_id"] == parent_id

    parent_logs = unify.get_logs(
        context=hive_parent,
        filter=f"id == {parent_id}",
        return_ids_only=True,
    )
    unify.delete_logs(context=hive_parent, logs=parent_logs[0])

    overlay_after = unify.get_logs(
        context=overlay_ctx,
        from_fields=["overlay_id", "target_id"],
    )
    assert len(overlay_after) == 1
    assert overlay_after[0].entries.get("target_id") is None  # SET NULL fired
