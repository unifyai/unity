from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
import unify

from tests.helpers import _handle_project
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.session_details import SESSION_DETAILS
from unity.contact_manager.types.contact import Contact
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message


def _space_id() -> int:
    return int(time.time_ns() % 1_000_000_000)


def _logs(context: str, filter_expr: str):
    return unify.get_logs(context=context, filter=filter_expr, return_ids_only=False)


def _delete_context_tree(root: str) -> None:
    try:
        children = list(unify.get_contexts(prefix=f"{root}/").keys())
    except Exception:
        children = []
    for context in sorted(children, key=len, reverse=True):
        try:
            unify.delete_context(context)
        except Exception:
            pass
    try:
        unify.delete_context(root)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_space_membership_state():
    yield
    for space_id in SESSION_DETAILS.space_ids:
        _delete_context_tree(f"Spaces/{space_id}")
    SESSION_DETAILS.space_ids = []
    SESSION_DETAILS.space_summaries = []


def _message_payload(
    content: str,
    *,
    exchange_id: int | None = None,
    medium: str = "email",
    sender_id=0,
    receiver_ids=None,
) -> dict:
    payload = {
        "medium": medium,
        "sender_id": sender_id,
        "receiver_ids": receiver_ids if receiver_ids is not None else [1],
        "timestamp": datetime.now(UTC),
        "content": content,
    }
    if exchange_id is not None:
        payload["exchange_id"] = exchange_id
    return payload


@_handle_project
def test_transcript_writes_route_to_space_and_reads_fan_out():
    space_id = _space_id()
    SESSION_DETAILS.space_ids = [space_id]
    tm = TranscriptManager()

    routed_content = f"shared launch transcript {space_id}"
    created = tm.log_messages(
        _message_payload(routed_content, exchange_id=81001),
        synchronous=True,
        destination=f"space:{space_id}",
    )
    assert isinstance(created, list)
    first_exchange_id, first_message_id = tm.log_first_message_in_new_exchange(
        _message_payload(f"shared kickoff transcript {space_id}"),
        exchange_initial_metadata={"topic": "kickoff"},
        destination=f"space:{space_id}",
    )

    personal_transcripts = tm._transcripts_ctx
    personal_exchanges = tm._exchanges_ctx
    space_transcripts = f"Spaces/{space_id}/Transcripts"
    space_exchanges = f"Spaces/{space_id}/Exchanges"

    assert not _logs(personal_transcripts, f"content == '{routed_content}'")
    assert _logs(space_transcripts, f"content == '{routed_content}'")
    assert _logs(space_exchanges, "exchange_id == 81001")
    assert not _logs(personal_exchanges, f"exchange_id == {first_exchange_id}")
    assert _logs(space_exchanges, f"exchange_id == {first_exchange_id}")

    messages = tm._filter_messages(filter=f"content == '{routed_content}'")["messages"]
    assert [message.content for message in messages] == [routed_content]
    semantic_messages = tm._search_messages(
        references={"content": "shared launch transcript"},
        k=3,
    )["messages"]
    assert any(message.content == routed_content for message in semantic_messages)
    assert tm.get_exchange_metadata(first_exchange_id).exchange_id == first_exchange_id
    assert (
        tm._reduce(metric="count", keys="message_id", group_by="medium")["email"] >= 2
    )

    image_refs = [{"raw_image_ref": {"image_id": 123}, "annotation": "diagram"}]
    tm.update_message_images(
        first_message_id,
        image_refs,
        destination=f"space:{space_id}",
    )
    [updated_log] = _logs(space_transcripts, f"message_id == {first_message_id}")
    assert updated_log.entries["images"] == image_refs


@_handle_project
def test_transcript_move_helpers_relocate_rows_and_surface_destination_errors():
    space_id = _space_id()
    SESSION_DETAILS.space_ids = [space_id]
    tm = TranscriptManager()

    exchange_id, message_id = tm.log_first_message_in_new_exchange(
        _message_payload(f"personal source transcript {space_id}"),
        exchange_initial_metadata={"owner": "personal"},
    )

    update = tm.update_exchange_metadata(
        exchange_id,
        {"owner": "space"},
        destination=f"space:{space_id}",
    )
    assert update.exchange_id == exchange_id

    message_move = tm.move_message(
        message_id,
        from_root="personal",
        to_destination=f"space:{space_id}",
    )
    exchange_move = tm.move_exchange(
        exchange_id,
        from_root="personal",
        to_destination=f"space:{space_id}",
    )

    assert message_move["details"]["to_context"] == f"Spaces/{space_id}/Transcripts"
    assert exchange_move["details"]["to_context"] == f"Spaces/{space_id}/Exchanges"
    assert not _logs(tm._transcripts_ctx, f"message_id == {message_id}")
    assert not _logs(tm._exchanges_ctx, f"exchange_id == {exchange_id}")
    assert _logs(f"Spaces/{space_id}/Transcripts", f"message_id == {message_id}")
    assert _logs(f"Spaces/{space_id}/Exchanges", f"exchange_id == {exchange_id}")

    invalid = tm.move_message(
        message_id,
        from_root=f"space:{space_id}",
        to_destination="space:999999999",
    )
    assert invalid["error_kind"] == "invalid_destination"

    with pytest.raises(ValueError):
        tm.move_message(
            987654321,
            from_root="personal",
            to_destination=f"space:{space_id}",
        )


@_handle_project
def test_invalid_transcript_destination_returns_tool_error():
    SESSION_DETAILS.space_ids = []
    tm = TranscriptManager()

    result = tm.log_messages(
        Message(**_message_payload("invalid destination transcript", exchange_id=99)),
        synchronous=True,
        destination="space:42",
    )

    assert result["error_kind"] == "invalid_destination"
    assert result["details"]["destination"] == "space:42"


@_handle_project
def test_transcript_image_tools_use_the_message_root_for_duplicate_image_ids():
    space_id = _space_id()
    SESSION_DETAILS.space_ids = [space_id]
    images = ImageManager()
    tm = TranscriptManager()
    personal_image_data = make_solid_png_base64(8, 8, (255, 0, 0))
    space_image_data = make_solid_png_base64(8, 8, (0, 0, 255))

    [personal_image_id] = images.add_images(
        [
            {
                "timestamp": datetime.now(UTC),
                "caption": "personal duplicate image",
                "data": personal_image_data,
            },
        ],
        synchronous=True,
    )
    [space_image_id] = images.add_images(
        [
            {
                "timestamp": datetime.now(UTC),
                "caption": "space duplicate image",
                "data": space_image_data,
            },
        ],
        synchronous=True,
        destination=f"space:{space_id}",
    )
    assert personal_image_id == space_image_id

    [message] = tm.log_messages(
        {
            **_message_payload(
                "shared transcript with duplicate image",
                exchange_id=99,
            ),
            "images": [
                {
                    "raw_image_ref": {"image_id": space_image_id},
                    "annotation": "shared duplicate",
                },
            ],
        },
        synchronous=True,
        destination=f"space:{space_id}",
    )

    metadata = tm._get_images_for_message(message_id=message.message_id)

    assert metadata == [
        {
            "image_id": space_image_id,
            "caption": "space duplicate image",
            "timestamp": metadata[0]["timestamp"],
            "annotation": "shared duplicate",
        },
    ]
    attached = tm._attach_image_to_context(image_id=space_image_id)
    assert attached["image"] == space_image_data


@_handle_project
def test_transcript_contact_search_and_reductions_read_shared_roots():
    space_id = _space_id()
    SESSION_DETAILS.space_ids = [space_id]
    tm = TranscriptManager()

    personal_marker = f"personal decoy transcript {space_id}"
    shared_sender_marker = f"shared sender transcript {space_id}"
    shared_receiver_marker = f"shared receiver transcript {space_id}"

    tm.log_messages(
        _message_payload(
            personal_marker,
            exchange_id=12001,
            sender_id=Contact(
                first_name="Nina",
                surname="Desk",
                bio="Private planning contact for personal browser notes.",
            ),
        ),
        synchronous=True,
    )
    tm.log_messages(
        _message_payload(
            shared_sender_marker,
            exchange_id=98001,
            medium="phone_call",
            sender_id=Contact(
                first_name="Mara",
                surname="Field",
                bio="Heliotrope relay supervisor for compressor callback bundles.",
            ),
        ),
        synchronous=True,
        destination=f"space:{space_id}",
    )
    tm.log_messages(
        _message_payload(
            shared_receiver_marker,
            exchange_id=99001,
            medium="sms_message",
            sender_id=Contact(
                first_name="Iris",
                surname="Research",
                bio="Market pricing analyst for quarterly customer interview synthesis.",
            ),
            receiver_ids=[
                Contact(
                    first_name="Omar",
                    surname="Dispatch",
                    bio="Zephyr dispatch owner for overnight compressor incidents.",
                ),
            ],
        ),
        synchronous=True,
        destination=f"space:{space_id}",
    )

    sender_results = tm._search_messages(
        references={"sender_bio": "heliotrope relay supervisor"},
        k=1,
    )["messages"]
    assert [message.content for message in sender_results] == [shared_sender_marker]

    receiver_results = tm._search_messages(
        references={"receiver_bio": "zephyr dispatch owner"},
        k=1,
    )["messages"]
    assert [message.content for message in receiver_results] == [shared_receiver_marker]

    counts_by_medium = tm._reduce(
        metric="count",
        keys="message_id",
        group_by="medium",
    )
    assert counts_by_medium["email"] == 1
    assert counts_by_medium["phone_call"] == 1
    assert counts_by_medium["sms_message"] == 1
    assert tm._reduce(metric="max", keys="exchange_id") == 99001.0


@_handle_project
def test_transcript_semantic_search_globally_ranks_across_roots():
    space_id = _space_id()
    SESSION_DETAILS.space_ids = [space_id]
    tm = TranscriptManager()

    target_content = (
        f"Shared compressor callback bundle {space_id}: use amber coupler routing."
    )
    tm.log_messages(
        _message_payload(
            f"Personal lunch planning note {space_id}: buy apples after work.",
            exchange_id=22001,
        ),
        synchronous=True,
    )
    tm.log_messages(
        _message_payload(target_content, exchange_id=22002),
        synchronous=True,
        destination=f"space:{space_id}",
    )

    results = tm._search_messages(
        references={"content": "compressor callback amber coupler routing"},
        k=1,
    )["messages"]

    assert [message.content for message in results] == [target_content]
