import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *
from unity.conversation_manager.domains.contact_index import Contact

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()


def build_response_models():
    """
    Create response models for ConversationManager's main brain.

    All actions (comms, task steering, etc.) are now tool calls.
    The response model only captures the LLM's reasoning.

    Returns:
        dict: Response models for different modes (call, unify_meet, text)
    """
    # Text mode: just thoughts
    TextResponse = create_model(
        "TextResponse",
        thoughts=(
            str,
            Field(..., description="Your concise reasoning before taking actions"),
        ),
        __base__=BaseModel,
    )

    # Voice mode: thoughts + guidance for the Voice Agent
    # Both TTS and Realtime modes use call_guidance - the Main CM Brain
    # provides guidance/data to the voice agent (fast brain) which handles
    # the actual conversation.
    VoiceResponse = create_model(
        "VoiceResponse",
        thoughts=(
            str,
            Field(..., description="Your concise reasoning before taking actions"),
        ),
        call_guidance=(
            str,
            Field(..., description="Guidance for the Voice Agent handling the call"),
        ),
        __base__=BaseModel,
    )

    return {
        "call": VoiceResponse,
        "unify_meet": VoiceResponse,
        "text": TextResponse,
    }


# utils
async def get_update_or_create_contact(
    cm: "ConversationManager",
    contact_id: int = None,
    details: dict = None,
):
    if not contact_id and not details:
        # bad
        ...
    # means update
    elif contact_id and details:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        data_to_insert = {}
        for k, v in details.items():
            if v:
                if contact[k] != v:
                    data_to_insert[k] = v
        updated_contacts_raw = cm.contact_manager.get_contact_info(
            contact_id=list(cm.contact_index.contacts.keys()),
        )
        # Update contacts dict with Contact objects
        for cid, uc in updated_contacts_raw.items():
            if cid in cm.contact_index.contacts:
                existing = cm.contact_index.contacts[cid]
                cm.contact_index.contacts[cid] = Contact(
                    **{**existing.model_dump(), **uc, "threads": existing.threads},
                )
            else:
                cm.contact_index.contacts[cid] = Contact(**uc)
        # Update active_conversations similarly
        for cid, c in cm.contact_index.active_conversations.items():
            if cid in updated_contacts_raw:
                uc = updated_contacts_raw[cid]
                cm.contact_index.active_conversations[cid] = Contact(
                    **{**c.model_dump(), **uc, "threads": c.threads},
                )
        phone_number, email_address = details.get("phone_number"), details.get(
            "email_address",
        )
        contact = (
            cm.contact_index.get_contact(phone_number=phone_number)
            if phone_number
            else cm.contact_index.get_contact(email=email_address)
        )
        return contact

    # means retrieve if exists, create if not
    elif details:
        phone_number, email_address = details.get("phone_number"), details.get(
            "email_address",
        )
        maybe_contact = cm.contact_index.get_contact(
            phone_number=phone_number,
        ) or cm.contact_index.get_contact(email=email_address)
        if maybe_contact:
            return maybe_contact
        tool_outcome = await asyncio.to_thread(
            cm.contact_manager._create_contact,
            **details,
        )
        new_contact_id = tool_outcome["details"]["contact_id"]
        new_contact = await asyncio.to_thread(
            cm.contact_manager.get_contact_info,
            new_contact_id,
        )
        cm.contact_index.contacts[new_contact_id] = Contact(
            **new_contact[new_contact_id],
        )
        # all good, maybe no need to get all contacts here
        return new_contact[new_contact_id]

    # just retrieve
    elif contact_id:
        # means just message this person directly
        return cm.contact_index.get_contact(contact_id=contact_id)


async def send_sms(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    # contact_id = kwargs.get("contact_id")
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    content = kwargs.get("content")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_number = contact.get("phone_number")
    response = await comms_utils.send_sms_message_via_number(
        to_number=to_number,
        content=content,
    )

    if response["success"]:
        contact = cm.contact_index.get_contact(phone_number=to_number)
        event = SMSSent(contact=contact, content=content)
    else:
        if not cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send sms to {to_number}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:sms_sent", event.to_json())


async def send_unify_message(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    content = kwargs.get("content")
    contact_id = kwargs.get("contact_id")
    if contact_id is None:
        raise ValueError("contact_id is required for send_unify_message")
    response = await comms_utils.send_unify_message(
        content=content,
        contact_id=contact_id,
    )
    if response["success"]:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        event = UnifyMessageSent(contact=contact, content=content)
    else:
        event = Error(f"Failed to send unify message")
    await event_broker.publish("app:comms:unify_message_sent", event.to_json())


async def send_email(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_email = contact.get("email_address")
    subject = kwargs.get("subject")
    body = kwargs.get("body")
    email_id_to_reply_to = kwargs.get("email_id_to_reply_to")

    # ------------------------------------------------------------------
    # Reduce flakiness: prefer the most recent inbound email's Message-ID
    # for this contact+subject, rather than trusting the LLM to copy it.
    #
    # In practice the model may output `null` (or an incorrect value) because
    # the schema allows Optional[str]. For replies, we can infer the correct
    # Message-ID from the active conversation email thread.
    # ------------------------------------------------------------------
    inferred_reply_id: str | None = None
    try:
        convo = None
        if contact_id is not None:
            convo = cm.contact_index.active_conversations.get(contact_id)
        if convo is None and to_email:
            convo = next(
                (
                    c
                    for c in cm.contact_index.active_conversations.values()
                    if getattr(c, "email_address", None) == to_email
                ),
                None,
            )
        if convo is not None:
            thread = getattr(convo, "threads", {}).get("email")
            if thread:
                for m in reversed(thread):
                    # Prefer the most recent *user* email (name != "You") with the
                    # same subject and a non-empty email_id.
                    if (
                        getattr(m, "name", None) != "You"
                        and getattr(m, "subject", None) == subject
                        and getattr(m, "email_id", None)
                    ):
                        inferred_reply_id = m.email_id
                        break
    except Exception:
        inferred_reply_id = None

    if inferred_reply_id and inferred_reply_id != email_id_to_reply_to:
        email_id_to_reply_to = inferred_reply_id
    response = await comms_utils.send_email_via_address(
        to_email=to_email,
        subject=subject,
        body=body,
        email_id=email_id_to_reply_to,
    )
    if response["success"]:
        contact = cm.contact_index.get_contact(email=to_email)
        event = EmailSent(
            contact=contact,
            body=body,
            subject=subject,
            email_id_replied_to=email_id_to_reply_to,
        )
    else:
        if not cm.assistant_email:
            error_msg = "You don't have an email address, please provision one."
        else:
            error_msg = f"Failed to send email to {to_email}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:email_sent", event.to_json())


async def make_call(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_number = contact.get("phone_number")
    response = await comms_utils.start_call(to_number=to_number)
    if response["success"]:
        contact = cm.contact_index.get_contact(phone_number=to_number)
        event = PhoneCallSent(contact=contact)
    else:
        if not cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send call to {to_number}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:make_call", event.to_json())


_next_handle_id = 0


async def start_task_action(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    """Start a new task, store handle, and publish started."""
    global _next_handle_id

    await managers_utils.wait_for_initialization(cm)
    query = kwargs["query"]

    handle = await cm.actor.act(
        query,
        _parent_chat_context=cm.chat_history,
    )

    # allocate handle id and register
    handle_id = _next_handle_id
    _next_handle_id += 1
    cm.active_tasks[handle_id] = {
        "handle": handle,
        "query": query,
        "handle_actions": [],
    }

    # publish started
    await event_broker.publish(
        f"app:actor:actor_started_handle_{handle_id}",
        ActorHandleStarted(
            handle_id=handle_id,
            action_name=action_name,
            query=query,
        ).to_json(),
    )

    # spawn watchers
    asyncio.create_task(managers_utils.actor_watch_result(handle_id, handle))
    asyncio.create_task(managers_utils.actor_watch_notifications(handle_id, handle))
    asyncio.create_task(
        managers_utils.actor_watch_clarifications(handle_id, handle),
    )
