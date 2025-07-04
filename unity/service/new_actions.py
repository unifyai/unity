import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from unity.service import start, stop
from unity.service.events import (
    PhoneUtteranceEvent,
    PhoneCallInitiatedEvent,
    PhoneCallStopEvent,
)
from unity.service.utils import (
    publish_event,
    find_assistant_whatsapp_number,
    assign_new_assistant_whatsapp_number,
    find_assistant_phone_number,
    check_conflict,
    send_sms_notification,
    admin_update_assistant,
)

import unify
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

load_dotenv()

headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}


# Low-level functions
async def _send_whatsapp_message_via_number(to_number: str, message: str) -> bool:
    """
    Send a WhatsApp message using the WhatsApp Business API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        from_number = os.getenv("ASSISTANT_WHATSAPP_NUMBER")  # for debugging, to remove
        if not from_number:
            # always use the assistant phone number (unique) to find whatsapp number
            from_number = await find_assistant_whatsapp_number()

        # check conflict
        conflict = await check_conflict(from_number, to_number)
        # if not conflict:
        #     print(f"Conflict check error. Message not sent.")
        #     return False

        if conflict in ("both", "single"):
            new_whatsapp_number, from_user_phone_number = (
                await assign_new_assistant_whatsapp_number(
                    os.getenv("ASSISTANT_NUMBER"),
                    from_number,
                )
            )
            if not new_whatsapp_number:
                print(f"Failed to assign new WhatsApp number. Message not sent.")
                return False

            if conflict == "both":
                target_assistant_phone_number = await find_assistant_phone_number(
                    to_number,
                )
                second_new_whatsapp_number, target_user_phone_number = (
                    await assign_new_assistant_whatsapp_number(
                        target_assistant_phone_number,
                        from_number,
                        conflict_number=new_whatsapp_number,
                    )
                )
                if not second_new_whatsapp_number:
                    print(
                        f"Both conflicting. Failed to assign new WhatsApp number. Message not sent.",
                    )
                    return False

                update_res = await admin_update_assistant(
                    target_assistant_phone_number,
                    from_number,
                    second_new_whatsapp_number,
                )
                if not update_res:
                    print(
                        f"Both conflicting. Failed to update assistant. Message not sent.",
                    )
                    return False

                send_res = await send_sms_notification(
                    target_assistant_phone_number,
                    target_user_phone_number,
                    second_new_whatsapp_number,
                )
                if not send_res:
                    print(
                        f"Both conflicting. Failed to send SMS notification. Message not sent.",
                    )
                    return False

            update_res = await admin_update_assistant(
                os.getenv("ASSISTANT_NUMBER"),
                from_number,
                new_whatsapp_number,
            )
            if not update_res:
                print(f"Failed to update assistant. Message not sent.")
                return False
            send_res = await send_sms_notification(
                os.getenv("ASSISTANT_NUMBER"),
                from_user_phone_number,
                new_whatsapp_number,
            )
            if not send_res:
                print(f"Failed to send SMS notification. Message not sent.")
                return False

            from_number = new_whatsapp_number

        # no conflict, or numbers reassigned. proceed to send message
        print(f"Sending WhatsApp message from {from_number} to {to_number}: {message}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/whatsapp/send-text",
                headers=headers,
                json={
                    "from": from_number,
                    "to": to_number,
                    "body": message,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send WhatsApp message. Status: {response.status}")
                    return False

                response_text = await response.text()
                print(f"Response: {response_text}")
                return True
    except aiohttp.ClientError as e:
        print(f"Network error while sending WhatsApp message: {e}")
        return False
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False


async def _send_sms_message_via_number(to_number: str, message: str) -> bool:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        from_number = os.getenv("ASSISTANT_NUMBER")
        print(f"Sending SMS from {from_number} to {to_number}: {message}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-text",
                headers=headers,
                json={
                    "From": from_number,
                    "To": to_number,
                    "Body": message,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send SMS. Status: {response.status}")
                    return False

                response_text = await response.text()
                print(f"Response: {response_text}")
                return True
    except aiohttp.ClientError as e:
        print(f"Network error while sending SMS: {e}")
        return False
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False


async def _send_email_via_address(to_email: str, content: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_email: The email address to send the email to
        content: The message content to send

    Returns:
        str: A string indicating the result of the action
    """
    from_email = os.getenv("ASSISTANT_EMAIL")
    if not from_email:
        from_email = "unity.agent@unify.ai"  # todo: temp placeholder
        # print("No email address found for assistant")
        # return "Message not sent: No email address found for assistant"

    try:
        print(f"Sending email from {from_email} to {to_email}: {content}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/email/send",
                headers=headers,
                json={
                    "from": from_email,
                    "to": to_email,
                    "body": content,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send email. Status: {response.status}")
                    return "Message not sent: Failed to send email"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Message sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending email: {e}")
        return "Message not sent: Network error"
    except Exception as e:
        print(f"Error sending email: {e}")
        return "Message not sent: Error"


async def _start_call(to_number: str, purpose: str) -> bool:
    """
    Send a call using the call provider API.

    Args:
        to_number: The recipient's phone number

    Returns:
        bool: True if call was sent successfully, False otherwise
    """
    from_number = os.getenv("ASSISTANT_NUMBER")

    await publish_event(
        {
            "topic": to_number,
            "event": {
                **PhoneCallInitiatedEvent().to_dict(),
                "voice_id": None,
                "tts_provider": None,
                "outbound": True,
            },
        },
    )

    try:
        print(f"Sending call from {from_number} to {to_number}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-call",
                headers=headers,
                json={"From": from_number, "To": to_number, "NewCall": "true"},
            ) as response:
                if response.status != 200:
                    print(f"Failed to send call. Status: {response.status}")
                    return "Call not sent: Failed to send call"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Call sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending call: {e}")
        return "Call not sent: Network error"
    except Exception as e:
        print(f"Error sending call: {e}")
        return "Call not sent: Error"


# High-level Actions
async def send_whatsapp_message(
    description: str,
    parent_chat_context: list[dict] | None = None,
) -> SteerableToolHandle:
    contact_manager = ContactManager()
    transcript_manager = TranscriptManager(contact_manager=contact_manager)
    knowledge_manager = KnowledgeManager()
    task_scheduler = TaskScheduler()

    client = unify.AsyncUnify("o4-mini@openai")
    client.set_system_message(description)
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        task_scheduler.ask,
        _send_whatsapp_message_via_number,
        include_class_name=True,
    )
    return start_async_tool_use_loop(
        client,
        description,
        tools,
        loop_id="send_whatsapp_message",
        parent_chat_context=parent_chat_context,
        tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
    )


async def send_sms_message(
    description: str,
    parent_chat_context: list[dict] | None = None,
) -> SteerableToolHandle:
    contact_manager = ContactManager()
    transcript_manager = TranscriptManager(contact_manager=contact_manager)
    knowledge_manager = KnowledgeManager()
    task_scheduler = TaskScheduler()

    client = unify.AsyncUnify("o4-mini@openai")
    client.set_system_message(description)
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        task_scheduler.ask,
        _send_sms_message_via_number,
        include_class_name=True,
    )
    return start_async_tool_use_loop(
        client,
        description,
        tools,
        loop_id="send_sms_message",
        parent_chat_context=parent_chat_context,
        tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
    )


async def send_email(
    description: str,
    parent_chat_context: list[dict] | None = None,
) -> SteerableToolHandle:
    contact_manager = ContactManager()
    transcript_manager = TranscriptManager(contact_manager=contact_manager)
    knowledge_manager = KnowledgeManager()
    task_scheduler = TaskScheduler()

    client = unify.AsyncUnify("o4-mini@openai")
    client.set_system_message(description)
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        task_scheduler.ask,
        _send_email_via_address,
        include_class_name=True,
    )
    return start_async_tool_use_loop(
        client,
        description,
        tools,
        loop_id="send_email",
        parent_chat_context=parent_chat_context,
        tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
    )


class Call(SteerableToolHandle):

    def __init__(self, phone_number: str, purpose: str):
        """
        Starts a new phone call session and exposes the steerable methods
        """
        self.contact_manager = ContactManager()
        self.transcript_manager = TranscriptManager(
            contact_manager=self.contact_manager,
        )
        self.knowledge_manager = KnowledgeManager()
        self.task_scheduler = TaskScheduler()

        self.phone_number = phone_number
        self.purpose = purpose

        self.client = unify.AsyncUnify("o4-mini@openai")
        self.client.set_system_message(
            f"You are a helpful assistant. You are calling {self.phone_number} for {self.purpose}.",
        )
        self.tools = methods_to_tool_dict(
            self.contact_manager.ask,
            self.transcript_manager.ask,
            self.transcript_manager.summarize,
            self.knowledge_manager.ask,
            self.task_scheduler.ask,
            _send_email_via_address,
            _send_sms_message_via_number,
        )

        start()
        asyncio.create_task(_start_call(phone_number, purpose))
        self.status = "started"

    async def ask(self, question: str) -> SteerableToolHandle:
        """
        Ask a question to the assistant.
        """
        await publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="User",
                    content=question,
                ).to_dict(),
            },
        )
        return start_async_tool_use_loop(
            self.client,
            question,
            self.tools,
            loop_id="call",
        )

    async def interject(self, text: str) -> str:
        """
        Interject a message to the assistant for them to speak it to the user.
        """
        await publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="User",
                    content=text,
                ).to_dict(),
            },
        )
        return "Acknowledged."

    async def stop(self):
        """
        End the call.
        """
        await publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": PhoneCallStopEvent().to_dict(),
            },
        )
        await asyncio.sleep(5)
        stop()
        self.status = "ended"

    def result(self) -> str:
        return self.status

    def pause(self) -> str:
        return "Not applicable."

    def resume(self) -> str:
        return "Not applicable."

    def done(self) -> bool:
        return self.status == "ended"
