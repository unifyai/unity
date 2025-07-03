import aiohttp
import os
from dotenv import load_dotenv

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


# High-level Actions
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
