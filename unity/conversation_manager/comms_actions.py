import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel
from enum import EnumType
import aiohttp
import os
import redis
import json
import ast
from dotenv import load_dotenv
from unity.conversation_manager.events import (
    EmailSentEvent,
    Event,
    PhoneUtteranceEvent,
    PhoneCallInitiatedEvent,
    PhoneCallStopEvent,
    InterruptEvent,
    SMSMessageSentEvent,
    WhatsappMessageSentEvent,
)
from unity.conversation_manager.prompt_builders import (
    build_call_ask_prompt,
    build_local_chat_search_prompt,
    build_message_prompt,
)
from unity.conversation_manager.utils import (
    create_connection,
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
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

load_dotenv()

# Global connection variables for comms_actions
headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
reader = None
writer = None

# Local chat history builder
# This is required as Call/Meet is a separate process and requires polling data
redis_client = None
local_chat_history = []


async def _publish_event(ev: dict):
    """Publish an event using the comms_actions connection"""
    global reader, writer

    if reader is None or writer is None:
        reader, writer = await create_connection("comms")
    return await publish_event(ev, writer=writer)


def _init_redis_client():
    global redis_client
    redis_client = redis.Redis(host="localhost", port=6379, db=0).pubsub()
    redis_client.subscribe("local_chat")


def _get_update_from_redis():
    global redis_client

    if not redis_client:
        _init_redis_client()

    msg = redis_client.get_message()
    if msg and msg["type"] == "message":
        data = msg["data"]
        try:
            return json.loads(data)
        except Exception:
            try:
                return ast.literal_eval(
                    data.decode() if isinstance(data, bytes) else data,
                )
            except Exception:
                return None
    return None


def _poll_updates():
    global local_chat_history

    while True:
        payload = _get_update_from_redis()
        if payload is None:
            break

        # ensure dict
        if not isinstance(payload, dict):
            continue
        local_chat_history.append(payload)


def _wrap_response_format(response_format):
    if isinstance(response_format, EnumType):

        class WrappedModel(BaseModel):
            answer: response_format

            def to_json(self):
                return json.dumps(
                    {
                        "answer": self.answer.value,
                    },
                )

        return WrappedModel, True

    return response_format, False


def build_local_chat_history():
    global local_chat_history

    _poll_updates()
    return (
        "\n".join(str(Event.from_dict(e)) for e in local_chat_history)
        if local_chat_history
        else ""
    )


# Low-level functions
async def _send_whatsapp_message_via_number(
    to_number: str,
    message: str,
    reply_to_user: bool = False,
) -> str:
    """
    Send a WhatsApp message using the WhatsApp Business API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send
        reply_to_user: `True` if replying to user's message. `False` if starting a new conversation.

    Returns:
        str: The response from the WhatsApp API
    """
    # always use the assistant phone number (unique) to find whatsapp number
    from_number = await find_assistant_whatsapp_number()
    if not from_number:
        from_number = os.getenv("ASSISTANT_NUMBER")  # for debugging, to remove

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
                from_number,  # 'to' user has assistant of same whatsapp number
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
        send_endpoint = "send-text" if reply_to_user else "send-greeting"
        json_payload = {
            "from": from_number,
            "to": to_number,
            "body": message,
        }
        if not reply_to_user:
            json_payload["user_name"] = os.getenv("USER_NAME")
            json_payload["agent_name"] = os.getenv("ASSISTANT_NAME")

        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/whatsapp/{send_endpoint}",
            headers=headers,
            json=json_payload,
        ) as response:
            response.raise_for_status()
            response_text = await response.text()
            print(f"Response: {response_text}")
            await _publish_event(
                {
                    "topic": to_number,
                    "to": "past",
                    "event": WhatsappMessageSentEvent(
                        content=message,
                        role="Assistant",
                        timestamp=datetime.now().isoformat(),
                    ).to_dict(),
                },
            )
            return response_text


async def _send_sms_message_via_number(
    to_number: str,
    message: str,
) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        str: The response from the SMS API
    """
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
            response.raise_for_status()
            response_text = await response.text()
            print(f"Response: {response_text}")
            await _publish_event(
                {
                    "topic": to_number,
                    "to": "past",
                    "event": SMSMessageSentEvent(
                        content=message,
                        role="Assistant",
                        timestamp=datetime.now().isoformat(),
                    ).to_dict(),
                },
            )
            return response_text


async def _send_email_via_address(
    to_email: str,
    subject: str,
    content: str,
    message_id: str,
) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_email: The email address to send the email to
        subject: The subject of the email
        content: The message content to send
        message_id: The message ID of the email to reply to

    Returns:
        str: The response from the email API
    """
    from_email = os.getenv("ASSISTANT_EMAIL")

    print(f"Sending email from {from_email} to {to_email}: {content}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/email/send",
            headers=headers,
            json={
                "from": from_email,
                "to": to_email,
                "subject": subject,
                "body": content,
                "in_reply_to": message_id,
            },
        ) as response:
            response.raise_for_status()
            response_text = await response.text()
            print(f"Response: {response_text}")
            await _publish_event(
                {
                    "topic": to_email,
                    "to": "past",
                    "event": EmailSentEvent(
                        content=content,
                        role="Assistant",
                        timestamp=datetime.now().isoformat(),
                        message_id=message_id,
                    ).to_dict(),
                },
            )
            return response_text


async def _start_call(
    from_number: str,
    to_number: str,
    purpose: str = "general",
    task_context: Dict[str, str] = None,
    ongoing_call: bool = False,
) -> str:
    """
    Send a call using the call provider API.

    Args:
        from_number: The sender's phone number
        to_number: The recipient's phone number
        purpose: The purpose of the call
        task_context: The broader task context for the call, with name and description attributes. Use None if there is no task context.

    Returns:
        str: The response from the email API
    """
    if not from_number:
        from_number = os.getenv("ASSISTANT_NUMBER")

    await _publish_event(
        {
            "topic": to_number,
            "event": {
                **PhoneCallInitiatedEvent(
                    purpose=purpose,
                    task_context=task_context,
                    target_number=to_number,
                    voice_id=os.getenv("VOICE_ID", None),
                    tts_provider=os.getenv("TTS_PROVIDER", "cartesia"),
                ).to_dict(),
            },
        },
    )

    if not ongoing_call:
        print(f"Sending call from {from_number} to {to_number}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-call",
                headers=headers,
                json={"From": from_number, "To": to_number, "NewCall": "true"},
            ) as response:
                response.raise_for_status()
                response_text = await response.text()
                print(f"Response: {response_text}")
                return response_text


async def _join_meet_call(
    meet_id: str,
    purpose: str = "general",
    task_context: Dict[str, str] = None,
    ongoing_call: bool = False,
) -> str:
    """
    Join a Google Meet call.

    Args:
        meet_id: The ID of the Google Meet call to join
        purpose: The purpose of the call
        task_context: The broader task context for the call, with name and description attributes. Use None if there is no task context.

    Returns:
        str: The response from the Google Meet API
    """
    await _publish_event(
        {
            "topic": os.getenv("USER_NUMBER"),
            "event": {
                **PhoneCallInitiatedEvent(
                    purpose=purpose,
                    task_context=task_context,
                    target_number=os.getenv("USER_NUMBER"),
                    meet_id=meet_id,
                    voice_id=os.getenv("VOICE_ID", None),
                    tts_provider=os.getenv("TTS_PROVIDER", "cartesia"),
                ).to_dict(),
            },
        },
    )

    if not ongoing_call:
        print(f"Joining Google Meet call with ID: {meet_id}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/meet-call",
                headers=headers,
                json={
                    "from": os.getenv("ASSISTANT_NUMBER"),
                    "to": os.getenv("USER_NUMBER"),
                    "meet_id": meet_id,
                },
            ) as response:
                response.raise_for_status()
                response_text = await response.text()
                print(f"Response: {response_text}")
                return response_text


# High-level Actions
async def send_whatsapp_message(
    description: str,
    parent_chat_context: list[dict] | None = None,
) -> SteerableToolHandle:
    contact_manager = ContactManager()
    transcript_manager = TranscriptManager(contact_manager=contact_manager)
    knowledge_manager = KnowledgeManager()

    client = unify.AsyncUnify("o4-mini@openai")
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        _send_whatsapp_message_via_number,
        include_class_name=True,
    )
    client.set_system_message(build_message_prompt(tools, description, "whatsapp"))
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

    client = unify.AsyncUnify("o4-mini@openai")
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        _send_sms_message_via_number,
        include_class_name=True,
    )
    client.set_system_message(build_message_prompt(tools, description, "sms"))
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

    client = unify.AsyncUnify("o4-mini@openai")
    tools = methods_to_tool_dict(
        contact_manager.ask,
        transcript_manager.ask,
        transcript_manager.summarize,
        knowledge_manager.ask,
        _send_email_via_address,
        include_class_name=True,
    )
    client.set_system_message(build_message_prompt(tools, description, "email"))
    return start_async_tool_use_loop(
        client,
        description,
        tools,
        loop_id="send_email",
        parent_chat_context=parent_chat_context,
        tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
    )


class Call(SteerableToolHandle):

    def __init__(
        self,
        phone_number: str,
        purpose: str,
        task_context: Dict[str, str] = None,
        tools: Dict[str, Any] = None,
    ):
        """
        Starts a new phone call session and exposes the steerable methods
        """

        self.phone_number = phone_number
        self.purpose = purpose
        self.task_context = task_context

        self.client = unify.AsyncUnify("o4-mini@openai")
        self.tools = methods_to_tool_dict(
            self._search_local_chat,
            self._ask_user_then_search,
        )
        if tools:
            self.tools = {
                **tools,
                **self.tools,
            }

        self.call_ready = asyncio.Event()
        self.call_ask_status = asyncio.Event()
        self.call_ask_status.set()

        self.status = "initiated"

    # init
    async def _start_call_task(self):
        """Internal helper: perform the call start then mark ready."""
        await _start_call(
            os.getenv("ASSISTANT_NUMBER"),
            self.phone_number,
            self.purpose,
            self.task_context,
        )
        # give time to start call and complete greeting
        await asyncio.sleep(20)
        self.call_ready.set()
        self.status = "started"

    @classmethod
    async def create(
        cls,
        phone_number: str,
        purpose: str,
        task_context: Dict[str, str] = None,
        tools: Dict[str, Any] = None,
    ) -> "Call":
        """Async factory for Call: constructs and schedules the call immediately."""
        instance = cls(phone_number, purpose, task_context, tools)
        await instance._start_call_task()
        return instance

    # ask/interject
    async def _search_local_chat(self, question: str):
        """Search local chat window if a user response relevant to the question is found"""

        client = unify.AsyncUnify("o4-mini@openai")

        # Use shared prompt builder for local chat search
        client.set_system_message(
            build_local_chat_search_prompt(build_local_chat_history()),
        )

        handle = start_async_tool_use_loop(
            client,
            f"This is the user's question: {question}.",
            {},
            loop_id="call_search_local_chat",
        )
        return await handle.result()

    async def _ask_user_then_search(self, question):
        """Ask user the question, then search for their response in local chat history."""
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": InterruptEvent().to_dict(),
            },
        ),
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Ask the user this question directly: {question}",
                ).to_dict(),
            },
        )

        # give time for utterance and response
        await asyncio.sleep(40)
        return await self._search_local_chat(question)

    async def ask(
        self,
        question: str,
        response_format: Optional[
            str | int | float | bool | BaseModel | EnumType
        ] = None,
    ) -> SteerableToolHandle:
        """
        Ask a question to the assistant.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()

        self.call_ask_status.clear()

        self.client.set_system_message(
            build_call_ask_prompt(self.tools, question),
        )

        is_enum = False
        if response_format:
            response_format, is_enum = _wrap_response_format(response_format)

        handle = start_async_tool_use_loop(
            self.client,
            f"The user is answering this question: {question}. Use available tools to get information of the user's answer.",
            self.tools,
            loop_id="call_ask",
            response_format=response_format,
        )

        if is_enum:
            original_result = handle.result

            async def _wrap():
                answer = await original_result()
                answer = json.loads(answer)
                return answer["answer"]

            handle.result = _wrap

        async def _reset_call_ask_status():
            try:
                await handle.result()
            finally:
                self.call_ask_status.set()

        asyncio.create_task(_reset_call_ask_status())
        return handle

    async def interject(self, text: str) -> str:
        """
        Interject a message to the assistant for them to log as instruction for next response.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()
        self.call_ask_status.clear()

        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Instruction on next response: {text}",
                ).to_dict(),
            },
        )

        self.call_ask_status.set()
        return f"Message interjected to user: {text}"

    async def stop(self):
        """
        End the call.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": InterruptEvent().to_dict(),
            },
        ),
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Say goodbye to user.",
                ).to_dict(),
            },
        )
        await asyncio.sleep(15)
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": PhoneCallStopEvent().to_dict(),
            },
        )
        self.status = "ended"

    def result(self) -> str:
        return self.status

    def pause(self) -> str:
        return "Not applicable."

    def resume(self) -> str:
        return "Not applicable."

    def done(self) -> bool:
        return self.status == "ended"


class GoogleMeet(SteerableToolHandle):
    def __init__(
        self,
        meet_id: str,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
        tools: Dict[str, Any] = None,
    ):
        self.meet_id = meet_id
        self.purpose = purpose
        self.task_context = task_context
        self.phone_number = os.getenv("USER_NUMBER")

        self.client = unify.AsyncUnify("o4-mini@openai")
        self.tools = methods_to_tool_dict(
            self._search_local_chat,
            self._ask_user_then_search,
        )
        if tools:
            self.tools = {
                **tools,
                **self.tools,
            }

        self.call_ready = asyncio.Event()
        self.call_ask_status = asyncio.Event()
        self.call_ask_status.set()

        self.status = "initiated"

    async def _join_meet_task(self):
        await _join_meet_call(self.meet_id, self.purpose, self.task_context)
        # give time to control browser and join meet
        await asyncio.sleep(30)
        self.call_ready.set()
        self.status = "started"

    @classmethod
    async def create(
        cls,
        meet_id: str,
        purpose: str,
        task_context: Dict[str, str] = None,
        tools: Dict[str, Any] = None,
    ) -> "GoogleMeet":
        instance = cls(meet_id, purpose, task_context, tools)
        await instance._join_meet_task()
        return instance

    async def _search_local_chat(self, question: str):
        """Search local chat window if a user response relevant to the question is found"""

        client = unify.AsyncUnify("o4-mini@openai")

        # Use shared prompt builder for local chat search
        client.set_system_message(
            build_local_chat_search_prompt(build_local_chat_history()),
        )

        handle = start_async_tool_use_loop(
            client,
            f"This is the user's question: {question}.",
            {},
            loop_id="meet_search_local_chat",
        )
        return await handle.result()

    async def _ask_user_then_search(self, question):
        """Ask user the question, then search for their response in local chat history."""
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": InterruptEvent().to_dict(),
            },
        ),
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Ask the user this question directly: {question}",
                ).to_dict(),
            },
        )

        # give time for utterance and response
        await asyncio.sleep(40)
        return await self._search_local_chat(question)

    async def ask(
        self,
        question: str,
        response_format: Optional[
            str | int | float | bool | BaseModel | EnumType
        ] = None,
    ) -> SteerableToolHandle:
        """
        Ask a question to the assistant.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()

        self.call_ask_status.clear()

        self.client.set_system_message(
            build_call_ask_prompt(self.tools, question),
        )

        is_enum = False
        if response_format:
            response_format, is_enum = _wrap_response_format(response_format)

        handle = start_async_tool_use_loop(
            self.client,
            f"The user is answering this question: {question}. Use available tools to get information of the user's answer.",
            self.tools,
            loop_id="meet_ask",
            response_format=response_format,
        )

        if is_enum:
            original_result = handle.result

            async def _wrap():
                answer = await original_result()
                answer = json.loads(answer)
                return answer["answer"]

            handle.result = _wrap

        async def _reset_call_ask_status():
            try:
                await handle.result()
            finally:
                self.call_ask_status.set()

        asyncio.create_task(_reset_call_ask_status())
        return handle

    async def interject(self, text: str) -> str:
        """
        Interject a message to the assistant for them to log as instruction for next response.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()
        self.call_ask_status.clear()

        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Instruction on next response: {text}",
                ).to_dict(),
            },
        )

        self.call_ask_status.set()
        return f"Message interjected to user: {text}"

    async def stop(self):
        """
        End the call.
        """
        await self.call_ready.wait()
        await self.call_ask_status.wait()
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": InterruptEvent().to_dict(),
            },
        ),
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="System",
                    content=f"Say goodbye to user.",
                ).to_dict(),
            },
        )
        await asyncio.sleep(15)
        await _publish_event(
            {
                "topic": self.phone_number,
                "to": "past",
                "event": PhoneCallStopEvent().to_dict(),
            },
        )
        self.status = "ended"

    def start_recording(self):
        pass

    def stop_recording(self):
        pass

    def result(self) -> str:
        return self.status

    def pause(self) -> str:
        return "Not applicable."

    def resume(self) -> str:
        return "Not applicable."

    def done(self) -> bool:
        return self.status == "ended"
