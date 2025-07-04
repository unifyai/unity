import asyncio
import unify
from typing import Any
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

from unity.service import new_actions, start, stop
from unity.service.utils import publish_event
from unity.service.events import PhoneUtteranceEvent, PhoneCallStopEvent
from unity.controller.browser import Browser
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


class ActionProvider:
    """
    Provides a library of high-level, agentic actions for the HierarchicalPlanner.
    Each public method is a tool that the planner can incorporate into its generated code.
    """

    def __init__(self, session_connect_url: str | None = None, headless: bool = False):
        self.browser = Browser(
            session_connect_url=session_connect_url,
            headless=headless,
        )
        self.contact_manager = ContactManager()
        self.transcript_manager = TranscriptManager(
            contact_manager=self.contact_manager,
        )
        self.knowledge_manager = KnowledgeManager()

        from unity.task_scheduler.task_scheduler import TaskScheduler

        self.task_scheduler = TaskScheduler()

    # --- Communication Actions ---

    async def send_sms_message(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        """
        Understands a natural language request to send an SMS, finds the contact,
        drafts the message, and sends it.
        """
        client = unify.AsyncUnify("o4-mini@openai")
        client.set_system_message(
            "Your task is to send an SMS message. First, use the ContactManager to find the recipient's phone number. Then, draft a message. Finally, use the `_send_sms_message_via_number` tool to send it.",
        )
        tools = methods_to_tool_dict(
            self.contact_manager.ask,
            self.transcript_manager.ask,
            self.knowledge_manager.ask,
            new_actions._send_sms_message_via_number,
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
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        """
        Understands a natural language request to send an email.
        (Implementation would be similar to send_sms_message)
        """
        client = unify.AsyncUnify("o4-mini@openai")
        client.set_system_message(
            "Your task is to send an email. First, use the ContactManager to find the recipient's email address. Then, draft a message. Finally, use the `_send_email_via_address` tool to send it.",
        )
        tools = methods_to_tool_dict(
            self.contact_manager.ask,
            self.transcript_manager.ask,
            self.knowledge_manager.ask,
            new_actions._send_email_via_address,
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

    async def send_whatsapp_message(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        """
        Understands a natural language request to send a WhatsApp message.
        (Implementation would be similar to send_sms_message)
        """
        client = unify.AsyncUnify("o4-mini@openai")
        client.set_system_message(
            "Your task is to send a WhatsApp message. First, use the ContactManager to find the recipient's phone number. Then, draft a message. Finally, use the `_send_whatsapp_message_via_number` tool to send it.",
        )
        tools = methods_to_tool_dict(
            self.contact_manager.ask,
            self.transcript_manager.ask,
            self.knowledge_manager.ask,
            new_actions._send_whatsapp_message_via_number,
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

    def start_call(self, phone_number: str, purpose: str) -> SteerableToolHandle:
        """
        Initiates a call and returns a steerable handle for interaction.
        """

        class Call(SteerableToolHandle):
            def __init__(cls, phone_number: str, purpose: str):
                """
                Starts a new phone call session and exposes the steerable methods
                """
                cls.phone_number = phone_number
                cls.purpose = purpose

                cls.client = unify.AsyncUnify("o4-mini@openai")
                cls.client.set_system_message(
                    f"You are a helpful assistant. You are calling {cls.phone_number} for {cls.purpose}.",
                )
                cls.tools = methods_to_tool_dict(
                    self.contact_manager.ask,
                    self.transcript_manager.ask,
                    self.transcript_manager.summarize,
                    self.knowledge_manager.ask,
                    self.task_scheduler.ask,
                    new_actions._send_email_via_address,
                    new_actions._send_sms_message_via_number,
                    new_actions._send_whatsapp_message_via_number,
                )

                start()
                asyncio.create_task(new_actions._start_call(phone_number, purpose))
                cls.status = "started"

            async def ask(cls, question: str) -> SteerableToolHandle:
                """
                Ask a question to the assistant.
                """
                await publish_event(
                    {
                        "topic": cls.phone_number,
                        "to": "pending",
                        "event": PhoneUtteranceEvent(
                            role="User",
                            content=question,
                        ).to_dict(),
                    },
                )
                return start_async_tool_use_loop(
                    cls.client,
                    question,
                    cls.tools,
                    loop_id="call",
                )

            async def interject(cls, text: str) -> str:
                """
                Interject a message to the assistant for them to speak it to the user.
                """
                await publish_event(
                    {
                        "topic": cls.phone_number,
                        "to": "pending",
                        "event": PhoneUtteranceEvent(
                            role="User",
                            content=text,
                        ).to_dict(),
                    },
                )
                return "Acknowledged."

            async def stop(cls):
                """
                End the call.
                """
                await publish_event(
                    {
                        "topic": cls.phone_number,
                        "to": "past",
                        "event": PhoneCallStopEvent().to_dict(),
                    },
                )
                await asyncio.sleep(5)
                stop()
                cls.status = "ended"

            def result(cls) -> str:
                return cls.status

            def pause(cls) -> str:
                return "Not applicable."

            def resume(cls) -> str:
                return "Not applicable."

            def done(cls) -> bool:
                return cls.status == "ended"

        return Call(phone_number, purpose)

    # --- Browser Actions ---
    async def browser_act(self, instruction: str) -> str:
        """Alias for browser.act to be exposed to the planner."""
        return await self.browser.act(instruction)

    async def browser_observe(self, query: str) -> Any:
        """Alias for browser.observe to be exposed to the planner."""
        return await self.browser.observe(query)

    # TODO: uncomment these once implemented
    # async def browser_reason(self, query: str) -> str:
    #     """Alias for browser.reason to be exposed to the planner."""
    #     return await self.browser.reason(query)

    # def browser_multi_step(self, description: str) -> SteerableToolHandle:
    #     """Alias for browser.multi_step to be exposed to the planner."""
    #     return self.browser.multi_step(description)

    # async def browser_start_recording(self):
    #     """Alias for browser.start_recording."""
    #     return self.browser.start_recording()
