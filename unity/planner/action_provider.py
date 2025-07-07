import asyncio
import unify
from typing import Any
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

from unity.service import new_actions
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
        Understands a natural language request to send an SMS. This tool orchestrates a multi-step process:
        1. It uses the ContactManager to find the recipient's phone number based on the description.
        2. It uses other tools to gather necessary information and draft a precise message.
        3. It then calls the low-level `_send_sms_message_via_number` to finally send the message.
        You should provide a clear and complete description, e.g., "Send a text to John Doe letting him know his appointment is confirmed for 3 PM tomorrow."
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
        Understands a natural language request to send an email. This tool orchestrates a multi-step process:
        1. It uses the ContactManager to find the recipient's email address based on the description.
        2. It uses other tools like the KnowledgeManager or TranscriptManager to draft the email content.
        3. It then calls the low-level `_send_email_via_address` to send the email.
        You should provide a clear and complete description, e.g., "Email Jane Doe to follow up on our conversation from yesterday about the project proposal."
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
        Understands a natural language request to send a WhatsApp message. This tool orchestrates a multi-step process:
        1. It uses the ContactManager to find the recipient's WhatsApp-enabled phone number.
        2. It drafts a message based on the provided description and context.
        3. It calls the low-level `_send_whatsapp_message_via_number` to dispatch the message.
        You should provide a clear and complete description, e.g., "Send a WhatsApp message to the team group to remind them of the 10 AM meeting."
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
        Initiates an outbound phone call to a specified number for a given purpose.
        This function returns a steerable 'Call' handle that allows for interactive, real-time conversation.
        Args:
            phone_number: The destination phone number to call.
            purpose: A clear and concise description of why the call is being made. This purpose will be used to guide the conversation.
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
        """
        Performs a **single**, high-level action in the browser based on a natural language instruction.
        This tool should be used for actions that change the state of the page, such as clicking, typing, scrolling, or navigating.
        It uses an LLM to translate your instruction into a precise, low-level browser command.

        Examples:
        - "Click the 'Login' button"
        - "Type 'hello world' into the search bar with ID 'search-input'"
        - "Scroll down to the footer"
        - "Navigate to https://unify.ai"
        """
        return await self.browser.act(instruction)

    async def browser_observe(self, query: str) -> Any:
        """
        Asks a question about the current state of the browser page and returns the answer.
        This tool is for read-only operations to gather information without changing the page state.
        It uses an LLM to analyze a screenshot and the page's DOM to answer the query.

        Examples:
        - "What is the title of the page?"
        - "Is there a button with the text 'Submit' visible on the screen?"
        - "What are the headlines of the articles in the main content area?"
        """
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
