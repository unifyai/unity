import unify
from typing import Any
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

from unity.service import comms_actions
from unity.controller.browser import Browser
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.task_scheduler.task_scheduler import TaskScheduler


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
            "Your task is to send an SMS message. First, use the ContactManager to find the recipient's phone number. Then, draft a message. Finally, use the `send_sms` tool to send it.",
        )
        tools = methods_to_tool_dict(
            self.contact_manager.ask,
            self.transcript_manager.ask,
            self.knowledge_manager.ask,
            comms_actions.send_sms,
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

    # TODO: uncomment these once implemented
    # async def send_email(
    #     self,
    #     description: str,
    #     parent_chat_context: list[dict] | None = None,
    # ) -> SteerableToolHandle:
    #     """
    #     Understands a natural language request to send an email.
    #     (Implementation would be similar to send_sms_message)
    #     """
    #     # TODO: Implement this using the same pattern as send_sms_message
    #     print(f"Placeholder for sending email: {description}")

    # async def send_whatsapp_message(
    #     self,
    #     description: str,
    #     parent_chat_context: list[dict] | None = None,
    # ) -> SteerableToolHandle:
    #     """
    #     Understands a natural language request to send a WhatsApp message.
    #     (Implementation would be similar to send_sms_message)
    #     """
    #     # TODO: Implement this using the same pattern as send_sms_message
    #     print(f"Placeholder for sending WhatsApp message: {description}")

    # def start_call(self, phone_number: str, purpose: str) -> SteerableToolHandle:
    #     """
    #     Initiates a call and returns a steerable handle for interaction.
    #     """
    #     # TODO: Implement this by returning an instance of new Call handle.
    #     print(f"Placeholder for starting call to {phone_number} for purpose: {purpose}")

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
