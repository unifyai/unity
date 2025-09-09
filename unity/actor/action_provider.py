import functools
import os
import unify
from typing import Any, Dict
from pydantic import BaseModel
import inspect
from unity.common.llm_helpers import (
    SteerableToolHandle,
)

from unity.conversation_manager import comms_actions
from unity.controller.browser import Browser
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.llm_helpers import methods_to_tool_dict


class ActionProvider:
    """
    Provides a library of high-level, agentic actions for the HierarchicalActor.
    Each public method is a tool that the actor can incorporate into its generated code.
    """

    def __init__(
        self,
        session_connect_url: str | None = None,
        headless: bool = False,
        browser_mode: str = "legacy",
        controller_mode: str = "hybrid",
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
    ):

        browser_kwargs = {
            "legacy": {
                "session_connect_url": session_connect_url,
                "headless": headless,
                "controller_mode": controller_mode,
            },
            "magnitude": {
                "headless": headless,
                "agent_mode": agent_mode,
                "agent_server_url": agent_server_url,
            },
        }

        self.browser = Browser(mode=browser_mode, **browser_kwargs[browser_mode])
        self._setup_browser_methods()

        self._contact_manager = None
        self._transcript_manager = None
        self._knowledge_manager = None
        self._task_scheduler = None

    @property
    def contact_manager(self):
        """Lazily initialize and return the ContactManager."""
        if self._contact_manager is None:
            self._contact_manager = ContactManager()
        return self._contact_manager

    @property
    def transcript_manager(self):
        """Lazily initialize and return the TranscriptManager."""
        if self._transcript_manager is None:
            self._transcript_manager = TranscriptManager(
                contact_manager=self.contact_manager,
            )
        return self._transcript_manager

    @property
    def knowledge_manager(self):
        """Lazily initialize and return the KnowledgeManager."""
        if self._knowledge_manager is None:
            self._knowledge_manager = KnowledgeManager()
        return self._knowledge_manager

    @property
    def task_scheduler(self):
        """Lazily initialize and return the TaskScheduler."""
        if self._task_scheduler is None:
            from unity.task_scheduler.task_scheduler import TaskScheduler

            self._task_scheduler = TaskScheduler()
        return self._task_scheduler

    def _setup_browser_methods(self):
        """Dynamically create tool methods and assign backend docstrings."""
        methods_to_proxy = {
            "act": self.browser.backend.act,
            "observe": self.browser.backend.observe,
            "query": self.browser.backend.query,
            "navigate": self.browser.backend.navigate,
        }

        for method_name, backend_method in methods_to_proxy.items():
            # Create a simple wrapper that preserves the backend method's behavior and docstring
            @functools.wraps(backend_method)
            async def wrapper(*args, _backend_method=backend_method, **kwargs):
                return await _backend_method(*args, **kwargs)

            # Preserve the original docstring
            wrapper.__doc__ = backend_method.__doc__
            setattr(self, method_name, wrapper)

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
        return await comms_actions.send_sms_message(description, parent_chat_context)

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
        return await comms_actions.send_email(description, parent_chat_context)

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
        return await comms_actions.send_whatsapp_message(
            description,
            parent_chat_context,
        )

    def start_call(
        self,
        phone_number: str,
        purpose: str,
        task_context: Dict[str, str] = None,
    ) -> SteerableToolHandle:
        """
        Initiates an outbound phone call to a specified number for a given purpose.
        This function returns a steerable 'Call' handle that allows for interactive, real-time conversation.
        Args:
            phone_number: The destination phone number to call.
            purpose: A clear and concise description of why the call is being made. This purpose will be used to guide the conversation.
            task_context: The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        return comms_actions.Call.create(
            phone_number,
            purpose,
            task_context,
            tools=methods_to_tool_dict(
                self.contact_manager.ask,
                self.transcript_manager.ask,
                self.knowledge_manager.ask,
                self.task_scheduler.ask,
            ),
        )

    def join_meet(
        self,
        meet_id: str,
        purpose: str,
        task_context: Dict[str, str] = None,
    ):
        """
        Joins a Google Meet call.
        Args:
            meet_id: The ID of the Google Meet call.
            purpose: A clear and concise description of why the call is being made. This purpose will be used to guide the conversation.
            task_context: The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        return comms_actions.GoogleMeet.create(
            meet_id,
            purpose,
            task_context,
            tools=methods_to_tool_dict(
                self.contact_manager.ask,
                self.transcript_manager.ask,
                self.knowledge_manager.ask,
                self.task_scheduler.ask,
            ),
        )

    # --- Generic Reasoning Action ---
    async def reason(
        self,
        request: str,
        context: str,
        response_format: Any = str,
    ) -> Any:
        """
        Performs general-purpose reasoning or analysis on provided text.
        This tool is for stateless tasks like summarizing, translating, classifying, or extracting information from the given context.

        Args:
            request: The core instruction for the LLM (e.g., "Summarize this text.", "Classify the sentiment.").
            context: The text content to be analyzed.
            response_format: Optional. A Pydantic model to structure the output.

        Returns:
            The processed text or a Pydantic object, depending on `response_format`.
        """
        client = unify.AsyncUnify(os.environ.get("UNIFY_MODEL", "gpt-4o-mini@openai"))
        client.set_system_message(request)

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            client.set_response_format(response_format)
            raw_response = await client.generate(context)
            return response_format.model_validate_json(raw_response)
        else:
            return await client.generate(context)
