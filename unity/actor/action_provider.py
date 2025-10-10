import functools
import os
import unify
from typing import Any, Dict
from pydantic import BaseModel
import inspect
from unity.common.async_tool_loop import (
    SteerableToolHandle,
)

from unity.conversation_manager import comms_actions
from unity.controller.browser import Browser
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.llm_helpers import methods_to_tool_dict
from unity.secret_manager.secret_manager import SecretManager
from unity.conversation_manager_2.event_broker import get_event_broker
from unity.conversation_manager_2.handle import ConversationManagerHandle


class ActionProvider:
    """
    Provides a library of high-level, agentic actions for the HierarchicalActor.
    Each public method is a tool that the actor can incorporate into its generated code.
    """

    def __init__(
        self,
        session_connect_url: str | None = None,
        headless: bool = False,
        browser_mode: str = "magnitude",
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

        self._secret_manager = None
        self.browser = Browser(
            mode=browser_mode,
            secret_manager=self.secret_manager,
            **browser_kwargs[browser_mode],
        )
        self._setup_browser_methods()

        self._contact_manager = None
        self._transcript_manager = None
        self._knowledge_manager = None
        self._task_scheduler = None
        self._conversation_manager = None

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

    @property
    def secret_manager(self):
        """Lazily initialize and return the SecretManager."""
        if self._secret_manager is None:
            self._secret_manager = SecretManager()
        return self._secret_manager

    @property
    def conversation_manager(self) -> ConversationManagerHandle:
        """Lazily initialize and return the ConversationManagerHandle."""
        if self._conversation_manager is None:
            event_broker = get_event_broker()
            assistant_id = os.environ.get("ASSISTANT_ID")
            if not assistant_id:
                raise RuntimeError(
                    "ASSISTANT_ID environment variable is not set. "
                    "Cannot create ConversationManagerHandle.",
                )
            self._conversation_manager = ConversationManagerHandle(
                event_broker=event_broker,
                conversation_id=assistant_id,
                contact_id=1,
            )
        return self._conversation_manager

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
        Performs general-purpose reasoning with automatic access to the live call stack.

        This powerful tool is designed for complex, stateless tasks like analysis,
        classification, strategic decision-making, and data transformation. It is
        automatically provided with a "scoped context" of the running plan, including the
        source code of the parent, current, and potential child functions, enabling it
        to make highly informed decisions.

        ### Example 1: Strategic Decision-Making (Look-Ahead)
        Use `reason` to analyze an ambiguous situation and decide which function to call next.
        It can "look ahead" by inspecting the code of potential child functions.

        ```python
        from pydantic import BaseModel, Field
        from typing import Literal

        class SupportCategory(BaseModel):
            category: Literal["technical", "billing", "account"]
            justification: str = Field(description="A brief explanation for the chosen category.")

        SupportCategory.model_rebuild()

        user_message = "I can't access my dashboard and my last payment didn't go through."

        # The proxy automatically provides the source for `handle_technical_support`, etc.
        decision = await action_provider.reason(
            request=(
                "Based on the user's message, I need to choose the correct support category. "
                "Analyze the available child functions in the provided call stack context "
                "to determine the most appropriate category."
            ),
            context=f"User's message: '{user_message}'",
            response_format=SupportCategory
        )

        if decision.category == "technical":
            await handle_technical_support()
        elif decision.category == "billing":
            await handle_billing_inquiry()
        else:
            await handle_account_management()
        ```

        ### Example 2: Data Transformation and Structuring
        Use `reason` to parse unstructured text into a clean, Pydantic model.

        ```python
        from pydantic import BaseModel, Field

        class UserDetails(BaseModel):
            first_name: str
            last_name: str
            user_id: int = Field(description="The user's numerical ID.")

        UserDetails.model_rebuild()

        raw_text = "The user is Jane Doe, ID number 4815162342."

        structured_data = await action_provider.reason(
            request="Parse the user's first name, last name, and ID from the text.",
            context=raw_text,
            response_format=UserDetails
        )

        print(f"Welcome, {structured_data.first_name}! Your ID is {structured_data.user_id}.")
        # Expected Output: Welcome, Jane! Your ID is 4815162342.
        ```

        ### Example 3: Intelligent Question Formulation (Composition)
        Use `reason` to formulate a high-quality, disambiguating question for a user,
        then pass that question to a communication tool like `action_provider.conversation_manager.ask`.

        ```python
        user_request = "I need help with my account."

        # Use `reason` to generate the best question based on its look-ahead context.
        clarifying_question = await action_provider.reason(
            request=(
                "The user's request is ambiguous. Based on the child functions available "
                "in my context (e.g., `reset_password`, `update_billing`, `close_account`), "
                "formulate a single, clear question to ask the user to determine "
                "which path to take."
            ),
            context=f"User's request: '{user_request}'"
        )

        # clarifying_question might be:
        # "I can help with that! Are you looking to reset your password, update your billing "
        # "information, or close your account?"

        # Now, use the generated question to get the required information.
        handle = await action_provider.conversation_manager.ask(clarifying_question)
        user_answer = await handle.wait()
        ```

        Args:
            request: The core instruction for the LLM (e.g., "Analyze the user's intent.").
            context: The primary text content to be analyzed. The call stack context is
                     automatically prepended to this by the actor.
            response_format: Optional. A Pydantic model to structure the output. Highly recommended.

        Returns:
            The processed text or a Pydantic object, depending on `response_format`.
        """
        client = unify.AsyncUnify("gemini-2.5-pro@vertex-ai")
        system_message = (
            f"{request}\n\n"
            "### CONTEXT\n"
            "Use the following context, including the provided call stack information, to inform your reasoning.\n\n"
            f"{context}"
        )
        client.set_system_message(system_message)

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            client.set_response_format(response_format)
            raw_response = await client.generate("")
            return response_format.model_validate_json(raw_response)
        else:
            return await client.generate("")
