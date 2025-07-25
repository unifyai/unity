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
    Provides a library of high-level, agentic actions for the HierarchicalPlanner.
    Each public method is a tool that the planner can incorporate into its generated code.
    """

    def __init__(
        self,
        session_connect_url: str | None = None,
        headless: bool = False,
        browser_mode: str = "legacy",
        controller_mode: str = "hybrid",
    ):

        browser_kwargs = {
            "legacy": {
                "session_connect_url": session_connect_url,
                "headless": headless,
                "controller_mode": controller_mode,
            },
            "magnitude": {
                "headless": headless,
            },
        }
        self.browser = Browser(mode=browser_mode, **browser_kwargs[browser_mode])
        self._setup_browser_methods()

        self.contact_manager = ContactManager()
        self.transcript_manager = TranscriptManager(
            contact_manager=self.contact_manager,
        )
        self.knowledge_manager = KnowledgeManager()

        from unity.task_scheduler.task_scheduler import TaskScheduler

        self.task_scheduler = TaskScheduler()

    def _setup_browser_methods(self):
        """Dynamically create tool methods and assign backend docstrings."""
        methods_to_proxy = {
            "browser_act": self.browser.backend.act,
            "browser_observe": self.browser.backend.observe,
            "browser_navigate": self.browser.backend.navigate,
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

    # --- Browser Actions ---
    # async def browser_multi_step(self, description: str) -> SteerableToolHandle:
    #     """
    #     Performs a complex, sequential browser task that may require multiple steps.
    #     Use this for high-level goals like "Log into my account" or "Find the latest blog post and summarize it."
    #     This tool is more powerful than `act` for tasks that are not single-step.
    #     It returns a handle to a sub-agent that will execute the task.
    #     """
    #     return await self.browser.multi_step(description)

    # async def browser_start_recording(self):
    #     """Alias for browser.start_recording."""
    #     return self.browser.start_recording()

    # TODO: move this to the FM
    # async def scroll_until_visible(
    #     self,
    #     element_description: str,
    #     direction: str = "down",
    #     max_retries: int = 5,
    # ) -> str:
    #     """
    #     Scrolls the page in a specified direction until a target element is visible.

    #     This is a robust tool for finding elements that may be off-screen. It is generally
    #     preferable to writing manual scroll loops in a plan.

    #     Args:
    #         element_description (str): A clear, natural-language description of the target
    #                                  element to find (e.g., "the 'Submit' button",
    #                                  "the footer section containing 'About Us'").
    #         direction (str, optional): The direction to scroll. Can be "down" or "up".
    #                                  Defaults to "down".
    #         max_retries (int, optional): The maximum number of times to scroll before giving up.
    #                                    Defaults to 5.

    #     Returns:
    #         str: A status message indicating success or failure.

    #     Example:
    #         await action_provider.scroll_until_visible(
    #             element_description="the 'Terms of Service' link in the footer"
    #         )
    #     """

    #     class ElementVisibility(BaseModel):
    #         is_visible: bool = Field(
    #             description="True if the element is visible on the screen, False otherwise.",
    #         )
    #         reason: str = Field(description="The reason for the visibility status.")

    #     for i in range(max_retries):
    #         # First, check if the element is already visible.
    #         visibility_status = await self.browser.observe(
    #             f"Is'{element_description}' currently visible on the screen?",
    #             response_format=ElementVisibility,
    #         )

    #         if visibility_status.is_visible:
    #             print(f"Success: Element '{element_description}' is now visible.")
    #             return f"Success: Element '{element_description}' is now visible."

    #         print(f"Continue scrolling. Reason: {visibility_status.reason}")
    #         # If not visible, perform the scroll action.
    #         await self.browser.act(
    #             f"Scroll {direction} slightly",
    #             expectation=f"The page should scroll {direction}.",
    #         )
    #         await asyncio.sleep(1)

    #     # If the loop finishes without finding the element, return a failure message.
    #     return f"Failure: Could not find element '{element_description}' after {max_retries} scrolls."

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
