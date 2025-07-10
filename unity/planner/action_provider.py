import asyncio
import os
import unify
from typing import Any
from pydantic import BaseModel
import inspect
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)

from unity.conversation_manager import comms_actions
from unity.conversation_manager.utils import publish_event
from unity.conversation_manager.events import PhoneUtteranceEvent, PhoneCallStopEvent
from unity.conversation_manager.prompt_builders import build_call_ask_prompt
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
            mode="hybrid",
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
            comms_actions._send_sms_message_via_number,
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
            comms_actions._send_email_via_address,
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
            comms_actions._send_whatsapp_message_via_number,
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
                cls.tools = methods_to_tool_dict(
                    self.contact_manager.ask,
                    self.transcript_manager.ask,
                    self.transcript_manager.summarize,
                    self.knowledge_manager.ask,
                    self.task_scheduler.ask,
                    # comms_actions._send_email_via_address,
                    # comms_actions._send_sms_message_via_number,
                    # new_actions._send_whatsapp_message_via_number,
                )

                cls.call_ready = asyncio.Event()
                cls.call_ask_status = asyncio.Event()
                cls.call_ask_status.set()

                async def do_call():
                    await comms_actions._start_call(
                        os.getenv("ASSISTANT_NUMBER"),
                        phone_number,
                        purpose,
                    )
                    # give time to start call and complete greeting
                    await asyncio.sleep(15)
                    cls.call_ready.set()

                asyncio.create_task(do_call())
                cls.status = "initiated"

            async def ask(cls, question: str) -> SteerableToolHandle:
                """
                Ask a question to the assistant.
                """
                await cls.call_ready.wait()
                await cls.call_ask_status.wait()

                cls.call_ask_status.clear()
                await publish_event(
                    {
                        "topic": cls.phone_number,
                        "to": "pending",
                        "event": PhoneUtteranceEvent(
                            role="User",
                            content=f"Ask the user this question directly: {question}",
                        ).to_dict(),
                    },
                )

                cls.client.set_system_message(
                    build_call_ask_prompt(cls.tools, question),
                )
                handle = start_async_tool_use_loop(
                    cls.client,
                    f"The user is answering the question: {question}. Use available tools to get information of the user's answer.",
                    cls.tools,
                    loop_id="call_ask",
                )

                async def _reset_call_ask_status():
                    try:
                        await handle.result()
                    finally:
                        cls.call_ask_status.set()

                asyncio.create_task(_reset_call_ask_status())
                return handle

            async def interject(cls, text: str) -> str:
                """
                Interject a message to the assistant for them to speak it to the user.
                """
                await cls.call_ready.wait()
                await cls.call_ask_status.wait()

                cls.call_ask_status.clear()
                await publish_event(
                    {
                        "topic": cls.phone_number,
                        "to": "pending",
                        "event": PhoneUtteranceEvent(
                            role="User",
                            content=f"Speak this content to the user directly: {text}",
                        ).to_dict(),
                    },
                )

                # give time for utterance after event publish
                await asyncio.sleep(8)
                cls.call_ask_status.set()
                return f"Message interjected to user: {text}"

            async def stop(cls):
                """
                End the call.
                """
                await cls.call_ready.wait()
                await cls.call_ask_status.wait()
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
    async def browser_act(self, instruction: str, expectation: str) -> str:
        """
        Performs a single, atomic high-level action in the browser and verifies its outcome.
        This tool is for discrete, state-changing operations like a single click, a typing sequence, or a navigation event.

        Args:
            instruction (str): The natural-language instruction for the action.
                            **IMPORTANT**: This must be a single command. Do not chain multiple actions
                            together (e.g., "click login and type username").
            expectation (str): A clear, verifiable description of the expected state of the page *after*
                            the action is successfully completed.

        Examples:
            # Good Example (Single Action)
            - instruction: "Click the 'Login' button"
            expectation: "The URL should now contain '/login'."

            # Good Example (Single Action)
            - instruction: "Type 'hello world' into the search bar with ID 'search-input'"
            expectation: "The search bar should contain the text 'hello world'."

            # Bad Example (Chained Actions - Do Not Do This)
            - instruction: "Click the login button and then enter 'my_user' into the username field."
        """
        return await self.browser.act(
            instruction, expectation=expectation, multi_step_mode=True
        )

    async def browser_observe(self, query: str, response_format: Any = str) -> Any:
        """
        Asks a question about the current state of the browser page and returns the answer.
        This tool is for read-only operations to gather information without changing the page state.
        It uses an LLM to analyze a screenshot and the page's DOM to answer the query.

        Args:
            query: The natural-language question to ask about the page.
            response_format: Optional. A Pydantic model to structure the output. If provided, the LLM will return a JSON object matching the model.

        Examples:
        - "What is the title of the page?"
        - "Is there a button with the text 'Submit' visible on the screen?"
        - "What are the headlines of the articles in the main content area?"
        """
        return await self.browser.observe(query, response_format=response_format)

    async def browser_multi_step(self, description: str) -> SteerableToolHandle:
        """
        Performs a complex, sequential browser task that may require multiple steps.
        Use this for high-level goals like "Log into my account" or "Find the latest blog post and summarize it."
        This tool is more powerful than `act` for tasks that are not single-step.
        It returns a handle to a sub-agent that will execute the task.
        """
        return await self.browser.multi_step(description)

    # async def browser_start_recording(self):
    #     """Alias for browser.start_recording."""
    #     return self.browser.start_recording()

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
