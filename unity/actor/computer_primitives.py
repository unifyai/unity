import os
import unify
from typing import Any
from pydantic import BaseModel
import inspect

from unity.controller.browser import Browser
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.secret_manager.secret_manager import SecretManager
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.handle import ConversationManagerHandle


class ComputerPrimitives:
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
        *,
        connect_now: bool = False,
    ):

        # Cache browser configuration for lazy initialization
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
        self._browser = None
        self._browser_mode = browser_mode
        self._browser_kwargs_map = browser_kwargs
        # Lazily create the Browser (and thus avoid connecting to agent-service) unless requested
        if connect_now:
            self._browser = Browser(
                mode=self._browser_mode,
                secret_manager=self.secret_manager,
                **self._browser_kwargs_map[self._browser_mode],
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
            from unity.singleton_registry import SingletonRegistry
            from unity.conversation_manager.conversation_manager import (
                ConversationManager,
            )

            cm_instance = SingletonRegistry.get(ConversationManager)
            self._conversation_manager = ConversationManagerHandle(
                event_broker=event_broker,
                conversation_id=assistant_id,
                contact_id=1,
                conversation_manager=cm_instance,
            )
        return self._conversation_manager

    def _setup_browser_methods(self):
        """Dynamically create tool methods without forcing an early backend connection."""
        from unity.controller.browser_backends import (
            LegacyBrowserBackend,
            MagnitudeBrowserBackend,
        )

        backend_class = (
            MagnitudeBrowserBackend
            if self._browser_mode == "magnitude"
            else LegacyBrowserBackend
        )

        def _make_lazy_wrapper(method_name: str, backend_class):
            async def wrapper(*args, **kwargs):
                backend_method = getattr(self.browser.backend, method_name)
                return await backend_method(*args, **kwargs)

            wrapper.__name__ = method_name
            wrapper.__qualname__ = method_name
            backend_method = getattr(backend_class, method_name, None)
            if backend_method and hasattr(backend_method, "__doc__"):
                wrapper.__doc__ = backend_method.__doc__
            return wrapper

        for method_name in [
            "act",
            "observe",
            "query",
            "navigate",
            "get_links",
            "get_content",
        ]:
            setattr(
                self,
                method_name,
                _make_lazy_wrapper(method_name, backend_class),
            )

    @property
    def browser(self) -> Browser:
        """Lazily initialize and return the Browser instance."""
        if self._browser is None:
            self._browser = Browser(
                mode=self._browser_mode,
                secret_manager=self.secret_manager,
                **self._browser_kwargs_map[self._browser_mode],
            )
        return self._browser

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
        decision = await computer_primitives.reason(
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

        structured_data = await computer_primitives.reason(
            request="Parse the user's first name, last name, and ID from the text.",
            context=raw_text,
            response_format=UserDetails
        )

        print(f"Welcome, {structured_data.first_name}! Your ID is {structured_data.user_id}.")
        # Expected Output: Welcome, Jane! Your ID is 4815162342.
        ```

        ### Example 3: Intelligent Question Formulation (Composition)
        Use `reason` to formulate a high-quality, disambiguating question for a user,
        then pass that question to a communication tool like `computer_primitives.conversation_manager.ask`.

        ```python
        user_request = "I need help with my account."

        # Use `reason` to generate the best question based on its look-ahead context.
        clarifying_question = await computer_primitives.reason(
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
        handle = await computer_primitives.conversation_manager.ask(clarifying_question)
        user_answer = await handle.result()
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
