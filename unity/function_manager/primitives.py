"""
Registry and introspection utilities for action primitives.

Action primitives are public methods from state managers (ContactManager.ask,
TaskScheduler.execute, etc.) that are exposed to the Actor for direct invocation
or composition within generated Python code.

Primitives are stored in a dedicated `Functions/Primitives` context, separate from
user-defined functions in `Functions/Compositional`. Each primitive receives a
stable `function_id` based on its position in `PRIMITIVE_SOURCES` - these IDs are
consistent across all users. User-defined functions have their own auto-incrementing
ID sequence in their separate context, so the two never collide.

IMPORTANT: PRIMITIVE_SOURCES is append-only. Never reorder or remove entries -
only add new primitives at the end. This ensures stable IDs across upgrades.

This module provides:
- `ComputerPrimitives` - Browser/desktop control and reasoning capabilities
- `Primitives` - Runtime interface for accessing all primitives from executed functions
- Registry functions for syncing primitives to the database
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel

from unity.common.llm_client import new_llm_client

if TYPE_CHECKING:
    from unity.contact_manager.contact_manager import ContactManager
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.secret_manager.secret_manager import SecretManager
    from unity.guidance_manager.guidance_manager import GuidanceManager
    from unity.web_searcher.web_searcher import WebSearcher
    from unity.controller.browser import Browser
    from unity.file_manager.managers.file_manager import FileManager

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# ComputerPrimitives - Browser/Desktop Control
# ────────────────────────────────────────────────────────────────────────────


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
            from unity.controller.browser import Browser

            self._browser = Browser(
                mode=self._browser_mode,
                secret_manager=self.secret_manager,
                **self._browser_kwargs_map[self._browser_mode],
            )
        self._setup_browser_methods()

    @property
    def secret_manager(self):
        """Lazily initialize and return the SecretManager via ManagerRegistry."""
        if self._secret_manager is None:
            from unity.manager_registry import ManagerRegistry

            self._secret_manager = ManagerRegistry.get_secret_manager()
        return self._secret_manager

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
    def browser(self) -> "Browser":
        """Lazily initialize and return the Browser instance."""
        if self._browser is None:
            from unity.controller.browser import Browser

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
        then pass that question to `request_clarification` (injected by the Actor at runtime).

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
        user_answer = await request_clarification(clarifying_question)
        ```

        Args:
            request: The core instruction for the LLM (e.g., "Analyze the user's intent.").
            context: The primary text content to be analyzed. The call stack context is
                     automatically prepended to this by the actor.
            response_format: Optional. A Pydantic model to structure the output. Highly recommended.

        Returns:
            The processed text or a Pydantic object, depending on `response_format`.
        """
        client = new_llm_client(
            "gemini-2.5-pro@vertex-ai",
            reasoning_effort=None,
            service_tier=None,
        )
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


# ────────────────────────────────────────────────────────────────────────────
# Primitive Registry
# ────────────────────────────────────────────────────────────────────────────


# Registry of (module.ClassName, [method_names]) to expose as primitives.
# Each entry maps a fully-qualified class path to the list of public methods
# that should be available as primitives.
PRIMITIVE_SOURCES: List[Tuple[str, List[str]]] = [
    # State managers - public API methods
    (
        "unity.contact_manager.contact_manager.ContactManager",
        ["ask", "update"],
    ),
    (
        "unity.transcript_manager.transcript_manager.TranscriptManager",
        ["ask"],
    ),
    (
        "unity.knowledge_manager.knowledge_manager.KnowledgeManager",
        ["ask", "update", "refactor"],
    ),
    (
        "unity.task_scheduler.task_scheduler.TaskScheduler",
        ["ask", "update", "execute"],
    ),
    (
        "unity.secret_manager.secret_manager.SecretManager",
        ["ask", "update"],
    ),
    (
        "unity.guidance_manager.guidance_manager.GuidanceManager",
        ["ask", "update"],
    ),
    (
        "unity.web_searcher.web_searcher.WebSearcher",
        ["ask"],
    ),
    # ComputerPrimitives browser/reasoning methods (now in this module)
    (
        "unity.function_manager.primitives.ComputerPrimitives",
        ["navigate", "act", "observe", "query", "reason"],
    ),
    # FileManager - file operations and data access
    (
        "unity.file_manager.managers.file_manager.FileManager",
        [
            "ask",
            "ask_about_file",
            "reduce",
            "filter_files",
            "search_files",
            "visualize",
        ],
    ),
]


def _import_class(class_path: str) -> Optional[type]:
    """
    Dynamically import a class from its fully-qualified path.

    Args:
        class_path: e.g. "unity.contact_manager.contact_manager.ContactManager"

    Returns:
        The class object, or None if import fails.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import {class_path}: {e}")
        return None


def _get_method_metadata(
    cls: type,
    method_name: str,
    class_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract metadata (signature, docstring) from a class method.

    Handles functools.wraps by looking for __wrapped__ attribute.

    Args:
        cls: The class containing the method.
        method_name: Name of the method to introspect.
        class_name: Short class name for building qualified name.

    Returns:
        Primitive metadata dict, or None if method not found.
    """
    method = getattr(cls, method_name, None)
    if method is None:
        return None

    # Unwrap functools.wraps to get the original function's metadata
    fn = method
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    # Build qualified name: "ContactManager.ask"
    qualified_name = f"{class_name}.{method_name}"

    try:
        signature = str(inspect.signature(fn))
    except (ValueError, TypeError):
        signature = "(...)"

    docstring = inspect.getdoc(fn) or ""

    return {
        "name": qualified_name,
        "argspec": signature,
        "docstring": docstring,
        "embedding_text": (
            f"Function Name: {qualified_name}\n"
            f"Signature: {signature}\n"
            f"Docstring: {docstring}"
        ),
        "implementation": None,
        "is_primitive": True,
        "calls": [],
        "precondition": None,
        "verify": False,
        "guidance_ids": [],
        # Store class path for execution routing
        "primitive_class": cls.__module__ + "." + cls.__name__,
        "primitive_method": method_name,
    }


def collect_primitives() -> Dict[str, Dict[str, Any]]:
    """
    Introspect all registered primitives and return their metadata with stable IDs.

    Iterates through PRIMITIVE_SOURCES in order, imports each class, and extracts
    signature and docstring information for each registered method. Each primitive
    receives an explicit `function_id` based on its sequential position.

    The IDs are stable across all users because they depend only on the order
    in PRIMITIVE_SOURCES, which is append-only.

    Returns:
        Dict mapping qualified_name (e.g. "ContactManager.ask") to primitive
        metadata suitable for insertion into the Functions/Primitives context.
    """
    primitives: Dict[str, Dict[str, Any]] = {}
    next_id = 0

    for class_path, method_names in PRIMITIVE_SOURCES:
        cls = _import_class(class_path)
        if cls is None:
            # Still increment IDs even for failed imports to maintain stable ordering
            next_id += len(method_names)
            continue

        class_name = class_path.rsplit(".", 1)[1]

        for method_name in method_names:
            metadata = _get_method_metadata(cls, method_name, class_name)
            if metadata is not None:
                metadata["function_id"] = next_id
                primitives[metadata["name"]] = metadata
            next_id += 1

    logger.debug(f"Collected {len(primitives)} primitives with IDs 0-{next_id - 1}")
    return primitives


def compute_primitives_hash(primitives: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute a stable hash of all primitive signatures.

    Used to detect when primitives have changed (docstrings updated, methods
    added/removed) and a sync is needed.

    Args:
        primitives: Dict from collect_primitives().

    Returns:
        16-character hex hash string.
    """
    parts = []
    for name in sorted(primitives.keys()):
        p = primitives[name]
        # Include name, signature, and docstring in hash
        parts.append(f"{name}|{p['argspec']}|{p['docstring']}")

    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Mapping from class names to ManagerRegistry getter method names
_CLASS_TO_GETTER: Dict[str, str] = {
    "ContactManager": "get_contact_manager",
    "TranscriptManager": "get_transcript_manager",
    "KnowledgeManager": "get_knowledge_manager",
    "TaskScheduler": "get_task_scheduler",
    "SecretManager": "get_secret_manager",
    "GuidanceManager": "get_guidance_manager",
    "WebSearcher": "get_web_searcher",
    "ImageManager": "get_image_manager",
    "FileManager": "get_file_manager",
}


def get_primitive_callable(
    primitive_data: Dict[str, Any],
    computer_primitives: Optional[ComputerPrimitives] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    For ComputerPrimitives methods, uses the provided computer_primitives instance.
    For state manager methods, uses ManagerRegistry to respect IMPL settings.

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        computer_primitives: ComputerPrimitives instance (required for ComputerPrimitives primitives).

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    # Special case: ComputerPrimitives methods use the provided instance
    if "ComputerPrimitives" in class_path:
        if computer_primitives is None:
            logger.warning(
                "Cannot resolve ComputerPrimitives primitive without computer_primitives instance",
            )
            return None
        return getattr(computer_primitives, method_name, None)

    # State managers: use ManagerRegistry typed getters to respect IMPL settings
    class_name = class_path.rsplit(".", 1)[-1]
    getter_name = _CLASS_TO_GETTER.get(class_name)

    if getter_name is None:
        logger.warning(f"Unknown manager class: {class_name}")
        return None

    try:
        from unity.manager_registry import ManagerRegistry

        getter = getattr(ManagerRegistry, getter_name)
        instance = getter()
        return getattr(instance, method_name, None)
    except Exception as e:
        logger.warning(f"Could not get manager via '{getter_name}': {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Primitives Runtime Class
# ────────────────────────────────────────────────────────────────────────────


class Primitives:
    """
    Runtime interface to all primitives for use within executed functions.

    All imports and instantiations are lazy - only the primitives actually
    accessed by a function are loaded. This means a function that only uses
    contacts and transcripts will NOT import or initialize the browser/desktop
    infrastructure.

    All state managers are obtained via ManagerRegistry typed methods
    (e.g., get_contact_manager()) to respect IMPL settings (real vs simulated).

    Usage in stored functions:
        async def my_function():
            # Only ContactManager is imported/initialized
            await primitives.contacts.update(text="Add Alice")

            # Only if accessed: browser/desktop infrastructure loaded
            await primitives.computer.navigate("https://example.com")
    """

    def __init__(self):
        # All managers lazily initialized via ManagerRegistry
        self._contacts: Optional["ContactManager"] = None
        self._transcripts: Optional["TranscriptManager"] = None
        self._knowledge: Optional["KnowledgeManager"] = None
        self._tasks: Optional["TaskScheduler"] = None
        self._secrets: Optional["SecretManager"] = None
        self._guidance: Optional["GuidanceManager"] = None
        self._web: Optional["WebSearcher"] = None
        self._computer: Optional[ComputerPrimitives] = None
        self._files: Optional["FileManager"] = None

    @property
    def contacts(self) -> "ContactManager":
        """Contact management primitives (ask, update)."""
        if self._contacts is None:
            from unity.manager_registry import ManagerRegistry

            self._contacts = ManagerRegistry.get_contact_manager()
        return self._contacts

    @property
    def transcripts(self) -> "TranscriptManager":
        """Transcript management primitives (ask)."""
        if self._transcripts is None:
            from unity.manager_registry import ManagerRegistry

            self._transcripts = ManagerRegistry.get_transcript_manager()
        return self._transcripts

    @property
    def knowledge(self) -> "KnowledgeManager":
        """Knowledge management primitives (ask, update, refactor)."""
        if self._knowledge is None:
            from unity.manager_registry import ManagerRegistry

            self._knowledge = ManagerRegistry.get_knowledge_manager()
        return self._knowledge

    @property
    def tasks(self) -> "TaskScheduler":
        """Task scheduling primitives (ask, update, execute)."""
        if self._tasks is None:
            from unity.manager_registry import ManagerRegistry

            self._tasks = ManagerRegistry.get_task_scheduler()
        return self._tasks

    @property
    def secrets(self) -> "SecretManager":
        """Secret management primitives (ask, update)."""
        if self._secrets is None:
            from unity.manager_registry import ManagerRegistry

            self._secrets = ManagerRegistry.get_secret_manager()
        return self._secrets

    @property
    def guidance(self) -> "GuidanceManager":
        """Guidance management primitives (ask, update)."""
        if self._guidance is None:
            from unity.manager_registry import ManagerRegistry

            self._guidance = ManagerRegistry.get_guidance_manager()
        return self._guidance

    @property
    def web(self) -> "WebSearcher":
        """Web search primitives (ask)."""
        if self._web is None:
            from unity.manager_registry import ManagerRegistry

            self._web = ManagerRegistry.get_web_searcher()
        return self._web

    @property
    def computer(self) -> ComputerPrimitives:
        """
        Computer use primitives (navigate, act, observe, query, reason).

        This provides browser and desktop control capabilities. Only imported
        and initialized when actually accessed, so functions that don't need
        computer use won't load browser/desktop infrastructure.
        """
        if self._computer is None:
            self._computer = ComputerPrimitives()
        return self._computer

    @property
    def files(self) -> "FileManager":
        """
        File management primitives (ask, ask_about_file, reduce, filter_files, search_files, visualize).

        Provides access to file parsing, data reduction, filtering, searching, and visualization
        capabilities. Only imported and initialized when actually accessed.
        """
        if self._files is None:
            from unity.manager_registry import ManagerRegistry

            self._files = ManagerRegistry.get_file_manager()
        return self._files
