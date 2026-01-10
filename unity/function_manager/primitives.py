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
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, TypedDict

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
    from unity.function_manager.browser import Browser
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
        headless: bool = False,
        browser_mode: str = "magnitude",
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
        *,
        connect_now: bool = False,
        # Deprecated parameters (kept for backward compatibility, ignored)
        session_connect_url: str | None = None,
        controller_mode: str = "hybrid",
    ):
        # Cache browser configuration for lazy initialization
        browser_kwargs = {
            "magnitude": {
                "headless": headless,
                "agent_mode": agent_mode,
                "agent_server_url": agent_server_url,
            },
            "mock": {
                # MockBrowserBackend accepts optional url, screenshot, etc.
                # but works fine with no kwargs
            },
        }

        self._secret_manager = None
        self._browser = None
        self._browser_mode = browser_mode
        self._browser_kwargs_map = browser_kwargs
        # Lazily create the Browser (and thus avoid connecting to agent-service) unless requested
        if connect_now:
            from unity.function_manager.browser import Browser

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
        from unity.function_manager.browser_backends import (
            MagnitudeBrowserBackend,
            MockBrowserBackend,
        )

        if self._browser_mode == "magnitude":
            backend_class = MagnitudeBrowserBackend
        elif self._browser_mode == "mock":
            backend_class = MockBrowserBackend
        else:
            raise ValueError(
                f"Unknown browser_mode: '{self._browser_mode}'. Must be 'magnitude' or 'mock'.",
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
            from unity.function_manager.browser import Browser

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
            # "ask",
            # "ask_about_file",
            # Discovery tools (synchronous - return dict directly)
            "tables_overview",
            "list_columns",
            "schema_explain",
            # Query/aggregation tools (synchronous)
            "reduce",
            "filter_files",
            "search_files",
            "visualize",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Manager Metadata Registry (Single Source of Truth for Prompt Generation)
# ---------------------------------------------------------------------------
# This metadata is used by StateManagerEnvironment.get_prompt_context() to
# dynamically generate manager descriptions without hardcoding.

MANAGER_METADATA: Dict[str, Dict[str, Any]] = {
    "contacts": {
        "domain": "People & Relationships",
        "description": "People, organizations, contact records (names, emails, phones, roles, locations)",
        "methods": {
            "ask": "Find contacts by name/email/attribute, query relationships, get contact details",
            "update": "Create, edit, delete, or merge contact records",
        },
        "use_when": "Questions about specific people, contact info, 'who is X?', 'find contact in Y location'",
        "examples": "'Who is our contact at Acme Corp?', 'Find Alice's email', 'Contacts in Berlin?'",
        "priority": 3,
    },
    "tasks": {
        "domain": "Durable Work & Tracking",
        "description": "Task management, work queues, assignments, deadlines, priorities",
        "methods": {
            "ask": "Query task status, what's due/scheduled, assignments, priorities",
            "update": "Create, edit, delete, or reorder tasks (NOT for starting work)",
            "execute": "Start durable, tracked execution (use this to run tasks, not `.update(...)`)",
        },
        "use_when": "Questions about tasks/work items, 'what's due?', 'tasks assigned to X?', 'high-priority items?'",
        "examples": "'What tasks are due today?', 'Show Alice's open tasks', 'List high-priority items'",
        "priority": 4,
    },
    "transcripts": {
        "domain": "Conversation History",
        "description": "Past messages, conversation history, communication records (chat/SMS/email)",
        "methods": {
            "ask": "Search messages, find what someone said, retrieve conversation context",
        },
        "use_when": "Questions about past communications, 'what did X say?', 'last message about Y?', 'conversation with Z?'",
        "examples": "'What did Bob say yesterday?', 'Last SMS with Alice?', 'Messages mentioning budget?'",
        "priority": 2,
    },
    "knowledge": {
        "domain": "Facts, Policies & Domain Knowledge",
        "description": "Organizational facts, policies, procedures, reference material, documentation, stored information",
        "methods": {
            "ask": "Query stored knowledge - company policies (return/refund/warranty/HR), procedures, facts, historical records",
            "update": "Add/change facts, ingest structured data, update policies",
            "refactor": "Restructure knowledge schemas (advanced)",
        },
        "use_when": "Questions about company policies, operational procedures, reference docs, 'what is our X policy?', 'summarize Y procedure'",
        "examples": "'What's our return policy?', 'Summarize onboarding procedure', 'Office hours?', 'Warranty terms for X?'",
        "priority": 1,
    },
    "web": {
        "domain": "Time-Sensitive & External Research",
        "description": "External/public information and research (including general concepts/definitions), plus current events and 'today/latest/now' queries",
        "methods": {
            "ask": "Web search for current information, news, weather, public data",
        },
        "use_when": "Questions answered from public/external knowledge (including definitions/concepts) or requiring up-to-date info: current events, weather, news",
        "examples": "'What is the Eisenhower Matrix?', 'Weather in Berlin today?', 'Latest AI news?', 'Current stock price?'",
        "priority": 5,
    },
    "guidance": {
        "domain": "Function & Task Guidance",
        "description": "Execution instructions, runbooks, how-to guides for functions/tasks",
        "methods": {
            "ask": "Query execution instructions, runbooks, best practices for specific operations",
            "update": "Create, edit, or delete guidance entries linked to functions",
        },
        "use_when": "Questions about HOW to execute something, operational runbooks, incident response procedures",
        "examples": "'How do I handle DB failover?', 'Incident response for API outage?'",
        "priority": 6,
    },
    "files": {
        "domain": "Files & Data Operations",
        "description": "Received/downloaded files, document parsing, file metadata, data queries",
        "methods": {
            "ask": "Query about specific files, parse document contents, extract information from files",
            "tables_overview": "Discover available tables with optional column info",
            "list_columns": "Get column names and types for a table",
            "schema_explain": "Natural language explanation of table structure",
            "reduce": "Aggregate data (count, sum, mean, min, max, etc.)",
            "filter_files": "Query raw records with filtering",
            "search_files": "Semantic search over table data",
            "visualize": "Generate chart visualizations",
            "get_tools": "Get tools dict for passing to functions that accept `tools: FileTools`",
        },
        "use_when": "Questions about specific files/documents, data operations, aggregations, visualizations",
        "examples": "'Parse the attached PDF', 'What's in document X?', 'Find files about Y'",
        "priority": 7,
        "special_note": "Use `get_tools()` ONLY when passing to functions with `tools: FileTools` parameter. For direct operations, use method syntax: `await primitives.files.reduce(...)`",
    },
    "secrets": {
        "domain": "Credentials & Secrets",
        "description": "API keys, passwords, tokens, credentials",
        "methods": {
            "ask": "Get metadata/placeholders only (never returns actual secret values)",
            "update": "Create, edit, or delete secrets",
        },
        "use_when": "Managing credentials, API keys, secrets (rarely used in plans)",
        "examples": "Rarely used directly in plans",
        "priority": 8,
    },
}


# Tools exposed by get_tools() on files primitive (subset of all FileManager tools)
_FILE_TOOLS_EXPOSED = frozenset(
    {
        "tables_overview",
        "list_columns",
        "schema_explain",
        "reduce",
        "filter_files",
        "visualize",
    },
)


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
        "depends_on": [],
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
# FileTools TypedDict - Tools dictionary type for FileManager operations
# ────────────────────────────────────────────────────────────────────────────


class FileTools(TypedDict, total=False):
    """
    Dictionary of FileManager tool callables.

    Returned by `primitives.files.get_tools()` for passing to functions
    that accept a `tools` parameter for data operations.

    For direct data operations in your own code, use method syntax instead:
        result = await primitives.files.reduce(table=..., metric="count", ...)

    Keys
    ----
    tables_overview : Callable
        Discover available tables with optional column info
    list_columns : Callable
        Get column names and types for a table
    schema_explain : Callable
        Natural language explanation of table structure
    reduce : Callable
        Aggregate data (count, sum, mean, min, max, etc.)
    filter_files : Callable
        Query raw records with filtering
    search_files : Callable
        Semantic search over table data
    visualize : Callable
        Generate chart visualizations
    """

    tables_overview: Callable[..., Dict[str, Any]]
    list_columns: Callable[..., Dict[str, Any]]
    schema_explain: Callable[..., str]
    reduce: Callable[..., Any]
    filter_files: Callable[..., List[Dict[str, Any]]]
    search_files: Callable[..., List[Dict[str, Any]]]
    visualize: Callable[..., Any]


# ────────────────────────────────────────────────────────────────────────────
# Async FileManager Wrapper
# ────────────────────────────────────────────────────────────────────────────


class _AsyncFileManagerWrapper:
    """
    Wrapper that makes synchronous FileManager methods awaitable.

    This ensures consistency across all `primitives.*` namespaces - the LLM can
    safely use `await` on all primitives methods without needing to know which
    underlying implementations are sync vs async.

    The wrapper delegates to the underlying FileManager but wraps each method
    in an async function that simply returns the sync result. Docstrings are
    copied from the underlying FileManager methods for discoverability.
    """

    # Methods whose docstrings should be copied from the underlying FileManager
    _PROXIED_METHODS = (
        "tables_overview",
        "list_columns",
        "schema_explain",
        "reduce",
        "filter_files",
        "search_files",
        "visualize",
    )

    def __init__(self, file_manager: "FileManager"):
        self._fm = file_manager
        # Copy docstrings from underlying FileManager methods
        # Note: We only copy __doc__, NOT using update_wrapper which sets __wrapped__
        # because __wrapped__ breaks inspect.signature() for async wrapper methods
        for method_name in self._PROXIED_METHODS:
            wrapper = getattr(self, method_name, None)
            original = getattr(self._fm, method_name, None)
            if wrapper and original and original.__doc__:
                wrapper.__func__.__doc__ = original.__doc__

    async def tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Show the information for all tables. Async wrapper for consistency.

        See FileManager.tables_overview for full documentation.
        """
        return self._fm.tables_overview(
            include_column_info=include_column_info,
            file=file,
        )

    async def list_columns(
        self,
        *,
        include_types: bool = True,
        table: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List columns for the FileRecords index or a resolved logical table.
        Async wrapper for consistency.

        See FileManager.list_columns for full documentation.
        """
        return self._fm.list_columns(
            include_types=include_types,
            table=table,
        )

    async def schema_explain(self, *, table: str) -> str:
        """
        Return a natural-language explanation of a table's structure.
        Async wrapper for consistency.

        See FileManager.schema_explain for full documentation.
        """
        return self._fm.schema_explain(table=table)

    async def reduce(
        self,
        *,
        table: Optional[str] = None,
        metric: str,
        keys: Any,
        filter: Optional[Any] = None,
        group_by: Optional[Any] = None,
    ) -> Any:
        """
        Compute reduction metrics over a table. Async wrapper for consistency.

        See FileManager.reduce for full documentation.
        """
        return self._fm.reduce(
            table=table,
            metric=metric,
            keys=keys,
            filter=filter,
            group_by=group_by,
        )

    async def filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter files or resolve-and-filter per-file Content/Tables.
        Async wrapper for consistency.

        See FileManager.filter_files for full documentation.
        """
        return self._fm.filter_files(
            filter=filter,
            offset=offset,
            limit=limit,
            tables=tables,
        )

    async def search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        table: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over a resolved context. Async wrapper for consistency.

        See FileManager.search_files for full documentation.
        """
        return self._fm.search_files(
            references=references,
            k=k,
            table=table,
            filter=filter,
        )

    async def visualize(
        self,
        *,
        tables: Any,
        plot_type: str,
        x_axis: str,
        y_axis: Optional[str] = None,
        group_by: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        aggregate: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> Any:
        """
        Generate plot visualizations from table data. Async wrapper for consistency.

        See FileManager.visualize for full documentation.
        """
        return self._fm.visualize(
            tables=tables,
            plot_type=plot_type,
            x_axis=x_axis,
            y_axis=y_axis,
            group_by=group_by,
            filter=filter,
            title=title,
            aggregate=aggregate,
            scale_x=scale_x,
            scale_y=scale_y,
            bin_count=bin_count,
            show_regression=show_regression,
        )

    def get_tools(self) -> FileTools:
        """
        Get FileManager tools as a dictionary for passing to other functions.

        Returns ONLY the tools compatible with metric/analysis functions:
        - tables_overview, list_columns, schema_explain
        - reduce, filter_files, visualize

        Use this ONLY when calling a function that accepts a `tools: FileTools`
        parameter. For direct data operations, use method syntax instead:
            result = await primitives.files.reduce(table=..., metric="count", ...)

        Example
        -------
        >>> # When a function signature shows `tools: FileTools`:
        >>> tools = primitives.files.get_tools()
        >>> result = await some_function(tools, other_args...)

        Returns
        -------
        FileTools
            Dictionary with keys: tables_overview, list_columns, schema_explain,
            reduce, filter_files, visualize
        """
        all_tools = dict(self._fm.get_tools("ask", include_sub_tools=True))
        # Filter to only the tools exposed for external function use
        return {k: v for k, v in all_tools.items() if k in _FILE_TOOLS_EXPOSED}


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
        self._files: Optional[_AsyncFileManagerWrapper] = None

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
    def files(self) -> _AsyncFileManagerWrapper:
        """
        File management primitives for data discovery, querying, and visualization.

        All methods are async for consistency with other primitives - use `await`:

        Discovery tools:
        - await tables_overview() - List all available tables
        - await list_columns(table=...) - Get column names and types
        - await schema_explain(table=...) - Get natural language schema explanation

        Query tools:
        - await reduce(table, metric, keys, filter, group_by) - Aggregate data
        - await filter_files(filter, tables, limit) - Query raw records
        - await search_files(references, k, table) - Semantic search
        - await visualize(tables, plot_type, x_axis, y_axis, ...) - Generate charts

        Only imported and initialized when actually accessed.
        """
        if self._files is None:
            from unity.manager_registry import ManagerRegistry

            fm = ManagerRegistry.get_file_manager()
            self._files = _AsyncFileManagerWrapper(fm)
        return self._files
