from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel

from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.state_managers import BaseStateManager
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
    from unity.function_manager.function_manager import FunctionManager
    from unity.function_manager.primitives import ComputerPrimitives

__all__ = [
    "BaseActor",
    "BaseActorHandle",
    "PhoneCallHandle",
    "ComputerSessionHandle",
    "ComsManager",
    "BaseCodeActActor",
]

# --------------------------------------------------------------------------- #
# BaseActor
# --------------------------------------------------------------------------- #


class BaseActorHandle(SteerableToolHandle, ABC):
    """
    Marker base class for all actor handles returned by Actor.act().

    This provides a common nominal type across actor implementations while
    preserving the unified steerable surface inherited from SteerableToolHandle.
    Implementations are free to add additional helpers or properties, but the
    core pause/resume/stop/interject/ask/result interface must remain intact.
    """


class BaseActor(ABC):
    """
    Abstract contract that every concrete actor must satisfy.

    An actor is a component capable of performing work based on a natural
    language description. It returns a steerable handle that can be paused,
    resumed, interjected, or stopped. This type is intentionally decoupled
    from any task-specific terminology or lifecycle.

    Purpose and positioning
    -----------------------
    The Actor provides a direct, real-time handle to "act" in the world and
    get things done – e.g. open a web page, click UI elements, or perform a
    short-lived sandbox session during a conversation.

    Intended use
    ------------
    Use the Actor for interactive, ephemeral sessions within a live
    conversation (onboarding, guided walkthroughs, ad‑hoc demonstrations).
    It returns a steerable handle suitable for pause/resume/interject/stop.

    Usage guidance (LLM‑facing)
    ---------------------------
    Prefer calling ``Actor.act`` when the user's instruction implies a live,
    ad‑hoc, conversational session that should happen "now" inside the current
    chat, especially when the activity involves controlling tools or a UI in
    short iterative steps. Typical phrasings include:

    - "open a web page", "open a window", "navigate/click/show me"
    - "walk me through", "let's set this up together", "guide me live"
    - "troubleshoot together", "pair on this", "step‑by‑step now"

    This interface starts a live session and returns a steerable handle; it does
    not create durable records or schedules.
    """

    _as_caller_description: str = (
        "the Actor, performing a live action on behalf of the end user"
    )

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        computer_primitives: Optional["ComputerPrimitives"] = None,
        function_manager: Optional["FunctionManager"] = None,
        # Computer-use params for default environment creation
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        computer_mode: str = "magnitude",
        agent_mode: str = "web",
        agent_server_url: str | None = None,
        connect_now: bool = False,
        # Clarification queue params for environment wiring
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        """
        Shared initialization for concrete actor implementations.

        This centralizes:
        - Environment setup with sensible defaults (computer + state managers)
        - FunctionManager resolution (registry fallback)
        - Extraction of computer primitives for backward compatibility
        """
        self.environments: Dict[str, "BaseEnvironment"] = self._setup_environments(
            environments=environments,
            computer_primitives=computer_primitives,
            session_connect_url=session_connect_url,
            headless=headless,
            computer_mode=computer_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
            connect_now=connect_now,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        # Resolve FunctionManager (used by multiple actors for memoized skills).
        from unity.manager_registry import ManagerRegistry

        self.function_manager = (
            function_manager or ManagerRegistry.get_function_manager()
        )

        # Backward-compat: some call sites expect an actor-level computer primitives instance.
        self._computer_primitives = self._extract_computer_primitives()

    def _setup_environments(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]],
        computer_primitives: Optional["ComputerPrimitives"],
        session_connect_url: Optional[str],
        headless: bool,
        computer_mode: str,
        agent_mode: str,
        agent_server_url: str | None,
        connect_now: bool,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
    ) -> Dict[str, "BaseEnvironment"]:
        """
        Setup execution environments with defaults and optional clarification queues.

        Returns:
            Dict keyed by environment namespace.
        """
        from unity.actor.environments import (
            ComputerEnvironment,
            StateManagerEnvironment,
        )
        from unity.function_manager.primitives import ComputerPrimitives

        # If environments are explicitly provided, honor them and do not implicitly
        # introduce a computer environment (domain-agnostic mode).
        if environments is None:
            if computer_primitives is not None:
                cp = computer_primitives
            else:
                cp = ComputerPrimitives(
                    session_connect_url=session_connect_url,
                    headless=headless,
                    computer_mode=computer_mode,
                    agent_mode=agent_mode,
                    agent_server_url=agent_server_url,
                    connect_now=connect_now,
                )

            environments = [
                ComputerEnvironment(
                    cp,
                    clarification_up_q=clarification_up_q,
                    clarification_down_q=clarification_down_q,
                ),
                StateManagerEnvironment(
                    clarification_up_q=clarification_up_q,
                    clarification_down_q=clarification_down_q,
                ),
            ]

        env_map: Dict[str, "BaseEnvironment"] = {}
        for env in environments:
            ns = env.namespace
            if ns in env_map:
                raise ValueError(
                    f"Duplicate environment namespace detected: {ns!r}. "
                    "Environment namespaces must be unique.",
                )
            env_map[ns] = env
        return env_map

    def _extract_computer_primitives(self) -> Optional[Any]:
        """Extract computer primitives instance for backward compatibility."""
        if "computer_primitives" in getattr(self, "environments", {}):
            try:
                return self.environments["computer_primitives"].get_instance()
            except Exception:
                return None
        return None

    # ─────────────────────────── Work management ────────────────────────── #

    @abstractmethod
    async def act(
        self,
        description: str,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Perform work from a natural language description and return a steerable handle.

        This is the all-purpose method for engaging with knowledge, resources, and
        the world beyond immediate conversational context. Use ``act`` for any work
        that requires searching, retrieving, manipulating, or acting on information.

        **Capabilities include (but are not limited to):**

        - **Retrieval**: Search contact records, query knowledge bases, look up past
          conversations, find calendar events, search the web, retrieve files
        - **Action**: Send communications, update records, modify spreadsheets, control
          the desktop/web interface, schedule tasks, create reminders
        - **Combined**: Find information and then act on it (e.g., "find David's email
          and send him a meeting invite")

        **When to use ``act``:**

        Call ``act`` whenever you need to access or manipulate anything beyond your
        immediate context. When uncertain whether information exists or an action is
        possible, **call ``act`` anyway** — if it cannot help, it will simply report
        back explaining what it couldn't do. There is no penalty for speculative
        delegation; it is better to try and fail than to not try at all.

        **Key properties:**

        - The returned handle supports pause/resume/interject/stop for mid-flight control
        - Results are returned as strings (or structured output if ``response_format`` specified)
        - The actor has access to persistent storage, external APIs, and system capabilities
        - Multiple ``act`` calls can run concurrently

        Args:
            description: Natural language description of what to do. Can be a question
                ("What is David's email?"), a command ("Send an email to David"), or
                a combination ("Find David's email and send him a reminder").
            clarification_enabled: Whether the actor can ask clarifying questions.
            response_format: Optional Pydantic model for structured output.
            _parent_chat_context: Optional conversation context for continuity.
            _clarification_up_q: Queue for clarification requests (internal).
            _clarification_down_q: Queue for clarification answers (internal).

        Returns:
            A SteerableToolHandle for controlling and awaiting the result.
        """


class BaseCodeActActor(BaseActor, BaseStateManager, ABC):
    """
    Abstract contract for the CodeAct-style actor.

    Notes
    -----
    - Still shares the global actor base class: `BaseActor`.
    - Adds CodeAct-specific `act()` parameters (notifications, entrypoint, images) while
      preserving manager tool-registration patterns via `BaseStateManager`.
    """

    _as_caller_description: str = (
        "the CodeActActor, executing code-first actions on behalf of the end user"
    )

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        computer_primitives: Optional["ComputerPrimitives"] = None,
        function_manager: Optional["FunctionManager"] = None,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        computer_mode: str = "magnitude",
        agent_mode: str = "web",
        agent_server_url: str | None = None,
        connect_now: bool = False,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        BaseActor.__init__(
            self,
            environments=environments,
            computer_primitives=computer_primitives,
            function_manager=function_manager,
            session_connect_url=session_connect_url,
            headless=headless,
            computer_mode=computer_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
            connect_now=connect_now,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        BaseStateManager.__init__(self)

    @abstractmethod
    async def act(
        self,
        description: str,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        persist: bool = True,
    ) -> SteerableToolHandle:
        """Perform work from a natural-language description and return a steerable handle."""
        raise NotImplementedError
