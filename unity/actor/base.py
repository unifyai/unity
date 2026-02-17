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
    from unity.guidance_manager.guidance_manager import GuidanceManager

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
    language request. It returns a steerable handle that can be paused,
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
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
    ) -> None:
        """
        Shared initialization for concrete actor implementations.

        This centralizes:
        - Environment dict construction from the provided list
        - FunctionManager resolution (registry fallback)
        - GuidanceManager resolution (registry fallback)
        - Extraction of computer primitives for backward compatibility
        """
        self.environments: Dict[str, "BaseEnvironment"] = self._setup_environments(
            environments=environments,
        )

        from unity.manager_registry import ManagerRegistry

        self.function_manager = (
            function_manager or ManagerRegistry.get_function_manager()
        )
        self.guidance_manager = (
            guidance_manager or ManagerRegistry.get_guidance_manager()
        )

        # Backward-compat: some call sites expect an actor-level computer primitives instance.
        self._computer_primitives = self._extract_computer_primitives()

    def _setup_environments(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]],
    ) -> Dict[str, "BaseEnvironment"]:
        """
        Build the environment namespace dict from the provided list.

        Returns:
            Dict keyed by environment namespace.
        """
        if environments is None:
            environments = []

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
        request: str,
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Perform work from a natural language request and return a steerable handle.

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
            request: Natural language request specifying what to do. Can be a question
                ("What is David's email?"), a command ("Send an email to David"), or
                a combination ("Find David's email and send him a reminder").
            guidelines: Optional meta-guidance on *how* to approach the task, as
                opposed to *what* to do. Examples: "don't install new python packages",
                "use sub-agents for solving this task", "prefer simple solutions over
                complex ones". When provided, these are injected into the system prompt
                so the actor follows them throughout the session.
            clarification_enabled: Whether the actor can request clarification from
                its **caller** (i.e. whichever process holds the returned handle).
                This does NOT surface questions to the end user directly — the
                caller decides how (or whether) to resolve them.
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
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
    ) -> None:
        BaseActor.__init__(
            self,
            environments=environments,
            function_manager=function_manager,
            guidance_manager=guidance_manager,
        )
        BaseStateManager.__init__(self)

    @abstractmethod
    async def act(
        self,
        request: str,
        *,
        guidelines: Optional[str] = None,
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
        """Perform work from a natural-language request and return a steerable handle."""
        raise NotImplementedError
