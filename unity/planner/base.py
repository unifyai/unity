from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from unity.common.llm_helpers import SteerableToolHandle
from unity.controller.controller import Controller

logger = logging.getLogger(__name__)

__all__ = [
    "BaseActiveTask",
    "BasePlanner",
    "PhoneCallHandle",
    "BrowserSessionHandle",
    "ComsManager",
]

# --------------------------------------------------------------------------- #
# BasePlan
# --------------------------------------------------------------------------- #


class BaseActiveTask(SteerableToolHandle, ABC):
    """
    Abstract contract that every concrete *active task* must satisfy.

    An active task represents a long-running task that can be steered at runtime
    (pause / resume / interject / ask / stop) and that ultimately resolves
    to a single result string.

    Sub-classes **must** provide concrete implementations of all abstract
    members below and expose them via ``valid_tools`` so that higher-level
    agents (or the UI) can discover the currently available controls.
    """

    # ───────────────────────────── Public API ───────────────────────────── #

    @abstractmethod
    async def ask(self, question: str) -> str:
        """
        Ask any question about the live (ongoing and active) task being worked on.
        """

    @property
    @abstractmethod
    def valid_tools(self) -> Dict[str, Callable]:
        """
        Map of *public-name* ➜ *callable* for the user-accessible controls
        that are *currently* valid in the plan's lifecycle state.
        """


# --------------------------------------------------------------------------- #
# BasePlanner
# --------------------------------------------------------------------------- #


class BasePlanner(ABC):
    """
    Abstract contract that every concrete *planner* must satisfy.

    A planner is a *factory* that spawns exactly one *active* plan at a time
    (for now).  It keeps a reference to that plan so that external callers
    can query its status or steer it later.
    """

    def __init__(self) -> None:
        self._active_task: Optional[BaseActiveTask] = None

    # ─────────────────────────── Plan management ────────────────────────── #

    async def execute(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> BaseActiveTask:
        """
        Create (and start) a new active task.

        Sub-classes implement the actual creation logic in
        :meth:`_make_plan`.  This thin wrapper only enforces the
        *single-active-plan* rule and stores the reference.
        """
        if self._active_task is not None:
            raise RuntimeError(
                "Another plan is still active. Stop it or wait for "
                "completion before starting a new one.",
            )

        active_task = await self._execute_task_and_return_handle(
            task_description,
            parent_chat_context=parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._active_task = active_task
        return active_task

    @abstractmethod
    async def _execute_task_and_return_handle(
        self,
        task_description: str,
        *,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> BaseActiveTask:
        """
        Concrete planner must build **and start** an active task implementation
        (e.g. ``SimulatedActiveTask``) and return it.
        """

    # ────────────────────────── Convenience API ─────────────────────────── #

    @property
    def active_task(self) -> Optional[BaseActiveTask]:
        """Return the currently running task (or *None* if idle)."""
        return self._active_task

    def clear_active_task(self) -> None:
        """Forget the active task (useful once it has completed)."""
        self._active_task = None


# --------------------------------------------------------------------------- #
# Call Handle (for make_call)
# --------------------------------------------------------------------------- #


class PhoneCallHandle(BaseActiveTask):
    """
    A steerable handle for a simulated, long-running phone call.

    This handle allows the planner to interact with the "call" in real-time,
    primarily by asking questions and receiving answers. It uses the same
    clarification queue mechanism as the HierarchicalPlanner for interaction.
    """

    def __init__(
        self,
        contact_id: int,
        purpose: str,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ):
        self._contact_id = contact_id
        self._purpose = purpose
        self.clarification_up_q = clarification_up_q
        self.clarification_down_q = clarification_down_q

        self._call_id = f"call_{uuid.uuid4().hex[:6]}"
        self._is_active = True
        self._is_complete = False
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()

        # The main task that simulates the call's duration and logic.
        self._call_task = asyncio.create_task(self._simulate_call())

        logger.info(
            f"[{self._call_id}] Phone call initiated with contact {self._contact_id} for purpose: '{self._purpose}'",
        )

    async def _simulate_call(self):
        """A background task simulating the phone call.
        In a real implementation, this would connect to a telephony service.
        Here, we just wait until `stop()` is called.
        """
        try:
            while self._is_active:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info(f"[{self._call_id}] Call was cancelled.")
        finally:
            if not self._final_result:
                self._final_result = f"Call with {self._contact_id} ended."
            self._is_complete = True
            self._completion_event.set()
            logger.info(f"[{self._call_id}] Simulation ended.")

    # --- Public API for the Planner ---

    async def ask(self, question: str) -> str:
        """
        Asks a question to the person on the other end of the call.

        This uses the clarification queues to pass the question to an external
        agent (e.g., the user or another LLM) and wait for a response.
        """
        if not self._is_active:
            return "The call has already ended."

        logger.info(f"[{self._call_id}] Asking question: '{question}'")
        await self.clarification_up_q.put(
            f"[{self._call_id}] Person on call was asked: '{question}'. What did they say?",
        )
        answer = await self.clarification_down_q.get()
        logger.info(f"[{self._call_id}] Received answer: '{answer}'")
        return answer

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager, ensuring the call is stopped."""
        await self.stop()

    # --- Implementation of Abstract Steerable Methods ---

    async def stop(self) -> str:
        """Ends the phone call."""
        if not self._is_active:
            return "Call already ended."
        logger.info(f"[{self._call_id}] Stop requested. Ending call.")
        self._is_active = False
        if not self._call_task.done():
            self._call_task.cancel()
        await self._call_task
        return self._final_result

    async def result(self) -> str:
        """Waits for the call to end and returns the final summary."""
        await self._completion_event.wait()
        return self._final_result

    def done(self) -> bool:
        """Returns True if the call has ended."""
        return self._is_complete

    async def pause(self) -> str:
        return "Pause is not supported for this call handle."

    async def resume(self) -> str:
        return "Resume is not supported for this call handle."

    async def interject(self, message: str) -> str:
        return "Interject is not supported for this call handle."

    @property
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Exposes the currently available methods to the planner."""
        if self._is_active:
            return {"ask": self.ask, "stop": self.stop}
        return {}


# --------------------------------------------------------------------------- #
# Browser Handle
# --------------------------------------------------------------------------- #


class BrowserSessionHandle(BaseActiveTask):
    """
    A steerable handle for an active browser session.

    This encapsulates the browser controller, exposing `act` and `observe`
    as methods on the handle. This aligns browser interaction with the
    handle-based pattern used for other complex tools like `make_call`.
    """

    def __init__(
        self,
        controller: Controller,
        interactions_log: Optional[List[Tuple[str, str, Optional[str]]]] = None,
    ):
        self.controller = controller
        self._is_active = True
        self._is_complete = False
        self._completion_event = asyncio.Event()
        self.session_id = f"browser_{uuid.uuid4().hex[:6]}"
        self.interactions_log = interactions_log
        logger.info(f"[{self.session_id}] Browser session handle created.")

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager, ensuring the handle is stopped."""
        await self.stop()

    # --- Public API for the Planner ---

    async def act(self, instruction: str) -> str:
        """Executes a high-level action in the browser and logs the interaction."""
        if not self._is_active:
            return "Browser session has been closed."
        logger.info(f"[{self.session_id}] Executing act: '{instruction}'")
        result = await self.controller.act(instruction)
        if self.interactions_log is not None:
            self.interactions_log.append(("act", instruction, result))
        return result

    async def observe(self, query: str) -> str:
        """Observes the current state of the browser and logs the interaction."""
        if not self._is_active:
            return "Browser session has been closed."
        logger.info(f"[{self.session_id}] Executing observe: '{query}'")
        result = await self.controller.observe(query)
        if self.interactions_log is not None:
            self.interactions_log.append(("observe", query, result))
        return result

    # --- Implementation of Abstract Steerable Methods ---

    async def stop(self) -> str:
        """Closes the browser session (handle)."""
        if not self._is_active:
            return "Browser session already closed."
        logger.info(f"[{self.session_id}] Closing browser session handle.")
        self._is_active = False
        self._is_complete = True
        self._completion_event.set()
        return "Browser session handle closed."

    async def result(self) -> str:
        """Waits for the session to be stopped."""
        await self._completion_event.wait()
        return "Browser session ended."

    def done(self) -> bool:
        return self._is_complete

    async def ask(self, question: str) -> str:
        """For browser handles, 'ask' is just an alias for 'observe'."""
        return await self.observe(question)

    async def pause(self) -> str:
        return "Pause is not supported for this browser handle."

    async def resume(self) -> str:
        return "Resume is not supported for this browser handle."

    async def interject(self, message: str) -> str:
        return "Interject is not supported for this browser handle."

    @property
    def valid_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Exposes the core browser primitives to the planner."""
        if self._is_active:
            return {
                "act": self.act,
                "observe": self.observe,
                "stop": self.stop,
            }
        return {}


# --------------------------------------------------------------------------- #
# Communications Manager (Dummy)
# --------------------------------------------------------------------------- #


class ComsManager:
    """
    A manager for handling various communication and session-based tools.

    This class would be instantiated and passed to the HierarchicalPlanner.
    The planner would then expose the public methods of this class as tools
    to the generated Python plans.
    """

    def __init__(self, controller: Controller):
        """
        Args:
            controller: A running instance of the browser Controller,
                        required for the BrowserSessionHandle.
        """
        self._controller = controller

    # --- Simple Communication Stubs ---

    async def send_email(
        self,
        contact_id: int,
        content: str,
        exchange_id: Optional[int] = None,
    ) -> str:
        """Sends an email to a specified contact."""
        log_msg = f"Simulating sending email to contact {contact_id}."
        if exchange_id:
            log_msg += f" (Replying to thread {exchange_id})"
        log_msg += f"\nContent: {content}"
        logger.info(log_msg)
        return f"Success: Email queued for delivery to contact {contact_id}."

    async def send_sms_message(self, contact_id: int, content: str) -> str:
        """Sends an SMS message to a specified contact."""
        logger.info(
            f"Simulating sending SMS to contact {contact_id}.\nContent: {content}",
        )
        return f"Success: SMS queued for delivery to contact {contact_id}."

    async def send_whatsapp_message(self, contact_id: int, content: str) -> str:
        """Sends a WhatsApp message to a specified contact."""
        logger.info(
            f"Simulating sending WhatsApp message to contact {contact_id}.\nContent: {content}",
        )
        return f"Success: WhatsApp message queued for delivery to contact {contact_id}."

    # --- Handle-based Tool Stubs ---

    def make_call(
        self,
        contact_id: int,
        purpose: str,
        *,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ) -> PhoneCallHandle:
        """
        Initiates a 'call' and returns an interactive handle.

        This function doesn't need to be `async` because it returns the handle
        immediately. The handle itself manages the asynchronous call process.
        The clarification queues are crucial for the handle's `ask` method to work.
        """
        return PhoneCallHandle(
            contact_id,
            purpose,
            clarification_up_q,
            clarification_down_q,
        )

    def start_browser_session(
        self,
        interactions_log: Optional[List[Tuple[str, str, Optional[str]]]] = None,
    ) -> BrowserSessionHandle:
        """
        Starts a browser session and returns an interactive handle.
        """
        return BrowserSessionHandle(self._controller, interactions_log=interactions_log)
