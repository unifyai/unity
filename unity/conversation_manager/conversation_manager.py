import asyncio
import logging

import json
from dataclasses import dataclass
from typing import Optional
import contextlib
import contextvars

from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.settings import SETTINGS
from unity.manager_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.hierarchical_logger import SessionLogger
from unity.conversation_manager import debug_logger
from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.brain import build_brain_spec
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.brain_tools import ConversationManagerBrainTools
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.events import *
from unity.conversation_manager.events import _get_now

from unity.common.llm_client import new_llm_client
from unity.common.single_shot import single_shot_tool_decision
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.utils import Debouncer, log_task_exc

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.actor.base import BaseActor
from unity.conversation_manager.domains.proactive_speech import ProactiveSpeech

logger = logging.getLogger(__name__)

# Set logging level and add handler if not already configured
log_level = SETTINGS.conversation.LOG_LEVEL.upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Ensure we have a console handler to actually display logs
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


MAX_CONV_MANAGER_MSGS = 50


@dataclass(frozen=True)
class StepResult:
    """Result of processing a single event step."""

    input_event: "Event"
    llm_requested: bool
    llm_ran: bool
    output_events: list["Event"]


_step_llm_requests: contextvars.ContextVar[list[tuple[float, bool]] | None] = (
    contextvars.ContextVar("_step_llm_requests", default=None)
)


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker,
        job_name: str,
        user_id: str,
        assistant_id: str,
        user_name: str,
        assistant_name: str,
        assistant_age: str,
        assistant_nationality: str,
        assistant_about: str,
        assistant_number: str,
        assistant_email: str,
        user_number: str,
        user_email: str = None,
        voice_provider: str = "cartesia",
        voice_id: str = None,
        voice_mode: str = "tts",
        assistant_timezone: str = "",
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
        stop: asyncio.Event = None,
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_nationality = assistant_nationality
        self.assistant_timezone = assistant_timezone
        self.assistant_about = assistant_about
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.voice_mode = voice_mode

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email

        # initialization state
        self.initialized: bool = False
        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name

        # inactivity & shutdown
        self.inactivity_timeout = 360  # 6 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self.stop = stop

        self.event_broker = event_broker

        # managers
        self.transcript_manager: TranscriptManager = None
        self.contact_manager: ContactManager = None
        self.memory_manager: MemoryManager = None
        self.actor: BaseActor | None = None

        # debouncer (used to debounce llm runs)
        self.debouncer = Debouncer()

        # call manager
        self.call_manager = LivekitCallManager(self.get_call_config())

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a property with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        # Note: uses_realtime_api flag is stored for prompt building in _run_llm
        self._uses_realtime_api = self.call_manager.uses_realtime_api

        self.mode = "text"
        self.chat_history = []
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.active_tasks: dict[int, dict] = (
            {}
        )  # dict[int, {"handle": "SteerableTool", "query": "str", "handle_actions": []}]
        self.last_snapshot = _get_now()
        self._current_snapshot = None
        self.is_summarizing = None
        self.max_messages = 30

        # proactive speech
        self.proactive_speech = ProactiveSpeech()
        self._proactive_speech_task: asyncio.Task | None = None

        # ask handles (for Actor tasks)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool]] = []

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.info(
            "session_start",
            "ConversationManager session initialized",
        )

    def snapshot(self):
        self._current_snapshot = _get_now()
        return self._current_snapshot

    def commit(self):
        self.last_snapshot = self._current_snapshot
        notifs = self.notifications_bar.notifications
        self.notifications_bar.notifications = [n for n in notifs if n.pinned]

    @property
    def session_logger(self) -> SessionLogger:
        """The hierarchical session logger for this ConversationManager instance."""
        return self._session_logger

    def get_active_contact(self) -> dict | None:
        """Get the contact for the current active call, or fall back to the boss contact."""
        return self.call_manager.call_contact or self.contact_index.get_contact(
            contact_id=1,
        )

    def get_recent_voice_transcript(
        self,
        contact: dict | None = None,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """Extract recent voice transcript from the active conversation.

        Args:
            contact: Contact to get transcript for. Defaults to active contact.
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        conversation_turns: list[dict] = []
        last_message_timestamp: datetime | None = None

        if contact is None:
            contact = self.get_active_contact()

        if not contact:
            return conversation_turns, last_message_timestamp

        contact_id = contact.get("contact_id")
        if contact_id not in self.contact_index.active_conversations:
            return conversation_turns, last_message_timestamp

        active_contact = self.contact_index.active_conversations[contact_id]
        voice_thread = active_contact.threads.get("voice", [])

        # Optionally limit to last N messages
        if max_messages is not None:
            voice_thread = list(voice_thread)[-max_messages:]

        for msg in voice_thread:
            role = "assistant" if msg.name == "You" else "user"
            content = (msg.content or "").strip()

            # Skip system messages (e.g., "<Call Started>")
            if content.startswith("<") and content.endswith(">"):
                continue

            conversation_turns.append({"role": role, "content": content})

            if hasattr(msg, "timestamp") and msg.timestamp:
                last_message_timestamp = msg.timestamp

        return conversation_turns, last_message_timestamp

    def _preprocess_messages(
        self,
        messages: str | dict | list,
    ) -> str | dict | list:
        """Keep only the latest state snapshot from message history.

        ConversationManager renders a full state snapshot each turn. We keep only the
        latest snapshot when calling the model, while preserving any system messages
        and user interjections.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            return messages
        if not isinstance(messages, list):
            return messages

        try:
            # Find all state snapshot messages
            state_indices = [
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
            ]
            if not state_indices:
                return messages

            # Keep only the latest state snapshot and non-state messages
            last_state = messages[state_indices[-1]]
            kept: list[dict] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    kept.append(m)
                elif role == "user" and not m.get("_cm_state_snapshot"):
                    kept.append(m)

            kept.append(last_state)
            return kept
        except Exception:
            return messages

    async def interject_or_run(self, content: str):
        """Interject the ask handle or run the LLM"""
        if self.active_ask_handle and not self.active_ask_handle.done():
            self._session_logger.info(
                "event",
                "Routing to active ask handle",
                icon_override="🔀",
            )
            await self.active_ask_handle.interject(content)
        else:
            # With single-shot LLM, there's no ongoing brain loop to interject.
            # Just trigger a new LLM run.
            self._session_logger.info(
                "llm_thinking",
                "Triggering main CM brain",
                icon_override="🔀",
            )
            await self.request_llm_run(delay=0, cancel_running=True)

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(self, delay=0, cancel_running=False):
        await self.debouncer.submit(
            self._run_llm,
            delay=delay,
            cancel_running=cancel_running,
        )

    async def request_llm_run(self, delay=0, cancel_running=False) -> None:
        """Request an LLM run.

        In normal operation, the request is recorded and later scheduled by
        the event loop after the current event is handled.
        When executed inside ConversationManager.step(), the request is recorded
        and executed immediately by step().
        """
        requests = _step_llm_requests.get()
        if requests is not None:
            requests.append((delay, cancel_running))
            return
        self._pending_llm_requests.append((delay, cancel_running))

    async def flush_llm_requests(self) -> None:
        """Schedule any pending LLM runs recorded during event handling."""
        if _step_llm_requests.get() is not None:
            return
        if not self._pending_llm_requests:
            return
        delay, cancel_running = self._pending_llm_requests[-1]
        self._pending_llm_requests.clear()
        await self.run_llm(delay=delay, cancel_running=cancel_running)

    async def _step(self, event: "Event", *, publish: bool = False) -> StepResult:
        """Process one event deterministically and return produced output events.

        This is a TEST-ONLY method that bypasses the normal async event-driven flow.
        It avoids relying on background tasks by:
        - recording any requested LLM runs during event handling
        - running the LLM immediately (if requested)
        - capturing and applying any published output events to local state

        Args:
            event: Input event to process.
            publish: Whether to forward published events to the broker.

        Returns:
            StepResult with output events produced during this step.
        """
        published_events: list[Event] = []
        output_events: list[Event] = []
        llm_requested = False
        llm_ran = False

        original_publish = self.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                published_events.append(evt)
            if publish:
                return await original_publish(channel, message)
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)
        try:
            self.event_broker.publish = publish_wrapper

            await EventHandler.handle_event(
                event,
                self,
                is_voice_call=self.call_manager.uses_realtime_api,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            if llm_requested:
                llm_ran = True
                await self._run_llm()

            # Apply any published events to local state so callers can inspect state
            # without depending on background broker subscribers.
            for evt in published_events:
                if isinstance(
                    evt,
                    (SMSSent, EmailSent, UnifyMessageSent, PhoneCallSent),
                ):
                    output_events.append(evt)
                await EventHandler.handle_event(
                    evt,
                    self,
                    is_voice_call=self.call_manager.uses_realtime_api,
                )
        finally:
            self.event_broker.publish = original_publish
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=llm_requested,
            llm_ran=llm_ran,
            output_events=output_events,
        )

    async def _step_until_wait(
        self,
        event: "Event",
        *,
        max_steps: int = 5,
        publish: bool = False,
    ) -> StepResult:
        """Process an event and keep running LLM until it calls 'wait'.

        This is a TEST-ONLY method that gives the LLM continuous control until
        it explicitly decides to stop by calling the 'wait' tool.

        Args:
            event: Input event to process.
            max_steps: Maximum LLM steps to prevent infinite loops (default 5).
            publish: Whether to forward published events to the broker.

        Returns:
            StepResult with all output events produced across all steps.
        """
        all_output_events: list[Event] = []
        llm_ran = False

        original_publish = self.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                if isinstance(
                    evt,
                    (SMSSent, EmailSent, UnifyMessageSent, PhoneCallSent),
                ):
                    all_output_events.append(evt)
                # Handle the event locally
                await EventHandler.handle_event(
                    evt,
                    self,
                    is_voice_call=self.call_manager.uses_realtime_api,
                )
            if publish:
                return await original_publish(channel, message)
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)
        try:
            self.event_broker.publish = publish_wrapper

            # First, handle the incoming event
            await EventHandler.handle_event(
                event,
                self,
                is_voice_call=self.call_manager.uses_realtime_api,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            # Run LLM in a loop until 'wait' is called or max_steps reached
            step_count = 0
            while llm_requested and step_count < max_steps:
                llm_ran = True
                tool_name = await self._run_llm()
                step_count += 1

                # Stop if 'wait' was called
                if tool_name == "wait":
                    break

                # Check if another LLM run was requested (e.g., by event handlers)
                llm_requested = bool(step_requests)
                step_requests.clear()

                # If no explicit request but we didn't call 'wait', continue
                if not llm_requested and tool_name != "wait":
                    llm_requested = True

        finally:
            self.event_broker.publish = original_publish
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=True,
            llm_ran=llm_ran,
            output_events=all_output_events,
        )

    async def _run_llm(self) -> str | None:
        """Run a single LLM decision and return the tool name that was called."""
        self.snapshot()
        brain_spec = build_brain_spec(self)
        self._session_logger.debug(
            "state_update",
            f"State prompt:\n{brain_spec.state_prompt}",
        )
        input_message = brain_spec.state_message()
        system_prompt = brain_spec.system_prompt

        # Log LLM thinking start
        self._session_logger.log_llm_thinking(f"mode={self.mode}")

        # Build response model dynamically with current active tasks
        response_model = brain_spec.response_model

        brain_tools = ConversationManagerBrainTools(self)
        action_tools = ConversationManagerBrainActionTools(self)
        # Combine static tools with dynamic task steering tools
        tools = {
            **brain_tools.as_tools(),
            **action_tools.as_tools(),
            **action_tools.build_task_steering_tools(),
        }

        # Single-shot LLM call: one decision, one action
        client = new_llm_client(SETTINGS.UNIFY_MODEL, reasoning_effort="low")
        client.set_system_message(system_prompt)
        messages = self._preprocess_messages(self.chat_history + [input_message])
        result = await single_shot_tool_decision(
            client,
            messages,
            tools,
            tool_choice="required" if tools else "auto",
            response_format=response_model,
        )

        # Extract structured output (thoughts, call_guidance)
        structured = result.structured_output
        thoughts = ""
        if structured is not None:
            thoughts = getattr(structured, "thoughts", "")

            # Handle call_guidance for voice modes
            if self.mode in ["call", "unify_meet"]:
                call_guidance = getattr(structured, "call_guidance", "")
                if call_guidance:
                    contact = self.get_active_contact()
                    event = CallGuidance(contact, call_guidance)
                    await self.event_broker.publish(
                        "app:call:call_guidance",
                        event.to_json(),
                    )
                    await self.event_broker.publish(
                        "app:comms:assistant_call_guidance",
                        event.to_json(),
                    )

        # Log LLM response
        self._session_logger.log_llm_response(
            (
                f"thoughts: {thoughts[:100]}..."
                if len(thoughts) > 100
                else f"thoughts: {thoughts}"
            )
            + (f" | action: {result.tool_name}" if result.tool_name else ""),
        )

        self.commit()
        self._session_logger.debug("state_update", "Committing state")

        # Build assistant message for chat history
        assistant_content = (
            structured.model_dump_json() if structured else result.text_response or ""
        )
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": assistant_content})

        if (
            len(self.chat_history) >= int(0.7 * self.max_messages)
            and not self.is_summarizing
        ):
            self._session_logger.info("summarize", "Summarizing conversation")
            await self.event_broker.publish(
                "app:comms:summarize",
                SummarizeContext().to_json(),
            )
            self.is_summarizing = True

        return result.tool_name

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:actor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            # Initialization is triggered by StartupEvent handler which
            # sets details before starting init. Do not duplicate here.

            while True:
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg:
                    continue
                self.last_activity_time = self.loop.time()
                # process events
                event = Event.from_json(msg["data"])
                await EventHandler.handle_event(
                    event,
                    self,
                    is_voice_call=self.call_manager.uses_realtime_api,
                )
                await self.flush_llm_requests()

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout"""
        while True:
            await asyncio.sleep(self.inactivity_check_interval)
            current_time = self.loop.time()
            if current_time - self.last_activity_time > self.inactivity_timeout:
                self._session_logger.info(
                    "session_end",
                    f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown",
                )
                self.stop.set()
                await self.event_broker.aclose()

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details into SESSION_DETAILS."""
        self.user_id = payload["user_id"]
        self.assistant_id = payload["assistant_id"]
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_nationality = payload["assistant_nationality"]
        self.assistant_timezone = payload.get("assistant_timezone", "")
        self.assistant_about = payload["assistant_about"]
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.user_name = payload["user_name"]
        self.user_number = payload["user_number"]
        self.user_email = payload["user_email"]
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        self.voice_mode = payload["voice_mode"]
        # Set API key on SESSION_DETAILS for runtime access
        if payload.get("api_key"):
            SESSION_DETAILS.unify_key = payload["api_key"]
        # Populate the global SessionDetails singleton
        SESSION_DETAILS.populate(
            assistant_id=self.assistant_id,
            assistant_name=self.assistant_name,
            assistant_age=self.assistant_age,
            assistant_nationality=self.assistant_nationality,
            assistant_timezone=self.assistant_timezone,
            assistant_about=self.assistant_about,
            assistant_number=self.assistant_number,
            assistant_email=self.assistant_email,
            user_id=self.user_id,
            user_name=self.user_name,
            user_number=self.user_number,
            user_email=self.user_email,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            voice_mode=self.voice_mode,
        )
        # Export to env vars for subprocess inheritance
        SESSION_DETAILS.export_to_env()

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_name": self.user_name,
            "assistant_name": self.assistant_name,
            "user_number": self.user_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def get_call_config(self) -> CallConfig:
        return CallConfig(
            assistant_id=self.assistant_id,
            assistant_bio=self.assistant_about,
            assistant_number=self.assistant_number,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            voice_mode=self.voice_mode,
        )

    async def store_chat_history(self):
        if len(self.chat_history) >= 2:
            await self.event_broker.publish(
                "app:comms:chat_history",
                StoreChatHistory(chat_history=self.chat_history[-2:]).to_json(),
            )
            await asyncio.sleep(2)

    async def cleanup(self):
        """Clean up any running call processes"""
        await self.store_chat_history()
        await self.call_manager.cleanup_call_proc()
        if self.job_name and self.assistant_id != DEFAULT_ASSISTANT_ID:
            self._session_logger.info(
                "session_end",
                f"Marking job {self.job_name} done",
            )
            debug_logger.mark_job_done(self.job_name)
        self.stop.set()

    # Proactive speech related methods

    async def schedule_proactive_speech(self, skip_initial_wait: bool = False):
        """Decides if and when to speak proactively, and schedules it.

        Args:
            skip_initial_wait: If True, skip the initial 8s wait (used when rescheduling after a false decision)
        """
        self._session_logger.debug(
            "proactive_speech",
            f"schedule_proactive_speech called, mode={self.mode}, skip_initial_wait={skip_initial_wait}",
        )
        await self.cancel_proactive_speech()

        # Only schedule if we are in a call/voice mode where silence matters
        if self.mode not in ["call", "unify_meet"]:
            self._session_logger.debug(
                "proactive_speech",
                f"Skipping: mode {self.mode} not in supported modes",
            )
            return

        self._session_logger.debug("proactive_speech", "Creating proactive speech task")
        # Create a task to run the decision and potential wait
        self._proactive_speech_task = asyncio.create_task(
            self._proactive_speech_loop(skip_initial_wait=skip_initial_wait),
        )
        self._proactive_speech_task.add_done_callback(log_task_exc)

    async def cancel_proactive_speech(self):
        if self._proactive_speech_task and not self._proactive_speech_task.done():
            # Don't cancel if we are running inside the task (recursion case)
            if self._proactive_speech_task == asyncio.current_task():
                return

            self._proactive_speech_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_speech_task
            self._proactive_speech_task = None

    async def _proactive_speech_loop(self, skip_initial_wait: bool = False):
        try:
            # Wait a reasonable amount of time first to allow for natural conversation flow
            if not skip_initial_wait:
                self._session_logger.debug(
                    "proactive_speech",
                    "Waiting 10s before checking for silence",
                )
                await asyncio.sleep(10)
            else:
                self._session_logger.debug(
                    "proactive_speech",
                    "Skipping initial wait (reschedule after false decision)",
                )

            self._session_logger.debug(
                "proactive_speech",
                "Entering _proactive_speech_loop",
            )

            # Get conversation turns and last message timestamp using helper
            conversation_turns, last_message_timestamp = (
                self.get_recent_voice_transcript()
            )

            # Calculate elapsed time from last message timestamp
            if last_message_timestamp:
                now = _get_now()
                if isinstance(last_message_timestamp, datetime):
                    elapsed_seconds = (now - last_message_timestamp).total_seconds()
                else:
                    elapsed_seconds = 0
            else:
                elapsed_seconds = 0

            self._session_logger.debug(
                "proactive_speech",
                f"Elapsed time since last message: {elapsed_seconds:.1f}s",
            )

            # Build system prompt dynamically (there's no self.system_prompt attribute)
            brain_spec = build_brain_spec(self)
            decision = await self.proactive_speech.decide(
                conversation_turns,
                brain_spec.system_prompt,
                elapsed_seconds=elapsed_seconds,
            )
            self._session_logger.debug(
                "proactive_speech",
                f"Decision: should_speak={decision.should_speak}, delay={decision.delay}s",
            )

            if not decision.should_speak:
                # Adaptive wait: if we're already past 10s, check more frequently (5s)
                # Otherwise, wait until we hit ~12s threshold (but cap at 7s max wait)
                if elapsed_seconds < 10:
                    wait_time = min(
                        12 - elapsed_seconds,
                        7,
                    )  # Wait until ~12s, but max 7s
                else:
                    wait_time = 5  # Already past 10s, check every 5s

                self._session_logger.debug(
                    "proactive_speech",
                    f"Not speaking (delay={decision.delay}s), will check again in {wait_time:.1f}s",
                )
                await asyncio.sleep(wait_time)
                # Skip initial wait when rescheduling since we just waited
                await self.schedule_proactive_speech(skip_initial_wait=True)
                return

            self._session_logger.info(
                "proactive_speech",
                f"Speaking in {decision.delay}s: {decision.content}",
            )
            await asyncio.sleep(decision.delay)

            # Record in contact_index
            contact = self.get_active_contact()
            if contact:
                self.contact_index.push_message(
                    contact,
                    "voice",
                    message_content=decision.content,
                    role="assistant",
                )

            # Publish to Voice Agent via call_guidance channel
            await self.event_broker.publish(
                "app:call:call_guidance",
                json.dumps({"content": decision.content}),
            )

            await self.schedule_proactive_speech()

        except asyncio.CancelledError:
            self._session_logger.debug("proactive_speech", "Task cancelled")
            raise
        except Exception as e:
            self._session_logger.error("proactive_speech", f"Error in loop: {e}")
