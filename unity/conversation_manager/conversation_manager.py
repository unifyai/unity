import asyncio
import logging

import json
from dataclasses import dataclass
import traceback
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

from unity.conversation_manager.domains.llm import LLM
from unity.conversation_manager.domains.actions import (
    Action,
    build_dynamic_response_models,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.utils import Debouncer, log_task_exc

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conductor.conductor import Conductor
from unity.conversation_manager.domains import managers_utils
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
        self.conductor: Conductor = None

        # llm - uses system default model for careful orchestration decisions
        self.llm = LLM(SETTINGS.UNIFY_MODEL, event_broker)
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

        # ask handles
        self.active_ask_handle: Optional["SteerableToolHandle"] = None
        # Main CM Brain handle (async tool loop) while an LLM run is in-flight.
        # This enables mid-flight interjections to interrupt/restart generation.
        self.active_brain_handle: Optional["SteerableToolHandle"] = None

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

    async def interject_or_run(self, content: str):
        """Interject the ask handle or run the LLM"""
        if self.active_ask_handle and not self.active_ask_handle.done():
            self._session_logger.info(
                "event",
                "Routing to active ask handle",
                icon_override="🔀",
            )
            await self.active_ask_handle.interject(content)
        elif self.active_brain_handle and not self.active_brain_handle.done():
            self._session_logger.info(
                "event",
                "Routing to active main brain handle",
                icon_override="🔀",
            )
            await self.active_brain_handle.interject(content)
        else:
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

    async def step(self, event: "Event", *, publish: bool = False) -> StepResult:
        """Process one event deterministically and return produced output events.

        This method is intended for synchronous-style testing and orchestration.
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

    async def _run_llm(self):
        self.snapshot()
        brain_spec = build_brain_spec(self)
        print(brain_spec.state_prompt)
        input_message = brain_spec.state_message()
        system_prompt = brain_spec.system_prompt

        # Log LLM thinking start
        self._session_logger.log_llm_thinking(f"mode={self.mode}")

        # Build response model dynamically with current active tasks
        response_model = brain_spec.response_model

        brain_tools = ConversationManagerBrainTools(self)
        action_tools = ConversationManagerBrainActionTools(self)
        tools = {**brain_tools.as_tools(), **action_tools.as_tools()}

        def _brain_tool_policy(step_index: int, tools: dict) -> tuple[str, dict]:
            # Keep the tool surface conservative: allow inspection tools on the first turn
            # only, then encourage immediate completion via final_answer.
            if step_index > 0:
                return "auto", {}
            return "auto", tools

        def _set_brain_handle(h: object) -> None:
            # Store as SteerableToolHandle (protocol-like) for interjection routing.
            try:
                self.active_brain_handle = h  # type: ignore[assignment]
            except Exception:
                pass

        def _clear_brain_handle(h: object) -> None:
            try:
                if self.active_brain_handle is h:
                    self.active_brain_handle = None
            except Exception:
                pass

        out = await self.llm.run(
            system_prompt=system_prompt,
            messages=self.chat_history + [input_message],
            response_model=response_model,
            _tools=tools,
            _tool_policy=_brain_tool_policy,
            _on_handle_created=_set_brain_handle,
            _on_handle_finished=_clear_brain_handle,
        )
        parsed_out = json.loads(out)
        if self.mode in ["call", "unify_meet"]:
            # Both TTS and Realtime modes use call_guidance - publish guidance events
            # The Voice Agent (fast brain) handles conversational responses independently
            if parsed_out.get("call_guidance"):
                contact = (
                    self.call_manager.call_contact
                    or self.contact_index.get_contact(contact_id=1)
                )
                event = CallGuidance(
                    contact,
                    parsed_out["call_guidance"],
                )
                await self.event_broker.publish(
                    "app:call:call_guidance",
                    event.to_json(),
                )
                await self.event_broker.publish(
                    "app:comms:assistant_call_guidance",
                    event.to_json(),
                )

        # Log LLM response
        actions = parsed_out.get("actions") or []  # sometimes actions exist but is None
        action_names = (
            [a.get("action_name", "unknown") for a in actions] if actions else []
        )
        self._session_logger.log_llm_response(
            f"{len(actions)} action(s): {action_names}" if actions else "no actions",
        )

        print(f"parsed_out {parsed_out}")
        action_coros = []
        for action in actions:
            print("taking actions...")
            coro = Action.take_action(
                self,
                action.pop("action_name"),
                _as_task=False,
                **action,
                is_voice_call=self.call_manager.uses_realtime_api,
            )
            if coro is not None:
                action_coros.append(coro)
            print("done taking actions...")

        if action_coros:
            results = await asyncio.gather(*action_coros, return_exceptions=True)
            for r in results:
                if isinstance(r, asyncio.CancelledError):
                    continue
                if isinstance(r, Exception):
                    traceback.print_exception(type(r), r, r.__traceback__)

        self.commit()
        print("commiting...")
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": out})

        if (
            len(self.chat_history) >= int(0.7 * self.max_messages)
            and not self.is_summarizing
        ):
            print("summarizing conversation...")
            await self.event_broker.publish(
                "app:comms:summarize",
                SummarizeContext().to_json(),
            )
            self.is_summarizing = True

        # Schedule proactive speech check after assistant turn
        await self.schedule_proactive_speech()

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:conductor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            if self.assistant_id != DEFAULT_ASSISTANT_ID:
                self.build_response_model()
                # asyncio.create_task(self.publish_startup())

                # Start initialization and operations listener
                asyncio.create_task(managers_utils.init_conv_manager(self))
                asyncio.create_task(managers_utils.listen_to_operations(self))
                print("Default startup")

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
                print(
                    f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown...",
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
        self.build_response_model()
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

    def build_response_model(self):
        # Initial build without active tasks - actual models are rebuilt per LLM call
        self.dynamic_response_models = build_dynamic_response_models(
            active_tasks={},
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
            print(f"Marking job {self.job_name} done")
            debug_logger.mark_job_done(self.job_name)
        self.stop.set()

    # Proactive speech related methods

    async def schedule_proactive_speech(self, skip_initial_wait: bool = False):
        """Decides if and when to speak proactively, and schedules it.

        Args:
            skip_initial_wait: If True, skip the initial 8s wait (used when rescheduling after a false decision)
        """
        print(
            f"[Proactive Speech] schedule_proactive_speech called, mode={self.mode}, skip_initial_wait={skip_initial_wait}",
        )
        await self.cancel_proactive_speech()

        # Only schedule if we are in a call/voice mode where silence matters
        if self.mode not in ["call", "unify_meet"]:
            print(
                f"[Proactive Speech] Skipping: mode {self.mode} not in supported modes",
            )
            return

        print("[Proactive Speech] Creating proactive speech task...")
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
                print("[Proactive Speech] Waiting 5s before checking for silence...")
                await asyncio.sleep(10)
            else:
                print(
                    "[Proactive Speech] Skipping initial wait (reschedule after false decision)",
                )

            print("[Proactive Speech] Entering _proactive_speech_loop")

            # Build conversation from contact_index and calculate elapsed time
            conversation_turns = []
            last_message_timestamp = None

            contact = self.call_manager.call_contact or self.contact_index.get_contact(
                contact_id=1,
            )
            if (
                contact
                and contact["contact_id"] in self.contact_index.active_conversations
            ):
                active_contact = self.contact_index.active_conversations[
                    contact["contact_id"]
                ]
                voice_thread = active_contact.threads.get("voice", [])

                for msg in voice_thread:
                    role = "assistant" if msg.name == "You" else "user"
                    content = msg.content

                    if content.startswith("<") and content.endswith(">"):
                        continue

                    conversation_turns.append({"role": role, "content": content})

                    if hasattr(msg, "timestamp") and msg.timestamp:
                        last_message_timestamp = msg.timestamp

            # Calculate elapsed time from last message timestamp
            if last_message_timestamp:
                now = _get_now()
                if isinstance(last_message_timestamp, datetime):
                    elapsed_seconds = (now - last_message_timestamp).total_seconds()
                else:
                    elapsed_seconds = 0
            else:
                elapsed_seconds = 0

            print(
                f"[Proactive Speech] Elapsed time since last message: {elapsed_seconds:.1f}s",
            )

            decision = await self.proactive_speech.decide(
                conversation_turns,
                self.system_prompt,
                elapsed_seconds=elapsed_seconds,
            )
            print(
                f"[Proactive Speech] Decision: should_speak={decision.should_speak}, delay={decision.delay}s",
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

                print(
                    f"[Proactive Speech] Not speaking (LLM chose delay={decision.delay}s), will check again in {wait_time:.1f}s",
                )
                await asyncio.sleep(wait_time)
                # Skip initial wait when rescheduling since we just waited
                await self.schedule_proactive_speech(skip_initial_wait=True)
                return

            print(
                f"Proactive Speech decided to speak in {decision.delay}s: {decision.content}",
            )
            await asyncio.sleep(decision.delay)

            # Record in contact_index
            contact = self.call_manager.call_contact or self.contact_index.get_contact(
                contact_id=1,
            )
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
            print("Proactive speech task cancelled.")
            raise
        except Exception as e:
            print(f"Error in proactive speech loop: {e}")
