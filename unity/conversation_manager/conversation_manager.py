import os
import asyncio
import logging

import json
from typing import Callable, Optional
import contextlib

from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.settings import SETTINGS
from unity.conversation_manager.prompt_builders import build_system_prompt
from unity.singleton_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.hierarchical_logger import SessionLogger
from unity.conversation_manager import debug_logger
from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.events import *

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
import redis.asyncio as redis


logger = logging.getLogger(__name__)

# Set logging level and add handler if not already configured
log_level = os.getenv("CONVERSATION_MANAGER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Ensure we have a console handler to actually display logs
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


MAX_CONV_MANAGER_MSGS = 50


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker: redis.Redis,
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
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
        stop: asyncio.Event = None,
        user_turn_end_callback: Optional[Callable[[list[dict]], str]] = None,
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_nationality = assistant_nationality
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
        # Note: realtime flag is stored for prompt building in _run_llm
        self._realtime_mode = self.call_manager.realtime

        self.mode = "text"
        self.chat_history = []
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.active_tasks: dict[int, dict] = (
            {}
        )  # dict[int, {"handle": "SteerableTool", "query": "str", "handle_actions": []}]
        self.last_snapshot = datetime.now()
        self._current_snapshot = None
        self.is_summarizing = None
        self.max_messages = 30

        # filler callback when user finishes speaking (phone/unify_meet only)
        self.user_turn_end_callback: Optional[Callable[[list[dict]], str]] = (
            user_turn_end_callback
        )
        self._filler_task: asyncio.Task | None = None
        self._filler_started: asyncio.Event = asyncio.Event()
        self._filler_done: asyncio.Event = asyncio.Event()

        # proactive speech
        self.proactive_speech = ProactiveSpeech()
        self._proactive_speech_task: asyncio.Task | None = None

        # ask handles
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.info(
            "session_start",
            "ConversationManager session initialized",
        )

    def snapshot(self):
        self._current_snapshot = datetime.now()
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
        else:
            self._session_logger.info(
                "llm_thinking",
                "Triggering main CM brain",
                icon_override="🔀",
            )
            await self.run_llm(delay=0, cancel_running=True)

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(self, delay=0, cancel_running=False):
        await self.debouncer.submit(
            self._run_llm,
            delay=delay,
            cancel_running=cancel_running,
        )

    async def _run_llm(self):
        self.snapshot()
        prompt = self.prompt_renderer.render_state(
            self.contact_index,
            self.notifications_bar,
            self.active_tasks,
            self.last_snapshot,
        )
        print(prompt)
        input_message = {"role": "user", "content": prompt}
        boss_contact = self.contact_index.boss_contact
        system_prompt = build_system_prompt(
            bio=self.assistant_about,
            contact_id=boss_contact.contact_id,
            first_name=boss_contact.first_name,
            surname=boss_contact.surname,
            phone_number=boss_contact.phone_number,
            email_address=boss_contact.email_address,
            is_voice_call=self._realtime_mode,
            active_tasks=self.active_tasks,
        )

        # Log LLM thinking start
        self._session_logger.log_llm_thinking(f"mode={self.mode}")

        # Build response model dynamically with current active tasks
        response_models = build_dynamic_response_models(
            active_tasks=self.active_tasks,
            is_voice_call=self.call_manager.realtime,
        )
        response_model = response_models[self.mode]
        out = await self.llm.run(
            system_prompt=system_prompt,
            messages=self.chat_history + [input_message],
            # realtime model will handle the call so no need to stream anything to the call
            stream_to_call=self.mode in ["call", "unify_meet"]
            and not self.call_manager.realtime,
            response_model=response_model,
            call_type=self.mode,
            before_stream_start=(
                self.before_stream_start
                if (
                    self.mode in ["call", "unify_meet"]
                    and not self.call_manager.realtime
                )
                else None
            ),
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
                    "app:call:realtime_guidance",
                    event.to_json(),
                )
                await self.event_broker.publish(
                    "app:comms:assistant_realtime_guidance",
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
        for action in actions:
            print("taking actions...")
            Action.take_action(
                self,
                action.pop("action_name"),
                **action,
                is_voice_call=self.call_manager.realtime,
            )
            print("done taking actions...")
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
                    is_voice_call=self.call_manager.realtime,
                )

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

    # Convenience setter to allow late binding of the callback
    def set_user_turn_end_callback(self, callback: Callable[[list[dict]], str]) -> None:
        """Set or replace the callback invoked at user turn end (phone).

        The callback receives the current chat_history (list of messages) and
        should return a short filler string to be injected just before the
        assistant's next streamed response begins.
        """
        self.user_turn_end_callback = callback

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details into SESSION_DETAILS."""
        self.user_id = payload["user_id"]
        self.assistant_id = payload["assistant_id"]
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_nationality = payload["assistant_nationality"]
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
        if payload.get("api_key"):
            os.environ["UNIFY_KEY"] = payload["api_key"]
        # Populate the global SessionDetails singleton
        SESSION_DETAILS.populate(
            assistant_id=self.assistant_id,
            assistant_name=self.assistant_name,
            assistant_age=self.assistant_age,
            assistant_nationality=self.assistant_nationality,
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
            is_voice_call=self.call_manager.realtime,
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
        self.call_manager.cleanup_call_proc()
        if self.job_name and self.assistant_id != DEFAULT_ASSISTANT_ID:
            print(f"Marking job {self.job_name} done")
            debug_logger.mark_job_done(self.job_name)
        self.stop.set()

    # ToDo: Refactor this to use the debouncer like the LLM run

    # Filler related methods

    async def run_filler_once(self):
        if self.call_manager.realtime or self.mode not in [
            "call",
            "unify_meet",
        ]:
            return

        # record the running task so before_stream_start can coordinate
        self._filler_task = asyncio.current_task()
        self._filler_started = asyncio.Event()
        self._filler_done = asyncio.Event()
        if not self.user_turn_end_callback:
            self._filler_task = None
            return

        # pre-compute filler so streaming isn't blocked after start
        try:
            filler_text = self.user_turn_end_callback(self.chat_history) or ""
        except Exception:
            filler_text = ""
        self._filler_started.set()
        channel = f"app:{self.mode}:response_gen"
        await self.event_broker.publish(channel, json.dumps({"type": "start_gen"}))
        if filler_text:
            await self.event_broker.publish(
                channel,
                json.dumps({"type": "gen_chunk", "chunk": filler_text}),
            )
        await self.event_broker.publish(channel, json.dumps({"type": "end_gen"}))
        self._filler_done.set()
        self._filler_task = None

    async def cancel_filler(self):
        # cancel the running filler task
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._filler_task
            self._filler_task = None

    async def before_stream_start(self):
        # called just before the LLM streaming emits first start_gen
        if self._filler_task and not self._filler_task.done():
            if not self._filler_started.is_set():
                await self.cancel_filler()
            else:
                await self._filler_done.wait()

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
                from datetime import datetime

                now = datetime.now()
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

            # Publish to voice layer
            if self.call_manager.realtime:
                await self.event_broker.publish(
                    "app:call:realtime_guidance",
                    json.dumps({"content": decision.content}),
                )
            else:
                channel = f"app:{self.mode}:response_gen"
                await self.event_broker.publish(
                    channel,
                    json.dumps({"type": "start_gen"}),
                )
                await self.event_broker.publish(
                    channel,
                    json.dumps({"type": "gen_chunk", "chunk": decision.content}),
                )
                await self.event_broker.publish(
                    channel,
                    json.dumps({"type": "end_gen"}),
                )

            await self.schedule_proactive_speech()

        except asyncio.CancelledError:
            print("Proactive speech task cancelled.")
            raise
        except Exception as e:
            print(f"Error in proactive speech loop: {e}")
