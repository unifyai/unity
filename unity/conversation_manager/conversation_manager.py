from datetime import timedelta
import os
import asyncio
import logging

# import threading
from jinja2 import Template
import json
from pathlib import Path
from typing import Callable, Optional
import contextlib

from unity.conversation_manager.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager.domains.call_manager import LivekitCallManager
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.new_events import *

from unity.conversation_manager.domains.llm import LLM
from unity.conversation_manager.domains.actions import (
    Action,
    build_dynamic_response_models,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.utils import Debouncer

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conductor.conductor import Conductor
from unity.conversation_manager.domains import managers_utils
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


class ConversationManager:
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
        user_whatsapp_number: str,
        user_email: str = None,
        voice_provider: str = "cartesia",
        voice_id: str = None,
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
        stop: asyncio.Event = None,
        user_turn_end_callback: Optional[Callable[[list[dict]], str]] = None,
        realtime=False,
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
        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email
        self.user_whatsapp_number = user_whatsapp_number

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

        # llm
        self.llm = LLM("gpt-4.1", event_broker)
        # debouncer (used to debounce llm runs)
        self.debouncer = Debouncer()

        # call manager
        self.call_manager = LivekitCallManager(
            self.assistant_id,
            self.assistant_number,
            self.voice_provider,
            self.voice_id,
            realtime=realtime,
        )

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a propery with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        if not realtime:
            with open(Path(__file__).parent.resolve() / "prompts" / "v2.md") as f:
                self.system_prompt = f.read()
        else:
            # This prompt needs to have the conductor stuff inserted
            with open(Path(__file__).parent.resolve() / "prompts" / "realtime.md") as f:
                self.system_prompt = f.read()

        self.mode = "text"
        self.realtime = realtime
        self.chat_history = []
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.conductor_handles: dict[int, dict] = (
            {}
        )  # dict[int, {"handle": "SteerableTool", "query": "str", "handle_actions": []}]
        self.last_snapshot = datetime.now()
        self._current_snapshot = None
        self.call_exchange_id = None
        self.unify_call_exchange_id = None
        self.call_start_timestamp = None
        self.unify_call_start_timestamp = None
        self.conference_name = ""
        self.call_contact = None
        self.is_summarizing = None
        self.max_messages = 30

        # filler callback when user finishes speaking (phone/gmeet only)
        self.user_turn_end_callback: Optional[Callable[[list[dict]], str]] = (
            user_turn_end_callback
        )
        self._filler_task: asyncio.Task | None = None
        self._filler_started: asyncio.Event = asyncio.Event()
        self._filler_done: asyncio.Event = asyncio.Event()

    def snapshot(self):
        self._current_snapshot = datetime.now()
        return self._current_snapshot

    def commit(self):
        self.last_snapshot = self._current_snapshot
        notifs = self.notifications_bar.notifications
        self.notifications_bar.notifications = [n for n in notifs if n.pinned]

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(self, delay=0, cancel_running=False):
        await self.debouncer.submit(
            self._run_llm, delay=delay, cancel_running=cancel_running
        )

    async def _run_llm(self):
        self.snapshot()
        prompt = self.prompt_renderer.render_state(
            self.contact_index,
            self.notifications_bar,
            self.conductor_handles,
            self.last_snapshot,
        )
        print(prompt)
        input_message = {"role": "user", "content": prompt}
        boss_contact = self.contact_index.boss_contact
        system_prompt = Template(self.system_prompt).render(
            contact_id=boss_contact.contact_id,
            first_name=boss_contact.first_name,
            surname=boss_contact.surname,
            phone_number=boss_contact.phone_number,
            email_address=boss_contact.email_address,
        )

        # Use dynamic response models (set_details must be called before run_llm)
        dynamic_response_models = build_dynamic_response_models(
            include_email=self.assistant_email not in [None, ""],
            include_sms=self.assistant_number not in [None, ""],
            include_call=self.assistant_number not in [None, ""],
            realtime=self.realtime,
        )
        response_model = dynamic_response_models[self.mode]
        out = await self.llm.run(
            system_prompt=system_prompt,
            messages=self.chat_history + [input_message],
            # realtime model will handle the call so no need to stream anything to the call
            stream_to_call=self.mode in ["call", "unify_call", "gmeet"]
            and not self.realtime,
            response_model=response_model,
            call_type=self.mode,
            before_stream_start=(
                self.before_stream_start
                if (self.mode in ["call", "unify_call", "gmeet"] and not self.realtime)
                else None
            ),
        )
        parsed_out = json.loads(out)
        if "call" in self.mode:
            if not self.realtime:
                if self.mode == "unify_call":
                    topic = "app:comms:unify_call_utterance"
                    event = AssistantUnifyCallUtterance(
                        1, parsed_out["phone_utterance"]
                    )
                else:
                    topic = "app:comms:phone_utterance"
                    event = AssistantPhoneUtterance(
                        self.call_contact["phone_number"], parsed_out["phone_utterance"]
                    )
                await self.event_broker.publish(topic, event.to_json())

            else:
                if parsed_out.get("phone_guidance"):
                    await self.event_broker.publish(
                        "app:call:call_notifs",
                        json.dumps({"content": parsed_out["phone_guidance"]}),
                    )

        print(f"parsed_out {parsed_out}")
        actions = parsed_out.get("actions") or []  # sometimes actions exist but is None
        for action in actions:
            print("taking actions...")
            Action.take_action(
                self, action.pop("action_name"), **action, realtime=self.realtime
            )
            print("done taking actions...")
        self.commit()
        print("commiting...")
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": out})

        # if len(self.chat_history) >= int(0.7 * self.max_messages) and not self.is_summarizing:
        #    print("summarizing conversation...")
        #    Action.take_action(self, "summarize_conversation")

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:conductor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            if self.assistant_id:
                # asyncio.create_task(self.publish_startup())

                # this feels like it should be its own method really buts its really big
                # so will keep that way for now
                # also this is now fully blocking, will discuss it again with everyone what is the best
                # way to deal with this
                await managers_utils.init_conv_manager(self)
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
                await EventHandler.handle_event(event, self, realtime=self.realtime)

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

    # This can be moved to event handlers actually
    # and sets the Assistant dataclass instead of calling the conversation manager's
    def set_details(self, payload: dict):
        """Populate assistant/user/voice details and update environment variables."""
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
        self.user_whatsapp_number = payload["user_whatsapp_number"]
        self.user_email = payload["user_email"]
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        self.voice_mode = payload["voice_mode"]
        self.build_response_model()
        if payload.get("api_key"):
            os.environ["UNIFY_KEY"] = payload["api_key"]
        os.environ["USER_ID"] = self.user_id
        os.environ["USER_NAME"] = self.user_name
        os.environ["USER_NUMBER"] = self.user_number
        os.environ["USER_WHATSAPP_NUMBER"] = self.user_whatsapp_number
        os.environ["USER_EMAIL"] = self.user_email
        os.environ["ASSISTANT_NAME"] = self.assistant_name
        os.environ["ASSISTANT_NUMBER"] = self.assistant_number
        os.environ["ASSISTANT_EMAIL"] = self.assistant_email
        os.environ["VOICE_PROVIDER"] = self.voice_provider
        os.environ["VOICE_ID"] = self.voice_id
        os.environ["VOICE_MODE"] = self.voice_mode

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_name": self.user_name,
            "assistant_name": self.assistant_name,
            "user_number": self.user_number,
            "user_whatsapp_number": self.user_whatsapp_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.job_name} done")
        mark_job_done(self.job_name)
        self.call_manager.cleanup_call_proc()
        self.stop.set()

    async def run_filler_once(self):
        if self.realtime or self.mode not in ["call", "unify_call", "gmeet"]:
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
                channel, json.dumps({"type": "gen_chunk", "chunk": filler_text})
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
