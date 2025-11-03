from datetime import timedelta
import os
import asyncio
import logging

# import threading
from jinja2 import Template
import json
from pathlib import Path
from typing import Callable, Optional

from unity.conversation_manager_2.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager_2.domains.call_manager import LivekitCallManager
from unity.conversation_manager_2.domains.contact_index import ContactIndex
from unity.conversation_manager_2.domains.event_handlers import EventHandler
from unity.conversation_manager_2.new_events import *

from unity.conversation_manager_2.domains.llm import LLM
from unity.conversation_manager_2.domains.actions import Action, build_dynamic_response_models
from unity.conversation_manager_2.domains.notifications import NotificationBar
from unity.conversation_manager_2.domains.utils import Debouncer


import redis.asyncio as redis
from openai import AsyncOpenAI


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

with open(Path(__file__).parent.resolve() / "prompts" / "v2.md") as f:
    SYS = f.read()


MAX_CONV_MANAGER_MSGS = 50

# so basically, whenever the total count of a contact message > 10
# we are going to ask the contact manager/transcript manager to provide an update rolling summary
# we will keep the last N messages still


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
        assistant_region: str,
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
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_region = assistant_region
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
        
        # llm
        self.llm = LLM("gpt-4.1", event_broker)

        # debouncer (used to debounce llm runs)
        self.debouncer = Debouncer()

        # call manager
        self.call_manager = LivekitCallManager()

        # state - TODO: put the state into a dict or state class
        # access is as a propery with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        
        self.mode = "text"
        self.chat_history = []
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.last_snapshot = None
        self._current_snapshot = None
    
    def snapshot(self):
        self._current_snapshot = datetime.now()
    
    def commit(self):
        self.last_snapshot = self._current_snapshot
    
    # this is non-blocking, it will quickly submit the 
    # coro and return
    async def run_llm(self, delay=0):
        await self.debouncer.submit(self._run_llm, delay=delay)

    async def _run_llm(self):
        self.snapshot()
        # TODO: change to the real state
        prompt = "REPLY WITH SMS"
        print(prompt)
        input_message = {"role": "user", "content": prompt}
        # boss_contact = next(
        #     c for c in self.state.inverted_contacts_map.values() if c.is_boss
        # )
        # system_message = Template(SYS).render(
        #     contact_id=boss_contact.contact_id,
        #     first_name=boss_contact.first_name,
        #     surname=boss_contact.surname,
        #     phone_number=boss_contact.phone_number,
        #     email_address=boss_contact.email_address,
        # )
        # print(system_message)

        # Use dynamic response models (set_details must be called before run_llm)
        # response_model = self.state.dynamic_response_models[self.state.mode]
        out = await self.llm.run(system="", messages=self.chat_history, stream_to_call=self.mode in ["call", "unify_call", "gmeet"])
        parsed_out = json.loads(out)
        if "call" in self.mode:
            if self.mode == "unify_call":
                topic = "app:comms:unify_call_utterance"
                event = AssistantUnifyCallUtterance(1, parsed_out["phone_utterance"])
            else:
                topic = "app:comms:phone_utterance"
                event = AssistantPhoneUtterance(
                    self.state.phone_contact.phone_number, parsed_out["phone_utterance"]
                )
            await self.event_broker.publish(topic, event.to_json())

        print(f"parsed_out {parsed_out}")
        actions = parsed_out.get("actions", [])
        for action in actions:
            Action.take_action(action["action_name"], **action)
        self.commit()
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": out})
        # event = LLMInput(chat_history=self.state.chat_history)
        # asyncio.create_task(self.publish_bus_events(event))


    
    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:conductor:*",
                "app:managers:output",
            )

            if self.assistant_id:
                asyncio.create_task(self.publish_startup())
                print("Default startup")

            while True:
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg: continue
                self.last_activity_time = self.loop.time()
                # process events
                event = Event.from_json(msg["data"])
                await EventHandler.handle_event(event, self)

    async def publish_startup(self):
        print("publishing startup")
        await self.event_broker.publish(
            "app:managers:input",
            ManagersStartupRequest(
                agent_id=self.assistant_id,
                first_name=self.assistant_name,
                age=self.assistant_age,
                region=self.assistant_region,
                about=self.assistant_about,
                phone=self.assistant_number,
                email=self.assistant_email,
                user_phone=self.user_number,
                user_whatsapp_number=self.user_whatsapp_number,
                assistant_whatsapp_number=self.assistant_number,
            ).to_json(),
        )

    async def publish_bus_events(self, event: Event):
        await self.event_broker.publish(
            "app:managers:input",
            PublishBusEventRequest(event=event.to_dict()).to_json(),
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
        """Populate assistant/user/voice details and update environment variables."""
        self.user_id = payload["user_id"]
        self.assistant_id = payload["assistant_id"]
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_region = payload["assistant_region"]
        self.assistant_about = payload["assistant_about"]
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.user_name = payload["user_name"]
        self.user_number = payload["user_number"]
        self.user_whatsapp_number = payload["user_whatsapp_number"]
        self.user_email = payload["user_email"]
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
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
    
    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.state.job_name} done")
        mark_job_done(self.state.job_name)
        self.call_manager.cleanup_call_proc()
        self.stop.set()
