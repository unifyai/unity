import os
import asyncio

# import threading
from jinja2 import Template
import json
from typing import Literal
import contextlib
from pathlib import Path

from pydantic_core import from_json

from unity.events.event_bus import EVENT_BUS
from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.actions import (
    RESPONSES_MODEL,
    _send_sms_message_via_number,
)
from unity.conversation_manager_2.state import ConversationManagerState
from unity.helpers import run_script, terminate_process


import redis.asyncio as redis
from openai import AsyncOpenAI


class ActionEvent:
    def __init__(self, action):
        self.action = action

    def __str__(self):
        return self.action


MAX_PENDING_EVENTS = 10


CONV_CONTEXT_LENGTH = 50

with open(Path(__file__).parent.resolve() / "prompts" / "v1.md") as f:
    SYS = f.read()


@dataclass
class Contact:
    id: str
    name: str
    is_boss: bool
    number: str
    email: str


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

        # events & state(history)
        self.conv_context_length = conv_context_length
        self.mode: Literal["call", "gmeet", "text"] = "text"
        # self.current_llm_run = None
        self.current_response: asyncio.Task | None = None
        self.scheduled_response: asyncio.Task | None = None

        # switches to "True" when in a call

        # conductor
        self.conductor = ...

        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name
        self.is_past_events_init = asyncio.Event()
        # asyncio.create_task(self._init_past_events())

        # inactivity & shutdown
        self.inactivity_timeout = 360  # 6 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self.stop = stop

        self.event_broker = event_broker
        self.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # this will probs be retrieved from a database or whatever
        self.phone_contacts_map = {
            "+12697784020": Contact(
                "1", "Yasser Ahmed", True, "+12697784020", "yasser@unify.ai"
            ),
            "+13502381308": Contact(
                "2", "Dan Lenton", False, "+13502381308", "dan@unify.ai"
            ),
            "+16605382869": Contact("3", "Ved", False, "+16605382869", "ved@unify.ai"),
        }
        self.email_contacts_map = {
            "yasser@unify.ai": Contact(
                "1", "Yasser Ahmed", True, "+12697784020", "yasser@unify.ai"
            ),
            "dan@unify.ai": Contact(
                "2", "Dan Lenton", False, "+13502381308", "dan@unify.ai"
            ),
            "ved@unify.ai": Contact("3", "Ved", False, "+16605382869", "ved@unify.ai"),
        }

        self.inverted_contacts_map = {v.id: v for v in self.phone_contacts_map.values()}
        self.state = ConversationManagerState(
            self.phone_contacts_map, self.email_contacts_map
        )
        self.chat_history = []
        self.call_proc = None

    # should be re-written to account for the new refactor
    async def _init_past_events(self):
        # TODO: this should be generalized to retrieve the entire
        # state, which inclues the current active tasks
        # the current active contacts etc
        print("Retrieving all past events...")
        bus_events = await EVENT_BUS.search(
            filter='type == "Comms"',
            limit=self.conv_context_length,
        )

        self.past_events = [Event.from_bus_event(e).to_dict() for e in bus_events][::-1]
        self.is_past_events_init.set()

    async def run_llm(self):
        prompt = str(self.state)
        print(prompt)
        input_message = [{"role": "user", "content": prompt}]
        print(input_message[0])
        if self.mode in ["call", "gmeet"]:
            print("running...")
            last_phone_utterance = ""
            out = ""
            async with self.openai_client.responses.stream(
                model="gpt-4.1",
                instructions=Template(SYS).render(
                    name=self.user_name, number=self.user_number
                ),
                input=self.chat_history + input_message,
                text_format=RESPONSES_MODEL[self.mode],
            ) as stream:
                first_chunk = True
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        # print(event.delta)
                        out += event.delta
                        parsed_out = from_json(out, allow_partial="trailing-strings")
                        if parsed_out.get("phone_utterance"):
                            if first_chunk:
                                await self.event_broker.publish(
                                    "app:call:response_gen",
                                    json.dumps({"type": "start_gen"}),
                                )
                                first_chunk = False
                            if len(last_phone_utterance) != len(
                                parsed_out["phone_utterance"]
                            ):
                                await self.event_broker.publish(
                                    "app:call:response_gen",
                                    json.dumps(
                                        {
                                            "type": "gen_chunk",
                                            "chunk": parsed_out["phone_utterance"][
                                                len(last_phone_utterance) :
                                            ],
                                        }
                                    ),
                                )
                            last_phone_utterance = parsed_out["phone_utterance"]
            await self.event_broker.publish(
                "app:call:response_gen", json.dumps({"type": "end_gen"})
            )
            print(parsed_out)

        else:
            out = await self.openai_client.responses.parse(
                model="gpt-4.1",
                instructions=Template(SYS).render(
                    name=self.user_name, number=self.user_number
                ),
                input=self.chat_history + input_message,
                text_format=RESPONSES_MODEL[self.mode],
            )
            parsed_out = out.output[0].content[0].parsed.model_dump()
            out = out.output[0].content[0].text

        print(parsed_out)
        self.state.clear_notifications()
        if parsed_out["actions"] is not None:
            for action in parsed_out["actions"]:
                if action["action_name"] == "send_sms":
                    print("sending sms message")
                    contact_num_id = action["number_or_id"]
                    contact = self.phone_contacts_map.get(
                        contact_num_id
                    ) or self.inverted_contacts_map.get(contact_num_id)
                    await _send_sms_message_via_number(
                        contact.number, action["message"]
                    )
                    event = SMSSent(contact=contact.number, content=action["message"])
                    self.state.push_event(event)

        self.chat_history.append(input_message[0])
        self.chat_history.append({"role": "assistant", "content": out})
        print(self.chat_history)

    async def schedule_llm_run(self, delay=1, cancel_running=False):
        if self.scheduled_response and not self.scheduled_response.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self.scheduled_response

        if cancel_running:
            if self.current_response and not self.current_response.done():
                self.current_response.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.current_response

        async def run_llm_delayed(delay):
            await asyncio.sleep(delay)
            if self.current_response and not self.current_response.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await self.current_response
            self.current_response = asyncio.create_task(self.run_llm())

        if delay > 0:
            self.scheduled_response = asyncio.create_task(run_llm_delayed(delay))
        else:
            if not cancel_running:
                with contextlib.suppress(asyncio.CancelledError):
                    await self.current_response
            self.current_response = asyncio.create_task(self.run_llm())

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*", "app:conductor:*")
            while True:
                msg = await pubsub.get_message(
                    timeout=2, ignore_subscribe_messages=True
                )

                if msg is not None:
                    print(msg)
                    self.last_activity_time = self.loop.time()

                # there are still pending messages and no scheduled responses or currently running responses
                # TODO: fix this branch
                if msg is None:
                    # if (
                    #     self.pending_events and (not self.scheduled_response or self.scheduled_response.done())
                    #     and (not self.current_response or self.current_response.done())
                    # ):
                    # await self.schedule_llm_run(0)
                    ...
                else:
                    event = Event.from_json(msg["data"])
                    print(event)
                    if isinstance(event, Ping):
                        print("ping received - keeping conversation manager alive")
                        continue
                    await self.handle_event(event)

    async def handle_event(self, event: Event):
        self.state.push_event(event)
        if isinstance(event, PhoneCallInitiated):
            # start phone call process and wait untils its done, we should probably make sure
            # first that any running llm calls are awaited, and any scheduled llm calls are canceled
            # llm inference should not start until the process is set up (through PhoneCallStartedEvent)
            if self.mode in ["call", "gmeet"]:
                # can't make the call
                ...
            else:
                if self.scheduled_response and not self.scheduled_response.done():
                    self.scheduled_response.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.scheduled_response
                if self.current_response and not self.current_response.done():
                    await self.current_response

                # start the process here
                target_path = (
                    Path(__file__).parent.resolve() / "medium_scripts" / "call.py"
                )
                self.call_proc = run_script(
                    str(target_path),
                    "dev",
                    self.user_number,
                    self.assistant_number,
                    self.voice_provider,
                    self.voice_id if self.voice_id else "None",
                    "None",
                    str(False),
                )

        elif isinstance(event, PhoneCallStarted):
            self.mode = "call"
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, PhoneCallEnded):
            self.mode = "text"
            terminate_process(self.call_proc)

        elif isinstance(event, PhoneUtterance):
            await self.schedule_llm_run(0, cancel_running=True)

        else:
            # otherwise (whatsapp, sms, email) just schedule another llm run after 2 seconds
            # if there is no response at the moment, if there is a response, cancel it, and scheduel
            # check if there is a scheduled response, reschedule
            await self.schedule_llm_run(2, cancel_running=True)

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


# think about the end behaviour (how the events should look like in the end)
# and design the system around it
