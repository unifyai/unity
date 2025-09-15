import os
import asyncio
# import threading
import json
from typing import Dict, Callable, Literal, Union, Optional
import contextlib
from pathlib import Path

from pydantic import BaseModel
from pydantic_core import from_json

from unity.events.event_bus import EVENT_BUS
from unity.conversation_manager.events import *
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

# actions

# conductor
class AskConductor(BaseModel):
    action_name: Literal["ask_conductor"]
    query: str

# wait
class WaitForNextEvent(BaseModel):
    action_name: Literal["wait"]

# comms actions (main user)
# whatsapp has some issues, will deal with it later
# class SendWhatsapp(BaseModel):
#     ...

class SendEmail(BaseModel):
    action_name: Literal["send_email"]
    subject: str
    body: str

class SendSMS(BaseModel):
    action_name: Literal["send_sms"]
    message: str


class MakeCall(BaseModel):
    action_name: Literal["make_call"]

# comms actions (other users)
...

actions = Union[AskConductor, WaitForNextEvent, SendSMS, SendEmail, MakeCall]
class Response(BaseModel):
    phone_utterance: str
    actions: Optional[list[actions]]



# conversation manager can:
# 1- send comms directly to boss user
# 2- send comms directly to someone else
# 3- ask conductor for anything else

# conversation manager should have it's "mode" switched
# is just a "switch" that changes based on incoming events to the conversation manager
# Call started should simply start a phone call process and switch the mode to "call"
# if the controller joins a google meet and uses mode "google_meet"
# default mode is "text"
# this will just change the system prompt and structured output slightly, also enables streaming
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
        tts_provider: str = "cartesia",
        voice_id: str = None,
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_region = assistant_region
        self.assistant_about = assistant_about
        self.tts_provider = tts_provider
        self.voice_id = voice_id

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email
        self.user_whatsapp_number = user_whatsapp_number

        # events & state(history)
        self.state = {
            "tasks": None,
            "current_active_contacts": None,
            "information_requests": None # when one of the managers asks for information?
        }
        self.conv_context_length = conv_context_length
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.past_events = past_events or []
        self.pending_events = []
        self.inflight_events = []


        self.mode: Literal["call", "gmeet", "text"] = "text"
        # self.current_llm_run = None
        self.current_response: asyncio.Task | None = None
        self.schedueled_response: asyncio.Task | None = None

        # switches to "True" when in a call
        # self.call_mode = False
        # self.call_purpose = "general"
        # self.task_context = task_context
        # self.user_turn_end_callback = user_turn_end_callback
        # self.pending_calls = []

        # meet conference
        # self.meet_id = None
        # self.meet_browser = None
        # self.meet_joined = asyncio.Event()

        # conductor
        self.conductor = ...

        # logging
        self.loop = asyncio.get_event_loop()
        # self.transcript_manager = None
        # self.redis = None
        # self.broader_context = ""
        self.project_name = project_name
        # self.logging_lock = threading.Lock()
        self.is_past_events_init = asyncio.Event()
        # asyncio.create_task(self._init_past_events())

        self.event_broker = event_broker
        self.openai_client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
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
        # format events as message
        input_messages = []
        for e in self.past_events + self.inflight_events:
            if isinstance(e, ActionEvent):
                input_messages.append({"role": "assistant", "content": str(e)})
            else:
                input_messages.append({"role": "user", "content": str(e)})
        print(input_messages)
        if self.mode in ["call", "gmeet"]:
            print("running...")
            last_phone_utterance = ""
            out = ""
            async with self.openai_client.responses.stream(
                model="gpt-4.1",
                instructions="""
You are a general purpose assistant that is conversing with a user on a call.
You will recieve a bunch of user events, these are events such as the user's phone utterance, whatsapp messages recieved, etc.
Reply to the user using the following format:
{
    "phone_utterance": <YOUR RESPONSE TO THE USER OVER THE PHONE>,
    "actions": <ACTION TO TAKE BASED ON USER INPUTS>
}
""".strip(),
                input=input_messages,
                text_format=Response
            ) as stream:
                first_chunk = True
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        # print(event.delta)
                        out += event.delta
                        parsed_out = from_json(out, allow_partial="trailing-strings")
                        if parsed_out.get("phone_utterance"):
                            if first_chunk:
                                await self.event_broker.publish("app:call:response_gen", json.dumps({
                                    "type": "start_gen"
                                }))
                                first_chunk = False
                            if len(last_phone_utterance) != len(parsed_out["phone_utterance"]):
                                await self.event_broker.publish("app:call:response_gen", json.dumps({
                                    "type": "gen_chunk",
                                    "chunk": parsed_out["phone_utterance"][len(last_phone_utterance):]
                                }))
                            last_phone_utterance = parsed_out["phone_utterance"]
            await self.event_broker.publish("app:call:response_gen", json.dumps({
                                    "type": "end_gen"
                                }))
            print(parsed_out)
            self.past_events.extend(self.inflight_events.copy())
            self.inflight_events.clear()
            self.past_events.append(ActionEvent(out))
                         
        else:
            ...
    
    async def scheduele_llm_run(self, delay=1, cancel_running=False):
        self.inflight_events = self.pending_events.copy()
        self.pending_events.clear()

        if self.schedueled_response and not self.schedueled_response.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self.schedueled_response
        
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
            self.schedueled_response = asyncio.create_task(run_llm_delayed(delay))
        else:
            if not cancel_running:
                with contextlib.suppress(asyncio.CancelledError):
                    await self.current_response
            self.current_response = asyncio.create_task(self.run_llm())


    
    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")
            while True:
                msg = await pubsub.get_message(timeout=2, ignore_subscribe_messages=True)
                
                # if msg is not None: print(msg)

                # there are still pending messages and no scheduled responses or currently running responses
                if msg is None:
                    # if (
                    #     self.pending_events and (not self.schedueled_response or self.schedueled_response.done()) 
                    #     and (not self.current_response or self.current_response.done())
                    # ):
                        # await self.scheduele_llm_run(0)
                        ...
                else:
                    event = Event.from_dict(json.loads(msg["data"])["event"])
                    if event.transient:
                        continue
                    self.pending_events.append(event)
                    if isinstance(event, PhoneCallInitiatedEvent):
                        # start phone call process and wait untils its done, we should probably make sure
                        # first that any running llm calls are awaited, and any schedueled llm calls are canceled
                        # llm inference should not start until the process is set up (through PhoneCallStartedEvent)
                        if self.mode in ["call", "gmeet"]:
                            # can't make the call
                            ...
                        else:
                            print("I WAS HERE...")
                            if self.schedueled_response and not self.schedueled_response.done():
                                self.schedueled_response.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await self.schedueled_response
                            if self.current_response and not self.current_response.done():
                                await self.current_response
                        
                            # start the process here
                            target_path = Path(__file__).parent.resolve() / "medium_scripts" / "call.py"
                            self.call_proc = run_script(
                                str(target_path),
                                "dev",
                                self.user_number,
                                self.assistant_number,
                                self.tts_provider,
                                self.voice_id if self.voice_id else "None",
                                "None",
                                str(False),
                            )
                    elif isinstance(event, PhoneCallStartedEvent):
                        self.mode = "call"
                        await self.scheduele_llm_run(0, cancel_running=True)
                    elif isinstance(event, PhoneCallEndedEvent):
                        terminate_process(self.call_proc)
                    elif event.is_urgent or isinstance(event, PhoneUtteranceEvent):
                        if event.role == "user":
                            await self.scheduele_llm_run(0, cancel_running=True)
                    elif len(self.pending_events) >= MAX_PENDING_EVENTS:
                        # check if there is any running responses, wait for the response and then run
                        # this should also probably wait for the run to fully complete to avoid filling pending events
                        await self.scheduele_llm_run(0)
                    else:
                        # otherwise (whatsapp, sms, email) just scheduele another llm run after 2 seconds
                        # if there is no response at the moment, if there is a response, cancel it, and scheduel
                        # check if there is a schedueled response, rescheduele
                        if self.mode == "text":
                            await self.scheduele_llm_run(2, cancel_running=True)



# think about the end behaviour (how the events should look like in the end)
# and design the system around it