import os
import asyncio
# import threading
import json
from typing import Dict, Callable, Literal, Union, Optional
import contextlib
from pathlib import Path
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_core import from_json

from unity.events.event_bus import EVENT_BUS
from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.prompt_utils import *
# from unity.conversation_manager.comms_actions import _send_sms_message_via_number
from unity.helpers import run_script, terminate_process
from unity.conversation_manager_2.prompt_utils import NotificationBar, ContactThread, ThreadMessage


import redis.asyncio as redis
from openai import AsyncOpenAI

@dataclass
class Contact:
    id: int
    name: str
    is_boss: bool
    number: str
    email: str
class ActionEvent:
    def __init__(self, action):
        self.action = action
    def __str__(self):
        return self.action

MAX_PENDING_EVENTS = 10


CONV_CONTEXT_LENGTH = 50

with open(Path(__file__).parent.resolve() / "prompts" / "v1.md") as f:
    SYS = f.read()
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
    number_or_id: str = Field(..., description="Exact number or contact id of the contact to sms")
    message: str


class MakeCall(BaseModel):
    action_name: Literal["make_call"]
    number_or_id: str = Field(..., description="Exact number or contact id of the contact to call")


# comms actions (other users)
...

actions = Union[WaitForNextEvent, SendSMS, MakeCall]
class ResponsePhone(BaseModel):
    thoughts: str
    phone_utterance: str
    actions: Optional[list[actions]]

class Response(BaseModel):
    thoughts: str
    actions: Optional[list[actions]]

responses_model = {
    "call": ResponsePhone,
    "gmeet": ResponsePhone,
    "text": Response
}

import aiohttp

headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
async def _send_sms_message_via_number(
    to_number: str,
    message: str,
    event_broker: redis.Redis
) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        str: The response from the SMS API
    """
    from_number = os.getenv("ASSISTANT_NUMBER")

    print(f"Sending SMS from {from_number} to {to_number}: {message}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/phone/send-text",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "Body": message,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                print(e)
            response_text = await response.text()
            print(f"Response: {response_text}")
            # await event_broker.publish(
            #     "app:comms:sms_sent",
            #     json.dumps({
            #         "topic": to_number,
            #         "to": "past",
            #         "event": SMSMessageSentEvent(
            #             content=message,
            #             role="Assistant",
            #             timestamp=datetime.now().isoformat(),
            #         ).to_dict(),
            #     }),
            # )
            return response_text


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

        # this will probs be retrieved from a database or whatever
        self.contacts_map = {
            "+12697784020": Contact("1", "Yasser Ahmed", True, "+12697784020", "yasser@unify.ai"),
            "+13502381308": Contact("2", "Dan Lenton", False, "+13502381308", "dan@unify.ai")
        }

        self.inverted_contacts_map = {v.id:v for v in self.contacts_map.values()}
        

        self.state = {
            "notifications": NotificationBar(),
            "active_conversations": {},
            "stale_conversations": {}
        }
        self.chat_history = []
    
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
        active_convs = "\n\n".join([str(c) for c in self.state["active_conversations"].values()])
        notif = str(self.state["notifications"])
        prompt = f"<notifications>\n{add_spaces(notif)}\n</notifications>\n<active_conversations>\n{add_spaces(active_convs)}\n</active_conversations>"
        input_message = [{"role": "user", "content": prompt}]
        print(input_message[0])
        if self.mode in ["call", "gmeet"]:
            print("running...")
            last_phone_utterance = ""
            out = ""
            async with self.openai_client.responses.stream(
                model="gpt-4.1",
                instructions=SYS,
                input=self.chat_history+input_message,
                text_format=responses_model[self.mode]
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
                         
        else:
            out = await self.openai_client.responses.parse(model="gpt-4.1", instructions=SYS, input=self.chat_history+input_message, text_format=responses_model[self.mode])  
            parsed_out = out.output[0].content[0].parsed.model_dump()
            out = out.output[0].content[0].text
        
        print(parsed_out)
        self.state["notifications"].clear()
        if parsed_out["actions"] is not None: 
            for action in parsed_out["actions"]:
                if action["action_name"] == "send_sms":
                    print("sending sms message")
                    contact_num_id = action["number_or_id"]
                    contact = self.contacts_map.get(contact_num_id) or self.inverted_contacts_map.get(contact_num_id)
                    await _send_sms_message_via_number(contact.number, action["message"], self.event_broker)
                    if contact.id not in self.state["active_conversations"]:
                        self.state["active_conversations"][contact.id] = ConversationContact(contact.id, contact.name, contact.is_boss, False)
                        self.state["notifications"].push_notif(f"Adding '{contact.name}' to active conversations")

                    self.state["active_conversations"][contact.id].push_message("sms", 
                                                                       message=ThreadMessage("You", action["message"], datetime.now()))

        self.chat_history.append(input_message[0])
        self.chat_history.append({"role": "assistant", "content": out})
        print(self.chat_history)
    
    async def scheduele_llm_run(self, delay=1, cancel_running=False):

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
                
                if msg is not None: print(msg)

                # there are still pending messages and no scheduled responses or currently running responses
                if msg is None:
                    # if (
                    #     self.pending_events and (not self.schedueled_response or self.schedueled_response.done()) 
                    #     and (not self.current_response or self.current_response.done())
                    # ):
                        # await self.scheduele_llm_run(0)
                        ...
                else:
                    event = Event.from_json(msg["data"])
                    print(event)
                    await self.handle_event(event)
                    
    async def handle_event(self, event: Event):
        if isinstance(event, PhoneCallInitiated):
            # start phone call process and wait untils its done, we should probably make sure
            # first that any running llm calls are awaited, and any schedueled llm calls are canceled
            # llm inference should not start until the process is set up (through PhoneCallStartedEvent)
            if self.mode in ["call", "gmeet"]:
                # can't make the call
                ...
            else:
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
                active_conversations = self.state["active_conversations"]
                stale_conversations = self.state["stale_conversations"]
                notifs = self.state["notifications"]
                
                contact = self.contacts_map.get(event.contact)
                if not contact:
                    # will deal with this later
                    ...
                if contact.id in active_conversations:
                    # just add a message to the phone thread
                    conv_c = active_conversations[contact.id]

                elif contact.id in stale_conversations:
                    conv_c = stale_conversations.pop(contact.id)
                    active_conversations[contact.id] = conv_c
                    
                else:
                    conv_c = ConversationContact(contact.id, name=contact.name, is_boss=contact.is_boss, on_phone=True)
                    active_conversations[contact.id] = conv_c
                    notifs.push_notif(f"Adding '{contact.name}' to active conversations")

                conv_c.push_message("phone", message=ThreadMessage(contact.name, "<Phone call Initiated...>", event.timestamp))
                notifs.push_notif(f"Phone call initiated by '{contact.name}'")

                

        elif isinstance(event, PhoneCallStarted):
            self.mode = "call"
            active_conversations = self.state["active_conversations"]
            stale_conversations = self.state["stale_conversations"]
            notifs = self.state["notifications"]
            
            contact = self.contacts_map.get(event.contact)
            if not contact:
                # will deal with this later
                ...
            if contact.id in active_conversations:
                # just add a message to the phone thread
                conv_c = active_conversations[contact.id]

            elif contact.id in stale_conversations:
                conv_c = stale_conversations.pop(contact.id)
                active_conversations[contact.id] = conv_c
                
            else:
                conv_c = ConversationContact(contact.id, name=contact.name, is_boss=contact.is_boss, on_phone=True)
                active_conversations[contact.id] = conv_c
                notifs.push_notif(f"Adding '{contact.name}' to active conversations")

            conv_c.push_message("phone", message=ThreadMessage(contact.name, "<Phone call Initiated...>", event.timestamp))
            notifs.push_notif("Phone call initiated by '{contact.name}'")

            await self.scheduele_llm_run(0, cancel_running=True)
        elif isinstance(event, PhoneCallEnded):
            terminate_process(self.call_proc)
        elif isinstance(event, PhoneUtterance):
            active_conversations = self.state["active_conversations"]
            stale_conversations = self.state["stale_conversations"]
            notifs = self.state["notifications"]
            
            contact = self.contacts_map.get(event.contact)
            if not contact:
                # will deal with this later
                ...
            if contact.id in active_conversations:
                # just add a message to the phone thread
                conv_c = active_conversations[contact.id]

            elif contact.id in stale_conversations:
                conv_c = stale_conversations.pop(contact.id)
                active_conversations[contact.id] = conv_c
                
            else:
                conv_c = ConversationContact(contact.id, name=contact.name, is_boss=contact.is_boss, on_phone=True)
                active_conversations[contact.id] = conv_c
                notifs.push_notif(f"Adding '{contact.name}' to active conversations")
            conv_c.push_message("phone", message=ThreadMessage(contact.name, event.content, event.timestamp))
            notifs.push_notif(f"Phone utterance recieved from '{contact.name}'")
            await self.scheduele_llm_run(0, cancel_running=True)

        else:
            # otherwise (whatsapp, sms, email) just scheduele another llm run after 2 seconds
            # if there is no response at the moment, if there is a response, cancel it, and scheduel
            # check if there is a schedueled response, rescheduele
            if self.mode == "text":
                active_conversations = self.state["active_conversations"]
                stale_conversations = self.state["stale_conversations"]
                notifs = self.state["notifications"]
                
                contact = self.contacts_map.get(event.contact)
                if not contact:
                    # will deal with this later
                    ...
                if contact.id in active_conversations:
                    # just add a message to the phone thread
                    conv_c = active_conversations[contact.id]

                elif contact.id in stale_conversations:
                    conv_c = stale_conversations.pop(contact.id)
                    active_conversations[contact.id] = conv_c
                    
                else:
                    conv_c = ConversationContact(contact.id, name=contact.name, is_boss=contact.is_boss, on_phone=True)
                    active_conversations[contact.id] = conv_c
                    notifs.push_notif(f"Adding '{contact.name}' to active conversations")

                conv_c.push_message("sms", message=ThreadMessage(contact.name, event.content, event.timestamp))
                notifs.push_notif(f"SMS message recieved from '{contact.name}'")
                await self.scheduele_llm_run(2, cancel_running=True)


# think about the end behaviour (how the events should look like in the end)
# and design the system around it