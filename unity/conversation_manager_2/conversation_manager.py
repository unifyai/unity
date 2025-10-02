import os
import asyncio

# import threading
from jinja2 import Template
import json
from typing import Literal
import contextlib
from pathlib import Path

from pydantic_core import from_json

from unity.conversation_manager_2.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.actions import (
    RESPONSES_MODEL,
    _send_sms_message_via_number,
    _send_email_via_address,
    _start_call,
)
from unity.conversation_manager_2.state import ConversationManagerState
from unity.conversation_manager_2.prompt_utils import ThreadMessage
from unity.helpers import run_script, terminate_process
from unity.transcript_manager.types.message import UNASSIGNED
from .prompt_utils import ConversationContact


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

        self.phone_contacts_map = {}
        self.email_contacts_map = {}

        self.inverted_contacts_map = {v.id: v for v in self.phone_contacts_map.values()}
        self.state = ConversationManagerState(
            self.phone_contacts_map,
            self.email_contacts_map,
            self.inverted_contacts_map,
        )
        self.chat_history = []
        self.call_proc = None
        self.call_contact = None
        self.initialized = False

    async def run_llm(self):
        now = None
        # this will be used at the end to only filter out notification before now
        # this can be done in a better way but for now, this ~ensures, notifications added mid llm run are not cleared after the run
        # TODO: abstract the llm stuff in another module to reduce clutter
        if self.state.notifications.notifs:
            now = datetime.now()
        prompt = str(self.state)
        print(self.state.notifications)
        print(prompt)
        input_message = [{"role": "user", "content": prompt}]
        if self.mode in ["call", "gmeet"]:
            print("running...")
            last_phone_utterance = ""
            out = ""
            async with self.openai_client.responses.stream(
                model="gpt-4.1",
                instructions=Template(SYS).render(
                    name=self.user_name,
                    number=self.user_number,
                ),
                # input=self.chat_history + input_message,
                input=input_message,
                text_format=RESPONSES_MODEL[self.mode],
            ) as stream:
                first_chunk = True
                async for event in stream:
                    if not self.call_proc:
                        print("call ended, stopping stream")
                        break
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
                                parsed_out["phone_utterance"],
                            ):
                                await self.event_broker.publish(
                                    "app:call:response_gen",
                                    json.dumps(
                                        {
                                            "type": "gen_chunk",
                                            "chunk": parsed_out["phone_utterance"][
                                                len(last_phone_utterance) :
                                            ],
                                        },
                                    ),
                                )
                            last_phone_utterance = parsed_out["phone_utterance"]
            await self.event_broker.publish(
                "app:call:response_gen",
                json.dumps({"type": "end_gen"}),
            )
            # print(parsed_out)
            self.state.active_conversations[self.call_contact.id].push_message(
                "phone",
                ThreadMessage("You", last_phone_utterance, datetime.now()),
            )

        else:
            out = await self.openai_client.responses.parse(
                model="gpt-4.1",
                instructions=Template(SYS).render(
                    name=self.user_name,
                    number=self.user_number,
                ),
                # input=self.chat_history + input_message,
                input=input_message,
                text_format=RESPONSES_MODEL[self.mode],
            )
            parsed_out = out.output[0].content[0].parsed.model_dump()
            out = out.output[0].content[0].text

        print(parsed_out)
        self.state.clear_notifications(now)
        if parsed_out["actions"] is not None:
            for action in parsed_out["actions"]:
                if action["action_name"] == "send_sms":
                    print("sending sms message")
                    contact_num_id = (
                        action["contact_id"]
                        if action["contact_id"] != "-1"
                        else action["number"]
                    )
                    # check if contact exists if not create a new contact
                    contact = self.phone_contacts_map.get(
                        contact_num_id,
                    ) or self.inverted_contacts_map.get(contact_num_id)
                    print(self.inverted_contacts_map)
                    print("contact", contact)
                    if contact is None:
                        contact = ConversationContact(
                            id=str(len(self.inverted_contacts_map) + 1),
                            name=f"{action['first_name']} {action['last_name'] or ''}".strip(),
                            is_boss=False,
                            number=action["number"],
                            email=None,
                        )
                        self.phone_contacts_map[contact.number] = contact
                        self.inverted_contacts_map[contact.id] = contact
                        self.state.push_notif(
                            "comms",
                            f"New contact created and added to contacts `{contact.name}`",
                            datetime.now(),
                        )

                    # overwrite contact number
                    if action.get("number") and contact.number != action.get("number"):
                        old_num = contact.number
                        contact.number = action.get("number")
                        self.phone_contacts_map[contact.number] = contact
                        self.state.push_notif(
                            "comms",
                            f"contact `{contact.name}` number was updated implicitly from {old_num} to {contact.number}",
                            datetime.now(),
                        )

                    res = await _send_sms_message_via_number(
                        contact.number,
                        action["message"],
                    )
                    if not res["success"]:
                        # self.state.push_notif("comms", f"Attempted to send an SMS to an invalid number {contact.number}", datetime.now())
                        await self.event_broker.publish(
                            "app:comms:error",
                            Error(
                                f"Attempted to send an SMS to an invalid number {contact.number}. Make sure the number is correct.",
                            ).to_json(),
                        )

                    else:
                        event = SMSSent(
                            contact=contact.number,
                            content=action["message"],
                        )
                        # self.state.push_event(event)
                        await self.event_broker.publish(
                            "app:comms:sms_sent",
                            event.to_json(),
                        )
                elif action["action_name"] == "send_email":
                    print("sending email")
                    contact_email_id = (
                        action["contact_id"]
                        if action["contact_id"] != "-1"
                        else action["email"]
                    )
                    contact = self.email_contacts_map.get(
                        contact_email_id,
                    ) or self.inverted_contacts_map.get(contact_email_id)
                    print(self.inverted_contacts_map)
                    print("contact", contact)
                    if contact is None:
                        contact = ConversationContact(
                            id=str(len(self.inverted_contacts_map) + 1),
                            name=f"{action['first_name']} {action['last_name'] or ''}".strip(),
                            is_boss=False,
                            email=action["email"],
                        )
                        self.email_contacts_map[contact.email] = contact
                        self.inverted_contacts_map[contact.id] = contact
                        self.state.push_notif(
                            "comms",
                            f"New contact created and added to contacts `{contact.name}`",
                            datetime.now(),
                        )
                    if action.get("email") and contact.email != action.get("email"):
                        print("setting new email....")
                        old_email = contact.email
                        contact.email = action.get("email")
                        self.email_contacts_map[contact.email] = contact
                        self.state.push_notif(
                            "comms",
                            f"contact `{contact.name}` email was updated implicitly from {old_email} to {contact.email}",
                            datetime.now(),
                        )

                    await _send_email_via_address(
                        contact.email,
                        action["subject"],
                        action["body"],
                        action.get("messge_id"),
                    )
                    event = EmailSent(
                        contact=contact.email,
                        subject=action["subject"],
                        body=action["body"],
                        message_id=action.get("message_id"),
                    )
                    # self.state.push_event(event)
                    await self.event_broker.publish(
                        "app:comms:email_sent",
                        event.to_json(),
                    )
                elif action["action_name"] == "make_call":
                    print("calling...")
                    contact_num_id = (
                        action["contact_id"]
                        if action["contact_id"] != "-1"
                        else action["number"]
                    )
                    contact = self.phone_contacts_map.get(
                        contact_num_id,
                    ) or self.inverted_contacts_map.get(contact_num_id)
                    print(self.inverted_contacts_map)
                    print("contact", contact)
                    if contact is None:
                        contact = ConversationContact(
                            id=str(len(self.inverted_contacts_map) + 1),
                            name=f"{action['first_name']} {action['last_name'] or ''}".strip(),
                            is_boss=False,
                            number=action["number"],
                            email=None,
                        )
                        self.phone_contacts_map[contact.number] = contact
                        self.inverted_contacts_map[contact.id] = contact
                        self.state.push_notif(
                            "comms",
                            f"New contact created and added to contacts `{contact.name}`",
                            datetime.now(),
                        )
                    if action.get("number") and contact.number != action.get("number"):
                        old_num = contact.number
                        contact.number = action.get("number")
                        self.phone_contacts_map[contact.number] = contact
                        self.state.push_notif(
                            "comms",
                            f"contact `{contact.name}` number was updated implicitly from {old_num} to {contact.number}",
                            datetime.now(),
                        )

                    res = await _start_call(self.assistant_number, contact.number)
                    if not res["success"]:
                        await self.event_broker.publish(
                            "app:comms:error",
                            Error(res["error"]).to_json(),
                        )
                    else:
                        await self.event_broker.publish(
                            "app:comms:call_initiated",
                            PhoneCallSent(contact=contact.number).to_json(),
                        )

        self.chat_history.append(input_message[0])
        self.chat_history.append({"role": "assistant", "content": out})
        # print(self.chat_history)

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
            await pubsub.psubscribe(
                "app:comms:*", "app:conductor:*", "app:managers:output"
            )

            # fetch contacts if env vars are already set
            if self.assistant_id:
                asyncio.create_task(self.publish_startup())
                print("Default startup")

            while True:
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
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
                    elif isinstance(event, ManagersStartupOutput):
                        if not event.initialized:
                            raise Exception("Managers failed to initialize")
                        self.initialized = True
                        continue
                    elif isinstance(event, StartupEvent):
                        payload = event.to_dict()["payload"]
                        self.set_details(payload)
                        kwargs = {
                            "job_name": self.job_name,
                            "timestamp": payload["timestamp"],
                            "medium": payload["medium"],
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
                        await self.publish_startup()
                        asyncio.create_task(
                            asyncio.to_thread(log_job_startup, **kwargs),
                        )
                        continue
                    await self.handle_event(event)

    def set_details(self, payload):
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
        self.current_user = {
            "user_name": self.user_name,
            "user_number": self.user_number,
            "user_whatsapp_number": self.user_whatsapp_number,
            "user_email": self.user_email,
        }
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        os.environ["UNIFY_KEY"] = payload.pop("api_key")
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

    def update_contact(self, contact):
        self.phone_contacts_map[contact.number] = contact
        self.email_contacts_map[contact.email] = contact
        self.inverted_contacts_map[contact.id] = contact
        self.state.phone_contacts_map = self.phone_contacts_map
        self.state.email_contacts_map = self.email_contacts_map
        self.state.inverted_contacts_map = self.inverted_contacts_map

    async def publish_startup(self):
        print("publishing startup")
        await self.event_broker.publish(
            "app:managers:input",
            ManagersStartupInput(
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

    async def publish_transcript(self, event: Event):
        event_name = event.to_dict()["event_name"].lower()
        print("publishing transcript", event_name)
        medium = (
            "phone_call"
            if "phone" in event_name
            else (
                "sms_message"
                if "sms" in event_name
                else ("email" if "email" in event_name else "whatsapp_message")
            )
        )
        role = (
            "Assistant" if "sent" in event_name or "assistant" in event_name else "User"
        )
        if isinstance(event, (EmailSent, EmailRecieved)):
            content = event.subject + "\n\n" + event.body
        else:
            content = event.content

        contact_id = None
        contacts_map = {
            **self.state.email_contacts_map,
            **self.state.phone_contacts_map,
        }
        if event.contact in contacts_map:
            contact_id = contacts_map[event.contact].id
        if role == "Assistant":
            sender_id, receiver_ids = 0, [contact_id]
        else:
            sender_id, receiver_ids = contact_id, [0]

        await self.event_broker.publish(
            "app:managers:input",
            LogMessageInput(
                medium=medium,
                sender_id=sender_id,
                receiver_ids=receiver_ids,
                content=content,
                exchange_id=UNASSIGNED,
                metadata=None,
            ).to_json(),
        )

    async def handle_event(self, event: Event):
        # add placeholder contact if we're yet to populate the contacts map
        if not self.initialized and hasattr(event, "contact"):
            contact = ConversationContact(
                id=1,
                name="Placeholder",
                number=event.contact,
                email=event.contact,
                is_boss=True,
            )
            self.update_contact(contact)
            print("Placeholder contact created")

        # push event to state
        self.state.push_event(event)

        if isinstance(event, (PhoneCallRecieved, PhoneCallSent)):
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
                    event.contact,
                    self.assistant_number,
                    self.voice_provider,
                    self.voice_id if self.voice_id else "None",
                    "None",
                    str(False),
                )

        elif isinstance(event, PhoneCallStarted):
            self.mode = "call"
            contact = self.phone_contacts_map.get(event.contact)
            self.call_contact = contact
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, PhoneCallEnded):
            self.mode = "text"
            self.call_contact = None
            self.cleanup_call_proc()

        elif isinstance(event, PhoneUtterance):
            asyncio.create_task(self.publish_transcript(event))
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, GetContactsOutput):
            conversation_contacts = [
                ConversationContact(
                    c["id"], c["name"], c["id"] == 1, c["number"], c["email"]
                )
                for c in event.contacts
            ]
            for c in conversation_contacts:
                self.update_contact(c)

        elif isinstance(event, LogMessageOutput):
            # ToDo: get exchange id handled properly
            pass

        elif isinstance(event, Error):
            await self.schedule_llm_run(0, cancel_running=True)

        else:
            # otherwise (whatsapp, sms, email) just schedule another llm run after 2 seconds
            # if there is no response at the moment, if there is a response, cancel it, and scheduel
            # check if there is a scheduled response, reschedule
            if isinstance(event, (SMSSent, SMSRecieved, EmailSent, EmailRecieved)):
                asyncio.create_task(self.publish_transcript(event))
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

    def cleanup_call_proc(self):
        if hasattr(self, "call_proc") and self.call_proc:
            print(f"Terminating call process")
            try:
                terminate_process(self.call_proc)
                self.call_proc = None
                print(f"Call process terminated")
            except Exception as e:
                print(f"Error terminating call process: {e}")

    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.job_name} done")
        mark_job_done(self.job_name)
        self.cleanup_call_proc()


# think about the end behaviour (how the events should look like in the end)
# and design the system around it
