import os
import asyncio

# import threading
from jinja2 import Template
import json
from typing import Literal
import contextlib
from pathlib import Path

from unity.conversation_manager_2.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.actions import (
    RESPONSES_MODEL,
    _send_sms_message_via_number,
    _send_email_via_address,
    _start_call,
)
from unity.conversation_manager_2.state import ConversationManagerState
from unity.helpers import run_script, terminate_process
from unity.conversation_manager_2.llm_utils import stream_llm_call, llm_call
from unity.transcript_manager.types.message import UNASSIGNED


import redis.asyncio as redis
from openai import AsyncOpenAI


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

        self.state = ConversationManagerState(
            job_name=job_name,
            user_id=user_id,
            assistant_id=assistant_id,
            assistant_name=assistant_name,
            assistant_age=assistant_age,
            assistant_region=assistant_region,
            assistant_about=assistant_about,
            voice_provider=voice_provider,
            voice_id=voice_id,
            assistant_number=assistant_number,
            assistant_email=assistant_email,
            user_name=user_name,
            user_number=user_number,
            user_email=user_email,
            user_whatsapp_number=user_whatsapp_number,
        )

        self.chat_history = []
        self.call_proc = None

    async def run_llm(self):
        self.state.snapshot()
        prompt = self.state.get_state_for_llm()
        print(prompt)
        input_message = {"role": "user", "content": prompt}
        system_message = Template(SYS).render(
            name=self.state.user_name,
            number=self.state.user_number,
        )
        if self.state.mode in ["call", "gmeet"]:
            print("running...")
            first_chunk = True
            async for event in stream_llm_call(
                self.openai_client,
                system_message,
                self.chat_history + [input_message],
                "gpt-4.1",
                RESPONSES_MODEL[self.state.mode],
                "phone_utterance",
            ):
                if event["type"] == "chunk":
                    if first_chunk:
                        await self.event_broker.publish(
                            "app:call:response_gen",
                            json.dumps({"type": "start_gen"}),
                        )
                        first_chunk = False
                    await self.event_broker.publish(
                        "app:call:response_gen",
                        json.dumps(
                            {"type": "gen_chunk", "chunk": event["content"]},
                        ),
                    )
                elif event["type"] == "end_streamed_field":
                    await self.event_broker.publish(
                        "app:call:response_gen",
                        json.dumps({"type": "end_gen"}),
                    )

            out = event["content"]
            parsed_out = json.loads(out)
            assistant_phone_utterance_event = AssistantPhoneUtterance(
                self.state.phone_contact.phone_number, parsed_out["phone_utterance"]
            )
            await self.event_broker.publish(
                "app:comms:phone_utterance",
                assistant_phone_utterance_event.to_json(),
            )

        else:
            out = await llm_call(
                self.openai_client,
                system_message,
                self.chat_history + [input_message],
                response_model=RESPONSES_MODEL[self.state.mode],
            )
            parsed_out = json.loads(out)

        print(parsed_out)
        if parsed_out["actions"] is not None:
            for action in parsed_out["actions"]:
                if action["action_name"] == "send_sms":
                    contact = self.state.update_or_create_new_contact(
                        action["contact_id"],
                        action["first_name"],
                        action["last_name"],
                        phone_number=action["number"],
                    )
                    res = await _send_sms_message_via_number(
                        contact.phone_number,
                        action["message"],
                    )
                    if not res["success"]:
                        # self.state.push_notif("comms", f"Attempted to send an SMS to an invalid number {contact.number}", datetime.now())
                        await self.event_broker.publish(
                            "app:comms:error",
                            Error(
                                f"Attempted to send an SMS to an invalid number {contact.phone_number}. Make sure the number is correct.",
                            ).to_json(),
                        )

                    else:
                        event = SMSSent(
                            contact=contact.phone_number,
                            content=action["message"],
                        )
                        # self.state.push_event(event)
                        await self.event_broker.publish(
                            "app:comms:sms_sent",
                            event.to_json(),
                        )
                elif action["action_name"] == "send_email":
                    print("sending email")
                    contact = self.state.update_or_create_new_contact(
                        action["contact_id"],
                        action["first_name"],
                        action["last_name"],
                        email=action["email"],
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
                    contact = self.state.update_or_create_new_contact(
                        action["contact_id"],
                        action["first_name"],
                        action["last_name"],
                        phone_number=action["number"],
                    )
                    if self.state.mode == "call":
                        error = Error(
                            "You can not make a call while on a call, wait till the call ends."
                        )
                        await self.event_broker.publish(
                            "app:comms:call_initiated",
                            error.to_json(),
                        )
                    else:
                        res = await _start_call(
                            self.state.assistant_number, contact.phone_number
                        )
                        if not res["success"]:
                            await self.event_broker.publish(
                                "app:comms:error",
                                Error(res["error"]).to_json(),
                            )
                        else:
                            await self.event_broker.publish(
                                "app:comms:call_initiated",
                                PhoneCallSent(contact=contact.phone_number).to_json(),
                            )
        # obviously all three ops here should be "atomic", but that's an edge case for
        # another day...
        self.state.commit()
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": out})

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
            if self.state.assistant_id:
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
                    event = Event.from_json(msg["data"])  # type: ignore[arg-type]
                    print(event)
                    await self.handle_event(event)

    async def publish_startup(self):
        print("publishing startup")
        await self.event_broker.publish(
            "app:managers:input",
            ManagersStartupInput(
                agent_id=self.state.assistant_id,
                first_name=self.state.assistant_name,
                age=self.state.assistant_age,
                region=self.state.assistant_region,
                about=self.state.assistant_about,
                phone=self.state.assistant_number,
                email=self.state.assistant_email,
                user_phone=self.state.user_number,
                user_whatsapp_number=self.state.user_whatsapp_number,
                assistant_whatsapp_number=self.state.assistant_number,
            ).to_json(),
        )

    async def publish_bus_events(self, event: Event):
        await self.event_broker.publish(
            "app:managers:input",
            PublishBusEvent(event=event.to_dict()).to_json(),
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

        exchange_id = UNASSIGNED
        if medium == "phone_call":
            exchange_id = self.state.call_exchange_id

        await self.event_broker.publish(
            "app:managers:input",
            LogMessageInput(
                medium=medium,
                sender_id=sender_id,
                receiver_ids=receiver_ids,
                content=content,
                exchange_id=exchange_id,
                metadata=None,
            ).to_json(),
        )

    async def handle_event(self, event: Event):
        # add placeholder contact if we're yet to populate the contacts map.
        # TODO: this is a hack to get the conversation manager to work without managers.
        # TODO: Needs to be removed as soon as manager instantiation is quick enough.
        if not self.state.initialized and hasattr(event, "contact"):
            self.state.create_new_contact(
                id="1",
                first_name="Placeholder",
                last_name="Contact",
                email=event.contact,
                phone_number=event.contact,
            )
            print("Placeholder contact created")

        # update state
        self.state.update_state(event)

        # every interaction with the managers worker happens through the conversation
        # manager instead of the state, which is why we need to publish the events here
        if event.__class__.loggable:
            asyncio.create_task(self.publish_bus_events(event))

        if isinstance(event, (PhoneCallRecieved, PhoneCallSent)):
            # start phone call process and wait untils its done, we should probably make sure
            # first that any running llm calls are awaited, and any scheduled llm calls are canceled
            # llm inference should not start until the process is set up (through PhoneCallStartedEvent)
            if self.state.mode in ["call", "gmeet"]:
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
                    self.state.assistant_number,
                    self.state.voice_provider,
                    self.state.voice_id if self.state.voice_id else "None",
                    "None",
                    str(False),
                )

        elif isinstance(event, PhoneCallStarted):
            # self.mode = "call"
            # contact = self.phone_contacts_map.get(event.contact)
            # self.call_contact = contact
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, PhoneCallEnded):
            self.state.mode = "text"
            self.call_contact = None
            self.cleanup_call_proc()
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, PhoneUtterance):
            asyncio.create_task(self.publish_transcript(event))
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, AssistantPhoneUtterance):
            # do not do anything here, let the user reply back or whatever
            asyncio.create_task(self.publish_transcript(event))

        elif isinstance(event, StartupEvent):
            payload = event.to_dict()["payload"]
            kwargs = {
                "timestamp": payload["timestamp"],
                "medium": payload["medium"],
                **self.state.get_details(),
            }
            await self.publish_startup()
            asyncio.create_task(asyncio.to_thread(log_job_startup, **kwargs))

        elif isinstance(event, Error):
            await self.schedule_llm_run(0, cancel_running=True)

        elif isinstance(event, Ping):
            print("ping received - keeping conversation manager alive")

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
                self.state.call_exchange_id = UNASSIGNED
                print(f"Call process terminated")
            except Exception as e:
                print(f"Error terminating call process: {e}")

    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.job_name} done")
        mark_job_done(self.job_name)
        self.cleanup_call_proc()
        self.stop.set()
