import os
import asyncio

# import threading
from jinja2 import Template
import json
import contextlib
from pathlib import Path

from unity.conversation_manager_2.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.actions import (
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


with open(Path(__file__).parent.resolve() / "prompts" / "v2.md") as f:
    SYS = f.read()


MAX_CONV_MANAGER_MSGS = 30

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
    ):
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

        self.chat_history = self.state.chat_history
        self.call_proc = None
        # self.summarizing = False

    async def run_llm(self):
        self.state.snapshot()
        # print(DUMMY_EVENT_BUS)
        prompt = self.state.get_state_for_llm()
        print(prompt)
        input_message = {"role": "user", "content": prompt}
        boss_contact = next(
            c for c in self.state.inverted_contacts_map.values() if c.is_boss
        )
        system_message = Template(SYS).render(
            contact_id=boss_contact.contact_id,
            first_name=boss_contact.first_name,
            surname=boss_contact.surname,
            phone_number=boss_contact.phone_number,
            email_address=boss_contact.email_address,
        )
        print(system_message)

        # Use dynamic response models (set_details must be called before run_llm)
        response_model = self.state.dynamic_response_models[self.state.mode]
        if self.state.mode in ["call", "gmeet"]:
            print("running...")
            first_chunk = True
            async for event in stream_llm_call(
                self.openai_client,
                system_message,
                self.state.chat_history + [input_message],
                "gpt-4.1",
                response_model,
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
                self.state.phone_contact.phone_number,
                parsed_out["phone_utterance"],
            )
            await self.event_broker.publish(
                "app:comms:phone_utterance",
                assistant_phone_utterance_event.to_json(),
            )

        else:
            out = await llm_call(
                self.openai_client,
                system_message,
                self.state.chat_history + [input_message],
                response_model=response_model,
            )
            parsed_out = json.loads(out)

        print(parsed_out)
        if parsed_out["actions"] is not None:
            for action in parsed_out["actions"]:
                if action["action_name"] == "send_sms":
                    contact = self.state.update_or_create_new_contact(
                        action["contact_id"],
                        action["first_name"],
                        action["surname"],
                        phone_number=action["phone_number"],
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
                        action["surname"],
                        email_address=action["email_address"],
                    )
                    await _send_email_via_address(
                        contact.email_address,
                        action["subject"],
                        action["body"],
                        action.get("messge_id"),
                    )
                    event = EmailSent(
                        contact=contact.email_address,
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
                        action["surname"],
                        phone_number=action["phone_number"],
                    )
                    print("contact found=", contact)
                    if self.state.mode == "call":
                        error = Error(
                            "You can not make a call while on a call, wait till the call ends.",
                        )
                        await self.event_broker.publish(
                            "app:comms:call_initiated",
                            error.to_json(),
                        )
                    else:
                        res = await _start_call(
                            self.state.assistant_number,
                            contact.phone_number,
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
                elif action["action_name"] == "send_unify_message":
                    # Boss-only chat; contact id is always "1"
                    content = action["message"]
                    event = UnifyMessageSent(contact="1", content=content)
                    await self.event_broker.publish(
                        "app:comms:unify_message_sent",
                        event.to_json(),
                    )
        # obviously all ops here should be "atomic", but that's an edge case for
        # another day...
        self.state.commit()
        self.state.chat_history.append(input_message)
        # DUMMY_EVENT_BUS.append(LLMInput(content=input_message["content"]))
        # event = LLMInput(content=input_message["content"])
        # asyncio.create_task(self.publish_bus_events(event))
        self.state.chat_history.append({"role": "assistant", "content": out})
        # DUMMY_EVENT_BUS.append(LLMOutput(content=out))
        event = LLMInput(content=self.state.chat_history)
        asyncio.create_task(self.publish_bus_events(event))

        print("**NUMBER OF MESSAGES **", len(self.state.chat_history))
        if (
            len(self.state.chat_history) >= MAX_CONV_MANAGER_MSGS
            and not self.state.summarizing
        ):
            print("CLEARING CHAT HISTORY, REACHED MAX NUM")
            # self.chat_history = []
            # DUMMY_EVENT_BUS.append(ClearContext())
            try:
                event = UpdateContactRollingSummaryRequest(
                    contacts_ids=[
                        int(c.contact_id)
                        for c in self.state.active_conversations.values()
                    ],
                    transcripts=[
                        self.state._render_contact_threads(c)
                        for c in self.state.active_conversations.values()
                    ],
                )
                print(event)
                asyncio.create_task(
                    self.event_broker.publish("app:managers:input", event.to_json()),
                )
                self.state.summarizing = True
                print("sent")
            except Exception as e:
                print(e)
                raise

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
                "app:comms:*",
                "app:conductor:*",
                "app:managers:output",
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
                    # process events
                    event = Event.from_json(msg["data"])
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
                else (
                    "email"
                    if "email" in event_name
                    else (
                        "unify_message" if "unify" in event_name else "whatsapp_message"
                    )
                )
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
        if isinstance(event, (UnifyMessageSent, UnifyMessageRecieved)):
            contact_id = 1
        elif event.contact in contacts_map:
            contact_id = contacts_map[event.contact].contact_id
        if role == "Assistant":
            sender_id, receiver_ids = 0, [contact_id]
        else:
            sender_id, receiver_ids = contact_id, [0]

        exchange_id = UNASSIGNED
        if medium == "phone_call":
            exchange_id = self.state.call_exchange_id

        call_utterance_timestamp = ""
        call_url = ""
        if self.state.call_start_timestamp:
            delta = datetime.now() - self.state.call_start_timestamp
            minutes, seconds = divmod(int(delta.total_seconds()), 60)
            # ToDo: Make this MM:SS once we have explicit types working
            call_utterance_timestamp = f"{minutes:02d}.{seconds:02d}"
        if "default-assistant" not in self.state.assistant_id:
            call_url = (
                "https://storage.cloud.google.com/assistant-call-recordings/staging/"
                f"{self.state.assistant_id}/{self.state.conference_name}.mp3"
            )

        await self.event_broker.publish(
            "app:managers:input",
            LogMessageInput(
                medium=medium,
                sender_id=sender_id,
                receiver_ids=receiver_ids,
                content=content,
                exchange_id=exchange_id,
                call_utterance_timestamp=call_utterance_timestamp,
                call_url=call_url,
                metadata=None,
            ).to_json(),
        )

    async def handle_event(self, event: Event):
        # update state
        self.state.update_state(event)

        # Centralized handler for steering notifications from ConversationManagerHandle
        if isinstance(event, NotificationInjectedEvent):
            # Check if this notification is intended for this CM instance
            if event.target_conversation_id == self.state.assistant_id:
                print(f"INFO: Received steering notification: '{event.content}'")
                await self.schedule_llm_run(delay=0.1, cancel_running=True)
            return

        # every interaction with the managers worker happens through the conversation
        # manager instead of the state, which is why we need to publish the events here
        # if event.__class__.loggable:
        #     asyncio.create_task(self.publish_bus_events(event))

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
            if isinstance(
                event,
                (
                    SMSSent,
                    SMSRecieved,
                    EmailSent,
                    EmailRecieved,
                    UnifyMessageSent,
                    UnifyMessageRecieved,
                ),
            ):
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
                self.state.call_start_timestamp = None
                self.state.conference_name = ""
                print(f"Call process terminated")
            except Exception as e:
                print(f"Error terminating call process: {e}")

    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.state.job_name} done")
        mark_job_done(self.state.job_name)
        self.cleanup_call_proc()
        self.stop.set()
