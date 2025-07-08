import asyncio
import json
import openai
import os
import traceback

from unity.helpers import run_script, terminate_process
from unity.conversation_manager.comms_actions import _start_call
from unity.conversation_manager.actions import *
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import (
    build_call_sys_prompt,
    build_non_call_sys_prompt,
)

client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

ONGOING_CALL = False


# new events to add:
# task status update
#


class WhatsappQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    def add_message_task(self, mt):
        self.queue.put_nowait(mt)

    async def run(self):
        while True:
            task = await self.queue.get()
            await task
            await asyncio.sleep(0.5)


class CommsAgent:
    def __init__(
        self,
        user_name: str,
        assistant_number: str,
        user_number: str,
        user_phone_call_number: str = None,
        past_events: list | None = None,
        conv_context_length: int = 50,
        with_conductor: bool = True,
    ):
        # contact data
        self.with_conductor = with_conductor
        self.assistant_number = assistant_number
        self.user_name = user_name
        self.user_number = user_number
        self.user_phone_call_number = (
            user_phone_call_number if user_phone_call_number else user_number
        )

        # events (history)
        self.conv_context_length = conv_context_length
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.past_events = past_events
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

        # switches to "True" when in a call
        self.call_mode = False
        self.call_purpose = "general"

        # conductor
        self.conductor = None
        self.conductor_handles = None
        self.handle_count = 0

        # logging
        self.transcript_manager = None

    async def get_bus_events(self):
        from unity.events.event_bus import EVENT_BUS

        await EVENT_BUS._prefill_done.wait()
        bus_events = await EVENT_BUS.search(
            filter=f"event_type in {json.dumps(EVENT_TYPES)}",
            limit=self.conv_context_length,
        )

        return [Event.from_bus_event(e).to_dict() for e in bus_events]

    def get_chat_history(self):
        chat_history = []
        for event in self.past_events:
            if event["event_name"] == "PhoneUtteranceEvent":
                chat_history.append(
                    {
                        "role": event["payload"]["role"].lower(),
                        "content": event["payload"]["content"],
                    },
                )
        return chat_history

    async def listen_for_events(self):
        print("COLLECTING...")
        while True:
            try:
                new_event = await asyncio.wait_for(self.events_queue.get(), 1)
                # print("comm agent got", new_event)
                # continue
                if new_event["payload"]["transient"]:
                    continue
                if new_event["event_name"] == "PhoneCallInitiatedEvent":
                    global ONGOING_CALL
                    if not ONGOING_CALL:
                        self.call_purpose = new_event["payload"]["purpose"]
                        self.call_proc = run_script(
                            "unity/conversation_manager/call.py",
                            "dev",
                            self.user_phone_call_number,  # "console" if a local call is needed
                            self.assistant_number,
                            (
                                new_event["tts_provider"]
                                if new_event["tts_provider"]
                                else "cartesia"
                            ),
                            new_event["voice_id"] if new_event["voice_id"] else "None",
                            "--outbound" if new_event.get("outbound") else "None",
                        )
                        self.call_mode = True
                        ONGOING_CALL = True
                        continue
                    else:
                        # append initated phone call and failed
                        ...

                self.pending_events.append(new_event)
                # urgent events should re-trigger, cancel events should cancel current running only
                if new_event["payload"]["is_urgent"]:
                    # must flush all events now
                    if self.current_llm_run and not self.current_llm_run.done():
                        self.current_llm_run.cancel()
                        try:
                            # cancel gracefully
                            await self.current_llm_run
                        except asyncio.CancelledError:
                            self.inflight_events = [
                                *self.inflight_events,
                                *self.pending_events,
                            ]
                    else:
                        self.inflight_events = self.pending_events.copy()

                    self.current_llm_run = asyncio.create_task(self.run())
                    self.current_llm_run.add_done_callback(self.on_run_end)
                    self.pending_events.clear()
            except asyncio.TimeoutError:
                if not self.pending_events:
                    continue
                if self.current_llm_run and not self.current_llm_run.done():
                    continue

                self.inflight_events = self.pending_events.copy()
                self.current_llm_run = asyncio.create_task(self.run())
                self.current_llm_run.add_done_callback(self.on_run_end)

                self.pending_events.clear()

    async def conductor_action(self, action: ConductorAction):
        """Handle conductor actions asynchronously"""
        from unity.conductor.conductor import Conductor
        from unity.common.llm_helpers import AsyncToolUseLoopHandle

        if self.conductor is None:
            self.conductor = Conductor()
            self.conductor_handles: dict[int, dict[AsyncToolUseLoopHandle, str]] = {}

        # get chat history
        chat_history = self.get_chat_history()

        # query the conductor
        fn = self.conductor.ask if action.type == "ask" else self.conductor.request
        conductor_handle = await fn(
            action.query,
            parent_chat_context=chat_history,
            _return_reasoning_steps=action.show_steps,
        )
        handle_id = self.handle_count
        self.conductor_handles[handle_id] = {
            "handle": conductor_handle,
            "query": action.query,
        }
        self.handle_count += 1

        # publish start event
        self.publish(
            {
                "topic": "conductor",
                "to": "past",
                "event": ConductorStartedEvent(chat_history, action.query).to_dict(),
            },
        )

        # wait for the handle to be done
        while not conductor_handle.done():
            print("waiting for handle to be done")
            await asyncio.sleep(1)

        # get handle result
        answer = await conductor_handle.result()
        self.conductor_handles.pop(handle_id)
        if isinstance(answer, tuple):
            answer, _ = answer

        # publish end event
        self.publish(
            {
                "topic": "conductor",
                "event": ConductorEndedEvent(answer).to_dict(),
            },
        )

    async def conductor_handle_action(self, action: ConductorHandleAction):
        """Handle conductor handle actions asynchronously"""
        # check if the conductor is running
        if self.conductor_handles is None or not self.conductor_handles.get(
            action.handle_id,
        ):
            # handle failed
            event_data = {
                "event": ConductorHandleFailedEvent(
                    f"conductor is not running currently, "
                    "please create a new action instead",
                    action.type,
                ).to_dict(),
            }
        else:
            # handle
            handle = self.conductor_handles[action.handle_id]["handle"]
            if action.type == "ask":
                await handle.ask(action.query)
            elif action.type == "interject":
                await handle.interject(action.query)
            elif action.type == "stop":
                handle.stop()
            elif action.type == "pause":
                handle.pause()
            elif action.type == "resume":
                handle.resume()
            event_data = {
                "event": ConductorHandleSuccessEvent(
                    action.query,
                    action.type,
                ).to_dict(),
                "to": "past",
            }
        self.publish({"topic": "conductor", **event_data})

    def on_run_end(self, t: asyncio.Task):
        try:
            t: AssistantOutput | CallAssistantOutput | None = t.result()
            # everything is fine, just run the actions and add stuff to past events
            if t:
                # if self.call_mode:
                self.past_events.extend(self.inflight_events.copy())
                self.inflight_events.clear()

                # this should launch async tasks
                if t.actions is not None:
                    print("actions", t.actions)
                    for action in t.actions:
                        if isinstance(action, SendCallAction):
                            asyncio.create_task(self.send_call())
                        elif isinstance(action, ConductorAction):
                            asyncio.create_task(self.conductor_action(action))
                        elif isinstance(action, ConductorHandleAction):
                            asyncio.create_task(self.conductor_handle_action(action))

        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        if self.past_events is None:
            self.past_events = []  # await self.get_bus_events()
        if self.call_mode:
            return await self.phone_call_llm_run()
        else:
            return await self.non_phone_call_llm_run()

    async def non_phone_call_llm_run(self):
        non_call_sys = build_non_call_sys_prompt(
            self.user_name,
            with_conductor=self.with_conductor,
        )
        user_msg = self.get_user_agent_prompt()
        print(user_msg, flush=True)

        res = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": non_call_sys},
                {"role": "user", "content": user_msg},
            ],
            response_format=AssistantOutput,
        )
        message = res.choices[0].message
        # print(message)
        # print("parsed: ", message.parsed)
        if message.parsed:
            return message.parsed

    async def phone_call_llm_run(self):
        ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(ev)

        call_sys = build_call_sys_prompt(
            self.user_name,
            with_conductor=self.with_conductor,
        )

        user_msg = self.get_user_agent_prompt()
        print(user_msg)

        async with client.beta.chat.completions.stream(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": call_sys},
                {"role": "user", "content": user_msg},
            ],
            response_format=CallAssistantOutput,
        ) as stream:

            async for event in stream:
                # print(event)
                if event.type == "content.delta":
                    ev = {
                        "topic": "call_process",
                        "type": "gen_chunk",
                        "chunk": event.delta,
                    }
                    self.publish(ev)

            ev = {"topic": "call_process", "type": "end_gen"}
            self.publish(ev)
        self.past_events.extend(self.inflight_events.copy())
        self.inflight_events.clear()
        return event.parsed

    async def send_call(self):
        print(self.assistant_number, self.user_phone_call_number)
        await _start_call(
            self.assistant_number,
            self.user_phone_call_number,
        )

    async def wait_for_seconds_or_next_event(self, time: int): ...

    def subscribe(self, topics):
        if not self.event_manager:
            raise Exception("Set an event manager first.")
        for topic in topics:
            self.event_manager.topic_to_subs[topic].add(self)

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager

    def get_user_agent_prompt(self):
        past_events_str = (
            "\n".join([str(Event.from_dict(e)) for e in self.past_events])
            if self.past_events
            else ""
        )
        new_events_str = "\n".join(
            str(Event.from_dict(e)) for e in self.inflight_events
        )
        conductor_handles_str = (
            "\n".join(
                f"Handle ID {h}: {self.conductor_handles[h]['query']}"
                for h in self.conductor_handles
            )
            if self.conductor_handles is not None
            else ""
        )
        user_msg = f"""CALL PURPOSE: {self.call_purpose}
Events Stream:
** PAST EVENTS **
{past_events_str.strip()}
** NEW EVENTS **
{new_events_str.strip()}
** CONDUCTOR HANDLES (USE THESE FOR THE CONDUCTOR HANDLE ACTION) **
{conductor_handles_str.strip()}
"""
        return user_msg

    def publish(self, event: dict):
        self.event_manager.publish(event)

    def cleanup(self):
        """Clean up any running call processes"""
        if hasattr(self, "call_proc") and self.call_proc:
            print(f"Terminating call process")
            try:
                terminate_process(self.call_proc)
                self.call_proc = None
                self.call_mode = False
                global ONGOING_CALL
                ONGOING_CALL = False
                print(f"Call process terminated")
            except Exception as e:
                print(f"Error terminating call process: {e}")

    def handle_logging(self, event: dict):
        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.transcript_manager.types.message import Message
        from unity.events.event_bus import EVENT_BUS

        if self.transcript_manager is None:
            self.transcript_manager = TranscriptManager()

        try:
            bus_event = Event.from_dict(event["event"]).to_bus_event()
            asyncio.run(EVENT_BUS.publish(bus_event))
            if event["event"]["event_name"] in [
                "PhoneUtteranceEvent",
                "WhatsappMessageSentEvent",
                "SMSMessageSentEvent",
                "WhatsappMessageRecievedEvent",
                "SMSMessageRecievedEvent",
            ]:
                event_name = event["event"]["event_name"]
                role = event["event"]["payload"]["role"]
                content = event["event"]["payload"]["content"]
                timestamp = event["event"]["payload"]["timestamp"]
                medium = (
                    "phone_call"
                    if "phone" in event_name
                    else "sms_message" if "sms" in event_name else "whatsapp_message"
                )
                sender_id, receiver_id = "", ""
                if medium == "phone_call":
                    if role == "Assistant":
                        sender_id = self.assistant_number
                        receiver_id = self.user_phone_call_number
                    else:
                        sender_id = self.user_phone_call_number
                        receiver_id = self.assistant_number
                else:
                    if "recieved" in event_name.lower():
                        sender_id = self.user_number
                        receiver_id = self.assistant_number
                    else:
                        sender_id = self.assistant_number
                        receiver_id = self.user_number
                self.transcript_manager.log_message(
                    Message(
                        medium=medium,
                        sender_id=sender_id,
                        receiver_id=receiver_id,
                        timestamp=timestamp,
                        content=content,
                    ),
                )
        except Exception as e:
            print(f"Error handling logging: {e}")
            traceback.print_exc()

    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        if event["event"]["event_name"] == "PhoneCallEndedEvent":
            if self.call_proc:
                self.call_proc.kill()
                self.call_proc.wait()
                self.call_proc = None
                self.call_mode = False
                ONGOING_CALL = False
        elif event["event"]["event_name"] == "PhoneCallStopEvent":
            self.publish(
                {
                    "topic": "call_process",
                    "type": "stop",
                },
            )
        if to == "past":
            self.past_events.append(event["event"])
        else:
            self.events_queue.put_nowait(event["event"])
        asyncio.create_task(asyncio.to_thread(self.handle_logging, event))
