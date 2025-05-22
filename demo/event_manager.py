# The event manager job is to listen for events coming from GUI and call processes
# and send llm responses to both / ui updates
# the event manager will accumulate events and trigger an llm call when timeout happens or
# urgent event is sent, and cancel any running llm calls
import asyncio
import json

import openai
from dotenv import load_dotenv

load_dotenv()
import os

from actions_2 import AssistantOutput, CallAssistantOutput
from events import *

client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


with open("call_sys.md") as f:
    call_sys = f.read()

with open("non_call_sys.md") as f:
    non_call_sys = f.read()


class EventManager:
    def __init__(self):
        self.servers = {}
        self.readers = {}
        self.writers: dict[str, asyncio.StreamWriter] = {}

        self.in_call = False
        self.events_queue = asyncio.Queue()
        self.past_events = []
        self.pending_events = []
        self.inflight_events = []
        self.running_agent = None

    async def serve(self):
        self.servers["gui"] = await asyncio.start_server(
            self.handle_gui_client,
            "127.0.0.1",
            8888,
        )
        self.servers["call"] = await asyncio.start_server(
            self.handle_call_client,
            "127.0.0.1",
            8889,
        )

        self.event_aggregator_task = asyncio.create_task(self.collect_events())
        async with self.servers["gui"], self.servers["call"]:
            await asyncio.gather(
                self.servers["gui"].serve_forever(),
                self.servers["call"].serve_forever(),
            )

    async def handle_gui_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.readers["gui"] = reader
        self.writers["gui"] = writer

        print("GUI connected")
        while True:
            try:
                raw = await reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                # handle msg
                if msg["type"] == "user_agent_event":
                    if msg["to"] == "past":
                        self.past_events.append(msg["event"])
                    elif msg["to"] == "pending":
                        print("msg", msg)
                        self.events_queue.put_nowait(msg["event"])
                else:
                    ...
            except:
                writer.close()
                await writer.wait_closed()

    async def handle_call_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.readers["call"] = reader
        self.writers["call"] = writer

        print("Call connected")
        self.in_call = True
        while True:
            try:
                raw = await reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                # handle msg
                if msg["type"] == "user_agent_event":
                    if msg["event"]["event_name"].startswith("InterruptEvent"):
                        if self.running_agent:
                            self.running_agent.cancel()
                            try:
                                await self.running_agent
                            except asyncio.CancelledError:
                                pass
                    if msg["to"] == "past":
                        self.past_events.append(msg["event"])
                        print("adding to past events")
                    elif msg["to"] == "pending":
                        # print("msg", msg)
                        self.events_queue.put_nowait(msg["event"])
                else:
                    ...
            except Exception as e:
                print(str(e))
                print("CALL CLOSED")
                self.past_events.append(PhoneCallEndedEvent().to_dict())
                self.in_call = False
                writer.close()
                await writer.wait_closed()

    async def collect_events(self):
        print("COLLECTING...")
        while True:
            try:
                new_event = await asyncio.wait_for(self.events_queue.get(), 3)
                print(new_event)
                if new_event["payload"]["transient"]:
                    continue
                self.pending_events.append(new_event)
                # urgent events should re-trigger, cancel events should cancel current running only
                if new_event["payload"]["is_urgent"]:
                    # must flush all events now
                    if self.running_agent and not self.running_agent.done():
                        self.running_agent.cancel()
                        try:
                            # cancel gracefully
                            await self.running_agent
                        except asyncio.CancelledError:
                            self.inflight_events = [
                                *self.inflight_events,
                                *self.pending_events,
                            ]
                    else:
                        self.inflight_events = self.pending_events.copy()

                    self.running_agent = asyncio.create_task(self.run())
                    self.running_agent.add_done_callback(self.on_run_end)
                    self.pending_events.clear()
            except asyncio.TimeoutError:
                if not self.pending_events:
                    continue
                if self.running_agent and not self.running_agent.done():
                    continue

                self.inflight_events = self.pending_events.copy()
                self.running_agent = asyncio.create_task(self.run())
                self.running_agent.add_done_callback(self.on_run_end)

                self.pending_events.clear()

    def on_run_end(self, t: asyncio.Task):
        try:
            t: AssistantOutput | CallAssistantOutput | None = t.result()
            # everything is fine, just run the actions and add stuff to past events
            print("t", t, type(t))
            if t:
                # if self.in_call:
                self.past_events.extend(self.inflight_events.copy())
                self.inflight_events.clear()

                gui_writer = self.writers.get("gui")

                # this should launch async tasks
                if t.actions is not None:
                    for action in t.actions:
                        if self.writers.get("gui"):
                            print("creating tasks")
                            # For WhatsApp messages, include phone numbers in the message content
                            if action.type in ["whatsapp", "sms"]:
                                # Find phone numbers from inflight events
                                phone_numbers = {}
                                for event in self.past_events[::-1]:
                                    if event.get("payload", {}).get("content"):
                                        try:
                                            content = json.loads(event["payload"]["content"])
                                            if "to_number" in content and "from_number" in content:
                                                phone_numbers = {
                                                    "to_number": content["to_number"],
                                                    "from_number": content["from_number"]
                                                }
                                                break
                                        except json.JSONDecodeError:
                                            continue

                                # Create message content with phone numbers
                                message_content = json.dumps({
                                    "message": action.message,
                                    **phone_numbers
                                })
                            else:
                                message_content = action.message

                            asyncio.create_task(
                                self.send_event(
                                    gui_writer,
                                    {
                                        "type": "update_gui",
                                        "thread": action.type,
                                        "content": message_content,
                                    },
                                ),
                            )

                            events_map = {
                                "whatsapp": WhatsappMessageSentEvent,
                                "telegram": TelegramMessageSentEvent,
                                "sms": SMSMessageSentEvent,
                            }

                            event = events_map[action.type](
                                content=action.message,
                            ).to_dict()
                            if self.in_call:
                                self.events_queue.put_nowait(event)
                            else:
                                self.past_events.append(event)
        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        if self.in_call:
            return await self.run_call_agent()
        else:
            return await self.run_non_call_agent()

    async def run_call_agent(self):
        call_w = self.writers.get("call")
        gui_w = self.writers.get("gui")
        ev = json.dumps({"type": "start_gen"}) + "\n"
        call_w.write(ev.encode())
        await call_w.drain()

        user_msg = self.get_user_agent_prompt()
        print(user_msg)

        async with client.beta.chat.completions.stream(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": call_sys,
                },
                {
                    "role": "user",
                    "content": user_msg,
                },
            ],
            response_format=CallAssistantOutput,
        ) as stream:

            async for event in stream:
                # print(event)
                if event.type == "content.delta":
                    ev = json.dumps({"type": "gen_chunk", "chunk": event.delta}) + "\n"
                    call_w.write(ev.encode())
                    await call_w.drain()
            ev = json.dumps({"type": "end_gen"}) + "\n"
            call_w.write(ev.encode())
            await call_w.drain()
        self.past_events.extend(self.inflight_events.copy())
        self.inflight_events.clear()
        return event.parsed

    async def run_non_call_agent(self):
        print("Running...")
        w = self.writers["gui"]
        user_msg = self.get_user_agent_prompt()
        print(user_msg)
        res = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": non_call_sys,
                },
                {
                    "role": "user",
                    "content": user_msg,
                },
            ],
            response_format=AssistantOutput,
        )
        message = res.choices[0].message
        print(message)
        print("parsed: ", message.parsed)
        if message.parsed:
            return message.parsed

    def get_user_agent_prompt(self):
        past_events_str = (
            "\n".join([str(Event.from_dict(e)) for e in self.past_events])
            if self.past_events
            else ""
        )
        new_events_str = "\n".join(
            str(Event.from_dict(e)) for e in self.inflight_events
        )

        task_status_str = "No Tasks are running"  # TODO
        user_msg = f"""Events Log:
** PAST EVENTS **
{past_events_str.strip()}
** NEW EVENTS **
{new_events_str.strip()}


# Tasks status:
# {task_status_str.strip()}"""
        return user_msg

    async def send_event(self, writer: asyncio.StreamWriter, event: dict):
        ev = json.dumps(event) + "\n"
        writer.write(ev.encode())
        await writer.drain()


if __name__ == "__main__":
    event_manager = EventManager()
    asyncio.run(event_manager.serve())
