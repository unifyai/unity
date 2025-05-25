import os
import asyncio
import json
from dataclasses import dataclass
from typing import Literal

import openai

import comms_actions
from actions_2 import AssistantOutput, CallAssistantOutput
from events import *
from new_terminal_helper import run_script

client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

ONGOING_CALL = False


with open("call_sys.md") as f:
    call_sys = f.read()

with open("non_call_sys.md") as f:
    non_call_sys = f.read()

@dataclass
class CommTask:
    task_id: int
    contact_name: str
    contact_number: str
    status: str
    task_description: str

# new events to add:
# task status update
# 


class CommsAgent:
    def __init__(
        self,
        user_name: str,
        assistant_number: str,
        user_number: str,
        user_phone_call_number: str = None,
        past_events: list = None,
        main_user_agent: bool = False
    ):
        
        self.main_user = main_user_agent

        self.assistant_number = assistant_number

        # contact data
        self.user_name = user_name
        self.user_number = user_number
        self.user_phone_call_number = (
            user_phone_call_number if user_phone_call_number else user_number
        )

        # events (history)
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.past_events = past_events
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

        # switches to "True" when in a call
        self.call_mode = False

        # tasks attributes
        # if the comm agent performing a specific task
        # should be switched false once all associated tasks are done 
        # (probs the comms agent will be removed all together actually if it is done with its task)
        self.performing_task = False

        # this only makes sense to the "main" comms agent (the user agent)
        self.contact_num_to_comm_agent: dict[str, CommsAgent] = {

        }
        self.running_tasks = []

        

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
                        self.call_proc = run_script(
                            "call.py",
                            "dev",  # "console" if a local call is needed
                            self.user_phone_call_number,
                            # to_number,
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
                    for action in t.actions:
                        # take actions

                        events_map = {
                            "whatsapp": WhatsappMessageSentEvent,
                            "sms": SMSMessageSentEvent,
                        }

                        # should be referenced in a set to avoid being garbage collected
                        # (i think)
                        asyncio.create_task(self.send_whatsapp(action.message))

                        event = events_map[action.type](
                            content=action.message,
                        ).to_dict()
                        if self.call_mode:
                            self.events_queue.put_nowait(event)
                        else:
                            self.past_events.append(event)
        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        if self.call_mode:
            return await self.phone_call_llm_run()
        else:
            return await self.non_phone_call_llm_run()
        
    async def non_phone_call_llm_run(self):
        print("Running...")
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
        
    async def phone_call_llm_run(self):
        ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(ev)

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
                    ev = {"topic": "call_process", "type": "gen_chunk", "chunk": event.delta}
                    self.publish(ev)

            ev = {"topic": "call_process", "type": "end_gen"}
            self.publish(ev)
        self.past_events.extend(self.inflight_events.copy())
        self.inflight_events.clear()
        return event.parsed

    async def send_whatsapp(self, msg):
        return await comms_actions.send_whatsapp_message(
            from_number=self.assistant_number, to_number=self.user_number, message=msg
        )

    async def send_sms(self, msg):
        return await comms_actions.send_sms(
            from_number=self.assistant_number, to_number=self.user_number, message=msg
        )

    async def create_communication_task(
        self, contact_name: str, contact_number: str, detailed_task_description: str,
        task_scheduele: str = None
    ):
        # we are assuming one task here is being active, in some cases, another task to the same contact
        # might be pushed as well, in that case, we should just increase the tasks
        # the task should be scheduele-able as well, will think about this later
        contact_comms_agent = CommsAgent(contact_name, self.assistant_number, contact_number)
        contact_comms_agent.set_event_manager(self.event_manager)
        contact_comms_agent.performing_task = True
        contact_comms_agent.subscribe([contact_number, f"{contact_number}_{self.assistant_number}"])
        asyncio.create_task(contact_comms_agent.listen_for_events())

        # should generate a random id
        task = CommTask("12345", contact_name, contact_number, "in_progress", detailed_task_description)
        contact_comms_agent.attach_task(task)
        self.contact_num_to_comm_agent[contact_number] = contact_comms_agent
        self.attach_task(task)
    
    # mark task as done (could be failing, which is fine)
    async def task_done(self, task_status: Literal["fail", "success"], task_result: str):
        # mark task as done and publish task done event
        # should stop listening to events 
        ...

    async def ask_user_agent(self, query: str):
        # publish an event for the user agent to pick up
        ...
    
    async def wait_for_seconds_or_next_event(self, time: int):
        ...

    def subscribe(self, topics):
        if not self.event_manager:
            raise Exception("Set an event manager first.")
        for topic in topics:
            self.event_manager.topic_to_subs[topic].append(self)

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager

    def attach_task(self, task):
        self.running_tasks.append(task)
    
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

    def publish(self, event: dict):
        self.event_manager.publish(event)
    
    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        if event["event"]["event_name"] == "PhoneCallEndedEvent":
            if self.call_proc:
                print("HELLLLLLLLLLLLLLO")
                self.call_proc.kill()
                self.call_proc.wait()
                print("done")
                self.call_proc = None
                self.call_mode = False
                ONGOING_CALL = False
        if to == "past":
            self.past_events.append(event["event"])
        else:
            self.events_queue.put_nowait(event["event"])


