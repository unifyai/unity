import os
import asyncio
from dataclasses import dataclass
from typing import Literal

import openai

import comms_actions
from actions import *
from events import *
from new_terminal_helper import run_script

client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

ONGOING_CALL = False


with open("prompts/call_sys.md") as f:
    call_sys = f.read()

with open("prompts/non_call_sys.md") as f:
    non_call_sys = f.read()


@dataclass
class CommTask:
    agent_id: str
    task_id: str
    contact_name: str
    contact_number: str
    status: str
    task_description: str


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
        past_events: list = None,
        main_user_agent: bool = False,
        agent_id: str = None,
        contact_name: str = None,
        contact_number: str = None,
    ):

        self.main_user = main_user_agent
        self.contact_name = contact_name
        self.agent_id = agent_id

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
        self.past_events = past_events or []
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

        # switches to "True" when in a call
        self.call_mode = False

        # tasks attributes

        # this only makes sense to the "main" comms agent (the user agent)
        self.contact_num_to_comm_agent: dict[str, CommsAgent] = {}
        self.running_tasks: list[CommTask] = []

        # id gen
        self.curr_id = 0
        self.curr_task_id = 0

        # queue for communication channels to make sure messages arrive in the right order
        self.whatsapp_queue = WhatsappQueue()

    async def listen_for_events(self):
        asyncio.create_task(self.whatsapp_queue.run())
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
                            "dev",
                            self.user_phone_call_number,  # "console" if a local call is needed
                            self.assistant_number,
                            "--outbound" if new_event.get("outbound") else "",
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
            print("FROM ", self.agent_id)
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
                        # take actions

                        # should be referenced in a set to avoid being garbage collected
                        # (i think)
                        if isinstance(action, SendCallAction):
                            asyncio.create_task(self.send_call())
                        # TODO: add sms
                        if isinstance(action, SendWhatsAppMessageAction):
                            print(action)
                            self.whatsapp_queue.add_message_task(
                                self.send_whatsapp(action.message),
                            )
                        elif isinstance(action, CreateCommunicationTask):
                            print(action)
                            self.create_communication_task(
                                action.contact_name,
                                action.contact_number.replace(" ", ""),
                                action.detailed_task_description,
                            )

                        elif isinstance(action, EndTask):
                            print(action)
                            self.task_done(
                                action.task_id,
                                action.task_status,
                                action.task_result,
                            )

                        elif isinstance(action, AskUserAgent):
                            self.ask_user_agent(action.task_id, action.query)

                        elif isinstance(action, RespondToAgent):
                            # respond to agent logic
                            self.respond_to_agent(
                                action.agent_id,
                                action.task_id,
                                action.response,
                            )

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
        print("Running from ...", self.agent_id)
        user_msg = self.get_user_agent_prompt()
        print(user_msg, flush=True)

        if self.main_user:
            with open("prompts/non_call_sys.md") as f:
                non_call_sys = f.read().format(name=self.user_name)
        else:
            with open("prompts/comm_non_call_sys.md") as f:
                non_call_sys = f.read().format(
                    main_user_name=self.user_name,
                    other_user_name=self.contact_name,
                )

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
            response_format=AssistantOutput if self.main_user else CommsAgentOutput,
        )
        message = res.choices[0].message
        # print(message)
        # print("parsed: ", message.parsed)
        if message.parsed:
            return message.parsed

    async def phone_call_llm_run(self):
        ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(ev)

        if self.main_user:
            with open("prompts/call_sys.md") as f:
                call_sys = f.read().format(name=self.user_name)
        else:
            with open("prompts/comm_call_sys.md") as f:
                call_sys = f.read().format(
                    main_user_name=self.user_name,
                    other_user_name=self.contact_name,
                )

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
            response_format=(
                CallAssistantOutput if self.main_user else CallCommsAgentOutput
            ),
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

    async def send_whatsapp(self, msg):
        status = await comms_actions.send_whatsapp_message(
            from_number=self.assistant_number,
            to_number=self.user_number,
            message=msg,
        )
        if status:
            event = WhatsappMessageSentEvent(
                content=msg,
            ).to_dict()
            if self.call_mode:
                self.events_queue.put_nowait(event)
            else:
                self.past_events.append(event)

    async def send_sms(self, msg):
        return await comms_actions.send_sms(
            from_number=self.assistant_number,
            to_number=self.user_number,
            message=msg,
        )

    async def send_call(self):
        print(self.assistant_number, self.user_phone_call_number)
        event = PhoneCallInitiatedEvent().to_dict()
        event["outbound"] = True
        self.events_queue.put_nowait(event)
        await comms_actions.send_call(
            self.assistant_number,
            self.user_phone_call_number,
        )

    def create_communication_task(
        self,
        contact_name: str,
        contact_number: str,
        detailed_task_description: str,
        task_scheduele: str = None,
    ):
        # we are assuming one task here is being active, in some cases, another task to the same contact
        # might be pushed as well, in that case, we should just increase the tasks
        # the task should be scheduele-able as well, will think about this later
        contact_comms_agent = self.contact_num_to_comm_agent.get(
            contact_number.replace(" ", ""),
        )
        # if not contact_name or contact_number:
        #     ...
        if not contact_comms_agent:
            contact_comms_agent = CommsAgent(
                self.user_name,
                self.assistant_number,
                contact_number.replace(" ", "").strip(),
                contact_name=contact_name,
                agent_id=self.curr_id,
            )
            print("created comms agent")
            self.curr_id += 1
            contact_comms_agent.set_event_manager(self.event_manager)
            contact_comms_agent.subscribe(
                [
                    contact_number.replace(" ", "").strip(),
                    str(contact_comms_agent.agent_id),
                ],
            )
            asyncio.create_task(contact_comms_agent.listen_for_events())
            print("listening")

        # should generate a random id
        task = CommTask(
            contact_comms_agent.agent_id,
            self.curr_task_id,
            contact_name,
            contact_number,
            "in_progress",
            detailed_task_description,
        )
        self.curr_task_id += 1
        print("created task")
        contact_comms_agent.attach_task(task)
        self.contact_num_to_comm_agent[contact_number.replace(" ", "")] = (
            contact_comms_agent
        )
        self.attach_task(task)
        print("attached tasks")

    # mark task as done (could be failing, which is fine)
    def task_done(
        self,
        task_id: int,
        task_status: Literal["fail", "success"],
        task_result: str,
    ):
        # mark task as done and publish task done event
        # should stop listening to events
        self.running_tasks = [
            t for t in self.running_tasks if str(t.task_id) != task_id
        ]

        self.publish(
            {
                "topic": "user_agent",
                "event": CommsTaskDoneEvent(
                    self.agent_id,
                    task_id,
                    task_status=task_status,
                    task_result=task_result,
                ).to_dict(),
            },
        )

        # other tasks remain
        # if self.running_tasks:
        #     self.events_queue.put_nowait(CommsTaskDoneEvent(self.agent_id, task_id, task_status=task_status, task_result=task_result).to_dict())

    def ask_user_agent(self, task_id: str, query: str):
        # publish an event for the user agent to pick up
        self.publish(
            {
                "topic": "user_agent",
                "event": AskUserAgentEvent(self.agent_id, task_id, query).to_dict(),
            },
        )

    def respond_to_agent(self, agent_id: str, task_id: str, response: str):
        print("RESPONDING TO AGENT")
        self.publish(
            {
                "topic": agent_id,
                "event": UserAgentResponseEvent(task_id, response).to_dict(),
            },
        )

    async def wait_for_seconds_or_next_event(self, time: int): ...

    def subscribe(self, topics):
        if not self.event_manager:
            raise Exception("Set an event manager first.")
        for topic in topics:
            self.event_manager.topic_to_subs[topic].add(self)

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager

    def attach_task(self, task: CommTask):
        self.running_tasks.append(task)
        if self.main_user:
            # put a task created event
            self.events_queue.put_nowait(
                CommsTaskCreatedEvent(
                    task.contact_name,
                    task.contact_number,
                    task.task_description,
                    task.agent_id,
                    task.task_id,
                ).to_dict(),
            )
        else:
            # put a task started event
            self.events_queue.put_nowait(
                CommsTaskStartedEvent(
                    task.contact_name,
                    task.contact_number,
                    task.task_description,
                    task.agent_id,
                    task.task_id,
                ).to_dict(),
            )

    def get_user_agent_prompt(self):
        past_events_str = (
            "\n".join([str(Event.from_dict(e)) for e in self.past_events])
            if self.past_events
            else ""
        )
        new_events_str = "\n".join(
            str(Event.from_dict(e)) for e in self.inflight_events
        )

        #         if self.running_tasks:
        #             task_status_str = ""
        #             for task in self.running_tasks:
        #                 task_status_str += (
        #                     f"""
        # TASK: {task.task_description}
        # TASK ID: {task.task_id}
        # TASK STATUS: {task.status}
        # """
        #                     + f"RUN BY AGENT: {task.agent_id}\n"
        #                 )
        #         else:
        #             task_status_str = "No Tasks are running"  # TODO
        user_msg = f"""Events Stream:
** PAST EVENTS **
{past_events_str.strip()}
** NEW EVENTS **
{new_events_str.strip()}"""
        return user_msg

    def publish(self, event: dict):
        self.event_manager.publish(event)

    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        if event["event"]["event_name"] == "CommsTaskDoneEvent":
            self.running_tasks = [
                t
                for t in self.running_tasks
                if str(t.task_id) != str(event["event"]["payload"]["task_id"])
            ]
        if event["event"]["event_name"] == "PhoneCallEndedEvent":
            if self.call_proc:
                self.call_proc.kill()
                self.call_proc.wait()
                self.call_proc = None
                self.call_mode = False
                ONGOING_CALL = False
        if to == "past":
            self.past_events.append(event["event"])
        else:
            self.events_queue.put_nowait(event["event"])
