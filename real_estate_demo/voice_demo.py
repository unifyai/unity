import os
import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

import openai
from pydantic import BaseModel, Field

from new_terminal_helper import run_script, terminate_process

from demo_flow import flow, get_action_event, GoBack, GoNext, SYS_SONNET_2


client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Agent:
    def __init__(self):
        self.flow = flow
        # events (history)
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.event_stream = []
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

    async def listen_for_events(self):
        print("COLLECTING...")
        self.call_proc = run_script(
                            "call.py",
                            "console",
                        )
        while True:
            try:
                new_event = await asyncio.wait_for(self.events_queue.get(), 0.5)
                self.pending_events.append(new_event)
                # urgent events should re-trigger, cancel events should cancel current running only
                if new_event:
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
            agent_output = t.result()
            if agent_output.response:
                self.event_stream.append({"content": f"Agent: {agent_output.response}"})
            if agent_output.action:
                self.flow.play_actions(agent_output.action)
                print(self.flow.current_node.title)
                for label, action in agent_output.action:
                    if action is not None:
                        if not isinstance(action, GoNext) and not isinstance(action, GoBack):
                            action_event = get_action_event(flow, action)
                        else:
                            if isinstance(action, GoNext):
                                action_event = f"GoNext and has advanced to the next node: '{flow.current_node.title}'"
                            elif isinstance(action, GoBack):
                                action_event = f"GoBack and went back to the previous node: '{flow.current_node.title}'"
                        self.events_queue.put_nowait({"content": f"Agent took action: {action_event}"})

        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        return await self.phone_call_llm_run()
    
    async def phone_call_llm_run(self):
        client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(ev)


        class AgentOutput(BaseModel):
            response: Optional[str] = Field(..., description="Your response to the user, show as [Agent: ...] in the event stream, note that this response is over the phone, so make sure to use appropriate language")
            action: Optional[self.flow.current_action_model()] = Field(..., 
                                                                        description="The one single action to take given the current state (state = events stream and agent script UI), all other actions beside the chosen action should null")
        print(self.event_stream)
        event_stream_str = "\n".join([e["content"] for e in self.event_stream + self.inflight_events])
        user_msg = f"<event_stream>\n{event_stream_str}\n</event_stream>\n\n<agent_script>\n{flow.render()}\n</agent_script>"
        print("\033[32m" + user_msg + "\033[0m", flush=True)
        async with client.beta.chat.completions.stream(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": SYS_SONNET_2,
                        },
                        {
                            "role": "user",
                            "content": user_msg,
                        },
                    ],
                    response_format=AgentOutput,
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
        agent_output = event.parsed
        print(agent_output, flush=True)
        self.event_stream.extend(self.inflight_events.copy())
        self.inflight_events.clear()
        return agent_output

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager


    def publish(self, event: dict):
        self.event_manager.publish(event)

    def cleanup(self):
        """Clean up any running call processes"""
        if hasattr(self, "call_proc") and self.call_proc:
            print(f"Terminating call process for agent {self.agent_id}")
            try:
                terminate_process(self.call_proc)
                self.call_proc = None
                self.call_mode = False
                global ONGOING_CALL
                ONGOING_CALL = False
                print(f"Call process terminated for agent {self.agent_id}")
            except Exception as e:
                print(f"Error terminating call process for agent {self.agent_id}: {e}")

    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        
        if to == "past":
            self.event_stream.append(event["event"])
        else:
            self.events_queue.put_nowait(event["event"])
