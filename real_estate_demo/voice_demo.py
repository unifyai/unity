import os
import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from datetime import datetime

import openai
from pydantic import BaseModel, Field
from pydantic_core import from_json

from new_terminal_helper import run_script, terminate_process

from demo_flow import flow, get_action_event, GoBack, GoNext, EndSession, PromptUser, create_human_readable_delta


client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


NO_RESPONSE_COUNTER = 0
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
        self.call_proc = run_script("call.py", "dev")
        while True:
            try:
                new_event = await asyncio.wait_for(self.events_queue.get(), 0.5)
                print("GOT NEW EVENT", new_event)
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
        global NO_RESPONSE_COUNTER
        try:
            agent_output = t.result()
            next_action = agent_output.next_action
            # print(next_action)
            print("RUN DONE")
            if hasattr(next_action, "prompt"):
                # NO_RESPONSE_COUNTER = 0
                self.event_stream.append({"type": "message", "role": "assistant", "content": agent_output.next_action.prompt, "timestamp": datetime.now()})
            if hasattr(next_action, "update"):
                self.event_stream.append({"type": "message", "role": "assistant", "content": agent_output.next_action.update, "timestamp": datetime.now()})

            # check actions
            if hasattr(next_action, "next"):
                self.flow.play_actions([agent_output.next_action])
                action_event = f"advanced to the next node: '{flow.current_node.title}'"
                self.events_queue.put_nowait({"type": "action", "action_name":next_action.__class__.__name__, "content": action_event, "timestamp": datetime.now()})

            elif hasattr(next_action, "node_id"):
                self.flow.play_actions([agent_output.next_action])
                action_event = f"went to node `{action.node_id}`"
                self.events_queue.put_nowait({"type": "action", "action_name":next_action.__class__.__name__, "content": action_event, "timestamp": datetime.now()})


            elif hasattr(next_action, "fields_actions"):
                self.flow.play_actions(agent_output.next_action.fields_actions)
                for action in agent_output.next_action.fields_actions:
                    print(action)
                    action_event = get_action_event(flow, action)
                    self.events_queue.put_nowait({"type": "action", "action_name":action.__class__.__name__, "content": action_event, "timestamp": datetime.now()})
                    
                    

        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        return await self.phone_call_llm_run()
    
    async def phone_call_llm_run(self):
        client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        first_ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(first_ev)


        class AgentOutput(BaseModel):
            thoughts: str = Field(..., description="Your inner thoughts before taking actions. Also determine if you need to give a small update to the user based on the conversation history")
            # phone_utterance: Optional[str] = Field(..., 
                                            # description="Your response to the user over the phone, shown as [Assistant] ... in the conversation history.")
            next_action: flow.current_action_model() | PromptUser | EndSession = Field(..., 
                                        description="next action to take given the current state.")
        events = self.event_stream + self.inflight_events
        conversation_history = [e for e in events if e["type"] == "message"]
        action_log = [e for e in self.event_stream if e["type"] == "action"]
        conversation_history_str = "\n".join([f'[{m["role"].title()}, {create_human_readable_delta(m["timestamp"])}]: {m["content"]}' for m in conversation_history])
        conversation_history_prompt = f'<conversation_history>\n{conversation_history_str}\n</conversation_history>'

        action_log_str = "\n".join([f'[{m["action_name"]}, {create_human_readable_delta(m["timestamp"])}]: {m["content"]}' for m in action_log])
        agent_script_prompt = f"""
<agent_script>
<action_log>
{action_log_str if action_log_str else 'No Actions Taken Yet'}
</action_log>

<current_node>
{flow.render()}
</current_node>
</agent_script>""".strip()
        user_msg = f"{conversation_history_prompt}\n\n{agent_script_prompt}"
        print("\033[32m" + user_msg + "\033[0m", flush=True)
        
        with open(r"C:\Users\LEGION\Desktop\unity\unity\real_estate_demo\prompts\v5.md") as f:
            sys = f.read()
        
        acc_text = ""
        last_response = ""
        async with client.beta.chat.completions.stream(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": sys,
                        },
                        {
                            "role": "user",
                            "content": user_msg,
                        },
                    ],
                    response_format=AgentOutput,
                ) as stream:
            async for event in stream:
                ev = None
                # print(event)
                if event.type == "content.delta":
                    # ev = {
                    #     "topic": "call_process",
                    #     "type": "gen_chunk",
                    #     "chunk": event.delta,
                    # }
                    if event.delta:
                        acc_text += event.delta
                        # print("ACC TEXT", acc_text)
                        output = from_json(acc_text, allow_partial="trailing-strings")
                        if output.get("next_action"):
                            if output["next_action"].get("closing_message"):
                                ev = {
                                        "topic": "call_process",
                                        "type": "gen_chunk",
                                        "chunk": output["next_action"]["closing_message"][len(last_response):],
                                    }
                                last_response = output["next_action"]["closing_message"]
                            elif output["next_action"].get("prompt"):
                                ev = {
                                        "topic": "call_process",
                                        "type": "gen_chunk",
                                        "chunk": output["next_action"]["prompt"][len(last_response):],
                                    }
                                last_response = output["next_action"]["prompt"]
                            elif output["next_action"].get("update"):
                                ev = {
                                        "topic": "call_process",
                                        "type": "gen_chunk",
                                        "chunk": output["next_action"]["update"][len(last_response):],
                                    }
                                last_response = output["next_action"]["update"]
                            if ev: self.publish(ev)

            ev = {"topic": "call_process", "type": "end_gen"}
            self.publish(ev)
        agent_output = event.parsed
        # print(agent_output, flush=True)
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
