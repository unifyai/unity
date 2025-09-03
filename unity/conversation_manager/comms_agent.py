import asyncio
import json
import threading
import openai
import os
import redis
import traceback
from typing import Callable
from pathlib import Path
from pydantic_core import from_json
import unify
from collections import deque
from unity.helpers import run_script, terminate_process
from unity.common.llm_helpers import start_async_tool_use_loop, methods_to_tool_dict
from unity.memory_manager.broader_context import get_broader_context
from unity.conversation_manager.debug_logger import log_job_startup, mark_job_done
from unity.conversation_manager.comms_actions import (
    _start_call,
    _join_meet_call,
    _send_email_via_address,
    _send_sms_message_via_number,
    _send_whatsapp_message_via_number,
    Call,
    send_email,
    send_sms_message,
    send_whatsapp_message,
)
from unity.conversation_manager.actions import *
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import (
    build_call_sys_prompt,
    build_non_call_sys_prompt,
    build_user_agent_prompt,
    build_action_prompt,
)
from unity.conversation_manager.utils import (
    enter_name_with_retry,
    get_meet_join_state,
    join_meet_on_browser,
    ensure_captions_enabled,
)

client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

ONGOING_CALL = False
DEFAULT_ASSISTANT_PAYLOAD = {
    "user_id": "default-user",
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
    "surname": "",
    "weekly_limit": None,
    "max_parallel": None,
    "profile_photo": None,
    "country": None,
    "voice_id": None,
    "tts_provider": "cartesia",
    "user_last_name": "",
}


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
        start_local: bool = False,
        enabled_tools: list | str | None = "conductor",
        task_context: Dict[str, str] = None,
        user_turn_end_callback: Callable = None,
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
        self.current_user = {
            "contact_id": 1,
            "user_name": user_name,
            "user_number": user_number,
            "user_whatsapp_number": user_whatsapp_number,
            "user_email": user_email,
        }

        # events (history)
        self.conv_context_length = conv_context_length
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.past_events = past_events or []
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

        # call config
        self.call_purpose = "general"
        self.task_context = task_context
        self.user_turn_end_callback = user_turn_end_callback
        self.pending_calls = []

        # meet conference
        self.meet_id = None
        self.meet_browser = None
        self.meet_joined = asyncio.Event()

        # conductor
        self.conductor = None
        self.tool_use_handles = None
        self.handle_count = 0
        self.enabled_tools = (
            enabled_tools if isinstance(enabled_tools, list) else [enabled_tools]
        )
        self.start_local = start_local

        # logging
        self.loop = asyncio.get_event_loop()
        self.transcript_manager = None
        self.contact_manager = None
        self.redis = None
        self.broader_context = ""
        self.project_name = project_name
        self.logging_lock = threading.Lock()

        # speaker tracking
        self.speaker_buffer = deque(maxlen=50)
        self.current_speaker = None

    def _build_enabled_tools_dict(self):
        from unity.common.llm_helpers import AsyncToolUseLoopHandle

        self.tool_use_handles: dict[int, dict[AsyncToolUseLoopHandle, str]] = {}

        if self.enabled_tools[0] is None:
            self.enabled_tools = {}
            return

        if "conductor" in self.enabled_tools:
            # # if conductor is enabled, add its methods only as it has all other tools
            # from unity.conductor.conductor import Conductor
            # from unity.transcript_manager.transcript_manager import TranscriptManager

            # if self.transcript_manager is None and self.contact_manager is None:
            #     try:
            #         self.transcript_manager = TranscriptManager()
            #     except AssertionError as e:
            #         # only needed temporarily until we move init to the start anyway
            #         print("Assertion error in transcript manager", e)

            # self.conductor = Conductor()
            self.enabled_tools = methods_to_tool_dict(
                # self.conductor.ask,
                # self.conductor.request,
                # self.transcript_manager.ask,
                self._start_screen_share,
                self._stop_screen_share,
                self._send_call,
                self._send_sms,
                self._send_email,
                self._send_whatsapp,
                self._send_call_to_third_party,
                self._send_sms_to_third_party,
                self._send_email_to_third_party,
                self._send_whatsapp_to_third_party,
                self._join_meet,
            )
            return

        tools_list = []
        for tool in self.enabled_tools:
            tool = tool.lower()
            if tool == "contact":
                from unity.contact_manager.contact_manager import ContactManager

                self.contact_manager = ContactManager()
                tools_list += [self.contact_manager.ask, self.contact_manager.update]

            elif tool == "transcript":
                if not self.transcript_manager:
                    from unity.transcript_manager.transcript_manager import (
                        TranscriptManager,
                    )

                    self.transcript_manager = TranscriptManager()
                tools_list += [
                    self.transcript_manager.ask,
                    self.transcript_manager.summarize,
                ]

            elif tool == "knowledge":
                from unity.knowledge_manager.knowledge_manager import KnowledgeManager

                self.knowledge_manager = KnowledgeManager()
                tools_list += [
                    self.knowledge_manager.ask,
                    self.knowledge_manager.update,
                ]

            elif tool == "scheduler":
                from unity.task_scheduler.task_scheduler import TaskScheduler

                self.task_scheduler = TaskScheduler()
                tools_list += [self.task_scheduler.ask, self.task_scheduler.update]

            elif tool == "comms":
                tools_list += [
                    self._send_call,
                    self._send_sms,
                    self._send_email,
                    self._send_whatsapp,
                    self._join_meet,
                ]

            elif tool == "browser":
                from unity.actor.hierarchical_actor import HierarchicalActor

                self.actor = HierarchicalActor()
                tools_list += [self.actor.execute]

        self.enabled_tools = methods_to_tool_dict(*tools_list)

    async def get_bus_events(self):
        from unity.events.event_bus import EVENT_BUS

        bus_events = await EVENT_BUS.search(
            filter='type == "Comms"',
            limit=self.conv_context_length,
        )

        return [Event.from_bus_event(e).to_dict() for e in bus_events][::-1]

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

    async def inactivity_check_for_meet(self):
        # wait for the agent to be admitted into the meet
        await asyncio.sleep(20)

        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            if self.meet_browser is None:
                break  # meet call ended, exit the loop
            ret = await self.meet_browser.observe(
                f"Is {self.assistant_name} the only participant in the meeting?",
                bool,
            )
            print("ASSISTANT ONLY PARTICIPANT:", ret)
            if ret:
                print("All participants left, shutting down agent...")
                await self.publish(
                    {
                        "topic": self.user_number,
                        "event": PhoneCallStopEvent().to_dict(),
                    },
                )
                break  # Exit the loop after shutdown

    async def track_active_speaker(self):
        """Track active speaker using Meet captions via screenshot-based observation.

        Attempts to enable captions once, then polls the UI for the current caption
        speaker label and records it with a timestamp.
        """
        if not self.meet_browser:
            return

        await ensure_captions_enabled(self.meet_browser)

        while self.meet_browser:
            try:
                name = await self.meet_browser.observe(
                    (
                        "From the current Google Meet screen, read the live captions/subtitles. "
                        "If someone is currently speaking, return exactly their display name as plain text. "
                        "If no one is speaking or no caption is visible, return an empty string. "
                        "Do not include any extra words, punctuation, or quotes. Output must be a single line."
                    ),
                    str,
                )
                # Parallel observation: detect active speaker via blue outline/label indicator
                # name = await self.meet_browser.observe(
                #     (
                #         "From the current Google Meet screen, identify the participant visually marked as speaking "
                #         "(e.g., blue outline, speaker badge, or active-speaker indicator). "
                #         "If a participant is visually indicated as speaking, return exactly their display name as plain text. "
                #         "If no one is visually indicated as speaking, return an empty string. "
                #         "Do not include any extra words, punctuation, or quotes. Output must be a single line."
                #     ),
                #     str,
                # )
                if isinstance(name, str):
                    name = name.strip().split("\n")[0]
                # if name:
                self.current_speaker = name
                self.speaker_buffer.append((asyncio.get_event_loop().time(), name))
                print("\n\ncurrent speaker (captions)", name)
                print("speaker buffer", self.speaker_buffer)
            except Exception:
                print("\nTRACKING ACTIVE SPEAKER ERROR!\n")
            # await asyncio.sleep(0.5)

    def _nearest_speaker(self, t: float, window: float = 3.0):
        for ts, name in reversed(self.speaker_buffer):
            if t - ts <= window:
                return name
            break
        return None

    async def listen_for_events(self):
        print("COLLECTING...")
        while True:
            try:
                new_event = await asyncio.wait_for(self.events_queue.get(), 1)
                call_mode = new_event["payload"].get("call_mode") or new_event[
                    "event_name"
                ] in [
                    "PhoneCallStartedEvent",
                    "PhoneCallInitiatedEvent",
                    "PhoneCallEndedEvent",
                    "PhoneCallStopEvent",
                    "PhoneUtteranceEvent",
                ]
                # print("comm agent got", new_event)
                # continue
                if new_event["payload"]["transient"]:
                    continue
                if new_event["event_name"] == "PhoneCallInitiatedEvent":
                    global ONGOING_CALL
                    if not ONGOING_CALL:
                        self.call_purpose = new_event["payload"]["purpose"]
                        self.task_context = new_event["payload"]["task_context"]
                        target_number = new_event["payload"]["target_number"]
                        self.meet_id = new_event["payload"]["meet_id"]

                        print("call_requested", self.assistant_number)
                        print("new_event", new_event)
                        if not self.start_local:
                            target_path = Path(__file__).parent.resolve() / "call.py"

                            self.call_proc = run_script(
                                str(target_path),
                                "dev",
                                (
                                    target_number
                                    if target_number
                                    else self.current_user["user_number"]
                                ),
                                self.assistant_number,
                                self.tts_provider,
                                self.voice_id if self.voice_id else "None",
                                self.meet_id if self.meet_id else "None",
                            )
                        else:
                            target_path = Path(__file__).parent.resolve() / "call.py"
                            self.call_proc = run_script(
                                str(target_path),
                                "console",
                                self.current_user["user_number"],
                                self.assistant_number,
                                self.tts_provider,
                                self.voice_id if self.voice_id else "None",
                                self.meet_id if self.meet_id else "None",
                            )
                        ONGOING_CALL = True

                        # Join meet conference programatically
                        if self.meet_id:
                            from unity.controller.controller import Controller
                            from unify.logging.utils.logs import initialize_trace_logger

                            initialize_trace_logger()
                            self.meet_browser = Controller(redis_db=10)
                            self.meet_browser.start()

                            # Join meet
                            await join_meet_on_browser(self.meet_browser, self.meet_id)
                            await enter_name_with_retry(
                                self.meet_browser,
                                self.assistant_name,
                                max_attempts=3,
                            )
                            while (
                                await get_meet_join_state(self.meet_browser) != "joined"
                            ):
                                await asyncio.sleep(0.5)

                            asyncio.create_task(self.inactivity_check_for_meet())
                            self.meet_joined.set()
                            # start captions-based speaker tracking
                            asyncio.create_task(self.track_active_speaker())

                        continue
                    else:
                        # append initated phone call and failed
                        self.pending_calls.append(new_event)
                        continue

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
                    self.current_llm_run = asyncio.create_task(
                        self.run(
                            add_filler=new_event["event_name"]
                            != "PhoneCallStartedEvent",
                            call_mode=call_mode,
                        ),
                    )
                    self.current_llm_run.add_done_callback(
                        lambda t: self.on_run_end(t, call_mode),
                    )
                    self.pending_events.clear()
            except asyncio.TimeoutError:
                if not self.pending_events:
                    continue
                if self.current_llm_run and not self.current_llm_run.done():
                    continue

                self.inflight_events = self.pending_events.copy()
                self.current_llm_run = asyncio.create_task(
                    self.run(call_mode=call_mode),
                )
                self.current_llm_run.add_done_callback(self.on_run_end)

                self.pending_events.clear()

    # broader context helper
    def _inject_broader_context(self, msgs: list[dict]) -> list[dict]:
        """Replace the `{broader_context}` placeholder inside *system* messages
        with a fresh snapshot from `MemoryManager` right before the LLM call."""

        import copy

        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local import to avoid cycles

        patched = copy.deepcopy(msgs)

        try:
            broader_ctx = MemoryManager.get_rolling_activity()
        except Exception:
            broader_ctx = ""

        for m in patched:
            if m.get("role") == "system" and "{broader_context}" in (
                m.get("content") or ""
            ):
                m["content"] = m["content"].replace("{broader_context}", broader_ctx)

        return patched

    async def tool_use_action(self, action: ToolUseAction, call_mode: bool = False):
        """Handle tool_use actions asynchronously"""

        if isinstance(self.enabled_tools, list):
            self._build_enabled_tools_dict()

        # get chat history
        chat_history = self.get_chat_history()

        # start handle
        unify_client = unify.AsyncUnify("o4-mini@openai")
        unify_client.set_system_message(
            build_action_prompt(self.enabled_tools, action.query),
        )
        tool_use_handle = start_async_tool_use_loop(
            unify_client,
            action.query,
            self.enabled_tools,
            parent_chat_context=chat_history,
            preprocess_msgs=self._inject_broader_context,
        )

        # if action.show_steps:
        #     async def _wrap():
        #         answer = await tool_use_handle.result()
        #         return answer, unify_client.messages

        #     tool_use_handle.result = _wrap  # type: ignore[attr-defined]

        handle_id = self.handle_count
        self.tool_use_handles[handle_id] = {
            "handle": tool_use_handle,
            "query": action.query,
            "client": unify_client,
        }
        self.handle_count += 1

        # publish start event
        self.publish(
            {
                "topic": "tool_use",
                "to": "past",
                "event": ToolUseStartedEvent(
                    chat_history,
                    action.query,
                    handle_id,
                    call_mode=call_mode,
                ).to_dict(),
            },
        )

        # wait for the handle to be done
        while not tool_use_handle.done():
            print("waiting for handle to be done")
            await asyncio.sleep(1)

        # get handle result
        answer = await tool_use_handle.result()
        self.tool_use_handles.pop(handle_id)
        if isinstance(answer, tuple):
            answer, _ = answer

        # publish end event
        self.publish(
            {
                "topic": "tool_use",
                "event": ToolUseEndedEvent(
                    answer,
                    handle_id,
                    call_mode=call_mode,
                ).to_dict(),
            },
        )

    async def tool_use_handle_action(
        self,
        action: ToolUseHandleAction,
        call_mode: bool = False,
    ):
        """Handle tool_use handle actions asynchronously"""
        # check if the tool_use is running
        if self.tool_use_handles is None or not self.tool_use_handles.get(
            action.handle_id,
        ):
            # handle failed
            event_data = {
                "event": ToolUseHandleFailedEvent(
                    f"tool_use is not running currently, "
                    "please create a new action instead",
                    action.type,
                    call_mode=call_mode,
                ).to_dict(),
            }
        else:
            # handle
            handle = self.tool_use_handles[action.handle_id]["handle"]
            client = self.tool_use_handles[action.handle_id]["client"]
            if action.type == "ask":
                await handle.ask(action.query)
                self.events_queue.put_nowait(
                    PhoneUtteranceEvent(
                        role="System",
                        content=f"This is the current status of the tool_use: {client.messages[-1]}. Formulate response by replacing the tool_use name with the appropriate analogy and verb.",
                    ).to_dict(),
                )
            elif action.type == "interject":
                await handle.interject(action.query)
            elif action.type == "stop":
                handle.stop()
            elif action.type == "pause":
                handle.pause()
            elif action.type == "resume":
                handle.resume()
            event_data = {
                "event": ToolUseHandleSuccessEvent(
                    action.query,
                    action.type,
                    call_mode=call_mode,
                ).to_dict(),
                "to": "past",
            }
        self.publish({"topic": "tool_use", **event_data})

    def on_run_end(self, t: asyncio.Task, call_mode: bool = False):
        try:
            t: AssistantOutput | CallAssistantOutput | None = t.result()
            # everything is fine, just run the actions and add stuff to past events
            if t:
                self.past_events.extend(self.inflight_events.copy())
                self.inflight_events.clear()

                # this should launch async tasks
                if t.actions is not None:
                    print("actions", t.actions)
                    for action in t.actions:
                        if isinstance(action, ToolUseAction):
                            asyncio.create_task(self.tool_use_action(action, call_mode))
                        elif isinstance(action, ToolUseHandleAction):
                            asyncio.create_task(
                                self.tool_use_handle_action(action, call_mode),
                            )

        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self, add_filler: bool = False, call_mode: bool = False):
        if self.past_events is None:
            self.past_events = await self.get_bus_events()

        if call_mode:
            if self.meet_id:
                await self.meet_joined.wait()
            return await self.phone_call_llm_run(add_filler=add_filler)
        else:
            return await self.non_phone_call_llm_run()

    # response handling
    async def non_phone_call_llm_run(self):
        non_call_sys = build_non_call_sys_prompt(
            self.current_user["user_name"],
            self.current_user["user_number"],
            self.current_user["user_whatsapp_number"],
            self.current_user["user_email"],
            self.assistant_name,
            self.assistant_age,
            self.assistant_region,
            self.assistant_about,
            self.task_context,
            broader_context=self.broader_context,
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

    async def phone_call_llm_run(self, add_filler: bool = False):
        global ONGOING_CALL

        first_ev = {"topic": "call_process", "type": "start_gen"}
        self.publish(first_ev)

        if add_filler and self.user_turn_end_callback:
            filler = self.user_turn_end_callback()
            ev = {
                "topic": "call_process",
                "type": "gen_chunk",
                "chunk": f'{filler}<break time="1s"/>',
            }
            self.publish(ev)

        call_sys = build_call_sys_prompt(
            self.current_user["user_name"],
            self.current_user["user_number"],
            self.current_user["user_whatsapp_number"],
            self.current_user["user_email"],
            self.assistant_name,
            self.assistant_age,
            self.assistant_region,
            self.assistant_about,
            self.task_context,
            broader_context=self.broader_context,
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
            acc_text = ""
            last_response = ""
            async for event in stream:
                if not ONGOING_CALL:
                    print("call ended, stopping stream")
                    return None
                # print(event)
                if event.type == "content.delta":
                    if event.delta:
                        acc_text += event.delta
                        try:
                            output = from_json(
                                acc_text,
                                allow_partial="trailing-strings",
                            )
                        except ValueError:
                            continue

                        if not output.get("phone_utterance"):
                            continue

                        new_delta = output["phone_utterance"][len(last_response) :]
                        if new_delta:
                            print("delta", new_delta)
                            ev = {
                                "topic": "call_process",
                                "type": "gen_chunk",
                                "chunk": new_delta,
                            }
                            self.publish(ev)
                        last_response = output["phone_utterance"]

            ev = {"topic": "call_process", "type": "end_gen"}
            self.publish(ev)
        self.past_events.extend(self.inflight_events.copy())
        for ev in self.inflight_events:
            self.redis.publish("local_chat", json.dumps(ev))

        self.inflight_events.clear()
        return event.parsed

    # google meet communications
    async def _join_meet(
        self,
        meet_id: str,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
    ):
        """
        Joins a Google Meet call.

        Args:
            meet_id (str): The ID of the Google Meet call.
            purpose (str): The purpose of the call. Use 'general' if there is no specific purpose.
            task_context (Dict[str, str]): The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        global ONGOING_CALL
        await _join_meet_call(meet_id, purpose, task_context, ongoing_call=ONGOING_CALL)

    async def _start_screen_share(self):
        """
        Starts browser screen sharing.
        """
        if self.meet_browser is None:
            return

        query = "Create a new tab and go to https://www.google.com/"
        unify_client = unify.AsyncUnify("o4-mini@openai")
        unify_client.set_system_message(
            build_action_prompt(self.enabled_tools, query),
        )
        tool_use_handle = start_async_tool_use_loop(
            unify_client,
            query,
            self.enabled_tools,
            parent_chat_context=[],
            preprocess_msgs=self._inject_broader_context,
        )
        await tool_use_handle.result()

        await self.meet_browser.act("Click on the 'Share screen' button")

    async def _stop_screen_share(self):
        """
        Stops browser screen sharing.
        """
        if self.meet_browser is None:
            return

        await self.meet_browser.act("Click on the 'Stop presenting' button")

    # inner communications (to the current user)
    async def _send_call(
        self,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
    ):
        """
        Sends a call from the assistant's number to the current user's number.
        This tool is particularly used when you need to send a "DIRECT" call.

        Args:
            to_number (str): The number to call prefixed with +.
            purpose (str): The purpose of the call. Use 'general' if there is no specific purpose.
            task_context (Dict[str, str]): The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        from unity.contact_manager.contact_manager import ContactManager

        if self.contact_manager is None:
            self.contact_manager = ContactManager()
        return await Call.create(
            self.current_user["user_number"],
            purpose,
            task_context,
            tools=methods_to_tool_dict(self.contact_manager.ask),
        )

    async def _send_sms(self, message: str):
        """
        Sends an SMS message from the assistant's number to the current user's number.
        This tool is particularly used when you need to send a "DIRECT" SMS.

        Args:
            message (str): The message to send.
        """
        return await _send_sms_message_via_number(
            self.current_user["user_number"],
            message,
        )

    async def _send_email(self, subject: str, message: str, message_id: str = None):
        """
        Sends an email from the assistant's email address to the current user's email
        address.
        This tool is particularly used when you need to send a "DIRECT" email.

        Args:
            subject (str): The subject of the email.
            message (str): The message of the email.
            message_id (str): The message id of the email to reply to (ignore for now).
        """
        # ToDo: Add this back to the docstring once the message_id works
        # If you are asked to reply to an email rather than sending a new email,
        # use the transcript manager to get the message id for the email you want to reply
        # to.
        #
        # You should ask the transcript manager based on the contents of the original
        # mail, and get the message_id from the "_metadata" field of that transcript and
        # pass that as the message_id to this tool.
        return await _send_email_via_address(
            self.current_user["user_email"],
            subject,
            message,
            message_id,
        )

    async def _send_whatsapp(self, message: str, reply_to_user: bool = False):
        """
        Sends a WhatsApp message from the assistant's number to the current user's number.
        This tool is particularly used when you need to send a "DIRECT" WhatsApp message.

        Args:
            message (str): The message to send.
            reply_to_user (bool): `True` if replying to user's message. `False` if starting a new conversation.
        """
        return await _send_whatsapp_message_via_number(
            self.current_user["user_number"],
            message,
            reply_to_user,
        )

    # outer communications (to a third party)
    async def _send_call_to_third_party(
        self,
        to_number: str,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
    ):
        """
        Sends a call from the assistant's number to a third party.
        This tool is particularly used when you need to send a "THIRD PARTY" call.

        Args:
            to_number (str): The number to call prefixed with +.
            purpose (str): The purpose of the call. Use 'general' if there is no specific purpose.
            task_context (Dict[str, str]): The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        from unity.contact_manager.contact_manager import ContactManager

        if self.contact_manager is None:
            self.contact_manager = ContactManager()
        return await Call.create(
            to_number,
            purpose,
            task_context,
            tools=methods_to_tool_dict(self.contact_manager.ask),
        )

    async def _send_sms_to_third_party(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ):
        """
        Sends an SMS message from the assistant's number to a third party.
        This tool is particularly used when you need to send a "THIRD PARTY" SMS.

        Args:
            description (str): The description of the contact and content of the SMS message.
            parent_chat_context (list[dict]): The parent chat context.
        """
        return await send_sms_message(description, parent_chat_context)

    async def _send_email_to_third_party(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ):
        """
        Sends an email from the assistant's email address to a third party.
        This tool is particularly used when you need to send a "THIRD PARTY" email.
        You are the sender and the receiver is the third party.

        Args:
            description (str): The description of the contact and content of the email.
            parent_chat_context (list[dict]): The parent chat context.
        """
        return await send_email(description, parent_chat_context)

    async def _send_whatsapp_to_third_party(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ):
        """
        Sends a WhatsApp message from the assistant's number to a third party.
        This tool is particularly used when you need to send a "THIRD PARTY" WhatsApp
        message.

        Args:
            description (str): The description of the WhatsApp message.
            parent_chat_context (list[dict]): The parent chat context.
        """
        return await send_whatsapp_message(description, parent_chat_context)

    async def wait_for_seconds_or_next_event(self, time: int): ...

    def subscribe(self, topics):
        if not self.event_manager:
            raise Exception("Set an event manager first.")
        for topic in topics:
            self.event_manager.topic_to_subs[topic].add(self)

    def unsubscribe(self, topics):
        if not self.event_manager:
            raise Exception("Set an event manager first.")
        for topic in topics:
            self.event_manager.topic_to_subs[topic].remove(self)

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager

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
        self.tts_provider = payload["tts_provider"]
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
        os.environ["TTS_PROVIDER"] = self.tts_provider
        os.environ["VOICE_ID"] = self.voice_id

    async def initialize_redis(self):
        """Initialize Redis connection after server is ready"""
        import socket

        # Wait for Redis to be available
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Check if Redis port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", 6379))
                sock.close()

                if result == 0:
                    # Try to connect to Redis
                    test_redis = redis.Redis(host="localhost", port=6379, db=0)
                    test_redis.ping()
                    test_redis.close()

                    # Redis is ready, initialize the connection
                    self.redis = redis.Redis(host="localhost", port=6379, db=0)
                    print("Redis connection initialized successfully")
                    return
                else:
                    retry_count += 1
                    print(
                        f"Redis not ready yet, retrying... ({retry_count}/{max_retries})",
                    )
                    await asyncio.sleep(2)
            except Exception as e:
                retry_count += 1
                print(
                    f"Redis connection attempt {retry_count}/{max_retries} failed: {e}",
                )
                await asyncio.sleep(2)

        print("Warning: Failed to initialize Redis connection after all retries")
        # Initialize with None to avoid errors, but log the issue
        self.redis = None

    def get_user_agent_prompt(self):
        return build_user_agent_prompt(
            call_purpose=self.call_purpose,
            past_events=self.past_events,
            inflight_events=self.inflight_events,
            tool_use_handles=self.tool_use_handles,
        )

    def publish(self, event: dict):
        self.event_manager.publish(event)

    def cleanup_call_proc(self):
        if hasattr(self, "call_proc") and self.call_proc:
            print(f"Terminating call process")
            try:
                terminate_process(self.call_proc)
                self.call_proc = None
                global ONGOING_CALL
                ONGOING_CALL = False
                print(f"Call process terminated")
            except Exception as e:
                print(f"Error terminating call process: {e}")

    def cleanup(self):
        """Clean up any running call processes"""
        print(f"Marking job {self.job_name} done")
        mark_job_done(self.job_name)
        self.cleanup_call_proc()

    def handle_logging(self, event: dict):
        with self.logging_lock:
            import unity
            from unity.transcript_manager.transcript_manager import TranscriptManager
            from unity.events.event_bus import EVENT_BUS

            try:
                # initialize unity if not already initialised
                if not unity.ASSISTANT:
                    assistant_id = os.environ.get("ASSISTANT_ID", "0")
                    unity.init(
                        project_name=self.project_name,
                        assistant_id=int(
                            assistant_id.replace("default-assistant-", ""),
                        ),
                        default_assistant={
                            **DEFAULT_ASSISTANT_PAYLOAD,
                            "agent_id": assistant_id,
                            "first_name": self.assistant_name,
                            "age": self.assistant_age,
                            "region": self.assistant_region,
                            "about": self.assistant_about,
                            "phone": self.assistant_number,
                            "email": self.assistant_email,
                            "user_phone": self.user_number,
                            "user_whatsapp_number": self.user_whatsapp_number,
                            "assistant_whatsapp_number": self.assistant_number,
                            "api_key": os.environ.get("UNIFY_KEY"),
                        },
                    )
                    EVENT_BUS._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {os.environ['UNIFY_KEY']}"

                    # event_bus auto-pinning registration
                    EVENT_BUS.set_window("Comms", self.conv_context_length)
                    EVENT_BUS.register_auto_pin(
                        event_type="Comms",
                        open_predicate=lambda e: e.payload.get("role", "")
                        == "tool_use start",
                        close_predicate=lambda e: e.payload.get("role", "")
                        == "tool_use end",
                        key_fn=lambda e: e.payload.get("handle_id", ""),
                    )

                    # poll past events
                    self.loop.create_task(self.handle_past_events())

            except Exception as e:
                print(f"Error initializing unity: {e}")
                traceback.print_exc()
                return

            if self.transcript_manager is None:
                self.transcript_manager = TranscriptManager()
                self.transcript_manager._get_logger().session.headers[
                    "Authorization"
                ] = f"Bearer {os.environ['UNIFY_KEY']}"

            try:
                bus_event = Event.from_dict(event["event"]).to_bus_event()
                bus_event.payload.pop("api_key", None)
                message_id = bus_event.payload.pop("message_id", None)
                self.loop.create_task(EVENT_BUS.publish(bus_event))
                if event["event"]["event_name"] in [
                    "PhoneUtteranceEvent",
                    "WhatsappMessageSentEvent",
                    "SMSMessageSentEvent",
                    "EmailSentEvent",
                    "WhatsappMessageRecievedEvent",
                    "SMSMessageRecievedEvent",
                    "EmailRecievedEvent",
                ]:
                    event_name = event["event"]["event_name"].lower()
                    role = event["event"]["payload"]["role"]
                    content = event["event"]["payload"]["content"]
                    timestamp = event["event"]["payload"]["timestamp"]
                    medium = (
                        "phone_call"
                        if "phone" in event_name
                        else (
                            "sms_message"
                            if "sms" in event_name
                            else (
                                "email" if "email" in event_name else "whatsapp_message"
                            )
                        )
                    )
                    sender_id, receiver_ids = "", [""]
                    user_contact_id = self.current_user.get("contact_id", 1)
                    if medium == "whatsapp_message":
                        if role == "Assistant":
                            sender_id = 0
                            receiver_ids = [user_contact_id]
                        else:
                            sender_id = user_contact_id
                            receiver_ids = [0]
                    else:
                        if "recieved" in event_name:
                            sender_id = user_contact_id
                            receiver_ids = [0]
                        else:
                            sender_id = 0
                            receiver_ids = [user_contact_id]
                    metadata = None
                    if medium == "email":
                        metadata = {"message_id": message_id}
                    self.transcript_manager.log_messages(
                        {
                            "medium": medium,
                            "sender_id": sender_id,
                            "receiver_ids": receiver_ids,
                            "timestamp": timestamp,
                            "content": content,
                            "_metadata": metadata,
                        },
                    )
            except Exception as e:
                print(f"Error handling logging: {e}")
                traceback.print_exc()

    async def handle_past_events(self):
        """
        Background task that periodically fetches recent events from the EventBus
        and merges them into self.past_events.
        """
        while True:
            try:
                self.past_events = await self.get_bus_events()
                self.broader_context = await asyncio.to_thread(get_broader_context)
            except Exception as e:
                print(f"Error fetching bus events: {e}")
                traceback.print_exc()
            await asyncio.sleep(2)

    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        if event["event"]["event_name"] == "StartupEvent":
            try:
                self.set_details(event["event"]["payload"])
            except Exception as e:
                print(f"Error setting details: {e}")
                traceback.print_exc()
                return
            asyncio.create_task(
                asyncio.to_thread(
                    log_job_startup,
                    job_name=self.job_name,
                    timestamp=event["event"]["payload"]["timestamp"],
                    medium=event["event"]["payload"]["medium"],
                    user_id=self.user_id,
                    assistant_id=self.assistant_id,
                    user_name=self.user_name,
                    assistant_name=self.assistant_name,
                    user_number=self.user_number,
                    user_whatsapp_number=self.user_whatsapp_number,
                    assistant_number=self.assistant_number,
                ),
            )

        if event["event"]["event_name"] == "PhoneCallEndedEvent":
            if self.meet_browser:
                self.meet_browser.stop()
                self.meet_browser = None
                self.meet_id = None
                self.meet_joined.clear()

            if self.call_proc:
                self.cleanup_call_proc()

                # check for queued calls
                if self.pending_calls:
                    next_call_event = self.pending_calls.pop(0)

                    if next_call_event["payload"]["meet_id"]:
                        asyncio.create_task(
                            _join_meet_call(
                                next_call_event["payload"]["meet_id"],
                                next_call_event["payload"]["purpose"],
                                next_call_event["payload"]["task_context"],
                                ongoing_call=ONGOING_CALL,
                            ),
                        )
                    else:
                        asyncio.create_task(
                            _start_call(
                                self.assistant_number,
                                next_call_event["payload"]["target_number"],
                                next_call_event["payload"]["purpose"],
                                next_call_event["payload"]["task_context"],
                                ongoing_call=ONGOING_CALL,
                            ),
                        )

        elif event["event"]["event_name"] == "PhoneCallStopEvent":
            self.publish(
                {
                    "topic": "call_process",
                    "type": "stop",
                },
            )

        if event["event"].get("contact_details"):
            self.current_user = {
                "contact_id": event["event"]["payload"]["contact_details"][
                    "contact_id"
                ],
                "user_name": (
                    event["event"]["payload"]["contact_details"]["first_name"]
                    + " "
                    + event["event"]["payload"]["contact_details"]["surname"]
                ),
                "user_number": event["event"]["payload"]["contact_details"][
                    "phone_number"
                ],
                "user_whatsapp_number": event["event"]["payload"]["contact_details"][
                    "whatsapp_number"
                ],
                "user_email": event["event"]["payload"]["contact_details"][
                    "email_address"
                ],
            }

        # Attach speaker metadata to user phone utterances using recent captions
        try:
            if (
                event["event"]["event_name"] == "PhoneUtteranceEvent"
                and event["event"]["payload"].get("role") == "User"
            ):
                now = asyncio.get_event_loop().time()
                speaker = self._nearest_speaker(now)
                print("\n\nuser speaker", speaker)
                # if speaker:
                #     event["event"]["payload"]["speaker"] = speaker
        except Exception:
            ...

        if to == "past":
            self.past_events.append(event["event"])
            self.redis.publish("local_chat", json.dumps(event["event"]))
        else:
            self.events_queue.put_nowait(event["event"])
        asyncio.create_task(asyncio.to_thread(self.handle_logging, event))
