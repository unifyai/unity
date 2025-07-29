import asyncio
import json
import openai
import os
import redis
import traceback
import unify
from unity.common.llm_helpers import start_async_tool_use_loop, methods_to_tool_dict
from unity.helpers import run_script, terminate_process
from unity.conversation_manager.comms_actions import (
    _start_call,
    _join_meet_call,
    _send_email_via_address,
    _send_sms_message_via_number,
    _send_whatsapp_message_via_number,
)
from unity.conversation_manager.actions import *
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import (
    build_call_sys_prompt,
    build_non_call_sys_prompt,
    build_user_agent_prompt,
    build_action_prompt,
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
    "tts_provider": "elevenlabs",
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
        user_name: str,
        assistant_name: str,
        assistant_age: str,
        assistant_region: str,
        assistant_about: str,
        assistant_number: str,
        assistant_email: str,
        user_number: str,
        user_phone_call_number: str = None,
        user_email: str = None,
        past_events: list | None = None,
        conv_context_length: int = 50,
        start_local: bool = False,
        enabled_tools: list | str | None = "conductor",
        task_context: Dict[str, str] = None,
        outer_comms_enabled: bool = False,
    ):
        # assistant details
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_region = assistant_region
        self.assistant_about = assistant_about

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email
        self.user_phone_call_number = (
            user_phone_call_number if user_phone_call_number else user_number
        )

        # events (history)
        self.conv_context_length = conv_context_length
        self.events_listener_task = None
        self.events_queue = asyncio.Queue()
        self.past_events = past_events or []
        self.pending_events = []
        self.inflight_events = []

        self.current_llm_run = None

        # switches to "True" when in a call
        self.call_mode = False
        self.call_purpose = "general"
        self.task_context = task_context
        self.outer_comms_enabled = outer_comms_enabled
        self.pending_calls = []

        # meet conference
        self.meet_id = None
        self.meet_browser = None

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
        self.redis = None

    def _build_enabled_tools_dict(self):
        from unity.common.llm_helpers import AsyncToolUseLoopHandle

        self.tool_use_handles: dict[int, dict[AsyncToolUseLoopHandle, str]] = {}

        if self.enabled_tools[0] is None:
            self.enabled_tools = {}
            return

        if "conductor" in self.enabled_tools:
            # if conductor is enabled, add its methods only as it has all other tools
            from unity.conductor.conductor import Conductor

            self.conductor = Conductor()
            self.enabled_tools = methods_to_tool_dict(
                self.conductor.ask,
                self.conductor.request,
                # todo: temporary adding them here explicitly
                self._inner_send_call,
                self._join_meet,
                self._inner_send_email,
                self._inner_send_sms,
            )
            return

        tools_list = []
        for tool in self.enabled_tools:
            tool = tool.lower()
            if tool == "contact":
                from unity.contact_manager.contact_manager import ContactManager

                manager = ContactManager()
                tools_list += [manager.ask, manager.update]

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

                manager = KnowledgeManager()
                tools_list += [manager.ask, manager.update]

            elif tool == "scheduler":
                from unity.task_scheduler.task_scheduler import TaskScheduler

                manager = TaskScheduler()
                tools_list += [manager.ask, manager.update]

            elif tool == "comms":
                tools_list += [self._join_meet]
                if self.outer_comms_enabled:
                    tools_list += [
                        self._outer_send_call,
                        self._outer_send_sms,
                        self._outer_send_email,
                        self._outer_send_whatsapp,
                    ]
                else:
                    tools_list += [
                        self._inner_send_call,
                        self._inner_send_sms,
                        self._inner_send_email,
                        self._inner_send_whatsapp,
                    ]

            elif tool == "browser":
                from unity.planner.hierarchical_planner import HierarchicalPlanner

                planner = HierarchicalPlanner()
                tools_list += [planner.execute]

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
            ret = await self.meet_browser.observe(
                f"Is {self.assistant_name} the only participant in the meeting?",
                bool,
            )
            print("ASSISTANT ONLY PARTICIPANT:", ret)
            if ret:
                print("All participants left, shutting down agent...")
                await self.publish(
                    {
                        "topic": self.user_phone_call_number,
                        "event": PhoneCallStopEvent().to_dict(),
                    },
                )
                break  # Exit the loop after shutdown

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
                        self.task_context = new_event["payload"]["task_context"]
                        target_number = new_event["payload"]["target_number"]
                        self.meet_id = new_event["payload"]["meet_id"]

                        print("call_requested", self.assistant_number)
                        if not self.start_local:
                            self.call_proc = run_script(
                                "unity/conversation_manager/call.py",
                                "dev",
                                (
                                    target_number
                                    if target_number
                                    else self.user_phone_call_number
                                ),
                                self.assistant_number,
                                (
                                    new_event["tts_provider"]
                                    if hasattr(new_event, "tts_provider")
                                    and new_event["tts_provider"]
                                    else "cartesia"
                                ),
                                (
                                    new_event["voice_id"]
                                    if hasattr(new_event, "voice_id")
                                    and new_event["voice_id"]
                                    else "None"
                                ),
                                "--outbound" if new_event.get("outbound") else "None",
                                self.meet_id if self.meet_id else "None",
                            )
                        else:
                            self.call_proc = run_script(
                                "unity/conversation_manager/call.py",
                                "console",
                                self.user_phone_call_number,
                                self.assistant_number,
                                "cartesia",
                                "None",
                                "None",
                                self.meet_id if self.meet_id else "None",
                            )
                        self.call_mode = True
                        ONGOING_CALL = True

                        # Join meet conference programatically
                        if self.meet_id:
                            from unity.controller.controller import Controller
                            from unify.logging.utils.logs import initialize_trace_logger

                            initialize_trace_logger()
                            self.meet_browser = Controller(redis_db=10)
                            self.meet_browser.start()

                            # Join meet
                            await self.meet_browser.act(
                                f"Go to the page: https://meet.google.com/{self.meet_id}",
                            )

                            # Enter name
                            await asyncio.sleep(1)
                            await self.meet_browser.act(
                                "Click 'your name' textbox",
                            )
                            await self.meet_browser.act(
                                f"Enter your name as {self.assistant_name}",
                            )

                            # Set agent mic
                            await self.meet_browser.act(
                                "Click on microphone default",
                            )
                            await asyncio.sleep(1)
                            await self.meet_browser.act(
                                "Select 'agent_sink.monitor'",
                            )

                            # Set user speaker
                            await self.meet_browser.act(
                                "Click on speaker default",
                            )
                            await asyncio.sleep(1)
                            await self.meet_browser.act("Select 'meet_sink'")

                            # Join meet
                            await self.meet_browser.act("Click the 'Join' button")

                            asyncio.create_task(self.inactivity_check_for_meet())

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
            broader_ctx = MemoryManager().get_rolling_activity()
        except Exception:
            broader_ctx = ""

        for m in patched:
            if m.get("role") == "system" and "{broader_context}" in (
                m.get("content") or ""
            ):
                m["content"] = m["content"].replace("{broader_context}", broader_ctx)

        return patched

    async def tool_use_action(self, action: ToolUseAction):
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
                "event": ToolUseEndedEvent(answer, handle_id).to_dict(),
            },
        )

    async def tool_use_handle_action(self, action: ToolUseHandleAction):
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
                ).to_dict(),
                "to": "past",
            }
        self.publish({"topic": "tool_use", **event_data})

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
                        if isinstance(action, ToolUseAction):
                            asyncio.create_task(self.tool_use_action(action))
                        elif isinstance(action, ToolUseHandleAction):
                            asyncio.create_task(
                                self.tool_use_handle_action(action),
                            )

        except asyncio.CancelledError:
            pass
        finally:
            ...

    async def run(self):
        if self.past_events is None:
            self.past_events = await self.get_bus_events()

        if self.call_mode:
            return await self.phone_call_llm_run()
        else:
            return await self.non_phone_call_llm_run()

    # response handling
    async def non_phone_call_llm_run(self):
        non_call_sys = build_non_call_sys_prompt(
            self.user_name,
            self.assistant_name,
            self.assistant_age,
            self.assistant_region,
            self.assistant_about,
            self.task_context,
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
            self.assistant_name,
            self.assistant_age,
            self.assistant_region,
            self.assistant_about,
            self.task_context,
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
        for ev in self.inflight_events:
            self.redis.publish("local_chat", json.dumps(ev))

        self.inflight_events.clear()
        return event.parsed

    # general communications
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

    # inner communications
    async def _inner_send_call(
        self,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
    ):
        """
        Sends a call from the assistant's number to the user's number.

        Args:
            purpose (str): The purpose of the call. Use 'general' if there is no specific purpose.
            task_context (Dict[str, str]): The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        global ONGOING_CALL
        await _start_call(
            self.assistant_number,
            self.user_phone_call_number,
            purpose,
            task_context,
            ongoing_call=ONGOING_CALL,
        )

    async def _inner_send_sms(self, message: str):
        """
        Sends an SMS message from the assistant's number to the user's number.

        Args:
            message (str): The message content to be sent via SMS.
        """
        await _send_sms_message_via_number(
            self.user_phone_call_number,
            message,
        )

    async def _inner_send_email(self, subject: str, message: str):
        """
        Sends an email from the assistant's email address to the user's email address.

        Args:
            subject (str): The subject of the email.
            message (str): The message content to be sent via email.
        """
        await _send_email_via_address(
            self.user_email,
            subject,
            message,
        )

    async def _inner_send_whatsapp(self, message: str, reply_to_user: bool = False):
        """
        Sends a WhatsApp message from the assistant's number to the user's number.

        Args:
            message (str): The message content to be sent via WhatsApp.
            reply_to_user (bool): `True` if replying to user's message. `False` if starting a new conversation.
        """
        await _send_whatsapp_message_via_number(
            self.user_number,
            message,
            reply_to_user,
        )

    # outer communications
    async def _outer_send_call(
        self,
        to_number: str,
        purpose: str = "general",
        task_context: Dict[str, str] = None,
    ):
        """
        Sends a call from the assistant's number to the user's number.

        Args:
            to_number (str): The number to call prefixed with +.
            purpose (str): The purpose of the call. Use 'general' if there is no specific purpose.
            task_context (Dict[str, str]): The broader task context for the call, with name and description attributes. Use None if there is no task context.
        """
        global ONGOING_CALL
        await _start_call(
            self.assistant_number,
            to_number,
            purpose,
            task_context,
            ongoing_call=ONGOING_CALL,
        )

    async def _outer_send_sms(self, to_number: str, message: str):
        """
        Sends an SMS message from the assistant's number to the user's number.

        Args:
            to_number (str): The number to send the SMS to prefixed with +.
            message (str): The message content to be sent via SMS.
        """
        await _send_sms_message_via_number(
            to_number,
            message,
        )

    async def _outer_send_email(self, to_email: str, subject: str, message: str):
        """
        Sends an email from the assistant's email address to the user's email address.

        Args:
            to_email (str): The email address to send the email to in the format of example@example.com.
            subject (str): The subject of the email.
            message (str): The message content to be sent via email.
        """
        await _send_email_via_address(
            to_email,
            subject,
            message,
        )

    async def _outer_send_whatsapp(
        self,
        to_number: str,
        message: str,
        reply_to_user: bool = False,
    ):
        """
        Sends a WhatsApp message from the assistant's number to the user's number.

        Args:
            to_number (str): The number to send the WhatsApp message to prefixed with +.
            message (str): The message content to be sent via WhatsApp.
            reply_to_user (bool): `True` if replying to user's message. `False` if starting a new conversation.
        """
        await _send_whatsapp_message_via_number(
            to_number,
            message,
            reply_to_user,
        )

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
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_region = payload["assistant_region"]
        self.assistant_about = payload["assistant_about"]
        self.assistant_number = payload["assistant_number"]
        self.user_name = payload["user_name"]
        self.user_number = payload["user_number"]
        self.user_phone_call_number = payload["user_phone_number"]
        self.user_email = payload["user_email"]
        os.environ["UNIFY_KEY"] = payload.pop("api_key")
        os.environ["USER_NAME"] = self.user_name
        os.environ["USER_PHONE_NUMBER"] = self.user_phone_call_number
        os.environ["USER_EMAIL"] = self.user_email
        os.environ["ASSISTANT_NAME"] = self.assistant_name
        os.environ["ASSISTANT_NUMBER"] = self.assistant_number
        os.environ["ASSISTANT_EMAIL"] = self.assistant_email

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
        import unity
        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.transcript_manager.types.message import Message
        from unity.events.event_bus import EVENT_BUS

        try:
            if not unity.ASSISTANT:
                assistant_id = os.environ.get("ASSISTANT_ID", "0")
                unity.init(
                    assistant_id=int(assistant_id.replace("default-assistant-", "")),
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
                        "user_whatsapp_number": self.user_phone_call_number,
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
            self.loop.create_task(EVENT_BUS.publish(bus_event))
            if event["event"]["event_name"] in [
                "PhoneUtteranceEvent",
                "WhatsappMessageSentEvent",
                "SMSMessageSentEvent",
                "WhatsappMessageRecievedEvent",
                "SMSMessageRecievedEvent",
            ]:
                event_name = event["event"]["event_name"].lower()
                role = event["event"]["payload"]["role"]
                content = event["event"]["payload"]["content"]
                timestamp = event["event"]["payload"]["timestamp"]
                medium = (
                    "phone_call"
                    if "phone" in event_name
                    else "sms_message" if "sms" in event_name else "whatsapp_message"
                )
                sender_id, receiver_ids = "", [""]
                if medium == "phone_call":
                    if role == "Assistant":
                        sender_id = self.assistant_number
                        receiver_ids = [self.user_phone_call_number]
                    else:
                        sender_id = self.user_phone_call_number
                        receiver_ids = [self.assistant_number]
                else:
                    if "recieved" in event_name:
                        sender_id = self.user_number
                        receiver_ids = [self.assistant_number]
                    else:
                        sender_id = self.assistant_number
                        receiver_ids = [self.user_number]
                self.transcript_manager.log_messages(
                    Message(
                        medium=medium,
                        sender_id=sender_id,
                        receiver_ids=receiver_ids,
                        timestamp=timestamp,
                        content=content,
                    ),
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
            except Exception as e:
                print(f"Error fetching bus events: {e}")
            await asyncio.sleep(2)

    def handle_event(self, event: dict):
        global ONGOING_CALL
        to = event.get("to")
        if event["event"]["event_name"] == "StartupEvent":
            self.set_details(event["event"]["payload"])

        if event["event"]["event_name"] == "PhoneCallEndedEvent":
            if self.meet_browser:
                self.meet_browser.stop()
                self.meet_browser = None
                self.meet_id = None

            if self.call_proc:
                self.call_proc.kill()
                self.call_proc.wait()
                self.call_proc = None
                self.call_mode = False
                ONGOING_CALL = False

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

        if to == "past":
            self.past_events.append(event["event"])
            self.redis.publish("local_chat", json.dumps(event["event"]))
        else:
            self.events_queue.put_nowait(event["event"])
        asyncio.create_task(asyncio.to_thread(self.handle_logging, event))
