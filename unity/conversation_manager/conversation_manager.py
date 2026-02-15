import asyncio
import logging

import json
from typing import Optional
import contextlib

from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.settings import SETTINGS
from unity.manager_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.hierarchical_logger import SessionLogger
from unity.conversation_manager import debug_logger
from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unity.conversation_manager.domains.contact_index import ContactIndex, CommsMessage
from unity.conversation_manager.domains.brain import build_brain_spec
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.brain_tools import ConversationManagerBrainTools
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.events import *
from unity.common.prompt_helpers import now as prompt_now

from unity.common.llm_client import new_llm_client
from unity.common.single_shot import single_shot_tool_decision
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.utils import Debouncer, log_task_exc

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager.types import Medium, Mode, ScreenshotEntry
from unity.actor.base import BaseActor
from unity.conversation_manager.domains.proactive_speech import ProactiveSpeech
from unity.conversation_manager.domains.guidance_filter import (
    GuidanceFilter,
    ConversationMessage,
)

logger = logging.getLogger(__name__)

# Set logging level and add handler if not already configured
log_level = SETTINGS.conversation.LOG_LEVEL.upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Ensure we have a console handler to actually display logs
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


MAX_CONV_MANAGER_MSGS = 50


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker,
        job_name: str,
        user_id: str,
        assistant_id: str,
        user_name: str,
        assistant_name: str,
        assistant_age: str,
        assistant_nationality: str,
        assistant_about: str,
        assistant_number: str,
        assistant_email: str,
        user_number: str,
        user_email: str = None,
        voice_provider: str = "cartesia",
        voice_id: str = None,
        voice_mode: str = "tts",
        assistant_timezone: str = "",
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
        self.assistant_nationality = assistant_nationality
        self.assistant_timezone = assistant_timezone
        self.assistant_about = assistant_about
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.voice_mode = voice_mode

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email

        # initialization state
        self.initialized: bool = False
        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name

        # inactivity & shutdown
        self.inactivity_timeout = 540  # 9 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self.stop = stop

        self.event_broker = event_broker

        # managers
        self.transcript_manager: TranscriptManager = None
        self.contact_manager: ContactManager = None
        self.memory_manager: MemoryManager = None
        self.actor: BaseActor | None = None

        # debouncer (used to debounce llm runs)
        self.debouncer = Debouncer()

        # call manager - pass event_broker for socket IPC with voice agent subprocess
        self.call_manager = LivekitCallManager(self.get_call_config(), event_broker)
        self.call_manager.on_user_screenshot = self._buffer_user_screenshot

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a property with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        # Note: uses_realtime_api flag is stored for prompt building in _run_llm
        self._uses_realtime_api = self.call_manager.uses_realtime_api

        self.mode: Mode = Mode.TEXT
        self.chat_history = []
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.in_flight_actions: dict[int, dict] = (
            {}
        )  # dict[int, {"handle": "SteerableTool", "query": "str", "handle_actions": []}]
        self.completed_actions: dict[int, dict] = (
            {}
        )  # Finished actions, kept for post-completion ask() queries
        self._pending_steering_tasks: set[asyncio.Task] = (
            set()
        )  # Background tasks from async steering ops (e.g., ask_*)
        self.last_snapshot = prompt_now(as_string=False)
        self._current_snapshot = None
        self._current_state_snapshot = (
            None  # Fresh rendered state for tools during _run_llm
        )
        self._current_snapshot_state = (
            None  # SnapshotState with element tracking for incremental diff computation
        )
        self.is_summarizing = None
        self.max_messages = 30

        # meet interaction state (screen share / remote control)
        self.assistant_screen_share_active: bool = False
        self.user_screen_share_active: bool = False
        self.user_remote_control_active: bool = False

        # screenshot buffer for slow brain visual context
        self._screenshot_buffer: list[ScreenshotEntry] = []

        # proactive speech
        self.proactive_speech = ProactiveSpeech()
        self._proactive_speech_task: asyncio.Task | None = None

        # ask handles (for Actor actions)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool]] = []

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.info(
            "session_start",
            "ConversationManager session initialized",
        )

    def snapshot(self):
        self._current_snapshot = prompt_now(as_string=False)
        # Track how many notifications were present at snapshot time.
        # Any notifications appended while the LLM is running (e.g., an action that
        # completes very quickly) must remain visible for at least the NEXT LLM run.
        # Otherwise, `commit()` would immediately drop them and the LLM would never
        # see the result, which can cause repeated duplicate actions.
        self._snapshot_notif_count = len(self.notifications_bar.notifications)
        return self._current_snapshot

    def commit(self):
        self.last_snapshot = self._current_snapshot
        notifs = self.notifications_bar.notifications
        snap_n = int(getattr(self, "_snapshot_notif_count", 0) or 0)
        # Keep:
        # - pinned notifications
        # - notifications that were appended AFTER the last snapshot was taken
        #   (these arrived during the LLM run and must be shown next turn)
        self.notifications_bar.notifications = [
            n for i, n in enumerate(notifs) if n.pinned or i >= snap_n
        ]

    @property
    def session_logger(self) -> SessionLogger:
        """The hierarchical session logger for this ConversationManager instance."""
        return self._session_logger

    def get_active_contact(self) -> dict | None:
        """Get the contact for the current active call, or fall back to the boss contact."""
        return self.call_manager.call_contact or self.contact_index.get_contact(
            contact_id=1,
        )

    async def capture_assistant_screenshot(self, user_utterance: str) -> None:
        """Capture the assistant's screen and buffer it for the next slow brain turn.

        Called when an inbound utterance arrives while assistant screen sharing
        is active. The screenshot is paired with the user's utterance text so
        the slow brain can align visual context with spoken instructions.
        """
        import aiohttp
        from datetime import datetime, timezone

        desktop_url = SESSION_DETAILS.assistant.desktop_url or "http://localhost:3000"
        try:
            auth_key = SESSION_DETAILS.unify_key
            headers = {"authorization": f"Bearer {auth_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{desktop_url}/screenshot",
                    json={},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 400:
                        self._session_logger.warning(
                            "screenshot_capture",
                            f"Screenshot capture failed: HTTP {resp.status}",
                        )
                        return
                    data = await resp.json()
                    b64 = data.get("screenshot")
                    if b64:
                        self._screenshot_buffer.append(
                            ScreenshotEntry(
                                b64,
                                user_utterance,
                                datetime.now(timezone.utc),
                                "assistant",
                            ),
                        )
                        self._session_logger.debug(
                            "screenshot_capture",
                            f"Buffered screenshot #{len(self._screenshot_buffer)} "
                            f"for utterance: {user_utterance[:60]}...",
                        )
        except Exception as e:
            self._session_logger.warning(
                "screenshot_capture",
                f"Screenshot capture error: {e}",
            )

    def drain_screenshot_buffer(self) -> list[ScreenshotEntry]:
        """Drain and return all buffered screenshots, clearing the buffer."""
        screenshots = list(self._screenshot_buffer)
        self._screenshot_buffer.clear()
        return screenshots

    def _buffer_user_screenshot(self, event_json: str) -> None:
        """Buffer a user screen share screenshot received from the fast brain via IPC."""
        import json as _json
        from datetime import datetime, timezone

        try:
            data = _json.loads(event_json)
            b64 = data.get("b64", "")
            utterance = data.get("utterance", "")
            ts_str = data.get("timestamp")
            ts = (
                datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            )
            if b64:
                self._screenshot_buffer.append(
                    ScreenshotEntry(b64, utterance, ts, "user"),
                )
                self._session_logger.debug(
                    "screenshot_capture",
                    f"Buffered user screenshot #{len(self._screenshot_buffer)} "
                    f"for utterance: {utterance[:60]}...",
                )
        except Exception as e:
            self._session_logger.warning(
                "screenshot_capture",
                f"Error buffering user screenshot: {e}",
            )

    def get_recent_voice_transcript(
        self,
        contact: dict | None = None,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """Extract recent voice transcript from the active conversation.

        Args:
            contact: Contact to get transcript for. Defaults to active contact.
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        conversation_turns: list[dict] = []
        last_message_timestamp: datetime | None = None

        if contact is None:
            contact = self.get_active_contact()

        if not contact:
            return conversation_turns, last_message_timestamp

        contact_id = contact.get("contact_id")
        conv_state = self.contact_index.get_conversation_state(contact_id)
        if not conv_state:
            return conversation_turns, last_message_timestamp

        voice_medium = (
            Medium.UNIFY_MEET if self.mode == Mode.MEET else Medium.PHONE_CALL
        )
        voice_thread = self.contact_index.get_messages_for_contact(
            contact_id,
            voice_medium,
        )

        # Optionally limit to last N messages
        if max_messages is not None:
            voice_thread = voice_thread[-max_messages:]

        for msg in voice_thread:
            role = "assistant" if msg.name == "You" else "user"
            content = (msg.content or "").strip()

            # Skip system messages (e.g., "<Call Started>")
            if content.startswith("<") and content.endswith(">"):
                continue

            conversation_turns.append({"role": role, "content": content})

            if hasattr(msg, "timestamp") and msg.timestamp:
                last_message_timestamp = msg.timestamp

        return conversation_turns, last_message_timestamp

    def get_recent_transcript(
        self,
        contact: dict | None = None,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """Extract recent transcript from ALL threads for a contact.

        Unlike get_recent_voice_transcript which only looks at the voice thread,
        this method uses the global_thread which contains messages from ALL mediums
        (sms, unify, voice, email).

        Args:
            contact: Contact to get transcript for. Defaults to active contact.
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        conversation_turns: list[dict] = []
        last_message_timestamp: datetime | None = None

        if contact is None:
            contact = self.get_active_contact()

        if not contact:
            return conversation_turns, last_message_timestamp

        contact_id = contact.get("contact_id")
        conv_state = self.contact_index.get_conversation_state(contact_id)
        if not conv_state:
            return conversation_turns, last_message_timestamp

        global_thread = self.contact_index.get_messages_for_contact(contact_id)

        # Optionally limit to last N messages
        if max_messages is not None:
            global_thread = global_thread[-max_messages:]

        for msg in global_thread:
            # Skip non-communication messages (e.g., GuidanceMessage for internal orchestration)
            if not isinstance(msg, CommsMessage):
                continue

            # Handle both Message and EmailMessage types
            if hasattr(msg, "content"):
                content = (msg.content or "").strip()
            elif hasattr(msg, "body"):
                content = (msg.body or "").strip()
            else:
                continue

            # Skip system messages (e.g., "<Call Started>")
            if content.startswith("<") and content.endswith(">"):
                continue

            conversation_turns.append({"role": msg.role, "content": content})

            if hasattr(msg, "timestamp") and msg.timestamp:
                last_message_timestamp = msg.timestamp

        return conversation_turns, last_message_timestamp

    def _preprocess_messages(
        self,
        messages: str | dict | list,
    ) -> str | dict | list:
        """Keep only the latest state snapshot from message history.

        ConversationManager renders a full state snapshot each turn. We keep only the
        latest snapshot when calling the model, while preserving any system messages
        and user interjections.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            return messages
        if not isinstance(messages, list):
            return messages

        try:
            # Find all state snapshot messages
            state_indices = [
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
            ]
            if not state_indices:
                return messages

            # Keep only the latest state snapshot and non-state messages
            last_state = messages[state_indices[-1]]
            kept: list[dict] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    kept.append(m)
                elif role == "user" and not m.get("_cm_state_snapshot"):
                    kept.append(m)

            kept.append(last_state)
            return kept
        except Exception:
            return messages

    async def _check_guidance_relevance(
        self,
        guidance_content: str,
        slow_brain_start_time: "datetime",
    ) -> bool:
        """
        Check if guidance is still relevant given conversation changes since slow brain started.

        The slow brain takes 10-20 seconds to think. During this time, the user may change
        topics, the fast brain may respond, etc. This method uses a fast LLM filter to
        determine if the guidance is still relevant or if it's stale.

        Args:
            guidance_content: The guidance text from the slow brain.
            slow_brain_start_time: When the slow brain started thinking.

        Returns:
            True if guidance should be sent, False if it's stale and should be blocked.
        """
        from datetime import datetime, timezone

        try:
            # Get the current voice conversation
            contact = self.get_active_contact()
            if not contact:
                return True  # No contact context, send guidance

            contact_id = contact.get("contact_id")
            conv_state = self.contact_index.get_conversation_state(contact_id)
            if not conv_state:
                return True  # No conversation state, send guidance

            # Get the voice thread
            voice_medium = (
                Medium.UNIFY_MEET if self.mode == Mode.MEET else Medium.PHONE_CALL
            )
            voice_thread = self.contact_index.get_messages_for_contact(
                contact_id,
                voice_medium,
            )

            if not voice_thread:
                return True  # No messages to compare, send guidance

            # Convert to ConversationMessage format with is_new flag
            conversation_messages = []
            for msg in voice_thread:
                content = (msg.content or "").strip()

                # Skip system messages (e.g., "<Call Started>")
                if content.startswith("<") and content.endswith(">"):
                    continue

                # Determine role
                if hasattr(msg, "role"):
                    role = msg.role
                else:
                    role = "assistant" if msg.name == "You" else "user"

                # Check if this message arrived AFTER slow brain started
                msg_time = getattr(msg, "timestamp", None)
                is_new = False
                if msg_time is not None:
                    # Ensure timezone-aware comparison
                    if msg_time.tzinfo is None:
                        msg_time = msg_time.replace(tzinfo=timezone.utc)
                    is_new = msg_time > slow_brain_start_time

                conversation_messages.append(
                    ConversationMessage(
                        role=role,
                        content=content,
                        timestamp=msg_time or datetime.now(timezone.utc),
                        is_new=is_new,
                    ),
                )

            # If no new messages, guidance is definitely still relevant
            if not any(m.is_new for m in conversation_messages):
                return True

            # Assistant-only new chatter should not force filtering; only new
            # user turns can make slow-brain guidance stale for the caller.
            if not any(
                m.is_new and (m.role or "").lower() == "user"
                for m in conversation_messages
            ):
                return True

            # Use the GuidanceFilter to make the decision
            guidance_filter = GuidanceFilter()
            decision = await guidance_filter.should_send_guidance(
                guidance_content,
                conversation_messages,
            )

            self._session_logger.debug(
                "guidance_filter",
                f"Filter decision: send={decision.send_guidance}, "
                f"thoughts={decision.thoughts[:100]}...",
            )

            return decision.send_guidance

        except Exception as e:
            # On error, default to sending guidance (fail-open)
            self._session_logger.error(
                "guidance_filter",
                f"Error in guidance filter, defaulting to send: {e}",
            )
            return True

    async def interject_or_run(self, content: str):
        """Interject the ask handle or run the LLM"""
        if self.active_ask_handle and not self.active_ask_handle.done():
            self._session_logger.info(
                "event",
                "Routing to active ask handle",
                icon_override="🔀",
            )
            await self.active_ask_handle.interject(content)
        else:
            # With single-shot LLM, there's no ongoing brain loop to interject.
            # Just trigger a new LLM run.
            self._session_logger.info(
                "llm_thinking",
                "Triggering main CM brain",
                icon_override="🔀",
            )
            # Voice mode: cancel_running=False so running LLM tasks complete
            # while only pending tasks are replaced ("queue of 2"). This
            # prevents rapid user speech from cancelling every LLM run.
            # Text mode: cancel_running=True — rapid messages should get
            # fresh responses with the latest context.
            cancel_running = not self.mode.is_voice
            await self.request_llm_run(delay=0, cancel_running=cancel_running)

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(self, delay=0, cancel_running=False):
        await self.debouncer.submit(
            self._run_llm,
            delay=delay,
            cancel_running=cancel_running,
        )

    async def request_llm_run(self, delay=0, cancel_running=False) -> None:
        """Request an LLM run.

        The request is recorded and later scheduled by the event loop after
        the current event is handled.
        """
        self._pending_llm_requests.append((delay, cancel_running))

    async def flush_llm_requests(self) -> None:
        """Schedule any pending LLM runs recorded during event handling."""
        if not self._pending_llm_requests:
            return
        delay, cancel_running = self._pending_llm_requests[-1]
        self._pending_llm_requests.clear()
        await self.run_llm(delay=delay, cancel_running=cancel_running)

    async def _run_llm(self) -> str | None:
        """Run a single LLM decision and return the tool name that was called."""
        # Capture when slow brain starts thinking (for guidance staleness detection)
        from datetime import datetime, timezone

        slow_brain_start_time = datetime.now(timezone.utc)

        # Drain buffered screenshots so the slow brain gets visual context
        # from screen sharing. The buffer is cleared atomically — new screenshots
        # captured while this turn is running will accumulate for the next turn.
        screenshots = self.drain_screenshot_buffer()

        self.snapshot()
        brain_spec = build_brain_spec(self, screenshots=screenshots)
        if screenshots:
            self._session_logger.info(
                "screen_share",
                f"Attaching {len(screenshots)} screenshot(s) to slow brain turn",
            )
        self._session_logger.debug(
            "state_update",
            f"State prompt:\n{brain_spec.state_prompt}",
        )
        input_message = brain_spec.state_message()
        system_prompt = brain_spec.system_prompt

        # Store current state snapshot for tools to access during execution.
        # Tools (act, steering) need the fresh rendered state, not the stale chat_history.
        self._current_state_snapshot = input_message

        # Also capture the structured snapshot state for incremental diff computation.
        # This enables interject operations to send only changes since the initial act().
        self._current_snapshot_state = self.prompt_renderer.render_state(
            self.contact_index,
            self.notifications_bar,
            self.in_flight_actions,
            self.completed_actions,
            self.last_snapshot,
            assistant_screen_share_active=self.assistant_screen_share_active,
            user_screen_share_active=self.user_screen_share_active,
            user_remote_control_active=self.user_remote_control_active,
        )

        # Log LLM thinking start
        self._session_logger.log_llm_thinking(f"mode={self.mode}")

        # Build response model dynamically with current in-flight actions
        response_model = brain_spec.response_model

        brain_tools = ConversationManagerBrainTools(self)
        action_tools = ConversationManagerBrainActionTools(self)
        # Combine static tools with dynamic action steering tools
        tools = {
            **brain_tools.as_tools(),
            **action_tools.as_tools(),
            **action_tools.build_action_steering_tools(),
            **action_tools.build_completed_action_tools(),
        }

        # Single-shot LLM call: one decision, one action
        client = new_llm_client(SETTINGS.UNIFY_MODEL)
        client.set_system_message(system_prompt.to_list())
        client.set_prompt_caching(["system"])
        messages = self._preprocess_messages(self.chat_history + [input_message])
        result = await single_shot_tool_decision(
            client,
            messages,
            tools,
            tool_choice="required" if tools else "auto",
            response_format=response_model,
        )

        # Extract structured output (thoughts, call_guidance)
        structured = result.structured_output
        thoughts = ""
        if structured is not None:
            thoughts = getattr(structured, "thoughts", "")

            # Handle call_guidance for voice modes
            if self.mode.is_voice:
                call_guidance = getattr(structured, "call_guidance", "")
                if call_guidance:
                    # Check if guidance is still relevant (conversation may have moved on)
                    should_send = await self._check_guidance_relevance(
                        call_guidance,
                        slow_brain_start_time,
                    )

                    if should_send:
                        contact = self.get_active_contact()
                        event = CallGuidance(contact, call_guidance)
                        await self.event_broker.publish(
                            "app:call:call_guidance",
                            event.to_json(),
                        )
                        await self.event_broker.publish(
                            "app:comms:assistant_call_guidance",
                            event.to_json(),
                        )
                    else:
                        self._session_logger.info(
                            "guidance_filtered",
                            f"Stale guidance blocked: {call_guidance[:50]}...",
                        )

        # Log LLM response
        self._session_logger.log_llm_response(
            (
                f"thoughts: {thoughts[:100]}..."
                if len(thoughts) > 100
                else f"thoughts: {thoughts}"
            )
            + (f" | action: {result.tool_name}" if result.tool_name else ""),
        )

        self.commit()
        self._session_logger.debug("state_update", "Committing state")

        # Clear the temporary state snapshots now that tools have executed
        self._current_state_snapshot = None
        self._current_snapshot_state = None

        # Build assistant message for chat history
        assistant_content = (
            structured.model_dump_json() if structured else result.text_response or ""
        )
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": assistant_content})

        if (
            len(self.chat_history) >= int(0.7 * self.max_messages)
            and not self.is_summarizing
        ):
            self._session_logger.info("summarize", "Summarizing conversation")
            await self.event_broker.publish(
                "app:comms:summarize",
                SummarizeContext().to_json(),
            )
            self.is_summarizing = True

        return result.tool_name

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:actor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            # Initialization is triggered by StartupEvent handler which
            # sets details before starting init. Do not duplicate here.

            while True:
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg:
                    continue
                self.last_activity_time = self.loop.time()
                # process events
                event = Event.from_json(msg["data"])

                await EventHandler.handle_event(
                    event,
                    self,
                    is_voice_call=self.call_manager.uses_realtime_api,
                )
                await self.flush_llm_requests()

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout"""
        while True:
            await asyncio.sleep(self.inactivity_check_interval)
            current_time = self.loop.time()
            if current_time - self.last_activity_time > self.inactivity_timeout:
                log_str = f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown"
                print(
                    log_str,
                )  # need console logging of inactivity to detect idle containers
                self._session_logger.info("session_end", log_str)
                self.stop.set()
                await self.event_broker.aclose()
                break  # Exit the loop after triggering shutdown

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details into SESSION_DETAILS."""
        self.user_id = payload["user_id"]
        self.assistant_id = payload["assistant_id"]
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_nationality = payload["assistant_nationality"]
        self.assistant_timezone = payload.get("assistant_timezone", "")
        self.assistant_about = payload["assistant_about"]
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.user_name = payload["user_name"]
        self.user_number = payload["user_number"]
        self.user_email = payload["user_email"]
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        self.voice_mode = payload["voice_mode"]
        self.desktop_mode = payload.get("desktop_mode", "ubuntu")
        self.desktop_url = payload.get("desktop_url")
        self.user_desktop_mode = payload.get("user_desktop_mode")
        self.user_desktop_filesys_sync = payload.get("user_desktop_filesys_sync", False)
        self.user_desktop_url = payload.get("user_desktop_url")
        # Set API key on SESSION_DETAILS for runtime access
        if payload.get("api_key"):
            SESSION_DETAILS.unify_key = payload["api_key"]
        # Populate the global SessionDetails singleton
        SESSION_DETAILS.populate(
            assistant_id=self.assistant_id,
            assistant_name=self.assistant_name,
            assistant_age=self.assistant_age,
            assistant_nationality=self.assistant_nationality,
            assistant_timezone=self.assistant_timezone,
            assistant_about=self.assistant_about,
            assistant_number=self.assistant_number,
            assistant_email=self.assistant_email,
            user_id=self.user_id,
            user_name=self.user_name,
            user_number=self.user_number,
            user_email=self.user_email,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            voice_mode=self.voice_mode,
            desktop_mode=self.desktop_mode,
            desktop_url=self.desktop_url,
            user_desktop_mode=self.user_desktop_mode,
            user_desktop_filesys_sync=self.user_desktop_filesys_sync,
            user_desktop_url=self.user_desktop_url,
        )
        # Export to env vars for subprocess inheritance
        SESSION_DETAILS.export_to_env()

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_name": self.user_name,
            "assistant_name": self.assistant_name,
            "user_number": self.user_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def get_call_config(self) -> CallConfig:
        return CallConfig(
            assistant_id=self.assistant_id,
            assistant_bio=self.assistant_about,
            assistant_number=self.assistant_number,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            voice_mode=self.voice_mode,
        )

    async def store_chat_history(self):
        if len(self.chat_history) >= 2:
            await self.event_broker.publish(
                "app:comms:chat_history",
                StoreChatHistory(chat_history=self.chat_history[-2:]).to_json(),
            )
            await asyncio.sleep(2)

    async def cleanup(self):
        """Clean up any running call processes and file sync.

        Always updates rolling summaries before shutdown, regardless of message count,
        to ensure conversation context is persisted for the next session.
        """
        # Import inline to avoid potential circular import issues with type checkers
        from unity.conversation_manager.domains import managers_utils

        # Always update rolling summaries before shutdown
        self._session_logger.info(
            "cleanup",
            "Updating rolling summaries before shutdown",
        )
        try:
            await managers_utils.update_rolling_summaries(self)
        except Exception as e:
            self._session_logger.error(
                "cleanup",
                f"Failed to update rolling summaries: {e}",
            )

        await self.store_chat_history()
        await self.call_manager.cleanup_call_proc()

        # Stop file sync to ensure final sync to VM
        await self._stop_file_sync()

        if self.job_name and self.assistant_id != DEFAULT_ASSISTANT_ID:
            self._session_logger.info(
                "session_end",
                f"Marking job {self.job_name} done",
            )
            debug_logger.mark_job_done(self.job_name)
        self.stop.set()

    async def _stop_file_sync(self) -> None:
        """Stop file sync with managed VM, performing final sync."""
        try:
            from unity.file_manager.managers.local import LocalFileManager

            local_fm = LocalFileManager()
            adapter = local_fm._adapter

            # Check if adapter supports sync
            if not hasattr(adapter, "sync_started"):
                return

            if adapter.sync_started:
                print("[ConversationManager] Stopping file sync...")
                await adapter.stop_sync()
                print("[ConversationManager] File sync stopped")
        except Exception as e:
            print(f"[ConversationManager] Failed to stop file sync: {e}")

    # Proactive speech related methods

    async def schedule_proactive_speech(self, skip_initial_wait: bool = False):
        """Decides if and when to speak proactively, and schedules it.

        Args:
            skip_initial_wait: If True, skip the initial 8s wait (used when rescheduling after a false decision)
        """
        self._session_logger.debug(
            "proactive_speech",
            f"schedule_proactive_speech called, mode={self.mode}, skip_initial_wait={skip_initial_wait}",
        )
        await self.cancel_proactive_speech()

        # Only schedule if we are in a call/voice mode where silence matters
        if not self.mode.is_voice:
            self._session_logger.debug(
                "proactive_speech",
                f"Skipping: mode {self.mode} not a voice mode",
            )
            return

        self._session_logger.debug("proactive_speech", "Creating proactive speech task")
        # Create a task to run the decision and potential wait
        self._proactive_speech_task = asyncio.create_task(
            self._proactive_speech_loop(skip_initial_wait=skip_initial_wait),
        )
        self._proactive_speech_task.add_done_callback(log_task_exc)

    async def cancel_proactive_speech(self):
        if self._proactive_speech_task and not self._proactive_speech_task.done():
            # Don't cancel if we are running inside the task (recursion case)
            if self._proactive_speech_task == asyncio.current_task():
                return

            self._proactive_speech_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_speech_task
            self._proactive_speech_task = None

    async def _proactive_speech_loop(self, skip_initial_wait: bool = False):
        try:
            # Wait a reasonable amount of time first to allow for natural conversation flow
            if not skip_initial_wait:
                self._session_logger.debug(
                    "proactive_speech",
                    "Waiting 10s before checking for silence",
                )
                await asyncio.sleep(10)
            else:
                self._session_logger.debug(
                    "proactive_speech",
                    "Skipping initial wait (reschedule after false decision)",
                )

            self._session_logger.debug(
                "proactive_speech",
                "Entering _proactive_speech_loop",
            )

            # Get conversation turns and last message timestamp using helper
            conversation_turns, last_message_timestamp = (
                self.get_recent_voice_transcript()
            )

            # Calculate elapsed time from last message timestamp
            if last_message_timestamp:
                now = prompt_now(as_string=False)
                if isinstance(last_message_timestamp, datetime):
                    elapsed_seconds = (now - last_message_timestamp).total_seconds()
                else:
                    elapsed_seconds = 0
            else:
                elapsed_seconds = 0

            self._session_logger.debug(
                "proactive_speech",
                f"Elapsed time since last message: {elapsed_seconds:.1f}s",
            )

            # Build system prompt dynamically (there's no self.system_prompt attribute)
            brain_spec = build_brain_spec(self)
            decision = await self.proactive_speech.decide(
                conversation_turns,
                brain_spec.system_prompt.flatten(),
                elapsed_seconds=elapsed_seconds,
            )
            self._session_logger.debug(
                "proactive_speech",
                f"Decision: should_speak={decision.should_speak}, delay={decision.delay}s",
            )

            if not decision.should_speak:
                # Adaptive wait: if we're already past 10s, check more frequently (5s)
                # Otherwise, wait until we hit ~12s threshold (but cap at 7s max wait)
                if elapsed_seconds < 10:
                    wait_time = min(
                        12 - elapsed_seconds,
                        7,
                    )  # Wait until ~12s, but max 7s
                else:
                    wait_time = 5  # Already past 10s, check every 5s

                self._session_logger.debug(
                    "proactive_speech",
                    f"Not speaking (delay={decision.delay}s), will check again in {wait_time:.1f}s",
                )
                await asyncio.sleep(wait_time)
                # Skip initial wait when rescheduling since we just waited
                await self.schedule_proactive_speech(skip_initial_wait=True)
                return

            self._session_logger.info(
                "proactive_speech",
                f"Speaking in {decision.delay}s: {decision.content}",
            )
            await asyncio.sleep(decision.delay)

            # Record in contact_index
            contact = self.get_active_contact()
            if contact:
                contact_id = contact.get("contact_id")
                voice_medium = (
                    Medium.UNIFY_MEET if self.mode == Mode.MEET else Medium.PHONE_CALL
                )
                self.contact_index.push_message(
                    contact_id=contact_id,
                    sender_name="You",
                    thread_name=voice_medium,
                    message_content=decision.content,
                    role="assistant",
                )

            # Publish to Voice Agent via call_guidance channel
            await self.event_broker.publish(
                "app:call:call_guidance",
                json.dumps({"content": decision.content}),
            )

            await self.schedule_proactive_speech()

        except asyncio.CancelledError:
            self._session_logger.debug("proactive_speech", "Task cancelled")
            raise
        except Exception as e:
            self._session_logger.error("proactive_speech", f"Error in loop: {e}")
