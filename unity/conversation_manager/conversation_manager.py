import asyncio
from typing import Any, Optional
import contextlib
from datetime import datetime

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS
from unity.manager_registry import SingletonABCMeta
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common.hierarchical_logger import SessionLogger
from unity.conversation_manager import assistant_jobs
from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unity.conversation_manager.domains.contact_index import (
    ContactIndex,
    CommsMessage,
    Message,
)
from unity.conversation_manager.domains.brain import build_brain_spec
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.brain_tools import ConversationManagerBrainTools
from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.conversation_manager.domains.comms_utils import publish_system_error
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.renderer import Renderer
from unity.conversation_manager.events import *
from unity.common.prompt_helpers import now as prompt_now

from unity.common.llm_client import new_llm_client
from unity.common.single_shot import single_shot_tool_decision
from unity.events.manager_event_logging import _EVENT_SOURCE
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.domains.utils import Debouncer, log_task_exc

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager.cm_types import Medium, Mode, ScreenshotEntry
from unity.conversation_manager.cm_types.screenshot import (
    generate_screenshot_path,
    write_screenshot_to_disk,
)
from unity.actor.base import BaseActor
from unity.conversation_manager.domains.proactive_speech import ProactiveSpeech
from unity.conversation_manager.medium_scripts.common import FastBrainLogger

MAX_CONV_MANAGER_MSGS = 50


def _render_action_context(
    in_flight_actions: dict,
    completed_actions: dict,
    notifications: list | None = None,
) -> str | None:
    """Build an action-status summary with recent progress for proactive speech."""
    lines: list[str] = []
    for handle_data in in_flight_actions.values():
        query = handle_data.get("query", "unknown")
        action_type = handle_data.get("action_type", "act")
        lines.append(f"- EXECUTING ({action_type}): {query}")
        for entry in handle_data.get("handle_actions", [])[-5:]:
            name = entry.get("action_name", "")
            if name in ("act_started", "desktop_act_started", "web_act_started"):
                continue
            detail = str(entry.get("query", ""))[:200]
            ts = entry.get("timestamp", "")
            lines.append(f"    [{ts}] {name}: {detail}")
    for handle_data in completed_actions.values():
        query = handle_data.get("query", "unknown")
        lines.append(f"- COMPLETED: {query}")
    if notifications:
        recent = [n for n in notifications[-5:] if n.type not in ("Meet",)]
        if recent:
            lines.append("")
            lines.append("Recent system events:")
            for n in recent:
                lines.append(f"  - [{n.type}] {n.content[:150]}")
    if not lines:
        return None
    header = "[action status] Current actions and recent progress:"
    return f"{header}\n" + "\n".join(lines)


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker,
        job_name: str,
        user_id: str,
        assistant_id: int | None,
        user_first_name: str,
        user_surname: str,
        assistant_first_name: str,
        assistant_surname: str,
        assistant_age: str,
        assistant_nationality: str,
        assistant_about: str,
        assistant_number: str,
        assistant_email: str,
        user_number: str,
        user_email: str = None,
        voice_provider: str = "cartesia",
        voice_id: str = None,
        assistant_timezone: str = "",
        assistant_whatsapp_number: str = "",
        assistant_discord_bot_id: str = "",
        assistant_email_provider: str = "",
        assistant_job_title: str = "",
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
        stop: asyncio.Event = None,
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_first_name = assistant_first_name
        self.assistant_surname = assistant_surname
        self.assistant_age = assistant_age
        self.assistant_nationality = assistant_nationality
        self.assistant_timezone = assistant_timezone
        self.assistant_about = assistant_about
        self.assistant_job_title = assistant_job_title
        self.voice_provider = voice_provider
        self.voice_id = voice_id

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.assistant_whatsapp_number = assistant_whatsapp_number
        self.assistant_discord_bot_id = assistant_discord_bot_id
        self.assistant_email_provider = assistant_email_provider
        self.user_first_name = user_first_name
        self.user_surname = user_surname
        self.user_number = user_number
        self.user_email = user_email

        # initialization state
        self.initialized: bool = False
        self.ready_for_brain: bool = True
        self.vm_ready: bool = False
        self.file_sync_complete: bool = False
        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name

        # inactivity & shutdown
        self.inactivity_timeout = 420  # 7 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self.shutdown_reason: str | None = None
        self.stop = stop

        self.event_broker = event_broker

        # managers
        self.transcript_manager: TranscriptManager = None
        self.contact_manager: ContactManager = None
        self.memory_manager: MemoryManager = None
        self.actor: BaseActor | None = None

        self.debouncer = Debouncer(name="ConversationManager")

        # call manager - pass event_broker for socket IPC with voice agent subprocess
        self.call_manager = LivekitCallManager(self.get_call_config(), event_broker)
        self.call_manager.on_screenshot = self._buffer_screenshot
        self.call_manager.on_fast_brain_generating = self._on_fast_brain_generating
        self.call_manager.on_pipeline_quiescent = self._on_pipeline_quiescent

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a property with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
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

        # meet interaction state (screen share / webcam / remote control)
        self.assistant_screen_share_active: bool = False
        self.user_screen_share_active: bool = False
        self.user_webcam_active: bool = False
        self.user_remote_control_active: bool = False

        # screenshot buffer for slow brain visual context
        self._screenshot_buffer: list[ScreenshotEntry] = []
        # mapping from local_message_id (ephemeral CM counter) to
        # global message_id (persistent backend TM id), populated by
        # log_message() for post-hoc screenshot image updates.
        self._local_to_global_message_ids: dict[int, int] = {}

        # mapping from conference_name/room_name to exchange_id, populated
        # at call/meet end so the async RecordingReady handler can resolve
        # the exchange without a database filter query.
        self._recording_exchange_ids: dict[str, int] = {}

        # proactive speech
        self.proactive_speech = ProactiveSpeech()
        self._proactive_speech_task: asyncio.Task | None = None
        self._proactive_speech_gen: int = 0
        self._voice_pipeline_quiescent = asyncio.Event()
        self._voice_pipeline_quiescent.set()
        self._proactive_logger = FastBrainLogger("ProactiveSpeech")

        # ask handles (for Actor actions)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool]] = []
        self._pending_llm_request_meta: list[dict[str, str]] = []
        self._current_event_trace: dict[str, str] | None = None
        self._event_trace_seq: int = 0
        self._llm_request_seq: int = 0
        self._llm_run_seq: int = 0
        self._llm_gen: int = 0
        self._outbound_suppress_gen: int = -1

        # WhatsApp messages that were sent via greeting template (outside 24h
        # window). When the contact replies, the brain is notified so it can
        # resend or rework the original message.  Maps contact_id → content.
        self._pending_whatsapp_resends: dict[int, str] = {}

        # Outbound WhatsApp call contexts stashed while awaiting call permission.
        # When the contact grants permission (taps "Call now"), the context is
        # injected as call_manager.initial_notification.  Maps contact_id → context.
        self._pending_whatsapp_call_contexts: dict[int, str] = {}
        self._startup_wake_reasons: list[dict[str, Any]] = []

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.debug(
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
    def assistant_has_teams(self) -> bool:
        """True when Microsoft Teams capabilities are available to this assistant.

        Derived from the email provider rather than a dedicated flag because
        Teams access is gated by the same MS365 OAuth grant that backs the
        assistant's email — provisioning the Teams scopes without an MS365
        mailbox is not a supported configuration. Update both `assistant_has_teams`
        and `_assistant_has_teams` in `unity.comms.primitives` together if a
        first-class Teams flag is ever introduced.
        """
        return self.assistant_email_provider == "microsoft_365"

    @property
    def session_logger(self) -> SessionLogger:
        """The hierarchical session logger for this ConversationManager instance."""
        return self._session_logger

    @property
    def computer_primitives(self):
        """Lazily resolve the ``ComputerPrimitives`` singleton via ManagerRegistry."""
        from unity.function_manager.primitives.runtime import ComputerPrimitives
        from unity.manager_registry import ManagerRegistry

        return ManagerRegistry.get_instance(ComputerPrimitives)

    @property
    def computer_fast_path_eligible(self) -> bool:
        """True when the CM should expose computer fast-path tools.

        Requires assistant screen share to be active.  The tools are available
        regardless of whether an in-flight ``act`` session has already invoked
        computer primitives — the prompt guides the LLM to spin up a concurrent
        ``act(persist=True)`` session when one isn't already running.
        """
        return self.assistant_screen_share_active

    def get_coordinator_goal_context(
        self,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        """Return Coordinator onboarding state for prompt rendering."""

        if not self.initialized:
            return None, None
        if not SESSION_DETAILS.is_coordinator:
            return None, None

        from unity.coordinator_manager.coordinator_manager import (
            CoordinatorOnboardingManager,
        )

        manager = CoordinatorOnboardingManager()
        state = manager.get_state()
        if state is None:
            return None, None
        return state, manager.get_checklist()

    def get_active_contact(self) -> dict | None:
        """Get the contact for the current active call, or fall back to the boss contact."""
        return self.call_manager.call_contact or self.contact_index.get_contact(
            contact_id=1,
        )

    async def capture_assistant_screenshot(
        self,
        user_utterance: str,
        local_message_id: int | None = None,
        *,
        cached: bool = False,
    ) -> None:
        """Capture the assistant's screen and buffer it for the next slow brain turn.

        When *cached* is True, reads from the agent-service screenshot cache
        (~0 ms) instead of doing a live Playwright capture (~500 ms).  The
        cache is updated automatically after every Magnitude action.

        Runs the HTTP call in a thread to avoid event loop starvation — the
        main process event loop is shared with the actor and managers, which
        can saturate it during heavy async work.
        """
        import asyncio
        import time as _time
        from datetime import datetime, timezone

        import requests as _requests

        from unity.conversation_manager.medium_scripts.common import (
            _resolve_agent_service_url,
            _ensure_jpeg,
        )

        base_url = _resolve_agent_service_url()
        if cached:
            url = f"{base_url}/screenshot/latest"
        else:
            url = f"{base_url}/screenshot"
        auth_key = SESSION_DETAILS.unify_key

        def _sync_capture() -> dict | None:
            t0 = _time.monotonic()
            try:
                resp = _requests.post(
                    url,
                    json={},
                    headers={"authorization": f"Bearer {auth_key}"},
                    timeout=10,
                )
                total_ms = (_time.monotonic() - t0) * 1000
                if resp.status_code >= 400:
                    self._session_logger.warning(
                        "screenshot_capture",
                        f"Screenshot capture failed: HTTP {resp.status_code} "
                        f"url={url} total={total_ms:.0f}ms "
                        f"body={resp.text[:200]}",
                    )
                    return None
                data = resp.json()
                self._session_logger.debug(
                    "screenshot_capture",
                    f"Screenshot capture OK: url={url} "
                    f"total={total_ms:.0f}ms "
                    f"b64_len={len(data.get('screenshot', ''))}",
                )
                return data
            except Exception as e:
                total_ms = (_time.monotonic() - t0) * 1000
                self._session_logger.warning(
                    "screenshot_capture",
                    f"Screenshot capture error: {type(e).__name__}: {e} "
                    f"url={url} total={total_ms:.0f}ms",
                )
                return None

        data = await asyncio.to_thread(_sync_capture)
        if data and self.assistant_screen_share_active:
            b64 = data.get("screenshot")
            if b64:
                b64 = _ensure_jpeg(b64)
                self._screenshot_buffer.append(
                    ScreenshotEntry(
                        b64,
                        user_utterance,
                        datetime.now(timezone.utc),
                        "assistant",
                        local_message_id,
                    ),
                )

    def peek_screenshot_buffer(self) -> list[ScreenshotEntry]:
        """Return a snapshot of buffered screenshots without clearing.

        The buffer remains intact so that if the consuming operation
        (e.g. an LLM turn) is cancelled before completion, the next
        attempt will re-process the same screenshots.  Call
        :meth:`commit_screenshot_buffer` after all side effects have
        succeeded to remove the consumed entries.
        """
        return list(self._screenshot_buffer)

    def commit_screenshot_buffer(self, count: int) -> None:
        """Remove the first *count* entries from the screenshot buffer.

        Called after the LLM turn has successfully consumed and persisted
        the screenshots returned by :meth:`peek_screenshot_buffer`.
        Any screenshots that arrived *during* the turn (appended after the
        peek) are preserved for the next turn.
        """
        del self._screenshot_buffer[:count]

    async def _register_screenshots_background(
        self,
        screenshots: list[ScreenshotEntry],
        screenshot_paths: list[str],
    ) -> None:
        """Persist screenshots to disk and register with ImageManager / TM.

        Runs as a fire-and-forget background task after a successful LLM turn.
        None of these operations affect the LLM prompt or decision — they are
        purely persistence bookkeeping (disk write, image storage, transcript
        annotation).
        """
        source_labels = {"assistant": "Assistant's screen", "user": "User's screen"}

        # 0. Write screenshots to disk (entries with filepath already set by
        #    the fast brain are skipped — write_screenshot_to_disk is a no-op
        #    for those).
        for entry, path in zip(screenshots, screenshot_paths):
            if not entry.filepath:
                write_screenshot_to_disk(entry, path)

        # 1. Register with ImageManager to get persistent image_ids.
        image_ids: list[int] = []
        try:
            from unity.manager_registry import ManagerRegistry

            image_manager = ManagerRegistry.get_image_manager()
            items = [
                {
                    "data": entry.b64,
                    "timestamp": entry.timestamp,
                    "filepath": path,
                }
                for entry, path in zip(screenshots, screenshot_paths)
            ]
            image_ids = await asyncio.to_thread(
                image_manager.add_images,
                items,
                synchronous=True,
            )
        except Exception as e:
            self._session_logger.warning(
                "screenshot_registration",
                f"ImageManager registration failed, skipping: {e}",
            )
            return

        # 2. Annotate CM Message objects with image_ids and build TM refs.
        msg_to_image_refs: dict[int, list[dict]] = {}
        for i, (entry, _path) in enumerate(zip(screenshots, screenshot_paths)):
            if entry.local_message_id is None or i >= len(image_ids):
                continue
            mid = entry.local_message_id
            img_id = image_ids[i]

            # Attach image_id to the Message object.
            for gte in self.contact_index.global_thread:
                msg = gte.message
                if isinstance(msg, Message) and msg.local_message_id == mid:
                    if not hasattr(msg, "image_ids") or msg.image_ids is None:
                        msg.image_ids = []
                    msg.image_ids.append(img_id)
                    break

            label = source_labels.get(entry.source, "Screenshot")
            msg_to_image_refs.setdefault(mid, []).append(
                {
                    "raw_image_ref": {"image_id": img_id},
                    "annotation": f"{label} -- '{entry.utterance}'",
                },
            )

        # 3. Post-hoc update TM messages with AnnotatedImageRefs.
        if msg_to_image_refs and self.transcript_manager is not None:
            for local_mid, refs in msg_to_image_refs.items():
                tm_msg_id = self._local_to_global_message_ids.get(local_mid)
                if tm_msg_id is not None:
                    try:
                        await asyncio.to_thread(
                            self.transcript_manager.update_message_images,
                            tm_msg_id,
                            refs,
                        )
                    except Exception as e:
                        self._session_logger.warning(
                            "screenshot_tm_update",
                            f"TM image update failed for msg {tm_msg_id}: {e}",
                        )

    def _claim_pending_user_screenshot(self, local_message_id: int) -> None:
        """Stamp the most recent unclaimed user screenshot with the given local_message_id."""
        if self._screenshot_buffer:
            last = self._screenshot_buffer[-1]
            if last.source == "user" and last.local_message_id is None:
                self._screenshot_buffer[-1] = last._replace(
                    local_message_id=local_message_id,
                )

    def _buffer_screenshot(self, event_json: str) -> None:
        """Buffer a screenshot received from the fast brain via IPC.

        Accepts both user and assistant screenshots, distinguished by the
        ``source`` field in the JSON payload.  When a ``filepath`` is included,
        the file has already been written to disk by the fast brain.
        """
        import json as _json
        from datetime import datetime, timezone

        try:
            data = _json.loads(event_json)
            b64 = data.get("b64", "")
            utterance = data.get("utterance", "")
            source = data.get("source", "user")
            filepath = data.get("filepath")
            ts_str = data.get("timestamp")
            ts = (
                datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            )
            if b64:
                self._screenshot_buffer.append(
                    ScreenshotEntry(b64, utterance, ts, source, filepath=filepath),
                )
                self._session_logger.debug(
                    "screenshot_capture",
                    f"Buffered {source} screenshot #{len(self._screenshot_buffer)} "
                    f"for utterance: {utterance[:60]}...",
                )
        except Exception as e:
            self._session_logger.warning(
                "screenshot_capture",
                f"Error buffering screenshot: {e}",
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

        if self.call_manager.has_active_google_meet:
            voice_medium = Medium.GOOGLE_MEET
        elif self.call_manager.has_active_teams_meet:
            voice_medium = Medium.TEAMS_MEET
        elif self.mode == Mode.MEET:
            voice_medium = Medium.UNIFY_MEET
        elif self.call_manager._call_channel == "whatsapp_call":
            voice_medium = Medium.WHATSAPP_CALL
        else:
            voice_medium = Medium.PHONE_CALL
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

    _USER_ORIGIN_EVENTS = frozenset(
        {
            "InboundPhoneUtterance",
            "InboundUnifyMeetUtterance",
            "InboundWhatsAppCallUtterance",
        },
    )

    async def interject_or_run(
        self,
        content: str,
        triggering_contact_id: int | None = None,
    ):
        """Interject the ask handle or run the LLM"""
        prev_utterance = getattr(self, "_last_inbound_utterance", None)
        self._last_inbound_utterance = content

        if self.active_ask_handle and not self.active_ask_handle.done():
            await self.active_ask_handle.interject(content)
        else:
            if self.mode.is_voice:
                running_origin = self.debouncer.running_task_trace_meta.get(
                    "origin_event_name",
                    "",
                )
                running_is_known_non_user = (
                    running_origin != ""
                    and running_origin not in self._USER_ORIGIN_EVENTS
                )
                has_running = (
                    self.debouncer.running_task is not None
                    and not self.debouncer.running_task.done()
                )
                # Deterministically preempt non-user slow-brain runs.
                # Only cancel when the running task is *known* to be a
                # non-user event. Unknown origin (empty) defaults to the
                # safe queue-of-2 behavior.
                cancel_running = has_running and running_is_known_non_user
            else:
                # Text mode: rapid messages should get fresh responses.
                cancel_running = True

            await self.request_llm_run(
                delay=0,
                cancel_running=cancel_running,
                triggering_contact_id=triggering_contact_id,
                is_user_origin=True,
            )

            if (
                self.mode.is_voice
                and not cancel_running
                and SETTINGS.conversation.SPEECH_URGENCY_PREEMPT_ENABLED
                and self.debouncer.running_task
                and not self.debouncer.running_task.done()
            ):
                stale_task = self.debouncer.running_task
                asyncio.create_task(
                    self._evaluate_speech_urgency(
                        content,
                        stale_task,
                        prev_utterance,
                    ),
                )

    async def _evaluate_speech_urgency(
        self,
        utterance: str,
        stale_task: asyncio.Task,
        previous_utterance: str | None = None,
    ) -> None:
        """Concurrent sidecar: evaluate whether *utterance* should preempt the slow brain."""
        from unity.conversation_manager.domains.speech_urgency import (
            SpeechUrgencyEvaluator,
        )

        trace_meta = self.debouncer.running_task_trace_meta
        origin_event = trace_meta.get("origin_event_name", "unknown")
        elapsed = self.loop.time() - self.debouncer.running_task_started_at

        actions_parts = []
        for info in self.in_flight_actions.values():
            action_type = info.get("action_type", "unknown")
            query = info.get("query", "")
            actions_parts.append(f"{action_type}: {query!r}")
        actions_summary = "; ".join(actions_parts) if actions_parts else "none"

        evaluator = SpeechUrgencyEvaluator(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        )
        decision = await evaluator.evaluate(
            utterance=utterance,
            origin_event=origin_event,
            elapsed_seconds=elapsed,
            actions_summary=actions_summary,
            previous_utterance=previous_utterance,
        )

        self._session_logger.debug(
            "speech_urgency",
            (
                f"Urgency eval: urgent={decision.urgent} "
                f"utterance={utterance!r} origin={origin_event} "
                f"elapsed={elapsed:.1f}s reasoning={decision.reasoning!r}"
            ),
        )

        if not decision.urgent:
            return

        # Only cancel if the same stale task is still the running task.
        # If it completed (or a new task started) the utterance is already
        # being processed — cancelling would be counterproductive.
        if self.debouncer.running_task is stale_task and not stale_task.done():
            self._session_logger.debug(
                "speech_urgency",
                "Preempting stale slow-brain run — pending task will promote",
            )
            stale_task.cancel()

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(
        self,
        delay: float = 0,
        cancel_running: bool = False,
        trace_meta: dict[str, str] | None = None,
        is_user_origin: bool = False,
    ):
        await self.debouncer.submit(
            self._run_llm_with_failure_notification,
            kwargs={"trace_meta": trace_meta or {}},
            delay=delay,
            cancel_running=cancel_running,
            label=(trace_meta or {}).get("origin_event_name", ""),
            trace_meta=trace_meta,
            is_user_origin=is_user_origin,
        )

    @staticmethod
    def _is_transient_llm_error(exc: BaseException) -> bool:
        """True if ``exc`` is a provider-side transient error after unillm retries.

        unillm (``retry_transient_400_async``) already retries these internally
        with exponential backoff. If one escapes, it means the provider stayed
        unhealthy for the whole retry budget — e.g. Anthropic HTTP 529
        ``overloaded_error`` surfaces as ``litellm.InternalServerError``.
        """
        import litellm

        return isinstance(
            exc,
            (
                litellm.InternalServerError,
                litellm.ServiceUnavailableError,
                litellm.RateLimitError,
            ),
        )

    async def _notify_fast_brain_of_slow_brain_failure(
        self,
        exc: BaseException,
    ) -> None:
        """Surface a slow-brain exhaustion failure to the fast brain.

        Publishes a ``FastBrainNotification`` with ``should_speak=True`` and
        an explicit ``response_text`` so the fast brain utters the error via
        TTS directly (bypassing its own LLM, which may be hitting the same
        provider outage). Also cancels any pending proactive-speech loop so
        it stops emitting "still looking" filler for a request the slow brain
        has given up on.
        """
        response_text = (
            "Sorry, I'm having trouble thinking right now — "
            "could you say that again in a moment?"
        )
        notification_content = (
            f"Slow-brain turn failed after retries were exhausted "
            f"({type(exc).__name__}). The user's last request was not processed. "
            "Acknowledge the error and ask them to try again; do NOT claim you "
            "are still working on the prior request."
        )
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact or {},
            content=notification_content,
            response_text=response_text,
            should_speak=True,
            source="slow_brain_failure",
        )
        self._session_logger.info(
            "slow_brain_failure",
            (
                f"Notifying fast brain of slow-brain failure: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        event_json = event.to_json()
        await self.event_broker.publish("app:call:notification", event_json)
        await self.event_broker.publish(
            "app:comms:assistant_notification",
            event_json,
        )

        with contextlib.suppress(Exception):
            await self.cancel_proactive_speech()

    async def _run_llm_with_failure_notification(
        self,
        trace_meta: dict[str, str] | None = None,
    ) -> list[str] | None:
        """Wrap ``_run_llm`` so transient provider failures reach the user.

        Previously, a failed slow-brain turn only produced a ``log_task_exc``
        line in the logs — the user was left in silence while
        ``ProactiveSpeech`` continued to emit "still looking…" filler. This
        wrapper catches transient LLM errors, publishes a
        ``FastBrainNotification`` so the fast brain explicitly apologises and
        asks the user to retry, then re-raises so the existing failure log is
        preserved.
        """
        try:
            return await self._run_llm(trace_meta=trace_meta)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.mode.is_voice and self._is_transient_llm_error(exc):
                with contextlib.suppress(Exception):
                    await self._notify_fast_brain_of_slow_brain_failure(exc)
            raise

    async def request_llm_run(
        self,
        delay=0,
        cancel_running=False,
        triggering_contact_id: int | None = None,
        is_user_origin: bool = False,
    ) -> None:
        """Request an LLM run.

        The request is recorded and later scheduled by the event loop after
        the current event is handled.
        """
        self._llm_request_seq += 1
        request_id = f"llmreq-{self._llm_request_seq:06d}"
        event_trace = self._current_event_trace or {}
        request_meta = {
            "request_id": request_id,
            "origin_event_id": event_trace.get("event_id", ""),
            "origin_event_name": event_trace.get("event_name", ""),
            "triggering_contact_id": triggering_contact_id,
            "is_user_origin": is_user_origin,
        }
        self._pending_llm_requests.append((delay, cancel_running, is_user_origin))
        self._pending_llm_request_meta.append(request_meta)
        self._session_logger.debug(
            "llm_queue",
            (
                f"Queued slow-brain run request_id={request_id} "
                f"origin_event_id={request_meta['origin_event_id'] or '-'} "
                f"origin_event={request_meta['origin_event_name'] or '-'} "
                f"delay={delay} cancel_running={cancel_running} "
                f"is_user_origin={is_user_origin}"
            ),
        )

    async def flush_llm_requests(self) -> None:
        """Schedule any pending LLM runs recorded during event handling."""
        if not self._pending_llm_requests:
            return
        if not self.ready_for_brain:
            return

        requests = self._pending_llm_requests
        metas = self._pending_llm_request_meta

        # Prefer the newest user-origin request; fall back to the newest overall.
        selected_idx = len(requests) - 1
        for i in range(len(requests) - 1, -1, -1):
            if requests[i][2]:  # is_user_origin
                selected_idx = i
                break

        dropped_requests = len(requests) - 1
        delay, cancel_running, is_user_origin = requests[selected_idx]
        selected_meta = dict(metas[selected_idx]) if metas else {}

        self._pending_llm_requests.clear()
        self._pending_llm_request_meta.clear()

        self._llm_run_seq += 1
        run_id = f"llmrun-{self._llm_run_seq:06d}"
        selected_meta["run_id"] = run_id
        selected_meta["dropped_requests"] = str(dropped_requests)

        self._session_logger.debug(
            "llm_thinking",
            (
                f"Dispatching slow-brain run_id={run_id} "
                f"request_id={selected_meta.get('request_id', '-')} "
                f"origin_event_id={selected_meta.get('origin_event_id', '-') or '-'} "
                f"origin_event={selected_meta.get('origin_event_name', '-') or '-'} "
                f"dropped_requests={dropped_requests} delay={delay} "
                f"cancel_running={cancel_running} is_user_origin={is_user_origin}"
            ),
        )
        await self.run_llm(
            delay=delay,
            cancel_running=cancel_running,
            trace_meta=selected_meta,
            is_user_origin=is_user_origin,
        )

    async def _run_llm(self, trace_meta: dict[str, str] | None = None) -> list[str]:
        """Run a single LLM decision and return all tool names that were called."""
        import time as _rl_time

        from datetime import datetime, timezone

        from ..events.cost_attribution import COST_ATTRIBUTION

        _preamble_t0 = _rl_time.perf_counter()

        def _ms_since_start() -> str:
            return f"{(_rl_time.perf_counter() - _preamble_t0) * 1000:.0f}ms"

        trace_meta = trace_meta or {}

        if SESSION_DETAILS.org_id is not None:
            triggering_contact_id = trace_meta.get("triggering_contact_id")
            attributed_user_id = None
            if triggering_contact_id is not None:
                contact = self.contact_index.get_contact(
                    contact_id=triggering_contact_id,
                )
                if contact and contact.get("is_system"):
                    attributed_user_id = contact.get("user_id")
            if attributed_user_id:
                COST_ATTRIBUTION.set([attributed_user_id])
            else:
                COST_ATTRIBUTION.set([SESSION_DETAILS.user.id])

            try:
                import unillm

                unillm.set_billing_context(
                    assistant_id=SESSION_DETAILS.assistant.agent_id,
                    user_id=attributed_user_id or SESSION_DETAILS.user.id,
                    organization_id=SESSION_DETAILS.org_id,
                    source="call" if self.mode.is_voice else "chat",
                )
            except (ImportError, Exception):
                pass

        self._llm_gen += 1
        run_id = trace_meta.get("run_id", "llmrun-unknown")
        request_id = trace_meta.get("request_id", "")
        origin_event_id = trace_meta.get("origin_event_id", "")
        origin_event_name = trace_meta.get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            (
                f"Slow-brain run started run_id={run_id} "
                f"request_id={request_id or '-'} "
                f"origin_event_id={origin_event_id or '-'} "
                f"origin_event={origin_event_name or '-'} "
                f"was_queued={self.debouncer.was_queued} mode={self.mode}"
            ),
        )

        slow_brain_start_time = datetime.now(timezone.utc)

        screenshots = self.peek_screenshot_buffer()

        screenshot_paths = [
            s.filepath or generate_screenshot_path(s) for s in screenshots
        ]

        if screenshots:
            msg_to_paths: dict[int, list[str]] = {}
            for entry, path in zip(screenshots, screenshot_paths):
                if entry.local_message_id is not None:
                    msg_to_paths.setdefault(entry.local_message_id, []).append(path)
            if msg_to_paths:
                for gte in self.contact_index.global_thread:
                    msg = gte.message
                    if (
                        isinstance(msg, Message)
                        and msg.local_message_id in msg_to_paths
                    ):
                        msg.screenshots = msg_to_paths.pop(msg.local_message_id)
                    if not msg_to_paths:
                        break

        self.snapshot()

        web_sessions = None
        if self.assistant_screen_share_active:
            cp = self.computer_primitives
            if cp is not None:
                try:
                    web_sessions = await cp.web.list_sessions_with_metadata(
                        visible_only=True,
                        active_only=True,
                    )
                except Exception:
                    web_sessions = cp.web.list_sessions(
                        visible_only=True,
                        active_only=True,
                    )

        _t0 = _rl_time.perf_counter()
        _has_desktop = SESSION_DETAILS.assistant.desktop_mode in ("ubuntu", "windows")
        coordinator_state, coordinator_checklist = self.get_coordinator_goal_context()
        snapshot_state = self.prompt_renderer.render_state(
            self.contact_index,
            self.notifications_bar,
            self.in_flight_actions,
            self.completed_actions,
            self.last_snapshot,
            assistant_screen_share_active=self.assistant_screen_share_active,
            user_screen_share_active=self.user_screen_share_active,
            user_webcam_active=self.user_webcam_active,
            user_remote_control_active=self.user_remote_control_active,
            google_meet_active=self.call_manager.has_active_google_meet,
            teams_meet_active=self.call_manager.has_active_teams_meet,
            active_web_sessions=web_sessions,
            coordinator_state=coordinator_state,
            coordinator_checklist=coordinator_checklist,
            managers_initialized=self.initialized,
            vm_ready=self.vm_ready,
            file_sync_complete=self.file_sync_complete,
            has_desktop=_has_desktop,
        )
        _render_ms = (_rl_time.perf_counter() - _t0) * 1000

        _t0 = _rl_time.perf_counter()
        brain_spec = build_brain_spec(
            self,
            snapshot_state=snapshot_state,
            screenshots=screenshots,
            screenshot_paths=screenshot_paths,
        )
        _brain_spec_ms = (_rl_time.perf_counter() - _t0) * 1000

        if screenshots:
            self._session_logger.debug(
                "screen_share",
                f"Attaching {len(screenshots)} screenshot(s) to slow brain turn",
            )
        input_message = brain_spec.state_message()
        system_prompt = brain_spec.system_prompt

        self._current_state_snapshot = input_message

        self._current_snapshot_state = snapshot_state

        reason = (trace_meta or {}).get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            f"LLM thinking... ({reason})" if reason else "LLM thinking...",
        )

        response_model = brain_spec.response_model

        _t0 = _rl_time.perf_counter()
        brain_tools = ConversationManagerBrainTools(self)
        action_tools = ConversationManagerBrainActionTools(self)
        tools = {
            **brain_tools.as_tools(),
            **action_tools.as_tools(),
            **action_tools.build_action_steering_tools(),
            **action_tools.build_completed_action_tools(),
        }
        if SESSION_DETAILS.is_coordinator:
            coordinator_tools = CoordinatorTools(self)
            tools = {**tools, **coordinator_tools.as_tools()}

        if self.computer_fast_path_eligible:
            tools["desktop_act"] = action_tools.desktop_act
            tools["web_act"] = action_tools.web_act
            tools["close_web_session"] = action_tools.close_web_session
        _tools_ms = (_rl_time.perf_counter() - _t0) * 1000

        _t0 = _rl_time.perf_counter()
        client = new_llm_client(
            SETTINGS.UNIFY_MODEL,
            origin="ConversationManager",
        )
        if hasattr(client, "_pending_thinking_log"):
            parts = [
                p
                for p in [reason, "from queue" if self.debouncer.was_queued else ""]
                if p
            ]
            suffix = f" ({', '.join(parts)})" if parts else ""
            client._pending_thinking_log.set_thinking_context(suffix)
        client.set_system_message(system_prompt.to_list())
        client.set_prompt_caching(["system"])
        messages = self._preprocess_messages(self.chat_history + [input_message])
        _client_ms = (_rl_time.perf_counter() - _t0) * 1000

        _source_token = _EVENT_SOURCE.set("ConversationManager")

        _rl_t0 = _rl_time.perf_counter()

        def _rl_ms() -> str:
            return f"{(_rl_time.perf_counter() - _rl_t0) * 1000:.0f}ms"

        self._session_logger.debug(
            "perf",
            (
                f"[_run_llm preamble={_ms_since_start()}] "
                f"render_state={_render_ms:.0f}ms brain_spec={_brain_spec_ms:.0f}ms "
                f"tools={_tools_ms:.0f}ms client={_client_ms:.0f}ms | "
                f"calling single_shot_tool_decision ({len(tools)} tools, {len(messages)} msgs)"
            ),
        )
        try:
            result = await single_shot_tool_decision(
                client,
                messages,
                tools,
                tool_choice="required" if tools else "auto",
                response_format=response_model,
                exclusive_tools={
                    "make_call",
                    "make_whatsapp_call",
                    "join_google_meet",
                    "join_teams_meet",
                },
            )
        finally:
            if hasattr(client, "_pending_thinking_log"):
                client._pending_thinking_log.emit_fallback()
            _EVENT_SOURCE.reset(_source_token)
        tool_names = [t.name for t in result.tools]
        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] single_shot returned tools={tool_names}",
        )

        # Extract structured output (thoughts)
        structured = result.structured_output
        thoughts = ""
        if structured is not None:
            thoughts = getattr(structured, "thoughts", "")

        # Handle guide_voice_agent tool calls for voice modes.
        # The slow brain decides BLOCK (omit the tool), NOTIFY (default),
        # or SPEAK (should_speak=True + response_text) by calling
        # guide_voice_agent in parallel with its action tool.
        # Dedup is handled in the fast brain subprocess at speak time.
        if self.mode.is_voice:
            notification_content = ""
            should_speak = False
            response_text = ""
            for tool_exec in result.tools:
                if tool_exec.name == "guide_voice_agent":
                    args = tool_exec.args or {}
                    notification_content = args.get("content", "")
                    should_speak = args.get("should_speak", False)
                    response_text = args.get("response_text", "")
                    break

            if notification_content:
                pending = getattr(client, "_pending_thinking_log", None)
                slow_brain_log_path = (
                    pending.last_path or "" if pending is not None else ""
                )
                contact = self.get_active_contact()
                event = FastBrainNotification(
                    contact=contact,
                    content=notification_content,
                    response_text=response_text,
                    should_speak=should_speak,
                    source="slow_brain",
                    llm_log_path=slow_brain_log_path,
                )
                self._session_logger.info(
                    "call_notification",
                    f"Guide FastBrain (speak={should_speak}): {notification_content}",
                )
                event_json = event.to_json()
                await self.event_broker.publish(
                    "app:call:notification",
                    event_json,
                )
                await self.event_broker.publish(
                    "app:comms:assistant_notification",
                    event_json,
                )

        self._session_logger.debug(
            "llm_response",
            (
                f"run_id={run_id} thoughts: {thoughts[:100]}..."
                if len(thoughts) > 100
                else f"run_id={run_id} thoughts: {thoughts}"
            )
            + (f" | actions: {tool_names}" if tool_names else ""),
        )

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] voice notification done, committing",
        )
        self.commit()
        self._session_logger.debug("state_update", "Committing state")

        # Clear the temporary state snapshots now that tools have executed
        self._current_state_snapshot = None
        self._current_snapshot_state = None

        # The turn completed successfully — commit the screenshot buffer so
        # these entries are not re-processed on the next turn.  Any new
        # screenshots that arrived during this turn (appended after the peek)
        # are preserved.
        if screenshots:
            self.commit_screenshot_buffer(len(screenshots))
            asyncio.create_task(
                self._register_screenshots_background(
                    screenshots,
                    screenshot_paths,
                ),
            )

        # Build assistant message for chat history
        assistant_content = (
            structured.model_dump_json() if structured else result.text_response or ""
        )
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": assistant_content})

        # If the LLM called wait(delay=N), schedule a delayed follow-up turn.
        for tool_exec in result.tools:
            if tool_exec.name == "wait":
                delay = (tool_exec.args or {}).get("delay")
                msg = (
                    f"Decided to wait {delay} seconds"
                    if delay is not None
                    else "Decided to wait"
                )
                self._session_logger.info("wait", msg)
                if delay is not None:
                    await self.run_llm(delay=delay)
                break

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] post-processing done",
        )
        self._session_logger.debug(
            "llm_response",
            (f"Slow-brain run completed run_id={run_id} " f"tools={tool_names or '-'}"),
        )

        return tool_names

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
                channel = msg.get("channel", "")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                self._event_trace_seq += 1
                event_id = f"evt-{self._event_trace_seq:06d}"
                event_name = event.__class__.__name__
                self._current_event_trace = {
                    "event_id": event_id,
                    "event_name": event_name,
                }
                if event.__class__.loggable:
                    self._session_logger.debug(
                        "event_trace",
                        (
                            f"Processing event_id={event_id} "
                            f"event={event_name} channel={channel or '-'}"
                        ),
                    )
                try:
                    await EventHandler.handle_event(
                        event,
                        self,
                    )
                    await self.flush_llm_requests()
                except Exception as exc:
                    LOGGER.error(
                        f"⚠️ [EventLoop] Unhandled error processing "
                        f"event_id={event_id} event={event_name} "
                        f"channel={channel or '-'}: {exc}",
                        exc_info=True,
                    )
                    publish_system_error(
                        "An unexpected error occurred. The assistant is "
                        "attempting to recover.",
                        error_type="recovering",
                    )
                finally:
                    self._current_event_trace = None

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout.

        Activity is detected from two sources:
        - External pubsub messages (updated by wait_for_events via last_activity_time)
        - Internal EventBus publishes (LLM calls, tool-loop turns, manager methods)

        Ghost-publish detection: if pubsub is idle past the timeout but
        eventbus_idle stays suspiciously low for many consecutive checks,
        something is periodically resetting last_publish_monotonic without real
        user-facing activity. After ``_GHOST_PUBLISH_CHECKS`` consecutive such
        observations we shut down to prevent indefinite hangs.
        """
        import time as _time

        from unity.events.event_bus import EventBus

        _GHOST_PUBLISH_CHECKS = 20  # 20 * 30s = 10 minutes
        ghost_counter = 0

        while True:
            await asyncio.sleep(self.inactivity_check_interval)
            current_time = self.loop.time()
            pubsub_idle = current_time - self.last_activity_time
            eventbus_idle = _time.monotonic() - EventBus.last_publish_monotonic
            idle_seconds = min(pubsub_idle, eventbus_idle)

            if (
                pubsub_idle > self.inactivity_timeout
                and eventbus_idle < self.inactivity_timeout
            ):
                ghost_counter += 1
            else:
                ghost_counter = 0

            ghost_publish = ghost_counter >= _GHOST_PUBLISH_CHECKS

            if int(current_time) % 180 < self.inactivity_check_interval:
                extra = ""
                if ghost_counter > 0:
                    extra = f" ghost_count={ghost_counter}/{_GHOST_PUBLISH_CHECKS}"
                self._session_logger.info(
                    "inactivity_check",
                    f"Idle check: pubsub_idle={pubsub_idle:.1f}s, "
                    f"eventbus_idle={eventbus_idle:.1f}s, "
                    f"min_idle={idle_seconds:.1f}s, "
                    f"timeout={self.inactivity_timeout}s{extra}",
                )

            if idle_seconds > self.inactivity_timeout or ghost_publish:
                if ghost_publish:
                    log_str = (
                        f"Ghost-publish shutdown: pubsub_idle={pubsub_idle:.0f}s "
                        f"but eventbus_idle stuck at {eventbus_idle:.1f}s "
                        f"for {ghost_counter} consecutive checks "
                        f"(timeout={self.inactivity_timeout}s)"
                    )
                else:
                    self.shutdown_reason = "idle_timeout"
                    log_str = f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown"
                LOGGER.info(f"{DEFAULT_ICON} {log_str}")
                self._session_logger.info("session_end", log_str)
                self.stop.set()
                await self.event_broker.aclose()
                break  # Exit the loop after triggering shutdown

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details into SESSION_DETAILS."""
        self.user_id = payload["user_id"]
        self.assistant_id = int(payload["assistant_id"])
        self.assistant_first_name = payload["assistant_first_name"]
        self.assistant_surname = payload["assistant_surname"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_nationality = payload["assistant_nationality"]
        self.assistant_timezone = payload.get("assistant_timezone", "")
        self.assistant_about = payload["assistant_about"]
        self.assistant_job_title = payload.get("assistant_job_title", "")
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.assistant_email_provider = payload.get(
            "assistant_email_provider",
            "google_workspace",
        )
        self.self_contact_id = payload.get("self_contact_id", 0)
        self.boss_contact_id = payload.get("boss_contact_id", 1)
        self.assistant_whatsapp_number = payload.get("assistant_whatsapp_number", "")
        self.assistant_discord_bot_id = payload.get("assistant_discord_bot_id", "")
        self.user_first_name = payload["user_first_name"]
        self.user_surname = payload["user_surname"]
        self.user_number = payload["user_number"]
        self.user_email = payload["user_email"]
        self.user_whatsapp_number = payload.get("user_whatsapp_number", "")
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        self.binding_id = payload.get("binding_id", "")
        self.desktop_mode = payload.get("desktop_mode", "ubuntu")
        self.user_desktop_mode = payload.get("user_desktop_mode")
        self.user_desktop_filesys_sync = payload.get("user_desktop_filesys_sync", False)
        self.user_desktop_url = payload.get("user_desktop_url")
        self.org_id: int | None = payload.get("org_id")
        self.org_name: str = payload.get("org_name", "")
        self.team_ids: list[int] = payload.get("team_ids") or []
        self.space_ids: list[int] = payload.get("space_ids") or []
        space_summaries = payload.get("space_summaries") or []
        is_coordinator = payload.get("is_coordinator", False)
        # Set API key on SESSION_DETAILS for runtime access
        if payload.get("api_key"):
            SESSION_DETAILS.unify_key = payload["api_key"]
        # Populate the global SessionDetails singleton
        SESSION_DETAILS.populate(
            agent_id=self.assistant_id,
            assistant_first_name=self.assistant_first_name,
            assistant_surname=self.assistant_surname,
            assistant_age=self.assistant_age,
            assistant_nationality=self.assistant_nationality,
            assistant_timezone=self.assistant_timezone,
            assistant_about=self.assistant_about,
            assistant_job_title=self.assistant_job_title,
            assistant_number=self.assistant_number,
            assistant_email=self.assistant_email,
            assistant_email_provider=self.assistant_email_provider,
            assistant_self_contact_id=self.self_contact_id,
            assistant_whatsapp_number=self.assistant_whatsapp_number,
            assistant_discord_bot_id=self.assistant_discord_bot_id,
            user_id=self.user_id,
            user_first_name=self.user_first_name,
            user_surname=self.user_surname,
            user_number=self.user_number,
            user_email=self.user_email,
            user_whatsapp_number=self.user_whatsapp_number,
            user_boss_contact_id=self.boss_contact_id,
            org_id=self.org_id,
            org_name=self.org_name,
            team_ids=self.team_ids,
            space_ids=self.space_ids,
            space_summaries=space_summaries,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            binding_id=self.binding_id,
            desktop_mode=self.desktop_mode,
            user_desktop_mode=self.user_desktop_mode,
            user_desktop_filesys_sync=self.user_desktop_filesys_sync,
            user_desktop_url=self.user_desktop_url,
            is_coordinator=is_coordinator,
        )
        self.space_summaries = SESSION_DETAILS.space_summaries
        # Export to env vars for subprocess inheritance
        SESSION_DETAILS.export_to_env()

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_first_name": self.user_first_name,
            "user_surname": self.user_surname,
            "assistant_first_name": self.assistant_first_name,
            "assistant_surname": self.assistant_surname,
            "user_number": self.user_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def get_call_config(self) -> CallConfig:
        return CallConfig(
            assistant_id=self.assistant_id,
            user_id=self.user_id,
            assistant_bio=self.assistant_about,
            assistant_number=self.assistant_number,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            assistant_name=f"{self.assistant_first_name} {self.assistant_surname}".strip(),
            job_name=self.job_name,
            is_coordinator=SESSION_DETAILS.is_coordinator,
        )

    async def store_chat_history(self):
        if len(self.chat_history) >= 2:
            await self.event_broker.publish(
                "app:comms:chat_history",
                StoreChatHistory(chat_history=self.chat_history[-2:]).to_json(),
            )
            await asyncio.sleep(2)

    async def cleanup(self):
        """Clean up any running call processes and file sync."""
        await self.store_chat_history()
        local_ingress = getattr(self, "_local_comms_ingress", None)
        if local_ingress is not None:
            await local_ingress.stop()
        if self.call_manager.has_active_google_meet:
            await self.call_manager.cleanup_google_meet()
        elif self.call_manager.has_active_teams_meet:
            await self.call_manager.cleanup_teams_meet()
        else:
            await self.call_manager.cleanup_call_proc()
        await self.call_manager.cleanup_persistent_worker()

        await self._stop_file_sync()

        if self.job_name and self.assistant_id is not None:
            self._session_logger.debug(
                "session_end",
                f"Marking job {self.job_name} done",
            )
            mark_done_kwargs = {}
            if self.shutdown_reason:
                mark_done_kwargs["shutdown_reason"] = self.shutdown_reason
            assistant_jobs.mark_job_done(
                self.job_name,
                self.inactivity_timeout,
                **mark_done_kwargs,
            )
        self.stop.set()

    async def _stop_file_sync(self) -> None:
        """Stop file sync with managed VM."""
        if not self.initialized:
            return
        try:
            from unity.file_manager.managers.local import LocalFileManager

            local_fm = LocalFileManager()
            adapter = local_fm._adapter

            if not hasattr(adapter, "sync_started"):
                return

            if adapter.sync_started:
                LOGGER.debug(
                    f"{DEFAULT_ICON} [ConversationManager] Stopping file sync...",
                )
                await adapter.stop_sync()
                LOGGER.debug(f"{DEFAULT_ICON} [ConversationManager] File sync stopped")
        except Exception as e:
            LOGGER.error(
                f"{DEFAULT_ICON} [ConversationManager] Failed to stop file sync: {e}",
            )

    # Proactive speech related methods

    PROACTIVE_DEBOUNCE_SECONDS = 5

    def _on_fast_brain_generating(self) -> None:
        """Called via IPC when the fast brain starts generating a reply.

        Restarts the proactive speech cycle so any in-flight decision is
        cancelled.  The quiescence gate in ``_proactive_speech_loop`` will
        prevent the countdown from starting until the pipeline is idle again.
        """
        asyncio.ensure_future(self.schedule_proactive_speech())

    def _on_pipeline_quiescent(self, quiescent: bool) -> None:
        """Called via IPC when the voice pipeline quiescence state changes."""
        if quiescent:
            self._voice_pipeline_quiescent.set()
        else:
            self._voice_pipeline_quiescent.clear()

    async def schedule_proactive_speech(self):
        """Cancel any pending proactive speech and start a fresh cycle.

        Called on every user/assistant utterance event to reset the silence
        timer.  Only operates in voice modes (call / meet).
        """
        self._proactive_speech_gen += 1
        my_gen = self._proactive_speech_gen
        await self.cancel_proactive_speech()

        if not self.mode.is_voice:
            return

        if self._proactive_speech_gen != my_gen:
            return

        self._proactive_speech_task = asyncio.create_task(
            self._proactive_speech_loop(my_gen),
        )
        self._proactive_speech_task.add_done_callback(log_task_exc)

    async def cancel_proactive_speech(self):
        if self._proactive_speech_task and not self._proactive_speech_task.done():
            if self._proactive_speech_task == asyncio.current_task():
                return

            self._proactive_speech_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_speech_task
            self._proactive_speech_task = None

    async def _proactive_speech_loop(self, gen: int = 0):
        _log = self._proactive_logger

        def _superseded() -> bool:
            return self._proactive_speech_gen != gen

        try:
            if not self._voice_pipeline_quiescent.is_set():
                _log.proactive_waiting_for_quiescence()
                await self._voice_pipeline_quiescent.wait()
                if _superseded():
                    return

            _log.proactive_debounce(self.PROACTIVE_DEBOUNCE_SECONDS)
            await asyncio.sleep(self.PROACTIVE_DEBOUNCE_SECONDS)

            if _superseded():
                return

            if not self._voice_pipeline_quiescent.is_set():
                _log.proactive_deferred("pipeline not quiescent")
                return

            # Gather context for the decision.
            conversation_turns, _ = self.get_recent_voice_transcript()

            # Attach the latest screenshot from each active visual source
            # so the proactive LLM can visually verify screen state.
            screenshots = self.peek_screenshot_buffer()
            latest_by_source: dict[str, ScreenshotEntry] = {}
            for entry in screenshots:
                latest_by_source[entry.source] = entry

            if latest_by_source:
                source_labels = {
                    "assistant": "Assistant's Screen",
                    "user": "User's Screen",
                    "webcam": "User's Webcam",
                }
                content_parts: list[dict] = []
                for source, entry in latest_by_source.items():
                    label = source_labels.get(source, "Screenshot")
                    content_parts.append(
                        {
                            "type": "text",
                            "text": (f'[{label}] User said: "{entry.utterance}"'),
                        },
                    )
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{entry.b64}",
                            },
                        },
                    )
                conversation_turns.append(
                    {"role": "user", "content": content_parts},
                )
            else:
                active_visuals = []
                if self.user_screen_share_active:
                    active_visuals.append("the user is sharing their screen")
                if self.user_webcam_active:
                    active_visuals.append("the user's webcam is on")
                if self.assistant_screen_share_active:
                    active_visuals.append(
                        "the assistant's desktop is being shared",
                    )
                if active_visuals:
                    conversation_turns.append(
                        {
                            "role": "system",
                            "content": (
                                f"[context] "
                                f"{', '.join(active_visuals).capitalize()}."
                            ),
                        },
                    )

            _has_desktop = SESSION_DETAILS.assistant.desktop_mode in (
                "ubuntu",
                "windows",
            )
            coordinator_state, coordinator_checklist = (
                self.get_coordinator_goal_context()
            )
            snapshot_state = self.prompt_renderer.render_state(
                self.contact_index,
                self.notifications_bar,
                self.in_flight_actions,
                self.completed_actions,
                self.last_snapshot,
                assistant_screen_share_active=self.assistant_screen_share_active,
                user_screen_share_active=self.user_screen_share_active,
                user_webcam_active=self.user_webcam_active,
                user_remote_control_active=self.user_remote_control_active,
                google_meet_active=self.call_manager.has_active_google_meet,
                teams_meet_active=self.call_manager.has_active_teams_meet,
                coordinator_state=coordinator_state,
                coordinator_checklist=coordinator_checklist,
                vm_ready=self.vm_ready,
                file_sync_complete=self.file_sync_complete,
                has_desktop=_has_desktop,
            )
            brain_spec = build_brain_spec(self, snapshot_state=snapshot_state)

            action_context = _render_action_context(
                self.in_flight_actions,
                self.completed_actions,
                notifications=self.notifications_bar.notifications,
            )

            decision, llm_log_path = await self.proactive_speech.decide(
                conversation_turns,
                brain_spec.system_prompt.flatten(),
                action_context=action_context,
            )

            if _superseded():
                return

            _log.proactive_decision(
                decision.should_speak,
                decision.delay,
                decision.content,
            )

            if not decision.should_speak:
                _log.proactive_dormant()
                return

            # Wait the requested delay (cancellable if an utterance arrives).
            if decision.delay > 0:
                _log.proactive_speaking(decision.delay, decision.content)
                await asyncio.sleep(decision.delay)

            if _superseded():
                return

            # Record in contact_index.
            contact = self.get_active_contact()
            if contact:
                contact_id = contact.get("contact_id")
                if self.call_manager.has_active_google_meet:
                    voice_medium = Medium.GOOGLE_MEET
                elif self.call_manager.has_active_teams_meet:
                    voice_medium = Medium.TEAMS_MEET
                elif self.mode == Mode.MEET:
                    voice_medium = Medium.UNIFY_MEET
                elif self.call_manager._call_channel == "whatsapp_call":
                    voice_medium = Medium.WHATSAPP_CALL
                else:
                    voice_medium = Medium.PHONE_CALL
                self.contact_index.push_message(
                    contact_id=contact_id,
                    sender_name="You",
                    thread_name=voice_medium,
                    message_content=decision.content,
                    role="assistant",
                )

            event = FastBrainNotification(
                contact=contact or {},
                content=decision.content,
                response_text=decision.content,
                should_speak=True,
                source="proactive_speech",
                llm_log_path=llm_log_path,
            )
            await self.event_broker.publish(
                "app:call:notification",
                event.to_json(),
            )
            _log.proactive_published(decision.content)

        except asyncio.CancelledError:
            _log.proactive_cancelled()
            raise
        except Exception as e:
            _log.proactive_error(str(e))
