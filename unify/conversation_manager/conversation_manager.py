import asyncio
import collections
import contextlib
import json
import math
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional, Reversible

from unify.logger import LOGGER
from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.common.startup_timing import log_startup_timing
from unify.common.diagnostic_logging import staging_diagnostics_enabled
from unify.session_details import SESSION_DETAILS
from unify.coordinator_voice import resolve_runtime_voice
from unify.settings import SETTINGS
from unify.conversation_manager.console_actions import (
    parse_console_actions,
    strip_markers,
)
from unify.manager_registry import SingletonABCMeta
from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.hierarchical_logger import SessionLogger
from unify.conversation_manager import assistant_jobs
from unify.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unify.conversation_manager.domains.contact_index import (
    ContactIndex,
    CommsMessage,
    Message,
)
from unify.conversation_manager.domains.brain import build_brain_spec
from unify.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unify.conversation_manager.domains.brain_tools import ConversationManagerBrainTools
from unify.conversation_manager.domains.comms_utils import publish_system_error
from unify.conversation_manager.domains.event_handlers import EventHandler
from unify.conversation_manager.domains.renderer import Renderer
from unify.conversation_manager.events import *
from unify.integrations.sync_state import IntegrationSyncCoordinator
from unify.common.prompt_helpers import now as prompt_now

from unify.common.llm_client import new_slow_brain_llm_client
from unify.common.single_shot import ToolExecution, single_shot_tool_decision
from unify.events.manager_event_logging import _EVENT_SOURCE
from unify.conversation_manager.domains.notifications import NotificationBar
from unify.conversation_manager.domains.utils import Debouncer, log_task_exc

from unify.memory_manager.memory_manager import MemoryManager
from unify.contact_manager.contact_manager import ContactManager
from unify.transcript_manager.transcript_manager import TranscriptManager
from unify.conversation_manager.cm_types import Medium, Mode, ScreenshotEntry
from unify.conversation_manager.cm_types.screenshot import (
    generate_screenshot_path,
    write_screenshot_to_disk,
)
from unify.actor.base import BaseActor
from unify.conversation_manager.domains.proactive_speech import ProactiveSpeech
from unify.conversation_manager.medium_scripts.common import FastBrainLogger
from unillm.limit_hooks import SpendingLimitExceededError
from unify.spending_limits import (
    GATE_BLOCK_ACCOUNT_SUSPENDED,
    GATE_BLOCK_CREDITS_DEPLETED,
    check_billing_gate_state,
)

MAX_CONV_MANAGER_MSGS = 50
# Upper bound a deferred hang-up waits for its explanatory line to be spoken
# before tearing down anyway (guards a line that never surfaces). Generous so it
# rarely cuts off a legitimately-playing line.
_HANG_UP_SPEECH_TIMEOUT_S = 20.0
IDLE_SMALLTALK_RECENT_COMMS_SECONDS = 20.0
RECENT_TOOL_EXECUTIONS_LIMIT = 20
RECENT_TOOL_PREVIEW_CHARS = 500
CREDIT_GATE_REPLY_THROTTLE_SECONDS = 300
# Upper bound a slow-brain turn holds for boot hydration before rendering
# anyway. Hydration is one EventBus search (seconds); a hold that outlives
# this bound means hydration is stuck, and an eager reply from the
# pre-hydration view beats indefinite silence.
BOOT_HYDRATION_MAX_WAIT_SECONDS = 30.0
ONBOARDING_OUTBOUND_CONTEXT_TTL_SECONDS = 120
# How long a Console presence heartbeat keeps the Console orientation block in
# the prompt. Comfortably longer than Console's keep-warm interval so a user who
# is present but idle does not drop out of it between beats.
CONSOLE_PRESENCE_TTL_S = 600.0
DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE = (
    "Your credits are depleted, so I can't continue helping with setup or tasks "
    "until you top up. Please add credits in billing, then I'll pick this back up."
)
DEPLETED_CREDITS_EMAIL_SUBJECT = "Credits depleted"
ACCOUNT_SUSPENDED_SLOW_BRAIN_RESPONSE = (
    "This account is suspended, so I can't run setup or tasks right now. Adding "
    "a payment method in billing usually lifts it — if that isn't it, email "
    "support@unify.ai and we'll sort it out."
)
ACCOUNT_SUSPENDED_EMAIL_SUBJECT = "Account suspended"
SLOW_BRAIN_FAILURE_REPLY_THROTTLE_SECONDS = 600
SLOW_BRAIN_FAILURE_RESPONSE = (
    "I hit a technical problem and couldn't respond just now. Please try "
    "again in a moment — if it keeps happening, contact support@unify.ai "
    "so we can look into it."
)
SLOW_BRAIN_FAILURE_EMAIL_SUBJECT = "Temporary problem responding"
# A spending gate refuses a turn for a stated reason — no credits, a paid-only
# provider, a suspended account. The reason is written for the account holder
# and names the remedy, so it is delivered verbatim rather than folded into the
# transient-failure copy above: telling someone to try again in a moment is
# actively wrong when nothing about a retry can change the answer.
SPENDING_GATE_EMAIL_SUBJECT = "Action needed to continue"
# The spoken form drops the written remedy: a caller cannot click a billing
# link mid-sentence, and reading a URL aloud is worse than pointing at it.
SPENDING_GATE_SPOKEN_PREFIX = (
    "I can't run that right now for a billing reason on this account. "
)
# Self-scheduled wait(delay) polling backoff. Timer wakes are full-priced
# slow-brain turns; a model that busy-polls a long-running act ("check
# again in 10 seconds", repeatedly) burns tokens with zero information
# gain — external events wake the brain immediately regardless, so only
# the self-scheduled timer needs a floor. A small budget of fast polls
# per window stays free; beyond it the effective delay doubles per extra
# poll up to the cap.
WAIT_POLL_WINDOW_SECONDS = 600.0
WAIT_POLL_FREE_BUDGET = 5
WAIT_POLL_MIN_CLAMPED_DELAY_SECONDS = 60
WAIT_POLL_MAX_CLAMPED_DELAY_SECONDS = 600

# Meet-interaction surfaces, split by whose lifetime owns them. Every surface in
# ``_MEET_STATE_FLAGS`` (the event-handler registry that applies them) belongs to
# exactly one of the two groups below, which a test checks against that registry.
#
# Call-scoped surfaces exist only for the duration of one call, so a call
# boundary closes them. Each maps to the screenshot sources it feeds, because
# closing a surface and keeping its frames leaves the two disagreeing: the flag
# says nobody is sharing while the buffer still offers the screen as current.
CALL_SCOPED_MEET_SURFACES: dict[str, tuple[str, ...]] = {
    "user_screen_share_active": ("user",),
    "user_webcam_active": ("webcam",),
    "meet_screen_share_active": ("google_meet", "teams_meet"),
}
# The assistant's own desktop is not call-scoped: the Console's Desktop tab
# opens it with no call in sight and reports its own close on unmount. Clearing
# these at a call boundary would tell the assistant nobody is watching while
# that pane is still open, and would hand back control the user still holds.
DESKTOP_SCOPED_MEET_SURFACES = (
    "assistant_screen_share_active",
    "user_remote_control_active",
)
# Viewer-source namespace for someone watching the assistant's desktop from
# inside a call, as ``call:<call_id>``. The Console composes the full source; the
# namespace alone is what a call boundary closes. Kept in step with
# ``console/src/lib/assistants/desktopViewer.ts``.
CALL_VIEWER_SOURCE = "call"
# Stands in for the whole room in a call viewer's key, in place of the person who
# happened to press the button. A call's stage is one switch, not a tally: it puts
# the desktop in front of *everyone* on the call, so anybody there may turn it on
# and anybody there may turn it off again. Keyed per person instead, a second
# participant's "stop" would discard a key they never held and the desktop would
# stay up with the button doing nothing.
#
# The standalone Desktop tab keeps per-person keys, where the tally is the point:
# two people with the pane open are two viewers, and one closing it must not take
# it from the other.
CALL_VIEWER_IDENTITY = "room"

COMMISSIONING_MUTATION_TOOL_NAMES = frozenset(
    {
        "act",
    },
)
COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS = frozenset(
    {
        "SMSSent",
        "WhatsAppMessageSent",
        "EmailSent",
        "UnifyMessageSent",
        "ApiMessageSent",
        "DiscordMessageSent",
        "DiscordChannelMessageSent",
        "TeamsMessageSent",
        "TeamsChannelMessageSent",
    },
)


def _idle_status_smalltalk_allowed(
    *,
    in_flight_actions: dict[int, dict],
    global_thread: Reversible[CommsMessage],
    inflight_voice_speech: str,
    now: datetime,
    recent_comms_seconds: float = IDLE_SMALLTALK_RECENT_COMMS_SECONDS,
) -> bool:
    if in_flight_actions:
        return False
    if inflight_voice_speech.strip():
        return False
    for message in reversed(global_thread):
        if not isinstance(message, CommsMessage):
            continue
        if getattr(message, "role", None) != "assistant":
            continue
        age_seconds = (now - message.timestamp).total_seconds()
        return age_seconds >= recent_comms_seconds
    return True


ACT_FOLLOWUP_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "response_format": None,
    "persist": False,
    "include_conversation_context": True,
}


def _log_slow_brain_single_shot_failure(
    *,
    run_id: str,
    request_id: str,
    origin_event_name: str,
    message_count: int,
    tool_count: int,
    state_chars: int,
) -> None:
    if not staging_diagnostics_enabled():
        return
    LOGGER.exception(
        (
            "Slow-brain single-shot failed "
            "run_id=%s request_id=%s origin_event=%s "
            "message_count=%d tool_count=%d state_chars=%d"
        ),
        run_id,
        request_id or "-",
        origin_event_name or "-",
        message_count,
        tool_count,
        state_chars,
    )
    LOGGER.error(
        "Slow-brain single-shot traceback text:\n%s",
        traceback.format_exc(),
    )


def _format_tool_thoughts_for_log(tools: list[ToolExecution]) -> str:
    parts: list[str] = []
    for tool_exec in tools:
        thoughts = getattr(tool_exec, "thoughts", None)
        if isinstance(thoughts, str) and thoughts.strip():
            parts.append(f"[{tool_exec.name}] {thoughts.strip()}")
    return " | ".join(parts)


def _append_context_to_state_message(message: dict, context: str) -> dict:
    if not context:
        return message
    updated = dict(message)
    content = updated.get("content")
    if isinstance(content, str):
        updated["content"] = f"{content}\n\n{context}"
        return updated
    if isinstance(content, list):
        updated["content"] = [
            *content,
            {"type": "text", "text": f"\n\n{context}"},
        ]
        return updated
    updated["content"] = f"{content or ''}\n\n{context}"
    return updated


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
        assistant_slack_bot_user_id: str = "",
        assistant_slack_team_id: str = "",
        assistant_has_ms_teams_bot: bool = False,
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
        self.assistant_slack_bot_user_id = assistant_slack_bot_user_id
        self.assistant_slack_team_id = assistant_slack_team_id
        # True when the assistant's org has a bound Unify Microsoft Teams bot
        # install (Bot Framework channel, distinct from the delegated-Graph
        # ``assistant_has_teams`` MS365 mailbox capability). Sourced from the
        # assistant profile and adopted at runtime from inbound bot activities.
        self.assistant_has_ms_teams_bot = assistant_has_ms_teams_bot
        # Global onboarding scaffolding gate, mirrored from Orchestra's
        # ``Coordinator/State`` and refreshed on a short TTL. When False the
        # slow-brain drops all onboarding scaffolding. Defaults to True until
        # the first refresh resolves.
        self.coordinator_onboarding_active: bool = True
        # Precomputed depends_on-aware onboarding picture (steps + statuses
        # + valid next targets with nudge copy), mirrored from Orchestra so
        # the slow brain reads a standing progress block instead of
        # deriving "what's next". None outside active onboarding.
        self.coordinator_onboarding_render: dict[str, Any] | None = None
        self.coordinator_intro_watched: bool = False
        self.coordinator_pending_chat_intro: bool = False
        self.coordinator_chat_intro_armed_at: str | None = None
        self._coordinator_chat_intro_delivery_task: asyncio.Task[None] | None = None
        # Trigger-step ids the user clicked in THIS session (ephemeral by
        # design): unlocks the matching reference-quiz comms tool until the
        # send durably completes the step. Lost on restart on purpose — the
        # row stays re-clickable, so a tool can never be permanently masked.
        self._onboarding_clicked_trigger_steps: set[str] = set()
        # Armed only inside the learning-demo chip-click -> step-complete
        # window (learning_beat_requested .. learn-from-correction
        # step_completed / session restart). Gates the StorageCheck-completion
        # wake in event_handlers.py's ActorNotification handler so storage
        # completing after every act does not wake the brain product-wide.
        self._learning_demo_storage_wake_armed: bool = False
        # Console orientation text and when the Console last reported the user
        # present, both set from AssistantPresenceObserved. See
        # ``console_guidance`` for why they are held together.
        self._console_guidance: dict[str, str] = {}
        self._console_guidance_version: str = ""
        self._console_presence_at: float | None = None
        self._coordinator_state_checked_at: float = 0.0
        # Shared keep-alive HTTP client for Orchestra state reads/writes, plus
        # the in-flight background refresh (see
        # _schedule_coordinator_onboarding_state_refresh). Closed in cleanup().
        self._coordinator_state_http: Any | None = None
        self._coordinator_state_refresh_task: asyncio.Task | None = None
        self.assistant_email_provider = assistant_email_provider
        self.user_first_name = user_first_name
        self.user_surname = user_surname
        self.user_number = user_number
        self.user_email = user_email

        # initialization state
        self.initialized: bool = False
        # Open ⇒ slow-brain turns may render. ``init_conv_manager`` closes it
        # for the window between boot and global-thread hydration; hydration
        # completion (restored, empty, or failed) reopens it. Steady-state
        # turns therefore never wait on it. See ``_run_llm`` for why a turn
        # must not render from a pre-hydration view.
        self._hydration_gate: asyncio.Event = asyncio.Event()
        self._hydration_gate.set()
        self.ready_for_brain: bool = True
        self.vm_ready: bool = False
        self.file_sync_complete: bool = False
        self.deployment_runtime_reconcile_status: Any | None = None
        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name

        # inactivity & shutdown
        self.inactivity_timeout = 600  # 10 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self._last_activity_source = "startup"
        self.shutdown_reason: str | None = None
        # Set when the process has established it cannot serve this assistant
        # at all -- currently only a failed manager init, which leaves no
        # actor and no managers for the rest of the pod's life. Liveness is
        # otherwise a question about traffic, and a pod being *talked to* says
        # nothing about whether it can answer: one that could not held a live
        # assistant for three hours, failing every scheduled task and every
        # message, while Console presence kept resetting its idle clock.
        self.unserviceable_reason: str | None = None
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
        self.call_manager.set_config_provider(self.get_call_config)
        self.call_manager.on_screenshot = self._buffer_screenshot
        self.call_manager.on_fast_brain_generating = self._on_fast_brain_generating
        self.call_manager.on_pipeline_quiescent = self._on_pipeline_quiescent
        self.call_manager.voice_profile_provider = self._get_voice_profiles

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a property with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        self.mode: Mode = Mode.TEXT
        self.chat_history = []
        # Line this turn just decided to speak, not yet confirmed spoken. Injected
        # render-only into the next run's prompt (as a transient [You] row) so it
        # is not repeated; cleared once the real Outbound utterance lands. Never
        # written to the stored transcript.
        self._inflight_voice_speech: str = ""
        # Deferred session teardown: the hang_up tool records intent here rather
        # than ending the call immediately, so _run_llm can wait for the turn's
        # explanatory line to be spoken before tearing down (no mid-sentence cut).
        self._pending_hang_up: bool = False
        self._pending_hang_up_teardown: Callable | None = None
        # Set when an outbound voice utterance matching the just-published spoken
        # guidance lands (full line, or a barge-in truncated prefix). Used to gate
        # the deferred hang-up on speech actually being delivered.
        self._inflight_speech_delivered: asyncio.Event = asyncio.Event()
        # Call-session id of an in-flight Unify Meet ring awaiting an answer.
        # Cleared when the owner answers (UnifyMeetReceived) or the no-answer
        # timeout fires and falls the conversation back to text.
        self._pending_meet_ring: str | None = None
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.integration_sync_coordinator = IntegrationSyncCoordinator()
        self.in_flight_actions: dict[
            int,
            dict,
        ] = (
            {}
        )  # dict[int, {"handle": SteerableTool, "query": str, "calling_id": str|None, ...}]
        self.completed_actions: dict[
            int,
            dict,
        ] = {}  # Finished actions, kept for post-completion ask() queries
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

        # meet interaction state (screen share / webcam / remote control).
        # Each flag below is classified as call- or desktop-scoped by the
        # ``*_MEET_SURFACES`` tuples above, which decide what a call boundary
        # closes.
        self.assistant_screen_share_active: bool = False
        # What currently has the assistant's desktop open, as
        # ``"<source>:<identity>"`` -- one key per Desktop tab, and one per call
        # for the whole room. ``assistant_screen_share_active`` above is this
        # set's emptiness, cached so the ~15 readers of the flag (and the generic
        # ``_MEET_STATE_FLAGS`` setattr path) keep working unchanged.
        # Membership, not the last event, decides: a call and a Desktop tab watch
        # the same desktop at once, and either one closing must not tell the
        # assistant that nobody is looking.
        self._assistant_screen_share_viewers: set[str] = set()
        self.user_screen_share_active: bool = False
        # Someone in a browser meeting is sharing a screen with us. Kept apart
        # from ``user_screen_share_active`` because that one also gates the
        # desktop fast path and web-session listing, which should not switch on
        # because a stranger in a Google Meet put up a slide.
        self.meet_screen_share_active: bool = False
        self.user_webcam_active: bool = False
        self.user_remote_control_active: bool = False
        # Flag names above that a frontend has reported on. A frontend is
        # authoritative for its surfaces, so track-inferred events stop being
        # applied to them once it speaks.
        self._frontend_reported_meet_surfaces: set[str] = set()

        # screenshot buffer for slow brain visual context
        self._screenshot_buffer: list[ScreenshotEntry] = []
        # mapping from local_message_id (ephemeral CM counter) to
        # global message_id (persistent backend TM id), populated by
        # log_message() for post-hoc screenshot image updates.
        self._local_to_global_message_ids: dict[int, int] = {}
        # Per-destination transcript message ids for fanout writes.
        self._local_to_global_message_ids_by_destination: dict[
            int,
            dict[str | None, int],
        ] = {}
        # Primary destination used when one id is needed for compatibility paths.
        self._local_message_destinations: dict[int, str | None] = {}

        # mapping from conference_name/room_name to the exchange's
        # (id, destination), populated at call/meet end so the async
        # RecordingReady handler can resolve the exchange without a database
        # filter query. The destination travels with the id because exchange
        # ids are root-local.
        self._recording_exchange_ids: dict[str, tuple[int, str | None]] = {}

        # Detached recording-start requests. Recording must never gate call
        # setup, so the call-started handler fires the request without awaiting
        # it; the set holds a strong reference so the task is not garbage
        # collected mid-flight.
        self._recording_start_tasks: set[asyncio.Task] = set()

        # Groups messages into conversation-thread exchanges (SMS / WhatsApp /
        # Discord / MS Teams bot / Slack DMs and channels, and email). Maps a
        # per-conversation key to its exchange_id. 1:1 DMs reuse a single
        # exchange per contact and provider-thread channels (group channels,
        # email) reuse for the whole thread — both with no inactivity window.
        # In-memory only; on a cold cache, durable mediums (DMs and email)
        # recover their exchange from Exchanges metadata so they survive a
        # restart.
        self._conversation_exchange_ids: dict[str, int] = {}

        # proactive speech
        self.proactive_speech = ProactiveSpeech()
        self._proactive_speech_task: asyncio.Task | None = None
        self._proactive_speech_gen: int = 0
        self._proactive_speech_enabled: bool = True
        self._voice_pipeline_quiescent = asyncio.Event()
        self._voice_pipeline_quiescent.set()
        self._proactive_logger = FastBrainLogger("ProactiveSpeech")

        # ask handles (for Actor actions)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool, bool]] = []
        self._pending_llm_request_meta: list[dict[str, Any]] = []
        self._current_event_trace: dict[str, str] | None = None
        self._event_trace_seq: int = 0
        self._llm_request_seq: int = 0
        self._llm_run_seq: int = 0
        self._llm_gen: int = 0
        self._active_llm_trace_meta: dict[str, Any] | None = None
        self._credit_gate_reply_sent_at: dict[tuple[str, str], float] = {}
        self._last_inbound_reply_context: dict[str, Any] | None = None
        self._recent_tool_executions: list[dict[str, Any]] = []
        self._recent_commissioning_successes: dict[str, int] = {}

        # WhatsApp messages that were sent via greeting template (outside 24h
        # window). When the contact replies, the brain is notified so it can
        # resend or rework the original message.  Maps contact_id → content.
        self._pending_whatsapp_resends: dict[int, str] = {}
        self._pending_whatsapp_resend_onboarding_metadata: dict[int, dict[str, str]] = (
            {}
        )

        # Best-effort estimate of whether each contact's 24-hour WhatsApp
        # free-form window is currently open, so the brain's send_whatsapp
        # docstring can warn it up front when an out-of-window send will only
        # deliver a generic template placeholder (not the verbatim body).
        # Maps contact_id → bool (absent = unknown). Seeded best-effort at
        # startup (Orchestra owns the authoritative window) and refreshed from
        # observed traffic: an inbound opens it, a templated outbound proves it
        # was closed, a free-form outbound proves it was open.
        self._whatsapp_window_open: dict[int, bool] = {}

        # Outbound WhatsApp call openers stashed while awaiting call permission.
        # When the contact grants permission (taps "Call now"), the stash is
        # injected as call_manager.pending_opener / pending_briefing. Maps
        # contact_id → {"opener": str, "briefing": str}.
        self._pending_whatsapp_call_openers: dict[int, dict[str, str]] = {}
        self._pending_onboarding_outbound: dict[str, Any] | None = None
        self._startup_wake_reasons: list[dict[str, Any]] = []

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.debug(
            "session_start",
            "ConversationManager session initialized",
        )

    def fast_brain_idle_smalltalk_allowed(self) -> bool:
        return _idle_status_smalltalk_allowed(
            in_flight_actions=self.in_flight_actions,
            global_thread=self.contact_index.global_thread,
            inflight_voice_speech=self._inflight_voice_speech,
            now=prompt_now(as_string=False),
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

    @staticmethod
    def _tool_result_is_error(result: Any) -> bool:
        return isinstance(result, dict) and "error_kind" in result

    @staticmethod
    def _preview_value(
        value: Any,
        *,
        max_chars: int = RECENT_TOOL_PREVIEW_CHARS,
    ) -> str:
        try:
            rendered = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            rendered = repr(value)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    @staticmethod
    def _normalize_followup_tool_args(
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(tool_args or {})
        if tool_name == "act":
            for key, default_value in ACT_FOLLOWUP_ARGUMENT_DEFAULTS.items():
                normalized.setdefault(key, default_value)
        return normalized

    @classmethod
    def _commissioning_tool_fingerprint(
        cls,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> str:
        stable_args = json.dumps(
            cls._normalize_followup_tool_args(tool_name, tool_args),
            sort_keys=True,
            default=str,
        )
        return f"{tool_name}:{stable_args}"

    def _is_immediate_commissioning_followup(self, origin_event_name: str) -> bool:
        return origin_event_name in COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS

    def suppress_duplicate_commissioning_tool(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Suppress immediate duplicate commissioning calls from outbound follow-ups."""
        if tool_name not in COMMISSIONING_MUTATION_TOOL_NAMES:
            return None
        trace_meta = self._active_llm_trace_meta or {}
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        if not self._is_immediate_commissioning_followup(origin_event_name):
            return None
        fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
        last_success_gen = self._recent_commissioning_successes.get(fingerprint)
        if last_success_gen != self._llm_gen - 1:
            return None
        return {
            "error_kind": "duplicate_suppressed",
            "message": (
                "Skipped duplicate commissioning tool call from immediate outbound "
                "follow-up event."
            ),
            "details": {
                "tool_name": tool_name,
                "origin_event_name": origin_event_name,
            },
        }

    def _record_recent_tool_executions(
        self,
        *,
        tools: list[Any],
        trace_meta: dict[str, Any],
    ) -> None:
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        for tool_exec in tools:
            tool_name = str(getattr(tool_exec, "name", ""))
            tool_args = getattr(tool_exec, "args", {}) or {}
            tool_result = getattr(tool_exec, "result", None)
            self._recent_tool_executions.append(
                {
                    "generation": self._llm_gen,
                    "origin_event_name": origin_event_name,
                    "tool_name": tool_name,
                    "args_preview": self._preview_value(tool_args),
                    "result_preview": self._preview_value(tool_result),
                },
            )
            if (
                tool_name in COMMISSIONING_MUTATION_TOOL_NAMES
                and not self._tool_result_is_error(tool_result)
            ):
                fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
                self._recent_commissioning_successes[fingerprint] = self._llm_gen
        if len(self._recent_tool_executions) > RECENT_TOOL_EXECUTIONS_LIMIT:
            self._recent_tool_executions = self._recent_tool_executions[
                -RECENT_TOOL_EXECUTIONS_LIMIT:
            ]
        for fingerprint, generation in list(
            self._recent_commissioning_successes.items(),
        ):
            if generation < self._llm_gen - 1:
                del self._recent_commissioning_successes[fingerprint]

    @property
    def is_coordinator(self) -> bool:
        """Whether this session is the workspace Coordinator.

        Delegates to ``SESSION_DETAILS`` so there is a single source of truth
        for the Coordinator role. Tool registration reads
        ``SESSION_DETAILS.is_coordinator``; if this returned a separately-stored
        attribute the two could disagree — offering a Coordinator-only tool that
        the runtime guards then reject.
        """
        return SESSION_DETAILS.is_coordinator

    @is_coordinator.setter
    def is_coordinator(self, value: bool) -> None:
        SESSION_DETAILS.is_coordinator = bool(value)

    @property
    def assistant_has_teams(self) -> bool:
        """True when Microsoft Teams capabilities are available to this assistant.

        Derived from the email provider rather than a dedicated flag because
        Teams access is gated by the same MS365 OAuth grant that backs the
        assistant's email — provisioning the Teams scopes without an MS365
        mailbox is not a supported configuration. Update both `assistant_has_teams`
        and `_assistant_has_teams` in `unify.comms.primitives` together if a
        first-class Teams flag is ever introduced.
        """
        return self.assistant_email_provider == "microsoft_365"

    @property
    def in_voice_session(self) -> bool:
        """True when a voice call or meeting of any kind is live (or joining).

        A single predicate spanning every voice surface — phone calls, WhatsApp
        calls, Unify Meet, and browser meetings (Google Meet / Microsoft Teams).
        Only one such session can exist at a time, so the call-starting tools are
        withheld whenever this is True. This is the single source of truth shared
        by the tool set (`as_tools`) and the system prompt so the two can never
        disagree on what is available mid-call.
        """
        call_state = self.call_manager
        return (
            self.mode.is_voice
            or call_state.has_active_call
            or call_state.has_active_google_meet
            or call_state.has_active_teams_meet
            or call_state._whatsapp_call_joining
        )

    @property
    def session_logger(self) -> SessionLogger:
        """The hierarchical session logger for this ConversationManager instance."""
        return self._session_logger

    @property
    def computer_primitives(self):
        """Lazily resolve the ``ComputerPrimitives`` singleton via ManagerRegistry."""
        from unify.function_manager.primitives.runtime import ComputerPrimitives
        from unify.manager_registry import ManagerRegistry

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

    @staticmethod
    def assistant_screen_share_viewer_key(user_id: str, source: str) -> str:
        """Identify one viewer of the assistant's desktop.

        A call collapses to one key for the whole room and ``user_id`` is ignored
        -- see ``CALL_VIEWER_IDENTITY`` for why the person who pressed the button
        is not the identity that matters there.

        A client that predates viewer tracking sends neither field; it collapses
        to a single legacy key so its start/stop pair still opens and closes the
        surface. Without that, an un-upgraded Console would add a viewer it could
        never remove and pin the desktop open for the rest of the session.
        """
        if source.startswith(f"{CALL_VIEWER_SOURCE}:"):
            return f"{source}:{CALL_VIEWER_IDENTITY}"
        return f"{source or 'legacy'}:{user_id or 'legacy'}"

    def note_assistant_screen_share_viewer(
        self,
        *,
        user_id: str,
        source: str,
        watching: bool,
    ) -> bool:
        """Add or remove one viewer; return whether anything is watching now.

        ``user_id`` names the viewer for a Desktop tab and is ignored for a call,
        where the key stands for the room rather than the person acting -- so any
        participant's stop closes the share any other participant opened.
        """
        key = self.assistant_screen_share_viewer_key(user_id, source)
        if watching:
            self._assistant_screen_share_viewers.add(key)
        else:
            self._assistant_screen_share_viewers.discard(key)
        return bool(self._assistant_screen_share_viewers)

    def assistant_desktop_watched_from_call(self, call_session_id: str = "") -> bool:
        """True when someone is watching the assistant's desktop *from a call*.

        Narrower than ``assistant_screen_share_active``, deliberately. That flag
        is desktop-scoped: a standalone Desktop tab holds it true across call
        boundaries, which is right for everything that reads it -- prompt state,
        screenshot capture, the computer fast path -- because somebody really is
        looking at the desktop.

        It is the wrong question for the room. Mounting the desktop on a call's
        stage shows it to *every* participant, so one person's Desktop tab must
        not decide it, and neither must a viewer left behind by an earlier call.
        Only viewers under ``CALL_VIEWER_SOURCE`` count here, and passing
        ``call_session_id`` narrows that to the call doing the asking.
        """
        prefix = (
            f"{CALL_VIEWER_SOURCE}:{call_session_id}:"
            if call_session_id
            else f"{CALL_VIEWER_SOURCE}:"
        )
        return any(
            key.startswith(prefix) for key in self._assistant_screen_share_viewers
        )

    def drop_stale_call_screen_share_viewers(self, call_session_id: str) -> None:
        """Drop call viewers belonging to any call other than this one.

        A new call cleaning up after its predecessors, because nothing else
        reliably does. A viewer can only be closed by a stop event naming the
        call it came from, and the call it came from is gone. Both resets that
        would otherwise catch it can be skipped in the same sequence: the
        call-start one when a dispatch arrives while the previous session is
        still winding down, and the call-end one when the stale-session guard
        drops a departed call's ``Ended`` event -- correctly, since a dying call
        must not clobber a live one, but it takes that call's cleanup with it.
        Left alone the viewer is immortal for the life of the pod, holding the
        desktop open on every later call with no way to take it down.

        Keyed on the call id rather than the ``call:`` namespace, so viewers this
        call has already registered survive: the Console can start a share before
        the runtime's own call-started event lands.
        """
        if not call_session_id:
            return
        keep_prefix = f"{CALL_VIEWER_SOURCE}:{call_session_id}:"
        drop_prefix = f"{CALL_VIEWER_SOURCE}:"
        self._assistant_screen_share_viewers = {
            key
            for key in self._assistant_screen_share_viewers
            if key.startswith(keep_prefix) or not key.startswith(drop_prefix)
        }
        self.assistant_screen_share_active = bool(
            self._assistant_screen_share_viewers,
        )

    def drop_assistant_screen_share_viewers(self, source: str) -> bool:
        """Drop every viewer watching through ``source``; return who is left.

        The reason the viewer set exists: a call ending closes the viewers it
        owns and leaves a standalone Desktop tab watching, which is why
        ``assistant_screen_share_active`` cannot simply be cleared at a call
        boundary -- see ``DESKTOP_SCOPED_MEET_SURFACES``.

        Sources are namespaced, so a namespace drops everything under it:
        ``CALL_VIEWER_SOURCE`` closes every call viewer without needing to know
        which call, which is what a call boundary wants -- a pod runs one call at
        a time, so no call viewer outlives one.
        """
        prefix = f"{source}:"
        self._assistant_screen_share_viewers = {
            key
            for key in self._assistant_screen_share_viewers
            if not key.startswith(prefix)
        }
        self.assistant_screen_share_active = bool(
            self._assistant_screen_share_viewers,
        )
        return self.assistant_screen_share_active

    def reset_meet_surfaces(self) -> None:
        """Close the call-scoped meet surfaces at a call boundary.

        These flags describe surfaces shared during one call, but they live on
        the CM, which outlives every call in the pod. Left alone they leak
        forward: a share still marked active after hangup, or a webcam whose
        "stopped" event never arrived because the Console unmounted first.

        The frontend-ownership entries go with them, and that is the expensive
        half. Ownership records which surfaces a frontend has spoken for, so
        that track-inferred events stop overriding it — right within a call,
        wrong across them. A 1:1 Console call claims both user surfaces; an org
        call has no frontend reporting on either, so a stale claim leaves the
        assistant capturing frames from a share it never registers as started.

        Each surface's unpaired frames go too, for the reason
        ``MeetScreenShareStopped`` already drops its own: an unpaired frame means
        "what this source shows now", so once the source is gone it shows
        nothing. Frames paired with an utterance stay — those are evidence for
        something somebody said, and remain true after the screen goes.

        The assistant's own desktop is not in ``CALL_SCOPED_MEET_SURFACES`` and
        is not blanket-cleared — see ``DESKTOP_SCOPED_MEET_SURFACES``. Its
        *call* viewers are still closed here, which is the distinction the viewer
        set exists to make: the people who were watching from the call stop
        watching when the call goes, and a Desktop tab open beside it does not.
        """

        for surface, screenshot_sources in CALL_SCOPED_MEET_SURFACES.items():
            setattr(self, surface, False)
            self._frontend_reported_meet_surfaces.discard(surface)
            for source in screenshot_sources:
                self.drop_unpaired_screenshots(source)

        still_watched = self.drop_assistant_screen_share_viewers(CALL_VIEWER_SOURCE)
        if not still_watched:
            # Nobody can be driving a desktop nobody is looking at, and a
            # remote-control flag left set makes the assistant refuse to act for
            # the rest of the session while it waits to be handed back.
            self.user_remote_control_active = False
            self._frontend_reported_meet_surfaces.discard(
                "assistant_screen_share_active",
            )
            self._frontend_reported_meet_surfaces.discard(
                "user_remote_control_active",
            )
            self.drop_unpaired_screenshots("assistant")

    def get_active_contact(self) -> dict | None:
        """Get the contact for the current active call, or fall back to the boss contact."""
        return self.call_manager.call_contact or self.contact_index.get_contact(
            contact_id=SESSION_DETAILS.boss_contact_id,
        )

    def record_console_presence(
        self,
        *,
        version: str = "",
        brief: str = "",
        full: str = "",
        actions: str = "",
    ) -> None:
        """Note that Console reported the user present, and keep any text it sent.

        Heartbeats that carry no text still refresh presence: most of them are
        version-only by design, and treating those as absence would blink the
        prompt section in and out between the ones that do carry it.
        """
        self._console_presence_at = time.monotonic()
        if not brief and not full:
            return
        if version and version != self._console_guidance_version:
            self._session_logger.info(
                "console_guidance",
                f"Console guidance updated to version {version}.",
            )
        self._console_guidance_version = version
        self._console_guidance = {"brief": brief, "full": full, "actions": actions}

    def console_is_open(self) -> bool:
        """Whether Console reported the user present recently enough to count."""
        if self._console_presence_at is None:
            return False
        return time.monotonic() - self._console_presence_at <= CONSOLE_PRESENCE_TTL_S

    def console_action_catalogue(self) -> str:
        """The most recently published navigation targets, or ``""`` if none.

        Not blanked by the presence window (see ``console_guidance`` for the
        cache rationale) — but unlike the guidance text, Console may
        legitimately publish an empty catalogue (navigation disallowed for
        this teammate), so empty means "no drivable targets", not "console
        never seen". The catalogue feeds the state snapshot's console pane;
        whether a move would land right now is ``console_is_open()``'s
        question, answered at ``show_in_console`` call time.
        """
        return self._console_guidance.get("actions", "")

    def console_guidance(self, detail: str = "brief") -> str:
        """Console orientation text, kept for the session once it arrives.

        The provider prompt cache is all-or-nothing over system+tools, so
        letting the presence window blank this section would wipe a warm cache
        every time the boss stepped away from the Console and again on their
        return. The text therefore stays put and changes only when the Console
        publishes a new content-hash version — at most one extra miss per
        session. Whether the Console is open right now is the state snapshot's
        job (its console pane), not this text's.
        """
        return self._console_guidance.get(detail, "")

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

        from unify.conversation_manager.medium_scripts.common import (
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

    def drop_unpaired_screenshots(self, source: str) -> int:
        """Forget buffered frames from *source* that no utterance depends on.

        Called when a visual source goes away. An unpaired frame is "what this
        source shows now", so once the source is gone it shows nothing and holding
        the last one lets the next turn describe a screen that has been taken
        down. Frames paired with an utterance are kept: those are evidence for a
        specific thing somebody said, and remain true after the screen goes.
        """
        before = len(self._screenshot_buffer)
        self._screenshot_buffer = [
            entry
            for entry in self._screenshot_buffer
            if entry.source != source or entry.utterance
        ]
        return before - len(self._screenshot_buffer)

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
        image_ids_by_destination: dict[str | None, list[int]] = {}
        implicit_destinations: list[str | None] = [None]
        try:
            from unify.manager_registry import ManagerRegistry
            from unify.common.context_registry import ContextRegistry

            image_manager = ManagerRegistry.get_image_manager()
            items = [
                {
                    "data": entry.b64,
                    "timestamp": entry.timestamp,
                    "filepath": path,
                }
                for entry, path in zip(screenshots, screenshot_paths)
            ]
            implicit_destinations = ContextRegistry.implicit_shared_destinations()
            for destination in implicit_destinations:
                destination_image_ids = await asyncio.to_thread(
                    image_manager.add_images,
                    items,
                    synchronous=True,
                    destination=destination,
                )
                image_ids_by_destination[destination] = destination_image_ids
            primary_destination = implicit_destinations[0]
            image_ids = image_ids_by_destination.get(primary_destination, [])
        except Exception as e:
            self._session_logger.warning(
                "screenshot_registration",
                f"ImageManager registration failed, skipping: {e}",
            )
            return

        # 2. Annotate CM Message objects with image_ids and build TM refs.
        msg_to_image_refs: dict[tuple[int, str | None], list[dict]] = {}
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
            for destination, destination_image_ids in image_ids_by_destination.items():
                if i >= len(destination_image_ids):
                    continue
                msg_to_image_refs.setdefault((mid, destination), []).append(
                    {
                        "raw_image_ref": {"image_id": destination_image_ids[i]},
                        "annotation": f"{label} -- '{entry.utterance}'",
                    },
                )

        # 3. Post-hoc update TM messages with AnnotatedImageRefs.
        if msg_to_image_refs and self.transcript_manager is not None:
            for (local_mid, destination), refs in msg_to_image_refs.items():
                destination_map = self._local_to_global_message_ids_by_destination.get(
                    local_mid,
                    {},
                )
                tm_msg_id = destination_map.get(destination)
                effective_destination = destination
                if tm_msg_id is None:
                    if destination is not None:
                        self._session_logger.warning(
                            "screenshot_tm_update",
                            (
                                "Skipping screenshot transcript update for "
                                f"local_mid={local_mid}, destination={destination!r}: "
                                "destination message mapping missing."
                            ),
                        )
                        continue
                    tm_msg_id = self._local_to_global_message_ids.get(local_mid)
                    effective_destination = self._local_message_destinations.get(
                        local_mid,
                        destination,
                    )
                if tm_msg_id is not None:
                    try:
                        await asyncio.to_thread(
                            self.transcript_manager.update_message_images,
                            tm_msg_id,
                            refs,
                            destination=effective_destination,
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
                entry = ScreenshotEntry(
                    b64,
                    utterance,
                    ts,
                    source,
                    filepath=filepath,
                    attribution=data.get("attribution"),
                )
                # An unpaired frame is "what this source shows now", so only the
                # newest is worth holding: a shared screen left up while nobody
                # speaks arrives every few seconds, and appending each one would
                # put a reel of near-identical images in one state message. Frames
                # paired with an utterance always append -- those are evidence for
                # a specific thing somebody said.
                previous = (
                    self._screenshot_buffer[-1] if self._screenshot_buffer else None
                )
                if (
                    not utterance
                    and previous is not None
                    and previous.source == source
                    and not previous.utterance
                ):
                    self._screenshot_buffer[-1] = entry
                else:
                    self._screenshot_buffer.append(entry)
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

    def _active_voice_medium(self) -> Medium:
        """The Medium for the currently-active voice thread."""
        if self.call_manager.has_active_google_meet:
            return Medium.GOOGLE_MEET
        if self.call_manager.has_active_teams_meet:
            return Medium.TEAMS_MEET
        if self.mode == Mode.MEET:
            return Medium.UNIFY_MEET
        if self.call_manager._call_channel == "whatsapp_call":
            return Medium.WHATSAPP_CALL
        return Medium.PHONE_CALL

    def _stash_inflight_voice_speech(self, message: str) -> None:
        """Stash the line this turn just decided to speak, for a render-only overlay.

        This is the slow brain's in-flight-speech overlay. The next run may start
        before the real spoken ``[You]`` utterance is recorded; without seeing
        this line it would re-derive "was that actually spoken?" and repeat it.
        So we stash it here and inject it into the NEXT render as a transient
        ``[You @ ...]`` row (see ``_run_llm``) - indistinguishable from confirmed
        speech for that one call, so the model treats it as already said.

        Crucially this is NEVER written to the stored transcript: it is a
        one-shot, render-only mutation. Once the real utterance lands (the
        ``Outbound*Utterance`` event), this stash is cleared so future turns see
        only what was *actually* spoken (e.g. the truncated prefix after a
        barge-in, with the ``VoiceInterrupt`` note carrying the remainder).
        """
        self._inflight_voice_speech = (message or "").strip()

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

        voice_medium = self._active_voice_medium()
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

    async def cancel_slow_brain_run(self, turn_id) -> None:
        """Cancel exactly the slow-brain run spawned by ``turn_id``.

        Used when a voice turn must be dropped before it produces speech (e.g.
        superseded by a newer user utterance). Targets only that turn's run
        wherever it sits in the queue (no-op if already gone). A run already in
        tool commit (speaking) is spared.
        """
        await self.debouncer.cancel_run_by_turn(turn_id)

    async def handle_voice_user_turn(
        self,
        content: str,
        triggering_contact_id: int | None = None,
        turn_id: int | None = None,
    ):
        """Route a completed voice user turn to the active ask handle or slow brain."""
        if self.active_ask_handle and not self.active_ask_handle.done():
            await self.active_ask_handle.interject(content)
            return

        await self.request_llm_run(
            delay=0,
            triggering_contact_id=triggering_contact_id,
            is_user_origin=True,
            turn_id=turn_id,
        )

    async def run_llm(
        self,
        delay: float = 0,
        trace_meta: dict[str, str] | None = None,
    ):
        await self.debouncer.submit(
            self._run_llm_with_failure_notification,
            kwargs={"trace_meta": trace_meta or {}},
            delay=delay,
            label=(trace_meta or {}).get("origin_event_name", ""),
            trace_meta=trace_meta,
        )

    # Grace window for the owner to answer a Unify Meet ring before the
    # conversation falls back to text.
    _MEET_RING_TIMEOUT_S = 25.0

    async def ring_unify_meet(
        self,
        opener: str,
        briefing: str | None = None,
        allow_hang_up: str | None = None,
    ) -> dict:
        """Ring the owner on Unify Meet and await an answer (no-answer -> text).

        Creates a ringing ``assistant_dm`` call session in Orchestra, which
        publishes the incoming-call frame the Console rings on. The assistant
        cannot join the owner's browser for them; when they answer, Orchestra
        dispatches this runtime into the room (landing here as
        ``UnifyMeetReceived``). ``opener`` is spoken verbatim once the call
        connects; ``briefing`` is unspoken context the voice agent uses to run
        the call's task itself; ``allow_hang_up`` pre-arms the hang-up gate for
        calls expected to be short. If unanswered within
        ``_MEET_RING_TIMEOUT_S``, the session is ended and a notification tells
        the brain to continue over text.
        """
        from unify.conversation_manager.domains import comms_utils

        reason = (opener or "").strip()
        if not reason:
            return {
                "status": "error",
                "message": (
                    "opener is required: provide the exact words to speak when "
                    "the call connects. No ring was sent."
                ),
            }
        # The opener rides the call session's opening config, which Orchestra
        # replays on the answer-triggered dispatch; the briefing and pre-armed
        # gate stay queued here and are reattached by start_unify_meet.
        self.call_manager.pending_briefing = (briefing or "").strip()
        self.call_manager.pending_hang_up_gate = " ".join(
            (allow_hang_up or "").split(),
        ).strip()
        result = await asyncio.to_thread(
            comms_utils.create_assistant_call,
            opening_config={
                "mode": "opener",
                "opener_text": reason,
                "source": "unify_meet_ring",
            },
        )
        call_session_id = result.get("call_id") or ""
        if not result.get("success") or not call_session_id:
            return {
                "status": "error",
                "message": "Could not ring the Unify Meet right now.",
            }
        self._pending_meet_ring = call_session_id
        asyncio.ensure_future(self._await_meet_ring_answer(call_session_id))
        return {
            "status": "ok",
            "message": (
                "Ringing my boss on Unify Meet — a pinged call window with an "
                "Answer button is now showing for them. I'll join when they answer."
            ),
        }

    async def _await_meet_ring_answer(self, call_session_id: str) -> None:
        """Fall back to text if a Unify Meet ring goes unanswered."""
        await asyncio.sleep(self._MEET_RING_TIMEOUT_S)
        if self._pending_meet_ring != call_session_id:
            return  # answered (or superseded) - nothing to do
        from unify.common.prompt_helpers import now as prompt_now
        from unify.conversation_manager.domains import comms_utils

        self._pending_meet_ring = None
        # End the ringing session so every Console surface clears the ring.
        await asyncio.to_thread(comms_utils.end_assistant_call, call_session_id)
        # The ring died unanswered: drop its queued pre-armed gate so it
        # cannot leak into a later, unrelated call.
        self.call_manager.pending_hang_up_gate = ""
        self.notifications_bar.push_notif(
            "Comms",
            (
                "My Unify Meet call went unanswered. Continue with the boss here "
                "over the current text channel instead - do not keep waiting on "
                "the call."
            ),
            prompt_now(as_string=False),
        )
        await self.run_llm(
            trace_meta={"origin_event_name": "unify_meet_ring_unanswered"},
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
        a dedicated ``spoken_message`` so the fast brain utters the error via
        TTS directly (bypassing its own LLM, which may be hitting the same
        provider outage). Also cancels any pending proactive-speech loop so
        it stops emitting "still looking" filler for a request the slow brain
        has given up on.
        """
        spoken_message = (
            "Sorry, I'm having trouble thinking right now — "
            "could you say that again in a moment?"
        )
        notification_message = (
            f"Slow-brain turn failed after retries were exhausted "
            f"({type(exc).__name__}). The user's last request was not processed. "
            "Acknowledge the error and ask them to try again; do NOT claim you "
            "are still working on the prior request."
        )
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact or {},
            message=notification_message,
            spoken_message=spoken_message,
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

    async def _publish_slow_brain_fast_brain_guidance(
        self,
        *,
        message: str,
        slow_brain_log_path: str = "",
        fast_brain_guidance: str = "",
        console_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        """Publish a slow-brain spoken line (``guide_voice_agent``) to the fast brain.

        ``guide_voice_agent`` is speak-only, so this always publishes a spoken
        line (``should_speak=True``); there is no silent-guidance path.
        ``fast_brain_guidance`` rides bundled with the spoken line (a short note
        the fast brain may use for a basic direct reply to the next message); it
        is always sent so an empty value clears any stale note.
        """
        if not message:
            return
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact,
            message=message,
            should_speak=True,
            source="slow_brain",
            llm_log_path=slow_brain_log_path,
            fast_brain_guidance=fast_brain_guidance,
            console_steps=console_steps or [],
        )
        self._session_logger.info(
            "call_notification",
            f"Guide FastBrain (speak): {message}",
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

    async def _perform_deferred_hang_up(self, *, awaiting_speech: bool) -> None:
        """Run a hang-up the ``hang_up`` tool deferred, after speech is delivered.

        When the same turn produced spoken guidance, wait for that line to land
        (the matching outbound utterance sets ``_inflight_speech_delivered``; a
        barge-in's truncated prefix counts too) before tearing the session down,
        so the call never ends mid-utterance. A timeout guards a line that never
        surfaces. Standalone hang-ups (no spoken line) tear down immediately.
        """
        teardown = self._pending_hang_up_teardown
        self._pending_hang_up = False
        self._pending_hang_up_teardown = None
        if teardown is None:
            return
        if awaiting_speech:
            try:
                await asyncio.wait_for(
                    self._inflight_speech_delivered.wait(),
                    timeout=_HANG_UP_SPEECH_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                self._session_logger.info(
                    "call_notification",
                    "Deferred hang-up proceeding without spoken-line ack (timeout)",
                )
        await teardown()

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
        except SpendingLimitExceededError as exc:
            # A refusal with a stated cause and remedy. The pre-turn billing
            # gate catches the account-wide cases, but it fails open and is
            # blind to per-call causes such as a paid-only provider, so the
            # spend boundary stays the only place every refusal is known.
            with contextlib.suppress(Exception):
                await self._surface_spending_gate_refusal(exc)
            raise
        except Exception as exc:
            if self.mode.is_voice and self._is_transient_llm_error(exc):
                with contextlib.suppress(Exception):
                    await self._notify_fast_brain_of_slow_brain_failure(exc)
            elif not self.mode.is_voice:
                # Text surfaces have no fast brain to apologise for a dead
                # slow brain — without this, a hard failure (e.g. an
                # unconstructable LLM client on a stale image) is pure
                # silence, and users have re-sent the same message for
                # hours. Throttled so repeated failures produce one
                # apology, not one per attempt.
                with contextlib.suppress(Exception):
                    await self._send_slow_brain_failure_reply()
            raise

    async def _send_slow_brain_failure_reply(self) -> None:
        """Tell the user their message hit a hard failure (throttled)."""
        reply_context = self._last_inbound_reply_context
        if not reply_context:
            return
        now = self.loop.time()
        last_sent = getattr(self, "_slow_brain_failure_reply_sent_at", None)
        if (
            last_sent is not None
            and now - last_sent < SLOW_BRAIN_FAILURE_REPLY_THROTTLE_SECONDS
        ):
            return
        self._slow_brain_failure_reply_sent_at = now
        await self._send_system_reply(
            reply_context,
            content=SLOW_BRAIN_FAILURE_RESPONSE,
            email_subject=SLOW_BRAIN_FAILURE_EMAIL_SUBJECT,
        )

    async def _surface_spending_gate_refusal(
        self,
        exc: SpendingLimitExceededError,
    ) -> None:
        """Deliver a spending-gate refusal to the user, in its own words.

        The reason string is authored for the account holder and names the
        remedy ("add a payment method", "switch to one of the included
        models"), so every surface repeats it rather than paraphrasing.
        Reusing the transient-failure copy here would tell someone to retry
        an operation whose outcome is fixed until they act on the account.

        Shares the billing-gate throttle with the pre-turn check so a blocked
        account that keeps sending gets one explanation, not one per message.
        """
        reason = str(exc).strip()
        if not reason:
            return

        # Console classifies on this type, so a refusal renders as its own
        # message instead of the generic "attempting to recover" copy.
        with contextlib.suppress(Exception):
            publish_system_error(reason, error_type="billing_blocked")

        if self.mode.is_voice:
            with contextlib.suppress(Exception):
                await self._speak_spending_gate_refusal(reason)
            return

        reply_context = self._last_inbound_reply_context
        if not reply_context:
            return
        if self._credit_gate_reply_is_throttled(reply_context):
            self._session_logger.info(
                "billing_gate",
                "Skipped repeated spending-gate refusal reply",
            )
            return
        await self._send_system_reply(
            reply_context,
            content=reason,
            email_subject=SPENDING_GATE_EMAIL_SUBJECT,
        )

    async def _speak_spending_gate_refusal(self, reason: str) -> None:
        """Have the fast brain state a spending refusal aloud.

        Speaks directly rather than notifying, for the same reason a provider
        outage does: the fast brain's own model call runs through the gate
        that just refused, so asking it to compose the apology would hit the
        same wall.
        """
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact or {},
            message=(
                f"The spending gate refused this turn: {reason} "
                "The user has been told; do not claim you are still working."
            ),
            spoken_message=f"{SPENDING_GATE_SPOKEN_PREFIX}{reason}",
            should_speak=True,
            source="spending_gate_refusal",
        )
        self._session_logger.info(
            "billing_gate",
            f"Speaking spending-gate refusal: {reason}",
        )
        event_json = event.to_json()
        await self.event_broker.publish("app:call:notification", event_json)
        await self.event_broker.publish(
            "app:comms:assistant_notification",
            event_json,
        )

    def record_last_inbound_reply(self, reply_context: dict[str, Any]) -> None:
        self._last_inbound_reply_context = reply_context

    def _clamp_wait_poll_delay(self, delay: int) -> int:
        """Apply escalating backoff to repeated self-scheduled wait polls.

        Keeps the first ``WAIT_POLL_FREE_BUDGET`` timer wakes per window at
        the model's requested cadence, then doubles the enforced minimum
        per extra poll (capped). External events are unaffected — they wake
        the brain immediately whether or not a timer is pending.
        """
        polls = getattr(self, "_wait_poll_times", None)
        if polls is None:
            polls = self._wait_poll_times = collections.deque(maxlen=64)
        now = self.loop.time()
        polls.append(now)
        recent = sum(1 for t in polls if now - t <= WAIT_POLL_WINDOW_SECONDS)
        excess = recent - WAIT_POLL_FREE_BUDGET
        if excess <= 0:
            return delay
        floor = min(
            WAIT_POLL_MIN_CLAMPED_DELAY_SECONDS * (2 ** (excess - 1)),
            WAIT_POLL_MAX_CLAMPED_DELAY_SECONDS,
        )
        if delay < floor:
            self._session_logger.info(
                "wait",
                f"Wait-poll backoff: raising delay {delay}s -> {floor}s "
                f"({recent} timer polls in the last "
                f"{int(WAIT_POLL_WINDOW_SECONDS)}s)",
            )
            return floor
        return delay

    def _credit_gate_throttle_key(
        self,
        reply_context: dict[str, Any],
    ) -> tuple[str, str]:
        medium = str(reply_context.get("medium") or "")
        target = (
            reply_context.get("api_message_id")
            or reply_context.get("channel_id")
            or reply_context.get("chat_id")
            or reply_context.get("contact_id")
            or ""
        )
        return (medium, str(target))

    def _credit_gate_reply_is_throttled(
        self,
        reply_context: dict[str, Any],
    ) -> bool:
        if reply_context.get("medium") == Medium.API_MESSAGE.value:
            return False

        throttle_key = self._credit_gate_throttle_key(reply_context)
        last_sent_at = self._credit_gate_reply_sent_at.get(throttle_key)
        now = self.loop.time()
        if (
            last_sent_at is not None
            and now - last_sent_at < CREDIT_GATE_REPLY_THROTTLE_SECONDS
        ):
            return True

        self._credit_gate_reply_sent_at[throttle_key] = now
        return False

    async def _send_system_reply(
        self,
        reply_context: dict[str, Any],
        *,
        content: str,
        email_subject: str,
    ) -> bool:
        """Deliver a canned system message over the inbound reply channel.

        Shared by the credit-gate reply and the slow-brain hard-failure
        apology: routes plain text back over whichever medium the last
        inbound message arrived on.
        """
        medium = reply_context.get("medium")
        contact_id = reply_context.get("contact_id")
        tools = ConversationManagerBrainActionTools(self)

        if medium == Medium.UNIFY_MESSAGE.value:
            send_kwargs: dict[str, Any] = {
                "content": content,
            }
            raw_group_id = reply_context.get("group_id")
            raw_team_id = reply_context.get("team_id")
            if raw_group_id not in (None, ""):
                send_kwargs["group_id"] = int(raw_group_id)
            elif raw_team_id not in (None, ""):
                send_kwargs["team_id"] = int(raw_team_id)
            elif contact_id is not None:
                send_kwargs["contact_id"] = contact_id
            else:
                return False
            await tools.send_unify_message(**send_kwargs)
        elif medium == Medium.SMS_MESSAGE.value and contact_id is not None:
            await tools.send_sms(
                contact_id=contact_id,
                content=content,
            )
        elif medium == Medium.WHATSAPP_MESSAGE.value and contact_id is not None:
            await tools.send_whatsapp(
                contact_id=contact_id,
                content=content,
            )
        elif medium == Medium.EMAIL.value:
            email_id = reply_context.get("email_id")
            thread_id = reply_context.get("thread_id")
            if email_id:
                await tools.send_email(
                    subject=email_subject,
                    body=content,
                    reply_all=True,
                    email_id_to_reply_to=email_id,
                    thread_id=thread_id,
                )
            elif contact_id is not None:
                await tools.send_email(
                    to=[contact_id],
                    subject=email_subject,
                    body=content,
                )
            else:
                return False
        elif medium == Medium.API_MESSAGE.value:
            await tools.send_api_response(
                contact_id=contact_id or SESSION_DETAILS.boss_contact_id,
                content=content,
                tags=reply_context.get("tags"),
            )
        elif medium == Medium.DISCORD_MESSAGE.value and contact_id is not None:
            await tools.send_discord_message(
                contact_id=contact_id,
                content=content,
            )
        elif medium == Medium.DISCORD_CHANNEL_MESSAGE.value and reply_context.get(
            "channel_id",
        ):
            await tools.send_discord_channel_message(
                channel_id=reply_context["channel_id"],
                guild_id=reply_context.get("guild_id") or "",
                contact_id=contact_id,
                content=content,
            )
        elif medium == Medium.SLACK_MESSAGE.value and contact_id is not None:
            await tools.send_slack_message(
                contact_id=contact_id,
                content=content,
                team_id=reply_context.get("team_id") or "",
                thread_ts=reply_context.get("thread_ts"),
            )
        elif medium == Medium.SLACK_CHANNEL_MESSAGE.value and reply_context.get(
            "channel_id",
        ):
            await tools.send_slack_channel_message(
                channel_id=reply_context["channel_id"],
                team_id=reply_context.get("team_id") or "",
                thread_ts=reply_context.get("thread_ts"),
                contact_id=contact_id,
                content=content,
            )
        elif medium == Medium.TEAMS_MESSAGE.value and contact_id is not None:
            await tools.send_teams_message(
                contact_id=contact_id,
                content=content,
                chat_id=reply_context.get("chat_id"),
            )
        elif medium == Medium.TEAMS_CHANNEL_MESSAGE.value and reply_context.get(
            "channel_id",
        ):
            await tools.send_teams_message(
                contact_id=contact_id or SESSION_DETAILS.boss_contact_id,
                content=content,
                channel_id=reply_context.get("channel_id"),
                team_id=reply_context.get("team_id"),
            )
        elif (
            medium == Medium.MS_TEAMS_BOT_MESSAGE.value
            and reply_context.get("tenant_id")
            and reply_context.get("conversation_id")
        ):
            await tools.send_ms_teams_bot_message(
                contact_id=contact_id,
                content=content,
                tenant_id=reply_context["tenant_id"],
                conversation_id=reply_context["conversation_id"],
            )
        elif (
            medium == Medium.MS_TEAMS_BOT_CHANNEL_MESSAGE.value
            and reply_context.get("tenant_id")
            and reply_context.get("conversation_id")
        ):
            await tools.send_ms_teams_bot_channel_message(
                contact_id=contact_id,
                content=content,
                tenant_id=reply_context["tenant_id"],
                conversation_id=reply_context["conversation_id"],
            )
        else:
            return False

        return True

    async def _send_billing_gate_reply(
        self,
        reply_context: dict[str, Any],
        blocked_by: str | None,
    ) -> bool:
        """Reply with the advice that actually clears this refusal.

        A suspension and an empty wallet need different actions, and neither
        resolves by trying again — so neither may fall back to the generic
        transient-failure copy.
        """
        if blocked_by == GATE_BLOCK_ACCOUNT_SUSPENDED:
            content = ACCOUNT_SUSPENDED_SLOW_BRAIN_RESPONSE
            subject = ACCOUNT_SUSPENDED_EMAIL_SUBJECT
        else:
            content = DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE
            subject = DEPLETED_CREDITS_EMAIL_SUBJECT
        return await self._send_system_reply(
            reply_context,
            content=content,
            email_subject=subject,
        )

    async def _maybe_handle_billing_gate(
        self,
        trace_meta: dict[str, Any],
    ) -> bool:
        reply_context = trace_meta.get("credit_gate_reply_context")
        if not reply_context:
            return False

        gate_state = await check_billing_gate_state()
        if gate_state.allowed:
            return False

        blocked_by = gate_state.blocked_by or GATE_BLOCK_CREDITS_DEPLETED
        if self._credit_gate_reply_is_throttled(reply_context):
            self._session_logger.info(
                "billing_gate",
                f"Skipped repeated billing-gate reply ({blocked_by})",
            )
            return True

        sent = await self._send_billing_gate_reply(reply_context, blocked_by)
        self._session_logger.info(
            "billing_gate",
            (
                f"Served billing-gate reply ({blocked_by})"
                if sent
                else f"Skipped billing-gate reply ({blocked_by}) "
                "without a deliverable channel"
            ),
        )
        return True

    async def request_llm_run(
        self,
        delay=0,
        triggering_contact_id: int | None = None,
        is_user_origin: bool = False,
        credit_gate_reply_context: dict[str, Any] | None = None,
        turn_id: int | None = None,
    ) -> str:
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
            # Carried onto the debouncer task so the fast brain can cancel exactly
            # this turn's run by id. ``None`` for non-voice / non-user triggers,
            # which must never be matched by a fast-brain cancel.
            "turn_id": turn_id,
        }
        if credit_gate_reply_context is not None:
            request_meta["credit_gate_reply_context"] = credit_gate_reply_context
        self._pending_llm_requests.append((delay, is_user_origin))
        self._pending_llm_request_meta.append(request_meta)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.request_llm_run queued "
                "request_id=%s origin_event=%s delay=%s "
                "is_user_origin=%s pending=%d ready_for_brain=%s"
            ),
            request_id,
            request_meta["origin_event_name"] or "-",
            delay,
            is_user_origin,
            len(self._pending_llm_requests),
            self.ready_for_brain,
        )
        self._session_logger.debug(
            "llm_queue",
            (
                f"Queued slow-brain run request_id={request_id} "
                f"origin_event_id={request_meta['origin_event_id'] or '-'} "
                f"origin_event={request_meta['origin_event_name'] or '-'} "
                f"delay={delay} is_user_origin={is_user_origin}"
            ),
        )
        return request_id

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
            if requests[i][1]:  # is_user_origin
                selected_idx = i
                break

        dropped_requests = len(requests) - 1
        delay, is_user_origin = requests[selected_idx]
        selected_meta = dict(metas[selected_idx]) if metas else {}

        self._pending_llm_requests.clear()
        self._pending_llm_request_meta.clear()

        self._llm_run_seq += 1
        run_id = f"llmrun-{self._llm_run_seq:06d}"
        selected_meta["run_id"] = run_id
        selected_meta["dropped_requests"] = str(dropped_requests)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.flush_llm_requests dispatch "
                "run_id=%s request_id=%s origin_event=%s dropped=%d delay=%s "
                "is_user_origin=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-"),
            selected_meta.get("origin_event_name", "-") or "-",
            dropped_requests,
            delay,
            is_user_origin,
        )

        self._session_logger.debug(
            "llm_thinking",
            (
                f"Dispatching slow-brain run_id={run_id} "
                f"request_id={selected_meta.get('request_id', '-')} "
                f"origin_event_id={selected_meta.get('origin_event_id', '-') or '-'} "
                f"origin_event={selected_meta.get('origin_event_name', '-') or '-'} "
                f"dropped_requests={dropped_requests} delay={delay} "
                f"is_user_origin={is_user_origin}"
            ),
        )
        if await self._maybe_handle_billing_gate(selected_meta):
            log_startup_timing(
                LOGGER,
                (
                    "⏱️ [StartupTiming] first_reply.credit_gate_blocked "
                    "run_id=%s request_id=%s origin_event=%s"
                ),
                run_id,
                selected_meta.get("request_id", "-") or "-",
                selected_meta.get("origin_event_name", "-") or "-",
            )
            return

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.run_llm_submitted "
                "run_id=%s request_id=%s origin_event=%s was_queued=%s mode=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-") or "-",
            selected_meta.get("origin_event_name", "-") or "-",
            self.debouncer.was_queued,
            self.mode,
        )
        await self.run_llm(
            delay=delay,
            trace_meta=selected_meta,
        )

    async def _open_slow_brain_follow_on_turn(
        self,
        *,
        origin_run_id: str,
        previous_tools: list[str],
    ) -> None:
        """Schedule another slow-brain turn when the prior turn omitted wait."""
        if not self.ready_for_brain:
            return

        from unify.conversation_manager.domains.event_handlers import EventHandler
        from unify.conversation_manager.events import OpenSlowBrainTurn

        event = OpenSlowBrainTurn(
            origin_run_id=origin_run_id,
            previous_tools=list(previous_tools),
        )
        await EventHandler.handle_event(event, self)
        await self.flush_llm_requests()

    async def _run_llm(self, trace_meta: dict[str, str] | None = None) -> list[str]:
        """Run a single LLM decision and return all tool names that were called."""
        import time as _rl_time

        from datetime import datetime, timezone

        from ..events.cost_attribution import COST_ATTRIBUTION

        # Hold the turn while boot hydration is still landing. A reply
        # rendered from a pre-hydration view answers with confident ignorance
        # about a conversation whose history is seconds from appearing, and
        # the post-init follow-up turn cannot reliably repair a wrong first
        # answer already sent. Serving during init is otherwise unchanged:
        # the gate reopens the moment hydration resolves, well before manager
        # init finishes, so pre-init replies still happen — just never from
        # an empty view of a non-empty conversation.
        if not self._hydration_gate.is_set():
            _gate_t0 = _rl_time.perf_counter()
            try:
                await asyncio.wait_for(
                    self._hydration_gate.wait(),
                    timeout=BOOT_HYDRATION_MAX_WAIT_SECONDS,
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    f"{DEFAULT_ICON} [ConversationManager] Boot hydration "
                    f"still pending after {BOOT_HYDRATION_MAX_WAIT_SECONDS:.0f}s "
                    "— rendering this turn without hydrated history",
                )
            log_startup_timing(
                LOGGER,
                "⏱️ [StartupTiming] first_reply.hydration_gate_wait duration=%.2fs",
                _rl_time.perf_counter() - _gate_t0,
            )

        _preamble_t0 = _rl_time.perf_counter()
        _last_preamble_step = _preamble_t0

        def _ms_since_start() -> str:
            return f"{(_rl_time.perf_counter() - _preamble_t0) * 1000:.0f}ms"

        def _mark_preamble_step() -> float:
            nonlocal _last_preamble_step
            now = _rl_time.perf_counter()
            elapsed_ms = (now - _last_preamble_step) * 1000
            _last_preamble_step = now
            return elapsed_ms

        trace_meta = trace_meta or {}

        # Resolve per-turn org member attribution (only meaningful in org
        # context, where a cost can be attributed to a specific member).
        attributed_user_id = None
        if SESSION_DETAILS.org_id is not None:
            triggering_contact_id = trace_meta.get("triggering_contact_id")
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

        # The acting user for this turn: the inbound message sender when it maps
        # to a system user (boss or provisioned org member), else the workspace
        # owner. Drives per-user linked-desktop resolution in the prompt so a
        # shared assistant reflects the *speaker's* machine, not the owner's.
        acting_user_id = attributed_user_id or SESSION_DETAILS.user.id

        # Re-bind the billing context for THIS turn so credit deductions are
        # attributed to the assistant (and the acting member, in org context).
        # This must run for personal workspaces too: the context set once at
        # init does not reliably propagate to the generation execution
        # context, so without this LLM transactions are recorded with a NULL
        # assistant_id and disappear when filtering usage by assistant.
        try:
            import unillm

            from unify.conversation_manager.events import billing_source

            source, label = billing_source(
                str(trace_meta.get("origin_event_name") or ""),
                is_voice=self.mode.is_voice,
            )
            unillm.set_billing_context(
                assistant_id=SESSION_DETAILS.assistant.agent_id,
                user_id=acting_user_id,
                organization_id=SESSION_DETAILS.org_id,
                source=source,
                label=label,
            )
        except (ImportError, Exception):
            pass
        _cost_attribution_ms = _mark_preamble_step()

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
        _run_metadata_ms = _mark_preamble_step()

        slow_brain_start_time = datetime.now(timezone.utc)

        screenshots = self.peek_screenshot_buffer()
        _screenshot_peek_ms = _mark_preamble_step()

        screenshot_paths = [
            s.filepath or generate_screenshot_path(s) for s in screenshots
        ]
        _screenshot_paths_ms = _mark_preamble_step()

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
        _screenshot_attach_ms = _mark_preamble_step()

        self.snapshot()
        _snapshot_ms = _mark_preamble_step()

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
        _web_sessions_ms = _mark_preamble_step()

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.setup.detail "
                "run_id=%s total=%.0fms cost_attribution=%.0fms metadata=%.0fms "
                "screenshot_peek=%.0fms screenshot_paths=%.0fms "
                "screenshot_attach=%.0fms snapshot=%.0fms web_sessions=%.0fms "
                "screenshots=%d global_thread=%d chat_history=%d "
                "screen_share_active=%s"
            ),
            run_id,
            (_rl_time.perf_counter() - _preamble_t0) * 1000,
            _cost_attribution_ms,
            _run_metadata_ms,
            _screenshot_peek_ms,
            _screenshot_paths_ms,
            _screenshot_attach_ms,
            _snapshot_ms,
            _web_sessions_ms,
            len(screenshots),
            len(self.contact_index.global_thread),
            len(self.chat_history),
            self.assistant_screen_share_active,
        )

        # Render-only overlay: inject the line we just decided to speak (not yet
        # confirmed spoken) as a transient `[You]` row so this render treats it as
        # already said and never repeats it, then remove it immediately. It is
        # never persisted, so future turns see only the actually-spoken transcript.
        _inflight_entry = None
        if self._inflight_voice_speech and self.mode.is_voice:
            _inflight_contact = self.get_active_contact()
            if _inflight_contact:
                _inflight_entry = self.contact_index.build_message(
                    contact_id=_inflight_contact.get("contact_id"),
                    sender_name="You",
                    thread_name=self._active_voice_medium(),
                    message_content=self._inflight_voice_speech,
                    role="assistant",
                )
                self.contact_index.global_thread.append(_inflight_entry)

        _t0 = _rl_time.perf_counter()
        try:
            snapshot_state = self.prompt_renderer.render_state(
                self.contact_index,
                self.notifications_bar,
                self.in_flight_actions,
                self.completed_actions,
                self.last_snapshot,
                recent_tool_executions=self._recent_tool_executions,
                assistant_screen_share_active=self.assistant_screen_share_active,
                user_screen_share_active=self.user_screen_share_active,
                user_webcam_active=self.user_webcam_active,
                user_remote_control_active=self.user_remote_control_active,
                google_meet_active=self.call_manager.has_active_google_meet,
                teams_meet_active=self.call_manager.has_active_teams_meet,
                meet_screen_share_active=self.meet_screen_share_active,
                active_web_sessions=web_sessions,
                managers_initialized=self.initialized,
                vm_ready=self.vm_ready,
                file_sync_complete=self.file_sync_complete,
                has_desktop=SESSION_DETAILS.assistant.has_managed_desktop,
                console_open=self.console_is_open(),
                console_action_catalogue=self.console_action_catalogue(),
            )
        finally:
            # render_state is synchronous, so the transient row is always the
            # last entry here; remove it so it never persists.
            if _inflight_entry is not None:
                gt = self.contact_index.global_thread
                if gt and gt[-1] is _inflight_entry:
                    gt.pop()
                else:
                    with contextlib.suppress(ValueError):
                        gt.remove(_inflight_entry)
        _render_ms = (_rl_time.perf_counter() - _t0) * 1000

        # Mirror the Coordinator's onboarding state (defer switch + the
        # precomputed progress render) so the prompt builder reads a
        # standing "what's done / what's next" block and can drop
        # scaffolding when deferred. TTL-cached + event-refreshed, and
        # fetched in the background: a turn never blocks on the HTTP
        # round-trip — it builds with the previous values and the fresh
        # snapshot lands for the next turn.
        self._schedule_coordinator_onboarding_state_refresh()

        _t0 = _rl_time.perf_counter()
        brain_spec = build_brain_spec(
            self,
            snapshot_state=snapshot_state,
            screenshots=screenshots,
            screenshot_paths=screenshot_paths,
            acting_user_id=acting_user_id,
        )
        _brain_spec_ms = (_rl_time.perf_counter() - _t0) * 1000

        if screenshots:
            self._session_logger.debug(
                "screen_share",
                f"Attaching {len(screenshots)} screenshot(s) to slow brain turn",
            )
        _t0 = _rl_time.perf_counter()
        input_message = brain_spec.state_message()
        integration_sync_context = self.integration_sync_coordinator.prompt_summary()
        input_message = _append_context_to_state_message(
            input_message,
            integration_sync_context,
        )
        _state_message_ms = (_rl_time.perf_counter() - _t0) * 1000
        _t0 = _rl_time.perf_counter()
        system_prompt = brain_spec.system_prompt
        _system_prompt_ref_ms = (_rl_time.perf_counter() - _t0) * 1000

        self._current_state_snapshot = input_message

        self._current_snapshot_state = snapshot_state

        reason = (trace_meta or {}).get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            f"LLM thinking... ({reason})" if reason else "LLM thinking...",
        )

        _t0 = _rl_time.perf_counter()
        _tools_step_t0 = _t0
        brain_tools = ConversationManagerBrainTools(self)
        _brain_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tools = ConversationManagerBrainActionTools(self)
        _action_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        brain_tool_dict = brain_tools.as_tools()
        _brain_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tool_dict = action_tools.as_tools()
        _action_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        steering_tool_dict = action_tools.build_action_steering_tools()
        _steering_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        completed_tool_dict = action_tools.build_completed_action_tools()
        _completed_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        tools = {
            **brain_tool_dict,
            **action_tool_dict,
            **steering_tool_dict,
            **completed_tool_dict,
        }
        _tools_merge_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000

        _tools_step_t0 = _rl_time.perf_counter()
        if self.computer_fast_path_eligible:
            tools["desktop_act"] = action_tools.desktop_act
            tools["web_act"] = action_tools.web_act
            tools["close_web_session"] = action_tools.close_web_session
        _fast_path_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.tools.detail "
                "run_id=%s total=%.0fms brain_init=%.0fms action_init=%.0fms "
                "brain_tools=%.0fms action_tools=%.0fms steering=%.0fms "
                "completed=%.0fms merge=%.0fms fast_path=%.0fms "
                "brain_tool_count=%d action_tool_count=%d steering_tool_count=%d "
                "completed_tool_count=%d total_tool_count=%d"
            ),
            run_id,
            _tools_ms,
            _brain_tools_init_ms,
            _action_tools_init_ms,
            _brain_tools_ms,
            _action_tools_ms,
            _steering_tools_ms,
            _completed_tools_ms,
            _tools_merge_ms,
            _fast_path_tools_ms,
            len(brain_tool_dict),
            len(action_tool_dict),
            len(steering_tool_dict),
            len(completed_tool_dict),
            len(tools),
        )

        _t0 = _rl_time.perf_counter()
        _client_step_t0 = _t0
        client = new_slow_brain_llm_client(
            origin="ConversationManager",
            # Slow brain pins "high" explicitly so an assistant-level or
            # SLOW_BRAIN_REASONING_EFFORT override is the only thing that
            # can change it. When the assistant carries a default-model
            # effort override, it takes priority inside
            # new_slow_brain_llm_client(). Screenshots use the same
            # multimodal slow-brain client as text turns.
            reasoning_effort="high",
        )
        _new_client_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        if hasattr(client, "_pending_thinking_log"):
            parts = [
                p
                for p in [reason, "from queue" if self.debouncer.was_queued else ""]
                if p
            ]
            suffix = f" ({', '.join(parts)})" if parts else ""
            client._pending_thinking_log.set_thinking_context(suffix)
        _thinking_context_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_system_message(system_prompt.to_list())
        _set_system_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_prompt_caching(["system"])
        _prompt_caching_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        messages = self._preprocess_messages(self.chat_history + [input_message])
        _preprocess_messages_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.client.detail "
                "run_id=%s total=%.0fms new_client=%.0fms thinking_context=%.0fms "
                "set_system=%.0fms prompt_caching=%.0fms preprocess_messages=%.0fms "
                "state_message=%.0fms system_prompt_ref=%.0fms chat_history=%d "
                "message_count=%d system_parts=%d state_chars=%d"
            ),
            run_id,
            _client_ms,
            _new_client_ms,
            _thinking_context_ms,
            _set_system_ms,
            _prompt_caching_ms,
            _preprocess_messages_ms,
            _state_message_ms,
            _system_prompt_ref_ms,
            len(self.chat_history),
            len(messages),
            len(system_prompt.to_list()),
            len(brain_spec.state_prompt),
        )

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
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.llm_preamble "
                "run_id=%s duration=%s render_state=%.0fms brain_spec=%.0fms "
                "tools=%.0fms client=%.0fms tool_count=%d message_count=%d"
            ),
            run_id,
            _ms_since_start(),
            _render_ms,
            _brain_spec_ms,
            _tools_ms,
            _client_ms,
            len(tools),
            len(messages),
        )
        self._active_llm_trace_meta = trace_meta

        try:
            try:
                from unify.conversation_manager.domains.prose_send_healing import (
                    build_slow_brain_completion_mutator,
                )

                completion_mutator = build_slow_brain_completion_mutator(
                    self,
                    trace_meta=trace_meta,
                    available_tool_names=set(tools.keys()),
                )
                result = await single_shot_tool_decision(
                    client,
                    messages,
                    tools,
                    tool_choice="required" if tools else "auto",
                    inject_tool_thoughts=True,
                    exclusive_tools={
                        "make_call",
                        "make_whatsapp_call",
                        "join_google_meet",
                        "join_teams_meet",
                    },
                    on_tool_execution_start=lambda: self._mark_tool_commit_started(
                        trace_meta,
                        run_id,
                    ),
                    completion_mutator=completion_mutator,
                )
            except Exception:
                _log_slow_brain_single_shot_failure(
                    run_id=run_id,
                    request_id=request_id,
                    origin_event_name=origin_event_name,
                    message_count=len(messages),
                    tool_count=len(tools),
                    state_chars=len(input_message),
                )
                raise
        finally:
            self._active_llm_trace_meta = None
            if hasattr(client, "_pending_thinking_log"):
                client._pending_thinking_log.emit_fallback()
            _EVENT_SOURCE.reset(_source_token)
        tool_names = [t.name for t in result.tools]
        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] single_shot returned tools={tool_names}",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.single_shot duration=%s run_id=%s tools=%s",
            _rl_ms(),
            run_id,
            tool_names,
        )
        self._record_recent_tool_executions(
            tools=result.tools,
            trace_meta=trace_meta or {},
        )

        thoughts_summary = _format_tool_thoughts_for_log(result.tools)

        # Handle guide_voice_agent tool calls for voice modes. The slow brain
        # either SPEAKs (guide_voice_agent with a message, spoken verbatim by the
        # fast brain subprocess) or WAITs (omits the tool). It may bundle an
        # optional fast_brain_guidance note alongside a spoken message — never on
        # its own — which the fast brain may use for a basic direct reply to the
        # caller's next message.
        if self.mode.is_voice:
            guidance_message = ""
            fast_brain_guidance = ""
            console_targets: list[str] = []
            for tool_exec in result.tools:
                if tool_exec.name == "guide_voice_agent":
                    args = tool_exec.args or {}
                    guidance_message = args.get("message", "")
                    fast_brain_guidance = args.get("fast_brain_guidance", "")
                elif tool_exec.name == "show_in_console":
                    raw = (tool_exec.args or {}).get("targets") or []
                    console_targets = [str(t) for t in raw if str(t).strip()]

            # The spoken line may carry [[n]] markers naming console moves. They
            # are stripped on every medium -- a line reaching TTS with one still
            # in it gets read aloud -- but only a Meet can act on them. A phone
            # call's room is the SIP leg, which Console is not in, so its moves
            # went out over the event stream when the tool ran instead.
            console_steps: list[dict[str, Any]] = []
            if guidance_message and "[[" in guidance_message:
                if (
                    console_targets
                    and self.mode == Mode.MEET
                    and self.console_is_open()
                ):
                    parsed = parse_console_actions(guidance_message, console_targets)
                    guidance_message = parsed.spoken_text
                    console_steps = [
                        {"target": step.target, "afterChars": step.after_chars}
                        for step in parsed.steps
                    ]
                    if parsed.dropped:
                        self._session_logger.info(
                            "console_actions",
                            f"Dropped console moves: {'; '.join(parsed.dropped)}.",
                        )
                else:
                    # Markers with nothing to drive them: still not speakable.
                    guidance_message = strip_markers(guidance_message)

            # A pending hang-up (recorded by the hang_up tool this turn) must not
            # tear down the session until the spoken line has been delivered,
            # otherwise the call ends mid-sentence. Reset the delivered signal
            # before publishing so we only observe THIS turn's delivery.
            if self._pending_hang_up and guidance_message:
                self._inflight_speech_delivered.clear()

            if guidance_message:
                pending = getattr(client, "_pending_thinking_log", None)
                slow_brain_log_path = (
                    pending.last_path or "" if pending is not None else ""
                )
                # guide_voice_agent is speak-only: every call is spoken. Guidance
                # rides bundled with the spoken line (never alone); a spoken turn
                # without guidance clears any stale note on the fast brain.
                await self._publish_slow_brain_fast_brain_guidance(
                    message=guidance_message,
                    slow_brain_log_path=slow_brain_log_path,
                    fast_brain_guidance=fast_brain_guidance,
                    console_steps=console_steps,
                )
                # Stash the spoken line for a render-only overlay so the next run
                # (which may start before the real `[You]` utterance is recorded)
                # sees what this turn just decided to say, treats it as already
                # said, and does not repeat it. Cleared once the real utterance
                # lands so future turns see only what was actually spoken.
                self._stash_inflight_voice_speech(guidance_message)

            # Perform any deferred hang-up only after the spoken line has actually
            # been delivered (or a barge-in truncated it), so the session never
            # ends mid-utterance.
            if self._pending_hang_up:
                await self._perform_deferred_hang_up(
                    awaiting_speech=bool(guidance_message),
                )

        llm_response_msg = f"run_id={run_id}"
        if thoughts_summary:
            preview = (
                f"{thoughts_summary[:100]}..."
                if len(thoughts_summary) > 100
                else thoughts_summary
            )
            llm_response_msg += f" thoughts: {preview}"
        if tool_names:
            llm_response_msg += f" | actions: {tool_names}"
        self._session_logger.debug("llm_response", llm_response_msg)

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] voice notification done, committing",
        )
        self.commit()
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.commit completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
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
        assistant_content = result.text_response or ""
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
                    delay = self._clamp_wait_poll_delay(delay)
                    await self.run_llm(delay=delay)
                break

        if "wait" not in tool_names:
            await self._open_slow_brain_follow_on_turn(
                origin_run_id=run_id,
                previous_tools=tool_names,
            )

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] post-processing done",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.post_processing completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
        self._session_logger.debug(
            "llm_response",
            (f"Slow-brain run completed run_id={run_id} " f"tools={tool_names or '-'}"),
        )

        return tool_names

    def _mark_tool_commit_started(
        self,
        trace_meta: dict[str, str] | None,
        run_id: str,
    ) -> None:
        if trace_meta is not None:
            trace_meta["tool_commit_started"] = "true"
        running_meta = getattr(self.debouncer, "running_task_trace_meta", None)
        if isinstance(running_meta, dict) and running_meta.get("run_id") == run_id:
            running_meta["tool_commit_started"] = "true"
        self._session_logger.debug(
            "llm_thinking",
            f"Slow-brain run entered tool commit run_id={run_id}",
        )

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

            # A retired session stops listening: once `stop` is set the broker
            # has been (or is about to be) closed, so nothing new can arrive,
            # and an in-process successor must not find this loop still
            # holding the old session's machinery.
            while not self.stop.is_set():
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg:
                    continue
                # process events
                event = Event.from_json(msg["data"])
                channel = msg.get("channel", "")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                # Only a message that means somebody needs the assistant
                # keeps the pod alive. This used to advance on *any* traffic,
                # so the system's own chatter — invisible, because those
                # events are not loggable either — held a pod open
                # indefinitely: it never idled out, so it never picked up a
                # new image, and deploys could not reach it.
                if event.__class__.counts_as_activity:
                    self.last_activity_time = self.loop.time()
                    self._last_activity_source = (
                        f"{event.__class__.__name__} on {channel or '-'}"
                    )
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
                    _event_t0 = self.loop.time()
                    await EventHandler.handle_event(
                        event,
                        self,
                    )
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.handle_event duration=%.2fs event_id=%s event=%s channel=%s",
                        self.loop.time() - _event_t0,
                        event_id,
                        event_name,
                        channel or "-",
                    )
                    _flush_t0 = self.loop.time()
                    await self.flush_llm_requests()
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.flush_llm_requests duration=%.2fs event_id=%s event=%s",
                        self.loop.time() - _flush_t0,
                        event_id,
                        event_name,
                    )
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

    def _busy_snapshot(self) -> tuple[bool, str]:
        """Whether anything real is in flight, and what it is.

        One predicate, consumed by every branch of the inactivity check. The
        idle clocks are traffic *proxies*: pubsub says somebody sent us
        something, the EventBus says we published something. Neither knows
        whether work is happening right now, so anything that outlives the call
        that started it -- or lives outside this process entirely -- has to say
        so here or it can be torn down mid-flight.

        The reason string is returned rather than logged from inside: the
        inactivity check throttles its own logging, and a predicate that logged
        on every evaluation would write every 30s forever.
        """
        from unify.events.active_work import ACTIVE_WORK

        active_work = ACTIVE_WORK.snapshot()
        if active_work.active_count > 0:
            # Named, not just counted. A bare count says a declaration is
            # holding the pod without saying which subsystem made it, which is
            # most of what you want to know when a pod outlives its welcome --
            # or when `proxy_load_bearing` says one is missing. Distinct labels
            # only: ten parallel chunk writers are one thing worth reporting.
            labels = ",".join(sorted({str(w["label"]) for w in active_work.works}))
            return True, f"active_work({active_work.active_count}:{labels})"

        # Voice lives in a separate process, so its LLM calls advance that
        # process's EventBus and never this one. The parent only sees per-turn
        # IPC events, which means a connected-but-quiet call looks exactly like
        # an empty pod. Shutting down here runs cleanup_call_proc: it hangs up.
        call_manager = self.call_manager
        if call_manager.has_active_call:
            return True, "active_call"
        # Channel-agnostic on purpose. The per-channel properties resolve
        # through `_call_channel`, which is not set yet while a meet is still
        # joining -- so asking about google_meet and teams_meet individually
        # answers False for a meeting that is half-way into the room.
        if call_manager.has_active_meet():
            return True, "browser_meet"
        if call_manager._whatsapp_call_joining:
            return True, "whatsapp_call_joining"

        # Somebody is watching a screen. Watching generates no traffic at all.
        if self.assistant_screen_share_active:
            return True, "assistant_screen_share"
        if self.user_screen_share_active:
            return True, "user_screen_share"
        if self._assistant_screen_share_viewers:
            return True, (
                f"screen_share_viewers({len(self._assistant_screen_share_viewers)})"
            )

        # A turn mid-flight. The EventBus only stamps on *completion*, so a
        # single long call is a silent window on that clock. `running_task` is
        # an asyncio Task, so `done()` is authoritative -- it cannot go stale.
        running_task = getattr(self.debouncer, "running_task", None)
        if running_task is not None and not running_task.done():
            return True, "slow_brain_turn"

        # `in_flight_actions` is deliberately NOT consulted. It answers "which
        # handles can I steer?", not "is work happening?" -- and a persist-mode
        # act parks in it indefinitely by design, waiting for an interjection
        # that may never come. Reading presence as liveness therefore made any
        # assistant that ever ran `act(persist=True)` immortal: one pod held a
        # parked handle for 99 minutes while the EventBus went 96 minutes
        # without a publish. A running action needs no help from this registry
        # -- its code holds ACTIVE_WORK and its turns show as slow_brain_turn.
        #
        # Retiring a pod does discard the parked session's Python state. That
        # has been true at every timeout this system has shipped; durable
        # session state is the fix for it, not a pod that never exits.

        return False, ""

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout.

        Two clocks and one predicate:
        - ``pubsub_idle`` -- somebody sent us something (``counts_as_activity``)
        - ``eventbus_idle`` -- we published something (LLM calls, tool loops,
          manager methods)
        - ``_busy_snapshot`` -- something real is in flight *right now*

        The clocks are traffic proxies and only ever protect, because the
        decision reads their ``min``. The predicate is what makes "idle"
        trustworthy: work that outlives its caller, or runs outside this
        process, generates no traffic and must declare itself.

        Ghost-publish detection: a publisher that keeps ``eventbus_idle`` fresh
        forever pins ``min`` below the timeout and disarms the ordinary path
        permanently. When pubsub has been idle past the timeout while the
        EventBus keeps looking fresh, and nothing is busy, we stop trusting the
        EventBus clock and shut down anyway. It is gated on the same predicate
        as the ordinary path: it used to be gated on active work alone, which
        made it a second way in past every floor the ordinary path respected.
        """
        import time as _time

        from unify.events.active_work import ACTIVE_WORK
        from unify.events.event_bus import EventBus

        ghost_counter = 0
        suspect_streak = 0

        while not self.stop.is_set():
            await asyncio.sleep(self.inactivity_check_interval)
            # An explicit retirement (stop_async, SIGTERM) sets `stop` from
            # outside this loop; every internal exit already breaks after
            # `_request_shutdown`, so this check only ends the watch when
            # somebody else decided the session is over.
            if self.stop.is_set():
                break
            # A duration, not a check count. The previous constant (20 checks)
            # was tuned when the timeout was 420s, so it silently became a
            # different policy every time the timeout moved. Read per pass
            # because the timeout is mutable at runtime. A disabled timeout
            # (``inf``, via UNIFY_INACTIVITY_TIMEOUT_SECONDS=0) falls back to
            # the 600s floor: the ghost branch can never fire then (pubsub
            # can never out-idle an infinite timeout), but ``int(inf)`` would
            # kill this whole loop with OverflowError on its first pass.
            ghost_checks_needed = max(
                2,
                int(
                    max(
                        600.0,
                        (
                            self.inactivity_timeout
                            if math.isfinite(self.inactivity_timeout)
                            else 600.0
                        ),
                    )
                    / self.inactivity_check_interval,
                ),
            )
            current_time = self.loop.time()
            pubsub_idle = current_time - self.last_activity_time
            monotonic_now = _time.monotonic()
            eventbus_idle = monotonic_now - EventBus.last_publish_monotonic
            idle_seconds = min(pubsub_idle, eventbus_idle)
            is_busy, busy_reason = self._busy_snapshot()
            effective_idle_seconds = 0.0 if is_busy else idle_seconds

            # Control-plane drain: once in-flight work is gone, shut down so
            # the next wake loads a fresh client bundle.
            #
            # Only the probe is shielded. Wrapping the shutdown too is what
            # let `await self.stop()` -- a TypeError, because `stop` is an
            # Event and not a coroutine -- read as a transient probe failure
            # for three weeks: the pod logged that it was shutting down every
            # 30s and never did, and drains completed only when the control
            # plane force-stopped the session at its deadline.
            if not is_busy:
                try:
                    from unify.runtime.drain_gate import is_admission_blocked

                    drain_armed = is_admission_blocked()
                except Exception:  # noqa: BLE001 — never break inactivity loop
                    LOGGER.debug(
                        "drain gate probe in inactivity failed",
                        exc_info=True,
                    )
                    drain_armed = False
                if drain_armed:
                    await self._request_shutdown(
                        "drain_restart",
                        "Drain in progress and nothing in flight; "
                        "shutting down for restart",
                    )
                    break

            # A pod that cannot serve retires regardless of how busy its
            # inbox looks. Nothing here recovers in place: the replacement is
            # scheduled fresh, which is also how it picks up the image that
            # fixed whatever broke this one. Retrying in process could not
            # have done that -- the credential check that started this ran
            # against an image that had already been superseded.
            if not is_busy and self.unserviceable_reason:
                await self._request_shutdown(
                    "unserviceable",
                    "Cannot serve this assistant "
                    f"({self.unserviceable_reason}); retiring so a fresh pod "
                    "takes over",
                )
                break

            # A pod with no assistant has nobody to be idle *from*: its only
            # traffic is the pre-startup keepalive, which is not presence.
            # Its lifetime belongs to the pool, which deletes stale-image
            # members and trims the rest to target, so retiring itself here
            # would empty the warm pool an hour after it was filled. Drain
            # still applies above — a draining idle pod exits for a restart.
            if SESSION_DETAILS.assistant.agent_id is None:
                continue

            # Gated on `is_busy`, not on active work alone. With the full
            # predicate a genuine autonomous stretch resets the counter on its
            # first check, so the ghost branch can only ever fire for
            # publishing with nothing running -- which is all it was for.
            if (
                not is_busy
                and pubsub_idle > self.inactivity_timeout
                and eventbus_idle < self.inactivity_timeout
            ):
                ghost_counter += 1
            else:
                ghost_counter = 0

            ghost_publish = ghost_counter >= ghost_checks_needed

            # A declaration standing while *both* clocks are long past the
            # timeout is the shape of one that has gone stale -- real work moves
            # at least one clock. Reported, never acted on: a call silent for
            # twenty minutes looks exactly like a call flag nobody cleared, and
            # the only way to tell them apart is to go and look. Retiring on a
            # suspicion would hang up on whoever was still holding, so the
            # absolute bound stays with the 12h stale-runtime sweep, which
            # defers live calls by design.
            #
            # On its own streak rather than the heartbeat's wall-clock window:
            # an anomaly should announce itself on the first check that sees it,
            # not whenever the clock next lands in a 30s slot of a 180s cycle.
            if (
                is_busy
                and pubsub_idle > self.inactivity_timeout
                and eventbus_idle > self.inactivity_timeout
            ):
                suspect_streak += 1
                if suspect_streak == 1 or suspect_streak % 20 == 0:
                    self._session_logger.info(
                        "inactivity_check",
                        f"Declaration suspect: busy={busy_reason} held while "
                        f"pubsub_idle={pubsub_idle:.0f}s and "
                        f"eventbus_idle={eventbus_idle:.0f}s both exceed "
                        f"timeout={self.inactivity_timeout}s "
                        f"(streak={suspect_streak}); not retiring on a "
                        "suspicion -- verify the declaration is real",
                    )
            else:
                suspect_streak = 0

            if int(current_time) % 180 < self.inactivity_check_interval:
                extra = ""
                if ghost_counter > 0:
                    extra = f" ghost_count={ghost_counter}/{ghost_checks_needed}"
                if is_busy:
                    extra += f" busy={busy_reason}"
                    active_work = ACTIVE_WORK.snapshot()
                    if active_work.active_count > 0:
                        active_heartbeat_age = (
                            monotonic_now - active_work.newest_heartbeat_at
                            if active_work.newest_heartbeat_at is not None
                            else 0.0
                        )
                        extra += (
                            f" active_work_count={active_work.active_count}"
                            f" active_elapsed={active_work.oldest_elapsed_s:.1f}s"
                            f" active_heartbeat_age={active_heartbeat_age:.1f}s"
                        )
                elif eventbus_idle < self.inactivity_timeout <= pubsub_idle:
                    # The EventBus clock is the only thing holding this pod
                    # open: nothing is declared busy and pubsub has gone quiet.
                    # Every appearance of this line is a work type that owes a
                    # declaration, which is the whole audit turned into a
                    # measurement. If it stops appearing, the proxy is
                    # redundant and can leave the decision entirely.
                    extra += " proxy_load_bearing=eventbus"

                self._session_logger.info(
                    "inactivity_check",
                    f"Idle check: last_activity={self._last_activity_source}, "
                    f"pubsub_idle={pubsub_idle:.1f}s, "
                    f"eventbus_idle={eventbus_idle:.1f}s, "
                    f"min_idle={idle_seconds:.1f}s, "
                    f"effective_idle={effective_idle_seconds:.1f}s, "
                    f"timeout={self.inactivity_timeout}s{extra}",
                )

            if effective_idle_seconds > self.inactivity_timeout or ghost_publish:
                if ghost_publish:
                    reason = "ghost_publish"
                    log_str = (
                        f"Ghost-publish shutdown: pubsub_idle={pubsub_idle:.0f}s "
                        f"but eventbus_idle stuck at {eventbus_idle:.1f}s "
                        f"for {ghost_counter} consecutive checks "
                        f"(timeout={self.inactivity_timeout}s)"
                    )
                else:
                    reason = "idle_timeout"
                    log_str = f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown"
                await self._request_shutdown(reason, log_str)
                break  # Exit the loop after triggering shutdown

    async def _request_shutdown(self, reason: str, log_str: str) -> None:
        """Signal the process to wind down, from any reason the loop recognises.

        One sequence for every exit so a new one cannot half-implement it. The
        drain branch used to open-code its own and got it wrong, which is why
        this exists rather than three call sites that look similar.

        Whether the pod *should* stop is the caller's question, decided once in
        ``check_inactivity`` against ``_busy_snapshot``. Re-deciding it here
        would put two predicates behind one outcome, and the looser of the two
        wins by accident.
        """

        self.shutdown_reason = reason
        LOGGER.info(f"{DEFAULT_ICON} {log_str}")
        self._session_logger.info("session_end", log_str)
        self.stop.set()
        await self.event_broker.aclose()

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
        self.self_contact_id = int(payload["self_contact_id"])
        self.boss_contact_id = int(payload["boss_contact_id"])
        self.assistant_whatsapp_number = payload.get("assistant_whatsapp_number", "")
        self.assistant_discord_bot_id = payload.get("assistant_discord_bot_id", "")
        self.assistant_slack_bot_user_id = payload.get(
            "assistant_slack_bot_user_id",
            "",
        )
        self.assistant_slack_team_id = payload.get(
            "assistant_slack_team_id",
            "",
        )
        # Default to the current value (not False) so a capability adopted at
        # runtime from an inbound bot activity — or forced on via the
        # ASSISTANT_HAS_MS_TEAMS_BOT env var at startup — survives assistant
        # updates whose payload omits the key (Orchestra does not yet emit it).
        self.assistant_has_ms_teams_bot = bool(
            payload.get(
                "assistant_has_ms_teams_bot",
                self.assistant_has_ms_teams_bot,
            ),
        )
        self.user_first_name = payload["user_first_name"]
        self.user_surname = payload["user_surname"]
        self.user_number = payload["user_number"]
        self.user_email = payload["user_email"]
        self.user_whatsapp_number = payload.get("user_whatsapp_number", "")
        # Only adopt voice from the payload when it carries a real value. A
        # sparse AssistantUpdateEvent (voice omitted / coerced None -> "") must
        # not wipe the assistant's current voice back to the provider default.
        if payload.get("voice_provider"):
            self.voice_provider = payload["voice_provider"]
        if payload.get("voice_id"):
            self.voice_id = payload["voice_id"]
        # Adopt the default model unconditionally: unlike voice, empty is a
        # meaningful value (reset to the platform default model).
        self.default_model = payload.get("default_model", "")
        self.default_reasoning_effort = payload.get("default_reasoning_effort", "")
        self.slow_brain_model = payload.get("slow_brain_model", "")
        self.slow_brain_reasoning_effort = payload.get(
            "slow_brain_reasoning_effort",
            "",
        )
        self.binding_id = payload.get("binding_id", "")
        self.desktop_mode = payload.get("desktop_mode", "none")
        # Default to the current value, for the same reason as the Teams bot
        # flag below: AssistantUpdateEvent does not carry this key at all, so
        # reading it as absent-means-None turned every assistant update -- a
        # rename, a voice change, a membership change, an OAuth re-auth -- into
        # a silent revocation of the managed desktop. ``desktop_url`` survives
        # the repopulate, so the result was a session claiming a desktop it was
        # no longer entitled to use: the browser kept working off the URL while
        # shell, python and file access refused.
        self.managed_desktop_status = payload.get(
            "managed_desktop_status",
            getattr(self, "managed_desktop_status", None),
        )
        self.user_desktops = payload.get("user_desktops") or []
        self.org_id: int | None = payload.get("org_id")
        self.org_name: str = payload.get("org_name", "")
        self.team_ids: list[int] = payload.get("team_ids") or []
        team_summaries = payload.get("team_summaries") or []
        # Arrives as an int from the bootstrap secret's JSON, but coerce
        # defensively: a string here would make team_owned truthy while every
        # integer comparison downstream silently failed.
        raw_owner_team_id = payload.get("owner_team_id")
        try:
            self.owner_team_id: int | None = (
                int(raw_owner_team_id) if raw_owner_team_id not in (None, "") else None
            )
        except (TypeError, ValueError):
            self.owner_team_id = None
        is_coordinator = bool(payload.get("is_coordinator", False))
        is_multiplayer = bool(payload.get("is_multiplayer", False))
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
            assistant_slack_bot_user_id=self.assistant_slack_bot_user_id,
            assistant_slack_team_id=self.assistant_slack_team_id,
            assistant_has_ms_teams_bot=self.assistant_has_ms_teams_bot,
            # Passed explicitly from the live session: omitting it let
            # ``populate`` apply its own default and clear a tenant id adopted
            # at runtime, which is exactly the failure the bot flag beside it
            # was already patched for.
            assistant_ms_teams_tenant_id=SESSION_DETAILS.assistant.ms_teams_tenant_id,
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
            team_summaries=team_summaries,
            owner_team_id=self.owner_team_id,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            default_model=self.default_model,
            default_reasoning_effort=self.default_reasoning_effort,
            slow_brain_model=self.slow_brain_model,
            slow_brain_reasoning_effort=self.slow_brain_reasoning_effort,
            binding_id=self.binding_id,
            desktop_mode=self.desktop_mode,
            managed_desktop_status=self.managed_desktop_status,
            user_desktops=self.user_desktops,
            is_coordinator=is_coordinator,
            is_multiplayer=is_multiplayer,
        )
        self.team_summaries = SESSION_DETAILS.team_summaries
        # Export to env vars for subprocess inheritance
        SESSION_DETAILS.export_to_env()
        # The payload's owner_team_id is a launcher-delivered hint; the
        # platform record is the identity. Binding here covers every
        # session-config lane (StartupEvent, AssistantUpdateEvent): an omitted
        # value self-heals from the record and a disagreement stops the
        # session, so a payload that forgot the field cannot silently route a
        # team-owned assistant's shared tables to the personal root.
        SESSION_DETAILS.bind_derived_ownership()
        self.owner_team_id = SESSION_DETAILS.owner_team_id

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
        # Resolve the voice from the live runtime source rather than a frozen
        # snapshot: the CM's own fields take precedence, but fall back to
        # SESSION_DETAILS.voice (populated from the OS env at boot in self-host,
        # where no StartupEvent ever arrives). Without this fallback an empty
        # CM voice field silently sends the provider default to the call agent.
        voice_provider = (
            self.voice_provider or SESSION_DETAILS.voice.provider or "cartesia"
        )
        voice_id = self.voice_id or SESSION_DETAILS.voice.id or ""
        voice_provider, voice_id = resolve_runtime_voice(
            is_coordinator=SESSION_DETAILS.is_coordinator,
            voice_provider=voice_provider,
            voice_id=voice_id,
        )
        return CallConfig(
            assistant_id=self.assistant_id,
            user_id=self.user_id,
            assistant_bio=self.assistant_about,
            assistant_number=self.assistant_number,
            voice_provider=voice_provider,
            voice_id=voice_id,
            assistant_name=f"{self.assistant_first_name} {self.assistant_surname}".strip(),
            job_name=self.job_name,
            is_coordinator=SESSION_DETAILS.is_coordinator,
        )

    def _orchestra_state_client(self) -> Any:
        """Shared keep-alive HTTP client for Orchestra state calls.

        A fresh client per call paid a new TCP+TLS handshake against the
        (cross-region) Orchestra endpoint every time, eating a large slice of
        the request budget before the server even responded. One pooled
        client amortises that to the first call; ``cleanup`` closes it.
        """
        import httpx as _httpx

        client = self._coordinator_state_http
        if client is None or client.is_closed:
            client = _httpx.AsyncClient(timeout=5.0)
            self._coordinator_state_http = client
        return client

    def _schedule_coordinator_onboarding_state_refresh(self) -> None:
        """Kick the onboarding-state refresh without blocking the caller.

        The refresh is a TTL backstop — onboarding events already push fresh
        renders in real time — so a slow-brain turn must never sit on its HTTP
        round-trip. The previous values stay in place until the background
        fetch lands; a still-running fetch is simply left to finish.
        """
        task = self._coordinator_state_refresh_task
        if task is not None and not task.done():
            return
        self._coordinator_state_refresh_task = asyncio.create_task(
            self._refresh_coordinator_onboarding_state(),
        )

    async def _refresh_coordinator_onboarding_state(
        self,
        *,
        force: bool = False,
    ) -> None:
        """Best-effort refresh of the cached onboarding state for the brain.

        Mirrors onboarding gate, render, and chat-intro intent from
        Orchestra's ``Coordinator/State`` onto the session so
        ``build_brain_spec`` never has to derive anything:
          - ``coordinator_onboarding_active``: whether onboarding scaffolding
            is live (narration, progress block, tool masking).
          - ``coordinator_onboarding_render``: the precomputed
            depends_on-aware picture (steps + statuses + valid next
            targets with nudge copy) that drives the standing progress
            block.
          - ``coordinator_intro_watched`` / ``coordinator_pending_chat_intro`` /
            ``coordinator_chat_intro_armed_at``: durable chat-opener intent.

        TTL-cached so we don't pay an HTTP round-trip every turn; the
        render is also refreshed in real time from each onboarding event
        (see ``set_coordinator_onboarding_render``), so this is mostly a
        backstop. Non-coordinator sessions and Console-less deployments
        skip it. Failures leave the previous values in place.
        """
        if not self.is_coordinator or not SETTINGS.UNIFY_CONSOLE_UI:
            return
        import time as _time

        now = _time.monotonic()
        # Refresh more eagerly while actively onboarding (a render is
        # present) so "what's next" stays fresh during a fast-moving
        # setup conversation; back off once onboarding is done/deferred.
        ttl = 10.0 if self.coordinator_onboarding_active else 30.0
        # Stamp before the await so concurrent runs don't stampede the
        # endpoint; a failed fetch still respects the TTL backoff.
        if not force and now - self._coordinator_state_checked_at < ttl:
            return
        self._coordinator_state_checked_at = now
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return

        try:
            client = self._orchestra_state_client()
            resp = await client.get(
                f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                headers={"Authorization": f"Bearer {SESSION_DETAILS.unify_key}"},
            )
            resp.raise_for_status()
            info = (resp.json() or {}).get("info") or {}
            self._apply_coordinator_state_info(info)
        except Exception as exc:
            # repr, not str: httpx timeout exceptions stringify to "".
            LOGGER.warning(
                "Coordinator onboarding-state refresh failed; "
                "keeping previous values (active=%s): %r",
                self.coordinator_onboarding_active,
                exc,
            )

    def _apply_coordinator_state_info(self, info: dict[str, Any]) -> None:
        """Mirror onboarding gate + render from a Coordinator/State snapshot."""
        self.coordinator_onboarding_active = bool(info.get("onboarding_active"))
        render = info.get("onboarding")
        self.coordinator_onboarding_render = (
            render if isinstance(render, dict) else None
        )
        self.coordinator_intro_watched = bool(info.get("intro_watched"))
        self.coordinator_pending_chat_intro = bool(info.get("pending_chat_intro"))
        armed_at = info.get("chat_intro_armed_at")
        self.coordinator_chat_intro_armed_at = (
            armed_at if isinstance(armed_at, str) and armed_at.strip() else None
        )

    async def _patch_coordinator_pending_chat_intro(
        self,
        *,
        pending: bool,
    ) -> dict[str, Any]:
        """PATCH ``pending_chat_intro`` on Orchestra and refresh the session cache."""
        from unify.settings import SETTINGS

        if not self.is_coordinator or not SETTINGS.UNIFY_CONSOLE_UI:
            return {
                "status": "error",
                "message": "Chat intro state can only be changed for the Coordinator.",
            }
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return {
                "status": "error",
                "message": "Coordinator agent id is missing.",
            }
        import time as _time

        try:
            resp = await self._orchestra_state_client().patch(
                f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                headers={
                    "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                },
                json={"pending_chat_intro": pending},
            )
            resp.raise_for_status()
            info = (resp.json() or {}).get("info") or {}
            if isinstance(info, dict):
                self._apply_coordinator_state_info(info)
            self._coordinator_state_checked_at = _time.monotonic()
            return {"status": "ok", "pending_chat_intro": pending}
        except Exception as exc:
            LOGGER.warning(
                "Coordinator pending_chat_intro PATCH failed (pending=%s): %r",
                pending,
                exc,
            )
            return {
                "status": "error",
                "message": f"Failed to update chat intro state: {exc!r}",
            }

    async def _patch_coordinator_onboarding_active(
        self,
        *,
        active: bool,
        clear_onboarding_step: bool = False,
    ) -> dict[str, Any]:
        """PATCH ``onboarding_active`` on Orchestra and refresh the session cache."""
        from unify.settings import SETTINGS

        if not self.is_coordinator or not SETTINGS.UNIFY_CONSOLE_UI:
            return {
                "status": "error",
                "message": "Onboarding can only be toggled for the workspace Coordinator.",
            }
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return {
                "status": "error",
                "message": "Coordinator agent id is missing.",
            }
        body: dict[str, Any] = {"onboarding_active": active}
        if clear_onboarding_step:
            body["clear_onboarding_step"] = True
        import time as _time

        try:
            resp = await self._orchestra_state_client().patch(
                f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                headers={
                    "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                },
                json=body,
            )
            if resp.status_code == 403:
                return {
                    "status": "error",
                    "message": (
                        "I do not have permission to change onboarding state "
                        "in this workspace."
                    ),
                }
            resp.raise_for_status()
            info = (resp.json() or {}).get("info") or {}
            if isinstance(info, dict):
                self._apply_coordinator_state_info(info)
            self._coordinator_state_checked_at = _time.monotonic()
            if active:
                message = (
                    "Onboarding is live again — the setup checklist and nudges "
                    "are back."
                )
            else:
                message = (
                    "Onboarding is paused — they can use the platform normally "
                    "and resume setup anytime from the Onboarding tab or by "
                    "asking me."
                )
            return {
                "status": "ok",
                "message": message,
                "onboarding_active": self.coordinator_onboarding_active,
            }
        except Exception as exc:
            LOGGER.warning(
                "Coordinator onboarding_active PATCH failed (active=%s): %r",
                active,
                exc,
            )
            return {
                "status": "error",
                "message": f"Failed to update onboarding state: {exc!r}",
            }

    async def _patch_coordinator_onboarding_step_state(
        self,
        *,
        step_id: str,
        completed: bool,
    ) -> dict[str, Any]:
        """PATCH manual onboarding step completion on Orchestra."""
        from unify.settings import SETTINGS

        if not self.is_coordinator or not SETTINGS.UNIFY_CONSOLE_UI:
            return {
                "status": "error",
                "message": (
                    "Onboarding step completion can only be changed for the "
                    "workspace Coordinator."
                ),
            }
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return {
                "status": "error",
                "message": "Coordinator agent id is missing.",
            }
        import time as _time

        try:
            resp = await self._orchestra_state_client().patch(
                f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                headers={
                    "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                },
                json={
                    "onboarding_step_completion": {
                        "step_id": step_id,
                        "completed": completed,
                    },
                },
                timeout=30.0,
            )
            payload = resp.json() if resp.content else {}
            if resp.status_code == 403:
                return {
                    "status": "error",
                    "message": (
                        "I do not have permission to change onboarding step "
                        "completion in this workspace."
                    ),
                }
            if resp.status_code == 400:
                detail = payload.get("detail")
                if isinstance(detail, dict):
                    message = detail.get("message") or detail.get("code")
                else:
                    message = detail
                return {
                    "status": "error",
                    "message": message or "Invalid onboarding step completion.",
                    "step_id": step_id,
                }
            resp.raise_for_status()
            info = payload.get("info") or {}
            if isinstance(info, dict):
                self._apply_coordinator_state_info(info)
            self._coordinator_state_checked_at = _time.monotonic()
            completed_step_ids = (
                info.get("completed_step_ids")
                if isinstance(info.get("completed_step_ids"), list)
                else []
            )
            if completed:
                message = f"Marked '{step_id}' complete in the onboarding checklist."
            elif step_id in completed_step_ids:
                message = (
                    f"Removed my manual completion for '{step_id}', but the "
                    "checklist still shows it as done because the platform "
                    "detects it is already complete."
                )
            else:
                message = f"Marked '{step_id}' incomplete in the onboarding checklist."
            return {
                "status": "ok",
                "message": message,
                "step_id": step_id,
                "completed": completed,
            }
        except Exception as exc:
            LOGGER.warning(
                "Coordinator onboarding step completion PATCH failed "
                "(step_id=%s, completed=%s): %r",
                step_id,
                completed,
                exc,
            )
            return {
                "status": "error",
                "message": f"Failed to update onboarding step completion: {exc!r}",
                "step_id": step_id,
            }

    def set_coordinator_onboarding_render(self, render: Any) -> None:
        """Update the cached onboarding render from a fresh event payload.

        Onboarding events carry the same ``onboarding`` rendering the
        state endpoint returns, so the standing progress block stays
        current between TTL fetches the instant an event lands.
        """
        if isinstance(render, dict):
            self.coordinator_onboarding_render = render
            self._coordinator_state_checked_at = 0.0

    @property
    def onboarding_clicked_trigger_steps(self) -> set[str]:
        """Trigger-step ids clicked in this session (reference-quiz gating)."""
        return self._onboarding_clicked_trigger_steps

    def record_onboarding_trigger_clicked(self, step_id: str) -> None:
        """Mark a reference-quiz trigger row as clicked this session.

        Unlocks the matching comms send tool until the send durably completes
        the step. No-op for blank ids.
        """
        if isinstance(step_id, str) and step_id.strip():
            self._onboarding_clicked_trigger_steps.add(step_id.strip())

    def clear_onboarding_clicked_trigger_steps(self) -> None:
        """Forget this session's clicked trigger rows (e.g. on onboarding reset)."""
        self._onboarding_clicked_trigger_steps.clear()

    @property
    def learning_demo_storage_wake_armed(self) -> bool:
        """Whether a StorageCheck-completion notification should wake the brain.

        Armed on ``learning_beat_requested``, cleared on the
        ``learn-from-correction`` step completing or on a fresh
        ``onboarding_session_started`` — so an abandoned demo can't leave
        the wake armed across sessions.
        """
        return self._learning_demo_storage_wake_armed

    def set_learning_demo_storage_wake_armed(self, armed: bool) -> None:
        self._learning_demo_storage_wake_armed = armed

    def active_pending_onboarding_outbound(self) -> dict[str, Any] | None:
        """Return armed onboarding outbound metadata, or None if unset or expired."""
        pending = self._pending_onboarding_outbound
        if not pending:
            return None
        if self.loop.time() > float(pending.get("expires_at", 0)):
            self._pending_onboarding_outbound = None
            return None
        return pending

    def set_pending_onboarding_outbound(
        self,
        details: dict[str, Any],
        *,
        origin_event_id: str = "",
    ) -> None:
        trigger_step_id = details.get("trigger_step_id")
        channel = details.get("channel")
        tool_name = details.get("tool_name")
        if not isinstance(trigger_step_id, str) or not trigger_step_id.strip():
            return
        if not isinstance(channel, str) or not channel.strip():
            return
        pending: dict[str, Any] = {
            "onboarding_trigger_step_id": trigger_step_id.strip(),
            "onboarding_reply_step_id": (
                details.get("reply_step_id").strip()
                if isinstance(details.get("reply_step_id"), str)
                else ""
            ),
            "onboarding_request_id": "",
            "onboarding_origin_event_id": origin_event_id,
            "channel": channel.strip(),
            "tool_name": tool_name.strip() if isinstance(tool_name, str) else "",
            "expires_at": self.loop.time() + ONBOARDING_OUTBOUND_CONTEXT_TTL_SECONDS,
        }
        self._pending_onboarding_outbound = pending

    def set_pending_onboarding_request_id(self, request_id: str) -> None:
        if self._pending_onboarding_outbound:
            self._pending_onboarding_outbound["onboarding_request_id"] = request_id

    def clear_pending_onboarding_outbound(self, step_id: str | None = None) -> None:
        if not self._pending_onboarding_outbound:
            return
        if (
            step_id
            and self._pending_onboarding_outbound.get(
                "onboarding_trigger_step_id",
            )
            != step_id
        ):
            return
        self._pending_onboarding_outbound = None

    def note_whatsapp_window_open(self, contact_id: int | None, is_open: bool) -> None:
        """Record the latest known WhatsApp free-form window state for a contact."""
        if contact_id is None:
            return
        self._whatsapp_window_open[int(contact_id)] = bool(is_open)

    def whatsapp_window_state(self, contact_id: int | None) -> bool | None:
        """Return True/False for the contact's WhatsApp window, or None if unknown.

        A pending template resend is authoritative proof the window is closed
        (the last send fell back to a placeholder and no reply has reopened it).
        Otherwise fall back to the last observed/seeded state.
        """
        if contact_id is None:
            return None
        cid = int(contact_id)
        if cid in self._pending_whatsapp_resends:
            return False
        return self._whatsapp_window_open.get(cid)

    async def seed_whatsapp_window(self, contact_id: int) -> None:
        """Best-effort: ask the gateway whether a contact's window is open.

        Used at startup so the brain knows up front whether a first send will
        deliver verbatim or only a placeholder. Failures are swallowed — the
        state simply stays unknown and the send_whatsapp docstring falls back to
        its window-agnostic guidance.
        """
        contact = self._get_contact_safe(contact_id)
        whatsapp_number = (contact or {}).get("whatsapp_number")
        if not whatsapp_number:
            return
        try:
            from unify.conversation_manager.domains import comms_utils

            is_open = await comms_utils.get_whatsapp_window(whatsapp_number)
        except Exception:
            is_open = None
        if is_open is not None:
            self.note_whatsapp_window_open(contact_id, is_open)

    def _get_contact_safe(self, contact_id: int) -> dict | None:
        try:
            return self.contact_index.get_contact(contact_id)
        except Exception:
            return None

    def stash_pending_whatsapp_resend_onboarding_metadata(
        self,
        contact_id: int,
        metadata: dict[str, str],
    ) -> None:
        if metadata:
            self._pending_whatsapp_resend_onboarding_metadata[contact_id] = dict(
                metadata,
            )

    def consume_pending_whatsapp_resend_onboarding_metadata(
        self,
        contact_id: int,
    ) -> dict[str, str] | None:
        return self._pending_whatsapp_resend_onboarding_metadata.pop(contact_id, None)

    def consume_pending_onboarding_outbound(self, medium: str) -> dict[str, str] | None:
        pending = self._pending_onboarding_outbound
        if not pending:
            return None
        if self.loop.time() > float(pending.get("expires_at", 0)):
            self._pending_onboarding_outbound = None
            return None
        expected_media = {
            "email": {"email"},
            "sms_message": {"sms_message"},
            "whatsapp_message": {"whatsapp_message"},
            "whatsapp_call": {"whatsapp_call"},
            "phone_call": {"phone_call"},
            "slack_message": {"slack_message", "slack_channel_message"},
            "discord_message": {"discord_message", "discord_channel_message"},
            "ms_teams_message": {"ms_teams_bot_message"},
        }.get(str(pending.get("channel", "")), set())
        if medium not in expected_media:
            return None
        self._pending_onboarding_outbound = None
        return {
            key: value
            for key in (
                "onboarding_trigger_step_id",
                "onboarding_reply_step_id",
                "onboarding_request_id",
                "onboarding_origin_event_id",
            )
            if isinstance((value := pending.get(key)), str) and value
        }

    def build_whatsapp_call_sent_event(self, contact: dict) -> WhatsAppCallSent:
        """Publish-ready outbound WhatsApp call event with onboarding metadata."""
        kwargs = self.consume_pending_onboarding_outbound("whatsapp_call") or {}
        return WhatsAppCallSent(contact=contact, **kwargs)

    async def store_chat_history(self):
        if len(self.chat_history) >= 2:
            await self.event_broker.publish(
                "app:comms:chat_history",
                StoreChatHistory(chat_history=self.chat_history[-2:]).to_json(),
            )
            await asyncio.sleep(2)

    async def _retire_in_flight_actions(self) -> None:
        """Discard the in-flight action registry the way a pod exit does.

        Idle retirement discards a parked session's Python state by exiting
        the process; an in-process retirement must do that discarding itself,
        and it must not wait on work that may never finish — a persist-mode
        act parked for an interjection, or a provider call that has hung.
        Every handle gets one stop request under a single shared grace
        period; whatever ignores it is abandoned along with the registry.
        """
        stops = []
        for handle_data in list(self.in_flight_actions.values()):
            handle = handle_data.get("handle")
            if handle is None:
                continue
            if hasattr(handle, "trigger_completion"):
                handle.trigger_completion()
            else:
                # Stopping an already-finished handle is a no-op, so no
                # done() probe — its signature varies across handle types.
                stops.append(
                    asyncio.ensure_future(handle.stop(reason="session retired")),
                )
        if stops:
            done, pending = await asyncio.wait(stops, timeout=5.0)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    self._session_logger.info(
                        "session_end",
                        f"In-flight action stop failed during retirement: "
                        f"{task.exception()!r}",
                    )
            for task in pending:
                task.cancel()
            if pending:
                self._session_logger.info(
                    "session_end",
                    f"Abandoned {len(pending)} in-flight action(s) that did "
                    "not stop within the retirement grace period",
                )
        self.in_flight_actions.clear()
        self.completed_actions.clear()

    async def cleanup(self):
        """Clean up any running call processes and file sync."""
        await self._retire_in_flight_actions()
        await self.store_chat_history()
        refresh_task = self._coordinator_state_refresh_task
        if refresh_task is not None and not refresh_task.done():
            refresh_task.cancel()
        state_client = self._coordinator_state_http
        if state_client is not None and not state_client.is_closed:
            await state_client.aclose()
        local_ingress = getattr(self, "_local_comms_ingress", None)
        if local_ingress is not None:
            await local_ingress.stop()
        activation_materializer = getattr(self, "_activation_materializer", None)
        if activation_materializer is not None:
            try:
                await activation_materializer.stop()
            except Exception as exc:
                LOGGER.warning(
                    f"{DEFAULT_ICON} [ConversationManager] "
                    f"Failed to stop activation materializer: {exc}",
                )
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
            from unify.file_manager.managers.local import LocalFileManager

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

    def _get_voice_profiles(self, contact_ids: list[int]) -> dict[int, list[float]]:
        """Return enrolled voice embeddings for the given contacts."""
        if self.contact_manager is None:
            return {}
        return self.contact_manager.get_voice_profiles(contact_ids)

    def _on_fast_brain_generating(self) -> dict[str, bool]:
        """Called via IPC when the fast brain starts generating a reply.

        Restarts the proactive speech cycle so any in-flight decision is
        cancelled.  The quiescence gate in ``_proactive_speech_loop`` will
        prevent the countdown from starting until the pipeline is idle again.
        """
        if self._proactive_speech_enabled:
            asyncio.ensure_future(self.schedule_proactive_speech())
        return {"idle_smalltalk_allowed": self.fast_brain_idle_smalltalk_allowed()}

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

        if not self._proactive_speech_enabled:
            return

        # While the hang-up gate is armed, silence is the close signal the
        # voice agent's own watcher acts on — proactive filler would keep a
        # finished conversation alive indefinitely.
        if self.call_manager.hang_up_gate_reason is not None:
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

    async def set_proactive_speech_enabled(self, enabled: bool):
        self._proactive_speech_enabled = enabled
        if not enabled:
            self._proactive_speech_gen += 1
            await self.cancel_proactive_speech()

    async def stop_in_flight_action_by_calling_id(
        self,
        calling_id: str,
        *,
        reason: str = "",
    ) -> bool:
        """Stop the in-flight act whose ManagerMethod ``calling_id`` matches.

        Console Live Actions identifies roots by EventBus ``calling_id`` (UUID).
        CM steering uses integer ``handle_id``; this bridges the two.
        """
        if not calling_id:
            return False

        for handle_id, handle_data in list(self.in_flight_actions.items()):
            handle = handle_data.get("handle")
            stored = handle_data.get("calling_id") or getattr(
                handle,
                "_manager_call_id",
                None,
            )
            if stored != calling_id:
                continue

            stop_reason = reason or "Stopped from Console Actions pane."
            handle_data.setdefault("handle_actions", []).append(
                {
                    "action_name": f"stop_{handle_id}",
                    "query": stop_reason,
                    "timestamp": prompt_now(),
                },
            )
            if handle is not None:
                await handle.stop(reason=stop_reason)
            stopped = self.in_flight_actions.pop(handle_id, None)
            if stopped is not None:
                self.completed_actions[handle_id] = stopped
            return True
        return False

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
                    "google_meet": "Google Meet Shared Screen",
                    "teams_meet": "Microsoft Teams Shared Screen",
                }
                content_parts: list[dict] = []
                for source, entry in latest_by_source.items():
                    label = source_labels.get(source, "Screenshot")
                    if entry.attribution:
                        label = f"{label} - shared by {entry.attribution}"
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

            # Nothing has been said yet — there is no silence to break.  The
            # cycle re-arms when a real utterance arrives.
            if not conversation_turns:
                return

            snapshot_state = self.prompt_renderer.render_state(
                self.contact_index,
                self.notifications_bar,
                self.in_flight_actions,
                self.completed_actions,
                self.last_snapshot,
                recent_tool_executions=self._recent_tool_executions,
                assistant_screen_share_active=self.assistant_screen_share_active,
                user_screen_share_active=self.user_screen_share_active,
                user_webcam_active=self.user_webcam_active,
                user_remote_control_active=self.user_remote_control_active,
                google_meet_active=self.call_manager.has_active_google_meet,
                teams_meet_active=self.call_manager.has_active_teams_meet,
                meet_screen_share_active=self.meet_screen_share_active,
                vm_ready=self.vm_ready,
                file_sync_complete=self.file_sync_complete,
                has_desktop=SESSION_DETAILS.assistant.has_managed_desktop,
                console_open=self.console_is_open(),
                console_action_catalogue=self.console_action_catalogue(),
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
                other_participants=tuple(
                    self.call_manager.other_call_participant_names,
                ),
                peer_assistants=tuple(self.call_manager.other_call_assistant_names),
            )

            if _superseded():
                return

            _log.proactive_decision(
                decision.delay,
                decision.content,
            )

            # Wait the requested delay (cancellable if an utterance arrives).
            # `delay` is unbounded: a few seconds when someone is waiting on a
            # reply, many minutes during a focused collaborative silence.
            if decision.delay > 0:
                _log.proactive_speaking(decision.delay, decision.content)
                await asyncio.sleep(decision.delay)

            if _superseded():
                return

            # The hang-up gate may have armed while the decision was in flight
            # (scheduling-time suppression cannot see that): silence is now the
            # close signal the voice agent's watcher acts on, so the pending
            # line is dropped rather than delivered.
            if self.call_manager.hang_up_gate_reason is not None:
                _log.proactive_deferred("hang-up gate armed")
                return

            # Do not pre-write to contact_index here: the line is recorded once,
            # via the actually-spoken Outbound utterance, only if it is genuinely
            # spoken (the fast brain discards proactive speech when the pipeline
            # is not quiescent). Pre-writing duplicated that record and logged
            # lines that were never said.
            contact = self.get_active_contact()

            event = FastBrainNotification(
                contact=contact or {},
                message=decision.content,
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
