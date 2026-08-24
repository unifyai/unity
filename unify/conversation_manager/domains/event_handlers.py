import asyncio
import json
import os
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from unify.contact_manager.types.contact import UNASSIGNED
from unify.common.context_registry import ContextRegistry
from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.conversation_manager import assistant_jobs
from unify.conversation_manager.events import *
from unify.conversation_manager.domains import managers_utils
from unify.conversation_manager.domains.canvas_actions import (
    handle_canvas_invocation_requested,
)
from unify.conversation_manager.domains.workflow_requests import (
    arm_workflows_for_connected_app,
    handle_workflow_request_requested,
)
from unify.conversation_manager.domains.comms_utils import (
    post_call_utterances_to_orchestra,
    publish_system_error,
    publish_voice_enrollment_suggested,
)
from unify.conversation_manager.domains.coordinator_onboarding import (
    _handle_coordinator_onboarding_event,
)
from unify.conversation_manager.domains.coordinator_delegate import (
    _handle_coordinator_delegate_event,
)
from unify.conversation_manager.domains.integration_sync import (
    _handle_integration_tools_sync_completed,
    _handle_integration_tools_sync_failed,
    _handle_integration_tools_sync_requested,
    _schedule_startup_integration_sync,
)
from unify.conversation_manager.domains.self_host_desktop import (
    apply_managed_desktop_ready,
    bootstrap_managed_desktop_on_startup,
    resolve_desktop_urls,
)
from unify.conversation_manager.domains.task_execution import (
    _consume_startup_wake_reasons,
    _handle_provider_event_dispatch_requested_event,
    _handle_task_due_event,
    _handle_task_trigger_requested_event,
    _surface_trigger_task_candidates,
)
from unify.conversation_manager.domains.console_script_result import (
    handle_console_script_result,
)
from unify.conversation_manager.domains.whatsapp_history import (
    whatsapp_sent_history_content,
)
from unify.memory_manager import broader_context
from unify.task_scheduler.machine_state import invalidate_task_machine_state_reads
from unify.conversation_manager.cm_types import Medium, Mode
from unify.common.startup_timing import log_startup_timing
from unify.logger import LOGGER
from unify.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from unify.conversation_manager.conversation_manager import ConversationManager


_AGENT_SERVICE_PID_FILE = Path("/tmp/agent-service.pid")
_AGENT_SERVICE_LOG = Path("/var/log/unity/agent-service.log")
# Upper bound on how long a restart waits for the respawned Node process to bind
# port 3000 before returning. Closes the race where a meet-join POST arrives in
# the cold-start window and is refused. Runs in a worker thread, so a blocking
# poll is fine.
_AGENT_SERVICE_READY_TIMEOUT_S = 30.0
_CALL_NOT_ANSWERED_REASON_DISPLAY = {
    "no-answer": "did not answer",
    "busy": "was busy",
    "canceled": "call was canceled",
    "failed": "call failed",
}


def _find_agent_service_dir() -> Path | None:
    """Locate the agent-service directory (Docker or local dev)."""
    candidates = [
        Path("/app/agent-service"),
        Path(__file__).resolve().parents[3] / "agent-service",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _update_env_file(env_path: Path, key: str, value: str) -> None:
    """Update or add a key=value line in a .env file, preserving other lines."""
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    prefix = f"{key}="
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")


def _find_pid_on_port(port: int) -> int | None:
    """Find the PID of the process listening on the given TCP port via /proc."""
    hex_port = f"{port:04X}"
    inode = None
    for tcp_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(tcp_path).read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) < 10:
                    continue
                # 0A = LISTEN state
                if fields[3] == "0A" and fields[1].endswith(f":{hex_port}"):
                    inode = fields[9]
                    break
        except FileNotFoundError:
            continue
        if inode:
            break
    if not inode:
        return None

    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            for fd in (pid_dir / "fd").iterdir():
                if os.readlink(str(fd)) == f"socket:[{inode}]":
                    return int(pid_dir.name)
        except (OSError, PermissionError):
            continue
    return None


def _kill_port(port: int) -> bool:
    """Kill the process listening on the given TCP port. Returns True if killed."""
    pid = _find_pid_on_port(port)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False


def _restart_agent_service_with_key(api_key: str) -> None:
    """Kill and restart the agent-service so it picks up the user's API key.

    ``entrypoint.sh`` starts the service with the container's own key as part of
    the desktop substrate, so this is a genuine respawn rather than a cold start.
    Do not treat it as the only start path: offline task pods have no
    ConversationManager and would otherwise have nothing listening on 3000.

    1. Updates the agent-service .env file (UNIFY_KEY + infrastructure URLs)
    2. Kills the process listening on port 3000
    3. Spawns a new agent-service with logs to /var/log/unity/agent-service.log
    4. Writes the new PID to /tmp/agent-service.pid for entrypoint.sh cleanup
    """
    from unify.settings import SETTINGS

    try:
        agent_dir = _find_agent_service_dir()
        if agent_dir is None:
            LOGGER.warning(
                "[agent-service-restart] agent-service directory not found, skipping",
            )
            return

        # 1. Update .env
        env_file = agent_dir / ".env"
        _update_env_file(env_file, "UNIFY_KEY", api_key)
        _update_env_file(env_file, "ORCHESTRA_URL", SETTINGS.ORCHESTRA_URL)
        _update_env_file(env_file, "UNIFY_COMMS_URL", SETTINGS.conversation.COMMS_URL)
        LOGGER.info(
            "[agent-service-restart] Updated agent-service .env",
        )

        # 2. Kill whatever is listening on port 3000 and wait for it to release
        killed = _kill_port(3000)
        if killed:
            for _ in range(50):
                time.sleep(0.1)
                if _find_pid_on_port(3000) is None:
                    break
        LOGGER.info(
            (
                "[agent-service-restart] Killed process on port 3000"
                if killed
                else "[agent-service-restart] No process found on port 3000"
            ),
        )

        # 3. Restart
        compiled = agent_dir / "dist" / "index.js"
        if compiled.exists():
            cmd = ["node", "--no-deprecation", str(compiled)]
        else:
            cmd = ["npx", "ts-node", str(agent_dir / "src" / "index.ts")]

        log_fh = open(_AGENT_SERVICE_LOG, "a")
        proc = subprocess.Popen(
            cmd,
            cwd=str(agent_dir),
            env={
                **os.environ,
                "UNIFY_KEY": api_key,
                "PLAYWRIGHT_BROWSERS_PATH": "/home/unity/.cache/ms-playwright",
            },
            stdout=log_fh,
            stderr=log_fh,
        )

        # 4. Write PID file for entrypoint.sh cleanup
        _AGENT_SERVICE_PID_FILE.write_text(str(proc.pid))
        LOGGER.info(
            f"[agent-service-restart] Restarted agent-service (PID: {proc.pid})",
        )

        # 5. Block until the respawned process is actually listening on 3000
        # (LISTEN state) before returning, so a meet-join POST that fires right
        # after this restart doesn't hit a dead socket. Bounded so a wedged
        # start surfaces rather than hanging.
        deadline = time.monotonic() + _AGENT_SERVICE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if _find_pid_on_port(3000) is not None:
                LOGGER.info(
                    "[agent-service-restart] agent-service listening on 3000 "
                    f"(PID: {proc.pid})",
                )
                break
            if proc.poll() is not None:
                LOGGER.warning(
                    "[agent-service-restart] agent-service exited during startup "
                    f"(code {proc.returncode})",
                )
                break
            time.sleep(0.2)
        else:
            LOGGER.warning(
                "[agent-service-restart] agent-service not listening on 3000 "
                f"within {_AGENT_SERVICE_READY_TIMEOUT_S:.0f}s",
            )
    except Exception as e:
        LOGGER.info(f"[agent-service-restart] Failed (non-fatal): {e}")


def _event_type_to_log_key(event_cls) -> str:
    """Convert an event class name to a log key for icon lookup."""
    name = event_cls.__name__

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _get_sender_name(contact: dict | None, fallback: str = "Unknown") -> str:
    """Get display name from contact dict."""
    if not contact:
        return fallback
    first = contact.get("first_name") or ""
    last = contact.get("surname") or ""
    name = f"{first} {last}".strip()
    return name or fallback


def _adopt_slack_bot_user_id(cm: "ConversationManager", event) -> None:
    """Enable Slack send capability from an inbound Slack event.

    Slack's ``bot_user_id`` is workteam-scoped (it lives on the
    ``slack_installs`` row), not on the assistant record, so the
    per-assistant ``assistant_slack_bot_user_id`` is never populated at
    session bootstrap. The brain gates the ``send_slack_message`` /
    ``send_slack_channel_message`` tools (and the Slack reply guidelines)
    on that flag, so without it the assistant can never reply on Slack.
    Adopt the workspace bot id from the inbound event so this session's
    subsequent brain run exposes the Slack tools.
    """
    inbound_bot_user_id = getattr(event, "bot_user_id", "") or ""
    if inbound_bot_user_id and not getattr(cm, "assistant_slack_bot_user_id", ""):
        cm.assistant_slack_bot_user_id = inbound_bot_user_id


def _active_voice_thread_medium(cm: "ConversationManager") -> Medium:
    """Return the active voice thread that should receive call-context messages."""

    if cm.call_manager.has_active_google_meet:
        return Medium.GOOGLE_MEET
    if cm.call_manager.has_active_teams_meet:
        return Medium.TEAMS_MEET
    if cm.mode == Mode.MEET:
        return Medium.UNIFY_MEET
    if cm.call_manager._call_channel == "whatsapp_call":
        return Medium.WHATSAPP_CALL
    return Medium.PHONE_CALL


async def _start_session_recording(event, cm: "ConversationManager") -> None:
    """Request a recording for the LiveKit room backing a just-started session.

    Browser meets are included. They were excluded while their audio was
    bridged through a pod-local audio device and never entered the LiveKit
    room, leaving the compositor nothing to mix; the meeting now arrives as an
    ordinary published track, so a room composite captures both sides.

    Best-effort throughout -- a recording problem must never disturb a live
    call, so every failure is logged and swallowed.
    """
    from unify.settings import SETTINGS

    if (
        SETTINGS.conversation.LOCAL_COMMS_ENABLED
        or SETTINGS.conversation.LOCAL_COMMS_MODE == "local"
    ):
        # Self-host owns recording in-process: its egress carries a
        # ``/local/livekit/recording-complete`` callback that only the local
        # ingress serves, whereas the gateway points completion webhooks at
        # ADAPTERS_URL. Routing through the gateway here would upload a
        # recording whose completion event nobody receives.
        return

    call_manager = cm.call_manager
    room_name = call_manager.room_name
    if not room_name:
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [Recording] No room name on the active session; "
            "skipping recording request",
        )
        return

    if isinstance(event, UnifyMeetStarted):
        call_session_id = call_manager.unify_meet_call_session_id or (
            event.call_session_id or ""
        )
        provider_call_sid = ""
        conference_name = ""
    elif isinstance(event, (GoogleMeetStarted, TeamsMeetStarted)):
        # Same key the utterances are written under, so the finished recording
        # resolves to the transcript of the meeting it came from. Browser meets
        # have no telephony identifiers.
        call_session_id = call_manager.meet_session_id
        provider_call_sid = ""
        conference_name = ""
    else:
        call_session_id = call_manager.call_session_id
        provider_call_sid = call_manager.provider_call_sid
        conference_name = call_manager.conference_name

    from unify.conversation_manager.utils import start_call_recording

    try:
        await asyncio.to_thread(
            start_call_recording,
            room_name,
            str(call_manager.assistant_id or ""),
            user_id=str(call_manager.user_id or ""),
            call_session_id=call_session_id,
            provider_call_sid=provider_call_sid,
            conference_name=conference_name,
        )
    except Exception as exc:
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [Recording] Failed to request recording for "
            f"room {room_name}: {exc}",
        )


def _call_not_answered_reason_text(reason: str) -> str:
    """Return the user-facing text for one telephony not-answered reason."""

    return _CALL_NOT_ANSWERED_REASON_DISPLAY.get(
        reason,
        f"not answered ({reason})",
    )


def _meet_join_failure_texts(
    meet_label: str,
    join_tool: str,
    reason: str | None,
) -> tuple[str, str]:
    """Build (user_text, llm_text) for a failed browser-meet join.

    ``user_text`` is a first-person, user-facing chat message (no tool names),
    sent so the person who shared the link isn't left in silence after the
    "joining now" ack and can ask for a retry. ``llm_text`` carries the
    machine detail (failure reason + the exact retry tool) into the transcript
    the brain reads, so it can retry with the right tool.

    When the agent-service reports a guest-admission block (``*_join_blocked``),
    the host was almost certainly not present to admit the guest, so say exactly
    that and what to do instead of a bare retry line.
    """

    reason_text = (reason or "").strip()
    if "join_blocked" in reason_text:
        user_text = (
            f"Couldn't get into the {meet_label} — the host wasn't in the "
            "meeting to let me in, so I was turned away at the door. Start the "
            "meeting and stay in it, then re-share the link and I'll rejoin."
        )
        llm_text = (
            f"Couldn't join the {meet_label}: the host wasn't in the meeting to "
            "let me in, so it turned me away at the door. Ask the user to start "
            "the meeting and stay in it, then re-share the link and I'll rejoin "
            f"with {join_tool}."
        )
        return user_text, llm_text

    user_text = (
        f"I couldn't get into the {meet_label}. Re-share the link and I'll try "
        "again."
    )
    llm_text = (
        f"Failed to join {meet_label}. You may retry by calling {join_tool} again."
    )
    if reason_text:
        llm_text = f"{llm_text} (reason: {reason_text})"
    return user_text, llm_text


class EventHandler:
    """Registry that maps event classes to their async handlers."""

    _registry: dict[type[Event], Any] = {}

    @classmethod
    def register(cls, event_cls: list[Event] | Event):
        """Register one handler for one or more event classes."""

        def wrapper(func):
            events_classes = (
                [event_cls] if not isinstance(event_cls, (list, tuple)) else event_cls
            )
            for e in events_classes:
                cls._registry[e] = func
            return func

        return wrapper

    @classmethod
    def handle_event(cls, event: Event, cm: "ConversationManager", *args, **kwargs):
        """Dispatch one event to its registered handler."""

        event_key = _event_type_to_log_key(event.__class__)
        if (
            hasattr(cm, "_session_logger")
            and not event.__class__.content_logged
            and event.__class__.loggable
        ):
            log_fn = (
                cm._session_logger.info
                if event.__class__.prominent
                else cm._session_logger.debug
            )
            log_fn(
                event_key,
                f"Event: {event.__class__.__name__}",
            )

        if event.__class__.loggable:
            asyncio.create_task(
                managers_utils.queue_operation(
                    managers_utils.publish_bus_events,
                    event,
                ),
            )

        f = cls._registry.get(event.__class__)
        if not f:
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(Ping)
async def _(event: Ping, cm: "ConversationManager", *args, **kwargs):
    log_str = "Ping received - keeping conversation manager alive"
    cm._session_logger.debug("ping", log_str)


@EventHandler.register((PhoneCallAnswered, WhatsAppCallAnswered))
async def _(
    event: PhoneCallAnswered | WhatsAppCallAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """
    Forward call answered status to the voice agent subprocess.

    This event arrives from the telephony system (via GCP PubSub) when the contact
    picks up for outbound calls. Forward it unconditionally - if this event arrives,
    we're in a call context by definition.
    """
    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "call_answered"}),
    )


CallInitEvents = Union[
    PhoneCallReceived,
    PhoneCallSent,
    UnifyMeetReceived,
    WhatsAppCallReceived,
    WhatsAppCallSent,
]


def _apply_stashed_whatsapp_call_context(
    cm: "ConversationManager",
    stashed: dict | str | None,
) -> None:
    """Apply a stashed WhatsApp call context to the call manager.

    The per-contact stash (``_pending_whatsapp_call_openers``) holds a dict
    with ``opener`` / ``briefing`` / ``hang_up_gate`` keys; the server-side
    pending-intent fallback yields a plain opener string. The call manager's
    pending fields are plain strings — assigning the dict wholesale crashes
    ``start_call`` when it strips the opener.
    """
    if isinstance(stashed, dict):
        cm.call_manager.pending_opener = stashed.get("opener") or ""
        cm.call_manager.pending_briefing = stashed.get("briefing") or ""
        cm.call_manager.pending_hang_up_gate = stashed.get("hang_up_gate") or ""
    elif isinstance(stashed, str):
        cm.call_manager.pending_opener = stashed


@EventHandler.register(
    (
        PhoneCallReceived,
        PhoneCallSent,
        UnifyMeetReceived,
        WhatsAppCallReceived,
        WhatsAppCallSent,
    ),
)
async def _(event: CallInitEvents, cm: "ConversationManager", *args, **kwargs):
    """
    Handle incoming/outgoing call initiation - spawn voice agent subprocess.
    """
    # Don't start a new call if we're already in voice mode. Unify Meet is the
    # exception: the browser can remain in the room while an assistant job drops,
    # so a fresh UnifyMeetReceived is allowed to redispatch the agent when the
    # previous job has already been cleaned up. Mid-call roster refreshes
    # (late human join / peer assistant add) update attribution without
    # restarting the voice job.
    if cm.mode.is_voice:
        if (
            isinstance(event, UnifyMeetReceived)
            and cm.mode == Mode.MEET
            and cm.call_manager.has_active_call
            and event.participants
            # Only the ACTIVE call's dispatches are roster refreshes. A
            # dispatch for a different session (e.g. a new call created
            # while the previous one is still tearing down) must not be
            # swallowed as a refresh of the dying call.
            and event.call_session_id == cm.call_manager.unify_meet_call_session_id
        ):
            await cm.call_manager.refresh_unify_meet_roster(event.participants)
            return
        if not (
            isinstance(event, UnifyMeetReceived)
            and cm.mode == Mode.MEET
            and not cm.call_manager.has_active_call
        ):
            return

    # Past the guards above this is a new call, so its shared surfaces start
    # closed. Also reset on hangup, but this is the one that holds: a frontend
    # event landing during the previous teardown re-dirties the state after
    # that reset has already run.
    cm.reset_meet_surfaces()

    boss_contact_id = SESSION_DETAILS.boss_contact_id
    boss = (
        cm.contact_index.get_contact(contact_id=int(boss_contact_id))
        if boss_contact_id is not None
        else None
    )
    if isinstance(event, UnifyMeetReceived):
        contact = event.contact or boss
    elif isinstance(event, (WhatsAppCallReceived, WhatsAppCallSent)):
        contact = cm.contact_index.get_contact(
            whatsapp_number=event.contact.get("whatsapp_number"),
        )
        if contact is None:
            contact = event.contact
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )
        if contact is None:
            contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else event.contact.get("contact_id")
    )
    sender_name = _get_sender_name(contact)

    # Inject stashed opener from make_whatsapp_call if a pending permission
    # grant triggered this inbound WhatsApp call.
    if isinstance(event, WhatsAppCallReceived) and contact_id is not None:
        stashed_opener = cm._pending_whatsapp_call_openers.pop(contact_id, None)
        if stashed_opener:
            _apply_stashed_whatsapp_call_context(cm, stashed_opener)

    if isinstance(event, WhatsAppCallSent) and contact_id is not None:
        if not cm.call_manager.pending_opener:
            stashed_opener = cm._pending_whatsapp_call_openers.pop(contact_id, None)
            if stashed_opener:
                _apply_stashed_whatsapp_call_context(cm, stashed_opener)
            else:
                from unify.conversation_manager.domains import comms_utils

                pool_number = (
                    getattr(cm, "assistant_whatsapp_number", "")
                    or getattr(SESSION_DETAILS.assistant, "whatsapp_number", "")
                    or ""
                )
                whatsapp_number = (contact or {}).get("whatsapp_number") or ""
                if pool_number and whatsapp_number:
                    try:
                        intent = await comms_utils.get_pending_whatsapp_call_intent(
                            pool_number=pool_number,
                            contact_number=whatsapp_number,
                        )
                    except Exception as exc:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to load pending WhatsApp call intent: {exc}",
                        )
                        intent = None
                    opener = (intent or {}).get("context")
                    if opener:
                        _apply_stashed_whatsapp_call_context(cm, opener)
                        try:
                            await comms_utils.clear_pending_whatsapp_call_intent(
                                pool_number=pool_number,
                                contact_number=whatsapp_number,
                            )
                        except Exception as exc:
                            LOGGER.error(
                                f"{DEFAULT_ICON} Failed to clear pending WhatsApp call intent: {exc}",
                            )
        cm.call_manager._whatsapp_call_joining = True

    match event:
        case PhoneCallReceived() as e:
            cm.call_manager.conference_name = e.conference_name
            cm.call_manager.call_session_id = e.call_session_id or ""
            cm.call_manager.provider_call_sid = e.provider_call_sid or ""
            await cm.call_manager.start_call(contact, boss, room_name=e.room_name)
            message_content = "<Recvieving Call...>"
            notif_content = f"Call received from {sender_name}"
        case WhatsAppCallReceived() as e:
            cm.call_manager.conference_name = e.conference_name
            cm.call_manager.call_session_id = e.call_session_id or ""
            cm.call_manager.provider_call_sid = e.provider_call_sid or ""
            await cm.call_manager.start_call(
                contact,
                boss,
                channel="whatsapp_call",
                room_name=e.room_name,
            )
            message_content = "<Receiving WhatsApp Call...>"
            notif_content = f"WhatsApp call received from {sender_name}"
        case WhatsAppCallSent():
            await cm.call_manager.start_call(
                contact,
                boss,
                outbound=True,
                channel="whatsapp_call",
            )
            message_content = "<Sending WhatsApp Call...>"
            notif_content = f"WhatsApp call sent to {sender_name}"
        case PhoneCallSent() as e:
            cm.call_manager.provider_call_sid = e.provider_call_sid or ""
            await cm.call_manager.start_call(contact, boss, outbound=True)
            message_content = "<Sending Call...>"
            notif_content = f"Call sent to {sender_name}"
        case UnifyMeetReceived() as e:
            # The owner answered; cancel any pending no-answer text fallback.
            cm._pending_meet_ring = None
            await cm.call_manager.start_unify_meet(
                e.contact or contact,
                boss,
                e.room_name,
                opening_config=e.opening_config,
                call_session_id=e.call_session_id,
                participants=e.participants,
            )
            message_content = "<Recieving Call...>"
            notif_content = f"Call received from {sender_name}"

    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)
    role = "user" if "received" in event.__class__.__name__.lower() else "assistant"
    medium = (
        Medium.UNIFY_MEET
        if isinstance(event, UnifyMeetReceived)
        else (
            Medium.WHATSAPP_CALL
            if isinstance(event, (WhatsAppCallReceived, WhatsAppCallSent))
            else Medium.PHONE_CALL
        )
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=message_content,
        role=role,
        timestamp=event.timestamp,
    )
    if isinstance(event, (PhoneCallSent, WhatsAppCallSent)):
        await managers_utils.queue_operation(managers_utils.log_message, cm, event)
    if role == "user" and await _surface_trigger_task_candidates(
        cm=cm,
        event=event,
        medium=medium,
        contact_id=contact_id,
        sender_name=sender_name,
        timestamp=event.timestamp,
    ):
        cm.record_last_inbound_reply(
            _credit_gate_reply_context(
                medium=medium,
                contact_id=contact_id,
            ),
        )
        await cm.request_llm_run(
            delay=0,
            triggering_contact_id=contact_id,
        )


def _meet_join_ack(meet_label: str) -> str:
    """User-facing acknowledgement sent the moment a browser-meet join starts.

    The browser join is slow (headless Chromium cold start + LLM-guided
    click-through of the pre-join screen, then a possible wait in the lobby),
    and the success/lobby message only lands once all of that completes.
    Without an immediate ack the user who just shared the link is left staring
    at nothing, unsure anything is happening. The handler owns this message and
    sends it deterministically for every assistant (general or coordinator), so
    the LLM must not announce the join itself — that split ownership produced a
    duplicate "joining" reply for general assistants and none for coordinators.
    """
    return (
        f"Getting into the {meet_label} now — give me a moment, joining can "
        "take a little while."
    )


def _meet_join_notif(meet_label: str, in_lobby: bool) -> tuple[str, str]:
    """Notification + transcript text for a successful browser-meet join.

    A join that lands in the waiting room is still a success — the browser has
    knocked ("Ask to join") and is waiting for the host to admit it — so word
    it as waiting to be let in rather than already being in the call. This
    keeps the LLM from treating a lobby state as a join failure. The "joining"
    ack has already been sent when the attempt started, so the non-lobby case
    reports arrival ("I'm in") rather than repeating "joining".
    """
    if in_lobby:
        notif = (
            f"In the {meet_label} waiting room — I've asked to join and am "
            "waiting for the host to let me in."
        )
        return notif, f"<Waiting in the {meet_label} waiting room to be admitted...>"
    return f"I'm in the {meet_label} now.", f"<Joined the {meet_label}.>"


@EventHandler.register(GoogleMeetReceived)
async def _(
    event: GoogleMeetReceived,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle request to join a Google Meet — spawn browser + audio bridge."""
    # Backstop for the tool-surface gate: a single-player twin never enters
    # a browser meeting, whatever path the request arrived through.
    if SESSION_DETAILS.is_private_coordinator:
        return
    if (
        cm.mode.is_voice
        or cm.call_manager.has_active_call
        or cm.call_manager.has_active_google_meet
        or cm.call_manager.has_active_teams_meet
    ):
        return

    boss_contact_id = SESSION_DETAILS.boss_contact_id
    boss = (
        cm.contact_index.get_contact(contact_id=int(boss_contact_id)) or {}
        if boss_contact_id is not None
        else {}
    )
    contact = boss

    contact_id = contact.get("contact_id") if contact else boss_contact_id
    sender_name = _get_sender_name(contact)

    # Acknowledge immediately, before the slow join: the browser cold-start +
    # click-through can take a minute, and the success/lobby message only fires
    # once that completes. The handler sends this deterministically (not the
    # LLM) so every assistant acknowledges exactly once — see _meet_join_ack.
    from unify.conversation_manager.domains import comms_utils

    ack_msg = _meet_join_ack("Google Meet")
    await comms_utils.send_unify_message(content=ack_msg, contact_id=contact_id)
    cm.notifications_bar.push_notif("Comms", ack_msg, event.timestamp)
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.GOOGLE_MEET,
        message_content=ack_msg,
        role="assistant",
        timestamp=event.timestamp,
    )

    joined = await cm.call_manager.start_google_meet(
        meet_url=event.meet_url,
        contact=contact,
        boss=boss,
    )

    if joined:
        notif_text, thread_text = _meet_join_notif(
            "Google Meet",
            cm.call_manager.meet_lobby_waiting,
        )
        await comms_utils.send_unify_message(
            content=notif_text,
            contact_id=contact_id,
        )
        cm.notifications_bar.push_notif(
            "Comms",
            notif_text,
            event.timestamp,
        )
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.GOOGLE_MEET,
            message_content=thread_text,
            role="assistant",
            timestamp=event.timestamp,
        )
        if await _surface_trigger_task_candidates(
            cm=cm,
            event=event,
            medium=Medium.GOOGLE_MEET,
            contact_id=contact_id,
            sender_name=sender_name,
            timestamp=event.timestamp,
        ):
            await cm.request_llm_run(
                delay=0,
                triggering_contact_id=contact_id,
            )
    else:
        await _surface_trigger_task_candidates(
            cm=cm,
            event=event,
            medium=Medium.GOOGLE_MEET,
            contact_id=contact_id,
            sender_name=sender_name,
            timestamp=event.timestamp,
        )
        # Surface the failure as a real outbound chat (not just the internal
        # notifications bar) so the user sees it after the "Joining now…" ack
        # and can request a retry — mirrors the ack/success paths above.
        user_text, llm_text = _meet_join_failure_texts(
            "Google Meet",
            "join_google_meet",
            cm.call_manager.meet_join_failure_reason,
        )
        await comms_utils.send_unify_message(
            content=user_text,
            contact_id=contact_id,
        )
        cm.notifications_bar.push_notif("Comms", user_text, event.timestamp)
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.GOOGLE_MEET,
            message_content=llm_text,
            role="assistant",
            timestamp=event.timestamp,
        )
        await cm.request_llm_run(
            delay=0,
            triggering_contact_id=contact_id,
        )


@EventHandler.register(TeamsMeetReceived)
async def _(
    event: TeamsMeetReceived,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle request to join a Microsoft Teams meeting — spawn browser + audio bridge."""
    # Backstop for the tool-surface gate: a single-player twin never enters
    # a browser meeting, whatever path the request arrived through.
    if SESSION_DETAILS.is_private_coordinator:
        return
    if (
        cm.mode.is_voice
        or cm.call_manager.has_active_call
        or cm.call_manager.has_active_google_meet
        or cm.call_manager.has_active_teams_meet
    ):
        return

    boss_contact_id = SESSION_DETAILS.boss_contact_id
    boss = (
        cm.contact_index.get_contact(contact_id=int(boss_contact_id)) or {}
        if boss_contact_id is not None
        else {}
    )
    contact = boss

    contact_id = contact.get("contact_id") if contact else boss_contact_id
    sender_name = _get_sender_name(contact)

    # Acknowledge immediately, before the slow join (see Google Meet handler).
    from unify.conversation_manager.domains import comms_utils

    ack_msg = _meet_join_ack("Teams meeting")
    await comms_utils.send_unify_message(content=ack_msg, contact_id=contact_id)
    cm.notifications_bar.push_notif("Comms", ack_msg, event.timestamp)
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.TEAMS_MEET,
        message_content=ack_msg,
        role="assistant",
        timestamp=event.timestamp,
    )

    joined = await cm.call_manager.start_teams_meet(
        meet_url=event.meet_url,
        contact=contact,
        boss=boss,
    )

    if joined:
        notif_text, thread_text = _meet_join_notif(
            "Teams meeting",
            cm.call_manager.meet_lobby_waiting,
        )
        await comms_utils.send_unify_message(
            content=notif_text,
            contact_id=contact_id,
        )
        cm.notifications_bar.push_notif(
            "Comms",
            notif_text,
            event.timestamp,
        )
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.TEAMS_MEET,
            message_content=thread_text,
            role="assistant",
            timestamp=event.timestamp,
        )
        if await _surface_trigger_task_candidates(
            cm=cm,
            event=event,
            medium=Medium.TEAMS_MEET,
            contact_id=contact_id,
            sender_name=sender_name,
            timestamp=event.timestamp,
        ):
            await cm.request_llm_run(
                delay=0,
                triggering_contact_id=contact_id,
            )
    else:
        await _surface_trigger_task_candidates(
            cm=cm,
            event=event,
            medium=Medium.TEAMS_MEET,
            contact_id=contact_id,
            sender_name=sender_name,
            timestamp=event.timestamp,
        )
        # Surface the failure as a real outbound chat (not just the internal
        # notifications bar) so the user sees it after the "Joining now…" ack
        # and can request a retry — mirrors the ack/success paths above.
        user_text, llm_text = _meet_join_failure_texts(
            "Teams meeting",
            "join_teams_meet",
            cm.call_manager.meet_join_failure_reason,
        )
        await comms_utils.send_unify_message(
            content=user_text,
            contact_id=contact_id,
        )
        cm.notifications_bar.push_notif("Comms", user_text, event.timestamp)
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.TEAMS_MEET,
            message_content=llm_text,
            role="assistant",
            timestamp=event.timestamp,
        )
        await cm.request_llm_run(
            delay=0,
            triggering_contact_id=contact_id,
        )


@EventHandler.register(
    (
        PhoneCallStarted,
        UnifyMeetStarted,
        GoogleMeetStarted,
        TeamsMeetStarted,
        WhatsAppCallStarted,
    ),
)
async def _(
    event: (
        PhoneCallStarted
        | UnifyMeetStarted
        | GoogleMeetStarted
        | TeamsMeetStarted
        | WhatsAppCallStarted
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, UnifyMeetStarted):
        active_session = cm.call_manager.unify_meet_call_session_id
        if (
            event.call_session_id
            and active_session
            and event.call_session_id != active_session
        ):
            LOGGER.info(
                "Ignoring stale UnifyMeetStarted for session %s (active=%s)",
                event.call_session_id,
                active_session,
            )
            return

    if isinstance(event, WhatsAppCallStarted):
        cm.mode = Mode.CALL
        whatsapp_number = event.contact.get("whatsapp_number")
        contact = cm.contact_index.get_contact(whatsapp_number=whatsapp_number)
    elif isinstance(event, PhoneCallStarted):
        cm.mode = Mode.CALL
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    elif isinstance(event, (GoogleMeetStarted, TeamsMeetStarted)):
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)
    else:
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)

    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)

    cm.call_manager.call_contact = contact
    if isinstance(event, (PhoneCallStarted, WhatsAppCallStarted)):
        cm.call_manager.call_start_timestamp = event.timestamp
        label = "Phone Call" if isinstance(event, PhoneCallStarted) else "WhatsApp Call"
    elif isinstance(event, GoogleMeetStarted):
        cm.call_manager.google_meet_start_timestamp = event.timestamp
        label = "Google Meet"
    elif isinstance(event, TeamsMeetStarted):
        cm.call_manager.teams_meet_start_timestamp = event.timestamp
        label = "Teams Meeting"
    elif isinstance(event, UnifyMeetStarted):
        cm.call_manager.unify_meet_call_session_id = event.call_session_id or ""
        cm.call_manager.unify_meet_start_timestamp = event.timestamp
        # Desktop viewers naming a call that is not this one cannot be closed by
        # anybody: the only stop event that would match them names a call nobody
        # is on any more. Taken from the event rather than the field just set, so
        # a stale id can never decide which viewers survive.
        cm.drop_stale_call_screen_share_viewers(event.call_session_id or "")
        label = "Unify Meet"
    else:
        raise ValueError(f"Unknown event type: {event.__class__.__name__}")

    cm.notifications_bar.push_notif(
        "Comms",
        f"{label} started with {sender_name}",
        timestamp=event.timestamp,
    )
    if isinstance(event, GoogleMeetStarted):
        medium = Medium.GOOGLE_MEET
    elif isinstance(event, TeamsMeetStarted):
        medium = Medium.TEAMS_MEET
    elif isinstance(event, PhoneCallStarted):
        medium = Medium.PHONE_CALL
    elif isinstance(event, WhatsAppCallStarted):
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.UNIFY_MEET
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content="<Call Started>",
        timestamp=event.timestamp,
    )
    conv_state = cm.contact_index.get_or_create_conversation(contact_id)
    conv_state.on_call = True

    # Start recording now rather than at dispatch: this is the first moment the
    # LiveKit room is known to exist and carry audio, which is what the Room
    # Composite Egress compositor waits for. Started earlier (at SIP bridge
    # setup or agent dispatch) it races the join and LiveKit kills the job with
    # "Start signal not received", producing no file.
    #
    # Detached deliberately: the request is a gateway round-trip that ends in a
    # LiveKit API call, and everything below (the outbound call_answered status,
    # the meet interaction-state sync, guidance delivery) is call-start UX that
    # must not wait on it. A slow or unreachable gateway would otherwise stall
    # the answer transition for as long as the request takes to time out.
    _recording_task = asyncio.create_task(_start_session_recording(event, cm))
    cm._recording_start_tasks.add(_recording_task)
    _recording_task.add_done_callback(cm._recording_start_tasks.discard)

    if isinstance(event, UnifyMeetStarted) and cm.call_manager.is_outbound:
        await cm.event_broker.publish(
            "app:call:status",
            json.dumps({"type": "call_answered"}),
        )

    # Sync meet interaction state that may have been set before the call started.
    # The fast brain starts with all flags as False and relies on guidance delivery,
    # so any state that was already active needs to be pushed now.
    if cm.assistant_screen_share_active and cm.call_manager._socket_server:
        from unify.conversation_manager.medium_scripts.common import (
            CALL_DESKTOP_SHARE_SURFACE,
            _resolve_agent_service_url,
        )

        guidance_text = _MEET_FAST_BRAIN_GUIDANCE[AssistantScreenShareStarted]
        notification_event = FastBrainNotification(
            contact=contact,
            message=guidance_text,
            source="meet_interaction",
            agent_service_url=_resolve_agent_service_url(),
            # Being watched and being on this call's stage are separate answers.
            # A Desktop tab open beside the call is the first without being the
            # second, so the room's copy is stated rather than inferred from the
            # flag above -- inferring it is what put one person's tab on
            # everybody's stage.
            meet_surface_state={
                "assistant_screen_share_active": True,
                CALL_DESKTOP_SHARE_SURFACE: cm.assistant_desktop_watched_from_call(
                    cm.call_manager.unify_meet_call_session_id,
                ),
            },
        )
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:notification",
            notification_event.to_json(),
        )

    # No LLM run here — outbound opener is pre-computed via make_call(opener=...).
    # The slow brain will be woken later by:
    # - InboundPhoneUtterance (user says something)
    # - ActorResult (action completes)
    # - NotificationInjectedEvent (cross-channel notification)
    # - SMSReceived/EmailReceived while on call


@EventHandler.register(
    (
        GoogleMeetParticipantJoined,
        GoogleMeetParticipantLeft,
        TeamsMeetParticipantJoined,
        TeamsMeetParticipantLeft,
    ),
)
async def _(
    event: (
        GoogleMeetParticipantJoined
        | GoogleMeetParticipantLeft
        | TeamsMeetParticipantJoined
        | TeamsMeetParticipantLeft
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Surface browser-meet roster changes as notifications so the LLM is aware."""
    name = event.participant_name
    if isinstance(event, (GoogleMeetParticipantJoined, GoogleMeetParticipantLeft)):
        meeting_label = "Google Meet"
        medium = Medium.GOOGLE_MEET
    else:
        meeting_label = "Teams meeting"
        medium = Medium.TEAMS_MEET

    if isinstance(event, (GoogleMeetParticipantJoined, TeamsMeetParticipantJoined)):
        label = f"{name} joined the {meeting_label}"
    else:
        label = f"{name} left the {meeting_label}"

    cm.notifications_bar.push_notif("Comms", label, event.timestamp)

    contact_id = (event.contact.get("contact_id") if event.contact else None) or 1
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=name,
        thread_name=medium,
        message_content=f"<{label}>",
        role="system",
        timestamp=event.timestamp,
    )


async def _record_meet_chat(
    event: (
        GoogleMeetChatMessage
        | TeamsMeetChatMessage
        | GoogleMeetChatSent
        | TeamsMeetChatSent
    ),
    cm: "ConversationManager",
    *,
    medium: Medium,
    sender_name: str,
    role: str,
) -> None:
    """Record one meeting-chat line, in either direction.

    It lands in three places, each serving a different reader. The contact
    thread is the brain's flat prompt view, so provenance rides in the body as a
    ``<meeting chat>`` marker -- nothing there carries structure. The Transcripts
    mirror and the call-utterance store are durable records, so they carry the
    bare text plus ``kind="chat"``: Console needs to tell a typed line from a
    spoken one to render it as chat and withhold a seek control it has no audio
    for.

    The store row is keyed on the meet session id, the same key the spoken lines
    use, which puts chat and speech in one chronological transcript. The
    exchange id needs no plumbing here -- ``log_message`` resolves it from the
    medium.
    """
    contact_id = (event.contact.get("contact_id") if event.contact else None) or 1
    message_id = cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=f"<meeting chat> {event.content}",
        role=role,
        timestamp=event.timestamp,
    )

    await managers_utils.queue_operation(
        managers_utils.log_message,
        cm,
        event,
        local_message_id=message_id,
    )

    call_key = cm.call_manager.meet_session_id
    if not call_key:
        LOGGER.error(
            f"{medium.value} chat message dropped from the call store: no "
            "session id on the active meet",
        )
        return

    is_assistant = role == "assistant"
    utterance = {
        "content": event.content,
        "speaker_name": sender_name,
        "speaker_assistant_id": (
            int(SESSION_DETAILS.assistant.agent_id)
            if is_assistant and SESSION_DETAILS.assistant.agent_id is not None
            else None
        ),
        "spoken_at": event.timestamp.isoformat(),
        "metadata": {
            "contact_id": contact_id,
            # Typed, not spoken. Consumers key the chat rendering and the
            # absence of a seek control off this rather than guessing from
            # a missing audio offset, which a spoken line can also have.
            "kind": "chat",
        },
    }
    asyncio.create_task(
        asyncio.to_thread(
            post_call_utterances_to_orchestra,
            str(call_key),
            [utterance],
        ),
    )


@EventHandler.register((GoogleMeetChatMessage, TeamsMeetChatMessage))
async def _(
    event: GoogleMeetChatMessage | TeamsMeetChatMessage,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Record an inbound meeting-chat message and wake the brain to answer it.

    Notify + wake, like the other prominent meet events: someone typing is
    addressing the assistant just as much as someone speaking, so this schedules
    an LLM run rather than leaving the message to be noticed by whatever turn
    happens next. Without the run the text still reaches the brain's context but
    only surfaces once somebody speaks -- so a meeting where the assistant is
    asked something purely in chat gets no reply at all.
    """
    medium = (
        Medium.GOOGLE_MEET
        if isinstance(event, GoogleMeetChatMessage)
        else Medium.TEAMS_MEET
    )
    label = f"{event.sender_name} in meeting chat: {event.content}"
    cm.notifications_bar.push_notif("Comms", label, event.timestamp)

    await _record_meet_chat(
        event,
        cm,
        medium=medium,
        sender_name=event.sender_name,
        role="user",
    )

    contact_id = (event.contact.get("contact_id") if event.contact else None) or 1
    await cm.request_llm_run(delay=0, triggering_contact_id=contact_id)


@EventHandler.register((GoogleMeetChatSent, TeamsMeetChatSent))
async def _(
    event: GoogleMeetChatSent | TeamsMeetChatSent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Record what the assistant posted into the meeting chat.

    No LLM run and no notification: the assistant just acted, so there is
    nothing to wake it for and nothing to tell its operator that it does not
    already know. Recording it is what stops the transcript showing a chat
    question with no answer beside it.
    """
    medium = (
        Medium.GOOGLE_MEET
        if isinstance(event, GoogleMeetChatSent)
        else Medium.TEAMS_MEET
    )
    await _record_meet_chat(
        event,
        cm,
        medium=medium,
        sender_name=SESSION_DETAILS.assistant.name or "Assistant",
        role="assistant",
    )


@EventHandler.register((GoogleMeetAlone, TeamsMeetAlone))
async def _(
    event: GoogleMeetAlone | TeamsMeetAlone,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Everyone else left the browser meet — wake the brain to decide the exit.

    Notify + LLM-decides: this pushes a system notification and wakes the slow
    brain, which then chooses whether to say a brief closing line and hang up
    (the same ``hang_up`` path a user's "leave the call" request takes). No
    automatic teardown happens here.
    """
    if isinstance(event, GoogleMeetAlone):
        meeting_label = "Google Meet"
        medium = Medium.GOOGLE_MEET
    else:
        meeting_label = "Teams meeting"
        medium = Medium.TEAMS_MEET

    label = (
        f"Everyone else has left the {meeting_label} — you're the only "
        f"participant left. Say a brief goodbye and leave the call unless "
        f"there's still something to do."
    )
    cm.notifications_bar.push_notif("Comms", label, event.timestamp)

    contact_id = (event.contact.get("contact_id") if event.contact else None) or 1
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name="Comms",
        thread_name=medium,
        message_content=f"<{label}>",
        role="system",
        timestamp=event.timestamp,
    )

    await cm.request_llm_run(delay=0)


@EventHandler.register(PhoneCallNotAnswered)
async def _(
    event: PhoneCallNotAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """
    Handle outbound call not answered (no-answer, busy, canceled, failed).

    This event arrives from the telephony system (via GCP PubSub) when the contact
    doesn't pick up for outbound calls. We need to:
    1. Tell the voice agent to stop (if running)
    2. Clean up the call process
    3. Notify the LLM brain so it can react appropriately
    """
    contact = cm.contact_index.get_contact(
        phone_number=event.contact.get("phone_number"),
    )
    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)
    reason = event.reason or "no-answer"

    # Forward stop status to voice agent subprocess
    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "stop", "reason": f"call_not_answered:{reason}"}),
    )

    # Reset mode if we were in call mode
    if cm.mode.is_voice:
        cm.mode = Mode.TEXT
        cm.call_manager.call_contact = None

    # Clean up the call process
    await cm.call_manager.cleanup_call_proc()

    # Clear session state
    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.google_meet_start_timestamp = None
    cm.call_manager.teams_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.google_meet_exchange_id = UNASSIGNED
    cm.call_manager.teams_meet_exchange_id = UNASSIGNED
    cm.call_manager.call_exchange_destination = None

    # Build display content
    reason_display = _call_not_answered_reason_text(reason)

    notif_content = f"Outbound call to {sender_name} {reason_display}"
    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.PHONE_CALL,
        message_content=f"<Call Not Answered: {reason_display}>",
        role="assistant",
        timestamp=event.timestamp,
    )

    # Trigger LLM run so the brain can decide next steps
    await cm.request_llm_run(
        delay=0,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallNotAnswered)
async def _(
    event: WhatsAppCallNotAnswered,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle outbound WhatsApp call not answered — same cleanup as phone."""
    contact = cm.contact_index.get_contact(
        whatsapp_number=event.contact.get("whatsapp_number"),
    )
    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)
    reason = event.reason or "no-answer"

    await cm.event_broker.publish(
        "app:call:status",
        json.dumps({"type": "stop", "reason": f"call_not_answered:{reason}"}),
    )

    if cm.mode.is_voice:
        cm.mode = Mode.TEXT
        cm.call_manager.call_contact = None

    await cm.call_manager.cleanup_call_proc()

    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.call_exchange_destination = None

    reason_display = _call_not_answered_reason_text(reason)

    notif_content = f"Outbound WhatsApp call to {sender_name} {reason_display}"
    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content=f"<WhatsApp Call Not Answered: {reason_display}>",
        role="assistant",
        timestamp=event.timestamp,
    )

    await cm.request_llm_run(
        delay=0,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallPermissionResponse)
async def _(
    event: WhatsAppCallPermissionResponse,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Handle call permission grant/rejection from a WhatsApp contact."""
    contact = event.contact
    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)

    permission_status = event.status or ("accepted" if event.accepted else "rejected")
    pending_openers = getattr(cm, "_pending_whatsapp_call_openers", None)
    whatsapp_number = contact.get("whatsapp_number")
    pool_number = (
        getattr(cm, "assistant_whatsapp_number", "")
        or getattr(SESSION_DETAILS.assistant, "whatsapp_number", "")
        or ""
    )

    if permission_status == "accepted":
        notif_content = f"{sender_name} granted WhatsApp call permission; calling now"
        message_content = "<WhatsApp Call Permission Granted: calling now>"
    elif permission_status == "rejected":
        notif_content = f"{sender_name} rejected WhatsApp call permission"
        if isinstance(pending_openers, dict):
            pending_openers.pop(contact_id, None)
        if pool_number and whatsapp_number:
            from unify.conversation_manager.domains import comms_utils

            try:
                await comms_utils.clear_pending_whatsapp_call_intent(
                    pool_number=pool_number,
                    contact_number=whatsapp_number,
                )
            except Exception as exc:
                LOGGER.error(
                    f"{DEFAULT_ICON} Failed to clear pending WhatsApp call intent: {exc}",
                )
        message_content = "<WhatsApp Call Permission Rejected>"
    else:
        notif_content = (
            f"{sender_name} interacted with WhatsApp call permission, but the "
            "provider did not include an accepted/rejected payload"
        )
        message_content = (
            "<WhatsApp Call Permission Unknown: waiting for reconciliation>"
        )

    cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content=message_content,
        role="user",
        timestamp=event.timestamp,
    )

    if permission_status == "accepted":
        # Permission granted after we sent a VOICE_CALL_REQUEST template.
        # Place the outbound call directly — no LLM round-trip needed.
        from unify.conversation_manager.domains import comms_utils
        from unify.conversation_manager.domains.call_manager import make_room_name

        opener = None
        if isinstance(pending_openers, dict):
            opener = pending_openers.pop(contact_id, None)
        if opener is None and pool_number and whatsapp_number:
            try:
                intent = await comms_utils.get_pending_whatsapp_call_intent(
                    pool_number=pool_number,
                    contact_number=whatsapp_number,
                )
            except Exception as exc:
                LOGGER.error(
                    f"{DEFAULT_ICON} Failed to load pending WhatsApp call intent: {exc}",
                )
                intent = None
            opener = (intent or {}).get("context")

        if opener is None:
            await cm.request_llm_run(
                delay=0,
                triggering_contact_id=contact_id,
            )
            return

        _apply_stashed_whatsapp_call_context(cm, opener)

        assistant_id = str(SESSION_DETAILS.assistant.agent_id)
        agent_name = SESSION_DETAILS.assistant.name or ""
        room_name = make_room_name(assistant_id, "whatsapp_call")

        cm.call_manager._whatsapp_call_joining = True
        response = await comms_utils.start_whatsapp_call(
            to_number=whatsapp_number,
            agent_name=agent_name,
            room_name=room_name,
        )
        if response.get("success"):
            if pool_number and whatsapp_number:
                try:
                    await comms_utils.clear_pending_whatsapp_call_intent(
                        pool_number=pool_number,
                        contact_number=whatsapp_number,
                    )
                except Exception as exc:
                    LOGGER.error(
                        f"{DEFAULT_ICON} Failed to clear pending WhatsApp call intent: {exc}",
                    )
            call_event = cm.build_whatsapp_call_sent_event(contact)
            await cm.event_broker.publish(
                "app:comms:whatsapp_call_sent",
                call_event.to_json(),
            )
            return

        cm.call_manager._whatsapp_call_joining = False
        cm.call_manager.pending_opener = ""
        cm.call_manager.pending_briefing = ""
        cm.call_manager.pending_hang_up_gate = ""
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.WHATSAPP_CALL,
            message_content="<WhatsApp Call Failed After Permission Granted>",
            role="assistant",
            timestamp=event.timestamp,
        )

    await cm.request_llm_run(
        delay=0,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(WhatsAppCallInviteSent)
async def _(
    event: WhatsAppCallInviteSent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Log the invite template send in the conversation thread."""
    contact = event.contact
    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)

    cm.notifications_bar.push_notif(
        "Comms",
        f"WhatsApp call permission request sent to {sender_name}; waiting for approval",
        event.timestamp,
    )

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=Medium.WHATSAPP_CALL,
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )


@EventHandler.register(
    (
        InboundPhoneUtterance,
        InboundUnifyMeetUtterance,
        InboundWhatsAppCallUtterance,
        OutboundPhoneUtterance,
        OutboundUnifyMeetUtterance,
        OutboundWhatsAppCallUtterance,
        InboundGoogleMeetUtterance,
        OutboundGoogleMeetUtterance,
        InboundTeamsMeetUtterance,
        OutboundTeamsMeetUtterance,
    ),
)
async def _(event: Event, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "event",
        f"Publishing transcript: {event.__class__.__name__}",
    )

    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    if contact is None:
        contact = event.contact
    sender_name = _get_sender_name(contact)
    role = "user" if event.__class__.__name__.startswith("Inbound") else "assistant"

    is_whatsapp_call = isinstance(
        event,
        (InboundWhatsAppCallUtterance, OutboundWhatsAppCallUtterance),
    )
    is_google_meet = isinstance(
        event,
        (InboundGoogleMeetUtterance, OutboundGoogleMeetUtterance),
    )
    is_teams_meet = isinstance(
        event,
        (InboundTeamsMeetUtterance, OutboundTeamsMeetUtterance),
    )
    is_unify_meet = isinstance(
        event,
        (InboundUnifyMeetUtterance, OutboundUnifyMeetUtterance),
    )
    if is_google_meet:
        medium = Medium.GOOGLE_MEET
    elif is_teams_meet:
        medium = Medium.TEAMS_MEET
    elif is_unify_meet:
        medium = Medium.UNIFY_MEET
    elif is_whatsapp_call:
        medium = Medium.WHATSAPP_CALL
    else:
        medium = Medium.PHONE_CALL

    # For diarized utterances, prefer the speaker_label from the event so we
    # attribute speech to the right participant. On enrolled-contact calls
    # this surfaces anonymous voices as "Speaker 2", "Speaker 3", …
    if (
        isinstance(
            event,
            (
                InboundPhoneUtterance,
                InboundWhatsAppCallUtterance,
                InboundUnifyMeetUtterance,
                InboundGoogleMeetUtterance,
                InboundTeamsMeetUtterance,
            ),
        )
        and event.speaker_label
    ):
        sender_name = event.speaker_label

    message_id = cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role=role,
    )

    await managers_utils.queue_operation(
        managers_utils.log_message,
        cm,
        event,
        local_message_id=message_id,
    )

    # Meet transcripts are first-class: Console reads them from Orchestra's
    # call-utterance store, so every utterance is appended there in addition to
    # the Transcripts mirror above.
    #
    # Browser meets are included now that their speaker names come from the
    # meeting platform rather than a screen scrape -- before, the only names
    # available were voice-cluster placeholders, and writing those into the
    # permanent record would have been worse than writing nothing. Their key is
    # the bot id: ``call_utterance.call_id`` is an opaque string, and readers
    # group by it rather than joining a session row.
    if is_unify_meet or is_google_meet or is_teams_meet:
        call_key = (
            cm.call_manager.unify_meet_call_session_id
            if is_unify_meet
            else cm.call_manager.meet_session_id
        )
        if not call_key:
            LOGGER.error(
                f"{medium.value} utterance dropped from the call store: no "
                "session id on the active meet",
            )
        if call_key:
            utterance = {
                "content": event.content,
                "speaker_name": (
                    sender_name
                    if role == "user"
                    else SESSION_DETAILS.assistant.name or "Assistant"
                ),
                "speaker_assistant_id": (
                    int(SESSION_DETAILS.assistant.agent_id)
                    if role == "assistant"
                    and SESSION_DETAILS.assistant.agent_id is not None
                    else None
                ),
                # `spoken_at` stays the commit instant: it orders the store
                # and predates this change. The audible start rides alongside
                # it, so a reader can place the line in the recording without
                # the meaning of an existing column shifting underneath it.
                "spoken_at": event.timestamp.isoformat(),
                "metadata": {
                    key: value
                    for key, value in {
                        "contact_id": contact_id,
                        # Who was in the meeting when this line was said. The
                        # platform reports names reliably and emails only
                        # sometimes, so the roster is the record of who could
                        # have spoken, and it is worth keeping per line because
                        # attendance changes mid-call.
                        # ``or None`` so an empty roster is dropped rather than
                        # stored as an empty list.
                        "participant_names": (
                            list(getattr(event, "participant_names", None) or [])
                            or None
                        ),
                        "speech_started_at": getattr(
                            event,
                            "speech_started_at",
                            None,
                        ),
                    }.items()
                    if value is not None
                },
            }
            asyncio.create_task(
                asyncio.to_thread(
                    post_call_utterances_to_orchestra,
                    str(call_key),
                    [utterance],
                ),
            )

    # Reset proactive speech on any utterance (user or assistant).
    await cm.schedule_proactive_speech()

    if role == "assistant":
        # The real spoken utterance has now landed (verbatim, or a truncated
        # prefix after a barge-in). Clear the render-only in-flight overlay for
        # this line so future turns rely on the actually-spoken transcript only.
        inflight = (getattr(cm, "_inflight_voice_speech", "") or "").strip()
        spoken = (event.content or "").strip()
        if inflight and spoken and inflight.startswith(spoken):
            cm._inflight_voice_speech = ""
            # Signal that the just-published spoken guidance has been delivered
            # (full line, or a barge-in's truncated prefix) so a deferred hang-up
            # can safely tear the session down without cutting speech off.
            delivered = getattr(cm, "_inflight_speech_delivered", None)
            if delivered is not None:
                delivered.set()

    if role == "user":
        # Link any pending user/webcam screenshot (forwarded from the fast
        # brain via IPC) to this message by stamping it with the message_id.
        cm._claim_pending_user_screenshot(message_id)

        if cm.assistant_screen_share_active:
            await cm.capture_assistant_screenshot(
                event.content,
                message_id,
                cached=True,
            )


@EventHandler.register(VoiceEnrollmentCaptured)
async def _(
    event: VoiceEnrollmentCaptured,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Persist an auto-captured voice enrollment onto the contact row."""
    contact_id = event.contact.get("contact_id") if event.contact else None
    if contact_id is None or not event.embedding:
        return
    contact = cm.contact_index.get_contact(contact_id=contact_id) or event.contact
    sender_name = _get_sender_name(contact)

    async def _persist(cm: "ConversationManager") -> None:
        from unify.contact_manager.voice_enrollment import (
            VOICE_ENROLLMENT_SOURCE_AUTO,
        )

        wav_bytes = None
        if event.wav_path:
            wav_file = Path(event.wav_path)
            if wav_file.exists():
                wav_bytes = wav_file.read_bytes()
                wav_file.unlink(missing_ok=True)
        await asyncio.to_thread(
            cm.contact_manager.set_voice_enrollment,
            contact_id=int(contact_id),
            embedding=[float(x) for x in event.embedding],
            wav_bytes=wav_bytes,
            source=VOICE_ENROLLMENT_SOURCE_AUTO,
        )
        LOGGER.info(
            f"{DEFAULT_ICON} Voice enrollment persisted for contact "
            f"{contact_id} ({event.duration_s:.0f}s of speech)",
        )

    await managers_utils.queue_operation(_persist, cm)

    cm.notifications_bar.push_notif(
        "Comms",
        f"Voice profile enrolled for {sender_name} from this call's audio; "
        "their turns on future calls will be voice-verified",
        event.timestamp,
    )


@EventHandler.register(VoiceEnrollmentSuggested)
async def _(
    event: VoiceEnrollmentSuggested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Surface degraded speaker attribution and nudge manual enrollment."""
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or {}
    sender_name = _get_sender_name(contact)

    cm.notifications_bar.push_notif(
        "Comms",
        f"{event.num_speakers} distinct voices detected on this call, but "
        f"{sender_name} has no voice enrollment, so turns cannot be "
        "attributed to individual speakers",
        event.timestamp,
    )

    # Only the account holder can use the in-app fallback recorder, so the
    # guidance nudge is limited to the boss.
    if contact_id == SESSION_DETAILS.boss_contact_id:
        await publish_voice_enrollment_suggested(num_speakers=event.num_speakers)
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=_active_voice_thread_medium(cm),
            message_content=(
                "Speaker attribution note: multiple distinct voices are "
                f"present on this call, but {sender_name} has no voice "
                "enrollment, so the transcript cannot reliably distinguish "
                "who said what. If a natural moment arises, you may mention "
                "that a short in-app voice recorder will appear when the call "
                "ends so they can enroll for future calls. Do not derail the "
                "current topic for this."
            ),
            role="guidance",
        )


@EventHandler.register(FastBrainTurnCompleted)
async def _(
    event: FastBrainTurnCompleted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Start the slow-brain run after the Voice Agent finishes a user turn."""
    from unify.conversation_manager.events import FAST_BRAIN_TURN_SILENCE
    from unify.conversation_manager.prompt_builders import (
        build_fast_brain_turn_guidance,
    )

    if event.classification == FAST_BRAIN_TURN_SILENCE:
        return

    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or {}
    sender_name = _get_sender_name(contact)

    note = build_fast_brain_turn_guidance(
        classification=event.classification,
        intended_speech=event.intended_speech,
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=_active_voice_thread_medium(cm),
        message_content=note,
        role="guidance",
    )

    user_content = (event.user_content or "").strip()
    if not user_content:
        return

    await cm.handle_voice_user_turn(
        user_content,
        triggering_contact_id=contact_id,
        turn_id=event.turn_id,
    )


@EventHandler.register(AssistantTurnInjected)
async def _(
    event: AssistantTurnInjected,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or cm.get_active_contact() or {}
    contact_id = contact.get("contact_id") if contact else contact_id
    sender_name = _get_sender_name(contact)

    message_id = cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=_active_voice_thread_medium(cm),
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )

    await managers_utils.queue_operation(
        managers_utils.log_message,
        cm,
        event,
        local_message_id=message_id,
    )

    if cm.call_manager and cm.call_manager._socket_server:
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:notification",
            event.to_json(),
        )
    else:
        cm.call_manager.pending_opener = event.content

    if event.schedule_proactive:
        await cm.schedule_proactive_speech()


@EventHandler.register(FastBrainNotification)
async def _(
    event: FastBrainNotification,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or {}
    sender_name = _get_sender_name(contact)

    # Only record silent guidance here. should_speak guidance is spoken by the
    # fast brain and recorded once via the actually-spoken Outbound utterance
    # (with the real, possibly rewritten text); pre-writing it duplicated that
    # record and surfaced a stray user-looking turn in the voice transcript.
    if not event.should_speak:
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=_active_voice_thread_medium(cm),
            message_content=event.message,
            role="guidance",
        )

    if event.should_speak:
        await cm.schedule_proactive_speech()


@EventHandler.register(VoiceInterrupt)
async def _(
    event: VoiceInterrupt,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Record that the caller interrupted before hearing the full message.

    Only the spoken prefix makes it into the transcript, so without this the slow
    brain would treat the prefix as the complete message it delivered. We append
    a guidance note naming the unheard remainder so the slow brain knows it was
    cut off and can re-surface the missed content if it still matters.
    """
    remainder = (event.unheard_remainder or "").strip()
    if not remainder:
        return
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = (
        cm.contact_index.get_contact(contact_id=contact_id) if contact_id else None
    )
    if contact is None:
        contact = event.contact or {}
    sender_name = _get_sender_name(contact)

    note = (
        "[The caller interrupted you mid-sentence and did NOT hear the rest of "
        f'your last message: "{remainder}". Weave it back in naturally if it '
        "still matters; otherwise move on with what they just said.]"
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=_active_voice_thread_medium(cm),
        message_content=note,
        role="guidance",
    )


@EventHandler.register(ProactiveSpeechControl)
async def _(
    event: ProactiveSpeechControl,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    await cm.set_proactive_speech_enabled(event.enabled)
    if event.enabled and event.schedule_now:
        await cm.schedule_proactive_speech()


@EventHandler.register(ActionStopRequested)
async def _(
    event: ActionStopRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    stopped = await cm.stop_in_flight_action_by_calling_id(
        event.calling_id,
        reason=event.reason or "Stopped from Console Actions pane.",
    )
    if not stopped:
        LOGGER.info(
            "%s Action stop requested for unknown calling_id=%s (source=%s)",
            DEFAULT_ICON,
            event.calling_id,
            event.source or "console",
        )


@EventHandler.register(
    (
        PhoneCallEnded,
        UnifyMeetEnded,
        GoogleMeetEnded,
        TeamsMeetEnded,
        WhatsAppCallEnded,
    ),
)
async def _(
    event: (
        PhoneCallEnded
        | UnifyMeetEnded
        | GoogleMeetEnded
        | TeamsMeetEnded
        | WhatsAppCallEnded
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, UnifyMeetEnded):
        active_session = cm.call_manager.unify_meet_call_session_id
        if (
            event.call_session_id
            and active_session
            and event.call_session_id != active_session
        ):
            LOGGER.info(
                "Ignoring stale UnifyMeetEnded for session %s (active=%s)",
                event.call_session_id,
                active_session,
            )
            return

    # Persist session identifiers in exchange metadata and stash the exchange's
    # (id, destination) so the async RecordingReady handler can find it. The
    # destination is part of the address: a shared assistant authors its
    # transcript into a team root, and an id resolved there means nothing in the
    # manager's home root.
    destination = cm.call_manager.call_exchange_destination
    if isinstance(event, (PhoneCallEnded, WhatsAppCallEnded)):
        exchange_id = cm.call_manager.call_exchange_id
        if exchange_id != UNASSIGNED:
            metadata = {
                key: value
                for key, value in {
                    "conference_name": cm.call_manager.conference_name,
                    "room_name": cm.call_manager.room_name,
                    "call_session_id": cm.call_manager.call_session_id,
                    "provider_call_sid": cm.call_manager.provider_call_sid,
                }.items()
                if value
            }
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                metadata,
                destination=destination,
            )
            for key in (
                cm.call_manager.call_session_id,
                cm.call_manager.provider_call_sid,
                cm.call_manager.room_name,
                cm.call_manager.conference_name,
            ):
                if key:
                    cm._recording_exchange_ids[key] = (exchange_id, destination)
    elif isinstance(event, GoogleMeetEnded):
        exchange_id = cm.call_manager.google_meet_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.room_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"room_name": cm.call_manager.room_name},
                destination=destination,
            )
            cm._recording_exchange_ids[cm.call_manager.room_name] = (
                exchange_id,
                destination,
            )
    elif isinstance(event, TeamsMeetEnded):
        exchange_id = cm.call_manager.teams_meet_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.room_name:
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                {"room_name": cm.call_manager.room_name},
                destination=destination,
            )
            cm._recording_exchange_ids[cm.call_manager.room_name] = (
                exchange_id,
                destination,
            )
    else:
        exchange_id = cm.call_manager.unify_meet_exchange_id
        if exchange_id != UNASSIGNED and cm.call_manager.room_name:
            metadata = {
                key: value
                for key, value in {
                    "room_name": cm.call_manager.room_name,
                    "call_session_id": cm.call_manager.unify_meet_call_session_id
                    or event.call_session_id,
                }.items()
                if value
            }
            cm.transcript_manager.update_exchange_metadata(
                exchange_id,
                metadata,
                destination=destination,
            )
            cm._recording_exchange_ids[cm.call_manager.room_name] = (
                exchange_id,
                destination,
            )

    cm.mode = Mode.TEXT
    cm.call_manager.call_contact = None
    await cm.call_manager.set_hang_up_gate(None)

    if isinstance(event, (UnifyMeetEnded, GoogleMeetEnded, TeamsMeetEnded)):
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)
    elif isinstance(event, WhatsAppCallEnded):
        contact = cm.contact_index.get_contact(
            whatsapp_number=event.contact.get("whatsapp_number"),
        )
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact.get("phone_number"),
        )

    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else event.contact.get("contact_id")
    )
    conv_state = cm.contact_index.get_conversation_state(contact_id)
    if conv_state:
        conv_state.on_call = False

    if isinstance(event, GoogleMeetEnded):
        await cm.call_manager.cleanup_google_meet()
    elif isinstance(event, TeamsMeetEnded):
        await cm.call_manager.cleanup_teams_meet()
    else:
        await cm.call_manager.cleanup_call_proc()
    await cm.cancel_proactive_speech()

    # Clear all session state after cleanup.
    cm.call_manager.conference_name = None
    cm.call_manager.room_name = None
    cm.call_manager.call_session_id = ""
    cm.call_manager.unify_meet_call_session_id = ""
    cm.call_manager.provider_call_sid = ""
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.google_meet_start_timestamp = None
    cm.call_manager.teams_meet_start_timestamp = None
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.google_meet_exchange_id = UNASSIGNED
    cm.call_manager.teams_meet_exchange_id = UNASSIGNED
    cm.call_manager.call_exchange_destination = None
    cm.reset_meet_surfaces()

    sender_name = _get_sender_name(contact)
    if isinstance(event, GoogleMeetEnded):
        label = "Google Meet"
    elif isinstance(event, TeamsMeetEnded):
        label = "Teams meeting"
    elif isinstance(event, UnifyMeetEnded):
        label = "Unify Meet"
    elif isinstance(event, WhatsAppCallEnded):
        label = "WhatsApp call"
    else:
        label = "Phone call"
    cm.notifications_bar.push_notif(
        "Comms",
        f"{label} with {sender_name} has ended.",
        event.timestamp,
    )

    await cm.request_llm_run(
        delay=0,
        triggering_contact_id=contact_id,
    )


@EventHandler.register(FastBrainHangUp)
async def _(
    event: FastBrainHangUp,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Disarm the voice close gate and tear down the active call."""
    await cm.call_manager.set_hang_up_gate(None)
    await cm.call_manager.end_call()
    cm.notifications_bar.push_notif(
        "Comms",
        f"Voice agent ended the call after saying: {event.farewell}",
        event.timestamp,
    )


@EventHandler.register(RecordingReady)
async def _(
    event: RecordingReady,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # Identifier -> the exchange metadata key it was stored under at call end.
    lookups = [
        ("call_session_id", event.call_session_id),
        ("provider_call_sid", event.provider_call_sid),
        ("room_name", event.room_name),
        ("conference_name", event.conference_name),
    ]
    keys = [value for _, value in lookups]
    name = next((key for key in keys if key), event.conference_name)
    # Both lookup paths yield (exchange_id, destination): the id alone does not
    # address a row, since the same id in another root is another exchange.
    located = None
    for key in keys:
        if key:
            located = cm._recording_exchange_ids.pop(key, None)
            if located is not None:
                break
    if located is None:
        # The in-process map only covers a recording that arrives while the
        # container that ran the call is still up. Egress finalises minutes
        # after the room closes, by which time the pod has often been recycled,
        # so fall back to the identifiers persisted on the exchange itself.
        for metadata_key, value in lookups:
            if not value:
                continue
            located = await asyncio.to_thread(
                cm.transcript_manager.resolve_exchange_id_by_metadata,
                metadata_key,
                value,
            )
            if located is not None:
                cm._session_logger.debug(
                    "recording",
                    f"{DEFAULT_ICON} [RecordingReady] Recovered exchange "
                    f"{located[0]} in {located[1] or 'personal'} from stored "
                    f"{metadata_key} for {name}",
                )
                break
    if located is not None:
        exchange_id, destination = located
        cm.transcript_manager.update_exchange_metadata(
            exchange_id,
            {
                key: value
                for key, value in {
                    "recording_url": event.recording_url,
                    "recording_started_at": event.recording_started_at,
                    "recording_call_session_id": event.call_session_id,
                    "recording_provider_call_sid": event.provider_call_sid,
                    "recording_room_name": event.room_name,
                }.items()
                if value
            },
            destination=destination,
        )
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [RecordingReady] Stored recording_url on exchange "
            f"{exchange_id} in {destination or 'personal'} for {name}",
        )
    else:
        cm._session_logger.debug(
            "recording",
            f"{DEFAULT_ICON} [RecordingReady] No exchange_id found for {name}",
        )


@EventHandler.register(
    (
        ActorResponse,
        ActorHandleResponse,
        ActorResult,
        ActorClarificationRequest,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    if isinstance(event, ActorClarificationRequest):
        if event.handle_id in cm.in_flight_actions:
            from unify.common.prompt_helpers import now as prompt_now

            cm.in_flight_actions[event.handle_id]["handle_actions"].append(
                {
                    "action_name": "clarification_request",
                    "query": event.query,
                    "call_id": event.call_id,
                    "timestamp": prompt_now(),
                },
            )
            await cm.request_llm_run()
    elif isinstance(event, ActorHandleResponse):
        # Handle response from an action steering operation.
        # Check both in-flight and completed actions — post-completion asks
        # publish on the same channel after ActorResult has already moved
        # the action to completed_actions.
        handle_data = cm.in_flight_actions.get(
            event.handle_id,
        ) or cm.completed_actions.get(event.handle_id)
        if handle_data:
            handle_actions = handle_data.get("handle_actions", [])
            action_name = event.action_name or "ask"
            expected_action_name = f"{action_name}_{event.handle_id}"

            # Find the pending action and update it with the response.
            for action in reversed(handle_actions):
                if (
                    action.get("action_name") == expected_action_name
                    and action.get("status") == "pending"
                ):
                    action["status"] = "completed"
                    action["response"] = event.response
                    break

            # Wake the brain LLM to process the response
            await cm.request_llm_run()
    else:
        ...


def _push_email_to_all_contacts(
    cm: "ConversationManager",
    event,
    sender_contact: dict | None,
    sender_name: str,
    subject: str,
    body: str,
    email_id: str | None,
    thread_id: str | None,
    attachments: list[str] | None,
    email_to: list[str],
    email_cc: list[str],
    email_bcc: list[str],
    role: str,
):
    """
    Push an email to ALL contacts involved (sender, to, cc, bcc).

    Emails are pushed to every known contact's thread to ensure no context is
    missing when viewing any contact-specific thread. Each message is tagged
    with `contact_role` to clarify the contact's relationship to the email.

    Args:
        cm: ConversationManager instance
        event: The email event (EmailSent or EmailReceived)
        sender_contact: The sender's contact dict (may be None for external senders)
        sender_name: Display name for the email sender
        subject: Email subject
        body: Email body
        email_id: Email ID for threading
        thread_id: Provider thread ID for deterministic thread targeting
        attachments: List of attachment filenames
        email_to: List of TO recipient email addresses
        email_cc: List of CC recipient email addresses
        email_bcc: List of BCC recipient email addresses
        role: "user" (received) or "assistant" (sent)
    """
    # Track which contact_ids we've already pushed to (avoid duplicates)
    pushed_contact_ids: set[int] = set()

    def _push_to_contact(contact_id: int, contact_role: str):
        """Helper to push email to a contact's thread."""
        if contact_id in pushed_contact_ids:
            return
        pushed_contact_ids.add(contact_id)
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=Medium.EMAIL,
            subject=subject,
            body=body,
            email_id=email_id,
            thread_id=thread_id,
            attachments=attachments,
            timestamp=event.timestamp,
            role=role,
            to=email_to,
            cc=email_cc,
            bcc=email_bcc,
            contact_role=contact_role,
        )

    def _resolve_contact_by_email(email_addr: str) -> dict | None:
        """Look up contact by email address."""
        return cm.contact_index.get_contact(email=email_addr)

    # 1. Push to sender's contact (if known)
    sender_contact_id = sender_contact.get("contact_id") if sender_contact else None
    if sender_contact_id is not None:
        _push_to_contact(sender_contact_id, "sender")

    # 2. Push to all TO recipients
    for email_addr in email_to or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "to")

    # 3. Push to all CC recipients
    for email_addr in email_cc or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "cc")

    # 4. Push to all BCC recipients
    for email_addr in email_bcc or []:
        contact = _resolve_contact_by_email(email_addr)
        if contact and contact.get("contact_id"):
            _push_to_contact(contact["contact_id"], "bcc")


def _should_wake_for_outbound_sent(event) -> bool:
    return not getattr(event, "suppress_slow_brain_wake", False)


def _credit_gate_reply_context(
    *,
    medium: Medium,
    contact_id: int | None,
    email_id: str | None = None,
    api_message_id: str | None = None,
    tags: list[str] | None = None,
    chat_id: str | None = None,
    channel_id: str | None = None,
    team_id: str | None = None,
    thread_id: str | None = None,
    group_id: str | None = None,
    thread_ts: str | None = None,
    guild_id: str | None = None,
    bot_id: str | None = None,
    message_id: str | None = None,
    routing_metadata: dict | None = None,
    tenant_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    context = {
        "medium": medium.value,
        "contact_id": contact_id,
    }
    optional_fields = {
        "email_id": email_id,
        "api_message_id": api_message_id,
        "tags": tags,
        "chat_id": chat_id,
        "channel_id": channel_id,
        "team_id": team_id,
        "thread_id": thread_id,
        "group_id": group_id,
        "thread_ts": thread_ts,
        "guild_id": guild_id,
        "bot_id": bot_id,
        "message_id": message_id,
        "routing_metadata": routing_metadata,
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
    }
    for key, value in optional_fields.items():
        if value:
            context[key] = value
    return context


@EventHandler.register(
    (
        SMSSent,
        SMSReceived,
        WhatsAppSent,
        WhatsAppReceived,
        EmailSent,
        EmailReceived,
        UnifyMessageSent,
        UnifyMessageReceived,
        ApiMessageSent,
        ApiMessageReceived,
        DiscordMessageReceived,
        DiscordMessageSent,
        DiscordChannelMessageReceived,
        DiscordChannelMessageSent,
        SlackMessageReceived,
        SlackMessageSent,
        SlackChannelMessageReceived,
        SlackChannelMessageSent,
        TeamsMessageReceived,
        TeamsMessageSent,
        TeamsChannelMessageReceived,
        TeamsChannelMessageSent,
        MsTeamsBotMessageSent,
        MsTeamsBotMessageReceived,
        MsTeamsBotChannelMessageSent,
        MsTeamsBotChannelMessageReceived,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    message_content = None
    attachments = None
    tags = None
    notif_content = None
    chat_id = None
    channel_id = None
    team_id = None
    thread_id = None
    group_id = None
    thread_ts = None
    event_ts = None
    message_id = None
    routing_metadata = None
    guild_id = None
    bot_id = None
    tenant_id = None
    conversation_id = None
    mentions = None

    # Get contact info from ContactManager, fallback to event.contact
    # Note: event.contact may be empty dict for emails to external addresses
    contact_id = event.contact.get("contact_id") if event.contact else None
    contact = cm.contact_index.get_contact(contact_id) if contact_id else None
    if contact is None:
        contact = event.contact or {}

    # contact_id may be None for external recipients not in contacts
    contact_id = contact.get("contact_id") if isinstance(contact, dict) else None
    sender_name = _get_sender_name(contact)

    match event:
        case SMSSent():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "sms_sent",
                f"SMS to {sender_name}: {event.content}",
            )
        case SMSReceived():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS Received from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "sms_received",
                f"SMS from {sender_name}: {event.content}",
            )
        case WhatsAppSent():
            medium = Medium.WHATSAPP_MESSAGE
            message_content = whatsapp_sent_history_content(event)
            attachments = event.attachments
            # A verbatim free-form send proves the window was open; a template
            # fallback proves it was closed.
            cm.note_whatsapp_window_open(contact_id, not event.via_template)
            notif_content = (
                f"WhatsApp template fallback sent to {sender_name}; original pending resend"
                if event.via_template
                else f"WhatsApp sent to {sender_name}"
            )
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "whatsapp_sent",
                f"WhatsApp to {sender_name}: {message_content}",
            )
        case WhatsAppReceived():
            medium = Medium.WHATSAPP_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"WhatsApp received from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            # An inbound reply (re)opens the 24-hour free-form window.
            cm.note_whatsapp_window_open(contact_id, True)
            cm._session_logger.info(
                "whatsapp_received",
                f"WhatsApp from {sender_name}: {event.content}",
            )
            pending_content = (
                cm._pending_whatsapp_resends.pop(contact_id, None)
                if contact_id is not None
                else None
            )
            if pending_content is not None:
                cm.notifications_bar.push_notif(
                    "comms",
                    (
                        f"Your earlier WhatsApp message to {sender_name} "
                        f"was not delivered verbatim. Original message: "
                        f'"{pending_content}". You can now resend or '
                        f"rework it via send_whatsapp."
                    ),
                    event.timestamp,
                )
        case EmailSent():
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            recipients = ", ".join((event.to or [])[:2])
            if len(event.to or []) > 2:
                recipients += "..."
            cm._session_logger.info(
                "email_sent",
                f"Email to {recipients}\n"
                f"Subject: {event.subject}\n\n"
                f"{event.body}",
            )
            # Email handling is special: push to ALL contacts involved
            email_to = event.to or []
            email_cc = event.cc or []
            email_bcc = event.bcc or []
            # For sent emails, the assistant is the sender
            _push_email_to_all_contacts(
                cm=cm,
                event=event,
                sender_contact=None,  # Assistant is sender, not a contact
                sender_name="You",
                subject=event.subject,
                body=event.body,
                email_id=event.email_id_replied_to,
                thread_id=event.thread_id,
                attachments=event.attachments,
                email_to=email_to,
                email_cc=email_cc,
                email_bcc=email_bcc,
                role="assistant",
            )
            notif_content = f"Email sent to {', '.join(email_to[:2])}{'...' if len(email_to) > 2 else ''}"
            cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
            if _should_wake_for_outbound_sent(event):
                await cm.request_llm_run(
                    triggering_contact_id=contact_id,
                )
            return  # Early return - email handling is complete

        case EmailReceived():
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "email_received",
                f"Email from {sender_name}\n"
                f"Subject: {event.subject}\n\n"
                f"{event.body}",
            )
            # Email handling is special: push to ALL contacts involved
            email_to = event.to or []
            email_cc = event.cc or []
            email_bcc = event.bcc or []
            _push_email_to_all_contacts(
                cm=cm,
                event=event,
                sender_contact=contact,  # The contact who sent the email
                sender_name=sender_name,
                subject=event.subject,
                body=event.body,
                email_id=event.email_id,
                thread_id=event.thread_id,
                attachments=event.attachments,
                email_to=email_to,
                email_cc=email_cc,
                email_bcc=email_bcc,
                role="user",
            )
            notif_content = f"Email Received from {sender_name}"
            cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
            await cm.cancel_proactive_speech()
            await _surface_trigger_task_candidates(
                cm=cm,
                event=event,
                medium=Medium.EMAIL,
                contact_id=contact_id,
                sender_name=sender_name,
                timestamp=event.timestamp,
            )
            email_reply_context = _credit_gate_reply_context(
                medium=Medium.EMAIL,
                contact_id=contact_id,
                email_id=event.email_id,
                thread_id=event.thread_id,
            )
            cm.record_last_inbound_reply(email_reply_context)
            await cm.request_llm_run(
                triggering_contact_id=contact_id,
                credit_gate_reply_context=email_reply_context,
            )
            return  # Early return - email handling is complete

        case UnifyMessageSent():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            thread_id = str(event.thread_id) if event.thread_id else None
            team_id = str(event.team_id) if event.team_id else None
            group_id = str(event.group_id) if event.group_id else None
            notif_content = f"Unify message sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "unify_message_sent",
                f"Message to {sender_name}: {event.content}",
            )
        case UnifyMessageReceived():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            thread_id = str(event.thread_id) if event.thread_id else None
            team_id = str(event.team_id) if event.team_id else None
            group_id = str(event.group_id) if event.group_id else None
            mentions = event.mentions
            notif_content = f"Unify message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "unify_message_received",
                f"Message from {sender_name}: {event.content}",
            )
            log_startup_timing(
                LOGGER,
                "⏱️ [StartupTiming] first_reply.message_received medium=unify_message contact_id=%s sender=%s",
                contact_id,
                sender_name,
            )
        case ApiMessageSent():
            medium = Medium.API_MESSAGE
            message_content = event.content
            attachments = event.attachments
            tags = event.tags
            notif_content = f"API response sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "api_message_sent",
                f"API response to {sender_name}: {event.content}",
            )
        case ApiMessageReceived():
            medium = Medium.API_MESSAGE
            message_content = event.content
            attachments = event.attachments
            tags = event.tags
            notif_content = f"API message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "api_message_received",
                f"API message from {sender_name}: {event.content}",
            )
            if event.api_message_id:
                cm._pending_api_message_id = event.api_message_id
                cm._pending_api_message_tags = event.tags
        case DiscordMessageSent():
            medium = Medium.DISCORD_MESSAGE
            message_content = event.content
            notif_content = f"Discord DM sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "discord_message_sent",
                f"Discord DM to {sender_name}: {event.content}",
            )
        case DiscordMessageReceived():
            medium = Medium.DISCORD_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Discord DM from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "discord_message_received",
                f"Discord DM from {sender_name}: {event.content}",
            )
        case DiscordChannelMessageSent():
            medium = Medium.DISCORD_CHANNEL_MESSAGE
            message_content = event.content
            channel_id = getattr(event, "channel_id", "") or None
            guild_id = getattr(event, "guild_id", "") or None
            notif_content = "Discord channel message sent"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "discord_channel_message_sent",
                f"Discord channel message: {event.content}",
            )
        case DiscordChannelMessageReceived():
            medium = Medium.DISCORD_CHANNEL_MESSAGE
            message_content = event.content
            attachments = event.attachments
            channel_id = getattr(event, "channel_id", "") or None
            guild_id = getattr(event, "guild_id", "") or None
            bot_id = getattr(event, "bot_id", "") or None
            message_id = getattr(event, "message_id", "") or None
            notif_content = f"Discord channel message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "discord_channel_message_received",
                f"Discord channel from {sender_name}: {event.content}",
            )
        case SlackMessageSent():
            medium = Medium.SLACK_MESSAGE
            message_content = event.content
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            thread_ts = getattr(event, "thread_ts", "") or None
            notif_content = f"Slack DM sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "slack_message_sent",
                f"Slack DM to {sender_name}: {event.content}",
            )
        case SlackMessageReceived():
            medium = Medium.SLACK_MESSAGE
            message_content = event.content
            attachments = event.attachments
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            thread_ts = getattr(event, "thread_ts", "") or None
            event_ts = getattr(event, "event_ts", "") or None
            message_id = getattr(event, "message_id", "") or None
            routing_metadata = getattr(event, "routing_metadata", None) or None
            notif_content = f"Slack DM from {sender_name}"
            role = "user"
            _adopt_slack_bot_user_id(cm, event)
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "slack_message_received",
                f"Slack DM from {sender_name}: {event.content}",
            )
        case SlackChannelMessageSent():
            medium = Medium.SLACK_CHANNEL_MESSAGE
            message_content = event.content
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            thread_ts = getattr(event, "thread_ts", "") or None
            notif_content = "Slack channel message sent"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "slack_channel_message_sent",
                f"Slack channel message: {event.content}",
            )
        case SlackChannelMessageReceived():
            medium = Medium.SLACK_CHANNEL_MESSAGE
            message_content = event.content
            attachments = event.attachments
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            thread_ts = getattr(event, "thread_ts", "") or None
            event_ts = getattr(event, "event_ts", "") or None
            message_id = getattr(event, "message_id", "") or None
            routing_metadata = getattr(event, "routing_metadata", None) or None
            notif_content = f"Slack channel message from {sender_name}"
            role = "user"
            _adopt_slack_bot_user_id(cm, event)
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "slack_channel_message_received",
                f"Slack channel from {sender_name}: {event.content}",
            )
        case TeamsMessageSent():
            medium = Medium.TEAMS_MESSAGE
            message_content = event.content
            attachments = event.attachments
            chat_id = getattr(event, "chat_id", "") or None
            notif_content = f"Teams message sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "teams_message_sent",
                f"Teams chat to {sender_name}: {event.content}",
            )
        case MsTeamsBotMessageSent():
            medium = Medium.MS_TEAMS_BOT_MESSAGE
            message_content = event.content
            chat_id = getattr(event, "conversation_id", "") or None
            notif_content = f"Teams bot message sent to {sender_name}"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "ms_teams_bot_message_sent",
                f"Teams bot message to {sender_name}: {event.content}",
            )
        case MsTeamsBotMessageReceived():
            # A routed bot activity proves the org has a bound install, so adopt
            # the capability on the live CM. This registers
            # send_ms_teams_bot_message for the responding turn even when the
            # boot-time activation payload didn't carry the flag.
            cm.assistant_has_ms_teams_bot = True
            medium = Medium.MS_TEAMS_BOT_MESSAGE
            message_content = event.content
            attachments = event.attachments
            # tenant_id + conversation_id are the routing keys the reply needs
            # to target the same Teams conversation via send_ms_teams_bot_message.
            tenant_id = getattr(event, "tenant_id", "") or None
            conversation_id = getattr(event, "conversation_id", "") or None
            chat_id = conversation_id
            channel_id = getattr(event, "channel_id", "") or None
            message_id = getattr(event, "message_id", "") or None
            routing_metadata = getattr(event, "routing_metadata", None) or None
            notif_content = f"Teams bot message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "ms_teams_bot_message_received",
                f"Teams bot message from {sender_name}: {event.content}",
            )
        case MsTeamsBotChannelMessageSent():
            medium = Medium.MS_TEAMS_BOT_CHANNEL_MESSAGE
            message_content = event.content
            chat_id = getattr(event, "conversation_id", "") or None
            notif_content = "Teams bot channel message sent"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "ms_teams_bot_channel_message_sent",
                f"Teams bot channel message to {sender_name}: {event.content}",
            )
        case MsTeamsBotChannelMessageReceived():
            # A routed channel activity proves the org has a bound install, so
            # adopt the capability on the live CM (registers the send tools for
            # the responding turn even if the boot payload lacked the flag).
            cm.assistant_has_ms_teams_bot = True
            medium = Medium.MS_TEAMS_BOT_CHANNEL_MESSAGE
            message_content = event.content
            attachments = event.attachments
            # tenant_id + conversation_id are the routing keys the reply needs
            # to target the same Teams conversation via
            # send_ms_teams_bot_channel_message.
            tenant_id = getattr(event, "tenant_id", "") or None
            conversation_id = getattr(event, "conversation_id", "") or None
            chat_id = conversation_id
            channel_id = getattr(event, "channel_id", "") or None
            team_id = getattr(event, "team_id", "") or None
            thread_id = getattr(event, "thread_id", "") or None
            message_id = getattr(event, "message_id", "") or None
            routing_metadata = getattr(event, "routing_metadata", None) or None
            notif_content = f"Teams bot channel message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "ms_teams_bot_channel_message_received",
                f"Teams bot channel message from {sender_name}: {event.content}",
            )
        case TeamsMessageReceived():
            medium = Medium.TEAMS_MESSAGE
            message_content = event.content
            attachments = event.attachments
            chat_id = getattr(event, "chat_id", "") or None
            message_id = getattr(event, "message_id", "") or None
            notif_content = f"Teams message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "teams_message_received",
                f"Teams chat from {sender_name}: {event.content}",
            )
        case TeamsChannelMessageSent():
            medium = Medium.TEAMS_CHANNEL_MESSAGE
            message_content = event.content
            attachments = event.attachments
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            notif_content = "Teams channel message sent"
            role = "assistant"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "teams_channel_message_sent",
                f"Teams channel message: {event.content}",
            )
        case TeamsChannelMessageReceived():
            medium = Medium.TEAMS_CHANNEL_MESSAGE
            message_content = event.content
            attachments = event.attachments
            team_id = getattr(event, "team_id", "") or None
            channel_id = getattr(event, "channel_id", "") or None
            thread_id = getattr(event, "thread_id", "") or None
            message_id = getattr(event, "message_id", "") or None
            notif_content = f"Teams channel message from {sender_name}"
            role = "user"
            event_trace = getattr(cm, "_current_event_trace", None) or {}
            cm._session_logger.info(
                "teams_channel_message_received",
                f"Teams channel from {sender_name}: {event.content}",
            )

    # Non-email messages: push to single contact only
    if contact_id is not None:
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=medium,
            message_content=message_content,
            attachments=attachments,
            timestamp=event.timestamp,
            role=role,
            tags=tags,
            chat_id=chat_id,
            channel_id=channel_id,
            team_id=team_id,
            thread_id=thread_id,
            group_id=group_id,
            thread_ts=thread_ts,
            event_ts=event_ts,
            message_id=message_id,
            routing_metadata=routing_metadata,
            guild_id=guild_id,
            bot_id=bot_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            mentions=mentions,
        )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
    if role == "user":
        _t0 = time.perf_counter()
        await _surface_trigger_task_candidates(
            cm=cm,
            event=event,
            medium=medium,
            contact_id=contact_id,
            sender_name=sender_name,
            timestamp=event.timestamp,
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.surface_trigger_candidates duration=%.2fs medium=%s contact_id=%s",
            time.perf_counter() - _t0,
            medium.value,
            contact_id,
        )

    if role == "user":
        _t0 = time.perf_counter()
        await cm.cancel_proactive_speech()
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.cancel_proactive_speech duration=%.2fs medium=%s",
            time.perf_counter() - _t0,
            medium.value,
        )

    if role == "user" or _should_wake_for_outbound_sent(event):
        _t0 = time.perf_counter()
        inbound_reply_context = (
            _credit_gate_reply_context(
                medium=medium,
                contact_id=contact_id,
                api_message_id=getattr(event, "api_message_id", None),
                tags=tags,
                chat_id=chat_id,
                channel_id=channel_id,
                team_id=team_id,
                thread_id=thread_id,
                group_id=group_id,
                thread_ts=thread_ts,
                guild_id=guild_id,
                bot_id=bot_id,
                message_id=message_id,
                routing_metadata=routing_metadata,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )
            if role == "user"
            else None
        )
        if inbound_reply_context is not None:
            cm.record_last_inbound_reply(inbound_reply_context)
        await cm.request_llm_run(
            triggering_contact_id=contact_id,
            credit_gate_reply_context=inbound_reply_context,
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.request_llm_run_call duration=%.2fs medium=%s role=%s",
            time.perf_counter() - _t0,
            medium.value,
            role,
        )


@EventHandler.register(Error)
async def _(event: Error, cm: "ConversationManager", *args, **kwargs):
    """Surface comms errors to the brain via the notification bar.

    When an outbound comms tool fails (send_sms, send_email, make_call, etc.),
    it publishes an Error event. Without a handler, the error is silently
    dropped and the brain never learns about the failure.

    The detailed error context is already pushed into the conversation thread
    by the tool itself (via ``_surface_comms_error``). This handler adds a
    lightweight notification and triggers a follow-up brain turn so the brain
    can see the failure and decide how to recover.
    """
    cm.notifications_bar.push_notif("Error", event.message, event.timestamp)
    await cm.request_llm_run(delay=0)


@EventHandler.register(BackupContactsEvent)
async def _(event: BackupContactsEvent, cm: "ConversationManager", *args, **kwargs):
    """
    Cache contacts from inbound messages for quick lookup.

    This handler is triggered when inbound messages arrive with contact data.
    Contacts are cached in ContactIndex and checked first in get_contact(),
    ensuring contacts from recent inbounds are always available even before
    or during ContactManager initialization.
    """
    if cm.contact_index._contact_manager:
        return
    cm._session_logger.debug(
        "backup_contacts",
        f"Caching {len(event.contacts)} contacts from inbound",
    )
    cm.contact_index.set_fallback_contacts(event.contacts)


@EventHandler.register(UnknownContactCreated)
async def _(event: UnknownContactCreated, cm: "ConversationManager", *args, **kwargs):
    """
    Handle new unknown contact creation from inbound messages.

    When an inbound SMS, email, or call arrives from an unknown sender (not in
    Contacts and not in BlackList), a minimal contact is automatically created
    with should_respond=False. This event notifies the ConversationManager so
    it can inform the boss and seek guidance on how to handle the contact.

    The assistant should use its judgement to decide the best course of action:
    - Inform the boss and ask for guidance
    - If clearly spam, potentially blacklist the contact
    - If legitimate, update contact details and enable responses
    """
    contact = event.contact
    contact_name = (
        contact.get("first_name")
        or contact.get("phone_number")
        or contact.get("email_address")
        or "Unknown"
    )

    cm._session_logger.info(
        "unknown_contact_created",
        f"New unknown contact created: {contact_name} via {event.medium}",
    )

    # Push notification so the assistant is aware
    notif_content = f"New unknown contact from {event.medium}: {contact_name}"
    if event.message_preview:
        notif_content += (
            f" - '{event.message_preview[:50]}...'"
            if len(event.message_preview) > 50
            else f" - '{event.message_preview}'"
        )
    cm.notifications_bar.push_notif("contacts", notif_content, event.timestamp)

    # Trigger LLM run so assistant can decide how to handle
    await cm.request_llm_run(
        triggering_contact_id=contact.get("contact_id"),
    )


async def _startup_sequence(cm: "ConversationManager", medium: str = ""):
    """Run job startup logging.

    VM readiness, desktop session warm-up, and file sync are now driven by
    the ``AssistantDesktopReady`` event handler rather than polling.
    """
    await asyncio.to_thread(
        assistant_jobs.log_job_startup,
        job_name=cm.job_name,
        user_id=cm.user_id,
        assistant_id=cm.assistant_id,
        medium=medium,
    )


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    try:
        cm._session_logger.info("startup", "Received startup event")
        cm._startup_wake_reasons = list(event.wake_reasons or [])
        if cm._startup_wake_reasons:
            wake_reason_types = sorted(
                {
                    str(reason.get("type") or "unknown")
                    for reason in cm._startup_wake_reasons
                    if isinstance(reason, dict)
                },
            )
            cm._session_logger.info(
                "startup",
                (
                    f"Stored {len(cm._startup_wake_reasons)} startup wake reason(s) "
                    f"of type(s): {', '.join(wake_reason_types) or 'unknown'}"
                ),
            )

        payload = event.to_dict()["payload"]
        previous_unify_key = SESSION_DETAILS.unify_key
        cm.set_details(payload)

        # Restart agent-service with the user's API key.
        # The agent-service is a sibling Node.js process the CM spawns in-pod on
        # localhost:3000 (entrypoint.sh only owns its teardown via
        # stop_agent_service). It first comes up with the container's original
        # UNIFY_KEY. set_details() + export_to_env() update os.environ in this
        # Python process, but the Node.js process still has the old key. Kill and
        # respawn so auth and LLM billing use the user's key.
        asyncio.create_task(
            asyncio.to_thread(
                _restart_agent_service_with_key,
                SESSION_DETAILS.unify_key,
            ),
        )

        cm.call_manager.set_config(cm.get_call_config())
        try:
            await cm.call_manager.refresh_persistent_worker_after_key_change(
                previous_unify_key,
                SESSION_DETAILS.unify_key,
            )
        except Exception as e:
            LOGGER.error(
                "LiveKit worker failed to start, voice calls unavailable: %s",
                e,
            )

        asyncio.create_task(_startup_sequence(cm, medium=event.medium))

        # Manager initialization runs in parallel
        asyncio.create_task(managers_utils.init_conv_manager(cm))
        asyncio.create_task(managers_utils.listen_to_operations(cm))
        _schedule_startup_integration_sync(cm)
    except Exception as e:
        cm._session_logger.error("startup", f"Error in startup sequence: {e}")
        traceback.print_exc()
        publish_system_error(
            "The assistant failed to start up. Please try again shortly.",
            error_type="startup_failed",
        )


@EventHandler.register(AssistantUpdateEvent)
async def _(event: AssistantUpdateEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("assistant_update", "Received assistant update event")
    if event.update_kind == "membership":
        team_ids = sorted(set(event.team_ids or []))
        SESSION_DETAILS.team_ids = team_ids
        SESSION_DETAILS.team_summaries = event.team_summaries or []
        SESSION_DETAILS.self_contact_id = event.self_contact_id
        SESSION_DETAILS.boss_contact_id = event.boss_contact_id
        SESSION_DETAILS.export_team_ids_to_env()
        SESSION_DETAILS.export_team_summaries_to_env()
        SESSION_DETAILS.export_contact_ids_to_env()
        cm.team_ids = team_ids
        cm.team_summaries = SESSION_DETAILS.team_summaries
        cm.self_contact_id = event.self_contact_id
        cm.boss_contact_id = event.boss_contact_id
        ContextRegistry.forget_departed_team_roots(team_ids)
        broader_context.reset()
        invalidate_task_machine_state_reads()
        return

    payload = event.to_dict()["payload"]
    old_key = SESSION_DETAILS.unify_key
    cm.set_details(payload)

    if SESSION_DETAILS.unify_key != old_key:
        asyncio.create_task(
            asyncio.to_thread(
                _restart_agent_service_with_key,
                SESSION_DETAILS.unify_key,
            ),
        )

    cm.call_manager.set_config(cm.get_call_config())
    try:
        await cm.call_manager.refresh_persistent_worker_after_key_change(
            old_key,
            SESSION_DETAILS.unify_key,
        )
    except Exception as e:
        LOGGER.error(
            "LiveKit worker failed to refresh after assistant update: %s",
            e,
        )

    # Sync OAuth tokens that may have changed after a re-authorization
    await managers_utils.queue_operation(managers_utils.sync_assistant_secrets)

    contact_sync_args = (
        cm,
        event.assistant_first_name,
        event.assistant_surname,
        event.assistant_number,
        event.assistant_email,
        event.user_first_name,
        event.user_surname,
        event.user_number,
        event.user_email,
        event.assistant_whatsapp_number,
        event.user_whatsapp_number,
        getattr(event, "assistant_job_title", "") or "",
    )
    # A profile contact update is immediately followed by onboarding actions that
    # read ContactManager. When the manager is live, sync before returning from
    # this event so later sends cannot observe stale boss contact details.
    if cm.contact_manager is not None:
        await managers_utils.update_session_contacts(*contact_sync_args)
    else:
        await managers_utils.queue_operation(
            managers_utils.update_session_contacts,
            *contact_sync_args,
        )

    from unify.settings import SETTINGS

    if (
        cm.is_coordinator
        and cm.coordinator_onboarding_active
        and SETTINGS.UNIFY_CONSOLE_UI
    ):
        cm._coordinator_state_checked_at = 0.0
        await cm._refresh_coordinator_onboarding_state()

    # Updates can carry wake reasons too (e.g. the multiplayer flip's
    # announcement for a pod that was already live). Replay through the
    # same chain startup uses; when managers are still initializing the
    # reasons queue up for the post-init replay instead.
    if event.wake_reasons:
        cm._startup_wake_reasons = [
            *(getattr(cm, "_startup_wake_reasons", None) or []),
            *event.wake_reasons,
        ]
        if cm.initialized:
            await _consume_startup_wake_reasons(cm)


@EventHandler.register(GetChatHistory)
async def _(event: GetChatHistory, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "state_update",
        f"Received chat history ({len(event.chat_history)} messages)",
    )
    cm.chat_history = event.chat_history + cm.chat_history


@EventHandler.register(ActorHandleStarted)
async def _(event: ActorHandleStarted, cm: "ConversationManager", *args, **kwargs):
    pass


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_injected",
        f"Notification: {event.content[:50]}...",
    )

    cm.notifications_bar.push_notif(
        event.source,
        event.content,
        event.timestamp,
        pinned=event.pinned,
        id=event.interjection_id,
    )

    await cm.schedule_proactive_speech()
    await cm.request_llm_run(delay=0)


@EventHandler.register(TaskDue)
async def _(
    event: TaskDue,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_task_due_event(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(TaskTriggerRequested)
async def _(
    event: TaskTriggerRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_task_trigger_requested_event(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(CanvasInvocationRequested)
async def _(
    event: CanvasInvocationRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # No `request_llm_run`: the run is deterministic work whose outcome is written
    # to the invocation row and streamed to the canvas, so waking the slow brain
    # would add a conversational turn nobody asked for.
    await handle_canvas_invocation_requested(event, cm)


@EventHandler.register(WorkflowRequestRequested)
async def _(
    event: WorkflowRequestRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # No `request_llm_run`: reconciling an installation is deterministic work
    # whose outcome is written to the request row Console reads, so waking the
    # slow brain would add a conversational turn nobody asked for.
    await handle_workflow_request_requested(event, cm)


@EventHandler.register(ProviderEventDispatchRequested)
async def _(
    event: ProviderEventDispatchRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_provider_event_dispatch_requested_event(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(AssistantPresenceObserved)
async def _(
    event: AssistantPresenceObserved,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.record_console_presence(
        version=event.console_guidance_version,
        brief=event.console_guidance_brief,
        full=event.console_guidance_full,
        actions=event.console_action_catalogue,
    )
    if hasattr(cm, "_session_logger"):
        cm._session_logger.debug(
            "assistant_presence_observed",
            f"Assistant presence observed from {event.source or 'console'}.",
        )


@EventHandler.register(ConsoleScriptResult)
async def _(
    event: ConsoleScriptResult,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # Only a move that failed is worth a turn. A landed one is recorded for the
    # next one, because the assistant already said what it was doing.
    if await handle_console_script_result(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(CoordinatorDelegate)
async def _(
    event: CoordinatorDelegate,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_coordinator_delegate_event(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(CoordinatorOnboardingEvent)
async def _(
    event: CoordinatorOnboardingEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # Drop the event into the notifications bar via the domain helper
    # and trigger an immediate LLM run so the brain composes the
    # acknowledgement while the user's action is still in the
    # foreground (a delayed narration during onboarding reads as stale).
    if await _handle_coordinator_onboarding_event(event, cm):
        request_id = await cm.request_llm_run(delay=0)
        cm.set_pending_onboarding_request_id(request_id)


@EventHandler.register(IntegrationToolsSyncRequested)
async def _(
    event: IntegrationToolsSyncRequested,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_integration_tools_sync_requested(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(IntegrationToolsSyncCompleted)
async def _(
    event: IntegrationToolsSyncCompleted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    # A workflow held on this app is armed by a reconcile, and the reconcile
    # that arms it is a repeat install. Doing it here is what makes the
    # Console promise ("connecting arms held jobs") true without the user
    # going back to reinstall. Best-effort: a failure here must not cost the
    # tool sync its own turn.
    try:
        await arm_workflows_for_connected_app(event.app_slug)
    except Exception:
        LOGGER.exception(
            "Could not re-check workflow requirements after %s connected",
            event.app_slug,
        )

    if await _handle_integration_tools_sync_completed(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(IntegrationToolsSyncFailed)
async def _(
    event: IntegrationToolsSyncFailed,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if await _handle_integration_tools_sync_failed(event, cm):
        await cm.request_llm_run(delay=0)


@EventHandler.register(NotificationUnpinnedEvent)
async def _(
    event: NotificationUnpinnedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_unpinned",
        f"Unpinned interjection: {event.interjection_id}",
    )

    cm.notifications_bar.remove_notif(event.interjection_id)


@EventHandler.register(ActorResult)
async def _(event: ActorResult, cm: "ConversationManager", *args, **kwargs):
    from unify.common.prompt_helpers import now as prompt_now

    action_data = cm.in_flight_actions.get(event.handle_id, {})
    action_type = event.action_type or action_data.get("action_type") or "act"
    completion_entry = {
        "action_name": "act_completed" if event.success else "act_failed",
        "query": event.result if event.success else (event.error or event.result),
        "timestamp": prompt_now(),
        "success": bool(event.success),
        "action_type": action_type,
    }
    if event.result is not None:
        completion_entry["result"] = event.result
    if event.error:
        completion_entry["error"] = event.error

    # Log completion in handle_actions before moving to completed_actions.
    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(completion_entry)

    # Move to completed_actions (preserves handle for post-completion ask queries)
    completed = cm.in_flight_actions.pop(event.handle_id, None)
    if completed:
        cm.completed_actions[event.handle_id] = completed
    await cm.request_llm_run()


@EventHandler.register(ActorSessionResponse)
async def _(event: ActorSessionResponse, cm: "ConversationManager", *args, **kwargs):
    """A persistent session completed a turn and is awaiting input.

    This is semantically distinct from ``ActorNotification`` (progress update):
    a response means the actor is *done with this turn* and will not proceed
    until the brain interjects with the next instruction.
    """
    action_data = cm.in_flight_actions.get(event.handle_id, {})

    from unify.common.prompt_helpers import now as prompt_now

    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(
            {
                "action_name": "response",
                "query": event.content,
                "status": "awaiting_input",
                "timestamp": prompt_now(),
            },
        )
    await cm.request_llm_run()


@EventHandler.register(ActorNotification)
async def _(event: ActorNotification, cm: "ConversationManager", *args, **kwargs):
    """A progress notification from an in-flight action.

    Unlike ``ActorResponse``, notifications arrive while the actor is still
    working.  Progress is recorded in the action's history so the slow brain
    sees accumulated progress when it next runs on a legitimate event.

    The fast brain receives actor progress via ``_render_boss_notifications``
    and the ``NotificationReplyEvaluator`` decides whether to speak.

    A notification can also arrive *late* -- after the handle has already
    moved from ``in_flight_actions`` to ``completed_actions`` (e.g. the
    StorageCheck phase finishing well after ``ActorResult`` resolved the
    action). Record those too instead of silently dropping them, so the
    handle's history reflects everything that actually happened.
    """
    from unify.common.prompt_helpers import now as prompt_now

    entry = {
        "action_name": "progress",
        "query": event.response,
        "timestamp": prompt_now(),
    }
    action_data = cm.in_flight_actions.get(event.handle_id) or cm.completed_actions.get(
        event.handle_id,
    )
    if action_data and "handle_actions" in action_data:
        action_data["handle_actions"].append(entry)

    # ponytail: single-consumer wake gate. If a second consumer needs to
    # react to a distinct notification `kind`, upgrade this to a dedicated
    # event class instead of growing this branch.
    if event.kind == "storage_review_complete" and cm.learning_demo_storage_wake_armed:
        await cm.request_llm_run()


@EventHandler.register(ComputerActCompleted)
async def _(
    event: ComputerActCompleted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """A visible computer session's act() call completed somewhere in the
    system. Push a notification and wake the slow brain so it can react."""
    snippet = event.summary[:120] if event.summary else event.instruction[:120]
    cm.notifications_bar.push_notif(
        "Computer",
        f"Computer action executed: {snippet}",
        event.timestamp,
    )
    await cm.request_llm_run()


@EventHandler.register(SyncContacts)
async def _(
    event: SyncContacts,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "state_update",
        f"SyncContacts: {event.reason or 'no reason'}",
    )

    async def _sync_contacts():
        try:
            await asyncio.to_thread(cm.contact_manager._sync_required_contacts)
            cm._session_logger.info("state_update", "Contacts synced successfully")
        except Exception as e:
            cm._session_logger.error("state_update", f"Error syncing contacts: {e}")
        try:
            await asyncio.to_thread(cm.contact_manager.sync_manual_voice_enrollment)
        except Exception as e:
            cm._session_logger.error(
                "state_update",
                f"Error syncing manual voice enrollment: {e}",
            )
        cm.notifications_bar.push_notif(
            "System",
            f"Contacts synced: {event.reason or 'manual sync'}",
            event.timestamp,
        )

    await managers_utils.queue_operation(_sync_contacts)


def _recent_conversation_snippet(cm: "ConversationManager", n: int = 4) -> str | None:
    """Extract the last *n* user/assistant messages from the global thread.

    Returns a compact multi-line string or None if no messages are available.
    """
    from unify.conversation_manager.domains.contact_index import CommsMessage

    lines: list[str] = []
    for entry in reversed(cm.contact_index.global_thread):
        if not isinstance(entry.message, CommsMessage):
            continue
        msg = entry.message
        content = getattr(msg, "content", None) or getattr(msg, "body", None) or ""
        content = content.strip()
        if not content or (content.startswith("<") and content.endswith(">")):
            continue
        role = getattr(msg, "role", "user")
        lines.append(f"{role}: {content}")
        if len(lines) >= n:
            break
    if not lines:
        return None
    lines.reverse()
    return "\n".join(lines)


async def _ensure_desktop_session(cm: "ConversationManager") -> None:
    """Create a desktop session in agent-service if one doesn't already exist.

    Sessions are lazy (created on first ``get_session`` call), so this must be
    called explicitly to guarantee the ``/screenshot`` endpoint has an active
    session to fall back to.  ``get_session`` is idempotent — calling it when a
    session already exists returns the cached instance.

    Retries with exponential backoff because the VM's Caddy reverse proxy may
    still be starting up or obtaining its TLS certificate from Let's Encrypt
    even after the Communication service reports the VM as "ready".
    """
    from unify.function_manager.primitives.runtime import ComputerPrimitives
    from unify.manager_registry import ManagerRegistry

    cp = ManagerRegistry.get_instance(ComputerPrimitives)
    if cp is None:
        return

    max_attempts = 12
    base_delay = 5.0
    max_delay = 30.0
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            session = await cp.backend.get_session("desktop")
            cm._session_logger.info(
                "desktop_session",
                f"Desktop session ready: {session._session_id}",
            )
            return
        except Exception as e:
            if attempt == max_attempts:
                cm._session_logger.warning(
                    "desktop_session",
                    f"Failed to create desktop session after {max_attempts} attempts: "
                    f"{type(e).__name__}: {e}",
                )
                return
            cm._session_logger.debug(
                "desktop_session",
                f"Attempt {attempt}/{max_attempts} failed ({type(e).__name__}), "
                f"retrying in {delay:.0f}s",
            )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            cp.backend.clear_session("desktop")


# --------------------------------------------------------------------------- #
# Desktop Lifecycle Events
# --------------------------------------------------------------------------- #


@EventHandler.register((AssistantDesktopReady,))
async def _(
    event: AssistantDesktopReady,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    from unify.session_details import SESSION_DETAILS

    current_binding_id = SESSION_DETAILS.assistant.binding_id or ""
    if current_binding_id and not event.binding_id:
        cm._session_logger.info(
            "desktop_ready_missing_binding",
            "Ignoring desktop_ready without binding_id because the session already has a current binding",
        )
        return
    if (
        event.binding_id
        and current_binding_id
        and event.binding_id != current_binding_id
    ):
        cm._session_logger.info(
            "desktop_ready_stale",
            f"Ignoring stale desktop_ready for binding {event.binding_id}; current binding is {current_binding_id}",
        )
        return

    resolved_binding_id = event.binding_id or current_binding_id
    browser_desktop_url, api_desktop_url = resolve_desktop_urls(event.desktop_url)

    cm._session_logger.info(
        "desktop_ready",
        f"VM ready: {event.vm_type} at {api_desktop_url or browser_desktop_url}",
    )
    log_startup_timing(
        LOGGER,
        "⏱️ [StartupTiming] desktop_ready.received vm_type=%s desktop_url_present=%s",
        event.vm_type,
        bool(api_desktop_url or browser_desktop_url),
    )

    await apply_managed_desktop_ready(
        cm,
        binding_id=resolved_binding_id,
        browser_desktop_url=browser_desktop_url,
        api_desktop_url=api_desktop_url,
        vm_type=event.vm_type,
        desktop_secret=event.desktop_secret,
        timestamp=event.timestamp,
    )


@EventHandler.register((FileSyncComplete,))
async def _(
    event: "FileSyncComplete",
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.file_sync_complete = True
    cm.notifications_bar.push_notif(
        "System",
        "File sync complete — all files from previous sessions are now available on disk.",
        event.timestamp,
    )
    cm._session_logger.debug("file_sync", "File sync complete")
    await cm.request_llm_run(delay=0)


OPEN_SLOW_BRAIN_TURN_NOTIFICATION = (
    "Open slow-brain turn — your previous turn finished without calling "
    "`wait`. You have another thinking turn now. Continue any outstanding "
    "work (reply to the user, send onboarding deliverables, follow up on "
    "in-flight actions). Recurring turns keep opening until you explicitly "
    "call `wait()` or `wait(delay=…)`. Do not call `wait` while the user is "
    "still waiting on you in chat."
)


@EventHandler.register((OpenSlowBrainTurn,))
async def _(
    event: OpenSlowBrainTurn,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.notifications_bar.push_notif(
        "System",
        OPEN_SLOW_BRAIN_TURN_NOTIFICATION,
        event.timestamp,
    )
    await cm.request_llm_run(delay=0)


# Notification text shown to the slow brain when initialization completes.
#
# Wording is deliberately strong about preferring `wait` over a follow-up
# message: in cold-start flows where the brain already replied to a user
# message during pre-init, the original "review … and follow up if needed
# (correct, elaborate, or confirm)" wording was being interpreted as
# permission to send a rephrased duplicate. We still need to allow the
# brain to follow up legitimately when it deferred work, gave an
# incomplete/incorrect answer due to missing context, or has new history
# from hydration — so the directive enumerates those cases explicitly and
# tells the brain to call `wait` otherwise.
INITIALIZATION_COMPLETE_NOTIFICATION = (
    "Initialization complete — all actions are now available and full "
    "conversation history has been loaded. If your previous reply during "
    "initialization (a) deferred work you can now perform, (b) was "
    "incorrect or incomplete because of missing context now revealed by "
    "hydrated history, or (c) needs a concrete update, follow up now "
    "(act, correct, or elaborate). Otherwise call wait — do NOT send a "
    "message that simply rephrases, restates, or confirms a reply you "
    "already gave."
)

# The truthful variant for a boot whose hydration restored nothing —
# because there genuinely is no prior conversation, or because the
# deployment does not persist the Comms stream (EVENTBUS_PUBLISHING_ENABLED
# off). Claiming "history has been loaded" on such a boot sent brains
# hunting for context that was never coming (observed 2026-08-22: an
# assistant told history was loaded, with an empty thread render, searched
# email and desktop for a spec that lived in the unrendered conversation).
INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION = (
    "Initialization complete — all actions are now available. No prior "
    "conversation history was found to load, so the rendered threads are "
    "the whole of what is known here. If your previous reply during "
    "initialization deferred work you can now perform or needs a concrete "
    "update, follow up now. Otherwise call wait — do NOT send a message "
    "that simply rephrases, restates, or confirms a reply you already "
    "gave."
)


@EventHandler.register((InitializationComplete,))
async def _(
    event: "InitializationComplete",
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    hydrated_count = int(getattr(cm, "_hydrated_history_count", 0) or 0)
    cm.notifications_bar.push_notif(
        "System",
        (
            INITIALIZATION_COMPLETE_NOTIFICATION
            if hydrated_count
            else INITIALIZATION_COMPLETE_NO_HISTORY_NOTIFICATION
        ),
        event.timestamp,
    )
    cm._session_logger.debug("initialization", "Initialization complete")

    await bootstrap_managed_desktop_on_startup(cm)

    # Notify the fast brain (voice agent) directly via the call manager's
    # IPC socket — the same channel used for meet_interaction and other
    # direct fast brain notifications.  This bypasses the CM's own event
    # broker subscription so the notification reaches only the subprocess.
    if cm.call_manager and cm.call_manager._socket_server:
        fast_brain_notification = FastBrainNotification(
            contact={},
            message=(
                "Initialization complete — all actions are now available. "
                + (
                    "Full conversation history has been loaded."
                    if hydrated_count
                    else "No prior conversation history was found to load."
                )
            ),
            should_speak=False,
            source="initialization",
        )
        await cm.call_manager._socket_server.queue_for_clients(
            "app:call:notification",
            fast_brain_notification.to_json(),
        )

    await _consume_startup_wake_reasons(cm)
    await cm.request_llm_run(delay=0)


# --------------------------------------------------------------------------- #
# Meet Interaction Events (screen share / remote control)
# --------------------------------------------------------------------------- #

# Mapping from event class to a human-readable notification for the slow brain.
_MEET_LOG_TYPES: dict[type, str] = {
    AssistantScreenShareStarted: "screen_share",
    AssistantScreenShareStopped: "screen_share_off",
    UserScreenShareStarted: "user_screen_share",
    UserScreenShareStopped: "user_screen_share_off",
    MeetScreenShareStarted: "meet_screen_share_on",
    MeetScreenShareStopped: "meet_screen_share_off",
    UserWebcamStarted: "webcam_on",
    UserWebcamStopped: "webcam_off",
    UserRemoteControlStarted: "remote_control",
    UserRemoteControlStopped: "remote_control_off",
}

_MEET_INTERACTION_NOTIFICATIONS: dict[type, str] = {
    AssistantScreenShareStarted: "Screen sharing is on — the people you are talking to can see your desktop.",
    AssistantScreenShareStopped: "Screen sharing is off — nobody can see your desktop now.",
    UserScreenShareStarted: "The user started sharing their screen with you.",
    UserScreenShareStopped: "The user stopped sharing their screen.",
    MeetScreenShareStarted: "Someone in the meeting started sharing their screen with you.",
    MeetScreenShareStopped: "Nobody in the meeting is sharing a screen any more.",
    UserWebcamStarted: "The user enabled their webcam — you can now see them.",
    UserWebcamStopped: "The user disabled their webcam.",
    UserRemoteControlStarted: "The user took remote control of your desktop — they now have mouse and keyboard control.",
    UserRemoteControlStopped: "The user released remote control of your desktop — you may resume computer actions.",
}

# Direct guidance sent to the fast brain when screen share modes change.
# These carry behavioral instructions so the fast brain knows how to respond
# to visual references immediately, without waiting for the slow brain.
_MEET_FAST_BRAIN_GUIDANCE: dict[type, str] = {
    AssistantScreenShareStarted: (
        "Screen sharing is now ON — the people you are talking to can see your "
        "desktop, and on a call with several of them that may be more than one "
        "person. Screenshots are being captured and processed in the "
        "background. If someone references something on screen, acknowledge "
        'briefly ("Got it", "I see", "Okay") and wait — visual context will '
        "be processed shortly. Do NOT describe or guess screen contents."
    ),
    AssistantScreenShareStopped: (
        "Screen sharing is now OFF — nobody can see your desktop any more."
    ),
    UserScreenShareStarted: (
        "The user is now sharing their screen with you. Visual context is "
        "being captured in the background. If they reference something on "
        "their screen, acknowledge naturally and wait for the processed "
        "details. Do NOT guess or fabricate what is on their screen."
    ),
    UserScreenShareStopped: ("The user stopped sharing their screen."),
    MeetScreenShareStarted: (
        "Someone in the meeting is sharing their screen. Snapshots of it are "
        "arriving in the background. If they point at something on it, "
        "acknowledge naturally and wait for the details. Do NOT guess what is "
        "on it."
    ),
    MeetScreenShareStopped: (
        "The shared screen is gone. Anything you saw on it is history now, not "
        "something you can still look at."
    ),
    UserWebcamStarted: (
        "The user enabled their webcam. Visual context is being captured "
        "in the background. If they reference their appearance or something "
        "visible on camera, acknowledge naturally and wait for the processed "
        "details. Do NOT guess or fabricate what you see."
    ),
    UserWebcamStopped: ("The user disabled their webcam."),
    UserRemoteControlStarted: (
        "The user now has remote control of your desktop. Do not perform "
        "any computer actions — wait for them to release control."
    ),
    UserRemoteControlStopped: (
        "The user released remote control. You may resume computer actions."
    ),
}

# State attribute name on the CM for each toggle pair.
_MEET_STATE_FLAGS: dict[type, tuple[str, bool]] = {
    AssistantScreenShareStarted: ("assistant_screen_share_active", True),
    AssistantScreenShareStopped: ("assistant_screen_share_active", False),
    UserScreenShareStarted: ("user_screen_share_active", True),
    UserScreenShareStopped: ("user_screen_share_active", False),
    MeetScreenShareStarted: ("meet_screen_share_active", True),
    MeetScreenShareStopped: ("meet_screen_share_active", False),
    UserWebcamStarted: ("user_webcam_active", True),
    UserWebcamStopped: ("user_webcam_active", False),
    UserRemoteControlStarted: ("user_remote_control_active", True),
    UserRemoteControlStopped: ("user_remote_control_active", False),
}


async def _publish_call_desktop_share_state(cm: "ConversationManager") -> None:
    """Tell the running call script whether the desktop belongs on its stage.

    Sent on every viewer change rather than on a transition, so the room can
    always be resynchronised. Each client holds its own copy of what is mounted,
    and an edge-triggered broadcast can only correct a client that happens to be
    wrong in the direction the edge is travelling -- which is no help at all when
    the aggregate the edge is taken from never moves.

    Carries no ``message``. The call script applies ``meet_surface_state`` before
    it decides whether to surface a notification, so an empty one reaches the
    surfaces and stops there: this is a state sync, and the assistant has already
    been told in prose whatever it needed to know.
    """
    if not cm.mode.is_voice:
        return
    contact = cm.get_active_contact()
    if not contact:
        return
    from unify.conversation_manager.medium_scripts.common import (
        CALL_DESKTOP_SHARE_SURFACE,
        _resolve_agent_service_url,
    )

    notification_event = FastBrainNotification(
        contact=contact,
        message="",
        source="meet_interaction",
        agent_service_url=_resolve_agent_service_url(),
        meet_surface_state={
            CALL_DESKTOP_SHARE_SURFACE: cm.assistant_desktop_watched_from_call(
                cm.call_manager.unify_meet_call_session_id,
            ),
        },
    )
    await cm.event_broker.publish(
        "app:call:notification",
        notification_event.to_json(),
    )


@EventHandler.register(
    (
        AssistantScreenShareStarted,
        AssistantScreenShareStopped,
        UserScreenShareStarted,
        UserScreenShareStopped,
        MeetScreenShareStarted,
        MeetScreenShareStopped,
        UserWebcamStarted,
        UserWebcamStopped,
        UserRemoteControlStarted,
        UserRemoteControlStopped,
    ),
)
async def _(
    event: (
        AssistantScreenShareStarted
        | AssistantScreenShareStopped
        | UserScreenShareStarted
        | UserScreenShareStopped
        | MeetScreenShareStarted
        | MeetScreenShareStopped
        | UserWebcamStarted
        | UserWebcamStopped
        | UserRemoteControlStarted
        | UserRemoteControlStopped
    ),
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    from unify.conversation_manager.medium_scripts.common import (
        TRACK_AUTODETECT_REASON,
    )

    event_name = event.__class__.__name__
    log_type = _MEET_LOG_TYPES.get(event.__class__, "meet_interaction")
    log_msg = f"Event: {event_name}"
    track_sourced = event.reason == TRACK_AUTODETECT_REASON
    if track_sourced:
        cm._session_logger.debug(log_type, log_msg)
    else:
        cm._session_logger.info(log_type, log_msg)

    attr, value = _MEET_STATE_FLAGS[event.__class__]

    # A frontend and LiveKit track subscription both describe this surface, but
    # they measure different things: the frontend reports what the user asked
    # for, the track reports what the transport is carrying. They diverge — a
    # camera switched off in the UI can keep its track subscribed until room
    # teardown — so letting both drive the flag makes the winner a matter of
    # arrival order. The frontend owns every surface it has reported on; track
    # events survive only where no frontend ever speaks, such as the Playground.
    if track_sourced:
        if attr in cm._frontend_reported_meet_surfaces:
            return
    else:
        cm._frontend_reported_meet_surfaces.add(attr)

    if isinstance(
        event,
        (AssistantScreenShareStarted, AssistantScreenShareStopped),
    ):
        # Membership decides this surface, so the viewer set is updated even when
        # the derived flag does not move: a Desktop tab opened beside a call has
        # to be counted, or closing it would take the desktop away from the call.
        # The flag check below then suppresses the notification, which is right --
        # nothing changed that the assistant needs telling about.
        value = cm.note_assistant_screen_share_viewer(
            user_id=event.viewer_user_id,
            source=event.viewer_source,
            watching=value,
        )
        # The room's copy is restated on every one of these, including the ones
        # the flag check below drops. What a call stages is its own switch, which
        # moves independently of the desktop-scoped flag: turning the call's share
        # off while a Desktop tab stays open leaves that flag true, so an
        # edge-triggered broadcast would have no edge to fire on and would strand
        # the desktop on every participant's stage.
        await _publish_call_desktop_share_state(cm)

    # Restating the current state carries no new information: notifying again
    # would tell the assistant the screen share just started for a second time.
    if getattr(cm, attr) == value:
        return

    # Update state flag on the CM.
    setattr(cm, attr, value)

    if isinstance(event, MeetScreenShareStopped):
        # The buffer is peeked non-destructively and only cleared after a
        # successful turn, so without this the frame from a screen that has just
        # been taken down survives into the next turn and gets described as
        # current. Utterance-paired frames stay: those answer a question somebody
        # actually asked.
        for source in ("google_meet", "teams_meet"):
            dropped = cm.drop_unpaired_screenshots(source)
            if dropped:
                cm._session_logger.debug(
                    log_type,
                    f"Dropped {dropped} stale {source} screenshot(s)",
                )

    if (
        isinstance(event, (UserWebcamStarted, UserWebcamStopped))
        and not cm.mode.is_voice
    ):
        return

    # Push a notification so the slow brain sees the state change.
    notification_text = _MEET_INTERACTION_NOTIFICATIONS[event.__class__]
    cm.notifications_bar.push_notif("Meet", notification_text, event.timestamp)

    # Send direct guidance to the fast brain so it can immediately adjust
    # its conversational behavior (e.g., acknowledge visual references
    # without hallucinating). This bypasses the slow brain for instant delivery.
    fast_brain_text = _MEET_FAST_BRAIN_GUIDANCE.get(event.__class__)
    if fast_brain_text and cm.mode.is_voice:
        contact = cm.get_active_contact()
        if contact:
            cm._session_logger.debug(
                "call_notification",
                f"Publishing meet interaction reason={event_name}",
            )
            from unify.conversation_manager.medium_scripts.common import (
                _resolve_agent_service_url,
            )

            notification_event = FastBrainNotification(
                contact=contact,
                message=fast_brain_text,
                source="meet_interaction",
                agent_service_url=_resolve_agent_service_url(),
                meet_surface_state={attr: value},
            )
            await cm.event_broker.publish(
                "app:call:notification",
                notification_event.to_json(),
            )

    await cm.schedule_proactive_speech()

    if isinstance(event, AssistantScreenShareStarted):
        asyncio.ensure_future(_ensure_desktop_session(cm))

    # Broadcast remote-control state change to all active CodeActActor loops
    # via the ComputerPrimitives singleton interject queue registry.
    if isinstance(event, (UserRemoteControlStarted, UserRemoteControlStopped)):
        try:
            from unify.function_manager.primitives.runtime import ComputerPrimitives
            from unify.manager_registry import ManagerRegistry

            cp = ManagerRegistry.get_instance(ComputerPrimitives)
            if cp is not None:
                cp.set_user_remote_control(
                    isinstance(event, UserRemoteControlStarted),
                    conversation_context=_recent_conversation_snippet(cm),
                )
        except Exception:
            pass

    if isinstance(event, UserRemoteControlStopped):
        if cm.assistant_screen_share_active:
            await cm.capture_assistant_screenshot("", cached=False)


@EventHandler.register((UserFilesysAccessStarted, UserFilesysAccessStopped))
async def _(
    event: UserFilesysAccessStarted | UserFilesysAccessStopped,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    """Apply a live filesystem-access consent change to in-flight actors.

    The standing Console link already mutated server-side; this propagates the
    change to a running session so reads/writebacks stop (or resume) at once
    instead of waiting for the next session load. Targets the desktop owner's
    link via ``user_id``; broadcasting is handled inside the primitive.
    """
    user_id = (event.user_id or "").strip()
    if not user_id:
        cm._session_logger.info(
            "user_filesys_access",
            f"Event: {event.__class__.__name__} with no user_id; ignoring",
        )
        return

    started = isinstance(event, UserFilesysAccessStarted)
    cm._session_logger.info(
        "user_filesys_access",
        f"Event: {event.__class__.__name__} user_id={user_id}",
    )

    from unify.function_manager.primitives.runtime import ComputerPrimitives
    from unify.manager_registry import ManagerRegistry

    cp = ManagerRegistry.get_instance(ComputerPrimitives)
    if cp is None:
        return
    if started:
        cp.grant_user_filesys_access(user_id)
    else:
        cp.revoke_user_filesys_access(
            user_id,
            conversation_context=_recent_conversation_snippet(cm),
        )


@EventHandler.register(LogMessageResponse)
async def _(event: LogMessageResponse, cm: "ConversationManager", *args, **kwargs):
    if (
        event.medium in (Medium.PHONE_CALL, Medium.WHATSAPP_CALL)
        and cm.call_manager.call_exchange_id == UNASSIGNED
    ):
        cm.call_manager.call_exchange_id = event.exchange_id
        cm.call_manager.call_exchange_destination = event.destination
    if (
        event.medium == Medium.UNIFY_MEET
        and cm.call_manager.unify_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_meet_exchange_id = event.exchange_id
        cm.call_manager.call_exchange_destination = event.destination
    if (
        event.medium == Medium.GOOGLE_MEET
        and cm.call_manager.google_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.google_meet_exchange_id = event.exchange_id
        cm.call_manager.call_exchange_destination = event.destination
    if (
        event.medium == Medium.TEAMS_MEET
        and cm.call_manager.teams_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.teams_meet_exchange_id = event.exchange_id
        cm.call_manager.call_exchange_destination = event.destination


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)


@EventHandler.register(DirectMessageEvent)
async def _(event: DirectMessageEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info(
        "direct_message",
        f"Direct message: {event.content[:50]}...",
    )

    if cm.mode.is_voice:
        cm._session_logger.info(
            "call_notification",
            "Publishing direct-message notification",
        )
        await cm.event_broker.publish(
            "app:call:notification",
            json.dumps({"content": event.content}),
        )

    contact = cm.get_active_contact()
    contact_id = (
        contact.get("contact_id") if contact else SESSION_DETAILS.boss_contact_id
    )
    sender_name = _get_sender_name(contact)

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=_active_voice_thread_medium(cm),
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
