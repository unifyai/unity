"""
Renderer: Renders conversation state for the ConversationManager LLM.

Contact information is fetched from ContactManager (source of truth).
Conversation state (threads) is fetched from ContactIndex.

SnapshotState: Tracks constituent elements of a rendered snapshot with identity,
enabling incremental diff computation for context propagation to Actor interjections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from unity.common._async_tool.utils import get_handle_paused_state
from unity.conversation_manager.domains.contact_index import (
    ApiMessage,
    Message,
    EmailMessage,
    UnifyMessage,
    WhatsAppMessage,
    TeamsMessage,
    TeamsChannelMessage,
    GuidanceMessage,
    ConversationState,
    ContactIndex,
    GlobalThreadEntry,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.task_actions import (
    derive_short_name,
    iter_steering_tools_for_action,
    iter_steering_tools_for_completed_action,
    build_action_name,
    safe_call_id_suffix,
)
from unity.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    pass


# =============================================================================
# Timezone Helpers for Participant Awareness
# =============================================================================


def _get_current_time_in_timezone(tz_name: str) -> str:
    """Get the current time formatted for a specific timezone.

    Args:
        tz_name: IANA timezone identifier (e.g., "America/New_York")

    Returns:
        Formatted time string like "3:45 PM"
    """
    from datetime import datetime, timezone as dt_timezone
    from zoneinfo import ZoneInfo

    utc_now = datetime.now(dt_timezone.utc)
    try:
        tz_info = ZoneInfo(tz_name)
        local_dt = utc_now.astimezone(tz_info)
        return local_dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "unknown"


_assistant_tz_cache: tuple[float, str | None] | None = None
_ASSISTANT_TZ_TTL = 300  # 5 minutes — timezone changes are very rare


def _get_assistant_timezone() -> str | None:
    """Get the assistant's timezone from contact_id=0.

    Uses a module-level TTL cache to avoid synchronous HTTP round-trips to
    Orchestra on every render_state() call (which runs in the hot path of the
    event loop).

    Returns:
        IANA timezone identifier or None if not available.
    """
    global _assistant_tz_cache
    import time

    now = time.monotonic()
    if _assistant_tz_cache is not None:
        cached_at, cached_val = _assistant_tz_cache
        if now - cached_at < _ASSISTANT_TZ_TTL:
            return cached_val

    import unify as _unify
    from unity.session_details import SESSION_DETAILS

    result: str | None = None
    _contacts_ctx = (
        f"{SESSION_DETAILS.user_context}/{SESSION_DETAILS.assistant_context}/Contacts"
    )

    try:
        rows = _unify.get_logs(
            context=_contacts_ctx,
            filter="contact_id == 0",
            limit=1,
            from_fields=["timezone"],
        )
        if rows:
            val = rows[0].entries.get("timezone")
            if isinstance(val, str) and val.strip():
                result = val.strip()
    except Exception:
        pass

    _assistant_tz_cache = (now, result)
    return result


def _format_timezone_block(
    assistant_tz: str | None,
    participants: list[tuple[str, str | None]],
) -> str | None:
    """Format a timezone block showing current local times for all participants.

    Groups participants by timezone and avoids duplication.

    Format examples:
    - Same timezone: "[Now: You and Alice 2:00 PM (America/New_York)]"
    - Different: "[Now: You 2:00 PM (America/New_York) | Alice 11:00 AM (America/Los_Angeles)]"
    - Multiple same: "[Now: You, Alice, and Bob 2:00 PM (America/New_York)]"

    Args:
        assistant_tz: Assistant's timezone (IANA identifier) or None
        participants: List of (name, timezone) tuples for other participants

    Returns:
        Formatted timezone block string, or None if no timezone data
    """
    if not assistant_tz and not any(tz for _, tz in participants):
        return None

    # Build timezone -> list of names mapping
    # Include "You" (assistant) in the mapping
    tz_to_names: dict[str, list[str]] = {}
    unknown_names: list[str] = []

    if assistant_tz:
        tz_to_names[assistant_tz] = ["You"]
    else:
        unknown_names.append("You")

    for name, tz in participants:
        if tz:
            if tz not in tz_to_names:
                tz_to_names[tz] = []
            tz_to_names[tz].append(name)
        else:
            unknown_names.append(name)

    if not tz_to_names and not unknown_names:
        return None

    # Format each timezone group
    parts: list[str] = []
    for tz_name in sorted(tz_to_names.keys()):
        names = tz_to_names[tz_name]
        current_time = _get_current_time_in_timezone(tz_name)
        # Format names: "You", "You and Alice", "You, Alice, and Bob"
        if len(names) == 1:
            names_str = names[0]
        elif len(names) == 2:
            names_str = f"{names[0]} and {names[1]}"
        else:
            names_str = ", ".join(names[:-1]) + f", and {names[-1]}"
        parts.append(f"{names_str} {current_time} ({tz_name})")

    if unknown_names:
        if len(unknown_names) == 1:
            names_str = unknown_names[0]
        elif len(unknown_names) == 2:
            names_str = f"{unknown_names[0]} and {unknown_names[1]}"
        else:
            names_str = ", ".join(unknown_names[:-1]) + f", and {unknown_names[-1]}"
        parts.append(f"{names_str} (unknown timezone)")

    if not parts:
        return None

    return "[Now: " + " | ".join(parts) + "]"


def _get_message_timezone_block(
    contact_name: str,
    contact_timezone: str | None,
    assistant_timezone: str | None,
) -> str | None:
    """Get timezone block for a simple message (SMS, phone, Unify).

    Args:
        contact_name: Name of the contact
        contact_timezone: Contact's timezone (IANA identifier) or None
        assistant_timezone: Assistant's timezone or None

    Returns:
        Formatted timezone block or None
    """
    return _format_timezone_block(
        assistant_tz=assistant_timezone,
        participants=[(contact_name, contact_timezone)],
    )


def _get_email_timezone_block(
    message: "EmailMessage",
    contact_index: "ContactIndex | None",
    assistant_timezone: str | None,
) -> str | None:
    """Get timezone block for an email message.

    Looks up all recipients and groups by timezone.

    Args:
        message: The email message with to/cc/bcc recipients
        contact_index: ContactIndex for looking up contacts by email
        assistant_timezone: Assistant's timezone or None

    Returns:
        Formatted timezone block or None
    """
    if contact_index is None:
        return None

    # Collect all participant emails
    all_emails: list[str] = []
    all_emails.extend(message.to or [])
    all_emails.extend(message.cc or [])
    all_emails.extend(message.bcc or [])

    if not all_emails:
        return None

    # Look up each contact and build participants list
    participants: list[tuple[str, str | None]] = []
    seen_emails: set[str] = set()

    for email in all_emails:
        if email.lower() in seen_emails:
            continue
        seen_emails.add(email.lower())

        contact = contact_index.get_contact(email=email)
        if contact:
            first_name = contact.get("first_name") or ""
            surname = contact.get("surname") or ""
            name = f"{first_name} {surname}".strip() or email
            tz = contact.get("timezone")
            participants.append((name, tz))
        else:
            participants.append((email, None))

    if not participants:
        return None

    return _format_timezone_block(
        assistant_tz=assistant_timezone,
        participants=participants,
    )


# =============================================================================
# Snapshot State Tracking for Incremental Context Propagation
# =============================================================================


@dataclass
class MessageElement:
    """A message element with identity for diff tracking.

    Identity is based on (contact_id, thread_name, index_in_thread, timestamp).
    """

    contact_id: int
    thread_name: str
    index_in_thread: int
    timestamp: datetime
    rendered: str


@dataclass
class NotificationElement:
    """A notification element with identity for diff tracking.

    Identity is based on (timestamp, content_hash, pinned).
    """

    timestamp: datetime
    content_hash: int
    pinned: bool
    rendered: str


@dataclass
class ActionElement:
    """An in-flight action element with identity for diff tracking.

    Identity is based on handle_id. State changes (new history events,
    status changes) are tracked via history_count and status.
    """

    handle_id: int
    query: str
    status: str
    history_count: int
    rendered: str


@dataclass
class SnapshotState:
    """Tracks the constituent elements of a rendered snapshot.

    This enables computing diffs between snapshots for incremental context
    propagation. Each element type has identity tracking:

    - Messages: (contact_id, thread_name, index, timestamp)
    - Notifications: (timestamp, content_hash, pinned)
    - Actions: (handle_id, with status/history tracking for state changes)

    The full_render contains the complete rendered snapshot string.
    """

    # The complete rendered snapshot
    full_render: str

    # Constituent elements with identity
    messages: list[MessageElement] = field(default_factory=list)
    notifications: list[NotificationElement] = field(default_factory=list)
    actions: list[ActionElement] = field(default_factory=list)

    # Snapshot metadata
    snapshot_time: datetime | None = None

    def message_ids(self) -> set[tuple[int, str, int, datetime]]:
        """Return set of message identity tuples for diff comparison."""
        return {
            (m.contact_id, m.thread_name, m.index_in_thread, m.timestamp)
            for m in self.messages
        }

    def notification_ids(self) -> set[tuple[datetime, int, bool]]:
        """Return set of notification identity tuples for diff comparison."""
        return {(n.timestamp, n.content_hash, n.pinned) for n in self.notifications}

    def action_states(self) -> dict[int, tuple[str, int]]:
        """Return dict of action_id -> (status, history_count) for diff comparison."""
        return {a.handle_id: (a.status, a.history_count) for a in self.actions}


def compute_snapshot_diff(
    old_snapshot: SnapshotState | None,
    new_snapshot: SnapshotState,
) -> str:
    """Compute the incremental diff between two snapshots.

    Returns a rendered string containing only the NEW or CHANGED elements:
    - New messages (not present in old snapshot)
    - New or changed notifications (new, or pinned state changed)
    - Action state changes (status changed, new history events)

    If old_snapshot is None, returns the full new snapshot (no diff possible).

    Args:
        old_snapshot: Previous snapshot state (or None for first snapshot)
        new_snapshot: Current snapshot state

    Returns:
        Rendered string containing only incremental updates, or empty string
        if nothing changed.
    """
    if old_snapshot is None:
        return new_snapshot.full_render

    diff_parts: list[str] = []

    # Find new messages
    old_msg_ids = old_snapshot.message_ids()
    new_messages = [
        m
        for m in new_snapshot.messages
        if (m.contact_id, m.thread_name, m.index_in_thread, m.timestamp)
        not in old_msg_ids
    ]
    if new_messages:
        msg_renders = [m.rendered for m in new_messages]
        diff_parts.append(
            "<new_messages>\n" + "\n".join(msg_renders) + "\n</new_messages>",
        )

    # Find new or changed notifications
    old_notif_ids = old_snapshot.notification_ids()
    new_notifications = [
        n
        for n in new_snapshot.notifications
        if (n.timestamp, n.content_hash, n.pinned) not in old_notif_ids
    ]
    if new_notifications:
        notif_renders = [n.rendered for n in new_notifications]
        diff_parts.append(
            "<new_notifications>\n"
            + "\n".join(notif_renders)
            + "\n</new_notifications>",
        )

    # Find action state changes
    old_action_states = old_snapshot.action_states()
    action_changes = []
    for action in new_snapshot.actions:
        old_state = old_action_states.get(action.handle_id)
        if old_state is None:
            # New action
            action_changes.append(action.rendered)
        elif old_state != (action.status, action.history_count):
            # Status or history changed
            action_changes.append(action.rendered)
    if action_changes:
        diff_parts.append(
            "<action_updates>\n" + "\n".join(action_changes) + "\n</action_updates>",
        )

    if not diff_parts:
        return ""

    return "\n\n".join(diff_parts)


def _get_assistant_email_role(message: EmailMessage) -> str | None:
    """
    Determine the assistant's role in an email (To, Cc, Bcc, or sender).

    Returns a human-readable description of the assistant's role, or None
    if the assistant's email is not found in any field.
    """
    assistant_email = SESSION_DETAILS.assistant.email
    if not assistant_email:
        return None

    # Normalize for comparison (lowercase)
    assistant_email_lower = assistant_email.lower()

    # Check if assistant sent this email
    if message.role == "assistant":
        return "You sent this email"

    # Check recipient fields for incoming emails
    if any(email.lower() == assistant_email_lower for email in (message.to or [])):
        return "You were a direct recipient (To)"
    if any(email.lower() == assistant_email_lower for email in (message.cc or [])):
        return "You were CC'd"
    if any(email.lower() == assistant_email_lower for email in (message.bcc or [])):
        return "You were BCC'd"

    # Assistant not found in any field (possible for forwarded emails, etc.)
    return None


def _format_coordinator_datetime(value: Any) -> str:
    """Return a stable ISO string for Coordinator state timestamps."""

    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class Renderer:

    def render_state(
        self,
        contact_index: ContactIndex,
        notification_bar: NotificationBar = None,
        in_flight_actions: dict = None,
        completed_actions: dict = None,
        last_snapshot: datetime = None,
        max_pinned_notifications: int = 50,
        max_contact_medium_messages: int = 25,
        max_action_history_events: int = 20,
        max_completed_actions: int = 20,
        max_completed_action_history_events: int = 5,
        assistant_screen_share_active: bool = False,
        user_screen_share_active: bool = False,
        user_webcam_active: bool = False,
        user_remote_control_active: bool = False,
        google_meet_active: bool = False,
        teams_meet_active: bool = False,
        active_web_sessions: list | None = None,
        coordinator_state: dict[str, Any] | None = None,
        coordinator_checklist: list[dict[str, Any]] | None = None,
        managers_initialized: bool = True,
        vm_ready: bool = True,
        file_sync_complete: bool = True,
        has_desktop: bool = False,
    ) -> SnapshotState:
        """Render the full conversation state.

        Returns a SnapshotState containing the rendered string and constituent
        element tracking for incremental diff computation.
        """
        from unity.common.prompt_helpers import now as prompt_now

        message_elements: list[MessageElement] = []
        notification_elements: list[NotificationElement] = []
        action_elements: list[ActionElement] = []

        infra_render = self.render_infrastructure_state(
            managers_initialized=managers_initialized,
            vm_ready=vm_ready,
            file_sync_complete=file_sync_complete,
            has_desktop=has_desktop,
        )

        meet_render = self.render_meet_interaction_state(
            assistant_screen_share_active=assistant_screen_share_active,
            user_screen_share_active=user_screen_share_active,
            user_webcam_active=user_webcam_active,
            user_remote_control_active=user_remote_control_active,
            google_meet_active=google_meet_active,
            teams_meet_active=teams_meet_active,
        )

        web_sessions_render = self.render_active_web_sessions(
            active_web_sessions or [],
        )
        coordinator_render = ""
        if SESSION_DETAILS.is_coordinator:
            coordinator_render = self.render_coordinator_goal_state(
                coordinator_state=coordinator_state,
                coordinator_checklist=coordinator_checklist,
            )

        notif_render = self.render_notification_bar(
            notification_bar,
            last_snapshot=last_snapshot,
            max_pinned=max_pinned_notifications,
            elements_out=notification_elements,
        )
        actions_render = self.render_in_flight_actions(
            in_flight_actions,
            max_history=max_action_history_events,
            elements_out=action_elements,
        )
        completed_render = self.render_completed_actions(
            completed_actions,
            max_completed=max_completed_actions,
            max_history=max_completed_action_history_events,
        )
        convs_render = self.render_active_conversations(
            contact_index,
            last_snapshot=last_snapshot,
            max_contact_medium_messages=max_contact_medium_messages,
            elements_out=message_elements,
        )

        sections = [
            s
            for s in [
                infra_render,
                meet_render,
                web_sessions_render,
                coordinator_render,
                notif_render,
                actions_render,
                completed_render,
                convs_render,
            ]
            if s
        ]
        full_render = "\n\n".join(sections)

        return SnapshotState(
            full_render=full_render,
            messages=message_elements,
            notifications=notification_elements,
            actions=action_elements,
            snapshot_time=prompt_now(as_string=False),
        )

    @staticmethod
    def render_coordinator_goal_state(
        *,
        coordinator_state: dict[str, Any] | None = None,
        coordinator_checklist: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render the Coordinator's onboarding goal state for the slow brain."""

        if not coordinator_state:
            return ""

        lines = ["<coordinator_goal>"]
        mode = coordinator_state.get("mode")
        if mode:
            lines.append(f"mode: {mode}")

        started_at = _format_coordinator_datetime(coordinator_state.get("started_at"))
        if started_at:
            lines.append(f"started_at: {started_at}")

        ready_at = _format_coordinator_datetime(coordinator_state.get("ready_at"))
        if ready_at:
            lines.append(f"ready_at: {ready_at}")

        if coordinator_checklist:
            lines.append("checklist:")
            for item in coordinator_checklist:
                item_id = item.get("item_id", "?")
                title = item.get("title", "")
                status = item.get("status", "pending")
                kind = item.get("kind")
                description = item.get("description")
                line = f"- [{status}] #{item_id}: {title}"
                if kind:
                    line += f" ({kind})"
                lines.append(line)
                if description:
                    lines.append(f"  {description}")

        lines.append("</coordinator_goal>")
        return "\n".join(lines)

    @staticmethod
    def render_infrastructure_state(
        *,
        managers_initialized: bool = True,
        vm_ready: bool,
        file_sync_complete: bool,
        has_desktop: bool,
    ) -> str:
        """Render pending infrastructure state (init, VM boot, filesystem sync).

        Returns XML sections only while the relevant subsystem is still
        initialising.  Once everything is ready, returns an empty string
        so the snapshot is not cluttered.
        """
        parts: list[str] = []

        if not managers_initialized:
            parts.append(
                "<infrastructure status='initializing'>\n"
                "Your system is still initializing. Complex actions (act, "
                "contact queries, transcript queries) are not available yet — "
                "these tools will appear once initialization completes. "
                "Conversation history from previous sessions is also still "
                "loading; messages above may only reflect the current session. "
                "If the user sends a message, acknowledge it promptly and "
                "respond as best you can with the context available. "
                "You will receive a notification once initialization is "
                "complete, at which point you will have full conversation "
                "history and all tools. Use that turn to review your earlier "
                "responses and follow up — correct, elaborate, or confirm "
                "as needed.\n"
                "</infrastructure>",
            )

        if not has_desktop:
            return "\n".join(parts)

        if not vm_ready:
            parts.append(
                "<infrastructure status='vm_pending'>\n"
                "Your managed desktop VM is still booting. Computer actions "
                "(desktop_act, web_act, screenshot, etc.) are not available "
                "yet. Do not attempt computer actions until you receive a "
                "notification that the VM is ready. If a user asks you to do "
                "something on the computer, let them know you will action it "
                "in just a moment.\n"
                "Note: join_google_meet and join_teams_meet do NOT depend on "
                "the desktop VM — they use a local browser and are available "
                "immediately.\n"
                "</infrastructure>",
            )

        if not file_sync_complete:
            parts.append(
                "<infrastructure status='sync_pending'>\n"
                "Your local filesystem is being synced from persistent "
                "storage. Files from previous sessions (Attachments/, "
                "functions/, downloaded files, etc.) DO NOT EXIST on disk "
                "yet — any attempt to open, read, or reference them will "
                "fail. You MUST wait for the sync-complete notification "
                "before performing any file operations. If the user asks "
                "about a file, tell them your files are still loading and "
                "you will get to it once they've finished syncing, and then "
                "wait for the explicit notification that the files have "
                "synced before proceeding with any action.\n"
                "</infrastructure>",
            )

        return "\n".join(parts)

    @staticmethod
    def render_meet_interaction_state(
        *,
        assistant_screen_share_active: bool = False,
        user_screen_share_active: bool = False,
        user_webcam_active: bool = False,
        user_remote_control_active: bool = False,
        google_meet_active: bool = False,
        teams_meet_active: bool = False,
    ) -> str:
        """Render active meet interaction states as top-level sections.

        Each active state gets its own prominent, self-contained XML section
        with clear context explaining what is happening and what it means.
        Inactive states produce no output — no clutter for text-only sessions.
        """
        parts: list[str] = []

        if assistant_screen_share_active:
            parts.append(
                "<assistant_screen_share status='active'>\n"
                "Your desktop is currently being shared with the user — they "
                "can see everything on your screen in real time. Any actions "
                "you take (navigation, typing, file operations) are visible "
                "to the user as you perform them.\n"
                "</assistant_screen_share>",
            )

        if user_screen_share_active:
            parts.append(
                "<user_screen_share status='active'>\n"
                "The user is currently sharing their screen with you. You can "
                "see what they are looking at. If they reference something on "
                "their screen or ask for help with what they see, you have "
                "visual context available.\n"
                "</user_screen_share>",
            )

        if user_webcam_active:
            parts.append(
                "<user_webcam status='active'>\n"
                "The user's webcam is currently on. You can see them via "
                "captured frames paired with their speech. If they reference "
                "their appearance or something visible on camera, you have "
                "visual context available.\n"
                "</user_webcam>",
            )

        if user_remote_control_active:
            parts.append(
                "<user_remote_control status='active'>\n"
                "The user currently has remote control of your desktop — they "
                "are operating your mouse and keyboard directly. Do not "
                "perform any computer actions that would conflict with or "
                "interrupt the user's input. Wait for them to release control "
                "before resuming desktop operations.\n"
                "</user_remote_control>",
            )

        if google_meet_active:
            parts.append(
                "<google_meet_visual status='active'>\n"
                "You are in a Google Meet call and receiving periodic "
                "screenshots of the meeting view. You can see participants, "
                "any content being presented, and the meeting UI. When users "
                "ask if you can see the meeting, confirm that you can.\n"
                "</google_meet_visual>",
            )

        if teams_meet_active:
            parts.append(
                "<teams_meet_visual status='active'>\n"
                "You are in a Microsoft Teams meeting and receiving periodic "
                "screenshots of the meeting view. You can see participants, "
                "any content being presented, and the meeting UI. When users "
                "ask if you can see the meeting, confirm that you can.\n"
                "</teams_meet_visual>",
            )

        return "\n\n".join(parts)

    @staticmethod
    def render_active_web_sessions(web_sessions: list) -> str:
        """Render active visible web sessions for the slow brain snapshot.

        Accepts either a list of ``WebSessionHandle`` objects (backward compat)
        or a list of metadata dicts from ``list_sessions_with_metadata``.
        Only produces output when there are active sessions to show.
        """
        if not web_sessions:
            return ""
        lines = ["<active_web_sessions>"]
        for entry in web_sessions:
            if isinstance(entry, dict):
                sid = entry.get("session_id", "?")
                label = entry.get("label", "")
                url = entry.get("url", "")
                lines.append(
                    f'  <session id="{sid}" label="{label}" url="{url}" />',
                )
            else:
                sid = getattr(entry, "session_id", "?")
                label = getattr(entry, "label", "")
                lines.append(f'  <session id="{sid}" label="{label}" />')
        lines.append("</active_web_sessions>")
        return "\n".join(lines)

    def render_notification_bar(
        self,
        notification_bar: NotificationBar,
        last_snapshot: datetime = None,
        max_pinned: int = 50,
        elements_out: list[NotificationElement] | None = None,
    ) -> str:
        """Render the notification bar."""
        if notification_bar is None:
            return "<notifications>\n</notifications>"

        pinned_notifs = sorted(
            (n for n in notification_bar.notifications if n.pinned),
            key=lambda n: n.timestamp,
        )[-max_pinned:]
        new_notifs = [
            n
            for n in notification_bar.notifications
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs

        rendered_lines = []
        for n in all_notifs:
            prefix = "[PINNED]" if n.pinned else ""
            line = f'{prefix}[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}'
            rendered_lines.append(line)

            if elements_out is not None:
                elements_out.append(
                    NotificationElement(
                        timestamp=n.timestamp,
                        content_hash=hash(n.content),
                        pinned=n.pinned,
                        rendered=line,
                    ),
                )

        return f"<notifications>\n" + "\n".join(rendered_lines) + "\n</notifications>"

    @staticmethod
    def _render_action_history(
        handle_actions: list[dict],
        short_name: str,
        handle_id: int,
        max_history: int,
    ) -> str:
        """Render the event history for an action, capped to the most recent events."""
        displayed = handle_actions[-max_history:]
        if not displayed:
            return ""
        out = "<history>\n"
        for a in displayed:
            action_type = a.get("action_name", "")
            action_query = a.get("query", "")
            action_status = a.get("status", "")
            action_ts = a.get("timestamp", "")

            attrs = f"type='{action_type}'"
            if action_ts:
                attrs += f" timestamp='{action_ts}'"
            if action_status:
                attrs += f" status='{action_status}'"
            out += f"<event {attrs}>\n"

            if action_query:
                out += f"  <content>{action_query}</content>\n"
            if a_res := a.get("response"):
                out += f"  <response>{a_res}</response>\n"

            if action_status == "pending" and action_type.startswith("ask_"):
                out += (
                    "  <note>Result pending - you will receive another "
                    "turn when the answer is ready.</note>\n"
                )

            if action_type == "clarification_request" and not a.get("response"):
                call_id = a.get("call_id", "")
                suffix = safe_call_id_suffix(call_id)
                action = build_action_name(
                    "answer_clarification",
                    short_name,
                    handle_id,
                    suffix,
                )
                out += f"  <pending>Use {action} to respond</pending>\n"
            out += "</event>\n"
        out += "</history>\n"
        return out

    def render_in_flight_actions(
        self,
        in_flight_actions: dict,
        max_history: int = 20,
        elements_out: list[ActionElement] | None = None,
    ) -> str:
        """Render in-flight actions with their status and history."""
        out = "<in_flight_actions>\n"
        if not in_flight_actions:
            out += "No actions currently executing.\n"
        else:
            for handle_id, handle_data in in_flight_actions.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle = handle_data.get("handle")
                handle_actions = handle_data.get("handle_actions", [])

                is_paused = get_handle_paused_state(handle)
                status = "paused" if is_paused else "executing"

                pending_clarifications = [
                    a
                    for a in handle_actions
                    if a.get("action_name") == "clarification_request"
                    and not a.get("response")
                ]

                is_persistent = handle_data.get("persist", False)
                mode_attr = " mode='persistent'" if is_persistent else ""
                action_type = handle_data.get("action_type", "act")
                type_attr = f" type='{action_type}'"
                action_render = f"<action id='{handle_id}' short_name='{short_name}' status='{status}'{type_attr}{mode_attr}>\n"
                action_render += f"<original_request>{query}</original_request>\n"
                if is_persistent:
                    action_render += (
                        "<note>Persistent session — will NOT self-complete. "
                        "Use stop_* to end it. Responses marked 'awaiting_input' "
                        "mean the actor finished its turn and needs your next "
                        "interject_* to continue.</note>\n"
                    )

                action_render += "<steering_tools>\n"
                for action_name, description in iter_steering_tools_for_action(
                    handle_id,
                    query,
                    pending_clarifications,
                    is_paused=is_paused,
                ):
                    action_render += f"  - {action_name}: {description}\n"
                action_render += "</steering_tools>\n"

                action_render += self._render_action_history(
                    handle_actions,
                    short_name,
                    handle_id,
                    max_history,
                )

                action_render += "</action>\n"
                out += action_render

                if elements_out is not None:
                    elements_out.append(
                        ActionElement(
                            handle_id=handle_id,
                            query=query,
                            status=status,
                            history_count=len(handle_actions),
                            rendered=action_render,
                        ),
                    )

        out += "</in_flight_actions>"
        return out

    def render_active_conversations(
        self,
        contact_index: ContactIndex,
        last_snapshot: datetime = None,
        max_contact_medium_messages: int = 25,
        elements_out: list[MessageElement] | None = None,
    ) -> str:
        """Render active conversations derived from the shared global thread.

        Only contacts with messages in the global thread are rendered. Per-contact
        and per-medium views are derived from the shared deque at render time.
        """
        # Fetch assistant's timezone once for all contacts
        assistant_timezone = _get_assistant_timezone()

        # Group global thread entries by contact_id
        grouped = contact_index.get_messages_grouped_by_contact()

        contacts = []
        for contact_id, entries in grouped.items():
            contact_info = contact_index.get_contact(contact_id) or {}
            conv_state = contact_index.get_or_create_conversation(contact_id)
            rendered = self.render_contact(
                contact_info=contact_info,
                conv_state=conv_state,
                entries=entries,
                max_contact_medium_messages=max_contact_medium_messages,
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                assistant_timezone=assistant_timezone,
            )
            contacts.append(rendered)

        contacts_str = "\n\n".join(contacts)
        return f"<active_conversations>\n{contacts_str}\n</active_conversations>"

    def render_contact(
        self,
        contact_info: dict,
        conv_state: ConversationState,
        entries: list[GlobalThreadEntry] | None = None,
        max_contact_medium_messages: int = 25,
        last_snapshot: datetime = None,
        elements_out: list[MessageElement] | None = None,
        contact_index: ContactIndex | None = None,
        assistant_timezone: str | None = None,
    ) -> str:
        """Render a single contact's conversation.

        The global thread view is the full list of entries. Per-medium views
        are derived by grouping entries by medium and capping each.
        """
        contact_id = conv_state.contact_id
        first_name = contact_info.get("first_name") or ""
        surname = contact_info.get("surname") or ""
        phone_number = contact_info.get("phone_number") or ""
        email_address = contact_info.get("email_address") or ""
        discord_id = contact_info.get("discord_id") or ""
        timezone = contact_info.get("timezone") or ""
        bio = contact_info.get("bio") or ""
        rolling_summary = contact_info.get("rolling_summary") or ""
        response_policy = contact_info.get("response_policy") or ""
        should_respond = contact_info.get("should_respond", True)
        is_boss = contact_id == 1

        # Compute contact name for timezone display
        contact_name = f"{first_name} {surname}".strip() or f"Contact #{contact_id}"
        contact_timezone = contact_info.get("timezone")

        if entries is None:
            entries = []

        # Global thread: all messages for this contact (already capped by deque size)
        all_messages = [e.message for e in entries]
        global_thread = ""
        if all_messages:
            global_thread = self.render_thread(
                "global",
                all_messages,
                contact_id=contact_id,
                max_messages=len(all_messages),
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )

        # Per-medium threads: group entries by medium, cap each
        medium_messages: dict[str, list] = {}
        for entry in entries:
            medium_key = str(entry.medium)
            if medium_key not in medium_messages:
                medium_messages[medium_key] = []
            medium_messages[medium_key].append(entry.message)

        per_medium_threads = "\n\n".join(
            self.render_thread(
                medium_name,
                msgs,
                contact_id=contact_id,
                max_messages=max_contact_medium_messages,
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            for medium_name, msgs in medium_messages.items()
            if msgs
        )
        threads_content = (
            f"{global_thread}\n\n{per_medium_threads}"
            if global_thread
            else per_medium_threads
        )

        return (
            f'<contact contact_id="{contact_id}" first_name="{first_name}" surname="{surname}" '
            f'is_boss="{is_boss}" phone_number="{phone_number}" email_address="{email_address}" '
            f'discord_id="{discord_id}" '
            f'timezone="{timezone}" on_call="{conv_state.on_call}" should_respond="{should_respond}">\n'
            f"<bio>{bio}</bio>\n"
            f"<rolling_summary>{rolling_summary}</rolling_summary>\n"
            f"<response_policy>{response_policy}</response_policy>\n"
            f"<threads>\n{threads_content}\n</threads>\n"
            f"</contact>"
        )

    def render_thread(
        self,
        thread_name: str,
        thread,
        contact_id: int = None,
        max_messages: int = 25,
        last_snapshot: datetime = None,
        elements_out: list[MessageElement] | None = None,
        contact_index: ContactIndex | None = None,
        contact_name: str | None = None,
        contact_timezone: str | None = None,
        assistant_timezone: str | None = None,
    ) -> str:
        """Render a thread."""
        thread_list = list(thread)
        displayed_messages = thread_list[-max_messages:]
        start_index = len(thread_list) - len(displayed_messages)

        rendered_messages = []
        for i, m in enumerate(displayed_messages):
            rendered = self.render_message(
                m,
                last_snapshot,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            rendered_messages.append(rendered)

            if elements_out is not None:
                elements_out.append(
                    MessageElement(
                        contact_id=contact_id,
                        thread_name=thread_name,
                        index_in_thread=start_index + i,
                        timestamp=m.timestamp,
                        rendered=rendered,
                    ),
                )

        return (
            f"<{thread_name}>\n" + "\n".join(rendered_messages) + f"\n</{thread_name}>"
        )

    def render_message(
        self,
        message: (
            Message
            | EmailMessage
            | UnifyMessage
            | WhatsAppMessage
            | TeamsMessage
            | TeamsChannelMessage
            | ApiMessage
            | GuidanceMessage
        ),
        last_snapshot: datetime = None,
        contact_index: ContactIndex | None = None,
        contact_name: str | None = None,
        contact_timezone: str | None = None,
        assistant_timezone: str | None = None,
    ):
        # Mark all recent messages as NEW (both incoming and outbound)
        is_new = last_snapshot < message.timestamp
        new_marker = "**NEW** " if is_new else ""
        timestamp_str = message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")

        if isinstance(message, EmailMessage):
            attachments_line = ""
            if message.attachments:

                def _email_att_detail(att, is_self: bool) -> str:
                    if isinstance(att, dict):
                        fname = att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                        att_id = att.get("id")
                        id_part = f"id: {att_id}, " if att_id else ""
                        if is_self:
                            fpath = att.get("filepath")
                            if fpath:
                                return f"{fname} ({id_part}attached from {fpath})"
                            return (
                                f"{fname} ({id_part}attached)"
                                if id_part
                                else f"{fname} (attached)"
                            )
                        return f"{fname} ({id_part}auto-downloaded to Attachments/{att_id}_{fname})"
                    return (
                        f"{att} (attached)" if is_self else f"{att} (auto-downloaded)"
                    )

                attachment_details = [
                    _email_att_detail(att, message.name == "You")
                    for att in message.attachments
                ]
                attachments_line = f"Attachments: {', '.join(attachment_details)}\n"
            # Render recipients (for reply-all context)
            recipients_lines = ""
            if message.to:
                recipients_lines += f"To: {', '.join(message.to)}\n"
            if message.cc:
                recipients_lines += f"Cc: {', '.join(message.cc)}\n"
            if message.bcc:
                recipients_lines += f"Bcc: {', '.join(message.bcc)}\n"
            # Show contact's role in this email for clarity in contact-specific threads
            # This helps the LLM understand the contact's relationship to the email
            contact_role_line = ""
            if message.contact_role:
                role_descriptions = {
                    "sender": "This contact SENT this email",
                    "to": "This contact was a DIRECT RECIPIENT (To)",
                    "cc": "This contact was CC'd on this email",
                    "bcc": "This contact was BCC'd on this email",
                }
                contact_role_line = f"[Context: {role_descriptions.get(message.contact_role, message.contact_role)}]\n"

            # Show assistant's role in this email (To, Cc, Bcc, or sender)
            # This helps the LLM understand its own relationship to the email
            assistant_role_line = ""
            assistant_role = _get_assistant_email_role(message)
            if assistant_role:
                assistant_role_line = f"[Your role: {assistant_role}]\n"

            # Show participant timezone info with current local times
            # This helps the assistant be aware of recipients' local times
            tz_block_line = ""
            tz_block = _get_email_timezone_block(
                message,
                contact_index,
                assistant_timezone,
            )
            if tz_block:
                tz_block_line = f"{tz_block}\n"

            return (
                f"{new_marker}[{message.name} @ {timestamp_str}]:\n"
                f"{contact_role_line}"
                f"{assistant_role_line}"
                f"Subject: {message.subject}\n"
                f"Email ID: {message.email_id}\n"
                f"{recipients_lines}"
                f"{tz_block_line}"
                f"{attachments_line}"
                f"Body:\n"
                f"{message.body}"
            )

        if isinstance(message, (UnifyMessage, WhatsAppMessage)):
            attachments_line = ""
            if message.attachments:

                def _unify_att_detail(att, is_self: bool) -> str:
                    if isinstance(att, dict):
                        fname = att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                        att_id = att.get("id")
                        id_part = f"id: {att_id}, " if att_id else ""
                        if is_self:
                            fpath = att.get("filepath")
                            if fpath:
                                return f"{fname} ({id_part}attached from {fpath})"
                            return (
                                f"{fname} ({id_part}attached)"
                                if id_part
                                else f"{fname} (attached)"
                            )
                        return f"{fname} ({id_part}auto-downloaded to Attachments/{att_id}_{fname})"
                    return (
                        f"{att} (attached)" if is_self else f"{att} (auto-downloaded)"
                    )

                attachment_details = [
                    _unify_att_detail(att, message.name == "You")
                    for att in message.attachments
                ]
                attachments_line = f" [Attachments: {', '.join(attachment_details)}]"

            # Show timezone info for the contact
            tz_block_line = ""
            if contact_name:
                tz_block = _get_message_timezone_block(
                    contact_name,
                    contact_timezone,
                    assistant_timezone,
                )
                if tz_block:
                    tz_block_line = f"\n{tz_block}"

            return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}{attachments_line}{tz_block_line}"

        if isinstance(message, TeamsMessage):
            attachments_line = ""
            if message.attachments:

                def _teams_att_detail(att, is_self: bool) -> str:
                    if isinstance(att, dict):
                        fname = att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                        att_id = att.get("id")
                        id_part = f"id: {att_id}, " if att_id else ""
                        if is_self:
                            fpath = att.get("filepath")
                            if fpath:
                                return f"{fname} ({id_part}attached from {fpath})"
                            return (
                                f"{fname} ({id_part}attached)"
                                if id_part
                                else f"{fname} (attached)"
                            )
                        return f"{fname} ({id_part}auto-downloaded to Attachments/{att_id}_{fname})"
                    return (
                        f"{att} (attached)" if is_self else f"{att} (auto-downloaded)"
                    )

                attachment_details = [
                    _teams_att_detail(att, message.name == "You")
                    for att in message.attachments
                ]
                attachments_line = f" [Attachments: {', '.join(attachment_details)}]"

            tz_block_line = ""
            if contact_name:
                tz_block = _get_message_timezone_block(
                    contact_name,
                    contact_timezone,
                    assistant_timezone,
                )
                if tz_block:
                    tz_block_line = f"\n{tz_block}"

            # Surface the Teams chat_id so the LLM can reply into the same
            # thread via send_teams_message(chat_id=...).
            ids_line = ""
            if message.chat_id:
                ids_line = f' [chat_id="{message.chat_id}"]'

            return (
                f"{new_marker}[{message.name} @ {timestamp_str}]: "
                f"{message.content}{ids_line}{attachments_line}{tz_block_line}"
            )

        if isinstance(message, TeamsChannelMessage):
            attachments_line = ""
            if message.attachments:

                def _teams_channel_att_detail(att, is_self: bool) -> str:
                    if isinstance(att, dict):
                        fname = att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                        att_id = att.get("id")
                        id_part = f"id: {att_id}, " if att_id else ""
                        if is_self:
                            fpath = att.get("filepath")
                            if fpath:
                                return f"{fname} ({id_part}attached from {fpath})"
                            return (
                                f"{fname} ({id_part}attached)"
                                if id_part
                                else f"{fname} (attached)"
                            )
                        return f"{fname} ({id_part}auto-downloaded to Attachments/{att_id}_{fname})"
                    return (
                        f"{att} (attached)" if is_self else f"{att} (auto-downloaded)"
                    )

                attachment_details = [
                    _teams_channel_att_detail(att, message.name == "You")
                    for att in message.attachments
                ]
                attachments_line = f" [Attachments: {', '.join(attachment_details)}]"

            tz_block_line = ""
            if contact_name:
                tz_block = _get_message_timezone_block(
                    contact_name,
                    contact_timezone,
                    assistant_timezone,
                )
                if tz_block:
                    tz_block_line = f"\n{tz_block}"

            # Surface team_id, channel_id, and thread_id so the LLM can reply
            # into the same channel thread via send_teams_message(...).
            id_bits: list[str] = []
            if message.team_id:
                id_bits.append(f'team_id="{message.team_id}"')
            if message.channel_id:
                id_bits.append(f'channel_id="{message.channel_id}"')
            if message.thread_id:
                id_bits.append(f'thread_id="{message.thread_id}"')
            ids_line = f" [{' '.join(id_bits)}]" if id_bits else ""

            return (
                f"{new_marker}[{message.name} @ {timestamp_str}]: "
                f"{message.content}{ids_line}{attachments_line}{tz_block_line}"
            )

        if isinstance(message, ApiMessage):
            attachments_line = ""
            if message.attachments:

                def _api_att_detail(att, is_self: bool) -> str:
                    if isinstance(att, dict):
                        fname = att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                        att_id = att.get("id")
                        id_part = f"id: {att_id}, " if att_id else ""
                        if is_self:
                            fpath = att.get("filepath")
                            if fpath:
                                return f"{fname} ({id_part}attached from {fpath})"
                            return (
                                f"{fname} ({id_part}attached)"
                                if id_part
                                else f"{fname} (attached)"
                            )
                        return f"{fname} ({id_part}auto-downloaded to Attachments/{att_id}_{fname})"
                    return (
                        f"{att} (attached)" if is_self else f"{att} (auto-downloaded)"
                    )

                attachment_details = [
                    _api_att_detail(att, message.name == "You")
                    for att in message.attachments
                ]
                attachments_line = f" [Attachments: {', '.join(attachment_details)}]"

            tags_line = ""
            if message.tags:
                tags_line = f" [Tags: {', '.join(message.tags)}]"

            tz_block_line = ""
            if contact_name:
                tz_block = _get_message_timezone_block(
                    contact_name,
                    contact_timezone,
                    assistant_timezone,
                )
                if tz_block:
                    tz_block_line = f"\n{tz_block}"

            return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}{attachments_line}{tags_line}{tz_block_line}"

        if isinstance(message, GuidanceMessage):
            return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}"

        # Simple Message (SMS, phone call utterances)
        # Show timezone info for the contact
        tz_block_line = ""
        if contact_name:
            tz_block = _get_message_timezone_block(
                contact_name,
                contact_timezone,
                assistant_timezone,
            )
            if tz_block:
                tz_block_line = f"\n{tz_block}"

        screenshots_line = ""
        if isinstance(message, Message) and message.screenshots:
            screenshots_line = f" [Screenshots: {', '.join(message.screenshots)}]"

        return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}{screenshots_line}{tz_block_line}"

    def render_completed_actions(
        self,
        completed_actions: dict,
        max_completed: int = 20,
        max_history: int = 5,
    ):
        """Render completed actions with their result and a brief history.

        Each entry is self-contained: original query, result, capped history,
        and steering tools for post-completion queries.
        """
        out = "<completed_actions>\n"
        if not completed_actions:
            out += "No completed actions.\n"
        else:
            # Cap to the most recent completed actions by handle_id (monotonic)
            items = list(completed_actions.items())[-max_completed:]

            for handle_id, handle_data in items:
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle_actions = handle_data.get("handle_actions", [])

                # Extract result from the act_completed event
                result = None
                for a in reversed(handle_actions):
                    if a.get("action_name") == "act_completed":
                        result = a.get("query", "")
                        break

                action_type = handle_data.get("action_type", "act")
                out += f"<action id='{handle_id}' short_name='{short_name}' status='completed' type='{action_type}'>\n"
                out += f"<original_request>{query}</original_request>\n"

                if result is not None:
                    out += f"<result>{result}</result>\n"

                out += self._render_action_history(
                    handle_actions,
                    short_name,
                    handle_id,
                    max_history,
                )

                out += "<steering_tools>\n"
                for (
                    action_name,
                    description,
                ) in iter_steering_tools_for_completed_action(
                    handle_id,
                    query,
                ):
                    out += f"  - {action_name}: {description}\n"
                out += "</steering_tools>\n"

                out += "</action>\n"
        out += "</completed_actions>"
        return out
