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
from typing import TYPE_CHECKING

from unity.common._async_tool.utils import get_handle_paused_state
from unity.conversation_manager.domains.contact_index import (
    Message,
    EmailMessage,
    UnifyMessage,
    GuidanceMessage,
    ConversationState,
    ContactIndex,
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


def _get_assistant_timezone() -> str | None:
    """Get the assistant's timezone from contact_id=0.

    Returns:
        IANA timezone identifier or None if not available.
    """
    import unify as _unify

    try:
        _ctxs = _unify.get_active_context()
        _read_ctx = _ctxs.get("read")
    except Exception:
        _read_ctx = None
    _contacts_ctx = f"{_read_ctx}/Contacts" if _read_ctx else "Contacts"

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
                return val.strip()
    except Exception:
        pass
    return None


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


class Renderer:

    def render_state(
        self,
        contact_index: ContactIndex,
        notification_bar: NotificationBar = None,
        in_flight_actions: dict = None,
        completed_actions: dict = None,
        last_snapshot: datetime = None,
    ) -> str:
        """Render the full state as a string (backward compatible)."""
        return (
            f"{self.render_notification_bar(notification_bar, last_snapshot=last_snapshot)}\n\n"
            f"{self.render_in_flight_actions(in_flight_actions)}\n\n"
            f"{self.render_completed_actions(completed_actions)}\n\n"
            f"{self.render_active_conversations(contact_index, last_snapshot=last_snapshot)}"
        )

    def render_state_with_tracking(
        self,
        contact_index: ContactIndex,
        notification_bar: NotificationBar = None,
        in_flight_actions: dict = None,
        completed_actions: dict = None,
        last_snapshot: datetime = None,
    ) -> SnapshotState:
        """Render state and return SnapshotState with constituent element tracking.

        This enables computing incremental diffs for context propagation.
        """
        from unity.common.prompt_helpers import now as prompt_now

        # Collect constituent elements while rendering
        message_elements: list[MessageElement] = []
        notification_elements: list[NotificationElement] = []
        action_elements: list[ActionElement] = []

        # Render notifications with tracking
        notif_render = self._render_notification_bar_with_tracking(
            notification_bar,
            last_snapshot=last_snapshot,
            elements_out=notification_elements,
        )

        # Render in-flight actions with tracking
        actions_render = self._render_in_flight_actions_with_tracking(
            in_flight_actions,
            elements_out=action_elements,
        )

        # Render completed actions (no element tracking needed)
        completed_render = self.render_completed_actions(completed_actions)

        # Render active conversations with tracking
        convs_render = self._render_active_conversations_with_tracking(
            contact_index,
            last_snapshot=last_snapshot,
            elements_out=message_elements,
        )

        full_render = f"{notif_render}\n\n{actions_render}\n\n{completed_render}\n\n{convs_render}"

        return SnapshotState(
            full_render=full_render,
            messages=message_elements,
            notifications=notification_elements,
            actions=action_elements,
            snapshot_time=prompt_now(as_string=False),
        )

    def _render_notification_bar_with_tracking(
        self,
        notification_bar: NotificationBar,
        last_snapshot: datetime = None,
        elements_out: list[NotificationElement] | None = None,
    ) -> str:
        """Render notification bar and track elements."""
        if notification_bar is None:
            return "<notifications>\n</notifications>"

        pinned_notifs = [n for n in notification_bar.notifications if n.pinned]
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

    def _render_in_flight_actions_with_tracking(
        self,
        in_flight_actions: dict,
        elements_out: list[ActionElement] | None = None,
    ) -> str:
        """Render in-flight actions and track elements."""
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

                action_render = f"<action id='{handle_id}' short_name='{short_name}' status='{status}'>\n"
                action_render += f"<original_request>{query}</original_request>\n"

                action_render += "<steering_tools>\n"
                for action_name, description in iter_steering_tools_for_action(
                    handle_id,
                    query,
                    pending_clarifications,
                    is_paused=is_paused,
                ):
                    action_render += f"  - {action_name}: {description}\n"
                action_render += "</steering_tools>\n"

                if handle_actions:
                    action_render += "<history>\n"
                    for a in handle_actions:
                        action_type = a.get("action_name", "")
                        action_query = a.get("query", "")
                        action_status = a.get("status", "")

                        if action_status:
                            action_render += f"<event type='{action_type}' status='{action_status}'>\n"
                        else:
                            action_render += f"<event type='{action_type}'>\n"

                        if action_query:
                            action_render += f"  <content>{action_query}</content>\n"
                        if a_res := a.get("response"):
                            action_render += f"  <response>{a_res}</response>\n"

                        if action_status == "pending" and action_type.startswith(
                            "ask_",
                        ):
                            action_render += (
                                "  <note>Result pending - you will receive another "
                                "turn when the answer is ready.</note>\n"
                            )

                        if action_type == "clarification_request" and not a.get(
                            "response",
                        ):
                            call_id = a.get("call_id", "")
                            suffix = safe_call_id_suffix(call_id)
                            action = build_action_name(
                                "answer_clarification",
                                short_name,
                                handle_id,
                                suffix,
                            )
                            action_render += (
                                f"  <pending>Use {action} to respond</pending>\n"
                            )
                        action_render += "</event>\n"
                    action_render += "</history>\n"

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

    def _render_active_conversations_with_tracking(
        self,
        contact_index: ContactIndex,
        last_snapshot: datetime = None,
        max_messages: int = 5,
        max_global_messages: int = 50,
        elements_out: list[MessageElement] | None = None,
    ) -> str:
        """Render active conversations and track message elements."""
        # Fetch assistant's timezone once for all contacts
        assistant_timezone = _get_assistant_timezone()

        contacts = []
        for contact_id, conv_state in contact_index.active_conversations.items():
            contact_info = contact_index.get_contact(contact_id) or {}
            rendered = self._render_contact_with_tracking(
                contact_info=contact_info,
                conv_state=conv_state,
                max_messages=max_messages,
                max_global_messages=max_global_messages,
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                assistant_timezone=assistant_timezone,
            )
            contacts.append(rendered)

        contacts_str = "\n\n".join(contacts)
        return f"<active_conversations>\n{contacts_str}\n</active_conversations>"

    def _render_contact_with_tracking(
        self,
        contact_info: dict,
        conv_state: ConversationState,
        max_messages: int = 5,
        max_global_messages: int = 50,
        last_snapshot: datetime = None,
        elements_out: list[MessageElement] | None = None,
        contact_index: ContactIndex | None = None,
        assistant_timezone: str | None = None,
    ) -> str:
        """Render a single contact's conversation and track message elements."""
        contact_id = conv_state.contact_id
        first_name = contact_info.get("first_name") or ""
        surname = contact_info.get("surname") or ""
        phone_number = contact_info.get("phone_number") or ""
        email_address = contact_info.get("email_address") or ""
        timezone = contact_info.get("timezone") or ""
        bio = contact_info.get("bio") or ""
        rolling_summary = contact_info.get("rolling_summary") or ""
        response_policy = contact_info.get("response_policy") or ""
        should_respond = contact_info.get("should_respond", True)
        is_boss = contact_id == 1

        # Compute contact name for timezone display
        contact_name = f"{first_name} {surname}".strip() or f"Contact #{contact_id}"
        contact_timezone = contact_info.get("timezone")

        # Render threads with tracking
        global_thread = ""
        if conv_state.global_thread:
            global_thread = self._render_thread_with_tracking(
                "global",
                conv_state.global_thread,
                contact_id=contact_id,
                max_messages=max_global_messages,
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )

        per_medium_threads = "\n\n".join(
            self._render_thread_with_tracking(
                str(t_name),
                t,
                contact_id=contact_id,
                max_messages=max_messages,
                last_snapshot=last_snapshot,
                elements_out=elements_out,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            for t_name, t in conv_state.threads.items()
            if t
        )
        threads_content = (
            f"{global_thread}\n\n{per_medium_threads}"
            if global_thread
            else per_medium_threads
        )

        return (
            f'<contact contact_id="{contact_id}" first_name="{first_name}" surname="{surname}" '
            f'is_boss="{is_boss}" phone_number="{phone_number}" email_address="{email_address}" '
            f'timezone="{timezone}" on_call="{conv_state.on_call}" should_respond="{should_respond}">\n'
            f"<bio>{bio}</bio>\n"
            f"<rolling_summary>{rolling_summary}</rolling_summary>\n"
            f"<response_policy>{response_policy}</response_policy>\n"
            f"<threads>\n{threads_content}\n</threads>\n"
            f"</contact>"
        )

    def _render_thread_with_tracking(
        self,
        thread_name: str,
        thread,
        contact_id: int,
        max_messages: int = 5,
        last_snapshot: datetime = None,
        elements_out: list[MessageElement] | None = None,
        contact_index: ContactIndex | None = None,
        contact_name: str | None = None,
        contact_timezone: str | None = None,
        assistant_timezone: str | None = None,
    ) -> str:
        """Render a thread and track message elements."""
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

    def render_active_conversations(
        self,
        contact_index: ContactIndex,
        max_messages: int = 5,
        max_global_messages: int = 50,
        last_snapshot: datetime = None,
    ):
        """Render all active conversations, fetching contact info from ContactManager."""
        # Fetch assistant's timezone once for all contacts
        assistant_timezone = _get_assistant_timezone()

        contacts = []
        for contact_id, conv_state in contact_index.active_conversations.items():
            # Fetch contact info from ContactManager (source of truth)
            contact_info = contact_index.get_contact(contact_id) or {}
            rendered = self.render_contact(
                contact_info=contact_info,
                conv_state=conv_state,
                max_messages=max_messages,
                max_global_messages=max_global_messages,
                last_snapshot=last_snapshot,
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
        max_messages: int = 5,
        max_global_messages: int = 50,
        last_snapshot: datetime = None,
        contact_index: ContactIndex | None = None,
        assistant_timezone: str | None = None,
    ):
        """
        Render a single contact's conversation.

        Args:
            contact_info: Contact data from ContactManager (name, email, response_policy, etc.)
            conv_state: Conversation state from ContactIndex (threads, on_call)
            contact_index: ContactIndex for looking up participant timezones in emails
            assistant_timezone: Assistant's timezone for timezone comparison display
        """
        contact_id = conv_state.contact_id
        first_name = contact_info.get("first_name") or ""
        surname = contact_info.get("surname") or ""
        phone_number = contact_info.get("phone_number") or ""
        email_address = contact_info.get("email_address") or ""
        timezone = contact_info.get("timezone") or ""
        bio = contact_info.get("bio") or ""
        rolling_summary = contact_info.get("rolling_summary") or ""
        response_policy = contact_info.get("response_policy") or ""
        should_respond = contact_info.get("should_respond", True)
        is_boss = contact_id == 1

        # Compute contact name for timezone display
        contact_name = f"{first_name} {surname}".strip() or f"Contact #{contact_id}"
        contact_timezone = contact_info.get("timezone")

        # Render threads
        global_thread = (
            self.render_thread(
                "global",
                conv_state.global_thread,
                max_messages=max_global_messages,
                last_snapshot=last_snapshot,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            if conv_state.global_thread
            else ""
        )
        per_medium_threads = "\n\n".join(
            self.render_thread(
                t_name,
                t,
                max_messages=max_messages,
                last_snapshot=last_snapshot,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            for t_name, t in conv_state.threads.items()
            if t
        )
        threads_content = (
            f"{global_thread}\n\n{per_medium_threads}"
            if global_thread
            else per_medium_threads
        )

        return (
            f'<contact contact_id="{contact_id}" first_name="{first_name}" surname="{surname}" '
            f'is_boss="{is_boss}" phone_number="{phone_number}" email_address="{email_address}" '
            f'timezone="{timezone}" on_call="{conv_state.on_call}" should_respond="{should_respond}">\n'
            f"<bio>{bio}</bio>\n"
            f"<rolling_summary>{rolling_summary}</rolling_summary>\n"
            f"<response_policy>{response_policy}</response_policy>\n"
            f"<threads>\n{threads_content}\n</threads>\n"
            f"</contact>"
        )

    def render_thread(
        self,
        thread_name,
        thread,
        max_messages=5,
        last_snapshot=None,
        contact_index: ContactIndex | None = None,
        contact_name: str | None = None,
        contact_timezone: str | None = None,
        assistant_timezone: str | None = None,
    ):
        messages = "\n".join(
            self.render_message(
                m,
                last_snapshot,
                contact_index=contact_index,
                contact_name=contact_name,
                contact_timezone=contact_timezone,
                assistant_timezone=assistant_timezone,
            )
            for m in list(thread)[-max_messages:]
        )
        return f"<{thread_name}>\n{messages}\n</{thread_name}>"

    def render_message(
        self,
        message: Message | EmailMessage | UnifyMessage | GuidanceMessage,
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
                if message.name == "You":
                    attachment_details = [
                        f"{fname} (attached)" for fname in message.attachments
                    ]
                else:
                    attachment_details = [
                        f"{fname} (auto-downloaded to Downloads/{fname})"
                        for fname in message.attachments
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

        if isinstance(message, UnifyMessage):
            attachments_line = ""
            if message.attachments:
                # Extract filename from attachment (supports both dict and string format)
                def get_filename(att):
                    if isinstance(att, dict):
                        return att.get(
                            "filename",
                            f"attachment_{att.get('id', 'unknown')}",
                        )
                    return att  # Already a string

                if message.name == "You":
                    attachment_details = [
                        f"{get_filename(att)} (attached)" for att in message.attachments
                    ]
                else:
                    attachment_details = [
                        f"{get_filename(att)} (auto-downloaded to Downloads/{get_filename(att)})"
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

        return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}{tz_block_line}"

    def render_notification_bar(
        self,
        notification_bar: NotificationBar,
        last_snapshot=None,
    ):
        pinned_notifs = [n for n in notification_bar.notifications if n.pinned]
        new_notifs = [
            n
            for n in notification_bar.notifications
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs
        rendered_notifs = "\n".join(
            f'{"[PINNED] " if n.pinned else ""}[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}'
            for n in all_notifs
        )
        return f"<notifications>\n{rendered_notifs}\n</notifications>"

    def render_in_flight_actions(self, in_flight_actions: dict):
        """Render currently in-flight actions with their status and history.

        These are actions that are ALREADY EXECUTING - work is in progress.
        Use steering tools to interact with them, don't duplicate with `act`.
        """
        out = "<in_flight_actions>\n"
        if not in_flight_actions:
            out += "No actions currently executing.\n"
        else:
            for handle_id, handle_data in in_flight_actions.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle = handle_data.get("handle")
                handle_actions = handle_data.get("handle_actions", [])

                # Determine status based on pause state
                is_paused = get_handle_paused_state(handle)
                status = "paused" if is_paused else "executing"

                pending_clarifications = [
                    a
                    for a in handle_actions
                    if a.get("action_name") == "clarification_request"
                    and not a.get("response")
                ]

                out += f"<action id='{handle_id}' short_name='{short_name}' status='{status}'>\n"
                out += f"<original_request>{query}</original_request>\n"

                out += "<steering_tools>\n"
                for action_name, description in iter_steering_tools_for_action(
                    handle_id,
                    query,
                    pending_clarifications,
                    is_paused=is_paused,
                ):
                    out += f"  - {action_name}: {description}\n"
                out += "</steering_tools>\n"

                if handle_actions:
                    out += "<history>\n"
                    for a in handle_actions:
                        action_type = a.get("action_name", "")
                        action_query = a.get("query", "")
                        action_status = a.get("status", "")

                        # Include status attribute if present (for ask operations)
                        if action_status:
                            out += f"<event type='{action_type}' status='{action_status}'>\n"
                        else:
                            out += f"<event type='{action_type}'>\n"

                        if action_query:
                            out += f"  <content>{action_query}</content>\n"
                        if a_res := a.get("response"):
                            out += f"  <response>{a_res}</response>\n"

                        # Show pending note for in-flight ask operations
                        if action_status == "pending" and action_type.startswith(
                            "ask_",
                        ):
                            out += (
                                "  <note>Result pending - you will receive another "
                                "turn when the answer is ready.</note>\n"
                            )

                        if action_type == "clarification_request" and not a.get(
                            "response",
                        ):
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

                out += "</action>\n"
        out += "</in_flight_actions>"
        return out

    def render_completed_actions(self, completed_actions: dict):
        """Render completed actions that are available for querying.

        These actions have finished execution but remain available for
        `ask` queries about their trajectory/results.
        """
        out = "<completed_actions>\n"
        if not completed_actions:
            out += "No completed actions.\n"
        else:
            for handle_id, handle_data in completed_actions.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)

                out += f"<action id='{handle_id}' short_name='{short_name}' status='completed'>\n"
                out += f"<original_request>{query}</original_request>\n"

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
