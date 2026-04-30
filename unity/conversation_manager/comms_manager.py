"""
CommsManager: External communications handler for ConversationManager.

This module bridges external communication channels (GCP PubSub for SMS, email,
calls, etc.) to the internal event broker.

Threading Model:
----------------
GCP PubSub uses a thread pool for message callbacks. The `handle_message` method
is called from these background threads, NOT from the asyncio event loop. Therefore:

- `handle_message` uses `asyncio.run_coroutine_threadsafe()` to safely publish
  events to the async event broker from a sync callback context.
- `send_pings` and `start` are async methods that run on the event loop and can
  use direct `await` for async operations.

Testing:
--------
For testing, CommsManager is typically disabled (enable_comms_manager=False) since
there are no real external events to receive. Tests can publish events directly
to the event broker instead.
"""

from __future__ import annotations

import asyncio
from functools import partial
import html
import json
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

try:
    from google.cloud import pubsub_v1
except ImportError:  # pragma: no cover - exercised in local-only installs
    pubsub_v1 = None

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS
from unity.settings import SETTINGS
from unity.deploy_runtime import (
    mark_job_container_ready,
    read_assistant_session,
    read_job_assignment_record,
    read_session_bootstrap_secret_record,
    wait_for_assistant_session_name,
)
from unity.conversation_manager.domains.comms_utils import (
    add_email_attachments,
    add_unify_message_attachments,
    publish_system_error,
)
from unity.conversation_manager.domains.inactivity import (
    _inactivity_followup_event_from_payload,
)
from unity.conversation_manager.events import *
from unity.conversation_manager.metrics import pubsub_e2e_latency
from unity.session_details import SESSION_DETAILS
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.cm_types import Medium

load_dotenv()

# Lock for unknown contact creation to prevent duplicates
_unknown_contact_lock = threading.Lock()

# In-memory dedup for Discord message snowflake IDs.
# Keyed on message ID; values are insertion timestamps for TTL eviction.
_seen_discord_ids: dict[str, float] = {}
_DISCORD_DEDUP_TTL = 300.0


def _already_seen_discord(message_id: str) -> bool:
    """Return True if this Discord message_id was already processed recently."""
    now = time.time()
    cutoff = now - _DISCORD_DEDUP_TTL
    expired = [k for k, t in _seen_discord_ids.items() if t < cutoff]
    for k in expired:
        del _seen_discord_ids[k]
    if message_id in _seen_discord_ids:
        return True
    _seen_discord_ids[message_id] = now
    return False


# In-memory dedup for Microsoft Graph Teams message IDs.
# Graph change-notification subscriptions can redeliver when acks fall behind,
# so we guard against duplicate ingestion the same way Discord does.
_seen_teams_ids: dict[str, float] = {}
_TEAMS_DEDUP_TTL = 300.0


def _already_seen_teams(message_id: str) -> bool:
    """Return True if this Teams message_id was already processed recently."""
    now = time.time()
    cutoff = now - _TEAMS_DEDUP_TTL
    expired = [k for k, t in _seen_teams_ids.items() if t < cutoff]
    for k in expired:
        del _seen_teams_ids[k]
    if message_id in _seen_teams_ids:
        return True
    _seen_teams_ids[message_id] = now
    return False


if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker

    EventBroker = InMemoryEventBroker


def _get_subscription_id() -> str:
    """Build subscription ID from current assistant context."""
    agent_id = SESSION_DETAILS.assistant.agent_id
    env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
    return f"unity-{agent_id}{env_suffix}-sub"


def _get_local_contact() -> dict:
    """Build local contact dict from current assistant context."""
    return {
        "contact_id": -1,
        "first_name": SESSION_DETAILS.user.first_name,
        "surname": SESSION_DETAILS.user.surname,
        "phone_number": SESSION_DETAILS.user.number,
        "email_address": SESSION_DETAILS.user.email,
        "whatsapp_number": SESSION_DETAILS.user.whatsapp_number,
    }


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer coercion for webhook payloads."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_TEAMS_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TEAMS_BLOCK_END_RE = re.compile(r"</(p|div|li|h[1-6]|tr)>", re.IGNORECASE)
_TEAMS_TAG_RE = re.compile(r"<[^>]+>")
_TEAMS_INLINE_WS_RE = re.compile(r"[ \t]+")
_TEAMS_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _teams_html_to_text(raw: str) -> str:
    """Convert a Teams HTML message body into plain text.

    Microsoft Graph returns Teams chat and channel message bodies as HTML
    whenever the user authored them in the Teams UI (e.g. ``<p>hi</p>``).
    Storing the raw markup in transcripts is noisy and bloats downstream
    LLM prompts, so we normalize to plain text at ingress:

    - ``<br>`` and block-closing tags (``</p>``, ``</div>`` etc.) become newlines
    - all remaining tags are dropped
    - HTML entities (``&amp;``, ``&nbsp;`` etc.) are unescaped
    - inline whitespace is collapsed; leading/trailing empty lines stripped
    """
    if not raw:
        return ""
    s = _TEAMS_BR_RE.sub("\n", raw)
    s = _TEAMS_BLOCK_END_RE.sub("\n", s)
    s = _TEAMS_TAG_RE.sub("", s)
    s = html.unescape(s)
    lines = [_TEAMS_INLINE_WS_RE.sub(" ", line).strip() for line in s.splitlines()]
    joined = "\n".join(line for line in lines).strip()
    # An empty paragraph (e.g. ``<p><br></p>``) emits both a ``<br>`` newline and
    # a closing-tag newline on top of the surrounding paragraphs' own breaks,
    # which adds up to 3+ consecutive newlines. Collapse runs to a single blank
    # line so empty paragraphs render as one visual gap, matching how Teams
    # itself displays them.
    return _TEAMS_BLANK_RUN_RE.sub("\n\n", joined)


def _task_due_event_from_payload(
    payload: dict[str, Any],
    *,
    reason: str = "",
) -> TaskDue | None:
    """Build a `TaskDue` event from a comms payload when complete."""

    task_id = _coerce_int(payload.get("task_id"))
    source_task_log_id = _coerce_int(payload.get("source_task_log_id"))
    activation_revision = str(payload.get("activation_revision") or "")
    scheduled_for = str(payload.get("scheduled_for") or "")
    if task_id is None or source_task_log_id is None:
        return None
    if not activation_revision or not scheduled_for:
        return None
    task_label = str(payload.get("task_label") or "")
    return TaskDue(
        task_id=task_id,
        source_task_log_id=source_task_log_id,
        activation_revision=activation_revision,
        scheduled_for=scheduled_for,
        destination=payload.get("destination"),
        execution_mode=str(payload.get("execution_mode") or "live"),
        source_type=str(payload.get("source_type") or "scheduled"),
        task_label=task_label,
        task_summary=str(payload.get("task_summary") or ""),
        visibility_policy=str(payload.get("visibility_policy") or "silent_by_default"),
        recurrence_hint=str(payload.get("recurrence_hint") or "one_off"),
        reason=reason
        or (
            f"Scheduled task '{task_label}' became due."
            if task_label
            else f"Scheduled task {task_id} became due."
        ),
    )


# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    "msg": SMSReceived,
    "whatsapp": WhatsAppReceived,
    "email": EmailReceived,
    "unify_message": UnifyMessageReceived,
    "api_message": ApiMessageReceived,
    "discord": DiscordMessageReceived,
    "teams_chat": TeamsMessageReceived,
    "teams_channel": TeamsChannelMessageReceived,
}


def _is_blacklisted(medium: str, contact_detail: str | None) -> bool:
    """
    Check if a contact detail is blacklisted for a given medium.

    This is a fail-open check: returns False on any error to avoid
    blocking legitimate messages due to infrastructure issues.

    Gated by SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED (default False).
    When disabled, returns False immediately without any manager initialization.

    Args:
        medium: The communication medium (e.g., "sms_message", "email", "phone_call")
        contact_detail: The phone number or email address to check

    Returns:
        True if the contact detail is blacklisted, False otherwise
    """
    # Fast path: skip all manager initialization when blacklist checks disabled
    if not SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED:
        return False

    if not contact_detail:
        return False

    try:
        from unity.blacklist_manager import BlackListManager

        blm = BlackListManager()
        result = blm.filter_blacklist(
            filter=f"medium == '{medium}' and contact_detail == '{contact_detail}'",
            limit=1,
        )
        return len(result.get("entries", [])) > 0
    except Exception:
        # Fail-open: don't block messages if blacklist check fails
        return False


def _get_or_create_unknown_contact(
    medium: str,
    contact_detail: str,
) -> dict | None:
    """
    Get an existing contact or create a new unknown contact.

    When an inbound message arrives from an unknown sender (not in Contacts
    and not in BlackList), we create a minimal contact record with:
    - Only the medium field populated (phone_number or email_address)
    - should_respond=False to prevent automatic responses
    - A response_policy guiding the assistant to seek boss guidance

    Uses a lock to prevent duplicate contact creation when multiple
    messages arrive from the same unknown sender simultaneously.

    Gated by SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED (default False).
    When disabled, returns None immediately without any manager initialization.

    Args:
        medium: The communication medium (determines which contact field to set)
        contact_detail: The phone number or email address

    Returns:
        The contact dict (existing or newly created), or None on error
    """
    # Fast path: skip all manager initialization when blacklist checks disabled
    if not SETTINGS.conversation.BLACKLIST_CHECKS_ENABLED:
        return None

    from unity.manager_registry import ManagerRegistry
    from unity.contact_manager.contact_manager import ContactManager

    with _unknown_contact_lock:
        try:
            cm = ManagerRegistry.get_contact_manager()

            # Determine which field to search/set based on medium
            if medium == "whatsapp_message":
                field_name = "whatsapp_number"
            elif medium in ("sms_message", "phone_call"):
                field_name = "phone_number"
            elif medium == "email":
                field_name = "email_address"
            elif medium in ("discord_message", "discord_channel_message"):
                field_name = "discord_id"
            elif medium in ("teams_message", "teams_channel_message"):
                field_name = "email_address"
            else:
                # For unify_message, we don't have external contact details
                return None

            # Check if contact already exists
            result = cm.filter_contacts(
                filter=f"{field_name} == '{contact_detail}'",
                limit=1,
            )
            existing = result.get("contacts", [])
            if existing:
                contact = existing[0]
                return (
                    contact.model_dump() if hasattr(contact, "model_dump") else contact
                )

            # Create new unknown contact
            create_kwargs = {
                field_name: contact_detail,
                "should_respond": False,
                "response_policy": ContactManager.UNKNOWN_INBOUND_RESPONSE_POLICY,
            }
            outcome = cm._create_contact(**create_kwargs)
            new_contact_id = outcome["details"]["contact_id"]

            # Fetch the newly created contact
            contact_info = cm.get_contact_info(new_contact_id)
            new_contact = contact_info.get(new_contact_id)
            return new_contact

        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error in _get_or_create_unknown_contact: {e}")
            return None


def _resolve_teams_participants(
    *,
    raw_participants: list[dict],
    sender_email: str,
    sender_contact_id: int | None,
    medium: str,
    unknown_contact_resolver,
) -> list[int]:
    """Collapse the adapter-supplied participants payload into contact IDs.

    Each entry is shaped as ``{"contact_id": int | None, "email": str | None,
    "display_name": str, "aad_user_id": str | None}``. Pre-resolved entries
    (``contact_id`` set) are used directly. Unresolved entries with a real
    email are finished via ``unknown_contact_resolver`` — the single
    canonical path for minting external contacts (gated by
    ``BLACKLIST_CHECKS_ENABLED``). Entries without a usable email (or with
    a synthetic ``@teams`` placeholder) are dropped rather than pollute the
    contact store.

    The sender's contact_id is seeded into the set because the ingress
    caller has already resolved them via the sender-specific branch;
    entries matching ``sender_email`` are skipped to avoid a redundant
    resolver call.
    """
    participant_contact_ids: set[int] = set()
    if sender_contact_id is not None:
        participant_contact_ids.add(sender_contact_id)

    for entry in raw_participants:
        entry_email = (entry.get("email") or "").strip()
        if entry_email and sender_email and entry_email == sender_email:
            continue
        entry_cid = entry.get("contact_id")
        if entry_cid is not None:
            participant_contact_ids.add(entry_cid)
            continue
        if not entry_email or entry_email.endswith("@teams"):
            continue
        resolved = unknown_contact_resolver(medium, entry_email)
        resolved_cid = (resolved or {}).get("contact_id")
        if resolved_cid is not None:
            participant_contact_ids.add(resolved_cid)

    return sorted(participant_contact_ids)


class CommsManager:
    """
    Handles external communications via GCP PubSub.

    Receives events from external channels (SMS, email, calls) and publishes
    them to the internal event broker for ConversationManager to process.
    """

    def __init__(self, event_broker: "EventBroker"):
        self.subscribers: dict = {}
        self.call_proc = None
        self.credentials = None
        # Store reference to event loop for thread-safe publishing from callbacks
        self.loop = asyncio.get_event_loop()
        self.event_broker: "EventBroker" = event_broker

    def _publish_from_callback(self, channel: str, message: str) -> None:
        """
        Publish to event broker from a sync callback (thread-safe).

        This method is called from GCP PubSub callbacks which run in a thread pool,
        NOT from the asyncio event loop. We use run_coroutine_threadsafe to safely
        schedule the async publish on the main event loop.
        """
        asyncio.run_coroutine_threadsafe(
            self.event_broker.publish(channel, message),
            self.loop,
        )

    def _ack_with_latency(self, message, publish_timestamp, topic):
        """Ack the message and record end-to-end Pub/Sub latency if available."""
        if publish_timestamp is not None:
            latency = time.time() - publish_timestamp
            pubsub_e2e_latency.record(latency, {"topic": topic})
        message.ack()

    def _ack_callback(self, ack, publish_timestamp, topic):
        """Ack a message callback while preserving latency metrics."""
        if publish_timestamp is not None:
            latency = time.time() - publish_timestamp
            pubsub_e2e_latency.record(latency, {"topic": topic})
        ack()

    def _log_dispatch_future(self, future) -> None:
        """Log unexpected failures from background envelope dispatch tasks."""
        exc = future.exception()
        if exc is not None:
            LOGGER.error(f"{DEFAULT_ICON} Error processing message: {exc}")

    async def _handle_attachment_ingestion_complete(self, event: dict) -> None:
        """Apply a worker-reported attachment ingestion outcome to FileRecords.

        Invoked when a ``thread="attachment_ingestion_complete"`` envelope
        arrives on the per-assistant Pub/Sub topic.  The event payload must
        include ``display_name`` and ``status`` (``"success"`` or ``"error"``);
        optional ``error`` carries a failure message.
        """
        display_name = event.get("display_name")
        status = event.get("status", "success")
        error = event.get("error")
        if not display_name:
            LOGGER.warning(
                f"{DEFAULT_ICON} attachment_ingestion_complete missing display_name; "
                "ignoring event.",
            )
            return
        try:
            from unity.conversation_manager.domains.managers_utils import (
                ManagerRegistry,
            )

            file_manager = ManagerRegistry.get_file_manager()
        except Exception:
            file_manager = None
        if file_manager is None:
            LOGGER.warning(
                f"{DEFAULT_ICON} attachment_ingestion_complete received but "
                "FileManager is unavailable; cannot update FileRecords.",
            )
            return
        from unity.file_manager.managers.utils.attachment_ingestion import (
            apply_attachment_completion,
        )

        await asyncio.to_thread(
            apply_attachment_completion,
            file_manager,
            display_name=display_name,
            status=status,
            error=error,
        )

    async def dispatch_envelope_payload(
        self,
        payload: dict,
        *,
        source_topic: str = "",
        ack=None,
        nack=None,
    ) -> None:
        """Dispatch a normalized {thread, event} payload to the event broker."""
        await self.dispatch_inbound_envelope(
            thread=payload["thread"],
            event=payload["event"],
            publish_timestamp=payload.get("publish_timestamp"),
            source_topic=source_topic,
            ack=ack,
            nack=nack,
        )

    async def dispatch_inbound_envelope(
        self,
        *,
        thread: str,
        event: dict,
        publish_timestamp: float | None = None,
        source_topic: str = "",
        ack=None,
        nack=None,
    ) -> None:
        """Map a comms envelope onto the existing app:comms:* broker contract."""

        async def publish(channel: str, payload: str) -> None:
            await self.event_broker.publish(channel, payload)

        async def publish_blocking(channel: str, payload: str) -> None:
            await self.event_broker.publish(channel, payload)

        def schedule(coro) -> None:
            asyncio.create_task(coro)

        def ack_now() -> None:
            if ack is not None:
                self._ack_callback(ack, publish_timestamp, source_topic)

        def nack_now() -> None:
            if nack is not None:
                nack()

        try:
            if thread == "assistant_update":
                details = {
                    "api_key": event["api_key"],
                    "binding_id": event.get("binding_id", ""),
                    "medium": event.get("medium", "assistant_update"),
                    "assistant_id": event["assistant_id"],
                    "user_id": event["user_id"],
                    "assistant_first_name": event["assistant_first_name"],
                    "assistant_surname": event["assistant_surname"],
                    "assistant_age": event["assistant_age"],
                    "assistant_nationality": event["assistant_nationality"],
                    "assistant_timezone": event.get("assistant_timezone", ""),
                    "assistant_about": event["assistant_about"],
                    "assistant_job_title": event.get("assistant_job_title", ""),
                    "assistant_number": event["assistant_number"],
                    "assistant_email": event["assistant_email"],
                    "assistant_email_provider": event.get(
                        "assistant_email_provider",
                        "google_workspace",
                    ),
                    "assistant_whatsapp_number": event.get(
                        "assistant_whatsapp_number",
                        "",
                    ),
                    "assistant_discord_bot_id": event.get(
                        "assistant_discord_bot_id",
                        "",
                    ),
                    "user_first_name": event["user_first_name"],
                    "user_surname": event["user_surname"],
                    "user_number": event["user_number"],
                    "user_email": event["user_email"],
                    "user_whatsapp_number": event.get("user_whatsapp_number", ""),
                    "voice_provider": event["voice_provider"],
                    "voice_id": event["voice_id"],
                    "desktop_mode": event.get("desktop_mode", "ubuntu"),
                    "user_desktop_mode": event.get("user_desktop_mode"),
                    "user_desktop_filesys_sync": event.get(
                        "user_desktop_filesys_sync",
                        False,
                    ),
                    "user_desktop_url": event.get("user_desktop_url"),
                    "org_id": event.get("org_id"),
                    "org_name": event.get("org_name", ""),
                    "team_ids": event.get("team_ids") or [],
                    "space_ids": event.get("space_ids") or [],
                    "update_kind": event.get("update_kind", "general"),
                    "demo_id": event.get("demo_id"),
                }
                await publish(
                    "app:comms:assistant_update",
                    AssistantUpdateEvent(**details).to_json(),
                )
                ack_now()
                return

            if thread == "ping":
                await publish(
                    "app:comms:ping",
                    Ping(kind="keepalive").to_json(),
                )
                ack_now()
                return

            if thread == "attachment_ingestion_complete":
                await self._handle_attachment_ingestion_complete(event)
                ack_now()
                return

            if thread == "unity_system_event":
                system_event_type = event.get("event_type")
                system_message = event.get("message")
                reason = str(system_message) if system_message is not None else ""

                desktop_ready_ttl = 300
                if (
                    system_event_type == "assistant_desktop_ready"
                    and publish_timestamp is not None
                    and time.time() - publish_timestamp > desktop_ready_ttl
                ):
                    age = time.time() - publish_timestamp
                    LOGGER.warning(
                        f"{DEFAULT_ICON} Discarding stale assistant_desktop_ready "
                        f"(age={age:.0f}s, TTL={desktop_ready_ttl}s)",
                    )
                    ack_now()
                    return

                system_event_map = {
                    "sync_contacts": lambda r: SyncContacts(
                        reason=r or "Contact sync requested via system event.",
                    ),
                    "task_due": lambda r: _task_due_event_from_payload(
                        event,
                        reason=r,
                    ),
                    "inactivity_followup": lambda r: _inactivity_followup_event_from_payload(
                        event,
                        reason=r,
                    ),
                    "assistant_screen_share_started": lambda r: AssistantScreenShareStarted(
                        reason=r or "User enabled assistant screen sharing.",
                    ),
                    "assistant_screen_share_stopped": lambda r: AssistantScreenShareStopped(
                        reason=r or "User disabled assistant screen sharing.",
                    ),
                    "user_screen_share_started": lambda r: UserScreenShareStarted(
                        reason=r or "User started sharing their screen.",
                    ),
                    "user_screen_share_stopped": lambda r: UserScreenShareStopped(
                        reason=r or "User stopped sharing their screen.",
                    ),
                    "user_webcam_started": lambda r: UserWebcamStarted(),
                    "user_webcam_stopped": lambda r: UserWebcamStopped(),
                    "user_remote_control_started": lambda r: UserRemoteControlStarted(
                        reason=r or "User took remote control of assistant desktop.",
                    ),
                    "user_remote_control_stopped": lambda r: UserRemoteControlStopped(
                        reason=r
                        or "User released remote control of assistant desktop.",
                    ),
                    "assistant_desktop_ready": lambda r: AssistantDesktopReady(
                        binding_id=event.get("binding_id") or "",
                        desktop_url=event.get("desktop_url")
                        or SESSION_DETAILS.assistant.desktop_url
                        or "",
                        vm_type=event.get("vm_type")
                        or SESSION_DETAILS.assistant.desktop_mode,
                    ),
                }

                factory = system_event_map.get(system_event_type)
                if factory is not None:
                    mapped_event = factory(reason)
                    if mapped_event is not None:
                        await publish(
                            f"app:comms:{system_event_type}",
                            mapped_event.to_json(),
                        )
                ack_now()
                return

            if thread in events_map:
                contacts = [*event.get("contacts", []), _get_local_contact()]
                await publish(
                    "app:comms:backup_contacts",
                    BackupContactsEvent(contacts=contacts).to_json(),
                )

                content = event["body"]

                if (
                    thread in ("teams_chat", "teams_channel")
                    and event.get("content_type") == "html"
                ):
                    content = _teams_html_to_text(content)

                if thread == "email":
                    content = "Subject: " + event["subject"] + "\n\n" + event["body"]
                    raw_from = event["from"]
                    contact_detail = (
                        raw_from.split("<")[1].rstrip(">")
                        if "<" in raw_from
                        else raw_from.strip()
                    )
                    medium_for_blacklist = Medium.EMAIL

                    if _is_blacklisted(medium_for_blacklist, contact_detail):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted email from: {contact_detail}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (c for c in contacts if c["email_address"] == contact_detail),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            contact_detail,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for email from: {contact_detail}",
                        )
                        ack_now()
                        return

                    attachments = event.get("attachments") or []

                    def _normalize_recipients(value):
                        if not value:
                            return []
                        if isinstance(value, str):
                            return [value] if value else []
                        return list(value)

                    await publish(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            subject=event["subject"],
                            body=event["body"],
                            contact=contact,
                            email_id=event["email_id"],
                            attachments=attachments,
                            to=_normalize_recipients(event.get("to")),
                            cc=_normalize_recipients(event.get("cc")),
                            bcc=_normalize_recipients(event.get("bcc")),
                        ).to_json(),
                    )

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=(
                                    event["subject"][:100]
                                    if event.get("subject")
                                    else ""
                                ),
                            ).to_json(),
                        )

                    if attachments:
                        schedule(
                            add_email_attachments(
                                attachments,
                                SESSION_DETAILS.assistant.email,
                                event.get("message_id")
                                or event.get("gmail_message_id", ""),
                            ),
                        )

                    ack_now()
                    return

                if thread == "unify_message":
                    target_contact_id = event.get("contact_id")
                    if target_contact_id is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Error: contact_id is required for unify_message, "
                            "skipping message",
                        )
                        ack_now()
                        return

                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        None,
                    )
                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Error: contact_id {target_contact_id} not found in "
                            f"contacts list, skipping message",
                        )
                        ack_now()
                        return

                    attachments = event.get("attachments") or []
                    await publish(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                            attachments=attachments,
                        ).to_json(),
                    )

                    if attachments:
                        schedule(add_unify_message_attachments(attachments))

                    ack_now()
                    return

                if thread == "api_message":
                    target_contact_id = event.get("contact_id", 1)
                    contact = next(
                        (c for c in contacts if c["contact_id"] == target_contact_id),
                        contacts[0] if contacts else {},
                    )
                    api_message_id = event.get("api_message_id", "")
                    attachments = event.get("attachments") or []
                    tags = event.get("tags") or []

                    await publish(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                            api_message_id=api_message_id,
                            attachments=attachments,
                            tags=tags,
                        ).to_json(),
                    )

                    if attachments:
                        schedule(add_unify_message_attachments(attachments))

                    ack_now()
                    return

                if thread == "whatsapp":
                    if event.get("type") == "call_permission_response":
                        contact_number = event.get("contact_number", "")
                        accepted = event.get("payload") == "ACCEPTED"
                        contact = next(
                            (
                                c
                                for c in contacts
                                if c.get("whatsapp_number") == contact_number
                            ),
                            None,
                        )
                        if contact is None:
                            contact = next(
                                (
                                    c
                                    for c in contacts
                                    if c.get("phone_number") == contact_number
                                ),
                                None,
                            )
                        if contact is None:
                            LOGGER.error(
                                f"{DEFAULT_ICON} Failed to resolve contact for WhatsApp call permission from: {contact_number}",
                            )
                            ack_now()
                            return

                        await publish(
                            "app:comms:whatsapp_call_permission",
                            WhatsAppCallPermissionResponse(
                                contact=contact,
                                accepted=accepted,
                            ).to_json(),
                        )
                        ack_now()
                        return

                    raw_from = event["from_number"].strip()
                    contact_detail = (
                        raw_from.replace("whatsapp:", "")
                        if raw_from.startswith("whatsapp:")
                        else raw_from
                    )
                    medium_for_blacklist = Medium.WHATSAPP_MESSAGE

                    if _is_blacklisted(medium_for_blacklist, contact_detail):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted WhatsApp from: {contact_detail}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (
                            c
                            for c in contacts
                            if c.get("whatsapp_number") == contact_detail
                            or c["phone_number"] == contact_detail
                        ),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            contact_detail,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for WhatsApp from: {contact_detail}",
                        )
                        ack_now()
                        return

                    attachments = event.get("attachments") or []
                    await publish(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=contact,
                            **({"attachments": attachments} if attachments else {}),
                        ).to_json(),
                    )

                    if attachments:
                        schedule(add_unify_message_attachments(attachments))

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=content[:100] if content else "",
                            ).to_json(),
                        )

                    ack_now()
                    return

                if thread == "discord":
                    sender_discord_id = event.get("sender_discord_id", "")
                    message_id = event.get("message_id", "")
                    is_channel = event.get("is_channel", False)
                    channel_id = event.get("channel_id", "")
                    guild_id = event.get("guild_id")
                    bot_id = event.get("bot_id", "")
                    attachments = event.get("attachments") or []

                    if message_id and _already_seen_discord(message_id):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Skipping duplicate Discord message {message_id}",
                        )
                        ack_now()
                        return

                    medium_for_blacklist = (
                        Medium.DISCORD_CHANNEL_MESSAGE
                        if is_channel
                        else Medium.DISCORD_MESSAGE
                    )

                    if _is_blacklisted(medium_for_blacklist, sender_discord_id):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted Discord from: {sender_discord_id}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (
                            c
                            for c in contacts
                            if c.get("discord_id") == sender_discord_id
                        ),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            sender_discord_id,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for Discord from: {sender_discord_id}",
                        )
                        ack_now()
                        return

                    if is_channel:
                        discord_event = DiscordChannelMessageReceived(
                            contact=contact,
                            content=content,
                            channel_id=channel_id,
                            guild_id=guild_id or "",
                            bot_id=bot_id,
                            message_id=message_id,
                            attachments=attachments,
                        )
                        await publish(
                            "app:comms:discord_channel_message",
                            discord_event.to_json(),
                        )
                    else:
                        await publish(
                            "app:comms:discord_message",
                            events_map[thread](
                                content=content,
                                contact=contact,
                                channel_id=channel_id,
                                bot_id=bot_id,
                                message_id=message_id,
                                attachments=attachments,
                            ).to_json(),
                        )

                    if attachments:
                        schedule(add_unify_message_attachments(attachments))

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=content[:100] if content else "",
                            ).to_json(),
                        )

                    ack_now()
                    return

                if thread in ("teams_chat", "teams_channel"):
                    sender_email = event.get("sender", "")
                    message_id = event.get("message_id", "")
                    is_channel = thread == "teams_channel"
                    attachments = event.get("attachments") or []

                    if message_id and _already_seen_teams(message_id):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Skipping duplicate Teams message {message_id}",
                        )
                        ack_now()
                        return

                    medium_for_blacklist = (
                        Medium.TEAMS_CHANNEL_MESSAGE
                        if is_channel
                        else Medium.TEAMS_MESSAGE
                    )

                    if _is_blacklisted(medium_for_blacklist, sender_email):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted Teams from: {sender_email}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (c for c in contacts if c.get("email_address") == sender_email),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            medium_for_blacklist,
                            sender_email,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for Teams from: {sender_email}",
                        )
                        ack_now()
                        return

                    if event.get("participants_incomplete"):
                        LOGGER.warning(
                            f"{DEFAULT_ICON} Teams participants incomplete "
                            f"(reason={event.get('participants_reason', 'ok')}, "
                            f"thread={thread})",
                        )

                    participants_list = _resolve_teams_participants(
                        raw_participants=event.get("participants") or [],
                        sender_email=sender_email,
                        sender_contact_id=contact.get("contact_id"),
                        medium=medium_for_blacklist,
                        unknown_contact_resolver=_get_or_create_unknown_contact,
                    )

                    if is_channel:
                        await publish(
                            "app:comms:teams_channel_message",
                            events_map[thread](
                                contact=contact,
                                content=content,
                                channel_id=event.get("channel_id", ""),
                                team_id=event.get("team_id", ""),
                                message_id=message_id,
                                is_reply=event.get("is_reply", False),
                                parent_message_id=event.get("parent_message_id"),
                                thread_id=event.get("thread_id", ""),
                                post_subject=event.get("post_subject"),
                                attachments=attachments,
                                participants=participants_list,
                            ).to_json(),
                        )
                    else:
                        await publish(
                            "app:comms:teams_message",
                            events_map[thread](
                                contact=contact,
                                content=content,
                                chat_id=event.get("chat_id", ""),
                                message_id=message_id,
                                chat_type=event.get("chat_type"),
                                chat_topic=event.get("chat_topic"),
                                attachments=attachments,
                                participants=participants_list,
                            ).to_json(),
                        )

                    if attachments:
                        schedule(add_unify_message_attachments(attachments))

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=medium_for_blacklist,
                                message_preview=content[:100] if content else "",
                            ).to_json(),
                        )

                    ack_now()
                    return

                contact_detail = event["from_number"].strip()
                medium_for_blacklist = Medium.SMS_MESSAGE

                if _is_blacklisted(medium_for_blacklist, contact_detail):
                    LOGGER.debug(
                        f"{DEFAULT_ICON} Ignoring blacklisted SMS from: {contact_detail}",
                    )
                    ack_now()
                    return

                contact = next(
                    (c for c in contacts if c["phone_number"] == contact_detail),
                    None,
                )
                is_new_unknown = False
                if contact is None:
                    contact = _get_or_create_unknown_contact(
                        medium_for_blacklist,
                        contact_detail,
                    )
                    is_new_unknown = contact is not None

                if contact is None:
                    LOGGER.error(
                        f"{DEFAULT_ICON} Failed to resolve contact for SMS from: {contact_detail}",
                    )
                    ack_now()
                    return

                await publish(
                    f"app:comms:{thread}_message",
                    events_map[thread](
                        content=content,
                        contact=contact,
                    ).to_json(),
                )

                if is_new_unknown:
                    await publish(
                        "app:comms:unknown_contact_created",
                        UnknownContactCreated(
                            contact=contact,
                            medium=medium_for_blacklist,
                            message_preview=content[:100] if content else "",
                        ).to_json(),
                    )

                ack_now()
                return

            elif thread == "log_pre_hire_chats":
                try:
                    assistant_id = event.get("assistant_id", "")
                    body = event.get("body", []) or []

                    published = 0
                    for item in body:
                        try:
                            msg_content = item.get("msg", "")
                            if not isinstance(msg_content, str):
                                msg_content = str(msg_content)

                            await publish(
                                "app:comms:pre_hire",
                                PreHireMessage(
                                    content=msg_content,
                                    role=item.get("role"),
                                    exchange_id=UNASSIGNED,
                                ).to_json(),
                            )
                            published += 1
                        except Exception as inner_exc:
                            LOGGER.debug(
                                f"{DEFAULT_ICON} Skipping malformed pre-hire item: {inner_exc}",
                            )

                    LOGGER.debug(
                        f"{DEFAULT_ICON} Logged {published} pre-hire chat message(s) for assistant {assistant_id}",
                    )
                    ack_now()
                except Exception as exc:
                    LOGGER.error(
                        f"{DEFAULT_ICON} Error processing pre-hire logs: {exc}",
                    )
                    nack_now()
                return

            if thread == "recording_ready":
                await publish(
                    "app:comms:recording_ready",
                    RecordingReady(
                        conference_name=event.get("conference_name", ""),
                        recording_url=event.get("recording_url", ""),
                    ).to_json(),
                )
                ack_now()
                return

            if "call" in thread or "meet" in thread:
                contacts = [*event.get("contacts", []), _get_local_contact()]
                await publish_blocking(
                    "app:comms:backup_contacts",
                    BackupContactsEvent(contacts=contacts).to_json(),
                )

                if thread == "unify_meet":
                    call_event = UnifyMeetReceived(
                        contact=next(c for c in contacts if c["contact_id"] == 1),
                        room_name=event.get("livekit_room"),
                    )
                    event_topic = "app:comms:unify_meet_received"
                elif thread == "whatsapp_call":
                    number = event.get("caller_number", event.get("user_number"))
                    if number and number.startswith("whatsapp:"):
                        number = number[len("whatsapp:") :]

                    if _is_blacklisted(Medium.WHATSAPP_MESSAGE, number):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted WhatsApp call from: {number}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (c for c in contacts if c.get("whatsapp_number") == number),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            Medium.WHATSAPP_CALL,
                            number,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for WhatsApp call from: {number}",
                        )
                        ack_now()
                        return

                    call_event = WhatsAppCallReceived(
                        contact=contact,
                        conference_name=event.get("conference_name", ""),
                    )
                    event_topic = "app:comms:whatsapp_call_received"

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=Medium.WHATSAPP_CALL,
                                message_preview="Incoming WhatsApp call",
                            ).to_json(),
                        )
                elif thread == "call":
                    number = event.get("caller_number", event.get("user_number"))

                    if _is_blacklisted(Medium.PHONE_CALL, number):
                        LOGGER.debug(
                            f"{DEFAULT_ICON} Ignoring blacklisted call from: {number}",
                        )
                        ack_now()
                        return

                    contact = next(
                        (c for c in contacts if c["phone_number"] == number),
                        None,
                    )
                    is_new_unknown = False
                    if contact is None:
                        contact = _get_or_create_unknown_contact(
                            Medium.PHONE_CALL,
                            number,
                        )
                        is_new_unknown = contact is not None

                    if contact is None:
                        LOGGER.error(
                            f"{DEFAULT_ICON} Failed to resolve contact for call from: {number}",
                        )
                        ack_now()
                        return

                    call_event = PhoneCallReceived(
                        contact=contact,
                        conference_name=event.get("conference_name", ""),
                    )
                    event_topic = "app:comms:call_received"

                    if is_new_unknown:
                        await publish(
                            "app:comms:unknown_contact_created",
                            UnknownContactCreated(
                                contact=contact,
                                medium=Medium.PHONE_CALL,
                                message_preview="Incoming phone call",
                            ).to_json(),
                        )
                elif thread == "whatsapp_call_answered":
                    number = event.get("user_number")
                    if number and number.startswith("whatsapp:"):
                        number = number[len("whatsapp:") :]
                    contact = next(
                        (c for c in contacts if c.get("whatsapp_number") == number),
                        None,
                    )
                    if contact is None:
                        contact = next(
                            (c for c in contacts if c.get("phone_number") == number),
                            None,
                        )
                    if contact is None:
                        contact = next(c for c in contacts if c["contact_id"] == 1)
                    call_event = WhatsAppCallAnswered(contact=contact)
                    event_topic = "app:comms:whatsapp_call_answered"
                elif thread == "whatsapp_call_not_answered":
                    number = event.get("user_number")
                    if number and number.startswith("whatsapp:"):
                        number = number[len("whatsapp:") :]
                    call_status = event.get("call_status", "no-answer")
                    contact = next(
                        (c for c in contacts if c.get("whatsapp_number") == number),
                        None,
                    )
                    if contact is None:
                        contact = next(
                            (c for c in contacts if c.get("phone_number") == number),
                            None,
                        )
                    if contact is None:
                        contact = next(c for c in contacts if c["contact_id"] == 1)
                    call_event = WhatsAppCallNotAnswered(
                        contact=contact,
                        reason=event.get("call_status", "no-answer"),
                    )
                    event_topic = "app:comms:whatsapp_call_not_answered"
                elif thread == "call_not_answered":
                    number = event.get("user_number")
                    contact = next(
                        (c for c in contacts if c["phone_number"] == number),
                        None,
                    )
                    if contact is None:
                        contact = next(c for c in contacts if c["contact_id"] == 1)
                    call_event = PhoneCallNotAnswered(
                        contact=contact,
                        reason=event.get("call_status", "no-answer"),
                    )
                    event_topic = "app:comms:call_not_answered"
                elif thread == "call_answered":
                    number = event.get("user_number")
                    contact = next(
                        (c for c in contacts if c["phone_number"] == number),
                        None,
                    )
                    if contact is None:
                        contact = next(c for c in contacts if c["contact_id"] == 1)
                    call_event = PhoneCallAnswered(contact=contact)
                    event_topic = "app:comms:call_answered"
                else:
                    LOGGER.warning(
                        f"{DEFAULT_ICON} Unhandled call/meet thread: {thread}",
                    )
                    ack_now()
                    return

                await publish_blocking(event_topic, call_event.to_json())
                ack_now()
                return

            if thread != "assistant_desktop_ready":
                LOGGER.error(f"{DEFAULT_ICON} Unknown event type: {thread}")
            ack_now()
        except Exception as exc:
            LOGGER.error(f"{DEFAULT_ICON} Error processing message: {exc}")
            publish_system_error(
                "An internal error occurred while processing a message. "
                "The assistant may not have received your last message.",
                error_type="message_failed",
            )
            ack_now()

    def handle_message(
        self,
        message: Any,
        subscription_id: str = "",
    ):
        """
        Handle incoming messages from PubSub subscriptions.

        NOTE: This method is called from a GCP PubSub thread pool thread,
        NOT from the asyncio event loop. It decodes the Pub/Sub payload and
        schedules the shared async envelope dispatcher on the main loop.
        """
        topic = subscription_id.removesuffix("-sub")
        try:
            payload = json.loads(message.data.decode("utf-8"))
            thread = payload["thread"]
            LOGGER.debug(
                f"{DEFAULT_ICON} Received message from {thread}: {message.data.decode('utf-8')}",
            )
            future = asyncio.run_coroutine_threadsafe(
                self.dispatch_envelope_payload(
                    payload,
                    source_topic=topic,
                    ack=message.ack,
                    nack=message.nack,
                ),
                self.loop,
            )
            future.add_done_callback(self._log_dispatch_future)
            if "call" in thread or "meet" in thread:
                future.result()
        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error processing message: {e}")
            publish_system_error(
                "An internal error occurred while processing a message. "
                "The assistant may not have received your last message.",
                error_type="message_failed",
            )
            message.ack()

    def subscribe_to_topic(self, subscription_id: str, max_messages: int | None = None):
        """Subscribe to a specific PubSub topic and process messages."""
        if pubsub_v1 is None:
            LOGGER.error(
                f"{ICONS['subscription']} Google Pub/Sub client is unavailable; "
                "hosted subscriptions are disabled in this environment.",
            )
            return
        if not SETTINGS.GCP_PROJECT_ID:
            LOGGER.error(
                f"{ICONS['subscription']} GCP_PROJECT_ID is not set — "
                f"cannot subscribe to Pub/Sub. Set the GCP_PROJECT_ID environment variable.",
            )
            return
        try:
            # Let GCP libraries handle authentication automatically
            if self.credentials:
                subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
            else:
                subscriber = pubsub_v1.SubscriberClient()
            subscription_path = subscriber.subscription_path(
                SETTINGS.GCP_PROJECT_ID,
                subscription_id,
            )

            LOGGER.debug(
                f"{ICONS['subscription']} Starting subscription to {subscription_path} (max_messages={max_messages})",
            )

            flow_control = (
                pubsub_v1.types.FlowControl(max_messages=max_messages)
                if max_messages
                else pubsub_v1.types.FlowControl()
            )

            callback = partial(self.handle_message, subscription_id=subscription_id)
            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=callback,
                flow_control=flow_control,
            )

            # Store the future for cleanup
            self.subscribers[subscription_id] = streaming_pull_future
            LOGGER.info(
                f"{ICONS['subscription']} Subscription active: {subscription_path} "
                f"(max_messages={max_messages})",
            )

        except Exception as e:
            LOGGER.error(
                f"{ICONS['subscription']} Error setting up subscription {subscription_id}: {e}",
            )

    async def _poll_for_assignment(self):
        """Wait for cluster-owned AssistantSession assignment.

        The session controller writes a session reference onto the real Job.
        Unity watches for that reference, reads the AssistantSession plus its
        bootstrap Secret, and emits the same StartupEvent path the existing
        ConversationManager already handles.
        """
        job_name = SETTINGS.conversation.JOB_NAME

        if not job_name:
            LOGGER.error(
                f"{DEFAULT_ICON} Cannot poll for assignment: "
                f"JOB_NAME not configured",
            )
            return

        LOGGER.debug(
            f"{DEFAULT_ICON} Waiting for AssistantSession assignment on {job_name}",
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                LOGGER.info(
                    f"{DEFAULT_ICON} Assignment poll attempt {attempt} for {job_name}",
                )
                session_name = await asyncio.to_thread(
                    wait_for_assistant_session_name,
                    job_name,
                )
                LOGGER.info(
                    f"{DEFAULT_ICON} Assignment session discovered for {job_name}: "
                    f"{session_name}",
                )
                job_assignment = await asyncio.to_thread(
                    read_job_assignment_record,
                    job_name,
                )
                if job_assignment.session_name != session_name:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Ignoring assignment on {job_name}: "
                        f"job now points at {job_assignment.session_name or 'no-session'} "
                        f"instead of {session_name}",
                    )
                    await asyncio.sleep(5)
                    continue
                if not job_assignment.binding_id:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Waiting for binding identity on {job_name} "
                        f"before bootstrapping {session_name}",
                    )
                    await asyncio.sleep(5)
                    continue
                session = await asyncio.to_thread(read_assistant_session, session_name)
                session_spec = session.get("spec") or {}
                session_status = session.get("status") or {}
                session_binding_id = str(
                    ((session_status.get("binding") or {}).get("id") or ""),
                )
                activation_id = str(session_spec.get("activationId", "") or "")
                secret_name = str(session_spec.get("startupSecretRef", "") or "")
                if not secret_name:
                    raise RuntimeError(
                        f"AssistantSession {session_name} missing startupSecretRef",
                    )
                if not activation_id:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Waiting for activation ownership on "
                        f"{session_name} before bootstrapping {job_name}",
                    )
                    await asyncio.sleep(5)
                    continue
                if not session_binding_id:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Waiting for current binding on "
                        f"{session_name} before bootstrapping {job_name}",
                    )
                    await asyncio.sleep(5)
                    continue
                if job_assignment.binding_id != session_binding_id:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Ignoring stale assignment on {job_name}: "
                        f"job binding {job_assignment.binding_id} != "
                        f"session binding {session_binding_id}",
                    )
                    await asyncio.sleep(5)
                    continue
                LOGGER.info(
                    f"{DEFAULT_ICON} Assignment session loaded for {job_name}: "
                    f"phase={(session_status.get('phase') or '')}, "
                    f"secret={secret_name}, "
                    f"binding_id={session_binding_id}",
                )

                secret_record = await asyncio.to_thread(
                    read_session_bootstrap_secret_record,
                    secret_name,
                )
                event = secret_record.payload
                if not event:
                    raise RuntimeError(
                        f"AssistantSession bootstrap secret {secret_name} is empty",
                    )
                if secret_record.owner_session_name != session_name:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Ignoring stale bootstrap Secret "
                        f"{secret_record.name} on {job_name}: owner session "
                        f"{secret_record.owner_session_name or 'missing'} != {session_name}",
                    )
                    await asyncio.sleep(5)
                    continue
                if secret_record.owner_activation_id != activation_id:
                    LOGGER.info(
                        f"{DEFAULT_ICON} Ignoring stale bootstrap Secret "
                        f"{secret_record.name} on {job_name}: owner activation "
                        f"{secret_record.owner_activation_id or 'missing'} != {activation_id}",
                    )
                    await asyncio.sleep(5)
                    continue
                expected_assistant_id = str(session_spec.get("assistantId", "") or "")
                event_assistant_id = str(event.get("assistant_id", "") or "")
                if (
                    expected_assistant_id
                    and event_assistant_id
                    and event_assistant_id != expected_assistant_id
                ):
                    LOGGER.info(
                        f"{DEFAULT_ICON} Ignoring bootstrap Secret {secret_record.name} "
                        f"on {job_name}: payload assistant {event_assistant_id} != "
                        f"session assistant {expected_assistant_id}",
                    )
                    await asyncio.sleep(5)
                    continue
                LOGGER.info(
                    f"{DEFAULT_ICON} Bootstrap secret read for {job_name}: "
                    f"assistant_id={event.get('assistant_id')} medium={event.get('medium')}",
                )

                LOGGER.debug(
                    f"{DEFAULT_ICON} Assignment detected for assistant "
                    f"{event.get('assistant_id')} via {session_name} on {job_name}",
                )

                SESSION_DETAILS.assistant.agent_id = int(event["assistant_id"])
                self.subscribe_to_topic(_get_subscription_id(), max_messages=10)
                LOGGER.info(
                    f"{DEFAULT_ICON} Assistant inbound subscription established for "
                    f"{job_name}: {_get_subscription_id()}",
                )

                details = {
                    "api_key": event["api_key"],
                    "binding_id": session_binding_id,
                    "medium": event.get("medium", "startup"),
                    "assistant_id": event["assistant_id"],
                    "user_id": event["user_id"],
                    "assistant_first_name": event["assistant_first_name"],
                    "assistant_surname": event["assistant_surname"],
                    "assistant_age": event["assistant_age"],
                    "assistant_nationality": event["assistant_nationality"],
                    "assistant_timezone": event.get("assistant_timezone", ""),
                    "assistant_about": event["assistant_about"],
                    "assistant_job_title": event.get("assistant_job_title", ""),
                    "assistant_number": event["assistant_number"],
                    "assistant_email": event["assistant_email"],
                    "assistant_email_provider": event.get(
                        "assistant_email_provider",
                        "google_workspace",
                    ),
                    "assistant_whatsapp_number": event.get(
                        "assistant_whatsapp_number",
                        "",
                    ),
                    "assistant_discord_bot_id": event.get(
                        "assistant_discord_bot_id",
                        "",
                    ),
                    "user_first_name": event["user_first_name"],
                    "user_surname": event["user_surname"],
                    "user_number": event["user_number"],
                    "user_email": event["user_email"],
                    "user_whatsapp_number": event.get("user_whatsapp_number", ""),
                    "voice_provider": event["voice_provider"],
                    "voice_id": event["voice_id"],
                    "desktop_mode": event.get("desktop_mode", "ubuntu"),
                    "user_desktop_mode": event.get("user_desktop_mode"),
                    "user_desktop_filesys_sync": event.get(
                        "user_desktop_filesys_sync",
                        False,
                    ),
                    "user_desktop_url": event.get("user_desktop_url"),
                    "org_id": event.get("org_id"),
                    "org_name": event.get("org_name", ""),
                    "team_ids": event.get("team_ids") or [],
                    "space_ids": event.get("space_ids") or [],
                    "wake_reasons": event.get("wake_reasons") or [],
                    "demo_id": event.get("demo_id"),
                }

                await self.event_broker.publish(
                    "app:comms:startup",
                    StartupEvent(**details).to_json(),
                )
                LOGGER.info(
                    f"{DEFAULT_ICON} StartupEvent published for assistant "
                    f"{event.get('assistant_id')} on {job_name}",
                )
                await asyncio.to_thread(mark_job_container_ready, job_name)
                LOGGER.info(
                    f"{DEFAULT_ICON} Container-ready signalled for {job_name}",
                )
                return
            except Exception as e:
                LOGGER.exception(
                    f"{DEFAULT_ICON} AssistantSession discovery failed for {job_name} "
                    f"on attempt {attempt}: {e}",
                )
                await asyncio.sleep(5)

    async def start(self):
        """Start all subscriptions and maintain connection to event manager."""
        if SESSION_DETAILS.assistant.agent_id is None:
            asyncio.create_task(self._poll_for_assignment())
            asyncio.create_task(self.send_pings())
        else:
            self.subscribe_to_topic(_get_subscription_id(), max_messages=10)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            LOGGER.debug(f"{ICONS['lifecycle']} Shutting down...")
            for future in self.subscribers.values():
                future.cancel()

    async def send_pings(self):
        """Send periodic pings to keep the event manager alive while waiting for startup."""
        LOGGER.debug(
            f"{ICONS['subscription']} Starting ping mechanism for idle container...",
        )
        while True:
            try:
                # Send ping to event manager (direct await since we're in async context)
                await self.event_broker.publish(
                    "app:comms:ping",
                    Ping(kind="keepalive").to_json(),
                )

                # Wait 30 seconds before next ping (half the inactivity timeout)
                await asyncio.sleep(30)

                # Check if we've received a startup message (indicated by assistant_id changed)
                if SESSION_DETAILS.assistant.agent_id is not None:
                    LOGGER.debug(
                        f"{ICONS['subscription']} Startup received, stopping ping mechanism",
                    )
                    break

            except Exception as e:
                LOGGER.error(f"{ICONS['subscription']} Error in ping mechanism: {e}")
                await asyncio.sleep(30)  # Continue trying


async def main():
    """Main entry point for the communication manager application."""
    from unity.conversation_manager.event_broker import get_event_broker

    event_broker = get_event_broker()
    manager = CommsManager(event_broker)
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main(), debug=SETTINGS.UNITY_ASYNCIO_DEBUG)
