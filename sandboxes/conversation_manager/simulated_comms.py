"""
Simulated comms layer for the ConversationManager sandbox.

In simulated mode, the sandbox should be able to exercise the ConversationManager
"brain" without requiring any external comms infrastructure (COMMS service, GCP
Pub/Sub, provisioned phone numbers, etc.).

We implement this by monkey-patching `unity.conversation_manager.domains.comms_utils`
to return successful responses without performing any network calls.

This module is sandbox-only. It must be applied AFTER `run_conversation_manager()`
so imports and settings are initialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _Originals:
    send_sms_message_via_number: Any
    send_email_via_address: Any
    send_unify_message: Any
    upload_unify_attachment: Any
    start_call: Any


def apply_simulated_comms() -> Callable[[], None]:
    """
    Patch comms_utils to avoid real side effects and always succeed.

    Returns:
        A restore() callback that reverts the monkey patches.
    """
    from unity.conversation_manager.domains import comms_utils

    originals = _Originals(
        send_sms_message_via_number=comms_utils.send_sms_message_via_number,
        send_email_via_address=comms_utils.send_email_via_address,
        send_unify_message=comms_utils.send_unify_message,
        upload_unify_attachment=getattr(comms_utils, "upload_unify_attachment", None),
        start_call=comms_utils.start_call,
    )

    async def simulated_send_sms_message_via_number(
        to_number: str,
        content: str,
        *args,
        **kwargs,
    ):
        return {"success": True}

    async def simulated_send_email_via_address(
        to_email: str,
        subject: str,
        body: str,
        *args,
        **kwargs,
    ):
        return {"success": True}

    async def simulated_send_unify_message(
        content: str,
        contact_id: int = 1,
        attachment: dict | None = None,
        *args,
        **kwargs,
    ):
        return {"success": True}

    async def simulated_upload_unify_attachment(
        file_content: bytes,
        filename: str,
        *args,
        **kwargs,
    ):
        # brain_action_tools treats any dict with no "error" key as a success.
        return {
            "id": "sandbox",
            "filename": filename,
            "url": f"sandbox://attachment/{filename}",
        }

    async def simulated_start_call(to_number: str, *args, **kwargs):
        return {"success": True}

    comms_utils.send_sms_message_via_number = simulated_send_sms_message_via_number
    comms_utils.send_email_via_address = simulated_send_email_via_address
    comms_utils.send_unify_message = simulated_send_unify_message
    if originals.upload_unify_attachment is not None:
        comms_utils.upload_unify_attachment = simulated_upload_unify_attachment
    comms_utils.start_call = simulated_start_call

    def restore() -> None:
        comms_utils.send_sms_message_via_number = originals.send_sms_message_via_number
        comms_utils.send_email_via_address = originals.send_email_via_address
        comms_utils.send_unify_message = originals.send_unify_message
        if originals.upload_unify_attachment is not None:
            comms_utils.upload_unify_attachment = originals.upload_unify_attachment
        comms_utils.start_call = originals.start_call

    return restore
