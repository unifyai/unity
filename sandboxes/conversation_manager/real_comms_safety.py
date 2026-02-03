"""
Real-comms safety layer for the ConversationManager sandbox.

Sandbox-only monkey patching of `unity.conversation_manager.domains.comms_utils`
to require an explicit Y/N confirmation before sending any *real* outbound comms.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

from sandboxes.conversation_manager.io_gate import gated_input

LG = logging.getLogger("conversation_manager_sandbox")


class RealCommsSafetyError(Exception):
    """Raised when a real-comms action is blocked by the user."""


def _yn_prompt(*, medium: str, recipient: str, content_preview: str) -> bool:
    print("\n⚠️ REAL-COMMS MODE: Confirm action")
    print(f"   Medium: {medium}")
    print(f"   Recipient: {recipient}")
    if content_preview:
        wrapped = "\n".join(
            "   " + line for line in textwrap.wrap(content_preview, width=90)
        )
        print("   Content:")
        print(wrapped)
    resp = gated_input("   Proceed? (Y/N) [default: N]: ").strip().upper()
    return resp == "Y"


@dataclass
class SafetyConfig:
    auto_confirm: bool = False
    debug: bool = True


def apply_real_comms_safety(*, config: SafetyConfig) -> None:
    """
    Apply monkey patches to comms_utils functions.

    Must be called AFTER `run_conversation_manager()` so imports are initialized.
    """
    from unity.conversation_manager.domains import comms_utils
    from unity.session_details import SESSION_DETAILS
    from unity.settings import SETTINGS

    original_send_sms = comms_utils.send_sms_message_via_number
    original_send_email = comms_utils.send_email_via_address
    original_send_unify = comms_utils.send_unify_message
    original_start_call = comms_utils.start_call

    def _debug_env_snapshot(*, medium: str) -> None:
        if not config.debug:
            return
        try:
            # Never print secrets; only booleans / non-sensitive strings.
            comms_url = getattr(SETTINGS.conversation, "COMMS_URL", "") or ""
            assistant_number_present = bool(
                getattr(SESSION_DETAILS.assistant, "number", "") or "",
            )
            user_number_present = bool(
                getattr(SESSION_DETAILS.user, "number", "") or "",
            )
            admin_key_present = bool(getattr(SETTINGS, "ORCHESTRA_ADMIN_KEY", None))
            LG.info(
                "[real-comms][debug] medium=%s comms_url=%s assistant_number_present=%s user_number_present=%s admin_key_present=%s",
                medium,
                comms_url,
                assistant_number_present,
                user_number_present,
                admin_key_present,
            )
        except Exception:
            pass

    async def safe_send_sms(to_number: str, content: str, *args, **kwargs):
        _debug_env_snapshot(medium="SMS")
        if not config.auto_confirm:
            ok = _yn_prompt(
                medium="SMS",
                recipient=to_number,
                content_preview=content,
            )
            if not ok:
                raise RealCommsSafetyError("Action blocked by user (real-comms safety)")
        try:
            res = await original_send_sms(to_number, content, *args, **kwargs)
            if config.debug:
                LG.info(
                    "[real-comms][debug] SMS result_success=%s",
                    bool(
                        getattr(res, "get", lambda _k, _d=None: None)("success", False),
                    ),
                )
            return res
        except Exception as exc:
            LG.exception(
                "[real-comms][debug] SMS send raised: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise

    async def safe_send_email(
        to_email: str,
        subject: str,
        body: str,
        email_id: str = None,
        attachment: dict | None = None,
        *args,
        **kwargs,
    ):
        _debug_env_snapshot(medium="Email")
        preview = f"Subject: {subject}\n\n{body}"
        if attachment and attachment.get("filename"):
            preview += f"\n\n(attachment: {attachment.get('filename')})"
        if not config.auto_confirm:
            ok = _yn_prompt(
                medium="Email",
                recipient=to_email,
                content_preview=preview,
            )
            if not ok:
                raise RealCommsSafetyError("Action blocked by user (real-comms safety)")
        try:
            res = await original_send_email(
                to_email,
                subject,
                body,
                email_id=email_id,
                attachment=attachment,
                *args,
                **kwargs,
            )
            if config.debug:
                LG.info(
                    "[real-comms][debug] Email result_success=%s",
                    bool(
                        getattr(res, "get", lambda _k, _d=None: None)("success", False),
                    ),
                )
            return res
        except Exception as exc:
            LG.exception(
                "[real-comms][debug] Email send raised: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise

    async def safe_send_unify(
        content: str,
        contact_id: int = 1,
        attachment: dict | None = None,
        *args,
        **kwargs,
    ):
        _debug_env_snapshot(medium="UnifyMessage")
        preview = content
        if attachment and attachment.get("filename"):
            preview += f"\n\n(attachment: {attachment.get('filename')})"
        if not config.auto_confirm:
            ok = _yn_prompt(
                medium="Unify Message",
                recipient=f"contact_id={contact_id}",
                content_preview=preview,
            )
            if not ok:
                raise RealCommsSafetyError("Action blocked by user (real-comms safety)")
        try:
            res = await original_send_unify(
                content,
                contact_id=contact_id,
                attachment=attachment,
                *args,
                **kwargs,
            )
            if config.debug:
                LG.info(
                    "[real-comms][debug] Unify message result_success=%s",
                    bool(
                        getattr(res, "get", lambda _k, _d=None: None)("success", False),
                    ),
                )
            return res
        except Exception as exc:
            LG.exception(
                "[real-comms][debug] Unify message send raised: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise

    async def safe_start_call(to_number: str, *args, **kwargs):
        _debug_env_snapshot(medium="PhoneCall")
        if not config.auto_confirm:
            ok = _yn_prompt(
                medium="Phone Call",
                recipient=to_number,
                content_preview="(start outbound call)",
            )
            if not ok:
                raise RealCommsSafetyError("Action blocked by user (real-comms safety)")
        try:
            res = await original_start_call(to_number, *args, **kwargs)
            if config.debug:
                LG.info(
                    "[real-comms][debug] Call result_success=%s",
                    bool(
                        getattr(res, "get", lambda _k, _d=None: None)("success", False),
                    ),
                )
            return res
        except Exception as exc:
            LG.exception(
                "[real-comms][debug] Call start raised: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise

    comms_utils.send_sms_message_via_number = safe_send_sms
    comms_utils.send_email_via_address = safe_send_email
    comms_utils.send_unify_message = safe_send_unify
    comms_utils.start_call = safe_start_call
