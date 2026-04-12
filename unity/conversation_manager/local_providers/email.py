from __future__ import annotations

import asyncio
import base64
import contextlib
from email import message_from_bytes
from email.message import EmailMessage, Message
from email.utils import getaddresses, make_msgid
import imaplib
import os
import smtplib
import ssl
import uuid

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def is_email_configured() -> bool:
    """Return whether the local IMAP/SMTP email path is configured."""
    return bool(
        _env("UNITY_LOCAL_EMAIL_ADDRESS")
        and _env("UNITY_LOCAL_EMAIL_PASSWORD")
        and _env("UNITY_LOCAL_EMAIL_IMAP_HOST")
        and _env("UNITY_LOCAL_EMAIL_SMTP_HOST"),
    )


def _email_address() -> str:
    return _env("UNITY_LOCAL_EMAIL_ADDRESS") or (SESSION_DETAILS.assistant.email or "")


def _email_password() -> str:
    return _env("UNITY_LOCAL_EMAIL_PASSWORD")


def _imap_port() -> int:
    return int(_env("UNITY_LOCAL_EMAIL_IMAP_PORT", "993"))


def _smtp_port() -> int:
    return int(_env("UNITY_LOCAL_EMAIL_SMTP_PORT", "465"))


def _use_starttls() -> bool:
    return _env("UNITY_LOCAL_EMAIL_SMTP_STARTTLS", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extract_body(message: Message) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        return ""
    payload = message.get_payload(decode=True) or b""
    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _attachment_dicts(message: Message) -> list[dict]:
    attachments: list[dict] = []
    for part in message.walk():
        if part.get_content_disposition() != "attachment":
            continue
        filename = part.get_filename() or f"attachment_{uuid.uuid4().hex[:8]}"
        payload = part.get_payload(decode=True) or b""
        attachments.append(
            {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "content_base64": base64.b64encode(payload).decode("ascii"),
                "content_type": part.get_content_type(),
                "size_bytes": len(payload),
            },
        )
    return attachments


def _recipient_list(header_value: str | None) -> list[str]:
    if not header_value:
        return []
    return [addr for _, addr in getaddresses([header_value]) if addr]


class LocalEmailPoller:
    """Poll an IMAP inbox and dispatch email envelopes into CommsManager."""

    def __init__(self, dispatch_envelope):
        self._dispatch_envelope = dispatch_envelope
        self._task: asyncio.Task | None = None
        self._processed_ids: set[str] = set()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _poll_loop(self) -> None:
        interval = SETTINGS.conversation.LOCAL_EMAIL_POLL_INTERVAL
        while True:
            try:
                envelopes = await asyncio.to_thread(self._poll_once_sync)
                for envelope in envelopes:
                    await self._dispatch_envelope(envelope)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Local email poll failed: {exc}",
                )
            await asyncio.sleep(interval)

    def _poll_once_sync(self) -> list[dict]:
        if not is_email_configured():
            return []

        imap_host = _env("UNITY_LOCAL_EMAIL_IMAP_HOST")
        username = _email_address()
        password = _email_password()
        mailbox_name = _env("UNITY_LOCAL_EMAIL_IMAP_MAILBOX", "INBOX")

        mail = imaplib.IMAP4_SSL(imap_host, _imap_port())
        try:
            mail.login(username, password)
            mail.select(mailbox_name)
            status, data = mail.search(None, "UNSEEN")
            if status != "OK":
                return []

            envelopes: list[dict] = []
            for msg_num in data[0].split():
                status, fetched = mail.fetch(msg_num, "(RFC822)")
                if status != "OK":
                    continue
                raw_message = next(
                    (
                        item[1]
                        for item in fetched
                        if isinstance(item, tuple) and len(item) > 1
                    ),
                    None,
                )
                if raw_message is None:
                    continue
                message = message_from_bytes(raw_message)
                message_id = message.get("Message-ID") or make_msgid()
                if message_id in self._processed_ids:
                    continue

                body = _extract_body(message)
                attachments = _attachment_dicts(message)
                envelopes.append(
                    {
                        "thread": "email",
                        "event": {
                            "from": message.get("From", ""),
                            "subject": message.get("Subject", ""),
                            "body": body,
                            "email_id": message_id,
                            "attachments": attachments,
                            "to": _recipient_list(message.get("To")),
                            "cc": _recipient_list(message.get("Cc")),
                            "bcc": _recipient_list(message.get("Bcc")),
                        },
                    },
                )
                self._processed_ids.add(message_id)
                mail.store(msg_num, "+FLAGS", "\\Seen")
            return envelopes
        finally:
            with contextlib.suppress(Exception):
                mail.close()
            with contextlib.suppress(Exception):
                mail.logout()


async def send_email(
    *,
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    email_id: str | None = None,
    attachment: dict | None = None,
) -> dict:
    """Send an email directly via SMTP."""
    if not is_email_configured():
        return {"success": False, "error": "Local email provider is not configured"}

    sender = _email_address()
    password = _email_password()
    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    if bcc:
        message["Bcc"] = ", ".join(bcc)
    if email_id:
        message["In-Reply-To"] = email_id
        message["References"] = email_id
    message["Subject"] = subject
    message["Message-ID"] = make_msgid()
    message.set_content(body)

    if attachment:
        file_bytes = base64.b64decode(attachment["content_base64"].encode("ascii"))
        content_type = attachment.get("content_type", "application/octet-stream")
        maintype, _, subtype = content_type.partition("/")
        if not subtype:
            maintype = "application"
            subtype = "octet-stream"
        message.add_attachment(
            file_bytes,
            maintype=maintype,
            subtype=subtype,
            filename=attachment["filename"],
        )

    smtp_host = _env("UNITY_LOCAL_EMAIL_SMTP_HOST")
    smtp_port = _smtp_port()

    def _send() -> dict:
        recipients = [*to, *(cc or []), *(bcc or [])]
        if _use_starttls():
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(sender, password)
                smtp.send_message(message, from_addr=sender, to_addrs=recipients)
        else:
            with smtplib.SMTP_SSL(
                smtp_host,
                smtp_port,
                context=ssl.create_default_context(),
                timeout=30,
            ) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message, from_addr=sender, to_addrs=recipients)
        return {"success": True, "id": message["Message-ID"]}

    return await asyncio.to_thread(_send)
