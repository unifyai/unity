import re
import sys
from pathlib import Path
import aiohttp
import asyncio
from dotenv import load_dotenv
import json
import os
import logging
from datetime import datetime, UTC
from typing import Optional, Any, Dict, Mapping
from pydantic import BaseModel
from google.cloud import pubsub_v1
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from helpers import get_thread_id

logger = logging.getLogger(__name__)

# Configuration
load_dotenv()
project_id = os.getenv("PROJECT_ID", "responsive-city-458413-a2")
subscription_id = os.getenv("TOPIC_ID", "intranet-sub")
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
assistant_email = "mh-policies@unify.ai"


def setup_project_path():
    """Add project root to Python path for imports."""

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


setup_project_path()


def _ensure_email_logging_context() -> str:
    """Create the `IntranetEmails` context (idempotent) and return its name.

    Mirrors the safe creation pattern used elsewhere to tolerate concurrency.
    """
    try:
        import unify  # type: ignore
    except Exception as e:
        logger.error(f"Error importing unify: {e}")
        return "IntranetEmails"

    # active_ctx = (unify.get_active_context() or {}).get("write") or ""
    # if not active_ctx:
    #     try:
    #         from unity import ensure_initialised as _ensure_initialised  # type: ignore

    #         _ensure_initialised()
    #         active_ctx = (unify.get_active_context() or {}).get("write") or ""
    #     except Exception as e:
    #         logger.error(f"Error ensuring initialised: {e}")

    ctx = "IntranetEmails"  # f"{active_ctx}/IntranetEmails" if active_ctx else "IntranetEmails"
    try:
        if ctx not in unify.get_contexts():
            unify.create_context(ctx)
            unify.create_fields(
                {
                    "sender_email": {"type": "str", "mutable": True},
                    "subject": {"type": "str", "mutable": True},
                    "body": {"type": "str", "mutable": True},
                    "message_id": {"type": "str", "mutable": True},
                    "timestamp": {"type": "datetime", "mutable": True},
                    "answer": {"type": "str", "mutable": True},
                    "error": {"type": "str", "mutable": True},
                    "success": {"type": "bool", "mutable": True},
                },
                context=ctx,
            )
    except Exception as e:
        logger.error(f"Error ensuring initialised: {e}")

    return ctx


class Email(BaseModel):
    sender_email: str
    subject: str
    body: str
    message_id: str
    timestamp: datetime
    answer: Optional[str] = None
    error: Optional[str] = None
    success: bool = False


def _to_mutable_payload(raw: Any) -> Dict[str, Any]:
    """Return a plain, mutable dict for safe mutation.

    - If `raw` is a Pydantic model, dump with mode="python" to preserve native types
    - If `raw` is a mapping, coerce to dict
    - Otherwise, fall back to a minimal dict
    """
    try:
        if isinstance(raw, BaseModel):
            return raw.model_dump(mode="python")
        if isinstance(raw, Mapping):
            return dict(raw)
    except Exception:
        pass
    return {}


async def send_email(to_email: str, subject: str, content: str, message_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{os.getenv('UNITY_COMMS_URL')}/email/send",
            headers=headers,
            json={
                "from": assistant_email,
                "to": to_email,
                "subject": subject,
                "body": content,
                "in_reply_to": message_id,
            },
        ) as response:
            response.raise_for_status()
            text = await response.text()
            logger.info(f"Email sent '{subject}' to {to_email}: {text}")


async def process_message(message):
    try:
        # fetch details of email
        payload = json.loads(message.data.decode("utf-8"))
        print("Payload", payload)
        email = payload["emailAddress"]
        if email != "mh-policies@unify.ai":
            print(f"Skipping message for {email}")
            message.ack()
            return

        history_id = payload["historyId"]
        gmail_creds = Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=[
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.modify",
            ],
            subject=assistant_email,
        )
        gmail_service = build("gmail", "v1", credentials=gmail_creds)
        _, message_id, last_message = get_thread_id(email, history_id, gmail_service)
        if not last_message:
            print("No last message found")
            message.ack()
            return
        print("Last Message", last_message)
        question = last_message.get("content", "Sample Message")
        subject = last_message.get("subject", "Sample Subject")
        res = re.search(r"<(.*?)>", last_message.get("sender", ""))
        if res:
            to_email = res.group(1)
        if not res:
            print("No email found in sender")
            message.ack()
            return
        print("Email Details", to_email, message_id, question)

        # prepare inbound email log payload (best-effort; will log after generating answer)
        ctx = None
        email_log_payload = None
        try:
            ctx = _ensure_email_logging_context()
            email_model = Email(
                sender_email=to_email,
                subject=subject,
                body=question,
                message_id=message_id,
                timestamp=datetime.now(UTC),
            )
            email_log_payload = _to_mutable_payload(email_model)
        except Exception as e:
            logger.error(f"Error ensuring email logging context: {e}")

        # acknowledge receipt
        ack_message = (
            "Thank you for your message. We’ve received your request and are now processing it.\n\n"
            "You’ll receive a detailed reply shortly. To help us respond efficiently, please wait to hear back from us "
            "before sending another reply in this email thread.\n\n"
            "Kind regards,\n"
            "Midland Heart Policies Assistant"
        )
        await send_email(
            to_email,
            subject,
            ack_message,
            message_id,
        )

        # call the /query endpoint
        fallback_message = (
            "We're sorry, but we couldn't complete your request due to an unexpected error.\n\n"
            "Please forward this email to hello@unify.ai with a brief description of what you "
            "were trying to do and any details that might help us investigate (for example, the "
            "original subject and the approximate time you sent your message). We'll look into this promptly and get back to you.\n\n"
            "You can also try sending your question again or try another question to see if the issue persists.\n\n"
            "Kind regards,\n"
            "Midland Heart Policies Assistant"
        )
        error_text = ""
        success_flag = False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/query",
                    json={"query": question, "retreival_mode": "llm"},
                ) as resp:
                    text = await resp.text()
                    print("Query response", text)
                    resp.raise_for_status()
                    result = await resp.json()
            success_flag = bool(result.get("success"))
            if success_flag:
                answer_body = result.get("answer") or "No answer returned."
            else:
                error_text = result.get("error") or "Unknown error"
                answer_body = fallback_message
        except Exception as e:
            logger.error(f"Error querying API: {e}")
            error_text = str(e)
            answer_body = fallback_message

        # best-effort logging of complete email record (include actual error, not fallback)
        try:
            if email_log_payload is not None:
                # Mutate dict first
                email_log_payload["answer"] = answer_body
                email_log_payload["error"] = error_text or ""
                # Use success flag from API response when available
                email_log_payload["success"] = success_flag

                # Validate back into model and convert to JSON-mode dict for logging
                final_model = Email.model_validate(email_log_payload)
                payload = final_model.model_dump(mode="json")

                import unify  # type: ignore

                unify.log(
                    context=ctx or "IntranetEmails",
                    new=True,
                    mutable=True,
                    **payload,
                )
        except Exception as e:
            logger.error(f"Error logging email: {e}")

        # send the answer back
        await send_email(
            to_email,
            subject,
            answer_body,
            message_id,
        )

        print(f"Processed message {message_id}")
        message.ack()
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()


def callback(message):
    print(f"Received notification: {message}")
    asyncio.run(process_message(message))


async def main():
    # Initialize Unity/RAG system before subscribing
    try:
        from intranet.core.api import ensure_system_initialized

        await ensure_system_initialized()
    except Exception as e:
        logger.error(f"Error during system initialization: {e}")

    # Use a flow control to limit in-flight messages
    flow_control = pubsub_v1.types.FlowControl(max_messages=10)
    streaming_pull = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=flow_control,
    )
    logger.info(f"Listening for messages on {subscription_path}...")
    try:
        streaming_pull.result()
    except KeyboardInterrupt:
        streaming_pull.cancel()


if __name__ == "__main__":
    asyncio.run(main())
