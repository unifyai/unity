import re
import aiohttp
import asyncio
from dotenv import load_dotenv
import json
import os
from datetime import datetime, UTC
from google.cloud import pubsub_v1
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from helpers import get_thread_id


# Configuration
load_dotenv()
project_id = os.getenv("PROJECT_ID", "responsive-city-458413-a2")
subscription_id = os.getenv("TOPIC_ID", "intranet-sub")
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
assistant_email = "mh-policies@unify.ai"

# Initialize Unity environment and system components
await ensure_system_initialized()


def _ensure_email_logging_context() -> str:
    """Create the `IntranetEmails` context (idempotent) and return its name.

    Mirrors the safe creation pattern used elsewhere to tolerate concurrency.
    """
    try:
        import unify  # type: ignore
    except Exception:
        return "IntranetEmails"

    active_ctx = (unify.get_active_context() or {}).get("write") or ""
    if not active_ctx:
        try:
            from unity import ensure_initialised as _ensure_initialised  # type: ignore

            _ensure_initialised()
            active_ctx = (unify.get_active_context() or {}).get("write") or ""
        except Exception:
            pass

    ctx = f"{active_ctx}/IntranetEmails" if active_ctx else "IntranetEmails"
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
                },
                context=ctx,
            )
    except Exception:
        # Tolerate races or partial creation in concurrent scenarios
        pass

    return ctx


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
            print(f"Email sent '{subject}' to {to_email}: {text}")


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
            email_log_payload = {
                "sender_email": to_email,
                "subject": subject,
                "body": question,
                "message_id": message_id,
                "timestamp": datetime.now(UTC),
            }
        except Exception:
            pass

        # acknowledge receipt
        ack_message = (
            "Thank you for your message.\n\n"
            "We’ve received your request and are now processing it. "
            "You’ll receive a detailed reply shortly. "
            "To help us respond efficiently, please wait to hear back from us "
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
            "We’re sorry, but we couldn’t complete your request due to an unexpected error.\n\n"
            "Please forward this email to hello@unify.ai with a brief description of what you "
            "were trying to do and any details that might help us investigate (for example, the "
            "original subject and the approximate time you sent your message). We’ll look into this "
            "promptly and get back to you.\n\n"
            "You can also try sending your question again or try another question to see if the issue persists.\n\n"
            "Kind regards,\n"
            "Midland Heart Policies Assistant"
        )
        error_text = ""
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
            if result.get("success"):
                answer_body = result.get("answer") or "No answer returned."
            else:
                error_text = result.get("error") or "Unknown error"
                answer_body = fallback_message
        except Exception as e:
            print(f"Error querying API: {e}", flush=True)
            error_text = str(e)
            answer_body = fallback_message

        # best-effort logging of complete email record (include actual error, not fallback)
        try:
            if email_log_payload is not None:
                email_log_payload["answer"] = answer_body
                email_log_payload["error"] = error_text or ""
                import unify  # type: ignore

                unify.log(
                    context=ctx or "IntranetEmails",
                    new=True,
                    mutable=True,
                    **email_log_payload,
                )
        except Exception:
            pass

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
        print(f"Error processing message: {e}", flush=True)
        message.nack()


def callback(message):
    print(f"Received notification: {message}")
    asyncio.run(process_message(message))


async def main():
    # Use a flow control to limit in-flight messages
    flow_control = pubsub_v1.types.FlowControl(max_messages=10)
    streaming_pull = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=flow_control,
    )
    print(f"Listening for messages on {subscription_path}...")
    try:
        streaming_pull.result()
    except KeyboardInterrupt:
        streaming_pull.cancel()


if __name__ == "__main__":
    asyncio.run(main())
