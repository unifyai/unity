import re
import aiohttp
import asyncio
import base64
from dotenv import load_dotenv
import json
import os
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

        # acknowledge receipt
        await send_email(
            to_email,
            subject,
            "We have received your request and are processing it.",
            message_id,
        )

        # call the /query endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/query", json={"query": question}
            ) as resp:
                text = await resp.text()
                print("Query response", text)
                resp.raise_for_status()
                result = await resp.json()
        answer = result.get("answer", "No answer returned.")

        # send the answer back
        await send_email(
            to_email,
            subject,
            answer,
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
