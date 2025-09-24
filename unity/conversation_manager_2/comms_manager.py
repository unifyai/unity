import time
from dotenv import load_dotenv

load_dotenv()

import asyncio
from google.cloud import pubsub_v1
import json
import os
from unity.conversation_manager_2.new_events import *
from unity.constants import ASYNCIO_DEBUG
import redis.asyncio as redis


# Subscription IDs
project_id = "responsive-city-458413-a2"
startup_subscription_id = (
    "unity-startup" + ("-staging" if os.getenv("STAGING") else "") + "-sub"
)
subscription_id = (
    "unity-"
    + (os.getenv("ASSISTANT_ID") if os.getenv("ASSISTANT_ID") else "default-assistant")
    + (
        "-staging"
        if (
            os.getenv("STAGING")
            and "default-assistant" not in os.getenv("ASSISTANT_ID", "")
        )
        else ""
    )
) + "-sub"

# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    # "whatsapp": WhatsappMessageRecievedEvent,
    "msg": SMSRecieved,
    "email": EmailRecieved,
}


class CommsManager:
    def __init__(self, event_broker):
        self.subscribers = {}
        self.call_proc = None
        self.credentials = None
        self.loop = asyncio.get_event_loop()
        self.message_queue: redis.Redis = event_broker

    def handle_message(
        self,
        message: pubsub_v1.types.PubsubMessage,
    ):
        """Handle incoming messages from PubSub subscriptions."""
        try:
            data = json.loads(message.data.decode("utf-8"))
            thread = data["thread"]
            event = data["event"]
            print(f"Received message from {thread}: {message.data.decode('utf-8')}")
            if thread == "startup":
                global subscription_id, startup_subscription_id

                # acknowledge message and cancel startup subscription
                message.ack()
                while startup_subscription_id not in self.subscribers:
                    time.sleep(0.1)
                self.subscribers[startup_subscription_id].cancel()
                self.subscribers.pop(startup_subscription_id)

                # subscribe to the assistant's subscription
                os.environ["ASSISTANT_ID"] = event["assistant_id"]
                subscription_id = (
                    "unity-"
                    + (
                        os.getenv("ASSISTANT_ID")
                        if os.getenv("ASSISTANT_ID")
                        else "default-assistant"
                    )
                    + ("-staging" if os.getenv("STAGING") else "")
                ) + "-sub"
                self.subscribe_to_topic(subscription_id)

                # publish
                task = asyncio.run_coroutine_threadsafe(
                    self.message_queue.publish(
                        "app:comms:start_up",
                        json.dumps(
                            {
                                "topic": "startup",
                                "to": "past",
                                "event": StartupEvent(
                                    api_key=event["api_key"],
                                    medium=event["medium"],
                                    assistant_id=event["assistant_id"],
                                    user_id=event["user_id"],
                                    assistant_name=event["assistant_name"],
                                    assistant_age=event["assistant_age"],
                                    assistant_region=event["assistant_region"],
                                    assistant_about=event["assistant_about"],
                                    assistant_number=event["assistant_number"],
                                    assistant_email=event["assistant_email"],
                                    user_name=event["user_name"],
                                    user_number=event["user_number"],
                                    user_whatsapp_number=event["user_whatsapp_number"],
                                    user_email=event["user_email"],
                                    voice_provider=event["voice_provider"],
                                    voice_id=event["voice_id"],
                                ).to_dict(),
                            }
                        ),
                    ),
                    self.loop,
                )
            elif thread in events_map:
                content = event["body"]
                topic = ""
                if thread == "email":
                    content = "Subject: " + event["subject"] + "\n\n" + event["body"]
                    topic = event["from"].split("<")[1][:-1]
                else:
                    topic = event["from_number"].replace("whatsapp:", "").strip()
                # Put the message in the queue instead of creating a task
                task = asyncio.run_coroutine_threadsafe(
                    self.message_queue.publish(
                        f"app:comms:{thread}_message",
                        events_map[thread](
                            content=content,
                            contact=topic,
                        ).to_json(),
                    ),
                    self.loop,
                )
                message.ack()
            elif thread == "call":
                try:
                    # Extract phone numbers from the message data
                    from_number = event.get("caller_number", "")
                    to_number = "+" + event.get("conference_name", "").replace(
                        "Unity_",
                        "",
                    )
                    task = asyncio.run_coroutine_threadsafe(
                        self.message_queue.publish(
                            "app:comms:call_initiated",
                            PhoneCallInitiated(
                                contact=event["caller_number"],
                                # voice_id=event.get("voice_id", None),
                                # voice_provider=event.get("voice_provider", None),
                            ).to_json(),
                        ),
                        self.loop,
                    )
                    # this should be handled through the comms agents i think
                    # self.call_proc = run_script(
                    #     "call.py",
                    #     "dev",
                    #     from_number,
                    #     to_number,
                    # )
                    task.result()
                    message.ack()
                except json.JSONDecodeError:
                    print("Invalid message format for call event")
                    message.nack()
                except Exception as e:
                    print(f"Error processing call event: {e}")
                    message.nack()
            else:
                print(f"Unknown event type: {thread}")
        except Exception as e:
            print(f"Error processing message: {e}")
            message.nack()

    def subscribe_to_topic(self, subscription_id: str):
        # async def subscribe_to_topic(self, subscription_id: str):
        """Subscribe to a specific PubSub topic and process messages."""
        try:
            # Let GCP libraries handle authentication automatically
            if self.credentials:
                subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
            else:
                subscriber = pubsub_v1.SubscriberClient()
            subscription_path = subscriber.subscription_path(
                project_id,
                subscription_id,
            )

            print(f"Starting subscription to {subscription_path}")

            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=self.handle_message,
            )

            # Store the future for cleanup
            self.subscribers[subscription_id] = streaming_pull_future

        except Exception as e:
            print(f"Error setting up subscription {subscription_id}: {e}")

    async def start(self):
        """Start all subscriptions and maintain connection to event manager."""
        if not os.getenv("ASSISTANT_ID"):
            # Start the startup subscription
            self.subscribe_to_topic(startup_subscription_id)
            # Start ping mechanism for idle containers
            asyncio.create_task(self.send_pings())
        else:
            # Start subscription
            self.subscribe_to_topic(subscription_id)

        # Keep the connection alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            # Cleanup subscriptions
            for future in self.subscribers.values():
                future.cancel()

    async def send_pings(self):
        """Send periodic pings to keep the event manager alive while waiting for startup."""
        print("Starting ping mechanism for idle container...")
        while True:
            try:
                # Send ping to event manager
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.publish(
                        f"app:comms:ping",
                        Ping(kind="keepalive").to_json(),
                    ),
                    self.loop,
                )

                # Wait 30 seconds before next ping (half the inactivity timeout)
                await asyncio.sleep(30)

                # Check if we've received a startup message (indicated by ASSISTANT_ID being set)
                current_assistant_id = os.getenv("ASSISTANT_ID")
                if current_assistant_id:
                    print("Startup received, stopping ping mechanism")
                    break

            except Exception as e:
                print(f"Error in ping mechanism: {e}")
                await asyncio.sleep(30)  # Continue trying


async def main():
    """Main entry point for the communication manager application."""
    manager = CommsManager()
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main(), debug=ASYNCIO_DEBUG)
