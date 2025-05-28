import asyncio
from datetime import datetime
from google.cloud import pubsub_v1
import json
import os
from events import *


# Subscription IDs
project_id = "responsive-city-458413-a2"
call_subscription_id = "call-sub"
email_subscription_id = "email-sub"
msg_subscription_id = "msg-sub"
whatsapp_subscription_id = "whatsapp-sub"

# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    "whatsapp-sub": WhatsappMessageRecievedEvent,
    "msg-sub": SMSMessageRecievedEvent,
}

class CommsManager:
    def __init__(self, events_queue):
        self.subscribers = {}
        self.call_proc = None
        self.credentials = None
        self.loop = asyncio.get_event_loop()
        self.message_queue = events_queue


    def handle_message(
        self,
        message: pubsub_v1.types.PubsubMessage,
        subscription_id: str,
    ):
        """Handle incoming messages from PubSub subscriptions."""
        try:
            print(
                "Received message from subscription: ",
                subscription_id,
                message.data.decode("utf-8"),
            )
            if subscription_id in events_map:
                # Put the message in the queue instead of creating a task
                self.loop.call_soon_threadsafe(
                    self.message_queue.put_nowait,
                    {
                        "topic": json.loads(message.data.decode("utf-8"))["from_number"].replace("whatsapp:", "").strip(),
                        "event": events_map[subscription_id](
                            content=json.loads(message.data.decode("utf-8"))["body"],
                            timestamp=datetime.now(),
                            role="User",
                        ).to_dict(),
                    },
                )
                message.ack()
            elif subscription_id == "call-sub":
                try:
                    # Extract phone numbers from the message data
                    message_data = json.loads(message.data.decode("utf-8"))
                    from_number = message_data.get("caller_number", "")
                    to_number = "+" + message_data.get("conference_name", "").replace(
                        "Unity_",
                        "",
                    )

                    self.loop.call_soon_threadsafe(
                    self.message_queue.put_nowait,
                    {
                        "topic": message_data["caller_number"],
                        "event": PhoneCallInitiatedEvent().to_dict(),
                    },
                )

                    # this should be handled through the comms agents i think
                    # self.call_proc = run_as_subprocess(
                    #     "call.py",
                    #     "dev",
                    #     from_number,
                    #     to_number,
                    # )
                    message.ack()
                except json.JSONDecodeError:
                    print("Invalid message format for call event")
                    message.nack()
                except Exception as e:
                    print(f"Error processing call event: {e}")
                    message.nack()
            else:
                print(f"Unknown event type: {subscription_id}")
        except Exception as e:
            print(f"Error processing message: {e}")
            message.nack()


    async def subscribe_to_topic(self, subscription_id: str):
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

            print(f"Starting subscription to {subscription_id}")

            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=lambda message: self.handle_message(message, subscription_id),
            )

            # Store the future for cleanup
            self.subscribers[subscription_id] = streaming_pull_future

        except Exception as e:
            print(f"Error setting up subscription {subscription_id}: {e}")

    async def start(self):
        """Start all subscriptions and maintain connection to event manager."""
        # Start all subscriptions
        subscriptions = [
            call_subscription_id,
            msg_subscription_id,
            whatsapp_subscription_id,
        ]

        for sub_id in subscriptions:
            await self.subscribe_to_topic(sub_id)

        # Keep the connection alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            # Cleanup subscriptions
            for future in self.subscribers.values():
                future.cancel()

async def main():
    """Main entry point for the communication manager application."""
    manager = CommsManager()
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
