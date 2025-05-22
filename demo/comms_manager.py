import asyncio
from datetime import datetime
from google.cloud import pubsub_v1
from google.oauth2.service_account import Credentials
import json
import os
from actions import handle_message_action
from events import Event, SMSMessageRecievedEvent, WhatsappMessageRecievedEvent
from new_terminal_helper import run_in_new_terminal

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
    def __init__(self):
        self.reader = None
        self.writer = None
        self.subscribers = {}
        self.credentials = None
        self.loop = asyncio.get_event_loop()
        self.message_queue = asyncio.Queue()

    async def connect_to_event_manager(self):
        """Connect to the event manager server."""
        self.reader, self.writer = await asyncio.open_connection("127.0.0.1", 8080)
        print("Connected to event manager")

    async def publish_event(self, ev: dict) -> None:
        """Publish an event to the event manager."""
        ev = json.dumps(ev) + "\n"
        self.writer.write(ev.encode())
        await self.writer.drain()

    async def handle_event_manager_events(self):
        """Handle events coming from the event manager."""
        while True:
            try:
                raw = await self.reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                if msg["type"] == "update_gui":
                    print(
                        f"Received GUI update for thread {msg['thread']}: {msg['content']}"
                    )

                    # Handle WhatsApp send events
                    if msg["thread"] in ["whatsapp", "sms"]:
                        # Extract phone numbers from the message content
                        # This assumes the message content contains the necessary information
                        # You might need to adjust this based on your actual message format
                        try:
                            message_data = json.loads(msg["content"])
                            success = await handle_message_action(
                                msg["thread"],
                                from_number=message_data.get("to_number", "").replace(
                                    "whatsapp:", ""
                                ),
                                to_number=message_data.get("from_number", "").replace(
                                    "whatsapp:", ""
                                ),
                                message=message_data.get("message", ""),
                            )
                            if not success:
                                print(f"Failed to send {msg['thread']} message")
                        except json.JSONDecodeError:
                            print(f"Invalid message format for {msg['thread']} send")
                        except Exception as e:
                            print(f"Error processing {msg['thread']} send event: {e}")

            except Exception as e:
                print(f"Error handling event manager event: {e}")
                if self.writer:
                    self.writer.close()
                    await self.writer.wait_closed()
                break

    def handle_message(
        self, message: pubsub_v1.types.PubsubMessage, subscription_id: str
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
                        "message": message,
                        "subscription_id": subscription_id,
                        "event": events_map[subscription_id](
                            content=message.data.decode("utf-8"),
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
                        "Unity_", ""
                    )

                    self.call_proc = run_in_new_terminal(
                        "call.py",
                        "dev",  # "console" if a local call is needed
                        from_number,
                        to_number,
                    )
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

    async def process_messages(self):
        """Process messages from the queue."""
        while True:
            try:
                message = await self.message_queue.get()
                data = message["message"].data.decode("utf-8")
                notification = json.loads(data)
                print("Processing message: ", notification)
                await self.publish_event(
                    {
                        "type": "user_agent_event",
                        "to": "pending",
                        "event": message["event"],
                    }
                )
            except Exception as e:
                print(f"Error processing message from queue: {e}")

    async def subscribe_to_topic(self, subscription_id: str):
        """Subscribe to a specific PubSub topic and process messages."""
        try:
            if not self.credentials:
                creds_json = json.loads(os.getenv("GCP_SA_KEY"))
                self.credentials = Credentials.from_service_account_info(creds_json)

            subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
            subscription_path = subscriber.subscription_path(
                project_id, subscription_id
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
        await self.connect_to_event_manager()

        # Start message processor and event manager event handler
        processor_task = asyncio.create_task(self.process_messages())
        event_handler_task = asyncio.create_task(self.handle_event_manager_events())

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
            processor_task.cancel()
            event_handler_task.cancel()
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()


async def main():
    """Main entry point for the communication manager application."""
    manager = CommsManager()
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
