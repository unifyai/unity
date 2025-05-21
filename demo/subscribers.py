import asyncio
from datetime import datetime
from google.cloud import pubsub_v1
from google.oauth2.service_account import Credentials
import json
import os
import threading
import queue
import time
from demo.events import Event, SMSMessageRecievedEvent, WhatsappMessageRecievedEvent

# Subscription IDs
project_id = "responsive-city-458413-a2"
call_subscription_id = "call-sub"
email_subscription_id = "email-sub"
msg_subscription_id = "msg-sub"
whatsapp_subscription_id = "whatsapp-sub"

# Message queue for thread-safe communication
message_queue = queue.Queue()

# Map subscription IDs to their corresponding event types
events_map: dict[str, Event] = {
    "whatsapp-sub": WhatsappMessageRecievedEvent,
    "msg-sub": SMSMessageRecievedEvent,
}

# Global connection to event manager
reader, writer = asyncio.open_connection("127.0.0.1", 8888)


async def publish_event(ev: dict) -> None:
    """
    Publish an event to the event manager.

    Args:
        ev (dict): The event dictionary to publish.

    The event is serialized to JSON and sent to the event manager
    running on localhost:8888.
    """
    ev = json.dumps(ev) + "\n"
    writer.write(ev.encode())
    await writer.drain()


def callback(message: pubsub_v1.types.PubsubMessage, subscription_id: str):
    """
    Handle incoming messages from PubSub subscriptions.

    Args:
        message (pubsub_v1.types.PubsubMessage): The incoming PubSub message.
        subscription_id (str): The ID of the subscription that received the message.

    This function:
    1. Creates an appropriate event based on the subscription type
    2. Publishes the event to the event manager
    3. Acknowledges the message to PubSub
    """
    try:
        if subscription_id in events_map:
            asyncio.create_task(
                publish_event(
                    {
                        "type": "user_agent_event",
                        "to": "pending",
                        "event": events_map[subscription_id](
                            content=message.data.decode("utf-8"),
                            timestamp=datetime.now(),
                            role="User",
                        ).to_dict(),
                    }
                )
            )
            message.ack()
        else:
            print(f"Unknown event type: {subscription_id}")
    except Exception as e:
        print(f"Error processing message: {e}")
        message.nack()


def subscribe_to_topic(subscription_id: str):
    """
    Subscribe to a specific PubSub topic and process messages.

    Args:
        subscription_id (str): The ID of the subscription to connect to.

    This function:
    1. Creates a new subscriber client
    2. Sets up the subscription
    3. Processes incoming messages using the callback function
    4. Handles errors and cleanup
    """
    try:
        creds_json = json.loads(os.getenv("GCP_SA_KEY"))
        creds = Credentials.from_service_account_info(creds_json)
        subscriber = pubsub_v1.SubscriberClient(credentials=creds)

        subscription_path = subscriber.subscription_path(project_id, subscription_id)

        print(f"Starting subscription to {subscription_id}")
        streaming_pull_future = subscriber.subscribe(
            subscription_path,
            callback=lambda message: callback(message, subscription_id),
        )

        # Keep the subscription running
        try:
            streaming_pull_future.result()
        except Exception as e:
            streaming_pull_future.cancel()
            print(f"Subscription {subscription_id} failed: {e}")

    except Exception as e:
        print(f"Error setting up subscription {subscription_id}: {e}")


def process_messages():
    """
    Process messages from the queue.

    This function runs in a separate thread and continuously:
    1. Retrieves messages from the queue
    2. Processes them according to their subscription type
    3. Handles any errors that occur during processing
    """
    while True:
        try:
            message = message_queue.get()
            print(f"Received message from {message['subscription']}: {message['data']}")
            # Add your message processing logic here
        except Exception as e:
            print(f"Error processing message from queue: {e}")
        time.sleep(0.1)


def main():
    """
    Main entry point for the subscriber application.

    This function:
    1. Creates separate threads for each subscription
    2. Starts a message processing thread
    3. Keeps the main thread alive until interrupted
    4. Handles graceful shutdown on keyboard interrupt
    """
    # Create threads for each subscription
    subscriptions = [
        call_subscription_id,
        email_subscription_id,
        msg_subscription_id,
        whatsapp_subscription_id,
    ]

    threads = []

    # Start subscription threads
    for sub_id in subscriptions:
        thread = threading.Thread(
            target=subscribe_to_topic, args=(sub_id,), daemon=True
        )
        threads.append(thread)
        thread.start()

    # Start message processing thread
    processor_thread = threading.Thread(target=process_messages, daemon=True)
    processor_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
