# # Starts the Unity service
# import signal
# import time
# import unity.service


# # Graceful shutdown handler
# def signal_handler(signum, frame):
#     print("Shutting down Unity service...")
#     unity.service.stop("signal_shutdown")
#     exit(0)


# signal.signal(signal.SIGTERM, signal_handler)
# signal.signal(signal.SIGINT, signal_handler)

# if __name__ == "__main__":
#     print("Starting Unity service...")

#     # Start the Unity service
#     if unity.service.start():
#         print("Unity service started successfully...")

#         # Keep running until the Unity service process is dead
#         while unity.service.is_running():
#             time.sleep(1)  # Check every second

#         # Get the final status to see why it stopped
#         status = unity.service.get_status()
#         print(
#             f"Unity service has stopped. Reason: {status.get('shutdown_reason', 'unknown')}"
#         )
#         if "message" in status:
#             print(f"Details: {status['message']}")
#     else:
#         print("Failed to start Unity service")
#         exit(1)


from google.cloud import pubsub_v1


def handle_message(message):
    print(f"Received message: {message}")


if __name__ == "__main__":
    project_id = "responsive-city-458413-a2"
    subscription_id = "test-topic-sub"

    print(f"project_id: {project_id}")
    print(f"subscription_id: {subscription_id}")

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=handle_message,
    )
    print(f"Listening for messages on {subscription_path}...")
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
