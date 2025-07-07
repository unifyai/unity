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


import os
import json
from google.cloud import pubsub_v1
from google.auth import default


def handle_message(message):
    print(f"Received message: {message}")


def debug_credentials():
    """Debug function to check what credentials are being used"""
    print("🔍 DEBUGGING CREDENTIALS:")
    print(
        f"   GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}"
    )

    # Check if the credentials file exists
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        print(f"   ✅ Credentials file exists: {creds_path}")
        try:
            with open(creds_path, "r") as f:
                creds_data = json.load(f)
            print(
                f"   📧 Service Account Email: {creds_data.get('client_email', 'NOT FOUND')}"
            )
            print(f"   🆔 Project ID: {creds_data.get('project_id', 'NOT FOUND')}")
            print(f"   🔑 Token URI: {creds_data.get('token_uri', 'NOT FOUND')}")
        except Exception as e:
            print(f"   ❌ Error reading credentials file: {e}")
    else:
        print(f"   ❌ Credentials file not found or not set")

    # Try to get default credentials
    try:
        credentials, project = default()
        print(f"   🔐 Default credentials type: {type(credentials).__name__}")
        print(f"   📋 Default project: {project}")

        # Try to get service account info from credentials
        if hasattr(credentials, "service_account_email"):
            print(
                f"   📧 Service Account from credentials: {credentials.service_account_email}"
            )
        else:
            print(f"   📧 Service Account from credentials: Not available")

    except Exception as e:
        print(f"   ❌ Error getting default credentials: {e}")

    print()


if __name__ == "__main__":
    project_id = "responsive-city-458413-a2"
    subscription_id = "test-topic-sub"

    print(f"project_id: {project_id}")
    print(f"subscription_id: {subscription_id}")

    # Debug credentials before trying to connect
    debug_credentials()

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
