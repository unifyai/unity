# Starts the Unity service
import signal
import time
import unity.service


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity.service.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting Unity service...")

    # Start the Unity service
    if unity.service.start():
        print("Unity service started successfully...")

        # Keep running until the Unity service process is dead
        while unity.service.is_running():
            time.sleep(1)  # Check every second

        # Get the final status to see why it stopped
        status = unity.service.get_status()
        print(
            f"Unity service has stopped. Reason: {status.get('shutdown_reason', 'unknown')}",
        )
        if "message" in status:
            print(f"Details: {status['message']}")
    else:
        print("Failed to start Unity service")
        exit(1)
