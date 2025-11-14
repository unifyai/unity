# Starts the convo manager
import signal
import time

import unity.conversation_manager as cm


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down convo manager...")
    cm.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting convo manager...")

    # Start the convo manager
    if cm.start():
        print("Convo manager started successfully...")

        # Keep running until the convo manager process is dead
        while cm.is_running():
            time.sleep(1)  # Check every second

        # Get the final status to see why it stopped
        status = cm.get_status()
        print(
            f"Convo manager has stopped. Reason: {status.get('shutdown_reason', 'unknown')}",
        )
        if "message" in status:
            print(f"Details: {status['message']}")
    else:
        print("Failed to start convo manager")
        exit(1)
