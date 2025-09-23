# Starts the convo manager
import signal
import time
# import unity.conversation_manager as conversation_manager
import unity.conversation_manager_2 as conversation_manager


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down convo manager...")
    conversation_manager.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting convo manager...")

    # Start the convo manager
    if conversation_manager.start():
        print("Convo manager started successfully...")

        # Keep running until the convo manager process is dead
        while conversation_manager.is_running():
            time.sleep(1)  # Check every second

        # Get the final status to see why it stopped
        status = conversation_manager.get_status()
        print(
            f"Convo manager has stopped. Reason: {status.get('shutdown_reason', 'unknown')}",
        )
        if "message" in status:
            print(f"Details: {status['message']}")
    else:
        print("Failed to start convo manager")
        exit(1)
