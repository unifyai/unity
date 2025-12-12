# Starts the convo manager
import signal
import time

import unity.conversation_manager as cm

# Flag to prevent double-handling of signals
shutting_down = False


# Graceful shutdown handler
def signal_handler(signum, frame):
    global shutting_down
    if shutting_down:
        return  # Already handling shutdown
    shutting_down = True

    # Shut down the process and get the exit code
    print("Shutting down convo manager...")
    exit_code = cm.stop("signal_shutdown")
    print(f"Convo manager exited with code {exit_code}")
    print(f"Checks: {type(exit_code)} {not exit_code} {exit_code != 0}")

    # Check if the convo manager exited with a non-zero exit code
    # This happens when: external signal + idle container or process crashed
    if not exit_code and exit_code != 0:
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
