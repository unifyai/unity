"""
Starts the conversation manager.

This script can be run as a CLI with the following arguments:
    --local         Enable local GUI mode (default).
    --full          Disable local GUI mode (real comms and no GUI).
    --enabled-tools Comma-separated list of enabled tools (choices: conductor, contact, transcript, knowledge, scheduler, comms). Default: None
"""

import signal
import time
import unity.conversation_manager
from dotenv import load_dotenv

load_dotenv(override=True)


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down convo manager...")
    unity.conversation_manager.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start the conversation manager in global or local mode",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--local",
        dest="start_local",
        action="store_true",
        default=True,
        help="Enable local GUI mode",
    )
    group.add_argument(
        "--full",
        dest="start_local",
        action="store_false",
        help="Disable local GUI mode (real comms and no GUI)",
    )
    parser.add_argument(
        "--enabled-tools",
        dest="enabled_tools",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=None,
        help="Comma-separated list of enabled tools with choices of conductor, contact, transcript, knowledge, scheduler, comms. Default: None",
    )
    args = parser.parse_args()

    # Start the convo manager
    print("Starting convo manager...")
    if unity.conversation_manager.start(
        start_local=args.start_local,
        enabled_tools=(
            ",".join(args.enabled_tools)
            if isinstance(args.enabled_tools, list)
            else args.enabled_tools
        ),
    ):
        print("Convo manager started successfully...")

        from unity.helpers import run_script

        if args.start_local:
            proc = run_script("sandboxes/conversation_manager/gui.py", terminal=True)
            proc.wait()
            unity.conversation_manager.stop("signal_shutdown")

        # Keep running until the convo manager process is dead
        while unity.conversation_manager.is_running():
            time.sleep(1)  # Check every second

        # Get the final status to see why it stopped
        status = unity.conversation_manager.get_status()
        print(
            f"Convo manager has stopped. Reason: {status.get('shutdown_reason', 'unknown')}",
        )
        if "message" in status:
            print(f"Details: {status['message']}")
    else:
        print("Failed to start convo manager")
        exit(1)
