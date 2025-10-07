"""
===================================================================
An interactive sandbox for the ScreenShareManager.

This sandbox allows you to stream a specific window from your screen and provide
voice or text input to simulate a user turn. It then fetches and displays the
richly annotated transcript message created by the ScreenShareManager.

Prerequisites:
- `pip install mss redis numpy Pillow`
- Ensure all environment variables from the root README are set (OPENAI_API_KEY, etc.).

--- IMPORTANT: Window Selection on Linux ---
The `XGetImage() failed` error is almost always due to incorrect window
coordinates or running on a Wayland display server without using a
Wayland-compatible method to get coordinates. Please follow these
instructions carefully.

**For X11 Users:**
1.  Run `xwininfo -frame` in your terminal.
2.  Your cursor will change. Click on the target window you want to capture.
3.  Note the following values from the output:
    - `Absolute upper-left X:`
    - `Absolute upper-left Y:`
    - `Width:`
    - `Height:`
4.  Use these exact values for the `--x`, `--y`, `--width`, and `--height` arguments.

**For Wayland Users:**
Standard X11 tools like `xwininfo` will not work correctly.
1.  You must use a Wayland-native tool to get screen coordinates. The recommended tool is `slop`.
    - Install it using your package manager (e.g., `sudo apt install slop` or `sudo dnf install slop`).
2.  Run the following command in your terminal:
    `slop -f "%x %y %w %h"`
3.  Your cursor will change. Click and drag to select the exact region of the window you want to capture.
4.  The geometry (`X Y W H`) will be printed to your terminal. Use these values for the
    `--x`, `--y`, `--width`, and `--height` arguments.

Example Usage (after getting coordinates):
------------------------------------------
python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720 --voice
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import mss
import redis
from dotenv import load_dotenv
from PIL import Image

# Ensure repository root is on the path for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables first
load_dotenv()

from sandboxes.utils import (
    activate_project,
    build_cli_parser,
    configure_sandbox_logging,
    record_until_enter,
    transcribe_deepgram,
    speak,
    _wait_for_tts_end,
)
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from unity.transcript_manager.transcript_manager import TranscriptManager

# Logger setup for the sandbox
LG = logging.getLogger("screen_share_sandbox")

# --- Globals for thread management ---
_capture_stop_event = threading.Event()
_main_stop_event = asyncio.Event()

# Help text displayed to the user in the REPL
_COMMANDS_HELP = """
ScreenShareManager Sandbox (Async Mode)
---------------------------------------
Type a message or use 'r' to record voice. Your utterance is sent for background
processing immediately. Results will appear below as they become available.

┌─────────────── Commands ───────────────┐
│ <your message>      - Send a text utterance.                      │
│ r                   - (Voice mode only) Record a voice utterance. │
│ help | h            - Show this help message.                     │
│ quit | exit         - Exit the sandbox.                           │
└────────────────────────────────────────┘
"""


def _capture_and_publish_frames(monitor: Dict[str, int], fps: int = 5):
    """
    Runs in a separate thread to capture and publish screen frames to Redis.
    """
    LG.info(f"Starting screen capture thread for monitor: {monitor}")
    LG.info("Capture will begin in 2 seconds. Please focus the target window.")
    time.sleep(2)  # Add a delay to allow window focus

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    start_time = time.time()
    frame_count = 0
    error_count = 0

    with mss.mss() as sct:
        while not _capture_stop_event.is_set():
            loop_start = time.time()
            try:
                sct_img = sct.grab(monitor)
                error_count = 0
            except mss.exception.ScreenShotError as e:
                error_count += 1
                if error_count == 1:
                    LG.error(
                        f"ScreenShotError: {e}. This is common on Wayland or with incorrect geometry."
                    )
                    LG.error(
                        "Please verify your --x, --y, --width, --height arguments. Capture will be retried."
                    )
                time.sleep(1)
                continue

            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_b64}"

            timestamp = time.time() - start_time
            event_payload = {
                "event_name": "ScreenFrame",
                "payload": {"timestamp": timestamp, "frame_b64": data_url},
            }
            try:
                redis_client.publish(
                    "app:comms:screen_frame", json.dumps(event_payload)
                )
                frame_count += 1
            except redis.exceptions.ConnectionError as e:
                LG.error(f"Redis connection error: {e}. Is Redis running?")
                break

            time_to_sleep = (1 / fps) - (time.time() - loop_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    LG.info(f"Screen capture thread stopped. Published {frame_count} frames.")


async def _result_fetcher_and_printer(
    transcript_manager: TranscriptManager, project_name: str, voice_enabled: bool
):
    """
    A background task that continuously polls for new transcript messages and prints them.
    """
    LG.info("Result fetcher started. Polling for new transcript logs.")
    # Initialize with the ID of the latest message at startup
    initial_messages = transcript_manager._filter_messages(limit=1)
    last_printed_message_id = initial_messages[0].message_id if initial_messages else -1
    context_name = transcript_manager._transcripts_ctx

    while not _main_stop_event.is_set():
        try:
            latest_messages = transcript_manager._filter_messages(limit=1)
            if latest_messages:
                latest_message = latest_messages[0]
                if latest_message.message_id > last_printed_message_id:
                    print(
                        f"\n\n✅ Event logged to Unify in {project_name}/{context_name} in log {latest_message.message_id}\n",
                        flush=True,
                    )
                    if voice_enabled:
                        speak("Analysis complete.")
                    # Update the last printed ID and redraw the input prompt
                    last_printed_message_id = latest_message.message_id
                    # This helps redraw the prompt cleanly after printing the async result
                    sys.stdout.write("command> ")
                    sys.stdout.flush()

            await asyncio.sleep(1)  # Poll every second
        except Exception as e:
            LG.error(f"Error in result fetcher: {e}", exc_info=True)
            await asyncio.sleep(2)


async def _main_async() -> None:
    """Main asynchronous function to run the sandbox REPL."""
    parser = build_cli_parser("Interactive ScreenShareManager Sandbox")
    parser.add_argument(
        "--x",
        type=int,
        required=True,
        help="The x-coordinate of the top-left corner.",
    )
    parser.add_argument(
        "--y",
        type=int,
        required=True,
        help="The y-coordinate of the top-left corner.",
    )
    parser.add_argument(
        "--width", type=int, required=True, help="The width of the capture area."
    )
    parser.add_argument(
        "--height", type=int, required=True, help="The height of the capture area."
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Frames per second for screen capture."
    )
    args = parser.parse_args()
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_screen_share_sandbox.txt",
    )
    LG.setLevel(logging.INFO)

    capture_monitor = {
        "top": args.y,
        "left": args.x,
        "width": args.width,
        "height": args.height,
    }

    screen_manager = None
    capture_thread = None
    redis_client = None
    manager_task = None
    result_fetcher_task = None

    session_start_time = time.time()

    try:
        screen_manager = ScreenShareManager()
        transcript_manager = TranscriptManager()
        redis_client = redis.asyncio.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=False,
        )

        manager_task = asyncio.create_task(screen_manager.start())
        LG.info("ScreenShareManager listener started.")

        capture_thread = threading.Thread(
            target=_capture_and_publish_frames,
            args=(capture_monitor, args.fps),
            daemon=True,
        )
        capture_thread.start()

        # Start the background task for fetching and printing results
        result_fetcher_task = asyncio.create_task(
            _result_fetcher_and_printer(
                transcript_manager, args.project_name, args.voice
            )
        )

        await asyncio.sleep(2)
        print(_COMMANDS_HELP)

        while not _main_stop_event.is_set():
            try:
                utterance = ""
                turn_start_time = 0.0
                turn_end_time = 0.0

                # Use asyncio.to_thread to run the blocking input() in a separate thread
                if args.voice:
                    _wait_for_tts_end()
                    prompt = await asyncio.to_thread(input, "command ('r' to record)> ")
                    prompt = prompt.strip()
                    if prompt.lower() == "r":
                        # Voice recording is also blocking, so run it in a thread
                        turn_start_time = time.time()
                        audio = await asyncio.to_thread(record_until_enter)
                        utterance = transcribe_deepgram(audio).strip()
                        turn_end_time = time.time()
                        if not utterance:
                            continue
                        print(f"▶️  {utterance}")
                    else:
                        turn_start_time = time.time()
                        utterance = prompt
                        turn_end_time = time.time()
                else:
                    turn_start_time = time.time()
                    utterance = await asyncio.to_thread(input, "command> ")
                    utterance = utterance.strip()
                    turn_end_time = time.time()

                if not utterance:
                    continue

                if utterance.lower() in {"quit", "exit"}:
                    break
                elif utterance.lower() in {"help", "h", "?"}:
                    print(_COMMANDS_HELP)
                    continue

                # --- Publish Utterance for Background Processing ---
                # FIX: Correctly calculate relative start and end times for the turn
                relative_start_time = turn_start_time - session_start_time
                relative_end_time = turn_end_time - session_start_time

                event_payload = {
                    "event_name": "PhoneUtterance",
                    "payload": {
                        "contact_details": {"contact_id": 1},
                        "timestamp": datetime.now().isoformat(),
                        "content": utterance,
                        "start_time": relative_start_time,
                        "end_time": relative_end_time,
                    },
                }
                await redis_client.publish(
                    "app:comms:phone_utterance", json.dumps(event_payload)
                )
                LG.info(f"Published utterance event for: '{utterance}'")

            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            except Exception as e:
                LG.error("An error occurred in the main loop: %s", e, exc_info=True)
                print(f"❌ An unexpected error occurred: {e}")

    finally:
        print("Shutting down...")
        _main_stop_event.set()

        if screen_manager:
            screen_manager.stop()
            if manager_task and not manager_task.done():
                await asyncio.sleep(0.5)
                manager_task.cancel()

        if result_fetcher_task and not result_fetcher_task.done():
            result_fetcher_task.cancel()

        if capture_thread:
            _capture_stop_event.set()
            capture_thread.join(timeout=2)

        if redis_client:
            await redis_client.close()

        print("Shutdown complete.")


def main() -> None:
    """Synchronous entry point for the sandbox."""
    try:
        asyncio.run(_main_async())
    except (Exception, KeyboardInterrupt) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"A critical error forced the sandbox to exit: {e}")
            LG.critical(
                "Sandbox forced to exit due to unhandled exception in main.",
                exc_info=True,
            )


if __name__ == "__main__":
    main()
