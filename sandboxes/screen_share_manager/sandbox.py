"""
===================================================================
An interactive sandbox for the ScreenShareManager.

This sandbox allows you to stream a specific window from your screen and provide
voice or text input to simulate a user turn. It then listens for and displays
the annotated image analysis result published by the ScreenShareManager.

Prerequisites:
- `pip install mss redis numpy Pillow opencv-python`

Example Usage (after getting coordinates):
------------------------------------------
python -m sandboxes.screen_share_manager.sandbox \
    --x 100 \
    --y 150 \
    --width 1280 \
    --height 720 \
    --voice \
    --save-images \
    --context "The user is Jane Doe, an administrator trying to reset a client's password."
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
import redis.asyncio as redis
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
processing immediately. Analysis results will appear below as they are published.

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


async def _result_listener_and_printer(voice_enabled: bool, save_images: bool):
    """
    A background task that listens for screen analysis results and prints them.
    If save_images is True, it also saves the raw images to a local folder.
    """
    LG.info("Result listener started. Subscribing to analysis results.")
    redis_client = redis.asyncio.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    async with redis_client.pubsub() as pubsub:
        await pubsub.subscribe("app:comms:screen_analysis_result")
        while not _main_stop_event.is_set():
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message:
                    data = json.loads(message["data"])
                    payload = data.get("payload", {})
                    images = payload.get("images", {}).get("root", [])
                    image_data_map = payload.get("image_data_map", {})

                    print("\n\n✅ Screen Analysis Result Received:", flush=True)
                    if not images:
                        print("   -> No annotated images were generated for this turn.")
                    else:
                        print(f"   -> Found {len(images)} annotated image(s):")
                        for i, ref in enumerate(images):
                            img_id = ref.get("raw_image_ref", {}).get("image_id", "N/A")
                            annotation = ref.get("annotation", "No annotation.")
                            print(f"      [{i+1}] Image ID: {img_id}")
                            print(f'          Annotation: "{annotation}"')

                    if save_images and image_data_map:
                        print("   -> Saving images locally...")
                        for image_id, data_url in image_data_map.items():
                            try:
                                img_data = base64.b64decode(data_url.split(",")[1])
                                img_path = Path("images") / f"{image_id}.png"
                                with open(img_path, "wb") as f:
                                    f.write(img_data)
                                print(f"      -> Saved {img_path}")
                            except Exception as e:
                                LG.error(f"Failed to save image {image_id}: {e}")

                    if voice_enabled:
                        speak("Analysis complete.")

                    # Redraw the input prompt cleanly
                    sys.stdout.write("\ncommand> ")
                    sys.stdout.flush()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                LG.error(f"Error in result listener: {e}", exc_info=True)
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
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional pre-conversation context about the user or their goal.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated images locally to an 'images' folder.",
    )
    args = parser.parse_args()
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    if args.save_images:
        Path("images").mkdir(exist_ok=True)
        LG.info(
            "Image saving enabled. Images will be stored in the 'images/' directory."
        )

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
    result_listener_task = None

    session_start_time = time.time()

    try:
        screen_manager = ScreenShareManager(
            include_raw_image_data_in_result=args.save_images
        )
        if args.context:
            screen_manager.set_session_context(args.context)
            LG.info(f"Initial session context set from CLI: '{args.context}'")

        redis_client = redis.asyncio.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=False,  # Use bytes for publishing
        )

        manager_task = asyncio.create_task(screen_manager.start())
        LG.info("ScreenShareManager listener started.")

        capture_thread = threading.Thread(
            target=_capture_and_publish_frames,
            args=(capture_monitor, args.fps),
            daemon=True,
        )
        capture_thread.start()

        # Start the background task for listening to and printing results
        result_listener_task = asyncio.create_task(
            _result_listener_and_printer(args.voice, args.save_images)
        )

        await asyncio.sleep(2)
        print(_COMMANDS_HELP)

        while not _main_stop_event.is_set():
            try:
                utterance = ""
                turn_start_time = 0.0
                turn_end_time = 0.0

                # Use asyncio.to_thread to run the blocking input()
                if args.voice:
                    _wait_for_tts_end()
                    prompt = await asyncio.to_thread(input, "command ('r' to record)> ")
                    prompt = prompt.strip()
                    if prompt.lower() == "r":
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

        if result_listener_task and not result_listener_task.done():
            result_listener_task.cancel()

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
