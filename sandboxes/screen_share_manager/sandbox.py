"""
===================================================================
An interactive sandbox for the ScreenShareManager.

This sandbox demonstrates the direct-control pattern for the ScreenShareManager.
It streams a specific screen region, allows you to provide voice or text
input to simulate a user turn, and then executes the two-stage analysis
process (detect and annotate) to get back annotated ImageHandles.

Prerequisites:
- `pip install mss Pillow opencv-python aiohttp unifyai`
- Set UNIFY_KEY environment variable.
- Optional: DEEPGRAM_API_KEY and CARTESIA_API_KEY for --voice mode.
===================================================================
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import mss
from dotenv import load_dotenv
from PIL import Image

# Ensure repository root is on the path for local execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from sandboxes.utils import (
    build_cli_parser,
    configure_sandbox_logging,  # Import the logging helper
    record_until_enter,
    transcribe_deepgram,
    speak,
    _wait_for_tts_end,
)
from unity.screen_share_manager.screen_share_manager import ScreenShareManager

# Use the named logger from the manager's module
logger = logging.getLogger("unity.screen_share_manager.screen_share_manager")

_stop_event = asyncio.Event()

# Help text displayed to the user in the REPL
_COMMANDS_HELP = """
ScreenShareManager Sandbox (Direct Control Mode)
-----------------------------------------------
Type a message or use 'r' to record voice. Your utterance triggers the
two-stage analysis pipeline. Results will appear below as they are generated.

┌─────────────── Commands ───────────────┐
│ <your message>      - Send a text utterance to trigger analysis.    │
│ r                   - (Voice mode only) Record a voice utterance.   │
│ help | h            - Show this help message.                       │
│ quit | exit         - Exit the sandbox.                             │
└────────────────────────────────────────┘
"""


async def _capture_and_push_frames(
    manager: ScreenShareManager, monitor: dict, fps: int
):
    logger.info(f"Starting screen capture for monitor: {monitor}")
    logger.info("Capture will begin in 2 seconds. Please focus the target window.")
    await asyncio.sleep(2)
    start_time = time.time()
    frame_count = 0
    with mss.mss() as sct:
        while not _stop_event.is_set():
            loop_start = time.time()
            try:
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                data_url = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

                frame_timestamp = time.time() - start_time
                await manager.push_frame(data_url, frame_timestamp)
                frame_count += 1
                logger.debug(
                    f"Pushed frame #{frame_count} at timestamp {frame_timestamp:.2f}s"
                )

                sleep_time = (1 / fps) - (time.time() - loop_start)
                await asyncio.sleep(max(0.01, sleep_time))
            except mss.exception.ScreenShotError as e:
                logger.error(f"ScreenShotError: {e}. Retrying...", exc_info=True)
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Capture loop error: {e}", exc_info=True)
                break
    logger.info(f"Screen capture task stopped after pushing {frame_count} frames.")


async def _process_turn_analysis(
    screen_manager: ScreenShareManager, utterance: str, turn_counter: int, args
):
    """
    Runs the full detection and annotation pipeline for a single turn in the background.
    """
    logger.info(f"--- Turn #{turn_counter} ---")

    # --- Stage 1: Detection ---
    start_time, end_time = time.time() - 5, time.time()
    await screen_manager.push_speech(utterance, start_time, end_time)
    logger.info(f"Pushed speech event for turn #{turn_counter}. Triggering analysis.")

    analysis_task = screen_manager.analyze_turn()

    print(f"\n[Turn #{turn_counter}] 🔍 Detecting key events...")
    logger.info(f"[Turn #{turn_counter}] Awaiting detection task...")
    detection_start_time = time.time()
    detected_events = await analysis_task
    detection_duration = time.time() - detection_start_time
    logger.info(
        f"[Turn #{turn_counter}] Detection stage completed in {detection_duration:.2f} seconds."
    )

    if not detected_events:
        print(f"   -> [Turn #{turn_counter}] No significant events were detected.")
        logger.info(
            f"[Turn #{turn_counter}] No events detected. Turn processing complete."
        )
        return

    logger.info(
        f"[Turn #{turn_counter}] Detected {len(detected_events)} candidate event(s) at timestamps: {[f'{e.timestamp:.2f}s' for e in detected_events]}"
    )
    print(
        f"   -> [Turn #{turn_counter}] Detected {len(detected_events)} candidate event(s). Now generating annotations..."
    )

    # --- Stage 2: Annotation ---
    annotation_context = f"The user just said: '{utterance}'"
    logger.info(
        f"[Turn #{turn_counter}] Starting annotation with consumer context: '{annotation_context}'"
    )
    annotation_start_time = time.time()
    annotated_handles = await screen_manager.annotate_events(
        detected_events, annotation_context
    )
    annotation_duration = time.time() - annotation_start_time
    logger.info(
        f"[Turn #{turn_counter}] Annotation stage completed in {annotation_duration:.2f} seconds."
    )

    print(f"\n[Turn #{turn_counter}] ✅ Analysis Complete:")
    if not annotated_handles:
        print(f"   -> [Turn #{turn_counter}] No final annotated images were generated.")
        logger.warning(
            f"[Turn #{turn_counter}] Annotation stage finished but produced no handles."
        )
    else:
        logger.info(
            f"[Turn #{turn_counter}] Generated {len(annotated_handles)} annotated image(s)."
        )
        print(
            f"   -> [Turn #{turn_counter}] Generated {len(annotated_handles)} annotated image(s):"
        )
        for i, handle in enumerate(annotated_handles):
            logger.debug(
                f"  [Turn #{turn_counter}] Handle #{i+1}: Pending ID={handle.image_id}, Annotation='{handle.annotation}'"
            )
            print(f"      [{i+1}] Image (Pending ID: {handle.image_id})")
            print(f'          Annotation: "{handle.annotation}"')

            if args.save_images:
                img_path = (
                    Path("images") / f"turn_{turn_counter}_{int(time.time())}_{i}.png"
                )
                try:
                    raw_data = handle.raw()
                    with open(img_path, "wb") as f:
                        f.write(raw_data)
                    logger.info(
                        f"[Turn #{turn_counter}] Successfully saved image for handle {handle.image_id} to '{img_path}' ({len(raw_data)} bytes)."
                    )
                    print(f"          -> Saved to {img_path}")
                except Exception as e:
                    logger.error(
                        f"[Turn #{turn_counter}] Failed to save image for handle {handle.image_id} to '{img_path}': {e}",
                        exc_info=True,
                    )

    if args.voice:
        speak("Analysis complete.")
    logger.info(f"--- End of Turn #{turn_counter} ---\n")


async def _main_async() -> None:
    parser = build_cli_parser("Interactive ScreenShareManager Sandbox")
    parser.add_argument("--x", type=int, required=True, help="The x-coordinate.")
    parser.add_argument("--y", type=int, required=True, help="The y-coordinate.")
    parser.add_argument(
        "--width", type=int, required=True, help="Width of capture area."
    )
    parser.add_argument(
        "--height", type=int, required=True, help="Height of capture area."
    )
    parser.add_argument("--fps", type=int, default=5, help="Frames per second.")
    parser.add_argument(
        "--context",
        type=str,
        default="User is navigating a web application.",
        help="Initial session context.",
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save annotated images locally."
    )
    args = parser.parse_args()

    # --- Setup Logging ---
    configure_sandbox_logging(
        log_in_terminal=args.log_in_terminal,
        log_file=".logs_screen_share_sandbox.txt",
    )
    # Ensure the logger level is set to capture detailed messages for the file.
    logging.getLogger("unity.screen_share_manager.screen_share_manager").setLevel(
        logging.DEBUG
    )

    if args.save_images:
        Path("images").mkdir(exist_ok=True)
        logger.info("Image saving enabled.")

    # --- Setup Manager and Capture ---
    screen_manager = ScreenShareManager()
    screen_manager.set_session_context(args.context)
    await screen_manager.start()

    capture_monitor = {
        "top": args.y,
        "left": args.x,
        "width": args.width,
        "height": args.height,
    }
    capture_task = asyncio.create_task(
        _capture_and_push_frames(screen_manager, capture_monitor, args.fps)
    )

    print(_COMMANDS_HELP)

    # --- Main Loop ---
    logger.info("Sandbox REPL started. Waiting for user input...")
    turn_counter = 0
    background_tasks: List[asyncio.Task] = []

    try:
        while not _stop_event.is_set():
            utterance = ""
            if args.voice:
                _wait_for_tts_end()
                prompt = await asyncio.to_thread(input, "command ('r' to record)> ")
                if prompt.strip().lower() == "r":
                    logger.info("Recording voice input...")
                    audio = await asyncio.to_thread(record_until_enter)
                    utterance = transcribe_deepgram(audio).strip()
                    if utterance:
                        print(f"▶️  {utterance}")
                        logger.info(f"Transcribed voice input: '{utterance}'")
                else:
                    utterance = prompt.strip()
                    if utterance:
                        logger.info(f"Received text input: '{utterance}'")
            else:
                utterance = await asyncio.to_thread(input, "command> ")
                utterance = utterance.strip()
                if utterance:
                    logger.info(f"Received text input: '{utterance}'")

            if not utterance or utterance.lower() in {"quit", "exit", "help", "h"}:
                if utterance.lower() in {"quit", "exit"}:
                    logger.info(f"'{utterance}' command received. Initiating shutdown.")
                    break
                if utterance.lower() in {"help", "h"}:
                    print(_COMMANDS_HELP)
                continue

            turn_counter += 1

            # Schedule the entire turn processing to run in the background.
            task = asyncio.create_task(
                _process_turn_analysis(screen_manager, utterance, turn_counter, args)
            )
            background_tasks.append(task)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Sandbox interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(
            f"An unhandled exception occurred in the main loop: {e}", exc_info=True
        )
    finally:
        print("\nShutting down...")
        logger.info("Starting sandbox shutdown sequence.")
        _stop_event.set()
        await screen_manager.stop()
        logger.info("ScreenShareManager stopped.")

        # Cancel any pending background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()

        if capture_task and not capture_task.done():
            logger.info("Waiting for capture task to complete...")
            await capture_task
            logger.info("Capture task completed.")

        print("Shutdown complete.")
        logger.info("Sandbox shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(_main_async())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nExiting sandbox.")
