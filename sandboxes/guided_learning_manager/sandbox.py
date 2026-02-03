"""
===================================================================
GuidedLearningManager Sandbox

Interactive sandbox for testing the guided learning keyframe capture.
This simulates a guided learning session where you demonstrate a workflow
while narrating your actions.

Quick Start:
    # Basic
    python -m sandboxes.guided_learning_manager.sandbox

    # With input listener for precise click/keyboard capture
    python -m sandboxes.guided_learning_manager.sandbox --input-listener

    # Custom region
    python -m sandboxes.guided_learning_manager.sandbox --no-fullscreen --x 100 --y 100 --width 800 --height 800

See README.md in this directory for more examples.
===================================================================
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import re
import signal
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from queue import Queue

if TYPE_CHECKING:
    pass

import mss
from dotenv import load_dotenv
from PIL import Image

# Ensure repository root is on the path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from sandboxes.utils import configure_sandbox_logging
from unity.guided_learning_manager import (
    GuidedLearningManager,
    GuidedLearningSettings,
    GuidedLearningStep,
    FrameCaptureMode,
    KeyframeSelectionMode,
)

logger = logging.getLogger("unity.guided_learning_manager.sandbox")

_stop_event = asyncio.Event()


async def _capture_frames(
    manager: GuidedLearningManager,
    monitor: dict,
    fps: int,
    session_start: float,
):
    """Background task to continuously capture and push frames."""
    logger.info(f"Starting frame capture at {fps} FPS for region: {monitor}")
    frame_count = 0

    with mss.mss() as sct:
        while not _stop_event.is_set():
            loop_start = time.time()
            try:
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                png_bytes = buffered.getvalue()
                data_url = (
                    f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
                )

                timestamp = time.time() - session_start
                await manager.push_frame(data_url, timestamp)
                frame_count += 1

                if frame_count % 50 == 0:
                    logger.debug(f"Pushed {frame_count} frames")

                sleep_time = (1 / fps) - (time.time() - loop_start)
                await asyncio.sleep(max(0.01, sleep_time))

            except mss.exception.ScreenShotError as e:
                logger.error(f"Screenshot error: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Capture error: {e}")
                break

    logger.info(f"Frame capture stopped after {frame_count} frames.")


def _transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Deepgram."""
    import os
    from deepgram import DeepgramClient

    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        logger.warning("DEEPGRAM_API_KEY not set")
        return ""

    dg = DeepgramClient(api_key=key)

    try:
        response = dg.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-2",
            smart_format=True,
            punctuate=True,
        )

        if response.results and response.results.channels:
            for channel in response.results.channels:
                if channel.alternatives:
                    return channel.alternatives[0].transcript or ""
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


async def _do_countdown(countdown: int) -> None:
    """Perform countdown with audio feedback."""
    if countdown <= 0:
        return

    import subprocess
    import platform

    system = platform.system()

    def play_beep():
        """Play a short beep sound (cross-platform)."""
        try:
            if system == "Darwin":  # macOS
                subprocess.Popen(
                    ["afplay", "-v", "0.5", "/System/Library/Sounds/Tink.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "Windows":
                import winsound

                winsound.Beep(800, 100)
            else:  # Linux
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    def say_go():
        """Say 'Go' using text-to-speech (cross-platform)."""
        try:
            if system == "Darwin":
                subprocess.Popen(
                    ["say", "-v", "Samantha", "Go"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "Windows":
                # Use Windows SAPI (async)
                subprocess.Popen(
                    [
                        "powershell",
                        "-Command",
                        "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Go')",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:  # Linux - use espeak if available
                subprocess.Popen(
                    ["espeak", "Go"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            print("\a", end="", flush=True)

    print(
        f"\n⏳ Starting in {countdown} seconds... (switch to your target window now!)",
    )
    for i in range(countdown, 0, -1):
        print(f"   {i}...", end=" ", flush=True)
        play_beep()
        await asyncio.sleep(1)
    print("GO! 🚀\n")
    say_go()


async def _run_live_mode(
    manager: GuidedLearningManager,
    session_start_holder: dict,
    args,
    start_capture_callback,
    aim=None,
):
    """
    Run live mode - continuous audio capture with automatic step detection.

    Args:
        manager: The GuidedLearningManager instance
        session_start_holder: Dict with "time" key - set when capture actually starts
        args: CLI arguments
        start_capture_callback: Callback to start frame capture (called after countdown)
    """
    import pyaudio
    import wave
    import struct

    print("\n" + "═" * 70)
    print("🎓 GUIDED LEARNING MODE")
    print("═" * 70)
    print("\nThis mode captures your demonstrations for guided learning:")
    print("  • Speak while performing actions")
    print("  • Pause for ~3s to complete a step")
    print("  • Steps are emitted automatically")
    print("  • Each step = (transcript, keyframes[])")
    print("\nPress Ctrl+C to exit\n")

    await asyncio.to_thread(input, "Press Enter when ready...")

    # Countdown timer with audio
    countdown = getattr(args, "countdown", 5)
    await _do_countdown(countdown)

    # Start frame capture after the countdown completes.
    capture_task = start_capture_callback()

    print("═" * 70)
    print("🔴 LIVE - Listening and watching...")
    print("═" * 70 + "\n")

    # Audio parameters
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK = 1024
    FORMAT = pyaudio.paInt16

    # Configurable thresholds
    SILENCE_THRESHOLD = getattr(args, "silence_threshold", 300)
    SILENCE_DURATION = getattr(args, "silence_duration", 3.0)
    MIN_SPEECH_DURATION = getattr(args, "min_speech", 1.0)

    print(f"   Audio: threshold={SILENCE_THRESHOLD}, silence={SILENCE_DURATION}s")
    print(f"   Visual: detecting keyframes automatically\n")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    sample_size = pa.get_sample_size(FORMAT)

    running = True
    step_count = 0
    all_steps: List[GuidedLearningStep] = []
    pending_step_for_teaching: GuidedLearningStep | None = None
    speech_queue: Queue = Queue()
    signal_queue: Queue = Queue()  # For speech start/end signals
    print_lock = threading.Lock()
    actor_processing = (
        threading.Event()
    )  # Set when Actor is processing, clear when done

    def safe_print(*args_print, **kwargs):
        with print_lock:
            print(*args_print, **kwargs)

    def _maybe_colorize_plan_tree(text: str) -> str:
        """Apply lightweight ANSI coloring to plan tree output (TTY only)."""

        if not getattr(sys.stdout, "isatty", lambda: False)():
            return text

        RESET = "\033[0m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        CYAN = "\033[36m"
        DIM = "\033[2m"

        out = text
        # Header + separators
        out = re.sub(
            r"^🤖 PLAN UPDATED.*$",
            lambda m: f"{CYAN}{m.group(0)}{RESET}",
            out,
            flags=re.M,
        )
        out = re.sub(
            r"^─{10,}$",
            lambda m: f"{DIM}{m.group(0)}{RESET}",
            out,
            flags=re.M,
        )

        # Status tags
        out = out.replace("[NEW]", f"{GREEN}[NEW]{RESET}")
        out = out.replace("[MODIFIED]", f"{YELLOW}[MODIFIED]{RESET}")
        out = out.replace("[REMOVED]", f"{RED}[REMOVED]{RESET}")

        # Icons (best-effort)
        out = out.replace("✨", f"{GREEN}✨{RESET}")
        out = out.replace("📝", f"{YELLOW}📝{RESET}")
        out = out.replace("❌", f"{RED}❌{RESET}")
        out = out.replace("📊", f"{DIM}📊{RESET}")
        return out

    def _maybe_colorize_unified_diff(text: str) -> str:
        """Apply lightweight ANSI coloring to unified diff output (TTY only)."""

        if not getattr(sys.stdout, "isatty", lambda: False)():
            return text

        RESET = "\033[0m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        CYAN = "\033[36m"
        DIM = "\033[2m"

        out_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("@@"):
                out_lines.append(f"{CYAN}{line}{RESET}")
            elif line.startswith("--- ") or line.startswith("+++ "):
                out_lines.append(f"{DIM}{line}{RESET}")
            elif line.startswith("+") and not line.startswith("+++"):
                out_lines.append(f"{GREEN}{line}{RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                out_lines.append(f"{RED}{line}{RESET}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    def calculate_rms(audio_data: bytes) -> float:
        count = len(audio_data) // 2
        shorts = struct.unpack(f"<{count}h", audio_data)
        return (sum(s * s for s in shorts) / count) ** 0.5

    def _merge_steps(
        a: GuidedLearningStep,
        b: GuidedLearningStep,
    ) -> GuidedLearningStep:
        """Merge two emitted steps into a single teaching step.

        This is used to smooth boundaries so visual-only / empty-transcript segments
        don't immediately trigger an Actor interjection.
        """

        merged_transcript = "\n".join(
            [t for t in [a.transcript.strip(), b.transcript.strip()] if t],
        ).strip()

        keyframes = sorted(
            [*a.keyframes, *b.keyframes],
            key=lambda kf: float(getattr(kf, "timestamp", 0.0)),
        )
        speech_segments = sorted(
            [*a.speech_segments, *b.speech_segments],
            key=lambda seg: float(getattr(seg, "start_time", 0.0)),
        )

        start_time = min(a.start_time, b.start_time)
        end_time = max(a.end_time, b.end_time)

        # Prefer earliest available context frame.
        context_frame = a.context_frame or b.context_frame

        has_visual_changes = bool(
            a.has_visual_changes or b.has_visual_changes or keyframes,
        )
        is_commentary_only = (not keyframes) and bool(speech_segments)

        return GuidedLearningStep(
            transcript=merged_transcript,
            keyframes=keyframes,
            speech_segments=speech_segments,
            start_time=start_time,
            end_time=end_time,
            has_visual_changes=has_visual_changes,
            is_commentary_only=is_commentary_only,
            context_frame=context_frame,
        )

    def audio_to_wav(frames: List[bytes]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_size)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    # Audio capture thread
    def audio_capture_thread():
        nonlocal running
        audio_frames: List[bytes] = []
        speech_start_time = None
        last_sound_time = time.time()
        is_speaking = False
        segment_counter = 0

        safe_print("🎤 Listening for speech...")

        while running and not _stop_event.is_set():
            try:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                rms = calculate_rms(audio_data)
                current_time = time.time()

                # In execute-plan mode, ignore speech while the Actor is executing.
                # This prevents new steps from being queued during plan execution.
                if args.execute_plan and actor_processing.is_set():
                    if is_speaking:
                        is_speaking = False
                        audio_frames = []
                        speech_start_time = None
                    last_sound_time = current_time
                    time.sleep(0.01)
                    continue

                if rms > SILENCE_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        speech_start_time = current_time
                        audio_frames = []
                        safe_print("   🎙️  Speaking...", end="", flush=True)
                        # Signal speech started
                        signal_queue.put(("speech_started", None))

                    audio_frames.append(audio_data)
                    last_sound_time = current_time

                    # Progress dots
                    speech_so_far = current_time - speech_start_time
                    if int(speech_so_far * 2) > int((speech_so_far - 0.05) * 2):
                        safe_print(".", end="", flush=True)

                elif is_speaking:
                    audio_frames.append(audio_data)
                    silence_elapsed = current_time - last_sound_time

                    if silence_elapsed > SILENCE_DURATION:
                        speech_duration = last_sound_time - speech_start_time
                        is_speaking = False
                        segment_counter += 1

                        safe_print(f" ({speech_duration:.1f}s)")

                        if speech_duration >= MIN_SPEECH_DURATION:
                            audio_bytes = audio_to_wav(audio_frames)
                            session_start = session_start_holder["time"]
                            speech_queue.put(
                                (
                                    audio_bytes,
                                    speech_start_time - session_start,
                                    last_sound_time - session_start,
                                ),
                            )
                            # Signal speech ended with pending transcription
                            signal_queue.put(("speech_ended", True))
                            safe_print(
                                f"   📥 Speech segment #{segment_counter} queued",
                            )
                        else:
                            # Speech ended but too short - no pending transcription
                            signal_queue.put(("speech_ended", False))
                            safe_print(f"   ⏭️  Too short ({speech_duration:.1f}s)")

                        # Only show "Listening" if Actor is not processing
                        if not actor_processing.is_set():
                            safe_print("🎤 Listening for speech...")
                        else:
                            safe_print("   ⏸️  (Actor processing, speech buffered)")
                        audio_frames = []

                time.sleep(0.01)

            except Exception as e:
                if running:
                    logger.error(f"Audio capture error: {e}")
                break

    # Step callback
    async def on_step_complete(step: GuidedLearningStep):
        nonlocal step_count
        nonlocal pending_step_for_teaching
        step_count += 1
        all_steps.append(step)

        safe_print(f"\n{'─' * 60}")
        safe_print(f"📋 STEP #{step_count} COMPLETE")
        safe_print(f"{'─' * 60}")
        safe_print(f'   📝 Transcript: "{step.transcript}"')
        safe_print(f"   🖼️  Keyframes: {step.num_keyframes}")
        safe_print(f"   ⏱️  Duration: {step.duration:.1f}s")
        safe_print(f"   📍 Time range: {step.start_time:.2f}s - {step.end_time:.2f}s")

        if step.is_commentary_only:
            safe_print(f"   ℹ️  (Commentary only - no visual changes)")

        # Show keyframe details
        if step.keyframes:
            safe_print(f"   🖼️  Keyframe details:")
            for i, kf in enumerate(step.keyframes):
                # Check if this is an LLM-selected keyframe (has reasoning in detection_reason)
                if kf.detection_reason and kf.detection_reason.startswith(
                    "llm_selected",
                ):
                    # Parse the LLM reasoning from detection_reason
                    # Format: "llm_selected [importance]: reason"
                    safe_print(f"      [{i+1}] t={kf.timestamp:.2f}s")
                    safe_print(f"          🤖 {kf.detection_reason}")
                else:
                    # Algorithmic detection - show metrics
                    metrics = []
                    if kf.ssim_score is not None:
                        metrics.append(f"SSIM={kf.ssim_score:.3f}")
                    if kf.change_ratio is not None:
                        metrics.append(f"ΔPx={kf.change_ratio:.4f}")
                    metrics_str = ", ".join(metrics) if metrics else ""
                    safe_print(f"      [{i+1}] t={kf.timestamp:.2f}s {metrics_str}")

        safe_print(f"{'─' * 60}")

        # Actor Integration: Process step via interjection
        if aim:
            teach_policy = getattr(args, "teach_policy", "auto")

            # If we buffered a previous segment, merge it into this step before teaching.
            step_to_teach = step
            if pending_step_for_teaching is not None:
                step_to_teach = _merge_steps(pending_step_for_teaching, step)
                pending_step_for_teaching = None

            # In narration mode, avoid teaching on empty transcript; buffer and merge into next narrated step.
            if teach_policy == "narration" and not step_to_teach.transcript.strip():
                pending_step_for_teaching = step_to_teach
                safe_print(
                    "\nℹ️  No narration detected for this segment. "
                    "Buffering and merging into the next narrated step (to avoid noisy interjections).",
                )
                safe_print("\n🎤 Listening for speech...")
                return

            actor_processing.set()  # Signal that Actor is processing
            if args.execute_plan:
                # Drop any queued speech/signals from the tail of the demonstration.
                while not speech_queue.empty():
                    speech_queue.get_nowait()
                while not signal_queue.empty():
                    signal_queue.get_nowait()
            safe_print("\n🤖 Teaching Actor... (please wait)")
            # Dynamic progress indicator while the Actor processes the interjection.
            spinner_frames = ["|", "/", "-", "\\"]
            spinner_stop = asyncio.Event()
            spinner_task: asyncio.Task[None] | None = None

            async def _spinner() -> None:
                start = time.time()
                i = 0
                interactive = getattr(sys.stdout, "isatty", lambda: False)()
                while not spinner_stop.is_set():
                    elapsed = time.time() - start
                    frame = spinner_frames[i % len(spinner_frames)]
                    i += 1
                    msg = f"   ⏳ Processing interjection {frame} ({elapsed:.1f}s)"
                    if interactive:
                        safe_print("\r" + msg, end="", flush=True)
                    else:
                        # Non-interactive: print occasional updates (avoid spam).
                        if i % 25 == 0:
                            safe_print(msg)
                    await asyncio.sleep(0.1)
                if interactive:
                    safe_print("\r" + (" " * 80), end="\r", flush=True)

            try:
                spinner_task = asyncio.create_task(_spinner())
                plan_state = await aim.process_step(step_to_teach)
                spinner_stop.set()
                if spinner_task:
                    with contextlib.suppress(Exception):
                        await spinner_task
                safe_print("   ✅ Interjection processed")

                # Display plan update
                if args.debug:
                    # Full plan code (debug mode)
                    full_plan = aim.get_full_plan()
                    safe_print(f"\n🤖 PLAN UPDATED (Step {step_count}):")
                    safe_print("═" * 60)
                    safe_print(full_plan)
                    safe_print("═" * 60)
                else:
                    # Tree view (default)
                    tree_view = aim.plan_formatter.format_tree_view(
                        plan_state,
                        plan_state.mode,
                    )
                    safe_print(_maybe_colorize_plan_tree(tree_view))

                    if args.show_plan_diff:
                        diff_text = (plan_state.git_diff or "").strip()
                        if diff_text:
                            diff_lines = diff_text.splitlines()
                            max_lines = max(
                                10,
                                int(getattr(args, "diff_max_lines", 220)),
                            )
                            if len(diff_lines) > max_lines:
                                shown = "\n".join(diff_lines[:max_lines])
                                shown += (
                                    f"\n... (diff truncated: {len(diff_lines)} lines total; "
                                    f"use --diff-max-lines to increase)"
                                )
                            else:
                                shown = diff_text
                            safe_print("\n🧾 PLAN DIFF (unified):")
                            safe_print("─" * 60)
                            safe_print(_maybe_colorize_unified_diff(shown))
                            safe_print("─" * 60)
                        else:
                            safe_print("\n🧾 PLAN DIFF: (no changes)")

                # Persist the latest learned plan after each successfully processed step.
                # This makes `learned_plan.py` available even if the user exits early.
                if not args.no_instrumentation and manager._instrumentation_dir:
                    try:
                        latest_plan = aim.get_full_plan()
                        plan_path = (
                            Path(manager._instrumentation_dir) / "learned_plan.py"
                        )
                        plan_path.write_text(latest_plan)
                    except Exception as e:
                        logger.warning(f"Failed to save plan during step update: {e}")

                # In execute-plan mode, wait for the Actor to finish execution and pause.
                if args.execute_plan and aim.actor_handle is not None:
                    safe_print("   ⏳ Waiting for plan execution to complete...")
                    await aim.actor_handle.awaiting_next_instruction()
                    safe_print(
                        "   ✅ Plan execution complete (paused for interjection).",
                    )

            except Exception as e:
                spinner_stop.set()
                if spinner_task:
                    with contextlib.suppress(Exception):
                        await spinner_task
                logger.error(f"Actor interjection failed: {e}")
                safe_print(f"\n⚠️  Actor failed to process Step #{step_count}: {e}")
                safe_print("    Capture continues normally.")
            finally:
                actor_processing.clear()  # Signal that Actor is done processing

        safe_print("\n🎤 Listening for speech...")

    # LLM progress callback (only used in LLM mode)
    async def on_llm_progress(status: str, data: dict):
        if status == "started":
            safe_print(
                f"\n   🤖 LLM keyframe selection started ({data.get('num_frames', '?')} frames)",
            )
            safe_print(f"   📝 Transcript: \"{data.get('transcript_preview', '')}\"")
        elif status == "prefilter_complete":
            orig = data.get("original_count", 0)
            kept = data.get("kept_count", 0)
            discarded = data.get("discarded_count", 0)
            safe_print(
                f"   🔍 Pre-filter: {orig} → {kept} frames ({discarded} near-duplicates removed)",
            )
        elif status == "calling_llm":
            safe_print(
                f"   ⏳ Calling {data.get('model', 'LLM')}... (please wait, this may take 10-30s)",
            )
        elif status == "completed":
            duration = data.get("llm_duration_sec", 0)
            num_kf = data.get("num_keyframes", 0)
            safe_print(
                f"   ✅ LLM completed in {duration:.1f}s - selected {num_kf} keyframes",
            )
            safe_print(f"   📋 Summary: {data.get('summary', 'N/A')}")
        elif status == "failed":
            safe_print(f"   ❌ LLM failed: {data.get('error', 'unknown error')}")
            safe_print(f"   ⚠️  Using fallback (first and last frame)")

    # Register callbacks
    manager.on_step_complete(on_step_complete)
    if manager.settings.selection_mode == KeyframeSelectionMode.LLM:
        manager.on_llm_progress(on_llm_progress)

    # Clarification callback for Actor integration
    async def on_clarification(question: str) -> str:
        """Prompt user for clarification answer (typed or spoken)."""
        safe_print(f"\n❓ ACTOR NEEDS CLARIFICATION:")
        safe_print(f'   "{question}"')
        safe_print()
        safe_print("   🎙️  Speak your answer or type: ", end="", flush=True)

        # Create queue for typed answer
        typed_answer_queue: asyncio.Queue = asyncio.Queue()

        # Wait for typed input in background
        async def wait_for_typed():
            answer = await asyncio.to_thread(input)
            await typed_answer_queue.put(answer)

        typed_task = asyncio.create_task(wait_for_typed())

        try:
            # Race: typed answer vs next speech segment
            while True:
                # Check for typed answer
                try:
                    answer = typed_answer_queue.get_nowait()
                    return answer
                except asyncio.QueueEmpty:
                    pass

                # Check for speech segment
                if not speech_queue.empty():
                    audio_bytes, start_time, end_time = speech_queue.get_nowait()
                    safe_print("\n   🔊 Transcribing...")
                    transcript = await asyncio.to_thread(_transcribe_audio, audio_bytes)
                    if transcript and len(transcript.strip()) >= 3:
                        safe_print(f'   ✅ "{transcript}"')
                        return transcript

                await asyncio.sleep(0.1)
        finally:
            typed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typed_task

    # Register clarification callback with Actor
    if aim:
        aim.on_clarification_callback = on_clarification

    # Start audio capture thread
    capture_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    capture_thread.start()

    try:
        # Process speech segments and signals
        while running and not _stop_event.is_set():
            try:
                # In execute-plan mode, pause processing while Actor is executing.
                if args.execute_plan and actor_processing.is_set():
                    await asyncio.sleep(0.1)
                    continue

                # Process any pending signals first
                while not signal_queue.empty():
                    signal_type, signal_data = signal_queue.get_nowait()
                    if signal_type == "speech_started":
                        await manager.signal_speech_started()
                    elif signal_type == "speech_ended":
                        pending = signal_data if signal_data is not None else False
                        await manager.signal_speech_ended(pending_transcription=pending)

                # Process speech queue
                if not speech_queue.empty():
                    audio_bytes, start_time, end_time = speech_queue.get_nowait()

                    # Transcribe
                    safe_print(f"   🔊 Transcribing... (please wait)")
                    transcript = await asyncio.to_thread(_transcribe_audio, audio_bytes)

                    if transcript and len(transcript.strip()) >= 3:
                        safe_print(f'   ✅ "{transcript}"')
                        # Push to manager (this clears pending_transcription flag)
                        # LLM progress will be shown via on_llm_progress callback
                        await manager.push_speech(transcript, start_time, end_time)
                    else:
                        safe_print(f"   ⚠️  No speech detected")
                        # Still need to clear pending flag
                        await manager.push_speech("", start_time, end_time)
                else:
                    await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                running = False
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        capture_thread.join(timeout=1.0)
        stream.stop_stream()
        stream.close()
        pa.terminate()

        # Flush any remaining step
        final_step = await manager.flush_current_step()
        if final_step and (final_step.transcript or final_step.keyframes):
            await on_step_complete(final_step)

        print("\n" + "═" * 70)
        print("✅ GUIDED LEARNING SESSION ENDED")
        print("═" * 70)
        print(f"\n📊 Summary: {step_count} step(s) captured\n")

        if all_steps:
            print("📜 ALL STEPS:")
            print("─" * 70)
            for i, step in enumerate(all_steps, 1):
                kf_info = (
                    f"{step.num_keyframes} keyframes"
                    if step.num_keyframes > 0
                    else "commentary"
                )
                print(
                    f"  [{i}] ({kf_info}) \"{step.transcript[:60]}{'...' if len(step.transcript) > 60 else ''}\"",
                )
            print("─" * 70)

        # Actor Learning Summary
        if aim:
            print("\n🤖 ACTOR LEARNING SUMMARY:")
            print("─" * 70)
            print(
                f"   ✅ Successfully processed: {aim.successful_steps}/{aim.total_steps} steps",
            )
            if aim.failed_steps > 0:
                print(f"   ❌ Failed: {aim.failed_steps} steps")
            print(
                f"   📝 Generated plan: {aim.num_functions} functions, {aim.num_lines} lines",
            )
            print(f"   ⏱️  Total learning time: {aim.total_time:.1f}s")
            print("─" * 70)

            # Display final plan
            final_plan = aim.get_full_plan()
            print("\n🤖 FINAL LEARNED PLAN:")
            print("═" * 70)
            print(final_plan)
            print("═" * 70)

            # Save plan to file
            try:
                plan_path = Path(manager._instrumentation_dir) / "learned_plan.py"
                plan_path.write_text(final_plan)
                print(f"\n💾 Plan saved to: {plan_path}")
            except Exception as e:
                logger.warning(f"Failed to save plan: {e}")

            # Execution flow (if --execute-plan)
            if args.execute_plan:
                print("\n✅ Plan learned and executed via demonstration.")
                print("   (Execution happened during the interjection phase)")

        print("═" * 70)


async def _main_async():
    import argparse

    parser = argparse.ArgumentParser(
        description="GuidedLearningManager Sandbox - Test keyframe capture with speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic:         python -m sandboxes.guided_learning_manager.sandbox
  With pynput:   python -m sandboxes.guided_learning_manager.sandbox --input-listener
  Custom region: python -m sandboxes.guided_learning_manager.sandbox --no-fullscreen --x 100 --y 100 --width 800 --height 800
        """,
    )

    # Screen region (fullscreen by default)
    region = parser.add_argument_group("Screen Region")
    region.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Disable fullscreen, use custom region instead",
    )
    region.add_argument(
        "--x",
        type=int,
        default=0,
        help="Left edge X (only with --no-fullscreen)",
    )
    region.add_argument(
        "--y",
        type=int,
        default=0,
        help="Top edge Y (only with --no-fullscreen)",
    )
    region.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width (only with --no-fullscreen)",
    )
    region.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height (only with --no-fullscreen)",
    )

    # Mode selection
    modes = parser.add_argument_group("Capture & Selection Modes")
    modes.add_argument(
        "--capture-mode",
        choices=["fps", "input_triggered", "hybrid"],
        default="fps",
        help="fps=regular interval, input_triggered=pynput only, hybrid=both (default: fps)",
    )
    modes.add_argument(
        "--selection-mode",
        choices=["direct", "algorithmic", "llm"],
        default="llm",
        help="direct=no filter, algorithmic=SSIM/MSE, llm=vision model (default: llm)",
    )
    modes.add_argument(
        "--input-listener",
        action="store_true",
        help="Enable pynput for click/keyboard capture (auto-enables hybrid mode)",
    )

    # LLM settings
    llm = parser.add_argument_group("LLM Settings (when --selection-mode llm)")
    llm.add_argument(
        "--llm-model",
        default="gemini-2.5-flash@vertex-ai",
        help="Vision model",
    )
    llm.add_argument(
        "--llm-fps",
        type=float,
        default=0.5,
        help="Frame buffer rate (default: 0.5)",
    )
    llm.add_argument(
        "--llm-resolution",
        default="768x432",
        help="Frame resolution WxH (default: 768x432)",
    )
    llm.add_argument(
        "--llm-max-frames",
        type=int,
        default=40,
        help="Max frames per segment (default: 40)",
    )
    llm.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable SSIM duplicate pre-filter (applies to LLM and DIRECT modes)",
    )
    llm.add_argument(
        "--prefilter-threshold",
        type=float,
        default=0.98,
        help="SSIM threshold for duplicate removal (default: 0.98)",
    )

    # Input listener settings
    pynput = parser.add_argument_group("Input Listener (pynput)")
    pynput.add_argument(
        "--pynput-fps",
        type=float,
        default=10.0,
        help="Frame buffer rate (default: 10.0)",
    )
    pynput.add_argument(
        "--pre-click-ms",
        type=float,
        default=100.0,
        help="Capture frame ms before click",
    )
    pynput.add_argument(
        "--post-click-ms",
        type=float,
        default=300.0,
        help="Capture frame ms after click",
    )
    pynput.add_argument(
        "--typing-interval",
        type=int,
        default=10,
        help="Capture every N chars typed",
    )
    pynput.add_argument(
        "--show-input-events",
        action="store_true",
        help="Print pynput events live",
    )

    # Audio settings
    audio = parser.add_argument_group("Audio Detection")
    audio.add_argument(
        "--silence-threshold",
        type=int,
        default=300,
        help="Audio level threshold",
    )
    audio.add_argument(
        "--silence-duration",
        type=float,
        default=3.0,
        help="Silence to end step (seconds). In INPUT_TRIGGERED mode, this is the only boundary signal.",
    )
    audio.add_argument(
        "--min-speech",
        type=float,
        default=1.0,
        help="Min speech duration to keep",
    )
    audio.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Countdown before start (0=none)",
    )

    # Algorithmic detection (when --selection-mode algorithmic)
    algo = parser.add_argument_group(
        "Algorithmic Detection (when --selection-mode algorithmic)",
    )
    algo.add_argument("--fps", type=int, default=5, help="Capture FPS (default: 5)")
    algo.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.95,
        help="SSIM threshold",
    )
    algo.add_argument("--mse-threshold", type=float, default=20.0, help="MSE threshold")
    algo.add_argument(
        "--change-ratio",
        type=float,
        default=0.01,
        help="Pixel change ratio",
    )
    algo.add_argument(
        "--hist-corr-threshold",
        type=float,
        default=0.98,
        help="Histogram correlation",
    )

    # Output & debugging
    output = parser.add_argument_group("Output & Debugging")
    output.add_argument(
        "--instrumentation-dir",
        default="captures/guided_learning/instrumentation",
        help="Directory for instrumentation output",
    )
    output.add_argument(
        "--no-instrumentation",
        action="store_true",
        help="Disable all instrumentation output",
    )
    output.add_argument(
        "--no-save-llm-frames",
        action="store_true",
        help="Don't save frames sent to LLM",
    )
    output.add_argument(
        "--no-save-discarded",
        action="store_true",
        help="Don't save prefilter-discarded frames",
    )
    output.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save rejected frame samples",
    )
    output.add_argument(
        "--sandbox-debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Actor Integration
    actor_group = parser.add_argument_group("Actor Integration")
    actor_group.add_argument(
        "--enable-actor",
        action="store_true",
        help="Enable Actor integration for learning from demonstrations",
    )
    actor_group.add_argument(
        "--execute-plan",
        action="store_true",
        help=(
            "Execute the learned plan after each demonstration step "
            "(requires agent-service). The sandbox waits until the Actor pauses "
            "before listening for the next step."
        ),
    )
    actor_group.add_argument(
        "--debug",
        action="store_true",
        help="Show full plan code instead of tree view",
    )
    actor_group.add_argument(
        "--teach-policy",
        choices=["auto", "narration"],
        default="narration",
        help=(
            "When --enable-actor is set: "
            "'auto'=teach on every emitted step; "
            "'narration'=buffer empty-transcript steps and merge into the next narrated step "
            "(smoother boundaries; avoids accidental interjections on silence)."
        ),
    )
    actor_group.add_argument(
        "--show-plan-diff",
        action="store_true",
        help="After each plan update (tree view), also print a git-style unified diff of changed functions.",
    )
    actor_group.add_argument(
        "--diff-max-lines",
        type=int,
        default=220,
        help="Max lines of diff to print per step (default: 220).",
    )
    actor_group.add_argument(
        "--course-correction",
        choices=["on", "off"],
        default="off",
        help=(
            "Enable/disable HierarchicalActor course correction (recovery sub-agent). "
            "Default: off (faster demos)."
        ),
    )
    actor_group.add_argument(
        "--headless",
        action="store_true",
        help="Run web mode in headless mode (no visible window). Default: False (web visible).",
    )

    args = parser.parse_args()

    # Setup logging
    configure_sandbox_logging(
        log_in_terminal=args.sandbox_debug,
        log_file=".logs_guided_learning_sandbox.txt",
    )

    if args.sandbox_debug:
        logging.getLogger("unity.guided_learning").setLevel(logging.DEBUG)
        logging.getLogger("unity.guided_learning_manager").setLevel(logging.DEBUG)

    # Parse LLM resolution
    llm_resolution = tuple(int(x) for x in args.llm_resolution.split("x"))

    # Create settings
    settings = GuidedLearningSettings(
        # Algorithmic detection
        ssim_threshold=args.ssim_threshold,
        mse_threshold=args.mse_threshold,
        change_ratio_threshold=args.change_ratio,
        hist_corr_threshold=args.hist_corr_threshold,
        # Instrumentation
        instrumentation_enabled=not args.no_instrumentation,
        save_keyframes=True,
        save_rejected_samples=args.save_rejected,
        instrumentation_dir=args.instrumentation_dir,
        # LLM settings
        llm_selection_model=args.llm_model,
        llm_selection_fps=args.llm_fps,
        llm_selection_resolution=llm_resolution,
        llm_selection_max_frames=args.llm_max_frames,
        save_llm_input_frames=not args.no_save_llm_frames,
        prefilter_enabled=not args.no_prefilter,
        prefilter_ssim_threshold=args.prefilter_threshold,
        save_prefilter_discarded=not args.no_save_discarded,
        # Input listener
        enable_input_listener=args.input_listener,
        verbose_input_events=args.show_input_events,
        pynput_buffer_fps=args.pynput_fps,
        pre_click_capture_ms=args.pre_click_ms,
        post_click_delay_ms=args.post_click_ms,
        typing_frame_interval_chars=args.typing_interval,
        # Activity window detection
        silence_threshold_sec=args.silence_duration,
    )

    # Determine capture and selection modes
    # --input-listener implies HYBRID mode if capture_mode is default "fps"
    if args.input_listener and args.capture_mode == "fps":
        settings.capture_mode = FrameCaptureMode.HYBRID
    else:
        settings.capture_mode = FrameCaptureMode(args.capture_mode)

    settings.selection_mode = KeyframeSelectionMode(args.selection_mode)

    # Determine effective FPS
    # When pynput is enabled (HYBRID/INPUT_TRIGGERED), use higher FPS for fresher frames
    if settings.capture_mode in (
        FrameCaptureMode.INPUT_TRIGGERED,
        FrameCaptureMode.HYBRID,
    ):
        effective_fps = args.pynput_fps
    elif settings.selection_mode == KeyframeSelectionMode.LLM:
        effective_fps = args.llm_fps
    else:
        effective_fps = args.fps

    # Display mode configuration
    print(f"\n📋 Mode Configuration:")
    print(f"   ├─ Capture:    {settings.capture_mode.value.upper()}")
    print(f"   └─ Selection:  {settings.selection_mode.value.upper()}")

    if settings.selection_mode == KeyframeSelectionMode.LLM:
        print(f"\n🤖 LLM Settings:")
        print(f"   ├─ Model:       {settings.llm_selection_model}")
        print(
            f"   ├─ Buffer FPS:  {effective_fps} {'(pynput mode - higher for fresh frames)' if settings.capture_mode in (FrameCaptureMode.INPUT_TRIGGERED, FrameCaptureMode.HYBRID) else ''}",
        )
        print(
            f"   ├─ Resolution:  {settings.llm_selection_resolution[0]}x{settings.llm_selection_resolution[1]}",
        )
        print(f"   ├─ Max frames:  {settings.llm_selection_max_frames}")
        if settings.prefilter_enabled:
            print(f"   └─ Pre-filter:  ✅ SSIM ≥ {settings.prefilter_ssim_threshold}")
        else:
            print(f"   └─ Pre-filter:  ❌ disabled")

    if settings.selection_mode == KeyframeSelectionMode.DIRECT:
        print(f"\n📋 DIRECT Mode Settings:")
        if settings.prefilter_enabled:
            print(
                f"   └─ Pre-filter:  ✅ SSIM ≥ {settings.prefilter_ssim_threshold} (removes near-duplicates)",
            )
        else:
            print(
                f"   └─ Pre-filter:  ❌ disabled (all captured frames become keyframes)",
            )

    if settings.capture_mode in (
        FrameCaptureMode.INPUT_TRIGGERED,
        FrameCaptureMode.HYBRID,
    ):
        print(f"\n🖱️ Input Listener (pynput):")
        print(f"   ├─ Pre-click:   {settings.pre_click_capture_ms}ms")
        print(f"   ├─ Post-click:  {settings.post_click_delay_ms}ms")
        print(f"   └─ Typing:      every {settings.typing_frame_interval_chars} chars")

        # Cross-platform permission warnings
        import platform

        system = platform.system()
        if system == "Darwin":  # macOS
            print(f"\n⚠️  PERMISSIONS REQUIRED (macOS):")
            print(
                f"   System Settings > Privacy & Security > Accessibility → Enable app",
            )
            print(
                f"   System Settings > Privacy & Security > Input Monitoring → Enable app",
            )
        elif system == "Windows":
            print(f"\n⚠️  PERMISSIONS NOTE (Windows):")
            print(f"   • Run as Administrator for full keyboard/mouse capture")
            print(f"   • Some apps (e.g., UAC dialogs) may block input monitoring")
        elif system == "Linux":
            print(f"\n⚠️  PERMISSIONS NOTE (Linux):")
            print(f"   • User must have access to /dev/input/* devices")
            print(f"   • Add user to 'input' group: sudo usermod -aG input $USER")
            print(f"   • For X11: xhost +local: may be needed")
            print(f"   • For Wayland: input capture may be limited")

    if settings.selection_mode == KeyframeSelectionMode.ALGORITHMIC:
        print(f"\n🔧 Algorithmic Detection Thresholds:")
        print(
            f"   ├─ SSIM:         < {settings.ssim_threshold} (lower = more sensitive)",
        )
        print(f"   ├─ MSE:          > {settings.mse_threshold}")
        print(f"   ├─ Change Ratio: > {settings.change_ratio_threshold}")
        print(f"   └─ Hist Corr:    < {settings.hist_corr_threshold}")

    if settings.instrumentation_enabled:
        print(f"\n📊 Instrumentation:")
        print(f"   ├─ Output dir: {settings.instrumentation_dir}")
        print(f"   ├─ Save keyframes: ✅")
        print(
            f"   └─ Save rejected samples: {'✅' if settings.save_rejected_samples else '❌'}",
        )

    # Actor Integration
    aim = None
    if args.enable_actor:
        from sandboxes.guided_learning_manager.actor_integration import (
            ActorIntegrationManager,
            ActorIntegrationConfig,
        )

        print(f"\n🤖 Actor Integration:")
        mode_str = "execution" if args.execute_plan else "learning"
        print(f"   ├─ Mode: {mode_str.upper()}")
        print(f"   ├─ Display: {'full code' if args.debug else 'tree view'}")
        print(f"   └─ Initializing...", end=" ", flush=True)

        config = ActorIntegrationConfig(
            enabled=True,
            execute_plan=args.execute_plan,
            debug_mode=args.debug,
            # "magnitude" for real execution (requires agent-service), "mock" for learning
            computer_mode="magnitude" if args.execute_plan else "mock",
            connect_now=args.execute_plan,
            enable_course_correction=(args.course_correction == "on"),
            headless=args.headless,
        )

        aim = ActorIntegrationManager()

        try:
            await aim.initialize(config)
            print("✅ Ready")
        except Exception as e:
            print(f"❌ Failed")
            if args.execute_plan:
                print("\n❌ Failed to initialize Actor for execution mode")
                print("ERROR: --execute-plan requires agent-service to be running")
                print("       Start agent-service and try again.")
                sys.exit(1)
            else:
                logger.error(f"Actor initialization failed: {e}")
                print(f"\n⚠️  Actor integration disabled due to error: {e}")
                aim = None

    # Create manager
    manager = GuidedLearningManager(settings=settings, debug=args.sandbox_debug)
    await manager.start()

    # Determine capture region (fullscreen by default)
    with mss.mss() as sct:
        # Get primary monitor (index 1 - index 0 is "all monitors combined")
        primary = sct.monitors[1]

        if args.no_fullscreen:
            # Custom region mode
            monitor = {
                "top": args.y,
                "left": args.x,
                "width": args.width if args.width else primary["width"],
                "height": args.height if args.height else primary["height"],
            }
            print(
                f"\n🖥️  Custom region: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})",
            )
        else:
            # Fullscreen (default)
            monitor = primary
            print(f"\n🖥️  Fullscreen: {monitor['width']}x{monitor['height']}")
    capture_task: Optional[asyncio.Task] = None
    session_start_holder = {"time": 0.0}  # Mutable holder for nonlocal access

    def start_capture():
        """Callback to start frame capture after countdown.

        Sets session_start to NOW so all frame timestamps are relative to
        when capture actually begins (after the countdown).
        """
        nonlocal capture_task
        session_start_holder["time"] = time.time()  # Reset to NOW
        capture_task = asyncio.create_task(
            _capture_frames(
                manager,
                monitor,
                effective_fps,
                session_start_holder["time"],
            ),
        )
        return capture_task

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()
    shutdown_count = [0]  # Mutable counter for nested access

    def signal_handler(sig, frame):
        """Handle interrupt signals - force exit on second attempt."""
        import os

        shutdown_count[0] += 1

        if shutdown_count[0] == 1:
            print("\n\n🛑 Shutting down... (press Ctrl+C again to force quit)")
            shutdown_event.set()
            _stop_event.set()
            # Stop pynput immediately to prevent it capturing more Ctrl+C
            if manager._input_listener:
                try:
                    manager._input_listener.stop()
                except Exception:
                    pass
        else:
            # Force exit on second Ctrl+C
            print("\n⚡ Force quitting...")
            os._exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run live mode (frame capture starts after countdown inside this function)
        # Pass session_start_holder so live_mode can access the actual start time
        await _run_live_mode(manager, session_start_holder, args, start_capture, aim)

    except KeyboardInterrupt:
        pass  # Handled by signal_handler
    except asyncio.CancelledError:
        pass
    finally:
        # Ensure stop event is set
        _stop_event.set()
        shutdown_event.set()

        # Stop manager first (this stops pynput listener)
        try:
            await asyncio.wait_for(manager.stop(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Manager stop timed out")
        except Exception as e:
            logger.warning(f"Error stopping manager: {e}")

        # Cancel and wait for capture task
        if capture_task and not capture_task.done():
            capture_task.cancel()
            try:
                await asyncio.wait_for(capture_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cleanup Actor integration
        if aim:
            try:
                await aim.cleanup()
            except Exception as e:
                logger.warning(f"Error during Actor cleanup: {e}")

        # Show where instrumentation was saved
        if not args.no_instrumentation and manager._instrumentation_dir:
            print(f"\n📁 Instrumentation saved to: {manager._instrumentation_dir}")
            print(f"   ├─ steps/           - Per-step directories:")
            print(f"   │   └─ step_NNN/")
            print(f"   │       ├─ keyframes/       - Step's detected keyframes")
            print(f"   │       └─ llm_input_frames/ - Frames sent to LLM")
            if args.save_rejected:
                print(f"   ├─ rejected_samples/ - Sample of rejected frames")
            print(f"   ├─ report.json      - Full JSON report")
            print(f"   └─ summary.txt      - Human-readable summary")
            if aim:
                print(f"   └─ learned_plan.py  - Learned Actor plan")

        print("\n✅ Done.")


def main():
    """Entry point with proper signal handling."""
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        pass  # Already handled gracefully
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
