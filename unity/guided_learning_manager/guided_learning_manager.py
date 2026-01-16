"""
GuidedLearningManager: Captures keyframes aligned to user speech for guided learning.

This manager is optimized for the guided learning use case where:
1. A user demonstrates a workflow while narrating their actions
2. We need to capture keyframes that correspond to their speech
3. The output is a lossless (transcript, images[]) tuple

Key design principles:
- Activity window based (not turn-based)
- Speech timing anchors the capture
- All visually significant frames are captured (not just before/after)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Callable, Awaitable

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from skimage.metrics import structural_similarity as ssim

from unity.manager_registry import ManagerRegistry
from unity.common.llm_client import new_llm_client

from .types import GuidedLearningStep, SpeechSegment, KeyframeEvent
from .input_listener import (
    InputEventListener,
    InputListenerSettings,
    InputEvent,
    InputEventType,
    Point,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Structured Output Models
# ─────────────────────────────────────────────────────────────────────────────


class _StrictBaseModel(BaseModel):
    """BaseModel configured for OpenAI Structured Outputs (strict JSON schema)."""

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"additionalProperties": False},
    }


class SelectedKeyframe(_StrictBaseModel):
    """A single keyframe selected by the LLM."""

    frame_index: int = Field(
        ...,
        description="The 0-indexed position of the selected frame in the sequence.",
    )
    reason: str = Field(
        ...,
        description="A concise explanation of why this frame is semantically important (e.g., 'User clicks on terminal icon', 'Command output appears', 'File contents visible').",
    )
    importance: str = Field(
        ...,
        description="Importance level: 'critical' (key action/result), 'important' (supporting context), or 'supplementary' (nice to have).",
    )


class KeyframeSelectionResult(_StrictBaseModel):
    """Structured output for LLM keyframe selection."""

    selected_keyframes: List[SelectedKeyframe] = Field(
        ...,
        description="List of selected keyframes with their indices and reasons.",
    )
    summary: str = Field(
        ...,
        description="A brief summary of what the user demonstrated in this segment.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation / Observability
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameComparisonResult:
    """Detailed result of comparing two frames."""

    timestamp: float
    is_keyframe: bool

    # Metrics
    mse: float
    ssim_score: float
    change_ratio: float
    hist_corr: float

    # Thresholds used (for reference)
    mse_threshold: float
    ssim_threshold: float
    change_ratio_threshold: float
    hist_corr_threshold: float

    # Which checks passed/failed
    passed_mse: bool
    passed_ssim: bool
    passed_change_ratio: bool
    passed_hist_corr: bool

    # Rejection reason (if not a keyframe)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "is_keyframe": self.is_keyframe,
            "metrics": {
                "mse": round(self.mse, 2),
                "ssim": round(self.ssim_score, 4),
                "change_ratio": round(self.change_ratio, 4),
                "hist_corr": round(self.hist_corr, 4),
            },
            "thresholds": {
                "mse": self.mse_threshold,
                "ssim": self.ssim_threshold,
                "change_ratio": self.change_ratio_threshold,
                "hist_corr": self.hist_corr_threshold,
            },
            "checks": {
                "mse": "PASS" if self.passed_mse else "FAIL",
                "ssim": "PASS" if self.passed_ssim else "FAIL",
                "change_ratio": "PASS" if self.passed_change_ratio else "FAIL",
                "hist_corr": "PASS" if self.passed_hist_corr else "FAIL",
            },
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class InstrumentationStats:
    """Aggregated statistics from the detection pipeline."""

    total_frames_processed: int = 0
    total_comparisons: int = 0
    keyframes_detected: int = 0

    # Rejection breakdown
    rejected_mse: int = 0
    rejected_ssim: int = 0
    rejected_change_ratio: int = 0
    rejected_hist_corr: int = 0
    rejected_cooldown: int = 0

    # Timing
    avg_comparison_time_ms: float = 0.0
    total_comparison_time_ms: float = 0.0

    # Metric distributions (for analysis)
    ssim_scores: List[float] = field(default_factory=list)
    change_ratios: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frames": {
                "total_processed": self.total_frames_processed,
                "total_comparisons": self.total_comparisons,
                "keyframes_detected": self.keyframes_detected,
                "keyframe_rate": f"{(self.keyframes_detected / max(1, self.total_comparisons)) * 100:.1f}%",
            },
            "rejections": {
                "mse_too_low": self.rejected_mse,
                "ssim_too_high": self.rejected_ssim,
                "change_ratio_too_low": self.rejected_change_ratio,
                "hist_corr_too_high": self.rejected_hist_corr,
                "cooldown": self.rejected_cooldown,
            },
            "timing": {
                "avg_comparison_ms": round(self.avg_comparison_time_ms, 2),
                "total_comparison_ms": round(self.total_comparison_time_ms, 2),
            },
            "metric_distributions": {
                "ssim_min": (
                    round(min(self.ssim_scores), 4) if self.ssim_scores else None
                ),
                "ssim_max": (
                    round(max(self.ssim_scores), 4) if self.ssim_scores else None
                ),
                "ssim_avg": (
                    round(sum(self.ssim_scores) / len(self.ssim_scores), 4)
                    if self.ssim_scores
                    else None
                ),
                "change_ratio_min": (
                    round(min(self.change_ratios), 4) if self.change_ratios else None
                ),
                "change_ratio_max": (
                    round(max(self.change_ratios), 4) if self.change_ratios else None
                ),
                "change_ratio_avg": (
                    round(sum(self.change_ratios) / len(self.change_ratios), 4)
                    if self.change_ratios
                    else None
                ),
            },
        }


class FrameCaptureMode(str, Enum):
    """How frames are captured for keyframe detection."""

    FPS = "fps"
    """Capture at regular intervals (e.g., 0.5 FPS). Good for general capture."""

    INPUT_TRIGGERED = "input_triggered"
    """Only capture around input events (clicks, typing). Requires pynput."""

    HYBRID = "hybrid"
    """Both FPS capture AND input-triggered. Maximum coverage."""


class KeyframeSelectionMode(str, Enum):
    """How keyframes are selected from captured frames."""

    DIRECT = "direct"
    """Use captured frames directly as keyframes (no filtering)."""

    ALGORITHMIC = "algorithmic"
    """Use SSIM/MSE/histogram analysis to detect visual changes."""

    LLM = "llm"
    """Use vision LLM to select semantically important keyframes."""


class GuidedLearningSettings(BaseModel):
    """Configuration for the GuidedLearningManager."""

    # --- Capture & Selection Modes (Composable) ---
    capture_mode: FrameCaptureMode = Field(
        default=FrameCaptureMode.FPS,
        description="How frames are captured: fps (regular interval), input_triggered (pynput), or hybrid (both).",
    )
    selection_mode: KeyframeSelectionMode = Field(
        default=KeyframeSelectionMode.ALGORITHMIC,
        description="How keyframes are selected: direct (no filtering), algorithmic (SSIM/MSE), or llm (vision model).",
    )

    # --- Activity Window Detection ---
    silence_threshold_sec: float = Field(
        default=3.0,
        description="Seconds of silence before potentially ending a step. "
        "This is the primary boundary signal - user must pause speech for this long.",
    )
    visual_stability_threshold_sec: float = Field(
        default=2.0,
        description="Seconds without visual changes before considering screen stable. "
        "Only used in FPS/HYBRID capture modes (ignored in INPUT_TRIGGERED).",
    )
    min_step_duration_sec: float = Field(
        default=2.0,
        description="Minimum duration for a step to be emitted. "
        "Prevents tiny steps from very brief pauses.",
    )
    max_step_duration_sec: float = Field(
        default=60.0,
        description="Force emit a step after this duration (prevents infinite accumulation).",
    )

    # --- Visual Change Detection Thresholds ---
    # These are tuned for detecting meaningful UI changes
    ssim_threshold: float = Field(
        default=0.95,
        description="SSIM below this = significant change (lower = more sensitive).",
    )
    mse_threshold: float = Field(
        default=20.0,
        description="MSE above this = potential change.",
    )
    change_ratio_threshold: float = Field(
        default=0.01,
        description="Fraction of pixels changed (lower = more sensitive).",
    )
    hist_corr_threshold: float = Field(
        default=0.98,
        description="Histogram correlation below this = scene change. Higher = more permissive.",
    )

    # --- Frame Handling ---
    frame_buffer_size: int = Field(
        default=300,
        description="Number of frames to keep in memory (~60s at 5fps).",
    )
    keyframe_cooldown_sec: float = Field(
        default=0.3,
        description="Minimum time between keyframes (prevents animation floods).",
    )
    comparison_resolution: Tuple[int, int] = Field(
        default=(512, 288),
        description="Resolution for frame comparison (smaller = faster).",
    )

    # --- Processing ---
    max_workers: int = Field(
        default=4,
        description="Thread pool size for frame processing.",
    )

    # --- Instrumentation ---
    instrumentation_enabled: bool = Field(
        default=True,
        description="Enable detailed instrumentation and logging.",
    )
    save_keyframes: bool = Field(
        default=True,
        description="Save detected keyframes to disk.",
    )
    save_rejected_samples: bool = Field(
        default=False,
        description="Save sample of rejected frames (for debugging thresholds).",
    )
    rejected_sample_rate: int = Field(
        default=20,
        description="Save every Nth rejected frame (to avoid disk spam).",
    )
    instrumentation_dir: str = Field(
        default="captures/guided_learning/instrumentation",
        description="Directory for instrumentation output.",
    )

    # --- LLM Keyframe Selection Mode ---
    llm_keyframe_selection: bool = Field(
        default=False,
        description="Use LLM to select keyframes instead of algorithmic detection.",
    )
    llm_selection_model: str = Field(
        default="gemini-2.5-flash@vertex-ai",
        description="Vision model to use for keyframe selection.",
    )
    llm_selection_fps: float = Field(
        default=0.5,
        description="Capture FPS when using LLM selection mode (lower = fewer frames to process).",
    )
    pynput_buffer_fps: float = Field(
        default=10.0,
        description="Frame buffer FPS when pynput is enabled. Higher = more frames captured during fast typing.",
    )
    llm_selection_resolution: Tuple[int, int] = Field(
        default=(768, 432),
        description="Resolution for frames sent to LLM (width, height). Lower = fewer tokens.",
    )
    llm_selection_max_frames: int = Field(
        default=20,
        description="Maximum frames to send to LLM per segment. If exceeded, frames are intelligently sampled.",
    )
    save_llm_input_frames: bool = Field(
        default=False,
        description="Save the frames sent to the LLM (after resizing) for debugging/tuning.",
    )
    # Pre-filter settings (conservative duplicate removal)
    # Works for both LLM and DIRECT modes to remove near-duplicate frames
    prefilter_enabled: bool = Field(
        default=True,
        description="Enable conservative pre-filter to remove near-duplicate frames. Works in LLM and DIRECT modes.",
    )
    prefilter_ssim_threshold: float = Field(
        default=0.98,
        description="SSIM threshold for duplicate detection. Higher = more conservative (0.98 = nearly identical only).",
    )
    save_prefilter_discarded: bool = Field(
        default=False,
        description="Save frames discarded by pre-filter for debugging/tuning.",
    )
    # Input listener settings (pynput - requires accessibility permissions)
    enable_input_listener: bool = Field(
        default=False,
        description="Enable pynput input listener for accurate click/keyboard capture. Requires local execution with accessibility permissions.",
    )
    verbose_input_events: bool = Field(
        default=False,
        description="Print input events to console in real-time for visibility.",
    )
    pre_click_capture_ms: float = Field(
        default=100.0,
        description="Capture frame this many ms BEFORE a click (to show pre-click state).",
    )
    post_click_delay_ms: float = Field(
        default=300.0,
        description="Capture frame this many ms AFTER a click (to show result).",
    )
    typing_frame_interval_chars: int = Field(
        default=10,
        description="Capture a frame every N characters typed.",
    )


@dataclass
class _FrameData:
    """Internal frame data with pre-computed comparison image."""

    timestamp: float
    raw_b64: str  # Original base64 data
    comparison_img: Image.Image  # Downscaled grayscale for comparison
    interaction_reason: Optional[str] = None  # Set for input-triggered frames


@dataclass
class _ActivityWindowState:
    """Tracks the current activity window being accumulated."""

    # Accumulated content
    speech_segments: List[SpeechSegment] = field(default_factory=list)
    keyframes: List[KeyframeEvent] = field(default_factory=list)

    # Timing
    start_time: Optional[float] = None
    last_speech_time: float = 0.0
    last_visual_change_time: float = 0.0

    # Context
    context_frame: Optional[_FrameData] = None  # First frame when window opened

    # Speech synchronization
    speech_in_flight: bool = False  # True when user is actively speaking
    speech_in_flight_since: float = 0.0  # When speech started
    pending_transcription: bool = False  # True when waiting for transcription

    # LLM mode: store all frames for LLM to select from
    all_frames: List[_FrameData] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.start_time is not None

    def reset(self):
        self.speech_segments.clear()
        self.keyframes.clear()
        self.start_time = None
        self.last_speech_time = 0.0
        self.last_visual_change_time = 0.0
        self.context_frame = None
        self.speech_in_flight = False
        self.speech_in_flight_since = 0.0
        self.pending_transcription = False
        self.all_frames.clear()


class GuidedLearningManager:
    """
    Captures keyframes and speech for guided learning scenarios.

    Usage:
    ------
    ```python
    manager = GuidedLearningManager()
    await manager.start()

    # Push frames continuously (from screen capture)
    await manager.push_frame(frame_b64, timestamp)

    # Push speech segments (from transcription)
    await manager.push_speech("I click the submit button", start=10.0, end=12.5)

    # Steps are emitted automatically when activity window closes
    # Or you can force-emit the current step
    step = await manager.flush_current_step()

    # Register a callback to receive steps as they're detected
    manager.on_step_complete(my_callback)

    await manager.stop()
    ```
    """

    def __init__(
        self,
        settings: Optional[GuidedLearningSettings] = None,
        image_manager=None,
        debug: bool = False,
    ):
        self.settings = settings or GuidedLearningSettings()
        self._debug = debug
        self._image_manager = image_manager or ManagerRegistry.get_image_manager()

        # Threading
        self._executor = ThreadPoolExecutor(max_workers=self.settings.max_workers)
        self._stop_event = asyncio.Event()

        # Frame buffer (ring buffer of recent frames)
        self._frame_buffer: Deque[_FrameData] = deque(
            maxlen=self.settings.frame_buffer_size,
        )
        self._frame_lock = asyncio.Lock()

        # Visual change detection state
        self._last_significant_frame: Optional[_FrameData] = None
        self._last_keyframe_time: float = 0.0

        # Activity window state
        self._activity_state = _ActivityWindowState()
        self._activity_lock = asyncio.Lock()

        # Step emission
        self._step_callbacks: List[Callable[[GuidedLearningStep], Awaitable[None]]] = []
        self._pending_steps: asyncio.Queue[GuidedLearningStep] = asyncio.Queue()

        # LLM progress callback (for sandbox feedback)
        self._on_llm_progress: Optional[Callable[[str, dict], Awaitable[None]]] = None

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None

        # Session timing
        self._session_start_time: float = 0.0

        # ─── Instrumentation ───
        self._stats = InstrumentationStats()
        self._comparison_results: List[FrameComparisonResult] = []
        self._instrumentation_dir: Optional[Path] = None
        self._session_id: str = ""
        self._rejected_frame_counter: int = 0
        self._keyframe_counter: int = 0
        self._step_counter: int = 0
        self._current_step_dir: Optional[Path] = None

        # ─── Input Listener (pynput) ───
        self._input_listener: Optional[InputEventListener] = None
        self._pending_input_events: List[InputEvent] = []
        self._input_event_lock = asyncio.Lock()
        self._last_typing_frame_time: float = 0.0
        self._last_llm_frame_time: float = (
            0.0  # Rate limit for regular FPS frames to all_frames
        )
        self._accumulated_text_start_time: float = (
            0.0  # When accumulated typing started
        )
        self._accumulated_text: str = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_event_counter: int = 0
        self._input_events_log: List[dict] = []  # For instrumentation

    async def start(self):
        """Start the manager and background monitoring."""
        logger.info("GuidedLearningManager starting...")
        self._session_start_time = time.time()
        self._stop_event.clear()

        # Store event loop for thread-safe callbacks (works on all platforms)
        self._loop = asyncio.get_running_loop()

        # Log session start info (useful for debugging timestamp conversions)
        logger.info(
            f"Session started at {self._session_start_time:.3f} (frame timestamps are relative to this)",
        )

        # Initialize instrumentation
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._step_counter = 0
        self._current_step_dir = None
        if self.settings.instrumentation_enabled:
            self._instrumentation_dir = (
                Path(self.settings.instrumentation_dir) / self._session_id
            )
            self._instrumentation_dir.mkdir(parents=True, exist_ok=True)
            # Create steps directory for step-specific keyframes/llm_input_frames
            (self._instrumentation_dir / "steps").mkdir(exist_ok=True)
            if self.settings.save_rejected_samples:
                (self._instrumentation_dir / "rejected_samples").mkdir(exist_ok=True)
            logger.info(f"Instrumentation output: {self._instrumentation_dir}")

        # Start input listener if needed (INPUT_TRIGGERED or HYBRID mode, or explicitly enabled)
        needs_input_listener = (
            self.settings.enable_input_listener
            or self.settings.capture_mode
            in (FrameCaptureMode.INPUT_TRIGGERED, FrameCaptureMode.HYBRID)
        )
        if needs_input_listener:
            try:
                self._input_listener = InputEventListener(
                    on_event=self._on_input_event,
                    settings=InputListenerSettings(),
                )
                self._input_listener.start()
                logger.info("🖱️ Input listener enabled (pynput)")
            except ImportError as e:
                logger.warning(f"Could not start input listener: {e}")
                logger.warning("Install pynput with: pip install pynput")
                logger.warning("Also ensure accessibility permissions are granted.")
                if self.settings.capture_mode == FrameCaptureMode.INPUT_TRIGGERED:
                    logger.error(
                        "INPUT_TRIGGERED mode requires pynput - falling back to FPS mode",
                    )
                    self.settings.capture_mode = FrameCaptureMode.FPS

        self._monitor_task = asyncio.create_task(self._activity_monitor_loop())
        logger.info("GuidedLearningManager started.")

    async def stop(self):
        """Stop the manager and flush any pending step."""
        logger.info("GuidedLearningManager stopping...")
        self._stop_event.set()

        # Flush any in-progress step
        if self._activity_state.is_active():
            step = await self._emit_current_step()
            if step:
                await self._pending_steps.put(step)

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Stop input listener
        if self._input_listener:
            self._input_listener.stop()
            self._input_listener = None
            logger.info("🖱️ Input listener stopped")

        # Generate instrumentation report
        if self.settings.instrumentation_enabled:
            report_dir = await self.generate_instrumentation_report()
            if report_dir:
                logger.info(f"Instrumentation report saved to: {report_dir}")

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("GuidedLearningManager stopped.")

    def on_step_complete(
        self,
        callback: Callable[[GuidedLearningStep], Awaitable[None]],
    ):
        """Register a callback to be called when a step is completed."""
        self._step_callbacks.append(callback)

    def on_llm_progress(
        self,
        callback: Callable[[str, dict], Awaitable[None]],
    ):
        """
        Register a callback for LLM processing progress updates.

        The callback receives (status, data) where status is one of:
        - "started": LLM keyframe selection started
        - "calling_llm": About to call the LLM API
        - "completed": LLM selection finished successfully
        - "failed": LLM selection failed (will use fallback)

        Data dict contains relevant info like num_frames, duration, etc.
        """
        self._on_llm_progress = callback

    # ─── Input Event Handlers (pynput) ──────────────────────────────────────────

    def _on_input_event(self, event: InputEvent) -> None:
        """
        Handle input events from pynput (called from background thread).

        Schedules async processing on the event loop.
        """
        if not self._loop or not self._loop.is_running():
            return

        # Count and log the event
        self._input_event_counter += 1
        event_log = {
            "counter": self._input_event_counter,
            "type": event.event_type.name,
            "timestamp": event.timestamp,
            "position": (
                {"x": event.position.x, "y": event.position.y}
                if event.position
                else None
            ),
            "button": event.button,
            "key": event.key,
            "modifiers": event.modifiers,
            "text": event.text,
        }
        self._input_events_log.append(event_log)

        # Log for visibility
        pos_str = (
            f"at ({event.position.x}, {event.position.y})" if event.position else ""
        )
        text_str = f' text="{event.text}"' if event.text else ""
        key_str = f" key={event.key}" if event.key else ""
        log_msg = f"🖱️ pynput #{self._input_event_counter}: {event.event_type.name} {pos_str}{text_str}{key_str}"
        logger.debug(log_msg)

        # Print to console if verbose mode enabled (for debugging/demos)
        if self.settings.verbose_input_events:
            relative_ts = event.timestamp - self._session_start_time
            print(f"   {log_msg} (t={relative_ts:.2f}s)")

        # Schedule async processing (thread-safe, works on all platforms)
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._process_input_event(event)),
        )

    async def _process_input_event(self, event: InputEvent) -> None:
        """Process an input event and capture relevant keyframes."""
        try:
            if event.event_type == InputEventType.CLICK:
                await self._capture_click_keyframes(event)
            elif event.event_type == InputEventType.DOUBLE_CLICK:
                await self._capture_click_keyframes(event, is_double=True)
            elif event.event_type == InputEventType.DRAG_END:
                await self._capture_drag_keyframes(event)
            elif event.event_type == InputEventType.TYPING:
                await self._capture_typing_keyframes(event)
            elif event.event_type == InputEventType.KEY_COMBO:
                await self._capture_key_combo_keyframes(event)
        except Exception as e:
            logger.error(f"Error processing input event: {e}", exc_info=True)

    async def _capture_click_keyframes(
        self,
        event: InputEvent,
        is_double: bool = False,
    ) -> None:
        """Capture pre-click and post-click keyframes."""
        # CRITICAL: pynput timestamps are ABSOLUTE (Unix epoch time)
        # Frame timestamps are RELATIVE (seconds since session start)
        # We must convert pynput timestamps to relative before comparing!

        # Convert absolute event timestamp to relative
        # Convert pynput absolute timestamp to relative (cross-platform compatible)
        event_time_rel = event.timestamp - self._session_start_time
        verbose = self.settings.verbose_input_events

        async with self._frame_lock:
            if not self._frame_buffer:
                logger.warning("Frame buffer is empty - cannot capture click keyframe")
                return

            # Find frame closest to click time (pre-click) - using RELATIVE time
            pre_click_frame = None
            pre_click_target_rel = event_time_rel - (
                self.settings.pre_click_capture_ms / 1000.0
            )

            for frame in reversed(self._frame_buffer):
                if frame.timestamp <= pre_click_target_rel:
                    pre_click_frame = frame
                    break
            if not pre_click_frame and self._frame_buffer:
                pre_click_frame = self._frame_buffer[-1]

            if verbose and pre_click_frame:
                logger.debug(
                    f"Click frame: target≤{pre_click_target_rel:.2f}s, selected={pre_click_frame.timestamp:.2f}s",
                )

        # Wait for post-click frame
        await asyncio.sleep(self.settings.post_click_delay_ms / 1000.0)

        async with self._frame_lock:
            post_click_frame = self._frame_buffer[-1] if self._frame_buffer else None

        # Record keyframes
        click_type = "double_click" if is_double else "click"
        position_str = (
            f"({event.position.x}, {event.position.y})" if event.position else "unknown"
        )

        if pre_click_frame:
            await self._record_interaction_keyframe(
                pre_click_frame,
                reason=f"pre_{click_type}",
                annotation=f"Before {click_type} at {position_str}",
                position=event.position,
            )

        if post_click_frame and post_click_frame != pre_click_frame:
            await self._record_interaction_keyframe(
                post_click_frame,
                reason=f"post_{click_type}",
                annotation=f"After {click_type} at {position_str}",
                position=event.position,
            )

    async def _capture_drag_keyframes(self, event: InputEvent) -> None:
        """Capture drag start and end keyframes."""
        # Get frame while holding lock
        end_frame = None
        async with self._frame_lock:
            if self._frame_buffer:
                end_frame = self._frame_buffer[-1]

        # Record outside the lock to avoid deadlock with _activity_lock
        if end_frame and event.drag_start and event.drag_end:
            await self._record_interaction_keyframe(
                end_frame,
                reason="drag_end",
                annotation=f"Drag from ({event.drag_start.x}, {event.drag_start.y}) to ({event.drag_end.x}, {event.drag_end.y})",
                position=event.drag_end,
            )

    async def _capture_typing_keyframes(self, event: InputEvent) -> None:
        """
        Capture keyframes when user types text.

        Cross-platform compatible: Uses time.time() for timestamps which works
        on Windows, Linux, and macOS. The pynput library handles platform-specific
        keyboard/mouse event capture internally.
        """
        if not event.text:
            return

        verbose = self.settings.verbose_input_events

        # Track when accumulated text started (first character of the accumulation)
        # BUT: Ignore leading newlines - they often come from Enter to start the demo
        # or from previous command execution and shouldn't set the timestamp
        # This is cross-platform: \n and \r are handled consistently across OS
        text_to_add = event.text
        if not self._accumulated_text:
            # Strip leading newlines when starting a new accumulation
            text_to_add = event.text.lstrip("\n\r")
            if text_to_add:
                # Only set start time if there's actual content after stripping
                self._accumulated_text_start_time = event.timestamp
            else:
                if verbose:
                    logger.debug(
                        "Ignoring leading newlines (not starting accumulation)",
                    )
                return  # Don't accumulate pure newlines at the start

        self._accumulated_text += text_to_add

        # Capture a frame every N characters
        if len(self._accumulated_text) >= self.settings.typing_frame_interval_chars:
            now = time.time()
            time_since_last = now - self._last_typing_frame_time

            if time_since_last > 0.5:  # Rate limit to avoid duplicate captures
                # CRITICAL: pynput timestamps are ABSOLUTE (Unix epoch time via time.time())
                # Frame timestamps are RELATIVE (seconds since session start)
                # We must convert pynput timestamps to relative before comparing!
                # This approach works identically on Windows, Linux, and macOS.

                # Use the CURRENT event's timestamp for frame lookup
                # (not the accumulated start time, which might be from old input)
                current_event_rel = event.timestamp - self._session_start_time

                frame = None
                best_frame = None
                best_diff = float("inf")

                # Target: a frame captured slightly before the event (to catch typing in progress)
                target_time_rel = current_event_rel - 0.1  # 100ms before the event

                async with self._frame_lock:
                    if self._frame_buffer:
                        # Find frame closest to target_time
                        for f in self._frame_buffer:
                            diff = abs(f.timestamp - target_time_rel)
                            if diff < best_diff:
                                best_diff = diff
                                best_frame = f

                        frame = best_frame

                        if verbose and frame:
                            logger.debug(
                                f"Typing frame: target={target_time_rel:.2f}s, "
                                f"selected={frame.timestamp:.2f}s (diff={best_diff:.3f}s)",
                            )
                    else:
                        logger.warning(
                            "Frame buffer is empty - cannot capture typing keyframe",
                        )

                # Record outside the lock to avoid deadlock with _activity_lock
                if frame:
                    text_to_record = self._accumulated_text
                    self._accumulated_text = ""
                    self._accumulated_text_start_time = (
                        0.0  # Reset for next accumulation
                    )
                    self._last_typing_frame_time = now

                    await self._record_interaction_keyframe(
                        frame,
                        reason="typing",
                        annotation=f'Typed: "{text_to_record}"',
                    )
                else:
                    logger.warning(
                        f"No frame found for typing event: {repr(self._accumulated_text)}",
                    )

    async def _capture_key_combo_keyframes(self, event: InputEvent) -> None:
        """
        Capture keyframes for key combinations (e.g., Ctrl+C, Cmd+V).

        Cross-platform compatible: pynput normalizes modifier keys across platforms.
        The 'ctrl' modifier works on Windows/Linux, and pynput maps Cmd to ctrl on macOS
        when appropriate, or reports it as 'cmd' for Mac-specific shortcuts.
        """
        # Convert pynput absolute timestamp to relative (cross-platform compatible)
        event_time_rel = event.timestamp - self._session_start_time
        verbose = self.settings.verbose_input_events

        # Get frame from BEFORE the key combo event (to show the state before the action)
        frame = None
        pre_combo_target_rel = event_time_rel - 0.15  # 150ms before event

        async with self._frame_lock:
            if self._frame_buffer:
                # Find frame closest to but before the pre-combo target
                for f in reversed(self._frame_buffer):
                    if f.timestamp <= pre_combo_target_rel:
                        frame = f
                        break
                # Fallback to oldest frame if none found before target
                if not frame:
                    frame = self._frame_buffer[0] if self._frame_buffer else None

                if verbose and frame:
                    logger.debug(
                        f"Key combo frame: target≤{pre_combo_target_rel:.2f}s, selected={frame.timestamp:.2f}s",
                    )
            else:
                logger.warning(
                    "Frame buffer is empty - cannot capture key combo keyframe",
                )

        if not frame:
            return

        modifiers = "+".join(event.modifiers) if event.modifiers else ""
        combo = f"{modifiers}+{event.key}" if modifiers else event.key

        # Record outside the lock to avoid deadlock with _activity_lock
        await self._record_interaction_keyframe(
            frame,
            reason="key_combo",
            annotation=f"Key combo: {combo}",
        )

    async def _record_interaction_keyframe(
        self,
        frame_data: _FrameData,
        reason: str,
        annotation: str,
        position: Optional[Point] = None,
    ) -> None:
        """
        Record a keyframe triggered by user interaction.

        Behavior depends on selection_mode:
        - LLM: Add frame to all_frames for later LLM selection
        - DIRECT with prefilter: Add to all_frames for later prefiltering
        - DIRECT without prefilter / ALGORITHMIC: Add directly to keyframes list
        """
        selection_mode = self.settings.selection_mode
        use_deferred_processing = selection_mode == KeyframeSelectionMode.LLM or (
            selection_mode == KeyframeSelectionMode.DIRECT
            and self.settings.prefilter_enabled
        )

        async with self._activity_lock:
            # Open activity window if not active (this creates the step directory)
            if not self._activity_state.is_active():
                await self._open_activity_window(frame_data.timestamp)

            if use_deferred_processing:
                # Store for later processing (LLM selection or DIRECT with prefilter)
                # Mark the frame with interaction metadata for context
                frame_data.interaction_reason = f"[{reason}] {annotation}"
                if (
                    len(self._activity_state.all_frames)
                    < self.settings.llm_selection_max_frames
                ):
                    self._activity_state.all_frames.append(frame_data)
                    mode_label = (
                        "LLM"
                        if selection_mode == KeyframeSelectionMode.LLM
                        else "prefilter"
                    )
                    log_msg = f"📸 Queued interaction frame #{len(self._activity_state.all_frames)} for {mode_label}: [{reason}] {annotation}"
                    logger.debug(log_msg)
                    if self.settings.verbose_input_events:
                        print(f"   {log_msg}")
                else:
                    logger.debug(
                        f"⚠️ Interaction frame dropped (max {self.settings.llm_selection_max_frames} reached): [{reason}] {annotation}",
                    )
            else:
                # DIRECT without prefilter or ALGORITHMIC: Add directly as keyframe
                raw_b64 = frame_data.raw_b64
                if raw_b64.startswith("data:image"):
                    raw_b64 = raw_b64.split(",", 1)[1]

                try:
                    image_handle = self._image_manager.add_images(
                        [{"data": raw_b64, "auto_caption": False}],
                        synchronous=False,
                        return_handles=True,
                    )[0]
                except Exception as e:
                    logger.warning(f"Failed to create image handle: {e}")
                    return

                keyframe = KeyframeEvent(
                    timestamp=frame_data.timestamp,
                    image_handle=image_handle,
                    detection_reason=f"[{reason}] {annotation}",
                )
                self._activity_state.keyframes.append(keyframe)
                logger.debug(
                    f"📸 Recorded interaction keyframe: {reason} - {annotation}",
                )

            self._activity_state.last_visual_change_time = time.time()

        # Save keyframe for instrumentation (only when adding directly, not when deferring)
        if self.settings.instrumentation_enabled and not use_deferred_processing:
            self._keyframe_counter += 1
            await self._save_keyframe(frame_data, comparison_result=None)

    async def get_next_step(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[GuidedLearningStep]:
        """
        Wait for and return the next completed step.

        Returns None if timeout expires or manager is stopped.
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._pending_steps.get(),
                    timeout=timeout,
                )
            else:
                return await self._pending_steps.get()
        except asyncio.TimeoutError:
            return None

    async def push_frame(self, frame_b64: str, timestamp: float):
        """
        Push a new frame from the screen capture.

        Args:
            frame_b64: Base64-encoded image (with or without data URL prefix)
            timestamp: Timestamp relative to session start
        """
        # Process frame in thread pool (CPU-bound image operations)
        loop = asyncio.get_running_loop()
        try:
            comparison_img = await loop.run_in_executor(
                self._executor,
                self._preprocess_frame,
                frame_b64,
            )
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return

        frame_data = _FrameData(
            timestamp=timestamp,
            raw_b64=frame_b64,
            comparison_img=comparison_img,
        )

        async with self._frame_lock:
            self._frame_buffer.append(frame_data)
            buffer_size = len(self._frame_buffer)

        # Periodic buffer status logging (every ~100 frames)
        self._frame_push_count = getattr(self, "_frame_push_count", 0) + 1
        if self._frame_push_count % 100 == 1:
            logger.debug(
                f"Frame buffer: #{self._frame_push_count} at t={timestamp:.2f}s ({buffer_size} frames)",
            )

        # Determine behavior based on capture and selection modes
        capture_mode = self.settings.capture_mode
        selection_mode = self.settings.selection_mode

        # In INPUT_TRIGGERED mode, don't store FPS frames for LLM (only input-triggered frames)
        # In FPS or HYBRID mode, store frames at regular intervals
        should_store_for_llm = (
            selection_mode == KeyframeSelectionMode.LLM
            and capture_mode in (FrameCaptureMode.FPS, FrameCaptureMode.HYBRID)
        )

        if should_store_for_llm:
            # In HYBRID mode, drastically reduce regular FPS frames
            # The interaction frames from pynput are what matter
            # Regular frames are just "background context" every ~5 seconds
            now = time.time()

            if capture_mode == FrameCaptureMode.HYBRID:
                # Much slower rate for HYBRID - interaction frames are the focus
                llm_frame_interval = 5.0  # One frame every 5 seconds as background
            else:
                # Normal rate for FPS-only mode
                llm_frame_interval = (
                    1.0 / self.settings.llm_selection_fps
                    if self.settings.llm_selection_fps > 0
                    else 2.0
                )

            time_since_last = now - self._last_llm_frame_time

            if time_since_last >= llm_frame_interval:
                async with self._activity_lock:
                    if self._activity_state.is_active():
                        # Respect max frames limit
                        if (
                            len(self._activity_state.all_frames)
                            < self.settings.llm_selection_max_frames
                        ):
                            self._activity_state.all_frames.append(frame_data)
                            self._last_llm_frame_time = now
            # Still check visual change for activity window management
            await self._check_visual_change(frame_data, record_keyframe=False)
        elif selection_mode == KeyframeSelectionMode.ALGORITHMIC:
            # Algorithmic mode: check for visual change and record keyframes
            await self._check_visual_change(frame_data)
        else:
            # DIRECT mode or INPUT_TRIGGERED with LLM: just check for activity window management
            await self._check_visual_change(frame_data, record_keyframe=False)

    async def signal_speech_started(self):
        """
        Signal that the user has started speaking.

        Call this when VAD detects speech onset. This prevents the activity
        window from closing while the user is speaking.
        """
        async with self._activity_lock:
            self._activity_state.speech_in_flight = True
            self._activity_state.speech_in_flight_since = time.time()

            # Open activity window if not already open
            if not self._activity_state.is_active():
                timestamp = time.time() - self._session_start_time
                await self._open_activity_window(timestamp)

            if self._debug:
                logger.debug("Speech started - activity window held open")

    async def signal_speech_ended(self, pending_transcription: bool = True):
        """
        Signal that the user has stopped speaking.

        Args:
            pending_transcription: If True, window stays open until push_speech() is called.
        """
        async with self._activity_lock:
            self._activity_state.speech_in_flight = False
            self._activity_state.pending_transcription = pending_transcription
            self._activity_state.last_speech_time = time.time()

            if self._debug:
                logger.debug(
                    f"Speech ended - pending_transcription={pending_transcription}",
                )

    async def push_speech(self, text: str, start_time: float, end_time: float):
        """
        Push a speech segment from transcription.

        Args:
            text: Transcribed text
            start_time: Start timestamp relative to session start
            end_time: End timestamp relative to session start
        """
        async with self._activity_lock:
            # Clear pending transcription flag
            self._activity_state.pending_transcription = False

            if not text or not text.strip():
                if self._debug:
                    logger.debug("Empty speech segment, skipping")
                return

            segment = SpeechSegment(
                text=text.strip(),
                start_time=start_time,
                end_time=end_time,
            )

            if self._debug:
                logger.debug(
                    f"Speech segment: '{text}' ({start_time:.2f}s - {end_time:.2f}s)",
                )

            # Open activity window if not already open
            if not self._activity_state.is_active():
                await self._open_activity_window(start_time)

            self._activity_state.speech_segments.append(segment)
            self._activity_state.last_speech_time = time.time()

    async def flush_current_step(self) -> Optional[GuidedLearningStep]:
        """
        Force-emit the current activity window as a step.

        Useful for forcing step boundaries (e.g., user explicitly says "done").
        """
        async with self._activity_lock:
            if self._activity_state.is_active():
                return await self._emit_current_step()
        return None

    # -------------------------------------------------------------------------
    # Frame Processing (CPU-bound, runs in thread pool)
    # -------------------------------------------------------------------------

    def _preprocess_frame(self, frame_b64: str) -> Image.Image:
        """Convert base64 frame to comparison-ready image."""
        # Strip data URL prefix if present
        if frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",", 1)[1]

        img_data = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(img_data))

        # Convert to grayscale and resize for fast comparison
        return img.convert("L").resize(self.settings.comparison_resolution)

    def _is_significant_change(
        self,
        before: Image.Image,
        after: Image.Image,
        timestamp: float,
    ) -> Tuple[bool, Dict[str, float], Optional[FrameComparisonResult]]:
        """
        Determine if there's a significant visual change between frames.

        Returns (is_significant, metrics_dict, comparison_result)
        """
        comparison_start = time.time()

        before_np = np.array(before, dtype=np.uint8)
        after_np = np.array(after, dtype=np.uint8)

        metrics = {}

        # 1. Quick MSE check
        mse = float(np.mean((before_np.astype(float) - after_np.astype(float)) ** 2))
        metrics["mse"] = mse
        passed_mse = mse >= self.settings.mse_threshold

        # 2. SSIM check (perceptual similarity)
        ssim_score, diff_map = ssim(before_np, after_np, full=True)
        ssim_score = float(ssim_score)
        metrics["ssim"] = ssim_score
        passed_ssim = ssim_score < self.settings.ssim_threshold

        # 3. Pixel change ratio
        diff_map = ((1 - diff_map) * 255).astype(np.uint8)
        _, mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        change_ratio = float(np.sum(mask > 0) / mask.size)
        metrics["change_ratio"] = change_ratio
        passed_change_ratio = change_ratio >= self.settings.change_ratio_threshold

        # 4. Histogram correlation
        hist1 = cv2.calcHist([before_np], [0], None, [32], [0, 256])
        hist2 = cv2.calcHist([after_np], [0], None, [32], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        metrics["hist_corr"] = hist_corr
        passed_hist_corr = hist_corr < self.settings.hist_corr_threshold

        # Combined decision
        is_significant = (
            passed_mse and passed_ssim and passed_change_ratio and passed_hist_corr
        )

        # Build comparison result for instrumentation
        comparison_time_ms = (time.time() - comparison_start) * 1000

        rejection_reason = None
        if not is_significant:
            reasons = []
            if not passed_mse:
                reasons.append(f"MSE={mse:.1f}<{self.settings.mse_threshold}")
            if not passed_ssim:
                reasons.append(f"SSIM={ssim_score:.3f}>{self.settings.ssim_threshold}")
            if not passed_change_ratio:
                reasons.append(
                    f"ΔPx={change_ratio:.4f}<{self.settings.change_ratio_threshold}",
                )
            if not passed_hist_corr:
                reasons.append(
                    f"Hist={hist_corr:.3f}>{self.settings.hist_corr_threshold}",
                )
            rejection_reason = ", ".join(reasons)

        comparison_result = FrameComparisonResult(
            timestamp=timestamp,
            is_keyframe=is_significant,
            mse=mse,
            ssim_score=ssim_score,
            change_ratio=change_ratio,
            hist_corr=hist_corr,
            mse_threshold=self.settings.mse_threshold,
            ssim_threshold=self.settings.ssim_threshold,
            change_ratio_threshold=self.settings.change_ratio_threshold,
            hist_corr_threshold=self.settings.hist_corr_threshold,
            passed_mse=passed_mse,
            passed_ssim=passed_ssim,
            passed_change_ratio=passed_change_ratio,
            passed_hist_corr=passed_hist_corr,
            rejection_reason=rejection_reason,
        )

        # Update stats
        self._stats.total_comparisons += 1
        self._stats.total_comparison_time_ms += comparison_time_ms
        self._stats.avg_comparison_time_ms = (
            self._stats.total_comparison_time_ms / self._stats.total_comparisons
        )
        self._stats.ssim_scores.append(ssim_score)
        self._stats.change_ratios.append(change_ratio)

        if is_significant:
            self._stats.keyframes_detected += 1
        else:
            # Track first failing check (in evaluation order)
            if not passed_mse:
                self._stats.rejected_mse += 1
            elif not passed_ssim:
                self._stats.rejected_ssim += 1
            elif not passed_change_ratio:
                self._stats.rejected_change_ratio += 1
            elif not passed_hist_corr:
                self._stats.rejected_hist_corr += 1

        # Store comparison result
        if self.settings.instrumentation_enabled:
            self._comparison_results.append(comparison_result)

        return is_significant, metrics, comparison_result

    # -------------------------------------------------------------------------
    # Visual Change Detection
    # -------------------------------------------------------------------------

    async def _check_visual_change(
        self,
        frame_data: _FrameData,
        record_keyframe: bool = True,
    ):
        """Check if this frame represents a significant visual change.

        Args:
            frame_data: The frame to check
            record_keyframe: If False, only update activity window timing without recording keyframe
                           (used in LLM mode where keyframes are selected later)
        """
        current_time = time.time()

        self._stats.total_frames_processed += 1

        # Cooldown check
        if (
            current_time - self._last_keyframe_time
            < self.settings.keyframe_cooldown_sec
        ):
            # Update reference frame even during cooldown
            self._last_significant_frame = frame_data
            self._stats.rejected_cooldown += 1
            return

        if self._last_significant_frame is None:
            self._last_significant_frame = frame_data
            return

        # Run comparison in thread pool
        loop = asyncio.get_running_loop()
        is_significant, metrics, comparison_result = await loop.run_in_executor(
            self._executor,
            self._is_significant_change,
            self._last_significant_frame.comparison_img,
            frame_data.comparison_img,
            frame_data.timestamp,
        )

        if is_significant:
            self._keyframe_counter += 1

            if self._debug:
                logger.debug(
                    f"✓ KEYFRAME #{self._keyframe_counter} at t={frame_data.timestamp:.2f}s "
                    f"(SSIM={metrics.get('ssim', 0):.3f}, "
                    f"ΔPx={metrics.get('change_ratio', 0):.4f})",
                )

            if record_keyframe:
                await self._record_keyframe(frame_data, metrics, comparison_result)
            else:
                # In LLM mode, just update activity window timing
                async with self._activity_lock:
                    if not self._activity_state.is_active():
                        await self._open_activity_window(frame_data.timestamp)
                    self._activity_state.last_visual_change_time = current_time

            self._last_significant_frame = frame_data
            self._last_keyframe_time = current_time
        else:
            # Optionally save rejected frame samples
            if self.settings.save_rejected_samples and self._instrumentation_dir:
                self._rejected_frame_counter += 1
                if (
                    self._rejected_frame_counter % self.settings.rejected_sample_rate
                    == 0
                ):
                    await self._save_rejected_frame(frame_data, comparison_result)

    async def _record_keyframe(
        self,
        frame_data: _FrameData,
        metrics: Dict[str, float],
        comparison_result: Optional[FrameComparisonResult] = None,
    ):
        """Record a keyframe in the current activity window."""
        # Create ImageHandle from the frame
        raw_b64 = frame_data.raw_b64
        if raw_b64.startswith("data:image"):
            raw_b64 = raw_b64.split(",", 1)[1]

        handle = self._image_manager.add_images(
            [{"data": raw_b64, "auto_caption": False}],
            synchronous=False,
            return_handles=True,
        )[0]

        keyframe = KeyframeEvent(
            timestamp=frame_data.timestamp,
            image_handle=handle,
            ssim_score=metrics.get("ssim"),
            change_ratio=metrics.get("change_ratio"),
        )

        # Save keyframe to disk for instrumentation
        if self.settings.save_keyframes and self._instrumentation_dir:
            await self._save_keyframe(frame_data, comparison_result)

        async with self._activity_lock:
            # Open activity window if not already open
            if not self._activity_state.is_active():
                await self._open_activity_window(frame_data.timestamp)

            self._activity_state.keyframes.append(keyframe)
            self._activity_state.last_visual_change_time = time.time()

    # -------------------------------------------------------------------------
    # Activity Window Management
    # -------------------------------------------------------------------------

    async def _open_activity_window(self, start_time: float):
        """Open a new activity window."""
        if self._debug:
            logger.debug(f"Opening activity window at t={start_time:.2f}s")

        self._activity_state.start_time = start_time
        self._activity_state.last_speech_time = time.time()
        self._activity_state.last_visual_change_time = time.time()

        # Create step-specific instrumentation directory
        self._step_counter += 1
        if self.settings.instrumentation_enabled and self._instrumentation_dir:
            self._current_step_dir = (
                self._instrumentation_dir / "steps" / f"step_{self._step_counter:03d}"
            )
            self._current_step_dir.mkdir(exist_ok=True)
            (self._current_step_dir / "keyframes").mkdir(exist_ok=True)
            if self._debug:
                logger.debug(f"Created step directory: {self._current_step_dir}")

        # Capture context frame (what the screen looks like when window opens)
        async with self._frame_lock:
            if self._frame_buffer:
                self._activity_state.context_frame = self._frame_buffer[-1]

    # -------------------------------------------------------------------------
    # LLM Keyframe Selection
    # -------------------------------------------------------------------------

    async def _select_keyframes_with_llm(
        self,
        frames: List[_FrameData],
        transcript: str,
    ) -> List[KeyframeEvent]:
        """
        Use LLM to select semantically important keyframes from a list of frames.

        Args:
            frames: All frames captured during the segment
            transcript: What the user said during the segment

        Returns:
            List of KeyframeEvents for the selected frames
        """
        if not frames:
            return []

        llm_start_time = time.time()
        logger.info(f"🤖 LLM keyframe selection starting: {len(frames)} frames")

        # Notify progress callback if registered
        if self._on_llm_progress:
            await self._on_llm_progress(
                "started",
                {
                    "num_frames": len(frames),
                    "transcript_preview": (
                        transcript[:100] + "..."
                        if len(transcript) > 100
                        else transcript
                    ),
                },
            )

        # Pre-filter: Remove near-duplicate frames (conservative)
        frames_to_process = frames
        if self.settings.prefilter_enabled:
            frames_to_process, discarded = await self._prefilter_duplicate_frames(
                frames,
            )
            if discarded:
                logger.info(
                    f"🔍 Pre-filter removed {len(discarded)} near-duplicate frames ({len(frames)} → {len(frames_to_process)})",
                )
                if self._on_llm_progress:
                    await self._on_llm_progress(
                        "prefilter_complete",
                        {
                            "original_count": len(frames),
                            "kept_count": len(frames_to_process),
                            "discarded_count": len(discarded),
                        },
                    )

        # Intelligently sample frames if still exceeding max limit
        if len(frames_to_process) > self.settings.llm_selection_max_frames:
            frames_to_process = self._sample_frames_intelligently(
                frames_to_process,
                self.settings.llm_selection_max_frames,
            )
            logger.info(
                f"🤖 Sampled {len(frames_to_process)} frames (max: {self.settings.llm_selection_max_frames})",
            )

        # Prepare frames for LLM (resize to configured resolution)
        loop = asyncio.get_running_loop()
        resized_frames = []

        for frame in frames_to_process:
            try:
                resized = await loop.run_in_executor(
                    self._executor,
                    self._resize_frame_for_llm,
                    frame.raw_b64,
                )
                resized_frames.append(
                    {
                        "frame": frame,
                        "b64": resized,
                    },
                )
            except Exception as e:
                logger.warning(f"Error resizing frame: {e}")
                continue

        if not resized_frames:
            return []

        # Save LLM input frames for debugging/tuning if enabled
        if self.settings.save_llm_input_frames and self._instrumentation_dir:
            await self._save_llm_input_frames(resized_frames)

        # Build LLM system prompt
        system_prompt = self._build_keyframe_selection_prompt(
            transcript,
            len(resized_frames),
        )

        # Build user message with images
        user_content = []
        for i, rf in enumerate(resized_frames):
            frame = rf["frame"]
            # Include interaction metadata if present (from pynput)
            label = f"[Frame {i}] (t={frame.timestamp:.2f}s)"
            if frame.interaction_reason:
                label += f" - {frame.interaction_reason}"
            user_content.append(
                {
                    "type": "text",
                    "text": label,
                },
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{rf['b64']}",
                    },
                },
            )

        # Add final instruction
        user_content.append(
            {
                "type": "text",
                "text": "Based on the transcript and frames above, select the most important keyframes.",
            },
        )

        messages = [
            {"role": "user", "content": user_content},
        ]

        # Call LLM using Unity's AsyncUnify client pattern with structured output
        selection_result: Optional[KeyframeSelectionResult] = None
        llm_client = new_llm_client(self.settings.llm_selection_model)

        try:
            llm_client.set_system_message(system_prompt)
            llm_client.set_response_format(KeyframeSelectionResult)

            # Notify that we're calling the LLM
            if self._on_llm_progress:
                await self._on_llm_progress(
                    "calling_llm",
                    {
                        "num_frames": len(resized_frames),
                        "model": self.settings.llm_selection_model,
                    },
                )

            llm_call_start = time.time()
            result_text = await llm_client.generate(messages=messages)
            llm_call_duration = time.time() - llm_call_start

            logger.info(f"🤖 LLM response received in {llm_call_duration:.1f}s")

            # Parse structured response
            selection_result = KeyframeSelectionResult.model_validate_json(result_text)
            logger.info(f"🤖 LLM summary: {selection_result.summary}")

            for kf in selection_result.selected_keyframes:
                logger.info(
                    f"   📍 Frame {kf.frame_index} [{kf.importance}]: {kf.reason}",
                )

            # Notify completion
            if self._on_llm_progress:
                await self._on_llm_progress(
                    "completed",
                    {
                        "num_keyframes": len(selection_result.selected_keyframes),
                        "llm_duration_sec": llm_call_duration,
                        "summary": selection_result.summary,
                    },
                )

        except Exception as e:
            logger.error(f"LLM keyframe selection failed: {e}")

            # Notify failure
            if self._on_llm_progress:
                await self._on_llm_progress("failed", {"error": str(e)})

            # Fallback: create a minimal result with first and last frame
            fallback_indices = (
                [0, len(resized_frames) - 1] if len(resized_frames) > 1 else [0]
            )
            selection_result = KeyframeSelectionResult(
                selected_keyframes=[
                    SelectedKeyframe(
                        frame_index=idx,
                        reason="Fallback selection (LLM call failed)",
                        importance="critical" if idx == 0 else "important",
                    )
                    for idx in fallback_indices
                ],
                summary="Fallback: LLM selection failed, using first and last frames.",
            )
        finally:
            llm_client.reset_response_format()

        # Create KeyframeEvents for selected frames
        keyframes = []
        for selected in selection_result.selected_keyframes:
            idx = selected.frame_index
            if 0 <= idx < len(resized_frames):
                frame_data = resized_frames[idx]["frame"]

                raw_b64 = frame_data.raw_b64
                if raw_b64.startswith("data:image"):
                    raw_b64 = raw_b64.split(",", 1)[1]

                handle = self._image_manager.add_images(
                    [{"data": raw_b64, "auto_caption": False}],
                    synchronous=False,
                    return_handles=True,
                )[0]

                # Include reason in detection_reason for instrumentation
                detection_reason = (
                    f"llm_selected [{selected.importance}]: {selected.reason}"
                )

                keyframe = KeyframeEvent(
                    timestamp=frame_data.timestamp,
                    image_handle=handle,
                    detection_reason=detection_reason,
                )
                keyframes.append(keyframe)

                # Save keyframe to instrumentation folder
                if self.settings.save_keyframes:
                    self._keyframe_counter += 1
                    await self._save_llm_keyframe(
                        frame_data,
                        selected.reason,
                        selected.importance,
                        selection_result.summary,
                    )

        logger.info(
            f"🤖 LLM selected {len(keyframes)} keyframes from {len(frames)} frames",
        )
        return keyframes

    def _sample_frames_intelligently(
        self,
        frames: List[_FrameData],
        max_frames: int,
    ) -> List[_FrameData]:
        """
        Intelligently sample frames to stay within max_frames limit.

        Strategy:
        1. Always keep first and last frame (temporal boundaries)
        2. For remaining slots, select frames that maximize visual diversity
           using SSIM-based greedy selection
        """
        if len(frames) <= max_frames:
            return frames

        if max_frames <= 2:
            # Just return first and last
            return [frames[0], frames[-1]] if len(frames) > 1 else [frames[0]]

        # Start with first and last frame
        selected_indices = {0, len(frames) - 1}
        remaining_slots = max_frames - 2

        # For the remaining slots, use greedy selection based on visual diversity
        # We select frames that are most different from already selected frames
        candidate_indices = set(range(1, len(frames) - 1))

        while remaining_slots > 0 and candidate_indices:
            best_idx = None
            best_min_distance = -1

            for idx in candidate_indices:
                # Calculate minimum SSIM distance to any selected frame
                # (lower SSIM = more different = higher distance)
                min_ssim = 1.0
                for sel_idx in selected_indices:
                    ssim_score = self._quick_frame_similarity(
                        frames[idx].comparison_img,
                        frames[sel_idx].comparison_img,
                    )
                    min_ssim = min(min_ssim, ssim_score)

                # We want the frame with the lowest max similarity (most different)
                distance = 1.0 - min_ssim
                if distance > best_min_distance:
                    best_min_distance = distance
                    best_idx = idx

            if best_idx is not None:
                selected_indices.add(best_idx)
                candidate_indices.remove(best_idx)
                remaining_slots -= 1
            else:
                break

        # Return frames in temporal order
        sorted_indices = sorted(selected_indices)
        return [frames[i] for i in sorted_indices]

    def _quick_frame_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Quick SSIM calculation between two comparison images."""
        try:
            arr1 = np.array(img1, dtype=np.uint8)
            arr2 = np.array(img2, dtype=np.uint8)
            score, _ = ssim(arr1, arr2, full=True)
            return float(score)
        except Exception:
            return 0.5  # Default to medium similarity on error

    async def _prefilter_duplicate_frames(
        self,
        frames: List[_FrameData],
    ) -> Tuple[List[_FrameData], List[Tuple[_FrameData, float, int]]]:
        """
        Conservative pre-filter to remove near-duplicate frames.

        Uses SSIM to detect frames that are nearly identical to the previous kept frame.
        Only removes frames with SSIM >= threshold (very high similarity).

        For interaction frames (from pynput), we deduplicate based on visual similarity
        alone. Pre-click and post-click frames that are visually identical will be
        reduced to a single representative frame. The key insight is that if the screen
        didn't change, keeping multiple frames provides no additional information.

        To preserve distinct interactions that happen to look similar (e.g., clicking
        same button twice), we only compare against the immediately previous kept frame,
        not all kept frames. This means rapid repeated actions on visually static UI
        will still be captured.

        Returns:
            (kept_frames, discarded_frames) where discarded_frames is a list of
            (frame, ssim_score, compared_to_index) tuples for instrumentation.
        """
        if len(frames) <= 1:
            return frames, []

        threshold = self.settings.prefilter_ssim_threshold
        kept: List[_FrameData] = [frames[0]]  # Always keep first frame
        discarded: List[Tuple[_FrameData, float, int]] = []

        for i, frame in enumerate(frames[1:], start=1):
            last_kept = kept[-1]

            # Compare visual similarity to the last kept frame
            similarity = self._quick_frame_similarity(
                frame.comparison_img,
                last_kept.comparison_img,
            )

            if similarity >= threshold:
                # Frames are visually nearly identical - discard this one
                # The first frame in a sequence of identical frames is kept
                discarded.append((frame, similarity, len(kept) - 1))
            else:
                # Visually different - keep this frame
                kept.append(frame)

        # Save discarded frames for debugging if enabled
        if (
            discarded
            and self.settings.save_prefilter_discarded
            and self._instrumentation_dir
        ):
            await self._save_prefilter_discarded(discarded, kept)

        return kept, discarded

    async def _save_prefilter_discarded(
        self,
        discarded: List[Tuple[_FrameData, float, int]],
        kept: List[_FrameData],
    ):
        """Save frames discarded by pre-filter for debugging/tuning."""
        if not self._instrumentation_dir:
            return

        # Use step-specific directory if available, fallback to flat structure
        if self._current_step_dir:
            discarded_dir = self._current_step_dir / "prefilter_discarded"
        else:
            discarded_dir = self._instrumentation_dir / "prefilter_discarded"
        discarded_dir.mkdir(exist_ok=True)

        loop = asyncio.get_running_loop()

        for i, (frame, ssim_score, compared_to_idx) in enumerate(discarded):
            # Decode frame
            raw_b64 = frame.raw_b64
            if raw_b64.startswith("data:image"):
                raw_b64 = raw_b64.split(",", 1)[1]
            img_data = base64.b64decode(raw_b64)

            # Build filename with SSIM score and what it was compared to
            ts = f"{frame.timestamp:.2f}".replace(".", "_")
            ssim_pct = int(ssim_score * 100)
            compared_ts = f"{kept[compared_to_idx].timestamp:.2f}".replace(".", "_")
            filename = f"discarded_{i:03d}_t{ts}_ssim{ssim_pct}_vs_t{compared_ts}.png"
            filepath = discarded_dir / filename

            await loop.run_in_executor(
                self._executor,
                lambda fp=filepath, data=img_data: fp.write_bytes(data),
            )

        # Also save a summary JSON
        summary = {
            "threshold": self.settings.prefilter_ssim_threshold,
            "total_frames": len(kept) + len(discarded),
            "kept_count": len(kept),
            "discarded_count": len(discarded),
            "discarded_details": [
                {
                    "timestamp": frame.timestamp,
                    "ssim_score": ssim_score,
                    "compared_to_timestamp": kept[compared_to_idx].timestamp,
                }
                for frame, ssim_score, compared_to_idx in discarded
            ],
        }
        summary_path = discarded_dir / "prefilter_summary.json"
        await loop.run_in_executor(
            self._executor,
            lambda: summary_path.write_text(json.dumps(summary, indent=2)),
        )

        logger.info(
            f"📁 Saved {len(discarded)} pre-filter discarded frames to {discarded_dir}",
        )

    def _resize_frame_for_llm(self, frame_b64: str) -> str:
        """Resize a frame to the configured LLM resolution and return as base64 JPEG."""
        if frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",", 1)[1]

        img_data = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(img_data))

        # Resize to configured resolution
        target_size = self.settings.llm_selection_resolution
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to JPEG (smaller than PNG)
        buffered = io.BytesIO()
        img.convert("RGB").save(buffered, format="JPEG", quality=85)

        return base64.b64encode(buffered.getvalue()).decode()

    def _build_keyframe_selection_prompt(self, transcript: str, num_frames: int) -> str:
        """Build the system prompt for keyframe selection."""
        return f"""You are an expert at analyzing screen recordings of user demonstrations.

## Context
The user is demonstrating a workflow while narrating their actions. Your job is to identify the most semantically important frames that capture key moments.

## User's Narration
"{transcript}"

## Frames
You are shown {num_frames} frames captured at regular intervals. Each frame is labeled [Frame N] with its timestamp.

## Selection Criteria

**Critical frames** (always include):
- Exact moment of user interaction (click landing, keystroke visible)
- Result of an action (page loaded, command output appeared, form submitted)
- Key information being displayed that matches narration

**Important frames** (include if adds context):
- Setup/context before an action
- Intermediate states during multi-step processes
- UI elements being referenced in narration

**Do NOT select**:
- Nearly identical consecutive frames
- Static frames with no change from neighbors
- Frames that don't relate to the narration

## Guidelines
- Select 2-5 frames total (prioritize fewer, high-quality selections)
- Correlate frames with the transcript - what did the user SAY they were doing?
- Each selected frame should add unique information
- Provide clear, specific reasons for each selection

## Summary
Write a brief (1-2 sentence) summary of what the user demonstrated in this segment."""

    async def _emit_current_step(self) -> Optional[GuidedLearningStep]:
        """Emit the current activity window as a GuidedLearningStep."""
        state = self._activity_state

        if not state.is_active():
            return None

        # Build transcript from all speech segments
        transcript = " ".join(seg.text for seg in state.speech_segments)

        selection_mode = self.settings.selection_mode

        # Handle keyframe selection based on mode
        if selection_mode == KeyframeSelectionMode.LLM and state.all_frames:
            logger.info(
                f"🤖 Using LLM to select keyframes from {len(state.all_frames)} frames",
            )
            llm_keyframes = await self._select_keyframes_with_llm(
                state.all_frames,
                transcript,
            )
            # Replace any previously collected keyframes with LLM-selected ones
            state.keyframes = llm_keyframes
        elif selection_mode == KeyframeSelectionMode.DIRECT and state.all_frames:
            # DIRECT mode: convert all_frames to keyframes (with optional duplicate removal)
            frames_to_use = state.all_frames

            # Apply prefilter if enabled (removes near-duplicates while keeping interaction frames)
            if self.settings.prefilter_enabled:
                frames_to_use, discarded = await self._prefilter_duplicate_frames(
                    state.all_frames,
                )
                if discarded:
                    logger.info(
                        f"🔍 Pre-filter removed {len(discarded)} near-duplicate frames ({len(state.all_frames)} → {len(frames_to_use)})",
                    )

            logger.info(
                f"📋 DIRECT mode: using {len(frames_to_use)} captured frames as keyframes",
            )
            direct_keyframes = []
            for frame in frames_to_use:
                # Create image handle via ImageManager
                raw_b64 = frame.raw_b64
                if raw_b64.startswith("data:image"):
                    raw_b64 = raw_b64.split(",", 1)[1]

                try:
                    image_handle = self._image_manager.add_images(
                        [{"data": raw_b64, "auto_caption": False}],
                        synchronous=False,
                        return_handles=True,
                    )[0]

                    reason = frame.interaction_reason or "captured_frame"
                    direct_keyframes.append(
                        KeyframeEvent(
                            timestamp=frame.timestamp,
                            image_handle=image_handle,
                            detection_reason=reason,
                        ),
                    )

                    # Save keyframe to instrumentation folder
                    if self.settings.save_keyframes and self._instrumentation_dir:
                        self._keyframe_counter += 1
                        await self._save_keyframe(frame, comparison_result=None)

                except Exception as e:
                    logger.warning(
                        f"Failed to create image handle for frame at t={frame.timestamp}: {e}",
                    )
            state.keyframes = direct_keyframes
        # ALGORITHMIC mode: keyframes already collected via _check_visual_change
        # or interaction events (if input listener is enabled and mode is ALGORITHMIC)

        # Determine end time
        end_time = state.start_time
        if state.speech_segments:
            end_time = max(end_time, max(seg.end_time for seg in state.speech_segments))
        if state.keyframes:
            end_time = max(end_time, max(kf.timestamp for kf in state.keyframes))

        # Create context frame handle if needed
        context_keyframe = None
        if state.context_frame and not state.keyframes:
            # Commentary only - include context frame
            raw_b64 = state.context_frame.raw_b64
            if raw_b64.startswith("data:image"):
                raw_b64 = raw_b64.split(",", 1)[1]

            handle = self._image_manager.add_images(
                [{"data": raw_b64, "auto_caption": False}],
                synchronous=False,
                return_handles=True,
            )[0]

            context_keyframe = KeyframeEvent(
                timestamp=state.context_frame.timestamp,
                image_handle=handle,
                detection_reason="context_frame",
            )

        step = GuidedLearningStep(
            transcript=transcript,
            keyframes=list(state.keyframes),
            speech_segments=list(state.speech_segments),
            start_time=state.start_time,
            end_time=end_time,
            has_visual_changes=len(state.keyframes) > 0,
            is_commentary_only=len(state.keyframes) == 0
            and len(state.speech_segments) > 0,
            context_frame=context_keyframe,
        )

        if self._debug:
            logger.debug(
                f"Emitting step: {len(state.speech_segments)} speech segments, "
                f"{len(state.keyframes)} keyframes, "
                f"duration={step.duration:.2f}s",
            )

        # Reset state
        state.reset()

        return step

    async def _activity_monitor_loop(self):
        """
        Background loop that monitors activity and emits steps when windows close.

        A step is emitted when ALL conditions are true:
        1. No speech for `silence_threshold_sec`
        2. No visual changes for `visual_stability_threshold_sec`
        3. User is not actively speaking (speech_in_flight = False)
        4. No pending transcription
        """
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(0.2)  # Check every 200ms

                async with self._activity_lock:
                    if not self._activity_state.is_active():
                        continue

                    current_time = time.time()
                    state = self._activity_state

                    # NEVER close if user is actively speaking or transcription pending
                    if state.speech_in_flight:
                        if (
                            self._debug and int(current_time * 5) % 25 == 0
                        ):  # Log every 5s
                            logger.debug("Activity window held open - speech in flight")
                        continue

                    if state.pending_transcription:
                        if self._debug and int(current_time * 5) % 25 == 0:
                            logger.debug(
                                "Activity window held open - pending transcription",
                            )
                        continue

                    # Calculate silence and stability durations
                    silence_duration = current_time - state.last_speech_time
                    stability_duration = current_time - state.last_visual_change_time
                    window_duration = current_time - (
                        self._session_start_time + state.start_time
                    )

                    # Check if we should close the activity window
                    should_close = False
                    close_reason = ""

                    # Max duration exceeded (force close even if waiting)
                    if window_duration > self.settings.max_step_duration_sec:
                        should_close = True
                        close_reason = "max_duration"

                    # For INPUT_TRIGGERED mode, only use silence threshold
                    # (visual stability doesn't make sense when visual detection is event-driven)
                    elif self.settings.capture_mode == FrameCaptureMode.INPUT_TRIGGERED:
                        if silence_duration > self.settings.silence_threshold_sec:
                            if window_duration >= self.settings.min_step_duration_sec:
                                should_close = True
                                close_reason = "silence_threshold"

                    # For FPS/HYBRID modes, require both silence and visual stability
                    elif (
                        silence_duration > self.settings.silence_threshold_sec
                        and stability_duration
                        > self.settings.visual_stability_threshold_sec
                    ):
                        # Also check minimum duration
                        if window_duration >= self.settings.min_step_duration_sec:
                            should_close = True
                            close_reason = "activity_complete"

                    if should_close:
                        if self._debug:
                            logger.debug(
                                f"Closing activity window (reason={close_reason}, "
                                f"silence={silence_duration:.1f}s, "
                                f"stability={stability_duration:.1f}s)",
                            )

                        step = await self._emit_current_step()
                        if step and (step.transcript or step.keyframes):
                            # Notify callbacks
                            for callback in self._step_callbacks:
                                try:
                                    await callback(step)
                                except Exception as e:
                                    logger.error(f"Step callback error: {e}")

                            # Add to pending queue
                            await self._pending_steps.put(step)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Activity monitor error: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def get_current_frame(self) -> Optional[_FrameData]:
        """Get the most recent frame from the buffer."""
        async with self._frame_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
        return None

    async def get_frame_at_time(self, timestamp: float) -> Optional[_FrameData]:
        """Get the frame closest to the given timestamp."""
        async with self._frame_lock:
            if not self._frame_buffer:
                return None

            # Binary search would be more efficient, but linear is fine for now
            closest = min(
                self._frame_buffer,
                key=lambda f: abs(f.timestamp - timestamp),
            )
            return closest

    def get_stats(self) -> Dict:
        """Get current manager statistics."""
        return {
            "frame_buffer_size": len(self._frame_buffer),
            "activity_window_active": self._activity_state.is_active(),
            "pending_steps": self._pending_steps.qsize(),
            "session_duration": time.time() - self._session_start_time,
        }

    # -------------------------------------------------------------------------
    # Instrumentation Methods
    # -------------------------------------------------------------------------

    async def _save_keyframe(
        self,
        frame_data: _FrameData,
        comparison_result: Optional[FrameComparisonResult],
    ):
        """Save a detected keyframe to the instrumentation directory."""
        if not self._instrumentation_dir:
            return

        # Use step-specific directory if available, fallback to flat structure
        if self._current_step_dir:
            keyframe_dir = self._current_step_dir / "keyframes"
        else:
            # Fallback (shouldn't happen in normal flow, but ensures backwards compatibility)
            keyframe_dir = self._instrumentation_dir / "keyframes"
            keyframe_dir.mkdir(exist_ok=True)

        # Decode and save image
        raw_b64 = frame_data.raw_b64
        if raw_b64.startswith("data:image"):
            raw_b64 = raw_b64.split(",", 1)[1]

        img_data = base64.b64decode(raw_b64)

        # Build filename with timestamp and metrics
        ts = f"{frame_data.timestamp:.2f}".replace(".", "_")
        if comparison_result:
            ssim_str = f"ssim{comparison_result.ssim_score:.3f}".replace(".", "")
            chg_str = f"chg{comparison_result.change_ratio:.4f}".replace(".", "")
            filename = f"kf_{self._keyframe_counter:04d}_t{ts}_{ssim_str}_{chg_str}.png"
        else:
            filename = f"kf_{self._keyframe_counter:04d}_t{ts}.png"

        filepath = keyframe_dir / filename

        # Save image
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: filepath.write_bytes(img_data),
        )

        # Save metadata JSON alongside
        if comparison_result:
            meta_path = filepath.with_suffix(".json")
            meta = comparison_result.to_dict()
            await loop.run_in_executor(
                self._executor,
                lambda: meta_path.write_text(json.dumps(meta, indent=2)),
            )

    async def _save_llm_keyframe(
        self,
        frame_data: _FrameData,
        reason: str,
        importance: str,
        segment_summary: str,
    ):
        """Save an LLM-selected keyframe with its reasoning metadata."""
        if not self._instrumentation_dir:
            return

        # Use step-specific directory if available, fallback to flat structure
        if self._current_step_dir:
            keyframe_dir = self._current_step_dir / "keyframes"
        else:
            keyframe_dir = self._instrumentation_dir / "keyframes"
            keyframe_dir.mkdir(exist_ok=True)

        # Decode and save image
        raw_b64 = frame_data.raw_b64
        if raw_b64.startswith("data:image"):
            raw_b64 = raw_b64.split(",", 1)[1]

        img_data = base64.b64decode(raw_b64)

        # Build filename with timestamp and importance
        ts = f"{frame_data.timestamp:.2f}".replace(".", "_")
        importance_short = importance[:4]  # 'crit', 'impo', or 'supp'
        filename = f"kf_{self._keyframe_counter:04d}_t{ts}_llm_{importance_short}.png"

        filepath = keyframe_dir / filename

        # Save image
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: filepath.write_bytes(img_data),
        )

        # Save LLM reasoning metadata JSON alongside
        meta_path = filepath.with_suffix(".json")
        meta = {
            "timestamp": frame_data.timestamp,
            "detection_method": "llm_selection",
            "llm_reasoning": {
                "reason": reason,
                "importance": importance,
                "segment_summary": segment_summary,
            },
        }
        await loop.run_in_executor(
            self._executor,
            lambda: meta_path.write_text(json.dumps(meta, indent=2)),
        )

    async def _save_llm_input_frames(self, resized_frames: List[dict]):
        """Save the frames being sent to the LLM for debugging/tuning FPS."""
        if not self._instrumentation_dir:
            return

        # Use step-specific directory if available, fallback to flat structure
        if self._current_step_dir:
            llm_frames_dir = self._current_step_dir / "llm_input_frames"
        else:
            llm_frames_dir = self._instrumentation_dir / "llm_input_frames"
        llm_frames_dir.mkdir(exist_ok=True)

        loop = asyncio.get_running_loop()

        for i, rf in enumerate(resized_frames):
            frame_data = rf["frame"]
            b64_data = rf["b64"]

            # Decode the resized JPEG
            img_data = base64.b64decode(b64_data)

            # Build filename with index and timestamp
            ts = f"{frame_data.timestamp:.2f}".replace(".", "_")
            filename = f"llm_frame_{i:03d}_t{ts}.jpg"
            filepath = llm_frames_dir / filename

            await loop.run_in_executor(
                self._executor,
                lambda fp=filepath, data=img_data: fp.write_bytes(data),
            )

        logger.info(
            f"📸 Saved {len(resized_frames)} LLM input frames to {llm_frames_dir}",
        )

    async def _save_rejected_frame(
        self,
        frame_data: _FrameData,
        comparison_result: Optional[FrameComparisonResult],
    ):
        """Save a sample rejected frame for debugging thresholds."""
        if not self._instrumentation_dir:
            return

        rejected_dir = self._instrumentation_dir / "rejected_samples"

        # Decode and save image
        raw_b64 = frame_data.raw_b64
        if raw_b64.startswith("data:image"):
            raw_b64 = raw_b64.split(",", 1)[1]

        img_data = base64.b64decode(raw_b64)

        # Build filename with timestamp and rejection reason
        ts = f"{frame_data.timestamp:.2f}".replace(".", "_")
        rejection = ""
        if comparison_result and comparison_result.rejection_reason:
            # Shorten rejection reason for filename
            if "MSE" in comparison_result.rejection_reason:
                rejection = "_mse"
            elif "SSIM" in comparison_result.rejection_reason:
                rejection = "_ssim"
            elif "ΔPx" in comparison_result.rejection_reason:
                rejection = "_chg"
            elif "Hist" in comparison_result.rejection_reason:
                rejection = "_hist"

        filename = f"rej_{self._rejected_frame_counter:06d}_t{ts}{rejection}.png"
        filepath = rejected_dir / filename

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: filepath.write_bytes(img_data),
        )

        # Save metadata JSON
        if comparison_result:
            meta_path = filepath.with_suffix(".json")
            meta = comparison_result.to_dict()
            await loop.run_in_executor(
                self._executor,
                lambda: meta_path.write_text(json.dumps(meta, indent=2)),
            )

    def get_instrumentation_stats(self) -> Dict:
        """Get detailed instrumentation statistics."""
        return self._stats.to_dict()

    def get_comparison_results(self) -> List[Dict]:
        """Get all frame comparison results (for detailed analysis)."""
        return [r.to_dict() for r in self._comparison_results]

    async def generate_instrumentation_report(self) -> str:
        """
        Generate a comprehensive instrumentation report and save to disk.

        Returns the path to the saved report.
        """
        if not self._instrumentation_dir:
            return ""

        report = {
            "session_id": self._session_id,
            "session_duration_sec": time.time() - self._session_start_time,
            "settings": {
                "ssim_threshold": self.settings.ssim_threshold,
                "mse_threshold": self.settings.mse_threshold,
                "change_ratio_threshold": self.settings.change_ratio_threshold,
                "hist_corr_threshold": self.settings.hist_corr_threshold,
                "keyframe_cooldown_sec": self.settings.keyframe_cooldown_sec,
                "comparison_resolution": list(self.settings.comparison_resolution),
            },
            "statistics": self._stats.to_dict(),
            "all_comparisons": [r.to_dict() for r in self._comparison_results],
        }

        report_path = self._instrumentation_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Also generate a human-readable summary
        summary_lines = [
            "=" * 60,
            f" GUIDED LEARNING INSTRUMENTATION REPORT",
            f" Session: {self._session_id}",
            "=" * 60,
            "",
            "## STATISTICS",
            f"  Total frames processed: {self._stats.total_frames_processed}",
            f"  Total comparisons:      {self._stats.total_comparisons}",
            f"  Keyframes detected:     {self._stats.keyframes_detected}",
            f"  Keyframe rate:          {(self._stats.keyframes_detected / max(1, self._stats.total_comparisons)) * 100:.1f}%",
            "",
            "## REJECTION BREAKDOWN",
            f"  MSE too low:            {self._stats.rejected_mse}",
            f"  SSIM too high:          {self._stats.rejected_ssim}",
            f"  Change ratio too low:   {self._stats.rejected_change_ratio}",
            f"  Hist corr too high:     {self._stats.rejected_hist_corr}",
            f"  Cooldown:               {self._stats.rejected_cooldown}",
            "",
            "## TIMING",
            f"  Avg comparison time:    {self._stats.avg_comparison_time_ms:.2f}ms",
            "",
            "## METRIC DISTRIBUTIONS",
        ]

        if self._stats.ssim_scores:
            summary_lines.extend(
                [
                    f"  SSIM range:             [{min(self._stats.ssim_scores):.4f}, {max(self._stats.ssim_scores):.4f}]",
                    f"  SSIM avg:               {sum(self._stats.ssim_scores) / len(self._stats.ssim_scores):.4f}",
                ],
            )

        if self._stats.change_ratios:
            summary_lines.extend(
                [
                    f"  Change ratio range:     [{min(self._stats.change_ratios):.4f}, {max(self._stats.change_ratios):.4f}]",
                    f"  Change ratio avg:       {sum(self._stats.change_ratios) / len(self._stats.change_ratios):.4f}",
                ],
            )

        summary_lines.extend(
            [
                "",
                "## THRESHOLDS USED",
                f"  SSIM threshold:         {self.settings.ssim_threshold} (below = keyframe)",
                f"  MSE threshold:          {self.settings.mse_threshold} (above = potential change)",
                f"  Change ratio threshold: {self.settings.change_ratio_threshold} (above = keyframe)",
                f"  Hist corr threshold:    {self.settings.hist_corr_threshold} (below = scene change)",
                "",
                f"## FILES",
                f"  Report:     {report_path}",
                f"  Keyframes:  {self._instrumentation_dir / 'keyframes'}/",
            ],
        )

        if self.settings.save_rejected_samples:
            summary_lines.append(
                f"  Rejected:   {self._instrumentation_dir / 'rejected_samples'}/",
            )

        # Add input events section if any were captured
        if self._input_events_log:
            summary_lines.extend(
                [
                    "",
                    "## INPUT EVENTS (pynput)",
                    f"  Total events captured: {len(self._input_events_log)}",
                ],
            )
            # Count by type
            event_type_counts = {}
            for evt in self._input_events_log:
                evt_type = evt.get("type", "UNKNOWN")
                event_type_counts[evt_type] = event_type_counts.get(evt_type, 0) + 1
            for evt_type, count in sorted(event_type_counts.items()):
                summary_lines.append(f"    {evt_type}: {count}")

            # Save detailed events log
            events_path = self._instrumentation_dir / "input_events.json"
            events_path.write_text(json.dumps(self._input_events_log, indent=2))
            summary_lines.append(f"  Events log: {events_path}")

        summary_lines.extend(["", "=" * 60])

        summary_text = "\n".join(summary_lines)
        summary_path = self._instrumentation_dir / "summary.txt"
        summary_path.write_text(summary_text)

        logger.info(f"Instrumentation report saved to: {self._instrumentation_dir}")

        return str(self._instrumentation_dir)

    def print_live_stats(self):
        """Print current stats to console (useful for live monitoring)."""
        stats = self._stats

        print(f"\n{'─' * 50}")
        print(f"📊 GUIDED LEARNING LIVE STATS")
        print(f"{'─' * 50}")
        print(f"  Frames processed:  {stats.total_frames_processed}")
        print(f"  Comparisons:       {stats.total_comparisons}")
        print(f"  Keyframes:         {stats.keyframes_detected}")

        if stats.total_comparisons > 0:
            rate = (stats.keyframes_detected / stats.total_comparisons) * 100
            print(f"  Detection rate:    {rate:.1f}%")

        if stats.ssim_scores:
            print(
                f"  SSIM range:        [{min(stats.ssim_scores):.3f}, {max(stats.ssim_scores):.3f}]",
            )

        if stats.change_ratios:
            print(
                f"  ΔPx range:         [{min(stats.change_ratios):.4f}, {max(stats.change_ratios):.4f}]",
            )

        # Rejection summary
        total_rejected = (
            stats.rejected_mse
            + stats.rejected_ssim
            + stats.rejected_change_ratio
            + stats.rejected_hist_corr
            + stats.rejected_cooldown
        )
        if total_rejected > 0:
            print(
                f"\n  Rejections: MSE={stats.rejected_mse}, SSIM={stats.rejected_ssim}, "
                f"ΔPx={stats.rejected_change_ratio}, Hist={stats.rejected_hist_corr}, "
                f"Cooldown={stats.rejected_cooldown}",
            )

        print(f"{'─' * 50}\n")
