from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from textwrap import dedent
import functools
import backoff
import unillm
import cv2

from unity.common.llm_client import new_llm_client
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from skimage.metrics import structural_similarity as ssim

from unity.image_manager.image_manager import ImageHandle
from unity.manager_registry import ManagerRegistry
from .prompt_builders import (
    build_summary_update_prompt,
    build_single_annotation_prompt,
    build_detection_prompt,
)
from .types import KeyEvent, DetectedEvent

logger = logging.getLogger(__name__)


# --- Configuration and State Models ---


class ScreenShareManagerSettings(BaseModel):
    """Configuration settings for the ScreenShareManager."""

    mse_threshold: float = Field(
        default=25.0,
        description="Mean Squared Error threshold for detecting initial frame differences.",
    )
    ssim_threshold: float = Field(
        default=0.985,
        description="Structural Similarity Index threshold for perceptual difference.",
    )
    change_ratio_threshold: float = Field(
        default=0.02,
        description="Fraction of pixels in the frame that changed noticeably compared to the previous one",
    )
    hist_corr_threshold: float = Field(
        default=0.9,
        description="Measures how similar the overall brightness and color distribution of the two frames are, using histogram correlation.",
    )
    vision_event_cooldown_sec: float = Field(
        default=0.25,
        description="Cooldown period after a visual event is detected to prevent floods of events from a single animation.",
    )
    inactivity_timeout_sec: float = Field(
        default=5.0,
        description="Time without any activity before flushing pending visual events.",
    )
    frame_buffer_size: int = Field(
        default=100,
        description="Number of recent frames to keep in memory.",
    )
    max_frame_workers: int = Field(
        default=os.cpu_count() or 4,
        description="Number of parallel workers for frame processing.",
    )
    frame_queue_size: int = Field(
        default=150,
        description="Maximum number of frames to buffer for processing.",
    )
    results_queue_size: int = Field(
        default=200,
        description="Maximum number of processed frame results to buffer.",
    )
    detection_queue_size: int = Field(
        default=10,
        description="Maximum number of pending detection results.",
    )
    adaptive_drop_threshold: float = Field(
        default=0.75,
        description="Queue fullness percentage to start proactively dropping frames.",
    )
    burst_detection_threshold_sec: float = Field(
        default=1.0,
        description="Time window to group consecutive visual events into a single burst.",
    )
    visual_event_sampling_threshold: int = Field(
        default=3,
        description="Number of events in a burst to trigger sampling.",
    )
    llm_retry_max_tries: int = Field(
        default=3,
        description="Maximum number of retries for critical LLM calls.",
    )
    llm_retry_base_delay_sec: float = Field(
        default=1.0,
        description="Initial delay for LLM retry backoff.",
    )
    use_auto_captions: bool = Field(
        default=False,
        description="Enable opportunistic auto-captioning to enrich the context for the event detection LLM.",
    )


@dataclass
class TurnState:
    """Encapsulates the state for a single analysis turn."""

    speech_events: List[Dict] = field(default_factory=list)
    visual_events: List[Dict] = field(default_factory=list)
    latest_frame: Optional[Tuple[float, str]] = None


# --- Helper Decorators ---


def llm_retry_decorator(func):
    """A decorator to add exponential backoff retries to an async function, using instance settings."""

    @functools.wraps(func)
    async def wrapper(self: "ScreenShareManager", *args, **kwargs):
        settings = self.settings

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=settings.llm_retry_max_tries,
            base=settings.llm_retry_base_delay_sec,
            logger=logger,
        )
        async def inner():
            return await func(self, *args, **kwargs)

        return await inner()

    return wrapper


# --- Main Manager Class ---


class ScreenShareManager:
    """
    A stateful component that analyzes screen share streams and user speech to
    detect key events and return annotated ImageHandles.
    """

    def __init__(
        self,
        settings: Optional[ScreenShareManagerSettings] = None,
        image_manager=None,
        detection_client: Optional[unillm.AsyncUnify] = None,
        analysis_client: Optional[unillm.AsyncUnify] = None,
        summary_client: Optional[unillm.AsyncUnify] = None,
        debug: bool = False,
    ):
        """
        Initializes the ScreenShareManager with configurable settings and injected dependencies.

        Args:
            settings: A Pydantic model containing all operational parameters.
            image_manager: An instance of ImageManager for handling image storage.
            detection_client: A Unify client for the fast detection stage.
            analysis_client: A Unify client for the rich annotation stage.
            summary_client: A Unify client for updating the session summary.
            debug: Flag to toggle logging statements
        """
        self.settings = settings or ScreenShareManagerSettings()
        self._debug = debug
        self._image_manager = image_manager or ManagerRegistry.get_image_manager()
        self._detection_client = detection_client or new_llm_client()
        self._analysis_client = analysis_client or new_llm_client()
        self._summary_client = summary_client or new_llm_client()
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self.settings.max_frame_workers,
        )
        self._stop_event = asyncio.Event()
        self._frame_sequence_id = 0
        self._frame_queue = asyncio.Queue(maxsize=self.settings.frame_queue_size)
        self._results_queue = asyncio.Queue(maxsize=self.settings.results_queue_size)
        self._detection_queue = asyncio.Queue(
            maxsize=self.settings.detection_queue_size,
        )
        self._frame_workers: List[asyncio.Task] = []
        self._sequencer_task: Optional[asyncio.Task] = None
        self._inactivity_task: Optional[asyncio.Task] = None
        self._frame_buffer: Deque[Tuple[float, str]] = deque(
            maxlen=self.settings.frame_buffer_size,
        )
        self._pending_vision_events: List[Dict] = []
        self._stored_silent_detected_events: List[DetectedEvent] = []
        self._last_significant_frame_b64: Optional[str] = None
        self._last_significant_frame_pil: Optional[Image.Image] = None
        self._last_activity_time: float = 0.0
        self._last_vision_event_time: float = 0.0
        self._state_lock = asyncio.Lock()
        self._turn_in_progress: bool = False
        self._current_turn_speech_events: List[Dict] = []
        self._session_summary: str = "The session has just begun."
        self._recent_key_events: Deque[KeyEvent] = deque(maxlen=5)
        self._unsummarized_events: List[KeyEvent] = []
        self._summary_update_lock = asyncio.Lock()
        self._summary_update_task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info("ScreenShareManager starting background workers...")
        self._last_activity_time = asyncio.get_event_loop().time()
        self._sequencer_task = asyncio.create_task(self._sequencer())
        self._frame_workers = [
            asyncio.create_task(self._frame_processing_worker())
            for _ in range(self.settings.max_frame_workers)
        ]
        self._inactivity_task = asyncio.create_task(self._inactivity_flush_loop())

    async def stop(self):
        logger.info("ScreenShareManager stopping...")
        self._stop_event.set()
        tasks = [
            self._sequencer_task,
            self._inactivity_task,
            self._summary_update_task,
            *self._frame_workers,
        ]

        active_tasks = [t for t in tasks if t and not t.done()]
        for task in active_tasks:
            task.cancel()

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        self._cpu_executor.shutdown(wait=False, cancel_futures=True)

    def set_session_context(self, context_text: str):
        self._session_summary = context_text.strip()
        if self._debug:
            logger.debug(f"Initial session context set: '{self._session_summary}'")

    async def push_frame(self, frame_b64: str, timestamp: float):
        if self._frame_queue.qsize() > (
            self.settings.frame_queue_size * self.settings.adaptive_drop_threshold
        ):
            logger.warning("Frame queue is backlogged. Proactively dropping frame.")
            return
        if self._frame_queue.full():
            logger.warning("Frame queue is full. Dropping incoming frame.")
            return
        self._frame_sequence_id += 1
        await self._frame_queue.put(
            (
                self._frame_sequence_id,
                {"payload": {"frame_b64": frame_b64, "timestamp": timestamp}},
            ),
        )

    async def push_speech(self, content: str, start_time: float, end_time: float):
        if self._debug:
            logger.debug(f"Received speech event: '{content}'")
        if not self._turn_in_progress:
            logger.warning(
                "Speech event received, but no turn is in progress. Ignoring.",
            )
            return
        speech_event = {
            "payload": {
                "content": content,
                "start_time": start_time,
                "end_time": end_time,
            },
        }
        self._current_turn_speech_events.append(speech_event)
        self._last_activity_time = asyncio.get_event_loop().time()

    def start_turn(self):
        """
        Starts a new analysis turn. All subsequent speech and visual events will
        be collected for this turn until end_turn() is called.
        """
        logger.info("Starting a new manual analysis turn.")
        self._current_turn_speech_events = []
        self._pending_vision_events.clear()
        self._turn_in_progress = True

    def end_turn(self) -> asyncio.Task[List[DetectedEvent]]:
        if not self._turn_in_progress:
            logger.warning(
                "end_turn called but no turn is in progress. Returning an empty task.",
            )

            async def empty_task():
                return []

            return asyncio.create_task(empty_task())
        if self._debug:
            logger.debug(
                f"Ending turn with {len(self._current_turn_speech_events)} speech event(s) and "
                f"{len(self._pending_vision_events)} potential visual event(s).",
            )
        visual_events = list(self._pending_vision_events)
        speech_events = list(self._current_turn_speech_events)
        self._pending_vision_events.clear()
        self._current_turn_speech_events.clear()
        self._turn_in_progress = False
        latest_frame = (
            self._frame_buffer[-1]
            if (speech_events and not visual_events and self._frame_buffer)
            else None
        )
        turn_state = TurnState(
            speech_events=speech_events,
            visual_events=visual_events,
            latest_frame=latest_frame,
        )
        asyncio.create_task(self._detect_key_moments(turn_state))
        return self.detect_events()

    def detect_events(self) -> asyncio.Task[List[DetectedEvent]]:
        """
        Returns a handle (a Task) to the result of the fast detection stage
        for a turn that has been initiated. The consumer must `await` this
        task to get the list of candidate events.
        """

        async def _detection_wrapper() -> List[DetectedEvent]:
            detected_events = await self._detection_queue.get()
            async with self._state_lock:
                all_detected_events = self._stored_silent_detected_events + (
                    detected_events or []
                )
                self._stored_silent_detected_events = []
            if self._debug:
                logger.debug(
                    f"Returning {len(all_detected_events)} DetectedEvent objects for this turn.",
                )
            return all_detected_events

        return asyncio.create_task(_detection_wrapper())

    async def annotate_events(
        self,
        events: List[DetectedEvent],
        context: Optional[str] = None,
    ) -> List[ImageHandle]:
        if not events:
            return []
        if self._debug:
            logger.debug(
                f"Starting sequential annotation for {len(events)} detected events.",
            )
        annotated_handles: List[ImageHandle] = []
        annotations_so_far: List[str] = []
        key_events_for_summary: List[KeyEvent] = []
        for event in events:
            try:
                annotation_text = await self._get_llm_annotation_for_event(
                    event,
                    context,
                    annotations_so_far,
                )
                if not annotation_text:
                    logger.warning(
                        f"Annotation for event at timestamp {event.timestamp:.2f}s returned empty.",
                    )
                    continue
                annotations_so_far.append(annotation_text)
                event.image_handle.annotation = annotation_text
                annotated_handles.append(event.image_handle)
                if self._debug:
                    logger.debug(
                        f"Attached annotation '{annotation_text}' to handle for timestamp {event.timestamp:.2f}s.",
                    )
                key_event = KeyEvent(
                    timestamp=event.timestamp,
                    image_annotation=annotation_text,
                    representative_timestamp=event.timestamp,
                )
                key_events_for_summary.append(key_event)
                self._recent_key_events.append(key_event)
            except Exception as e:
                logger.error(
                    f"Failed to generate annotation for event at timestamp {event.timestamp:.2f}s. Error: {e}",
                    exc_info=True,
                )
                continue
        if key_events_for_summary:
            async with self._state_lock:
                self._unsummarized_events.extend(key_events_for_summary)
            self._trigger_summary_update()
        if self._debug:
            logger.debug(
                f"Successfully annotated {len(annotated_handles)} of {len(events)} events sequentially.",
            )
        return annotated_handles

    def _strip_data_url_prefix(self, data_url: str) -> str:
        if data_url.startswith("data:image"):
            return data_url.split(",", 1)[1]
        return data_url

    def _b64_to_image(self, b64_string: str) -> Image.Image:
        try:
            img_data_b64 = self._strip_data_url_prefix(b64_string)
            img_data = base64.b64decode(img_data_b64)
            return Image.open(io.BytesIO(img_data)).convert("L").resize((512, 288))
        except Exception as e:
            raise ValueError("Invalid image data") from e

    def _is_significant_visual_change(
        self,
        img_before: Image.Image,
        img_after: Image.Image,
    ) -> bool:
        before = np.array(img_before, dtype=np.uint8)
        after = np.array(img_after, dtype=np.uint8)

        # --- Quick reject: MSE or identical check ---
        err = np.sum(
            (before - after) ** 2,
        )
        mse = err / (img_before.size[0] * img_before.size[1])
        if mse < self.settings.mse_threshold:
            return False

        # --- Perceptual SSIM check ---
        score, diff_map = ssim(before, after, full=True)
        if score > self.settings.ssim_threshold:
            return False

        # --- Pixel-level change ratio ---
        diff_map = (1 - diff_map) * 255
        diff_map = diff_map.astype(np.uint8)
        _, mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)

        # Remove noise and fill small holes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        change_ratio = np.sum(mask > 0) / mask.size

        # --- Histogram correlation for scene-level changes ---
        hist1 = cv2.calcHist([before], [0], None, [32], [0, 256])
        hist2 = cv2.calcHist([after], [0], None, [32], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # --- Combined decision ---
        return (
            change_ratio > self.settings.change_ratio_threshold
            and hist_corr < self.settings.hist_corr_threshold
            and (score < self.settings.ssim_threshold)
        )

    async def _inactivity_flush_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.settings.inactivity_timeout_sec)
            if not self._turn_in_progress and (
                asyncio.get_event_loop().time() - self._last_activity_time
                >= self.settings.inactivity_timeout_sec
                and self._pending_vision_events
            ):
                if self._debug:
                    logger.debug(
                        "Inactivity timeout. Flushing pending vision events for silent detection.",
                    )
                async with self._state_lock:
                    visual_events = list(self._pending_vision_events)
                    self._pending_vision_events.clear()
                turn_state = TurnState(speech_events=[], visual_events=visual_events)
                await self._detect_key_moments(turn_state)

    async def _frame_processing_worker(self):
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                seq_id, event_data = await self._frame_queue.get()
                pil_img = await loop.run_in_executor(
                    self._cpu_executor,
                    self._b64_to_image,
                    event_data["payload"]["frame_b64"],
                )
                await self._results_queue.put((seq_id, event_data, pil_img))
                self._frame_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame worker: {e}", exc_info=True)

    async def _sequencer(self):
        next_seq_id, results_buffer = 1, {}
        loop = asyncio.get_running_loop()
        cooldown_period = self.settings.vision_event_cooldown_sec
        while not self._stop_event.is_set():
            try:
                seq_id, event_data, pil_img = await self._results_queue.get()
                if seq_id != next_seq_id:
                    results_buffer[seq_id] = (seq_id, event_data, pil_img)
                    continue
                self._last_activity_time = loop.time()
                ts, b64 = (
                    event_data["payload"]["timestamp"],
                    event_data["payload"]["frame_b64"],
                )
                self._frame_buffer.append((ts, b64))
                if loop.time() - self._last_vision_event_time < cooldown_period:
                    (
                        self._last_significant_frame_b64,
                        self._last_significant_frame_pil,
                    ) = (b64, pil_img)
                    next_seq_id += 1
                    continue
                if self._last_significant_frame_pil:
                    if self._is_significant_visual_change(
                        self._last_significant_frame_pil,
                        pil_img,
                    ):
                        if self._debug:
                            logger.debug(
                                f"Sequencer detected significant visual change at t={ts:.2f}s.",
                            )
                        async with self._state_lock:
                            self._pending_vision_events.append(
                                {
                                    "timestamp": ts,
                                    "before_frame_b64": self._last_significant_frame_b64,
                                    "after_frame_b64": b64,
                                },
                            )
                        self._last_vision_event_time = loop.time()
                        (
                            self._last_significant_frame_b64,
                            self._last_significant_frame_pil,
                        ) = (b64, pil_img)
                else:
                    (
                        self._last_significant_frame_b64,
                        self._last_significant_frame_pil,
                    ) = (b64, pil_img)
                next_seq_id += 1
                while next_seq_id in results_buffer:
                    _, _, _ = results_buffer.pop(next_seq_id, (None, None, None))
                    next_seq_id += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sequencer: {e}", exc_info=True)

    @llm_retry_decorator
    async def _detect_key_moments(self, turn_state: TurnState):
        if (
            not turn_state.speech_events
            and not turn_state.visual_events
            and not turn_state.latest_frame
        ):
            await self._detection_queue.put([])
            return

        burst_events_info: List[str] = []
        timestamp_to_handle_map: Dict[float, ImageHandle] = {}
        all_handles: List[ImageHandle] = []
        items_to_add = []

        if turn_state.visual_events:
            turn_state.visual_events.sort(key=lambda x: x["timestamp"])
            bursts: List[List[Dict]] = []
            if turn_state.visual_events:
                current_burst = [turn_state.visual_events[0]]
                for i in range(1, len(turn_state.visual_events)):
                    if (
                        turn_state.visual_events[i]["timestamp"]
                        - turn_state.visual_events[i - 1]["timestamp"]
                    ) <= self.settings.burst_detection_threshold_sec:
                        current_burst.append(turn_state.visual_events[i])
                    else:
                        bursts.append(current_burst)
                        current_burst = [turn_state.visual_events[i]]
                bursts.append(current_burst)

            consolidated_visual_events: List[Dict] = []
            for burst in bursts:
                if len(burst) > self.settings.visual_event_sampling_threshold:
                    final_event = burst[-1]
                    consolidated_visual_events.append(final_event)
                    burst_events_info.append(
                        f"A rapid sequence of {len(burst)} visual changes occurred, "
                        f"ending at t={final_event['timestamp']:.2f}s.",
                    )
                else:
                    consolidated_visual_events.extend(burst)

            for event in consolidated_visual_events:
                items_to_add.append(
                    {
                        "data": self._strip_data_url_prefix(event["after_frame_b64"]),
                        "auto_caption": self.settings.use_auto_captions,
                        "_timestamp": event["timestamp"],
                    },
                )

        if turn_state.latest_frame:
            ts, b64 = turn_state.latest_frame
            items_to_add.append(
                {
                    "data": self._strip_data_url_prefix(b64),
                    "auto_caption": self.settings.use_auto_captions,
                    "_timestamp": ts,
                },
            )

        if items_to_add:
            handles = self._image_manager.add_images(
                items_to_add,
                synchronous=False,
                return_handles=True,
            )
            all_handles.extend(h for h in handles if h)
            for i, handle in enumerate(handles):
                if handle:
                    ts = items_to_add[i]["_timestamp"]
                    timestamp_to_handle_map[ts] = handle

        if all_handles and self.settings.use_auto_captions:
            if self._debug:
                logger.debug(f"Awaiting {len(all_handles)} auto-caption(s)...")
            await asyncio.gather(
                *[h.wait_for_caption(timeout=5.0) for h in all_handles],
                return_exceptions=True,
            )
            if self._debug:
                logger.debug("Auto-captions resolved or timed out.")

        visual_events_info = []
        for ts, handle in timestamp_to_handle_map.items():
            caption = handle.caption
            if caption:
                visual_events_info.append(
                    f'Visual change at t={ts:.2f}s, showing: "{caption}"',
                )
            else:
                visual_events_info.append(f"Visual change at t={ts:.2f}s.")
        async with self._state_lock:
            current_summary = self._session_summary
        if self._debug:
            logger.debug(
                dedent(
                    f"""
                --- PROMPT INPUTS: build_detection_prompt ---
                - current_summary: "{current_summary}"
                - speech_events: {json.dumps(turn_state.speech_events, indent=2)}
                - visual_events_info: {json.dumps(visual_events_info, indent=2)}
                - burst_events_info: {json.dumps(burst_events_info, indent=2)}
                ---------------------------------------------
                """,
                ),
            )
        system_prompt = build_detection_prompt(
            current_summary,
            turn_state.speech_events,
            visual_events_info,
            burst_events_info,
        )
        self._detection_client.set_system_message(system_prompt)
        response_str = await self._detection_client.generate(
            user_message="Identify key moments based on the context provided.",
        )
        result = json.loads(response_str)
        key_moments = result.get("moments", [])
        final_events: List[DetectedEvent] = []
        for moment in key_moments:
            ts = moment["timestamp"]
            handle = timestamp_to_handle_map.get(ts)
            if not handle and timestamp_to_handle_map:
                closest_ts = min(
                    timestamp_to_handle_map.keys(),
                    key=lambda k: abs(k - ts),
                )
                handle = timestamp_to_handle_map[closest_ts]
                ts = closest_ts
            if handle:
                final_events.append(
                    DetectedEvent(
                        timestamp=ts,
                        detection_reason=moment.get("reason", "unknown"),
                        image_handle=handle,
                    ),
                )
        if not turn_state.speech_events:
            if self._debug:
                logger.debug(f"Detected {len(final_events)} silent visual events.")
            async with self._state_lock:
                self._stored_silent_detected_events.extend(final_events)
        else:
            await self._detection_queue.put(final_events)

    @llm_retry_decorator
    async def _get_llm_annotation_for_event(
        self,
        event_to_annotate: DetectedEvent,
        consumer_context: Optional[str],
        previous_annotations_in_turn: List[str],
    ) -> Optional[str]:
        async with self._state_lock:
            current_summary = self._session_summary
            recent_events = self._recent_key_events
        if self._debug:
            recent_events_list = [evt.model_dump() for evt in recent_events]
            logger.debug(
                dedent(
                    f"""
                --- PROMPT INPUTS: build_single_annotation_prompt ---
                - current_summary: "{current_summary}"
                - consumer_context: "{consumer_context}"
                - previous_annotations_in_turn: {json.dumps(previous_annotations_in_turn, indent=2)}
                - recent_key_events: {json.dumps(recent_events_list, indent=2)}
                -------------------------------------------------------
                """,
                ),
            )
        system_prompt = build_single_annotation_prompt(
            current_summary,
            consumer_context,
            previous_annotations_in_turn,
            recent_events,
        )
        self._analysis_client.set_system_message(system_prompt)
        raw_bytes = event_to_annotate.image_handle.raw()
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_data}"
        user_content = [
            {
                "type": "text",
                "text": f"Moment at t={event_to_annotate.timestamp:.2f}s:",
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        if self._debug:
            logger.debug(
                f"Calling annotation LLM for image at timestamp {event_to_annotate.timestamp:.2f}s...",
            )
        response = await self._analysis_client.generate(user_message=user_content)
        if isinstance(response, str) and response.strip():
            if self._debug:
                logger.debug(f"Annotation LLM returned: '{response.strip()}'")
            previous_annotations_in_turn.append(response.strip())
            return response.strip()
        return None

    def _trigger_summary_update(self):
        if self._summary_update_task and not self._summary_update_task.done():
            return
        self._summary_update_task = asyncio.create_task(self._update_summary())

    @llm_retry_decorator
    async def _update_summary(self):
        await asyncio.sleep(1.0)
        async with self._summary_update_lock:
            if not self._unsummarized_events:
                return
            async with self._state_lock:
                events = list(self._unsummarized_events)
                self._unsummarized_events.clear()
                current_summary = self._session_summary
            try:
                if self._debug:
                    events_list = [evt.model_dump() for evt in events]
                    logger.debug(
                        dedent(
                            f"""
                        --- PROMPT INPUTS: build_summary_update_prompt ---
                        - current_summary: "{current_summary}"
                        - new_events: {json.dumps(events_list, indent=2)}
                        --------------------------------------------------
                        """,
                        ),
                    )
                prompt = build_summary_update_prompt(current_summary, events)
                new_summary = await self._summary_client.generate(prompt)
                if new_summary and isinstance(new_summary, str):
                    async with self._state_lock:
                        self._session_summary = new_summary.strip()
                    if self._debug:
                        logger.debug("Session summary updated.")
            except Exception as e:
                logger.error(f"Error updating summary: {e}", exc_info=True)
                async with self._state_lock:
                    self._unsummarized_events = events + self._unsummarized_events
