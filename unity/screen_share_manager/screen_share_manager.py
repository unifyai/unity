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

import backoff
import unify
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from skimage.metrics import structural_similarity as ssim

from unity.image_manager.image_manager import ImageManager, ImageHandle
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
    min_contour_area: int = Field(
        default=100,
        description="Minimum contour area to be considered a significant semantic change.",
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


@dataclass
class TurnState:
    """Encapsulates the state for a single analysis turn."""

    speech_events: List[Dict] = field(default_factory=list)
    visual_events: List[Dict] = field(default_factory=list)
    latest_frame: Optional[Tuple[float, str]] = None


# --- Helper Decorators ---


def llm_retry_decorator(max_tries: int, base_delay: float):
    """A decorator to add exponential backoff retries to an async function."""

    def decorator(func):
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            base=base_delay,
            logger=logger,
        )
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# --- Main Manager Class ---


class ScreenShareManager:
    """
    A stateful component that analyzes screen share streams and user speech to
    detect key events and return annotated ImageHandles.
    """

    def __init__(
        self,
        settings: Optional[ScreenShareManagerSettings] = None,
        image_manager: Optional[ImageManager] = None,
        detection_client: Optional[unify.AsyncUnify] = None,
        analysis_client: Optional[unify.AsyncUnify] = None,
        summary_client: Optional[unify.AsyncUnify] = None,
    ):
        """
        Initializes the ScreenShareManager with configurable settings and injected dependencies.

        Args:
            settings: A Pydantic model containing all operational parameters.
            image_manager: An instance of ImageManager for handling image storage.
            detection_client: A Unify client for the fast detection stage.
            analysis_client: A Unify client for the rich annotation stage.
            summary_client: A Unify client for updating the session summary.
        """
        self.settings = settings or ScreenShareManagerSettings()

        self._image_manager = image_manager or ImageManager()
        self._detection_client = detection_client or unify.AsyncUnify(
            "gpt-4o-mini@openai",
        )
        self._analysis_client = analysis_client or unify.AsyncUnify("gpt-4o@openai")
        self._summary_client = summary_client or unify.AsyncUnify("gpt-4o-mini@openai")

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
        for task in tasks:
            if task and not task.done():
                task.cancel()

        # Yield control to allow cancellation to propagate
        await asyncio.sleep(0)

        self._cpu_executor.shutdown(wait=False, cancel_futures=True)

    def set_session_context(self, context_text: str):
        self._session_summary = context_text.strip()
        logger.info(f"Initial session context set: '{self._session_summary}'")

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
        logger.info(f"Received speech event: '{content}'")
        if not self._turn_in_progress:
            logger.warning("Speech event received, but no turn is in progress. Ignoring.")
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
        """
        Ends the current turn, triggers analysis of all accumulated events,
        and returns a Task to await the detection results.
        """
        if not self._turn_in_progress:
            logger.warning("end_turn called but no turn is in progress. Returning an empty task.")
            async def empty_task():
                return []
            return asyncio.create_task(empty_task())

        logger.info(
            f"Ending turn with {len(self._current_turn_speech_events)} speech event(s) and "
            f"{len(self._pending_vision_events)} potential visual event(s)."
        )

        visual_events = list(self._pending_vision_events)
        speech_events = list(self._current_turn_speech_events)

        self._pending_vision_events.clear()
        self._current_turn_speech_events.clear()
        self._turn_in_progress = False

        turn_state = TurnState(
            speech_events=speech_events,
            visual_events=visual_events,
        )

        asyncio.create_task(self._detect_key_moments(turn_state))

        return self.detect_events()


    def detect_events(self) -> asyncio.Task[List[DetectedEvent]]:
        """
        Returns a handle (a Task) to the result of the fast detection stage
        for a turn that has been initiated. The consumer must `await` this
        task to get the list of candidate events.
        """
        async def _analysis_wrapper() -> List[DetectedEvent]:
            detection_result = await self._detection_queue.get()
            if not detection_result:
                return []

            key_moments, frame_map = detection_result
            logger.debug(f"Detection result received with {len(key_moments)} moments.")

            async with self._state_lock:
                all_detected_events = self._stored_silent_detected_events
                self._stored_silent_detected_events = []

            events_to_return, images_to_add, moment_map = [], [], {}
            for i, moment in enumerate(key_moments):
                screenshot_data_url = frame_map.get(moment["timestamp"])
                if screenshot_data_url:
                    raw_b64 = self._strip_data_url_prefix(screenshot_data_url)
                    images_to_add.append(
                        {"data": raw_b64, "caption": "Detected screen event"},
                    )
                    moment_map[i] = moment

            if not images_to_add:
                return all_detected_events

            handles = await asyncio.to_thread(
                self._image_manager.add_images,
                images_to_add,
                synchronous=False,
                return_handles=True,
            )

            for i, handle in enumerate(handles):
                if handle and i in moment_map:
                    moment = moment_map[i]
                    all_detected_events.append(
                        DetectedEvent(
                            timestamp=moment["timestamp"],
                            detection_reason=moment.get("reason", "visual_change"),
                            image_handle=handle,
                        ),
                    )

            logger.info(
                f"Created/retrieved {len(all_detected_events)} DetectedEvent objects for this turn.",
            )
            return all_detected_events

        return asyncio.create_task(_analysis_wrapper())

    async def annotate_events(
        self,
        events: List[DetectedEvent],
        context: Optional[str] = None,
    ) -> List[ImageHandle]:
        if not events:
            return []
        logger.info(
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
        logger.info(
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

    def _calculate_mse(self, img1: Image.Image, img2: Image.Image) -> float:
        err = np.sum(
            (np.array(img1, dtype=np.float64) - np.array(img2, dtype=np.float64)) ** 2,
        )
        return err / (img1.size[0] * img1.size[1])

    def _is_semantically_significant(
        self,
        img_before: Image.Image,
        img_after: Image.Image,
    ) -> bool:
        diff = cv2.absdiff(np.array(img_before), np.array(img_after))
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return any(
            cv2.contourArea(c) > self.settings.min_contour_area for c in contours
        )

    async def _inactivity_flush_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.settings.inactivity_timeout_sec)
            if not self._turn_in_progress and (
                asyncio.get_event_loop().time() - self._last_activity_time
                >= self.settings.inactivity_timeout_sec
                and self._pending_vision_events
            ):
                logger.info(
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
                    if (
                        self._calculate_mse(pil_img, self._last_significant_frame_pil)
                        > self.settings.mse_threshold
                    ):
                        score = await loop.run_in_executor(
                            self._cpu_executor,
                            ssim,
                            np.array(self._last_significant_frame_pil),
                            np.array(pil_img),
                        )
                        if (
                            score < self.settings.ssim_threshold
                            and self._is_semantically_significant(
                                self._last_significant_frame_pil,
                                pil_img,
                            )
                        ):
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

    @llm_retry_decorator(max_tries=3, base_delay=1.0)
    async def _detect_key_moments(self, turn_state: TurnState):
        if not turn_state.speech_events and not turn_state.visual_events:
            return

        consolidated_visual_events, frame_map = [], {}
        burst_events_info: List[str] = []

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

            for burst in bursts:
                if len(burst) > self.settings.visual_event_sampling_threshold:
                    logger.info(
                        f"Detected a burst of {len(burst)} events. Consolidating to a single event.",
                    )
                    final_event_in_burst = burst[-1]
                    consolidated_visual_events.append(final_event_in_burst)
                    burst_info_str = (
                        f"A rapid sequence of {len(burst)} visual changes occurred, "
                        f"ending at t={final_event_in_burst['timestamp']:.2f}s. This is the final state of that action."
                    )
                    burst_events_info.append(burst_info_str)
                else:
                    consolidated_visual_events.extend(burst)
        
        if turn_state.speech_events and not consolidated_visual_events and self._frame_buffer:
            turn_state.latest_frame = self._frame_buffer[-1]

        async with self._state_lock:
            current_summary = self._session_summary
        system_prompt = build_detection_prompt(
            current_summary,
            turn_state.speech_events,
            bool(consolidated_visual_events or turn_state.latest_frame),
            burst_events_info,
        )
        self._detection_client.set_system_message(system_prompt)
        text_prompts = []
        for speech_event in turn_state.speech_events:
            text_prompts.append(
                f"User speech at t={speech_event['payload']['start_time']:.2f}s.",
            )
        for ve in consolidated_visual_events:
            text_prompts.append(f"Visual change at t={ve['timestamp']:.2f}s.")
            frame_map[ve["timestamp"]] = ve["after_frame_b64"]
        if turn_state.latest_frame:
            frame_map[turn_state.latest_frame[0]] = turn_state.latest_frame[1]
        if not text_prompts:
            await self._detection_queue.put(([], {}))
            return

        logger.debug("Calling detection LLM...")
        response_str = await self._detection_client.generate(
            user_message="Identify key moments based on the context provided.",
        )
        result = json.loads(response_str)
        key_moments = result.get("moments", [])
        key_moments.sort(key=lambda x: x["timestamp"])
        for moment in key_moments:
            ts = moment["timestamp"]
            if ts not in frame_map and self._frame_buffer:
                _, closest_frame = min(self._frame_buffer, key=lambda x: abs(x[0] - ts))
                frame_map[ts] = closest_frame
        if not turn_state.speech_events:
            logger.info(f"Detected {len(key_moments)} silent visual events.")
            images_to_add, moment_map = [], {}
            for i, moment in enumerate(key_moments):
                if frame_map.get(moment["timestamp"]):
                    images_to_add.append(
                        {
                            "data": self._strip_data_url_prefix(
                                frame_map[moment["timestamp"]],
                            ),
                            "caption": "Silent screen event",
                        },
                    )
                    moment_map[i] = moment
            if images_to_add:
                handles = await asyncio.to_thread(
                    self._image_manager.add_images,
                    images_to_add,
                    synchronous=False,
                    return_handles=True,
                )
                async with self._state_lock:
                    for i, handle in enumerate(handles):
                        if handle and i in moment_map:
                            self._stored_silent_detected_events.append(
                                DetectedEvent(
                                    timestamp=moment_map[i]["timestamp"],
                                    detection_reason=moment_map[i].get(
                                        "reason",
                                        "visual_change",
                                    ),
                                    image_handle=handle,
                                ),
                            )
        else:
            await self._detection_queue.put((key_moments, frame_map))

    @llm_retry_decorator(max_tries=3, base_delay=1.0)
    async def _get_llm_annotation_for_event(
        self,
        event_to_annotate: DetectedEvent,
        consumer_context: Optional[str],
        previous_annotations_in_turn: List[str],
    ) -> Optional[str]:
        async with self._state_lock:
            current_summary = self._session_summary
            recent_events = self._recent_key_events
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
        try:
            logger.debug(
                f"Calling annotation LLM for image at timestamp {event_to_annotate.timestamp:.2f}s...",
            )
            response = await self._analysis_client.generate(user_message=user_content)
            if isinstance(response, str) and response.strip():
                logger.debug(f"Annotation LLM returned: '{response.strip()}'")
                previous_annotations_in_turn.append(response.strip())
                return response.strip()
            return None
        except Exception as e:
            logger.error(
                f"Error during single-event LLM annotation: {e}",
                exc_info=True,
            )
            return None

    def _trigger_summary_update(self):
        if self._summary_update_task and not self._summary_update_task.done():
            return
        self._summary_update_task = asyncio.create_task(self._update_summary())

    @llm_retry_decorator(max_tries=3, base_delay=1.0)
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
                prompt = build_summary_update_prompt(current_summary, events)
                new_summary = await self._summary_client.generate(prompt)
                if new_summary and isinstance(new_summary, str):
                    async with self._state_lock:
                        self._session_summary = new_summary.strip()
                    logger.info("Session summary updated.")
            except Exception as e:
                logger.error(f"Error updating summary: {e}", exc_info=True)
                async with self._state_lock:
                    self._unsummarized_events = events + self._unsummarized_events
