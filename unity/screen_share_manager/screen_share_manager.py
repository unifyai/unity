# FILE: unity/screen_share_manager/screen_share_manager.py

import asyncio
import base64
import io
import json
import logging
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Tuple, Dict, Any

import redis.asyncio as redis
import unify
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from unity.conversation_manager_2.event_broker import get_event_broker
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import (
    AnnotatedImageRef,
    RawImageRef,
    AnnotatedImageRefs,
)

from .prompt_builders import build_summary_update_prompt, build_turn_analysis_prompt
from .types import KeyEvent, TurnAnalysisResponse

logger = logging.getLogger(__name__)


class ScreenShareManager:
    """
    A background service that analyzes screen share streams and user speech to
    detect key events and publish annotated image references.

    How It Works
    ------------
    The manager operates by listening to a Redis event stream for two main types
    of events: `app:comms:screen_frame` for video frames and
    `app:comms:phone_utterance` for user speech. It processes these events
    asynchronously to build a comprehensive narrative of the user's actions.

    1. Three-Stage Vision Event Detection:
       To accurately detect meaningful UI changes while filtering out noise like
       cursor movements or minor animations, a three-stage pipeline is used for
       every new frame:
       a. MSE Pre-filter: A cheap Mean Squared Error check on downsampled,
          grayscale images quickly discards frames with no significant pixel change.
       b. SSIM Perceptual Check: If MSE detects a change, a more accurate but
          CPU-intensive Structural Similarity Index (SSIM) check is performed
          to verify the change is perceptually significant.
       c. Semantic Contour Analysis: If SSIM also confirms a difference, a final
          semantic filter is applied. This uses OpenCV to find the contours
          (outlines) of all changed regions and filters them based on size and
          shape. This step effectively ignores noise like thin scrollbars or
          small cursors. A "vision event" is created only if a change passes
          all three stages.

    2. Turn Analysis and Annotation:
       When a user turn occurs (triggered by speech or a silent visual action),
       the manager gathers context: the visual frames, user speech transcript,
       and a rolling session summary. This package is sent to a vision-capable
       LLM, which identifies key moments. For each moment, the LLM generates a
       single, context-rich `image_annotation` that explains why the
       associated screenshot is important.

    3. Output Generation:
       The manager's final output for each turn is an `AnnotatedImageRefs` object.
       This object pairs each AI-generated `image_annotation` with the permanent
       ID of its corresponding screenshot (obtained from the `ImageManager`). This
       structured data is then published to the `app:comms:screen_analysis_result`
       Redis channel for other services to consume.

    Event Processing Scenarios
    --------------------------
    The manager handles three primary scenarios to ensure all actions are captured:

    - Scenario 1: Speech and Vision Events Occur Together
      When a user speaks while interacting with the screen (e.g., "I'll click
      this button"), the utterance triggers an analysis that includes both the
      speech content and the visual evidence of the click. The LLM receives
      this combined context and generates an annotation like: "The confirmation
      dialog is now visible, which appeared after the user stated their
      intention to click the button."

    - Scenario 2: Only Speech Events Occur
      If the user speaks without a corresponding visual change, the utterance
      still triggers an analysis. The LLM processes the speech against the
      *current* screen view. The resulting annotation describes the screen state
      as it relates to the user's speech, for example: "The user is expressing
      confusion while viewing the main dashboard."

    - Scenario 3: Only Vision Events Occur (Silent Actions)
      If the user performs an action without speaking (e.g., clicking a link),
      the visual change is detected and stored as a pending event. If no speech
      occurs within an inactivity timeout, the manager "flushes" this event for
      analysis. The resulting `KeyEvent` is temporarily stored. It is then
      merged with the events from the *next* user utterance, ensuring the
      context of the silent action is not lost and is published alongside the

    Architecture for Parallelism (Sequencer Pattern)
    ------------------------------------------------
    To handle high-frequency frame streams, the manager uses a sequencer pattern:
    1.  **Producer (`_listen_for_events`)**: Receives frames, tags them with a
        sequence ID, and places them on a queue. It adaptively drops frames
        if the processing pipeline is backlogged.
    2.  **Concurrent Workers (`_frame_processing_worker`)**: A pool of workers
        performs stateless, CPU-intensive image analysis in parallel.
    3.  **Sequencer (`_sequencer`)**: A single task processes results in strict
        chronological order, updating the shared screen state and detecting
        vision events without race conditions.
    """

    def __init__(self, include_raw_image_data_in_result: bool = False):
        # Configuration
        self.MSE_THRESHOLD = 10
        self.SSIM_THRESHOLD = 0.995
        self.MIN_CONTOUR_AREA = 50
        self.MAX_ASPECT_RATIO = 20
        self.DEBOUNCE_DELAY_SEC = 0.5
        self.INACTIVITY_TIMEOUT_SEC = 5.0
        self.FRAME_BUFFER_SIZE = 100
        self.MAX_FRAME_WORKERS = os.cpu_count() or 4
        self.FRAME_QUEUE_SIZE = 150
        self.RESULTS_QUEUE_SIZE = 200
        self.OUTPUT_QUEUE_SIZE = 50
        self.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
        self.BURST_DETECTION_THRESHOLD_SEC = 1.0
        self.MAX_CONCURRENT_OUTPUT_TASKS = 5
        self.IMAGE_UPLOAD_MAX_RETRIES = 3
        self.IMAGE_UPLOAD_INITIAL_BACKOFF = 1.0
        self.ADAPTIVE_DROP_THRESHOLD = 0.75
        self.include_raw_image_data_in_result = include_raw_image_data_in_result

        # Clients and Managers
        self._event_broker: redis.Redis = get_event_broker()
        self._analysis_client = unify.AsyncUnify(
            "gpt-4o@openai",
            response_format=TurnAnalysisResponse,
        )
        self._summary_client = unify.AsyncUnify("gpt-4o-mini@openai")
        self._image_manager = ImageManager()
        self._cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

        # State Variables
        self._stop_event = asyncio.Event()
        self._frame_sequence_id = 0
        self._frame_queue = asyncio.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._results_queue = asyncio.Queue(maxsize=self.RESULTS_QUEUE_SIZE)
        self._frame_workers: List[asyncio.Task] = []
        self._sequencer_task: Optional[asyncio.Task] = None
        self._frame_buffer: Deque[Tuple[float, str]] = deque(
            maxlen=self.FRAME_BUFFER_SIZE,
        )
        self._pending_vision_events: List[Dict] = []
        self._stored_silent_key_events: List[KeyEvent] = []
        self._stored_silent_frame_map: Dict[float, str] = {}
        self._last_significant_frame_b64: Optional[str] = None
        self._last_significant_frame_pil: Optional[Image.Image] = None
        self._last_activity_time: float = asyncio.get_event_loop().time()
        self._state_lock = asyncio.Lock()
        self._analyses_in_flight = 0
        self._output_queue = asyncio.Queue(maxsize=self.OUTPUT_QUEUE_SIZE)
        self._output_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_OUTPUT_TASKS)
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None

        # State for rolling summary and recent events
        self._session_summary: str = "The session has just begun."
        self._recent_key_events: Deque[KeyEvent] = deque(maxlen=5)
        self._unsummarized_events: List[KeyEvent] = []
        self._summary_update_lock = asyncio.Lock()
        self._summary_update_task: Optional[asyncio.Task] = None

    def set_session_context(self, context_text: str):
        """
        Sets the initial context for the screen share session. This should be
        called before any events are processed to prime the analysis LLM.
        """
        if not context_text or not isinstance(context_text, str):
            logger.warning("Invalid session context provided. Ignoring.")
            return

        logger.info(f"Setting initial session context to: '{context_text}'")

        async def update_summary():
            async with self._state_lock:
                self._session_summary = context_text.strip()

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(update_summary())
            else:
                loop.run_until_complete(update_summary())
        except RuntimeError:
            asyncio.run(update_summary())

    async def start(self):
        """Starts the main event listening and the output dispatcher."""
        logger.info("ScreenShareManager started. Initializing background tasks...")
        self._dispatcher_task = asyncio.create_task(self._output_dispatcher())
        self._sequencer_task = asyncio.create_task(self._sequencer())
        self._frame_workers = [
            asyncio.create_task(self._frame_processing_worker())
            for _ in range(self.MAX_FRAME_WORKERS)
        ]
        await self._listen_for_events()

    def stop(self):
        """Signals the manager to gracefully shut down."""
        self._stop_event.set()
        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
        if self._sequencer_task and not self._sequencer_task.done():
            self._sequencer_task.cancel()
        for worker in self._frame_workers:
            if not worker.done():
                worker.cancel()
        if self._summary_update_task and not self._summary_update_task.done():
            self._summary_update_task.cancel()
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._cpu_executor.shutdown(wait=False)
        logger.info("ScreenShareManager stopping...")

    def _b64_to_image(self, b64_string: str) -> Image.Image:
        """Converts a base64 data URL to a resized, grayscale PIL Image."""
        try:
            img_data = base64.b64decode(b64_string.split(",")[1])
            img = Image.open(io.BytesIO(img_data)).convert("L").resize((512, 288))
            return img
        except Exception as e:
            logger.warning(
                "Failed to decode or process base64 image string.", exc_info=True
            )
            raise ValueError("Invalid base64 image data") from e

    def _calculate_mse(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculates the Mean Squared Error between two images."""
        err = np.sum(
            (np.array(img1).astype("float") - np.array(img2).astype("float")) ** 2
        )
        err /= float(img1.size[0] * img1.size[1])
        return err

    def _is_semantically_significant(
        self, img_before: Image.Image, img_after: Image.Image
    ) -> bool:
        """
        Performs a semantic check on the visual change by analyzing the contours
        of the difference between two frames. It filters out noise like cursors
        or small, irrelevant artifacts.
        """
        cv_before, cv_after = np.array(img_before), np.array(img_after)
        diff = cv2.absdiff(cv_before, cv_after)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            aspect_ratio = float(w) / h if h > w else float(h) / w
            if aspect_ratio > self.MAX_ASPECT_RATIO:
                continue
            return True
        return False

    async def _listen_for_events(self):
        """The core loop that subscribes to Redis and dispatches events."""
        async with self._event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:screen_frame",
                "app:comms:phone_utterance",
            )
            logger.info("Subscribed to event channels. Listening for events...")
            while not self._stop_event.is_set():
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if message:
                        channel = message["channel"]
                        event_data = json.loads(message["data"])
                        if channel == "app:comms:screen_frame":
                            if self._frame_queue.qsize() > (
                                self.FRAME_QUEUE_SIZE * self.ADAPTIVE_DROP_THRESHOLD
                            ):
                                logger.info(
                                    "Frame queue backlogged. Proactively dropping frame."
                                )
                                continue
                            try:
                                self._frame_sequence_id += 1
                                self._frame_queue.put_nowait(
                                    (self._frame_sequence_id, event_data)
                                )
                            except asyncio.QueueFull:
                                logger.warning(
                                    "Frame queue is full. Dropping incoming frame to maintain stability."
                                )
                        elif channel == "app:comms:phone_utterance":
                            self._trigger_turn_analysis(speech_event=event_data)
                    await self._flush_pending_events_on_timeout()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in event listener loop: {e}", exc_info=True)
                    await asyncio.sleep(1)

    async def _frame_processing_worker(self):
        """
        Pulls a frame from the queue, performs CPU-intensive analysis, and
        places the result onto the results queue for sequencing. This worker
        is stateless.
        """
        logger.info("Frame processing worker started.")
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                seq_id, event_data = await self._frame_queue.get()
                frame_b64 = event_data["payload"]["frame_b64"]
                current_img_pil = await loop.run_in_executor(
                    self._cpu_executor, self._b64_to_image, frame_b64
                )
                await self._results_queue.put((seq_id, event_data, current_img_pil))
                self._frame_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Frame processing worker cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in frame processing worker: {e}", exc_info=True)
        logger.info("Frame processing worker stopped.")

    async def _sequencer(self):
        """
        Processes analysis results in strict order to prevent race conditions.
        This is the only task that modifies `_last_significant_frame_b64` and
        `_pending_vision_events`.
        """
        logger.info("Sequencer task started.")
        next_seq_id = 1
        results_buffer: Dict[int, Any] = {}
        loop = asyncio.get_running_loop()

        while not self._stop_event.is_set():
            try:
                if next_seq_id in results_buffer:
                    seq_id, event_data, current_img_pil = results_buffer.pop(
                        next_seq_id
                    )
                else:
                    (
                        seq_id,
                        event_data,
                        current_img_pil,
                    ) = await self._results_queue.get()

                if seq_id != next_seq_id:
                    results_buffer[seq_id] = (seq_id, event_data, current_img_pil)
                    continue

                self._last_activity_time = asyncio.get_event_loop().time()
                timestamp = event_data["payload"]["timestamp"]
                frame_b64 = event_data["payload"]["frame_b64"]
                self._frame_buffer.append((timestamp, frame_b64))

                if self._last_significant_frame_b64 is None:
                    self._last_significant_frame_b64 = frame_b64
                    self._last_significant_frame_pil = current_img_pil
                else:
                    mse = self._calculate_mse(
                        current_img_pil, self._last_significant_frame_pil
                    )
                    if mse > self.MSE_THRESHOLD:
                        score = await loop.run_in_executor(
                            self._cpu_executor,
                            ssim,
                            np.array(self._last_significant_frame_pil),
                            np.array(current_img_pil),
                        )
                        if score < self.SSIM_THRESHOLD:
                            is_significant = await loop.run_in_executor(
                                self._cpu_executor,
                                self._is_semantically_significant,
                                self._last_significant_frame_pil,
                                current_img_pil,
                            )
                            if is_significant:
                                logger.info(
                                    f"Sequencer: Significant visual event detected at t={timestamp:.2f}s"
                                )
                                self._pending_vision_events.append(
                                    {
                                        "timestamp": timestamp,
                                        "before_frame_b64": self._last_significant_frame_b64,
                                        "after_frame_b64": frame_b64,
                                    }
                                )
                                self._last_significant_frame_b64 = frame_b64
                                self._last_significant_frame_pil = current_img_pil

                next_seq_id += 1
            except asyncio.CancelledError:
                logger.info("Sequencer task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in sequencer: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("Sequencer task stopped.")

    async def _output_dispatcher(self):
        """
        Waits for jobs in the queue, acquires a semaphore, and spawns
        a one-shot worker task for each job to handle output generation.
        """
        logger.info("Output dispatcher started.")
        while not self._stop_event.is_set():
            try:
                output_job = await self._output_queue.get()
                await self._output_semaphore.acquire()
                asyncio.create_task(self._output_worker(output_job))
                self._output_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Output dispatcher received cancellation request.")
                break
            except Exception as e:
                logger.error(f"Error in output dispatcher: {e}", exc_info=True)
        logger.info("Output dispatcher stopped.")

    async def _output_worker(self, output_job: tuple):
        """
        Processes a single output job: uploads images, constructs the final
        AnnotatedImageRefs payload, and publishes it to Redis.
        """
        try:
            speech_event, key_events, frame_map = output_job
            await self._build_and_publish_analysis_result(
                speech_event, key_events, frame_map
            )
        except Exception as e:
            logger.error(f"Error in output worker: {e}", exc_info=True)
        finally:
            self._output_semaphore.release()

    def _trigger_summary_update(self):
        """Schedules the summary update task, ensuring only one is running or scheduled (debouncing)."""
        if self._summary_update_task and not self._summary_update_task.done():
            logger.debug("Summary update task already scheduled. Skipping.")
            return
        logger.info("Scheduling session summary update.")
        self._summary_update_task = asyncio.create_task(self._update_summary())

    async def _update_summary(self):
        """A serialized task that updates the session summary using the latest events."""
        await asyncio.sleep(3.0)
        async with self._summary_update_lock:
            events_to_summarize, current_summary = [], ""
            async with self._state_lock:
                if not self._unsummarized_events:
                    logger.debug("No new events to summarize.")
                    return
                events_to_summarize = list(self._unsummarized_events)
                self._unsummarized_events.clear()
                current_summary = self._session_summary
            logger.info(
                f"Updating summary with {len(events_to_summarize)} new event(s)."
            )
            try:
                prompt = build_summary_update_prompt(
                    current_summary, events_to_summarize
                )
                new_summary = await self._summary_client.generate(prompt)
                if new_summary and isinstance(new_summary, str):
                    async with self._state_lock:
                        self._session_summary = new_summary.strip()
                    logger.info("Session summary updated successfully.")
                    logger.debug(f"New Summary: {self._session_summary}")
                else:
                    logger.warning(
                        "Summary update LLM call did not return a valid string."
                    )
            except asyncio.CancelledError:
                logger.info("Summary update task cancelled.")
                async with self._state_lock:
                    self._unsummarized_events = (
                        events_to_summarize + self._unsummarized_events
                    )
                raise
            except Exception as e:
                logger.error(f"Error during summary update: {e}", exc_info=True)
                async with self._state_lock:
                    self._unsummarized_events = (
                        events_to_summarize + self._unsummarized_events
                    )

    def _trigger_turn_analysis(self, speech_event: Optional[dict]):
        """
        Schedules a debounced turn analysis, cancelling any previously scheduled one.
        """
        self._last_activity_time = asyncio.get_event_loop().time()
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        logger.info(f"Debouncing turn analysis for {self.DEBOUNCE_DELAY_SEC}s...")
        self._debounce_task = asyncio.create_task(
            self._debounced_analysis_runner(speech_event)
        )

    async def _debounced_analysis_runner(self, speech_event: Optional[dict]):
        """Waits for the debounce delay then runs the analysis."""
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY_SEC)
            visual_events_for_turn, latest_frame_for_turn = [], None

            if self._pending_vision_events:
                visual_events_for_turn = list(self._pending_vision_events)
                self._pending_vision_events.clear()

            if speech_event and not visual_events_for_turn and self._frame_buffer:
                latest_frame_for_turn = self._frame_buffer[-1]

            if not speech_event and not visual_events_for_turn:
                return
            logger.info("Debounce window ended. Triggering turn analysis...")
            await self._analyze_turn(
                speech_event, visual_events_for_turn, latest_frame_for_turn
            )
        except asyncio.CancelledError:
            logger.info("Debounced analysis was cancelled by a newer event.")

    async def _flush_pending_events_on_timeout(self):
        """
        On inactivity, triggers a silent turn analysis, but only if no other
        analyses are in progress.
        """
        time_since_activity = asyncio.get_event_loop().time() - self._last_activity_time
        is_debouncing = self._debounce_task and not self._debounce_task.done()
        if (
            time_since_activity > self.INACTIVITY_TIMEOUT_SEC
            and self._analyses_in_flight == 0
            and not is_debouncing
            and self._pending_vision_events
        ):
            logger.info("Inactivity timeout reached. Flushing pending vision events.")
            self._trigger_turn_analysis(speech_event=None)

    async def _analyze_turn(
        self,
        speech_event: Optional[dict],
        visual_events: List[Dict],
        latest_frame: Optional[Tuple[float, str]] = None,
    ):
        """
        Analyzes a turn. If it's a silent (vision-only) turn, it stores the
        resulting events. If it's a speech turn, it queues the results for output.
        """
        async with self._state_lock:
            self._analyses_in_flight += 1
        try:
            response, frame_map = await self._get_llm_analysis(
                speech_event, visual_events, latest_frame
            )
            key_events = response.events if response else []

            if not speech_event:
                if key_events:
                    logger.info(f"Storing {len(key_events)} silent visual event(s).")
                    valid_timestamps = {
                        evt.representative_timestamp for evt in key_events
                    }
                    filtered_frame_map = {
                        ts: frame
                        for ts, frame in frame_map.items()
                        if ts in valid_timestamps
                    }
                    async with self._state_lock:
                        self._stored_silent_key_events.extend(key_events)
                        self._stored_silent_frame_map.update(filtered_frame_map)
                return

            if key_events:
                async with self._state_lock:
                    for event in key_events:
                        self._recent_key_events.append(event)
                        self._unsummarized_events.append(event)
                self._trigger_summary_update()

            output_job = (speech_event, key_events, frame_map)
            try:
                self._output_queue.put_nowait(output_job)
            except asyncio.QueueFull:
                logger.error("Output queue is full. Dropping latest analysis result.")

            for event in key_events:
                await self._event_broker.publish(
                    "app:comms:screen_annotation",
                    json.dumps(
                        {
                            "event_name": "ScreenAnnotationEvent",
                            "payload": {"annotation": event.image_annotation},
                        }
                    ),
                )
        except Exception as e:
            logger.error(
                f"An unhandled exception occurred during turn analysis: {e}",
                exc_info=True,
            )
            if speech_event:
                try:
                    self._output_queue.put_nowait((speech_event, [], {}))
                except asyncio.QueueFull:
                    logger.error("Output queue full. Dropping error-fallback result.")
        finally:
            async with self._state_lock:
                self._analyses_in_flight -= 1

    async def _get_llm_analysis(
        self,
        speech_event: Optional[dict],
        visual_events: List[Dict],
        latest_frame: Optional[Tuple[float, str]] = None,
    ) -> Tuple[Optional[TurnAnalysisResponse], Dict[float, str]]:
        """Constructs the prompt and calls the LLM to get turn analysis."""
        async with self._state_lock:
            current_summary = self._session_summary
            recent_key_events_copy = self._recent_key_events.copy()
        system_prompt = build_turn_analysis_prompt(
            current_summary, recent_key_events_copy
        )
        user_content, timestamp_to_frame_map = [], {}
        if speech_event:
            payload = speech_event["payload"]
            user_content.append(
                {"type": "text", "text": f"User Speech: \"{payload['content']}\""}
            )
            if "start_time" in payload and "end_time" in payload:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Speech Timestamps: Start={payload['start_time']:.2f}s, End={payload['end_time']:.2f}s",
                    }
                )
        if visual_events:
            user_content.append({"type": "text", "text": "\n--- Key Visual Frames ---"})
            bursts: List[List[Dict]] = []
            if visual_events:
                current_burst = [visual_events[0]]
                for i in range(1, len(visual_events)):
                    if (
                        visual_events[i]["timestamp"]
                        - visual_events[i - 1]["timestamp"]
                    ) <= self.BURST_DETECTION_THRESHOLD_SEC:
                        current_burst.append(visual_events[i])
                    else:
                        bursts.append(current_burst)
                        current_burst = [visual_events[i]]
                bursts.append(current_burst)
            frame_counter = 0
            for burst in bursts:
                events_to_process = burst
                if len(burst) > self.VISUAL_EVENT_SAMPLING_THRESHOLD:
                    logger.info(
                        f"Detected a burst of {len(burst)} events. Sampling down to 3."
                    )
                    user_content.append(
                        {
                            "type": "text",
                            "text": "\nNOTE: The following frames are a sampled summary (first, middle, last) of a rapid sequence of screen changes.",
                        }
                    )
                    events_to_process = [burst[0], burst[len(burst) // 2], burst[-1]]
                for ve in events_to_process:
                    frame_counter += 1
                    timestamp_to_frame_map[ve["timestamp"]] = ve["after_frame_b64"]
                    user_content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"\nVisual Change #{frame_counter} at t={ve['timestamp']:.2f}s:",
                            },
                            {"type": "text", "text": "BEFORE:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": ve["before_frame_b64"]},
                            },
                            {"type": "text", "text": "AFTER:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": ve["after_frame_b64"]},
                            },
                        ]
                    )
        elif latest_frame:
            ts, b64 = latest_frame
            timestamp_to_frame_map[ts] = b64
            user_content.append(
                {"type": "text", "text": "\n--- Current Screen View ---"}
            )
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"This is the screen view at t={ts:.2f}s, when the user was speaking:",
                    },
                    {"type": "image_url", "image_url": {"url": b64}},
                ]
            )
        if not user_content:
            return None, {}
        try:
            self._analysis_client.set_system_message(system_prompt)
            response = await self._analysis_client.generate(user_message=user_content)
            if isinstance(response, TurnAnalysisResponse):
                return response, timestamp_to_frame_map
            elif isinstance(response, str):
                logger.warning(
                    "LLM analysis returned raw string. Attempting manual parse."
                )
                try:
                    return (
                        TurnAnalysisResponse.model_validate_json(response),
                        timestamp_to_frame_map,
                    )
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(
                        f"Failed to manually parse LLM string response: {e}",
                        exc_info=True,
                    )
                    return None, {}
            else:
                logger.error(
                    f"Unexpected response type from LLM analysis: {type(response)}"
                )
                return None, {}
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            return None, {}

    async def _build_and_publish_analysis_result(
        self,
        speech_event: Dict,
        key_events: List[KeyEvent],
        frame_map: Dict[float, str],
    ):
        """
        Handles image uploads, constructs AnnotatedImageRefs, and publishes the
        final result to Redis.
        """
        async with self._state_lock:
            all_events = sorted(
                self._stored_silent_key_events + key_events, key=lambda e: e.timestamp
            )
            self._stored_silent_key_events.clear()
            combined_frame_map = self._stored_silent_frame_map.copy()
            self._stored_silent_frame_map.clear()

        combined_frame_map.update(frame_map)
        images_to_add, event_to_image_map = [], {}
        for event in all_events:
            rep_ts = event.representative_timestamp
            screenshot_b64 = combined_frame_map.get(rep_ts)
            if not screenshot_b64 and combined_frame_map:
                closest_ts = min(
                    combined_frame_map.keys(), key=lambda ts: abs(ts - rep_ts)
                )
                if abs(closest_ts - rep_ts) < 1.0:
                    screenshot_b64 = combined_frame_map[closest_ts]
            if screenshot_b64:
                images_to_add.append(
                    {"data": screenshot_b64, "caption": event.image_annotation}
                )
                event_to_image_map[event.timestamp] = len(images_to_add) - 1

        logged_image_ids = []
        if images_to_add:
            for attempt in range(self.IMAGE_UPLOAD_MAX_RETRIES):
                try:
                    # Use synchronous=True to get back image IDs immediately
                    logged_image_ids = await asyncio.to_thread(
                        self._image_manager.add_images, images_to_add, synchronous=True
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"Image upload failed on attempt {attempt + 1}/{self.IMAGE_UPLOAD_MAX_RETRIES}: {e}"
                    )
                    if attempt + 1 == self.IMAGE_UPLOAD_MAX_RETRIES:
                        logger.error("Image upload failed after all retries.")
                        return
                    await asyncio.sleep(
                        self.IMAGE_UPLOAD_INITIAL_BACKOFF * (2**attempt)
                        + random.uniform(0, 1)
                    )

        timestamp_to_image_id: Dict[float, int] = {}
        for ts, index in event_to_image_map.items():
            if index < len(logged_image_ids):
                image_id = logged_image_ids[index]
                if image_id is not None:
                    timestamp_to_image_id[ts] = image_id

        image_refs_list = []
        for event in all_events:
            image_id = timestamp_to_image_id.get(event.timestamp)
            if image_id is not None:
                image_refs_list.append(
                    AnnotatedImageRef(
                        raw_image_ref=RawImageRef(image_id=image_id),
                        annotation=event.image_annotation,
                    )
                )

        final_images = AnnotatedImageRefs.model_validate(image_refs_list)

        # Prepare the final event payload
        result_payload = {
            "speech_event": speech_event,
            "key_events": [ke.model_dump() for ke in all_events],
            "images": final_images.model_dump(),
        }

        # Optionally include raw image data for consumers like the sandbox
        if self.include_raw_image_data_in_result:
            image_data_map = {
                str(logged_image_ids[i]): images_to_add[i]["data"]
                for i in range(len(logged_image_ids))
                if logged_image_ids[i] is not None
            }
            result_payload["image_data_map"] = image_data_map

        final_event = {
            "event_name": "ScreenAnalysisResult",
            "payload": result_payload,
        }

        await self._event_broker.publish(
            "app:comms:screen_analysis_result", json.dumps(final_event)
        )
        logger.info(
            f"Published screen analysis result with {len(image_refs_list)} annotated image(s)."
        )
