import asyncio
import base64
import io
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple, Dict

import redis.asyncio as redis
import unify
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

from unity.conversation_manager_2.event_broker import get_event_broker
from unity.image_manager.image_manager import ImageManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import (
    Medium,
    Message,
    ScreenShareAnnotation,
)

from .prompt_builders import build_summary_update_prompt, build_turn_analysis_prompt
from .types import KeyEvent, TurnAnalysisResponse

logger = logging.getLogger(__name__)


class ScreenShareManager:
    """
    A background service that analyzes screen share streams and user speech to
    detect and annotate key events, feeding context to both the live
    ConversationManager and the historical TranscriptManager.

    How It Works
    ------------
    The manager operates by listening to a Redis event stream for two main types
    of events: `app:comms:screen_frame` for video frames and
    `app:comms:phone_utterance` for user speech. It processes these events
    asynchronously to build a comprehensive narrative of the user's actions.

    1. Vision Event Detection (SSIM):
       - The manager maintains a buffer of recent frames.
       - Each new frame is compared against the last "significant" frame using
         the Structural Similarity Index (SSIM).
       - If the SSIM score falls below a threshold (SSIM_THRESHOLD), it signifies
         a meaningful visual change on the screen.
       - A "vision event" is created, capturing the timestamp and the frames
         immediately before and after the change. These are stored in a pending queue.

    2. Speech Event Handling:
       - When a user speaks, a `phone_utterance` event is received.
       - This event immediately triggers a full "turn analysis," which gathers the
         user's speech and any pending vision events that occurred recently.

    3. Decoupled Architecture (Analysis vs. Logging):
       - To maintain responsiveness, the manager separates fast analysis from slow logging.
       - The `_analyze_turn` method, protected by a lock, performs the quick task
         of gathering context and calling the LLM.
       - As soon as the LLM responds, `_analyze_turn` places the results into a
         dedicated `_logging_queue` and immediately releases its lock.
       - A separate, long-running `_logging_worker` task continuously pulls from
         this queue and handles the slow, I/O-bound operations of batch-uploading
         images to the backend and persisting the final transcript message.
       - This ensures the manager is always ready to process new screen or speech
         events without getting blocked by network latency from previous events.

    Event Processing Scenarios
    --------------------------
    The manager is designed to handle three primary scenarios to ensure all
    actions are captured accurately:

    - Scenario 1: Speech and Vision Events Occur Together
      When a user speaks while interacting with the screen (e.g., "I'll click
      this button"), the utterance triggers an analysis that includes both the
      speech content and the visual evidence of the click. The Language Model (LLM)
      receives this combined context and generates a rich `KeyEvent` that links
      the action to the speech.

    - Scenario 2: Only Speech Events Occur
      If the user speaks without performing a visual action, the utterance still
      triggers an analysis. The LLM processes the speech to identify the user's
      intent, and this is logged to the transcript without a corresponding visual.

    - Scenario 3: Only Vision Events Occur (Silent Actions)
      If the user performs an action without speaking (e.g., clicking a link),
      the visual change is detected and stored as a pending vision event. If no
      speech occurs within an inactivity timeout (INACTIVITY_TIMEOUT_SEC), the
      manager "flushes" these pending events, sending them to the LLM for analysis.
      The resulting `KeyEvents` are temporarily stored in `_stored_silent_key_events`.
      They are then merged with the events from the *next* speech utterance,
      ensuring that the context of the silent action is not lost and is logged
      alongside the user's subsequent thoughts.

    Semantic Association (`[x:y]` Notation)
    ---------------------------------------
    A key feature is the ability to semantically link a visual event (a screenshot)
    to the exact words the user spoke. This is achieved via the `triggering_phrase`
    identified by the LLM.

    - The LLM is prompted to find the specific text span in the user's speech
      that corresponds to a visual action (e.g., for a click on a "Submit" button,
      the triggering phrase might be "click this").
    - In the `_log_turn_to_transcript` method, if a `KeyEvent` contains this
      `triggering_phrase`, the code searches for the phrase in the full speech
      content to find its start and end character indices.
    - It then creates a mapping in the format `{'[start:end]': image_id}`.
    - This mapping is stored in the `images` field of the `Message` object,
      creating a durable, precise link between the user's words and their actions.
      This association is only created when speech and vision events are analyzed
      together.

    Architecture for Parallelism
    ----------------------------
    This manager uses a highly concurrent, dynamic architecture to ensure
    responsiveness under variable load.

    1. State Protection with a Narrow Lock:
       - A lock (`_state_lock`) protects shared lists that accumulate events
         between turns. It is held only for the brief moment required to
         atomically capture the current state.

    2. Lock-Free Parallel Analysis:
       - The `_analyze_turn` method is stateless, allowing multiple analysis
         tasks (slow LLM calls) to run concurrently without interference.

    3. Dynamic Logging Worker Pool with a Semaphore:
       - Instead of a fixed pool, workers are spawned on-demand.
       - An `asyncio.Semaphore` limits the number of concurrent logging tasks
         to a configured maximum (`MAX_CONCURRENT_LOGGING_TASKS`), preventing
         system overload.
       - A single `_logging_dispatcher` task waits for results from the analysis
         phase. When a job arrives, it acquires the semaphore and spawns a
         one-shot `_logging_worker` task to handle the slow I/O.
       - This is highly resource-efficient, as no workers are active during
         idle periods, but the system can instantly scale up to handle bursts.
    """

    def __init__(self):
        # Configuration
        self.SSIM_THRESHOLD = 0.80
        self.INACTIVITY_TIMEOUT_SEC = 10.0
        self.FRAME_BUFFER_SIZE = 100
        self.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
        self.BURST_DETECTION_THRESHOLD_SEC = 2.0
        self.MAX_CONCURRENT_LOGGING_TASKS = 5

        # Clients and Managers
        self._event_broker: redis.Redis = get_event_broker()
        self._openai_client = unify.AsyncUnify(
            "gpt-4o@openai",
            response_format=TurnAnalysisResponse,
        )
        self._summary_client = unify.AsyncUnify("gpt-4o-mini@openai")
        self._image_manager = ImageManager()
        self._transcript_manager = TranscriptManager()

        # State Variables
        self._stop_event = asyncio.Event()
        self._frame_buffer: Deque[Tuple[float, str]] = deque(
            maxlen=self.FRAME_BUFFER_SIZE,
        )
        self._pending_vision_events: List[Dict] = []
        self._stored_silent_key_events: List[KeyEvent] = []
        self._stored_silent_frame_map: Dict[float, str] = {}
        self._last_significant_frame_b64: Optional[str] = None
        self._last_user_utterance_message_id: Optional[int] = None
        self._last_activity_time: float = asyncio.get_event_loop().time()
        self._state_lock = asyncio.Lock()
        self._analyses_in_flight = 0
        self._logging_queue = asyncio.Queue()
        self._logging_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LOGGING_TASKS)
        self._dispatcher_task: Optional[asyncio.Task] = None

        # State for rolling summary and recent events
        self._session_summary: str = "The session has just begun."
        self._recent_key_events: Deque[KeyEvent] = deque(maxlen=5)
        self._unsummarized_events: List[KeyEvent] = []
        self._summary_update_lock = asyncio.Lock()
        self._summary_update_task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the main event listening and the logging dispatcher."""
        logger.info("ScreenShareManager started. Initializing background tasks...")
        self._dispatcher_task = asyncio.create_task(self._logging_dispatcher())
        await self._listen_for_events()

    def stop(self):
        """Signals the manager to gracefully shut down."""
        self._stop_event.set()
        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
        if self._summary_update_task and not self._summary_update_task.done():
            self._summary_update_task.cancel()
        logger.info("ScreenShareManager stopping...")

    def _b64_to_image(self, b64_string: str) -> Image.Image:
        """Converts a base64 data URL to a resized, grayscale PIL Image."""
        try:
            img_data = base64.b64decode(b64_string.split(",")[1])
            img = Image.open(io.BytesIO(img_data)).convert("L").resize((512, 288))
            return img
        except Exception:
            logger.warning("Failed to decode or process base64 image string.")
            return None

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
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message:
                        channel = message["channel"]
                        event_data = json.loads(message["data"])
                        if channel == "app:comms:screen_frame":
                            asyncio.create_task(self._handle_frame_event(event_data))
                        elif channel == "app:comms:phone_utterance":
                            await self._trigger_turn_analysis(speech_event=event_data)

                    await self._flush_pending_events_on_timeout()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in event listener loop: {e}", exc_info=True)
                    await asyncio.sleep(1)

    async def _logging_dispatcher(self):
        """
        Waits for jobs in the queue, acquires the semaphore, and spawns
        a one-shot worker task for each job.
        """
        logger.info("Logging dispatcher started.")
        while not self._stop_event.is_set():
            try:
                log_job = await self._logging_queue.get()
                await self._logging_semaphore.acquire()
                asyncio.create_task(self._logging_worker(log_job))
                self._logging_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Logging dispatcher received cancellation request.")
                break
            except Exception as e:
                logger.error(f"Error in logging dispatcher: {e}", exc_info=True)
        logger.info("Logging dispatcher stopped.")

    async def _logging_worker(self, log_job: tuple):
        """
        Processes a single log job and releases the semaphore upon completion.
        """
        try:
            speech_event, key_events, frame_map = log_job
            await self._log_turn_to_transcript(speech_event, key_events, frame_map)
        except Exception as e:
            logger.error(f"Error in logging worker: {e}", exc_info=True)
        finally:
            self._logging_semaphore.release()

    async def _handle_frame_event(self, event_data: dict):
        """Processes a single video frame event."""
        self._last_activity_time = asyncio.get_event_loop().time()
        timestamp = event_data["payload"]["timestamp"]
        frame_b64 = event_data["payload"]["frame_b64"]
        self._frame_buffer.append((timestamp, frame_b64))

        if self._last_significant_frame_b64 is None:
            self._last_significant_frame_b64 = frame_b64
            return

        current_img = self._b64_to_image(frame_b64)
        last_img = self._b64_to_image(self._last_significant_frame_b64)
        if current_img is None or last_img is None:
            return

        score = ssim(np.array(last_img), np.array(current_img))
        if score < self.SSIM_THRESHOLD:
            logger.info(
                f"Significant visual change detected at t={timestamp:.2f}s (SSIM: {score:.2f})"
            )
            async with self._state_lock:
                self._pending_vision_events.append(
                    {
                        "timestamp": timestamp,
                        "before_frame_b64": self._last_significant_frame_b64,
                        "after_frame_b64": frame_b64,
                    }
                )
            self._last_significant_frame_b64 = frame_b64

    def _trigger_summary_update(self):
        """Schedules the summary update task, ensuring only one is running or scheduled (debouncing)."""
        if self._summary_update_task and not self._summary_update_task.done():
            logger.debug("Summary update task already scheduled. Skipping.")
            return
        logger.info("Scheduling session summary update.")
        self._summary_update_task = asyncio.create_task(self._update_summary())

    async def _update_summary(self):
        """A serialized task that updates the session summary using the latest events."""
        # Allow a small delay to batch events that arrive close together
        await asyncio.sleep(3.0)

        async with self._summary_update_lock:
            events_to_summarize = []
            current_summary = ""
            async with self._state_lock:
                if not self._unsummarized_events:
                    logger.debug("No new events to summarize.")
                    return
                # Create a copy and clear the original list
                events_to_summarize = list(self._unsummarized_events)
                self._unsummarized_events.clear()
                current_summary = self._session_summary

            logger.info(f"Updating summary with {len(events_to_summarize)} new event(s).")
            try:
                prompt = build_summary_update_prompt(current_summary, events_to_summarize)
                new_summary = await self._summary_client.generate(prompt)

                if new_summary and isinstance(new_summary, str):
                    async with self._state_lock:
                        self._session_summary = new_summary.strip()
                    logger.info("Session summary updated successfully.")
                    logger.debug(f"New Summary: {self._session_summary}")
                else:
                    logger.warning("Summary update LLM call did not return a valid string.")

            except asyncio.CancelledError:
                logger.info("Summary update task cancelled.")
                # Put the events back if cancelled before completion
                async with self._state_lock:
                    self._unsummarized_events = (
                        events_to_summarize + self._unsummarized_events
                    )
                raise
            except Exception as e:
                logger.error(f"Error during summary update: {e}", exc_info=True)
                # Put the events back on failure to be retried next time
                async with self._state_lock:
                    self._unsummarized_events = (
                        events_to_summarize + self._unsummarized_events
                    )

    async def _trigger_turn_analysis(self, speech_event: Optional[dict]):
        """
        Atomically captures the current pending visual events and launches a
        stateless, parallel-safe analysis task.
        """
        self._last_activity_time = asyncio.get_event_loop().time()

        visual_events_for_turn = []
        async with self._state_lock:
            if self._pending_vision_events:
                visual_events_for_turn = list(self._pending_vision_events)
                self._pending_vision_events.clear()

        if not speech_event and not visual_events_for_turn:
            return

        logger.info("Triggering turn analysis...")
        asyncio.create_task(self._analyze_turn(speech_event, visual_events_for_turn))

    async def _flush_pending_events_on_timeout(self):
        """
        On inactivity, triggers a silent turn analysis, but only if no other
        analyses are in progress.
        """
        time_since_activity = asyncio.get_event_loop().time() - self._last_activity_time

        if (
            time_since_activity > self.INACTIVITY_TIMEOUT_SEC
            and self._analyses_in_flight == 0
            and self._pending_vision_events
        ):
            logger.info("Inactivity timeout reached. Flushing pending vision events.")
            await self._trigger_turn_analysis(speech_event=None)

    async def _analyze_turn(
        self, speech_event: Optional[dict], visual_events: List[Dict]
    ):
        """
        A stateless method that analyzes a turn and queues the result for logging.
        """
        async with self._state_lock:
            self._analyses_in_flight += 1

        try:
            response, frame_map = await self._get_llm_analysis(
                speech_event, visual_events
            )

            if response:
                key_events = response.events
                # Add new events to state and trigger summarizer
                if key_events:
                    async with self._state_lock:
                        for event in key_events:
                            # Add to recent events for prompt context
                            self._recent_key_events.append(event)
                            # Add to unsummarized events for the summary task
                            self._unsummarized_events.append(event)
                    self._trigger_summary_update()

                # Queue the turn for logging to the transcript
                if key_events or speech_event:
                    log_job = (speech_event, key_events, frame_map)
                    await self._logging_queue.put(log_job)

                # Publish real-time annotations
                for event in key_events:
                    await self._event_broker.publish(
                        "app:comms:screen_annotation",
                        json.dumps(
                            {
                                "event_name": "ScreenAnnotationEvent",
                                "payload": {"event_description": event.event_description},
                            }
                        ),
                    )
            # If analysis fails but there was speech, still log the speech turn
            elif speech_event:
                log_job = (speech_event, [], {})
                await self._logging_queue.put(log_job)

        except Exception as e:
            logger.error(
                f"An unhandled exception occurred during turn analysis task: {e}",
                exc_info=True,
            )
        finally:
            async with self._state_lock:
                self._analyses_in_flight -= 1

    async def _get_llm_analysis(
        self,
        speech_event: Optional[dict],
        visual_events: List[Dict],
    ) -> Tuple[Optional[TurnAnalysisResponse], Dict[float, str]]:
        """Constructs the prompt and calls the LLM to get turn analysis."""
        # Get a consistent snapshot of the context state for this analysis turn
        async with self._state_lock:
            current_summary = self._session_summary
            recent_event_descriptions = [
                event.event_description for event in self._recent_key_events
            ]

        system_prompt = build_turn_analysis_prompt(
            current_summary, recent_event_descriptions
        )
        user_content = []
        timestamp_to_frame_map: Dict[float, str] = {}
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
                    prev_event = visual_events[i - 1]
                    current_event = visual_events[i]
                    time_diff = current_event["timestamp"] - prev_event["timestamp"]
                    if time_diff <= self.BURST_DETECTION_THRESHOLD_SEC:
                        current_burst.append(current_event)
                    else:
                        bursts.append(current_burst)
                        current_burst = [current_event]
                bursts.append(current_burst)
            # Process each burst (sampling if necessary) and build prompt
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
                    middle_index = len(burst) // 2
                    events_to_process = [burst[0], burst[middle_index], burst[-1]]
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
        if not self._openai_client:
            logger.warning("Unify client not initialized. Skipping analysis.")
            return None, {}
        # If there's no speech and no visual events, there's nothing to analyze.
        if not user_content and not speech_event:
            return None, {}
        try:
            self._openai_client.set_system_message(system_prompt)
            response = await self._openai_client.generate(user_message=user_content)
            if isinstance(response, TurnAnalysisResponse):
                return response, timestamp_to_frame_map
            elif isinstance(response, str):
                logger.warning(
                    "LLM analysis returned a raw string, indicating a schema validation failure. Attempting manual parse."
                )
                try:
                    validated_response = TurnAnalysisResponse.model_validate(
                        json.loads(response)
                    )
                    return validated_response, timestamp_to_frame_map
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(
                        f"Failed to manually parse or validate LLM string response: {e}",
                        exc_info=True,
                    )
                    logger.debug(f"Malformed LLM response string: {response}")
                    return None, {}
            else:
                logger.error(
                    f"Unexpected response type from LLM analysis: {type(response)}"
                )
                return None, {}
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            return None, {}

    async def _log_turn_to_transcript(
        self,
        speech_event: Optional[dict],
        key_events: List[KeyEvent],
        frame_map: Dict[float, str],
    ):
        """
        Handles the slow I/O of logging a turn to the TranscriptManager.
        """
        if not speech_event:
            if key_events:
                logger.info(f"Storing {len(key_events)} silent visual event(s).")
                async with self._state_lock:
                    self._stored_silent_key_events.extend(key_events)
                    self._stored_silent_frame_map.update(frame_map)
            return

        async with self._state_lock:
            all_events = sorted(
                self._stored_silent_key_events + key_events, key=lambda e: e.timestamp
            )
            self._stored_silent_key_events.clear()

            combined_frame_map = self._stored_silent_frame_map.copy()
            self._stored_silent_frame_map.clear()

        combined_frame_map.update(frame_map)
        speech_payload = speech_event["payload"]

        images_to_add = []
        event_to_image_map = {}

        for event in all_events:
            # 1. Register image, get ID
            rep_ts = event.representative_timestamp
            screenshot_b64 = combined_frame_map.get(rep_ts)

            # Implement closest-match fallback for timestamps
            if not screenshot_b64 and combined_frame_map:
                closest_ts = min(
                    combined_frame_map.keys(), key=lambda ts: abs(ts - rep_ts)
                )
                if abs(closest_ts - rep_ts) < 1.0:
                    screenshot_b64 = combined_frame_map[closest_ts]

            if screenshot_b64:
                images_to_add.append(
                    {"data": screenshot_b64, "caption": event.event_description}
                )
                # Use the event's timestamp as a temporary unique key
                event_to_image_map[event.timestamp] = len(images_to_add) - 1

        # Batch log all images in a single call
        logged_image_ids = self._image_manager.add_images(images_to_add)

        # Re-associate logged IDs back to events
        timestamp_to_image_id = {}
        for ts, index in event_to_image_map.items():
            if index < len(logged_image_ids):
                timestamp_to_image_id[ts] = logged_image_ids[index]

        images_dict = {}
        screen_share_dict = {}

        for event in all_events:
            image_id = timestamp_to_image_id.get(event.timestamp)

            screenshot_b64 = combined_frame_map.get(event.representative_timestamp)
            if not screenshot_b64 and combined_frame_map:
                closest_ts = min(
                    combined_frame_map.keys(),
                    key=lambda ts: abs(ts - event.representative_timestamp),
                )
                if abs(closest_ts - event.representative_timestamp) < 1.0:
                    screenshot_b64 = combined_frame_map[closest_ts]

            if not screenshot_b64:
                logger.warning(
                    f"Could not find screenshot for event at {event.timestamp}"
                )
                continue

            # 2. Build screen_share entry
            event_type = "speech" if event.triggering_phrase else "vision"
            ts_key = f"{event.timestamp:.2f}-{event.timestamp:.2f}"
            screen_share_dict[ts_key] = ScreenShareAnnotation(
                caption=event.event_description,
                image=screenshot_b64,
                type=event_type,
            )

            # 3. Build images entry if there's a triggering phrase
            if event.triggering_phrase and image_id:
                try:
                    start_index = speech_payload["content"].index(
                        event.triggering_phrase
                    )
                    end_index = start_index + len(event.triggering_phrase)
                    span_key = f"[{start_index}:{end_index}]"
                    images_dict[span_key] = image_id
                except ValueError:
                    logger.warning(
                        f"Triggering phrase '{event.triggering_phrase}' not found in content."
                    )

        # 4. Construct and log the Message
        message_to_log = Message(
            medium=Medium.PHONE_CALL,
            sender_id=speech_payload["contact_details"]["contact_id"],
            receiver_ids=[0],
            timestamp=datetime.fromisoformat(speech_payload["timestamp"]),
            content=speech_payload["content"],
            screen_share=screen_share_dict,
            images=images_dict,
        )
        logged_messages = self._transcript_manager.log_messages([message_to_log])
        if logged_messages:
            self._last_user_utterance_message_id = logged_messages[0].message_id
            logger.info(
                f"Logged turn to transcript with message_id: {self._last_user_utterance_message_id}"
            )