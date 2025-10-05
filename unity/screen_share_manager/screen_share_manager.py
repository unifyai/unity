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
from openai import AsyncOpenAI
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

from unity.conversation_manager_2.event_broker import get_event_broker
from unity.file_manager.parser.types.document import DocumentImage
from unity.image_manager.image_manager import ImageManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import (
    Medium,
    Message,
    ScreenShareAnnotation,
)

from .prompt_builders import build_turn_analysis_prompt
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
    """

    def __init__(self):
        # Configuration
        self.SSIM_THRESHOLD = 0.97
        self.INACTIVITY_TIMEOUT_SEC = 10.0
        self.FRAME_BUFFER_SIZE = 100  # Approx 10 seconds at 10fps
        self.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
        self.BURST_DETECTION_THRESHOLD_SEC = 2.0

        # Clients and Managers
        self._event_broker: redis.Redis = get_event_broker()
        self._openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._image_manager = ImageManager()
        self._transcript_manager = TranscriptManager()

        # State Variables
        self._stop_event = asyncio.Event()
        self._frame_buffer: Deque[Tuple[float, str]] = deque(
            maxlen=self.FRAME_BUFFER_SIZE
        )
        self._pending_vision_events: List[Dict] = []
        self._stored_silent_key_events: List[KeyEvent] = []  # Store silent events here
        self._last_significant_frame_b64: Optional[str] = None
        self._last_user_utterance_message_id: Optional[int] = None
        self._last_activity_time: float = asyncio.get_event_loop().time()
        self._analysis_lock = asyncio.Lock()

    async def start(self):
        """Starts the main event listening loop for the manager."""
        logger.info("ScreenShareManager started. Listening for events...")
        await self._listen_for_events()

    def stop(self):
        """Signals the manager to gracefully shut down."""
        self._stop_event.set()
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
        """
        The core loop that subscribes to Redis and dispatches events.
        Also handles the inactivity timeout for flushing silent visual events.
        """
        async with self._event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:screen_frame", "app:comms:phone_utterance"
            )

            while not self._stop_event.is_set():
                try:
                    # Wait for a message with a timeout to check for inactivity
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )

                    if message:
                        channel = message["channel"]
                        event_data = json.loads(message["data"])

                        if channel == "app:comms:screen_frame":
                            asyncio.create_task(self._handle_frame_event(event_data))
                        elif channel == "app:comms:phone_utterance":
                            asyncio.create_task(
                                self._handle_utterance_event(event_data)
                            )

                    # Check for inactivity and flush pending events
                    await self._flush_pending_events_on_timeout()

                except asyncio.TimeoutError:
                    continue  # No message, just loop to check timeout
                except Exception as e:
                    logger.error(f"Error in event listener loop: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Prevent rapid-fire errors

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
            self._pending_vision_events.append(
                {
                    "timestamp": timestamp,
                    "before_frame_b64": self._last_significant_frame_b64,
                    "after_frame_b64": frame_b64,
                }
            )
            self._last_significant_frame_b64 = frame_b64

    async def _handle_utterance_event(self, event_data: dict):
        """Processes a speech event, triggering a full turn analysis."""
        self._last_activity_time = asyncio.get_event_loop().time()
        logger.info(f"Utterance event received. Triggering turn analysis.")
        asyncio.create_task(self._analyze_turn(speech_event=event_data))

    async def _flush_pending_events_on_timeout(self):
        """Analyzes buffered vision-only events if no speech has occurred recently."""
        time_since_activity = asyncio.get_event_loop().time() - self._last_activity_time
        if (
            time_since_activity > self.INACTIVITY_TIMEOUT_SEC
            and self._pending_vision_events
        ):
            logger.info("Inactivity timeout reached. Flushing pending vision events.")
            # Reset activity time to prevent continuous flushing
            self._last_activity_time = asyncio.get_event_loop().time()
            asyncio.create_task(self._analyze_turn(speech_event=None))

    async def _analyze_turn(self, speech_event: Optional[dict]):
        """
        Orchestrates the analysis of a user's turn, gathering context,
        calling the LLM, and dispatching the results.
        """
        async with self._analysis_lock:
            # Capture and clear pending visual events
            visual_events = list(self._pending_vision_events)
            self._pending_vision_events.clear()

            if not speech_event and not visual_events:
                return

            key_events = await self._get_llm_analysis(speech_event, visual_events)
            if not key_events:
                # If there was a speech event but no key events (e.g., LLM error),
                # we should still log the basic speech message.
                if speech_event:
                    await self._log_turn_to_transcript(speech_event, [])
                return

            if speech_event:
                # If there's speech, log everything immediately
                await self._log_turn_to_transcript(speech_event, key_events)
            else:
                # If it's a silent visual event, store it for the next turn
                logger.info(f"Storing {len(key_events)} silent visual event(s).")
                self._stored_silent_key_events.extend(key_events)

            # Publish real-time annotations for the ConversationManager
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

    async def _get_llm_analysis(
        self, speech_event: Optional[dict], visual_events: List[Dict]
    ) -> List[KeyEvent]:
        """Constructs the prompt and calls the LLM to get turn analysis."""
        system_prompt = build_turn_analysis_prompt()

        user_content = []
        if speech_event:
            payload = speech_event["payload"]
            user_content.append(
                {"type": "text", "text": f"User Speech: \"{payload['content']}\""}
            )
            # Check for optional timestamp keys before accessing them
            if "start_time" in payload and "end_time" in payload:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Speech Timestamps: Start={payload['start_time']:.2f}s, End={payload['end_time']:.2f}s",
                    }
                )

        if visual_events:
            user_content.append({"type": "text", "text": "\n--- Key Visual Frames ---"})

            # Burst detection logic
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
                bursts.append(current_burst)  # Add the last burst

            # Process each burst (sampling if necessary) and build prompt
            frame_counter = 0
            for burst in bursts:
                events_to_process_for_burst = burst
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
                    events_to_process_for_burst = [
                        burst[0],
                        burst[middle_index],
                        burst[-1],
                    ]

                for ve in events_to_process_for_burst:
                    frame_counter += 1
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"\nVisual Change #{frame_counter} at t={ve['timestamp']:.2f}s:",
                        }
                    )
                    user_content.append({"type": "text", "text": "BEFORE:"})
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": ve["before_frame_b64"]},
                        }
                    )
                    user_content.append({"type": "text", "text": "AFTER:"})
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": ve["after_frame_b64"]},
                        }
                    )

        if not self._openai_client:
            logger.warning("OpenAI client not initialized. Skipping analysis.")
            return []

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_model=TurnAnalysisResponse,
            )
            return response.events
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            return []

    async def _log_turn_to_transcript(
        self, speech_event: dict, key_events: List[KeyEvent]
    ):
        """Logs a speech-based turn to the TranscriptManager, including any stored silent events."""
        # Merge current key events with any stored silent events
        all_events = sorted(
            self._stored_silent_key_events + key_events, key=lambda e: e.timestamp
        )
        self._stored_silent_key_events.clear()

        speech_payload = speech_event["payload"]
        images_dict = {}
        screen_share_dict = {}

        for event in all_events:
            # 1. Register image, get ID
            image_ids = self._image_manager.add_images(
                [{"data": event.screenshot_b64, "caption": event.event_description}]
            )
            if not image_ids:
                continue
            image_id = image_ids[0]

            # 2. Build screen_share entry
            ts_key = f"{event.timestamp:.2f}-{event.timestamp:.2f}"
            screen_share_dict[ts_key] = ScreenShareAnnotation(
                caption=event.event_description, image_b64=event.screenshot_b64
            )

            # 3. Build images entry if there's a triggering phrase
            if event.triggering_phrase:
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
            medium=Medium.PHONE_CALL,  # Or derive from event context if available
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
