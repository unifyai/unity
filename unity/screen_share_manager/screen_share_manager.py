# FILE: unity/screen_share_manager/screen_share_manager.py

import asyncio
import base64
import io
import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Tuple, Dict, Any

import unify
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from unity.image_manager.image_manager import ImageManager, ImageHandle
from .prompt_builders import build_summary_update_prompt, build_single_annotation_prompt, build_detection_prompt
from .types import KeyEvent, DetectedEvent

logger = logging.getLogger(__name__)


class ScreenShareManager:
    """
    A stateful component that analyzes screen share streams and user speech to
    detect key events and return annotated ImageHandles.
    """

    def __init__(self, include_raw_image_data_in_result: bool = False):
        self.MSE_THRESHOLD = 10
        self.SSIM_THRESHOLD = 0.995
        self.MIN_CONTOUR_AREA = 50
        self.DEBOUNCE_DELAY_SEC = 0.5
        self.INACTIVITY_TIMEOUT_SEC = 5.0
        self.FRAME_BUFFER_SIZE = 100
        self.MAX_FRAME_WORKERS = os.cpu_count() or 4
        self.FRAME_QUEUE_SIZE = 150
        self.RESULTS_QUEUE_SIZE = 200
        self.DETECTION_QUEUE_SIZE = 10
        self._analysis_client = unify.AsyncUnify("gpt-4o@openai")
        self._detection_client = unify.AsyncUnify("gpt-4o-mini@openai")
        self._summary_client = unify.AsyncUnify("gpt-4o-mini@openai")
        self._image_manager = ImageManager()
        self._cpu_executor = ThreadPoolExecutor(max_workers=self.MAX_FRAME_WORKERS)
        self._stop_event = asyncio.Event()
        self._frame_sequence_id = 0
        self._frame_queue = asyncio.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._results_queue = asyncio.Queue(maxsize=self.RESULTS_QUEUE_SIZE)
        self._detection_queue = asyncio.Queue(maxsize=self.DETECTION_QUEUE_SIZE)
        self._frame_workers: List[asyncio.Task] = []
        self._sequencer_task: Optional[asyncio.Task] = None
        self._inactivity_task: Optional[asyncio.Task] = None
        self._frame_buffer: Deque[Tuple[float, str]] = deque(maxlen=self.FRAME_BUFFER_SIZE)
        self._pending_vision_events: List[Dict] = []
        self._last_significant_frame_b64: Optional[str] = None
        self._last_significant_frame_pil: Optional[Image.Image] = None
        self._last_activity_time: float = 0.0
        self._state_lock = asyncio.Lock()
        self._debounce_task: Optional[asyncio.Task] = None
        self._session_summary: str = "The session has just begun."
        self._unsummarized_events: List[KeyEvent] = []
        self._summary_update_lock = asyncio.Lock()
        self._summary_update_task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info("ScreenShareManager starting background workers...")
        self._last_activity_time = asyncio.get_event_loop().time()
        self._sequencer_task = asyncio.create_task(self._sequencer())
        self._frame_workers = [asyncio.create_task(self._frame_processing_worker()) for _ in range(self.MAX_FRAME_WORKERS)]
        self._inactivity_task = asyncio.create_task(self._inactivity_flush_loop())

    def stop(self):
        logger.info("ScreenShareManager stopping...")
        self._stop_event.set()
        for task in [self._sequencer_task, self._inactivity_task, self._summary_update_task, self._debounce_task, *self._frame_workers]:
            if task and not task.done(): task.cancel()
        self._cpu_executor.shutdown(wait=False, cancel_futures=True)

    def set_session_context(self, context_text: str):
        self._session_summary = context_text.strip()
        logger.info(f"Initial session context set: '{self._session_summary}'")

    async def push_frame(self, frame_b64: str, timestamp: float):
        if self._frame_queue.full():
            logger.warning("Frame queue is full. Dropping incoming frame.")
            return
        self._frame_sequence_id += 1
        await self._frame_queue.put((self._frame_sequence_id, {"payload": {"frame_b64": frame_b64, "timestamp": timestamp}}))

    async def push_speech(self, content: str, start_time: float, end_time: float):
        logger.info(f"Received speech event: '{content}'")
        speech_event = {"payload": {"content": content, "start_time": start_time, "end_time": end_time}}
        self._trigger_turn_analysis(speech_event=speech_event)

    def analyze_turn(self) -> asyncio.Task[List[DetectedEvent]]:
        async def _analysis_wrapper() -> List[DetectedEvent]:
            detection_result = await self._detection_queue.get()
            if not detection_result: return []
            key_moments, frame_map = detection_result
            logger.debug(f"Detection result received with {len(key_moments)} moments.")
            events_to_return, images_to_add, moment_map = [], [], {}
            for i, moment in enumerate(key_moments):
                screenshot_data_url = frame_map.get(moment['timestamp'])
                if screenshot_data_url:
                    raw_b64 = self._strip_data_url_prefix(screenshot_data_url)
                    images_to_add.append({"data": raw_b64, "caption": "Detected screen event"})
                    moment_map[i] = moment
            if not images_to_add: return []
            handles = await asyncio.to_thread(self._image_manager.add_images, images_to_add, synchronous=False, return_handles=True)
            for i, handle in enumerate(handles):
                if handle and i in moment_map:
                    moment = moment_map[i]
                    events_to_return.append(DetectedEvent(timestamp=moment['timestamp'], detection_reason=moment.get('reason', 'visual_change'), image_handle=handle))
            logger.info(f"Created {len(events_to_return)} DetectedEvent objects.")
            return events_to_return
        return asyncio.create_task(_analysis_wrapper())

    async def annotate_events(self, events: List[DetectedEvent], context: Optional[str] = None) -> List[ImageHandle]:
        if not events:
            return []
        
        logger.info(f"Starting concurrent annotation for {len(events)} detected events.")

        # Create and run an independent annotation task for each event
        annotation_tasks = []
        annotations_so_far: List[str] = []
        for event in events:
            # Pass the list of previous annotations for better context
            task = self._get_llm_annotation_for_event(event, context, annotations_so_far)
            annotation_tasks.append(task)
        
        results = await asyncio.gather(*annotation_tasks, return_exceptions=True)

        annotated_handles: List[ImageHandle] = []
        key_events_for_summary: List[KeyEvent] = []

        for i, result in enumerate(results):
            event = events[i]
            if isinstance(result, Exception) or not result:
                logger.warning(f"Failed to generate annotation for event at timestamp {event.timestamp:.2f}s. Error: {result}")
                continue

            annotation_text = result
            event.image_handle.annotation = annotation_text
            annotated_handles.append(event.image_handle)
            logger.debug(f"Attached annotation '{annotation_text}' to handle for timestamp {event.timestamp:.2f}s.")
            key_events_for_summary.append(KeyEvent(timestamp=event.timestamp, image_annotation=annotation_text, representative_timestamp=event.timestamp))

        if key_events_for_summary:
            async with self._state_lock:
                self._unsummarized_events.extend(key_events_for_summary)
            self._trigger_summary_update()
            
        logger.info(f"Successfully annotated {len(annotated_handles)} handles.")
        return annotated_handles

    def _strip_data_url_prefix(self, data_url: str) -> str:
        if data_url.startswith('data:image'): return data_url.split(',', 1)[1]
        return data_url

    def _b64_to_image(self, b64_string: str) -> Image.Image:
        try:
            img_data_b64 = self._strip_data_url_prefix(b64_string)
            img_data = base64.b64decode(img_data_b64)
            return Image.open(io.BytesIO(img_data)).convert("L").resize((512, 288))
        except Exception as e:
            raise ValueError("Invalid image data") from e

    def _calculate_mse(self, img1, img2):
        err = np.sum((np.array(img1, dtype=np.float64) - np.array(img2, dtype=np.float64)) ** 2)
        return err / (img1.size[0] * img1.size[1])

    def _is_semantically_significant(self, img_before, img_after):
        diff = cv2.absdiff(np.array(img_before), np.array(img_after))
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > self.MIN_CONTOUR_AREA for c in contours)

    async def _inactivity_flush_loop(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.INACTIVITY_TIMEOUT_SEC)
            is_debouncing = self._debounce_task and not self._debounce_task.done()
            if (asyncio.get_event_loop().time() - self._last_activity_time >= self.INACTIVITY_TIMEOUT_SEC and not is_debouncing and self._pending_vision_events):
                logger.info("Inactivity timeout. Flushing pending vision events for detection.")
                self._trigger_turn_analysis(speech_event=None)

    async def _frame_processing_worker(self):
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                seq_id, event_data = await self._frame_queue.get()
                pil_img = await loop.run_in_executor(self._cpu_executor, self._b64_to_image, event_data["payload"]["frame_b64"])
                await self._results_queue.put((seq_id, event_data, pil_img))
                self._frame_queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Error in frame worker: {e}", exc_info=True)

    async def _sequencer(self):
        next_seq_id, results_buffer = 1, {}
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                seq_id, event_data, pil_img = await self._results_queue.get()
                if seq_id != next_seq_id:
                    results_buffer[seq_id] = (seq_id, event_data, pil_img)
                    continue
                self._last_activity_time = loop.time()
                ts, b64 = event_data["payload"]["timestamp"], event_data["payload"]["frame_b64"]
                self._frame_buffer.append((ts, b64))
                if self._last_significant_frame_pil:
                    if self._calculate_mse(pil_img, self._last_significant_frame_pil) > self.MSE_THRESHOLD:
                        score = await loop.run_in_executor(self._cpu_executor, ssim, np.array(self._last_significant_frame_pil), np.array(pil_img))
                        if score < self.SSIM_THRESHOLD and self._is_semantically_significant(self._last_significant_frame_pil, pil_img):
                            logger.debug(f"Sequencer detected significant visual change at t={ts:.2f}s.")
                            async with self._state_lock:
                                self._pending_vision_events.append({"timestamp": ts, "before_frame_b64": self._last_significant_frame_b64, "after_frame_b64": b64})
                            self._last_significant_frame_b64, self._last_significant_frame_pil = b64, pil_img
                else:
                    self._last_significant_frame_b64, self._last_significant_frame_pil = b64, pil_img
                next_seq_id += 1
                while next_seq_id in results_buffer:
                    _, _, _ = results_buffer.pop(next_seq_id, (None, None, None))
                    next_seq_id += 1
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Error in sequencer: {e}", exc_info=True)

    def _trigger_turn_analysis(self, speech_event):
        self._last_activity_time = asyncio.get_event_loop().time()
        if self._debounce_task and not self._debounce_task.done(): self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_detection_runner(speech_event))

    async def _debounced_detection_runner(self, speech_event):
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY_SEC)
            logger.info("Debounce window ended. Starting detection.")
            visual_events, latest_frame = [], None
            async with self._state_lock:
                if self._pending_vision_events:
                    visual_events = list(self._pending_vision_events)
                    self._pending_vision_events.clear()
            if speech_event and not visual_events and self._frame_buffer:
                latest_frame = self._frame_buffer[-1]
            if speech_event or visual_events:
                await self._detect_key_moments(speech_event, visual_events, latest_frame)
        except asyncio.CancelledError: logger.info("Debounced detection was cancelled.")

    async def _detect_key_moments(self, speech_event, visual_events, latest_frame):

        consolidated_visual_events = []
        time_window = 1.0  # Time window in seconds to group events
        if visual_events:
            visual_events.sort(key=lambda x: x['timestamp'])
            
            i = 0
            while i < len(visual_events):
                # Find the end of the current cluster of events
                j = i
                while j + 1 < len(visual_events) and \
                      (visual_events[j+1]['timestamp'] - visual_events[j]['timestamp']) <= time_window:
                    j += 1
                # Add the last event of the cluster
                consolidated_visual_events.append(visual_events[j])
                i = j + 1
            
            if len(visual_events) > len(consolidated_visual_events):
                logger.debug(f"Consolidated {len(visual_events)} visual events down to {len(consolidated_visual_events)} to reduce noise.")
            visual_events = consolidated_visual_events
        
        async with self._state_lock: current_summary = self._session_summary
        system_prompt = build_detection_prompt(current_summary, speech_event, bool(visual_events or latest_frame))
        self._detection_client.set_system_message(system_prompt)
        user_content, frame_map = [], {}
        if speech_event: user_content.append({"type": "text", "text": f"User speech at t={speech_event['payload']['start_time']:.2f}s."})
        for ve in visual_events:
            user_content.append({"type": "text", "text": f"Visual change at t={ve['timestamp']:.2f}s."})
            frame_map[ve['timestamp']] = ve['after_frame_b64']
        if latest_frame: frame_map[latest_frame[0]] = latest_frame[1]
        if not user_content:
            await self._detection_queue.put(([], {}))
            return
        try:
            logger.debug("Calling detection LLM...")
            response_str = await self._detection_client.generate(user_message="\n".join(c['text'] for c in user_content))
            result = json.loads(response_str)
            key_moments = result.get("moments", [])
            key_moments.sort(key=lambda x: x['timestamp'])
            for moment in key_moments:
                ts = moment['timestamp']
                if ts not in frame_map and self._frame_buffer:
                    _, closest_frame = min(self._frame_buffer, key=lambda x: abs(x[0] - ts))
                    frame_map[ts] = closest_frame
            await self._detection_queue.put((key_moments, frame_map))
        except Exception as e:
            logger.error(f"Failed to detect key moments: {e}", exc_info=True)
            await self._detection_queue.put(([], {}))

    async def _get_llm_annotation_for_event(self, event_to_annotate: DetectedEvent, consumer_context: Optional[str], previous_annotations_in_turn: List[str]) -> Optional[str]:
        """Performs a single, reliable annotation call for one event."""
        async with self._state_lock:
            current_summary = self._session_summary
        
        system_prompt = build_single_annotation_prompt(current_summary, consumer_context, previous_annotations_in_turn)
        self._analysis_client.set_system_message(system_prompt)

        raw_bytes = event_to_annotate.image_handle.raw()
        b64_data = base64.b64encode(raw_bytes).decode('utf-8')
        data_url = f"data:image/png;base64,{b64_data}"
        
        user_content = [
            {"type": "text", "text": f"Moment at t={event_to_annotate.timestamp:.2f}s:"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        try:
            logger.debug(f"Calling annotation LLM for image at timestamp {event_to_annotate.timestamp:.2f}s...")
            response = await self._analysis_client.generate(user_message=user_content)

            if isinstance(response, str) and response.strip():
                logger.debug(f"Annotation LLM returned: '{response.strip()}'")
                return response.strip()
            else:
                logger.warning(f"Annotation LLM returned an empty or invalid response for timestamp {event_to_annotate.timestamp:.2f}s. Type: {type(response)}")
                return None
        except Exception as e:
            logger.error(f"Error during single-event LLM annotation: {e}", exc_info=True)
            return None

    def _trigger_summary_update(self):
        if self._summary_update_task and not self._summary_update_task.done(): return
        self._summary_update_task = asyncio.create_task(self._update_summary())

    async def _update_summary(self):
        await asyncio.sleep(1.0)
        async with self._summary_update_lock:
            if not self._unsummarized_events: return
            events = list(self._unsummarized_events)
            self._unsummarized_events.clear()
            try:
                prompt = build_summary_update_prompt(self._session_summary, events)
                new_summary = await self._summary_client.generate(prompt)
                if new_summary and isinstance(new_summary, str):
                    self._session_summary = new_summary.strip()
                    logger.info("Session summary updated.")
            except Exception as e:
                logger.error(f"Error updating summary: {e}", exc_info=True)
                self._unsummarized_events = events + self._unsummarized_events