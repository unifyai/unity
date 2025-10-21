# Screen Share Manager – Controlled Analysis Flow

The `ScreenShareManager` is a stateful component responsible for analyzing real-time screen share video and user speech to detect and annotate meaningful events. It is designed to be directly instantiated and controlled by a consuming manager (like `ConversationManager`).

Implementation lives in `unity/screen_share_manager/`; representative tests live in `tests/test_screen_share_manager/`.

## Motivation and Design

This manager follows a direct-control model. The consumer is responsible for its lifecycle (`start`/`stop`) and for pushing data into it (`push_frame`/`push_speech`). This design offers several key advantages:

-   **Clear Ownership:** The flow of data is explicit. The consumer drives the analysis, eliminating ambiguity about when and why analysis occurs.
-   **Testability:** The manager can be tested in complete isolation without dependencies on infrastructure like Redis.
-   **Performance:** The architecture is optimized for concurrency, preventing the consumer from being blocked by slow analysis tasks.

## Core Architectural Pattern: The Two-Stage Flow

To maximize performance and responsiveness, the analysis process is decoupled into two distinct stages: a fast **detection** stage and an on-demand **annotation** stage.

### Stage 1: Fast Event Detection

The first stage quickly identifies *potential* moments of interest without performing expensive analysis.

1.  The consumer pushes a stream of video frames and speech events into the manager.
2.  The consumer calls `manager.analyze_turn()`, which returns an `asyncio.Task` immediately.
3.  This task is non-blocking. It internally triggers a lightweight analysis (using heuristics or a fast model) to identify timestamps of significant visual changes or speech events.
4.  When awaited, the task resolves to a `List[DetectedEvent]`. A `DetectedEvent` is a simple data object containing a timestamp and a pending `ImageHandle` for the relevant screenshot.

### Stage 2: On-Demand Contextual Annotation

The second stage generates rich, user-facing descriptions for the events identified in Stage 1.

1.  The consumer receives the `List[DetectedEvent]` from Stage 1.
2.  It then calls `manager.annotate_events(events, context: Optional[str] = None)`. The consumer can **optionally** provide its own high-level context (e.g., "The user was trying to log in").
3.  This method performs the expensive, high-quality vision LLM call. It intelligently combines its own internal, long-term `session_summary` with the optional, immediate context from the consumer to generate the most relevant annotations.
4.  It attaches the generated annotation string to the `.annotation` property of the original `ImageHandle` objects and returns the now-enriched list of handles.

This two-stage flow allows the consumer to run the expensive annotation step in parallel with its own logic, awaiting the results only when they are needed for its final output.

## End-to-End Consumer Example

This example shows how a consumer like `ConversationManager` would use the two-stage API.

```python
from __future__ import annotations
import asyncio

from unity.screen_share_manager.screen_share_manager import ScreenShareManager

async def run_consumer_workflow():
    # 1. Consumer instantiates and starts the manager
    screen_manager = ScreenShareManager()
    await screen_manager.start()
    screen_manager.set_session_context("User is attempting to navigate a web portal.")

    # In a real system, frame pushing would happen in a background loop.
    # For this example, we simulate a single user turn.
    await screen_manager.push_speech("Okay, I'm clicking the 'Submit' button now.", 10.0, 11.5)

    # --- Stage 1: Detection (Fast and Non-Blocking) ---
    analysis_task = screen_manager.analyze_turn()
    print("Detection task started...")
    
    # The consumer can do other work here while detection runs...
    
    detected_events = await analysis_task
    if not detected_events:
        print("No key events were detected.")
        screen_manager.stop()
        return

    print(f"Detected {len(detected_events)} candidate event(s).")

    # --- Stage 2: Annotation (Concurrent) ---
    # The consumer provides its own context to improve annotation quality.
    annotation_context = "The user is filling out a profile form and just stated their intent to submit."
    
    # This call runs annotation in the background.
    annotation_task = asyncio.create_task(
        screen_manager.annotate_events(detected_events, annotation_context)
    )
    print("Annotation task started in the background...")
    
    # Await the final result only when the annotated handles are needed.
    final_annotated_handles = await annotation_task
    print("Annotation task finished.")

    for handle in final_annotated_handles:
        print(f"  - Image (Pending ID: {handle.image_id}): {handle.annotation}")
    
    # 4. Clean up
    screen_manager.stop()

```

## Public API Reference

-   `__init__(...)`: Initializes the manager and its internal components.
-   `start() -> Awaitable[None]`: Starts all internal background tasks. Must be called before any other methods.
-   `stop() -> None`: Signals all background tasks to shut down gracefully.
-   `set_session_context(context_text: str) -> None`: Sets the initial long-term context for the session.
-   `push_frame(frame_b64: str, timestamp: float) -> Awaitable[None]`: Pushes a single video frame into the processing queue.
-   `push_speech(content: str, start_time: float, end_time: float) -> Awaitable[None]`: Pushes a user utterance, which triggers the debounced detection process.
-   `analyze_turn() -> asyncio.Task[List[DetectedEvent]]`: Returns a handle (a Task) to the result of the fast detection stage. The consumer must `await` this task to get the list of candidate events.
-   `annotate_events(events: List[DetectedEvent], context: str) -> Awaitable[List[ImageHandle]]`: Triggers the expensive annotation process for the given events and returns the list of handles, which will have their `.annotation` property populated upon completion.

## Key Features and Mechanics

The manager uses a robust set of internal patterns to handle real-time streams efficiently:

-   **Concurrent and Ordered Processing:** The manager employs a sophisticated producer-consumer architecture. Incoming frames are placed on a queue and processed by a pool of parallel workers (`_frame_processing_worker`) that handle CPU-intensive tasks like image decoding. A dedicated `_sequencer` task then reassembles these processed frames in their original chronological order, ensuring state is updated consistently and preventing race conditions. This guarantees high throughput without blocking the consumer.

-   **Adaptive Frame Dropping:** To maintain stability under high load, the manager monitors its internal frame processing queue. If the queue becomes backlogged beyond a configurable threshold (`adaptive_drop_threshold`), it will proactively drop incoming frames to prevent memory overload and ensure the analysis stays close to real-time.

-   **Multi-Stage Visual Change Detection:** A three-stage pipeline is used to accurately identify significant UI changes while filtering out visual noise:
    1.  **Mean Squared Error (MSE):** A fast, pixel-level check (`_calculate_mse`) acts as a pre-filter. If the difference between two frames is below the `mse_threshold`, no further checks are needed.
    2.  **Structural Similarity Index (SSIM):** For frames that pass the MSE check, SSIM is used to measure perceptual difference, which is more robust to minor shifts in lighting or rendering.
    3.  **Semantic Contour Analysis:** The final stage (`_is_semantically_significant`) analyzes the geometric shapes of the differences. This is highly effective at filtering out insignificant changes like blinking text cursors, mouse pointer movements, and subtle anti-aliasing shifts that might otherwise pass the first two checks.

-   **Burst Event Sampling:** The system identifies rapid sequences of visual changes occurring within a short time window (`burst_detection_threshold_sec`). Instead of analyzing every single frame in such a "burst" (e.g., during an animation or fast typing), it intelligently samples keyframes—typically the first, middle, and last. It then informs the detection LLM that a burst occurred, providing a concise yet representative summary of the rapid action.

-   **Rolling Session Summary:** The manager maintains a long-term, narrative summary of the session's events (`_session_summary`). After each turn's annotations are generated, a background task (`_update_summary`) uses a separate LLM client to integrate the new information, ensuring the summary evolves and remains coherent. This provides deep, chronological context for all subsequent analysis.

-   **Multi-Layered Annotation Context:** To generate the most relevant and informative annotations, the prompt for the vision LLM is constructed from four distinct layers of context:
    1.  **Long-Term Context:** The rolling session summary.
    2.  **Turn-Specific Context:** The optional context string provided by the consumer (e.g., "User is trying to log in").
    3.  **Intra-Turn Context:** The sequence of annotations that have already been generated *within the same analysis turn*, allowing the model to build a coherent narrative and avoid repetition.
    4.  **Recent Event History:** A list of the last few annotated events, giving the model immediate historical context.

-   **Debouncing and Inactivity Flushing:** Analysis is not triggered on every single event.
    -   **Debouncing:** When a user speaks (`push_speech`), the manager waits for a brief period (`debounce_delay_sec`) before starting the analysis. This allows related visual events that occur shortly after the speech to be grouped into the same analysis turn.
    -   **Inactivity Flushing:** If a significant visual change occurs without any accompanying speech, an inactivity timer (`inactivity_timeout_sec`) begins. If no further activity occurs before the timer expires, the pending visual events are "flushed" for analysis. This ensures purely visual actions are still captured.