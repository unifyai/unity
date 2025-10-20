# Screen Share Manager – Controlled Analysis Flow

The `ScreenShareManager` is a stateful component responsible for analyzing real-time screen share video and user speech to detect and annotate meaningful events. It is designed to be directly instantiated and controlled by a consuming manager (like `ConversationManager`), providing a clear, performant, and testable integration pattern.

Implementation lives in `unity/screen_share_manager/`; representative tests live in `tests/test_screen_share_manager/`.

## Motivation and Design

Unlike a background service that listens passively on an event bus, this manager follows a **direct-control** model. The consumer is responsible for its lifecycle (`start`/`stop`) and for pushing data into it (`push_frame`/`push_speech`). This design offers several key advantages:

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

    # In a real system, this would be a loop handling real-time events
    # For this example, we simulate a single user turn.
    
    # Consumer pushes data from its event source
    # (Simulated push_frame calls would happen in a background task)
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
    # The consumer can optionally provide its own context to improve annotation quality.
    # This will be combined with the ScreenShareManager's internal session summary.
    annotation_context = "The user is filling out a profile form and just stated their intent to submit."
    
    # This call returns a task immediately, allowing annotation to run in parallel.
    annotation_task = asyncio.create_task(
        screen_manager.annotate_events(detected_events, annotation_context)
    )
    print("Annotation task started in the background...")

    # The consumer can now prepare its own logic or prompt...
    
    # Await the final result just-in-time when the annotated handles are needed.
    final_annotated_handles = await annotation_task
    print("Annotation task finished.")

    for handle in final_annotated_handles:
        print(f"  - Image (Pending ID: {handle.image_id}): {handle.annotation}")
    
    # 4. Clean up
    screen_manager.stop()

```

## Public API Reference

-   `__init__()`: Initializes the manager and its internal queues and workers.
-   `start() -> Awaitable[None]`: Starts all internal background tasks. Must be called before any other methods.
-   `stop() -> None`: Signals all background tasks to shut down gracefully.
-   `push_frame(frame_b64: str, timestamp: float) -> Awaitable[None]`: Pushes a single video frame into the processing queue.
-   `push_speech(content: str, start_time: float, end_time: float) -> Awaitable[None]`: Pushes a user utterance, which triggers the debounced detection process.
-   `analyze_turn() -> asyncio.Task[List[DetectedEvent]]`: Returns a handle (a Task) to the result of the fast detection stage. The consumer must `await` this task to get the list of candidate events.
-   `annotate_events(events: List[DetectedEvent], context: str) -> Awaitable[List[ImageHandle]]`: Triggers the expensive annotation process for the given events and returns the list of handles, which will have their `.annotation` property populated upon completion.

## Internal Mechanics

The manager uses a robust set of internal patterns to handle real-time streams efficiently:
-   **Sequencer Pattern:** A producer-consumer model with concurrent workers and a single sequencer ensures that frames are processed in parallel but state is updated in strict chronological order to prevent race conditions.
-   **Three-Stage Vision Pipeline:** A cascade of Mean Squared Error (MSE), Structural Similarity Index (SSIM), and semantic contour analysis is used to accurately detect significant visual changes while filtering out noise like cursor movements.