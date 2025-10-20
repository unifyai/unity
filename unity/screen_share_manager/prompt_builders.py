import json
from typing import List, Deque
from unity.screen_share_manager.types import KeyEvent, TurnAnalysisResponse
from ..common.prompt_helpers import now_utc_str


def _now() -> str:
    return now_utc_str()


def build_turn_analysis_prompt(
    current_summary: str,
    recent_events: Deque[KeyEvent],
) -> str:
    """
    Builds the system prompt for the screen share turn analysis LLM.
    """
    schema = TurnAnalysisResponse.model_json_schema()

    recent_events_formatted = (
        "\n".join([f"- {evt.image_annotation}" for evt in recent_events])
        if recent_events
        else "No recent events have been identified."
    )

    prompt = f"""
You are an expert AI assistant specializing in analyzing user interactions during screen share sessions. Your task is to watch a video stream, listen to the user's speech, and identify all key moments for the CURRENT TURN ONLY.

CONTEXT PROVIDED:
----------------
1.  **Current Session Summary:** A rolling summary of what has happened in the session so far. This provides the backstory for the current turn.
    <summary>
    {current_summary}
    </summary>
2.  **Recent Key Events:** A list of the last 5 annotations that were generated.
    <recent_events>
    {recent_events_formatted}
    </recent_events>
3.  **User Speech (Optional):** The full transcript of what the user said during their turn.
4.  **Speech Timestamps (Optional):** The start and end time of the user's speech.
5.  **Key Visual Frames:** A list of 'before' and 'after' screenshots representing significant visual changes that occurred. Each visual change has a precise timestamp.

YOUR TASK:
----------
- Analyze all the provided information to create a complete, chronological narrative of the user's CURRENT TURN.
- Identify every distinct, meaningful event that occurred IN THIS TURN. An event can be either a spoken intent or a visual action.
- For each event, you must generate a single, clear `image_annotation`. This annotation must describe what the 'AFTER' screenshot visually contains and explain its significance in the context of the user's entire turn, their speech, and the session summary. It should answer the question: **"Why is this screenshot important right now?"**
- You must also identify which of the provided 'AFTER' frames best illustrates the event and return its exact timestamp in the `representative_timestamp` field.

RULES FOR `image_annotation` (VERY IMPORTANT):
----------------------------------------------
- **Be Descriptive:** The annotation should describe the visual evidence in the 'AFTER' frame.
- **Be Contextual:** The annotation must explain the relevance of the visual evidence to the user's speech, their likely intent, and the ongoing session narrative.
- **Combine Action and Evidence:** Instead of separate fields, merge the user's action and the visual proof into one coherent sentence.

**Example 1 (Speech + Vision):**
- User says: "Okay, I will submit my profile now."
- Visual: User clicks "Submit" and a confirmation appears.
- **`image_annotation`**: "A confirmation message stating 'Your profile has been updated' is visible, confirming the user's stated intention to submit their profile."

**Example 2 (Vision-Only):**
- *No user speech occurs.* A modal dialog appears on the screen.
- Session Context: "User was on the main dashboard."
- **`image_annotation`**: "The 'Account Settings' modal has appeared, which is the expected outcome after the user previously clicked on their profile icon."

CRITICAL RULES:
---------------
1.  **Representative Timestamp is Mandatory:** For every event, the `representative_timestamp` field must contain the exact timestamp of the corresponding 'AFTER' frame from the input. Do NOT invent timestamps.
2.  **Timestamp Format:** All `timestamp` values must be floating-point numbers representing seconds relative to the start of the media stream (e.g., `12.34`).
3.  **Chronological Order:** The final list of events in your response MUST be sorted by timestamp.
4.  **JSON ONLY:** Your entire response must be a single, valid JSON object that strictly conforms to the provided schema. Do not include any other text, notes, or markdown.
5.  **Speech Intent:** Always create an event for the user's primary spoken intent, timestamped at the beginning of their speech. For speech events, the `representative_timestamp` should be the timestamp of the visual frame that best shows the screen state *while they were speaking*.

SCHEMA FOR YOUR RESPONSE:```json
{json.dumps(schema, indent=2)}```
"""
    # Append current time to stabilize cache keys in tests
    return prompt + f"\n\nCurrent UTC time is {_now()}."


def build_summary_update_prompt(
    current_summary: str,
    new_events: List[KeyEvent],
) -> str:
    """
    Builds the system prompt for the summary update LLM.
    """
    new_events_formatted = "\n".join(
        [f"- At t={evt.timestamp:.2f}s: {evt.image_annotation}" for evt in new_events],
    )

    prompt = f"""
You are an expert summarization assistant. Your task is to update a session summary with new events that have just occurred.

CURRENT SUMMARY:
<summary>
{current_summary}
</summary>

NEW EVENTS THAT JUST OCCURRED:
<new_events>
{new_events_formatted}
</new_events>

YOUR TASK:
- Read the current summary and the list of new events.
- Create a new, updated summary that integrates the new events into the narrative of the existing summary.
- The summary should remain concise, coherent, and chronological.
- Do not simply append the new events. Re-write the summary to naturally include them.
- Your response must be ONLY the new summary text, with no preamble or other text.
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."
