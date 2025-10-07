import json
from typing import List
from unity.screen_share_manager.types import KeyEvent, TurnAnalysisResponse


def build_turn_analysis_prompt(
    current_summary: str,
    recent_events: List[str],
) -> str:
    """
    Builds the system prompt for the screen share turn analysis LLM.
    """
    schema = TurnAnalysisResponse.model_json_schema()

    recent_events_formatted = (
        "\n".join([f"- {evt}" for evt in recent_events])
        if recent_events
        else "No recent events have been identified."
    )

    prompt = f"""
You are an expert AI assistant specializing in analyzing user interactions during screen share sessions. Your task is to watch a video stream, listen to the user's speech, and identify all key moments for the CURRENT TURN ONLY.

CONTEXT PROVIDED:
1.  **Current Session Summary:** A rolling summary of what has happened in the session so far. Use this for context, but DO NOT update it.
    <summary>
    {current_summary}
    </summary>
2.  **Recent Key Events:** A list of the last 5 events that were identified.
    <recent_events>
    {recent_events_formatted}
    </recent_events>
3.  **User Speech (Optional):** The full transcript of what the user said during their turn.
4.  **Speech Timestamps (Optional):** The start and end time of the user's speech.
5.  **Key Visual Frames:** A list of 'before' and 'after' screenshots representing significant visual changes that occurred. Each visual change has a precise timestamp.

YOUR TASK:
- Analyze all the provided information to create a complete, chronological narrative of the user's CURRENT TURN.
- Identify every distinct, meaningful event that occurred IN THIS TURN. An event can be either a spoken intent or a visual action.
- For each event, you must provide a precise timestamp and a clear description.
- Crucially, you must also identify which of the provided 'AFTER' frames best illustrates the event and return its exact timestamp in the `representative_timestamp` field.
- If a user's speech directly refers to an action (e.g., "I'll click **this button**"), you MUST identify the exact text span ("this button") as the `triggering_phrase`.

CRITICAL RULES:
1.  **Representative Timestamp is Mandatory:** For every event, the `representative_timestamp` field must contain the exact timestamp of the corresponding 'AFTER' frame from the input. Do NOT invent timestamps.
2.  **Timestamp Format:** All `timestamp` values must be floating-point numbers representing seconds relative to the start of the media stream (e.g., `12.34`). Do NOT use Unix epoch timestamps.
3.  **Chronological Order:** The final list of events in your response MUST be sorted by timestamp.
4.  **Disentangle Events:** If speech and a visual change happen around the same time, create separate events for both the spoken intent and the visual action.
5.  **Speech Intent:** Always create an event for the user's primary spoken intent, timestamped at the beginning of their speech. For speech events, the `representative_timestamp` should be the timestamp of the visual frame that best shows the screen state *while they were speaking*.
6.  **Be Concise:** Event descriptions should be brief, factual, and in the third person (e.g., "User navigated to the profile page.").
7.  **JSON ONLY:** Your entire response must be a single, valid JSON object that strictly conforms to the provided schema. Do not include any other text, notes, or markdown.

SCHEMA FOR YOUR RESPONSE:```json
{json.dumps(schema, indent=2)}```
"""
    return prompt


def build_summary_update_prompt(current_summary: str, new_events: List[KeyEvent]) -> str:
    """
    Builds the system prompt for the summary update LLM.
    """
    new_events_formatted = "\n".join(
        [f'- At t={evt.timestamp:.2f}s: {evt.event_description}' for evt in new_events]
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
    return prompt