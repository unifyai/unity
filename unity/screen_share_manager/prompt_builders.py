import json
from typing import List, Deque, Optional, Dict
from unity.screen_share_manager.types import KeyEvent
from ..common.prompt_helpers import now_utc_str


def _now() -> str:
    return now_utc_str()


def build_detection_prompt(
    current_summary: str,
    speech_event: Optional[Dict],
    has_visual_events: bool,
    burst_events_info: List[str],
) -> str:
    """
    Builds a lightweight system prompt for the *detection* stage.
    """
    speech_text = (
        f"User Speech: \"{speech_event['payload']['content']}\""
        if speech_event
        else "No user speech occurred."
    )
    visual_text = (
        "Key visual frames representing screen changes were also provided."
        if has_visual_events
        else "No significant visual changes were detected."
    )
    burst_section = ""
    if burst_events_info:
        burst_details = "\n- ".join(burst_events_info)
        burst_section = f"""
ADDITIONAL VISUAL CONTEXT:
- {burst_details}
"""
    prompt = f"""
You are an ultra-fast analysis assistant. Your job is to identify a small, selective list of key moments from a screen share session.

CONTEXT:
- Session Summary: {current_summary}
- This Turn: {speech_text} {visual_text}
{burst_section}

TASK:
Your goal is to be selective. Instead of listing every visual change, identify only the timestamps that represent the completion of a distinct user action. If a user's speech implies multiple steps, return a moment for the outcome of each step. For a single action that causes multiple visual changes (like an animation), only return the timestamp for the final, stable frame of that action.

Respond with a JSON object containing a single key "moments", which is a list of objects. Each object must have a "timestamp" (float) and a "reason" (string, e.g., "user_speech" or "visual_change").

Example of a GOOD, selective response for a multi-step action:
{{
  "moments": [
    {{ "timestamp": 15.2, "reason": "user_speech" }},
    {{ "timestamp": 16.8, "reason": "visual_change" }},
    {{ "timestamp": 18.1, "reason": "visual_change" }}
  ]
}}

Provide ONLY the JSON object.
"""
    return prompt.strip()


def build_single_annotation_prompt(
    current_summary: str,
    consumer_context: Optional[str],
    previous_annotations_in_turn: List[str],
    recent_key_events: Deque[KeyEvent],
) -> str:
    """
    Builds a robust system prompt for the single-event annotation stage.
    """
    consumer_context_section = ""
    if consumer_context:
        consumer_context_section = f"""
2.  **Immediate Turn Context:** The user's most recent request or statement.
    <consumer_context>
    {consumer_context}
    </consumer_context>
"""

    previous_annotations_section = ""
    if previous_annotations_in_turn:
        previous_annotations_section = f"""
3.  **Previous Annotations from this Turn:** Descriptions of what has already been noted in the last few seconds.
    <previous_annotations>
    {json.dumps(previous_annotations_in_turn, indent=2)}
    </previous_annotations>
"""

    recent_events_formatted = (
        "\n".join([f"- {evt.image_annotation}" for evt in recent_key_events])
        if recent_key_events
        else "No recent events have been identified."
    )
    recent_events_section = f"""
4.  **Recent Key Events:** A list of the last 5 events that were identified.
    <recent_events>
    {recent_events_formatted}
    </recent_events>
"""

    prompt = f"""
You are an expert AI assistant specializing in analyzing screen share sessions. Your task is to view a single image and write a clear, descriptive annotation for it.

CONTEXT PROVIDED:
----------------
1.  **Overall Session Summary:** The backstory of the session.
    <summary>
    {current_summary}
    </summary>
{consumer_context_section}
{previous_annotations_section}
{recent_events_section}
5.  **Key Image:** A single image will be provided in the user message.

YOUR TASK:
----------
- Write a single, clear annotation string for the image.
- Your annotation must describe what the screenshot visually contains and explain its significance relative to the available context.

CRITICAL RULES:
---------------
1.  **Be Informative:** Your annotation must provide new information. Focus on what has changed or what the outcome of the user's last action is. While you should avoid repeating old information verbatim, you can briefly reference it to provide context. For example, instead of "The user clicked submit," a better annotation for the next frame would be "Following the click, a confirmation modal appeared..
2.  **Raw String Output:** Your entire response must be ONLY the annotation text, as a raw string. Do NOT wrap it in JSON or markdown.

Example of a GOOD response:
The user has clicked on the 'Context' dropdown menu, revealing a list of available options including 'Sandbox' and 'Default'.
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."


def build_summary_update_prompt(
    current_summary: str,
    new_events: List[KeyEvent],
) -> str:
    new_events_formatted = "\n".join(
        f"- At t={evt.timestamp:.2f}s: {evt.image_annotation}" for evt in new_events
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
