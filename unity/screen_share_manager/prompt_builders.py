import json
from typing import List, Deque, Optional, Dict
from unity.screen_share_manager.types import KeyEvent, SingleAnnotationResponse
from ..common.prompt_helpers import now_utc_str


def _now() -> str:
    return now_utc_str()


def build_detection_prompt(
    current_summary: str,
    speech_event: Optional[Dict],
    has_visual_events: bool,
) -> str:
    """
    Builds a lightweight system prompt for the *detection* stage.
    """
    speech_text = f"User Speech: \"{speech_event['payload']['content']}\"" if speech_event else "No user speech occurred."
    visual_text = "Key visual frames representing screen changes were also provided." if has_visual_events else "No significant visual changes were detected."
    prompt = f"""
You are an ultra-fast analysis assistant. Your job is to identify a small, selective list of key moments from a screen share session.

CONTEXT:
- Session Summary: {current_summary}
- This Turn: {speech_text} {visual_text}

TASK:
Your goal is to be selective. Do not list every single visual change. Instead, identify the key timestamps that represent distinct user actions or milestones.
- If user speech is present, that is almost always a key moment.
- If a single user action causes a rapid series of visual changes (e.g., a menu appearing and animating), only return the timestamp for the final, stable frame of that action.
- If the user describes multiple actions (e.g., "click the close button and then delete the tile"), identify the distinct visual moments for each completed action.

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
4.  **Key Image:** A single image will be provided in the user message.

YOUR TASK:
----------
- Write a single, clear annotation string for the image.
- Your annotation must describe what the screenshot visually contains and explain its significance, using all available context. Answer the question: **"Why is this screenshot important right now?"**

CRITICAL RULES:
---------------
1.  **Be Concise and Context-Aware:** Do NOT repeat information that is already present in the "Previous Annotations from this Turn" section. For example, if a previous annotation already mentioned "the user is viewing a log table," your new annotation should focus only on what is new or different in the current image (e.g., "The user has now filtered the logs to show only errors.").
2.  **Raw String Output:** Your entire response must be ONLY the annotation text, as a raw string. Do NOT wrap it in JSON or markdown.

Example of a GOOD response:
The user has clicked on the 'Context' dropdown menu, revealing a list of available options including 'Sandbox' and 'Default'.
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."


def build_summary_update_prompt(
    current_summary: str,
    new_events: List[KeyEvent],
) -> str:
    new_events_formatted = "\n".join(f"- At t={evt.timestamp:.2f}s: {evt.image_annotation}" for evt in new_events)
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
- Create a new, updated summary that integrates the new events.
- The summary should remain concise, coherent, and chronological.
- Your response must be ONLY the new summary text.
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."