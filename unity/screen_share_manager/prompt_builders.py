import json
from typing import List, Deque, Optional, Dict
from unity.screen_share_manager.types import KeyEvent, TurnAnalysisResponse
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

    This prompt instructs a fast LLM to simply identify timestamps of interest
    from the provided context (speech, visual changes) without generating
    expensive, detailed annotations. Its goal is speed.
    """
    speech_text = f"User Speech: \"{speech_event['payload']['content']}\"" if speech_event else "No user speech occurred."
    visual_text = "Key visual frames representing screen changes were also provided." if has_visual_events else "No significant visual changes were detected."

    prompt = f"""
You are an ultra-fast analysis assistant. Your only job is to identify timestamps of potentially important moments in a screen share session based on limited information. DO NOT describe the images.

CONTEXT:
- Session Summary: {current_summary}
- This Turn: {speech_text} {visual_text}

TASK:
Based on the provided text and image placeholders, identify the timestamps of key moments. A key moment is either the start of a user's speech or a significant visual change.

Respond with a JSON object containing a single key "moments", which is a list of objects. Each object must have a "timestamp" (float) and a "reason" (string, either "user_speech" or "visual_change").

Example Response:
{{
  "moments": [
    {{ "timestamp": 15.2, "reason": "user_speech" }},
    {{ "timestamp": 16.8, "reason": "visual_change" }}
  ]
}}

Provide ONLY the JSON object and nothing else.
"""
    return prompt.strip()


def build_annotation_prompt(
    current_summary: str,
    consumer_context: Optional[str],
) -> str:
    """
    Builds the detailed system prompt for the *annotation* stage.

    This prompt instructs a powerful vision LLM to generate rich, contextual
    annotations by combining the manager's long-term summary with optional,
    immediate context from the consumer.
    """
    schema = TurnAnalysisResponse.model_json_schema()

    consumer_context_section = ""
    if consumer_context:
        consumer_context_section = f"""
2.  **Immediate Turn Context:** High-level context about what the user was doing in this specific turn, provided by the consuming system. Use this as the primary focus for the annotation's relevance.
    <consumer_context>
    {consumer_context}
    </consumer_context>
"""

    prompt = f"""
You are an expert AI assistant specializing in analyzing user interactions during screen share sessions. Your task is to view a curated set of key screenshots and generate a rich, contextual annotation for each one.

CONTEXT PROVIDED:
----------------
1.  **Overall Session Summary:** A rolling summary of what has happened in the session so far. Use this for historical backstory.
    <summary>
    {current_summary}
    </summary>
{consumer_context_section}
3.  **Key Moment Images:** A list of screenshots, each representing a significant moment that you need to describe.

YOUR TASK:
----------
- For each image provided in the user message, you must generate a single, clear `image_annotation`.
- This annotation must describe what the screenshot visually contains and explain its significance, prioritizing the **Immediate Turn Context** (if provided) and using the **Overall Session Summary** for background. It should answer the question: **"Why is this screenshot important right now?"**
- You must also identify the exact timestamp of the image you are annotating and return it in the `representative_timestamp` field.

CRITICAL RULES:
---------------
1.  **Representative Timestamp is Mandatory:** For every event, `representative_timestamp` must exactly match the timestamp of the corresponding frame from the input.
2.  **Chronological Order:** The final list of events must be sorted by timestamp.
3.  **JSON ONLY:** Your response must be a single, valid JSON object that strictly conforms to the schema below.

SCHEMA FOR YOUR RESPONSE:```json
{json.dumps(schema, indent=2)}```
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."


def build_summary_update_prompt(
    current_summary: str,
    new_events: List[KeyEvent],
) -> str:
    """
    Builds the system prompt for the summary update LLM. (Unchanged)
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
- Your response must be ONLY the new summary text, with no preamble or other text.
"""
    return prompt + f"\n\nCurrent UTC time is {_now()}."