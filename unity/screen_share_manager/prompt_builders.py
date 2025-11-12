import json
from typing import List, Deque, Optional, Dict
from unity.screen_share_manager.types import KeyEvent
from ..common.prompt_helpers import now


def build_detection_prompt(
    current_summary: str,
    speech_events: List[Dict],
    visual_events_info: List[str],
    burst_events_info: List[str],
) -> str:
    """
    Builds a lightweight system prompt for the *detection* stage.
    """
    if speech_events:
        speech_details = "\n".join(
            [
                f"- At t={evt['payload']['start_time']:.2f}s: \"{evt['payload']['content']}\""
                for evt in speech_events
            ],
        )
        speech_text = f"User Speech:\n{speech_details}"
    else:
        speech_text = "No user speech occurred during this turn."

    if visual_events_info:
        visual_details = "\n".join([f"- {info}" for info in visual_events_info])
        visual_text = f"Visual Changes:\n{visual_details}"
    else:
        visual_text = "No significant visual changes were detected."

    burst_section = ""
    if burst_events_info:
        burst_details = "\n- ".join(burst_events_info)
        burst_section = f"""
ADDITIONAL VISUAL CONTEXT:
- {burst_details}
"""
    prompt = f"""
You are an expert, ultra-fast analysis assistant. Your job is to consolidate and identify key moments from a screen share session.

PRIMARY DIRECTIVE:
Your main goal is to consolidate speech and visual events that describe the same user action into a single, definitive moment. When this happens, you MUST prioritize the timestamp of the **visual event**, as it represents the tangible outcome of the user's stated intent.

---
EXAMPLE OF CONSOLIDATION:

CONTEXT:
- User Speech: "Okay, I'm clicking on the 'Submit' button now." (Occurred at t=10.5s)
- Visual Change: A "Success!" modal appears on the screen. (Occurred at t=11.2s)

❌ BAD RESPONSE (Incorrect):
{{
  "moments": [
    {{ "timestamp": 10.5, "reason": "user_speech" }},
    {{ "timestamp": 11.2, "reason": "visual_change" }}
  ]
}}
(Reasoning: This is redundant. The speech and the visual change are part of the same 'submit' action.)

✅ GOOD RESPONSE (Correct):
{{
  "moments": [
    {{ "timestamp": 11.2, "reason": "user_speech" }}
  ]
}}
(Reasoning: This correctly identifies the visual outcome as the single key moment and discards the redundant speech event.)
---

YOUR CURRENT TURN CONTEXT:
- Session Summary: {current_summary}
- This Turn: {speech_text}\n{visual_text}
{burst_section}

CRITICAL RULES:
1.  **Prioritize Outcomes:** Always favor the visual result of an action over the spoken intent when they refer to the same event.
2.  **Preserve Speech Without Visuals:** If a speech event occurs but has NO corresponding visual change within the turn, the speech event ITSELF IS the key moment. **Do NOT discard it.** Return a moment with the speech event's timestamp.
3.  **Be Selective with Animations:** For a single action that causes multiple visual changes (like an animation or a window opening), **only return the timestamp for the final, stable frame of that action.**
4.  **Handle Multi-Step Actions:** If a user's speech implies multiple distinct steps, it is appropriate to return a moment for the outcome of each step.

Respond with a JSON object containing a single key "moments", which is a list of objects. Each object must have a "timestamp" (float) and a "reason" (string). Provide ONLY the JSON object.

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
3.  **Previous Annotations from this Turn:** Descriptions of what has already been noted in this specific analysis turn.
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
4.  **Recent Key Events:** A list of the last events that were identified.
    <recent_events>
    {recent_events_formatted}
    </recent_events>
"""

    prompt = f"""
You are an expert AI assistant specializing in analyzing screen share sessions. Your task is to view a single image and write a clear, narrative annotation that builds upon previous events.

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
- Examine the Key Image and compare it to the description of the 'Immediately Preceding Event'.
- Write a single, clear annotation string that describes what is new or different in the current image. Your annotation must create a narrative flow.

CRITICAL RULES:
---------------
1.  **Focus on the Delta:** Your primary goal is to explain the change. For example, instead of "The screen shows a file tree," a better annotation is "The user has now expanded the 'unity' folder in the file tree, revealing its subdirectories."
2.  **Be Informative and Concise:** Explain the significance of the change in relation to the overall context.
3.  **Handle First Event:** If this is the first event, it's okay to provide a complete description of the scene as a baseline.
4.  **Raw String Output:** Your entire response must be ONLY the annotation text, as a raw string. Do NOT wrap it in JSON or markdown.

Example of a GOOD, narrative response:
Following the user's action, a dropdown menu for 'Context' has opened, showing 'Sandbox' and 'Default' as available options.
"""
    return prompt + f"\n\nCurrent UTC time is {now()}."


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
    return prompt + f"\n\nCurrent UTC time is {now()}."
