import json
from typing import List, Deque
from unity.screen_share_manager.types import KeyEvent, TurnAnalysisResponse


def build_turn_analysis_prompt(
    current_summary: str,
    recent_events: Deque[KeyEvent],
) -> str:
    """
    Builds the system prompt for the screen share turn analysis LLM.
    """
    schema = TurnAnalysisResponse.model_json_schema()

    recent_events_formatted = (
        "\n".join([f"- {evt.event_description}" for evt in recent_events])
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
2.  **Recent Key Events:** A list of the last 5 events that were identified.
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
- For each event, you must provide a precise timestamp and a clear, visually-grounded description.
- Crucially, you must also identify which of the provided 'AFTER' frames best illustrates the event and return its exact timestamp in the `representative_timestamp` field.

RULES FOR `event_description`:
------------------------------
- **Be Specific and Visually Grounded:** Your description MUST incorporate specific details from the visual evidence. Mention the text on buttons, links, or page titles seen in the 'AFTER' frame.
- **Focus on GOAL and OUTCOME:** Explain the *semantic meaning* of the event.
- **Maintain a Third-Person Narrative:** (e.g., "User navigates...", "User enters...").

**Examples of Rich vs. Poor Event Descriptions**
    1.  **Clicking a link:**
        - Visual: User clicks a link and a new page titled "Account Settings" loads.
        - **Poor Description:** "User clicked on a link."
        - **Rich Description:** "User navigated to the 'Account Settings' page."

    2.  **Form submission:**
        - Visual: A form is filled out, the user clicks "Submit", and a confirmation "Your profile has been updated." appears.
        - **Poor Description:** "User clicked the submit button."
        - **Rich Description:** "User submitted their updated profile information, displaying a 'profile updated' confirmation."

    3.  **Combined Speech + Visual:**
        - User says: "I need to find the latest invoice from Acme Corp."
        - Visual: The user types "Acme Corp" into a search bar labeled "Search invoices".
        - **Poor Description:** "User searched."
        - **Rich Description:** "User searched for invoices from 'Acme Corp' using the invoice search bar."

RULES FOR `triggering_phrase`:
------------------------------
- **Be Precise:** The `triggering_phrase` MUST be the **shortest possible, exact substring** from the user's speech that refers to the action.
- **If no specific phrase exists, this field MUST be `null`.** Do not invent a phrase.

**Examples of Precise vs. Imprecise Triggering Phrases**
    1.  **User says:** "Okay, I'm going to add the new quarterly report to the folder now."
        - Action: User clicks a button labeled "Upload".
        - **Imprecise Phrase:** "Okay, I'm going to add the new quarterly report to the folder now."
        - **Precise Phrase:** "add the new quarterly report"

    2.  **User says:** "Let's click the red button to continue."
        - Action: User clicks a red "Continue" button.
        - **Imprecise Phrase:** "click the red button to continue"
        - **Precise Phrase:** "click the red button"

FORMATTING RULES:
-----------------
1.  **Representative Timestamp is Mandatory:** For every event, the `representative_timestamp` field must contain the exact timestamp of the corresponding 'AFTER' frame from the input. Do NOT invent timestamps.
2.  **Timestamp Format:** All `timestamp` values must be floating-point numbers representing seconds relative to the start of the media stream (e.g., `12.34`). Do NOT use Unix epoch timestamps.
3.  **Chronological Order:** The final list of events in your response MUST be sorted by timestamp.
4.  **JSON ONLY:** Your entire response must be a single, valid JSON object that strictly conforms to the provided schema. Do not include any other text, notes, or markdown.

SCHEMA FOR YOUR RESPONSE:```json
{json.dumps(schema, indent=2)}```
"""
    return prompt


def build_summary_update_prompt(
    current_summary: str, new_events: List[KeyEvent]
) -> str:
    """
    Builds the system prompt for the summary update LLM.
    """
    new_events_formatted = "\n".join(
        [f"- At t={evt.timestamp:.2f}s: {evt.event_description}" for evt in new_events]
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
