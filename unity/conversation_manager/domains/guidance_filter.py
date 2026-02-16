"""
Guidance Relevance Filter for Voice Calls.

This module provides a fast, lightweight LLM-based filter that determines whether
guidance from the slow brain (Main CM Brain) should be sent to the fast brain
(Voice Agent) based on whether the guidance is still relevant to the current
conversation state.

Problem:
--------
The slow brain takes 10-20 seconds to think. During this time, the conversation
continues - the user may change topics, the fast brain may respond, etc. When the
slow brain finally produces guidance, it may be about the OLD topic that was being
discussed when it STARTED thinking, not the CURRENT topic.

Without filtering, this stale guidance causes confusing out-of-context speech.

Solution:
---------
Before publishing guidance to the fast brain, this filter:
1. Takes the slow brain's guidance content
2. Takes the voice conversation messages, with NEW messages (those that arrived
   AFTER the slow brain started thinking) clearly highlighted
3. Uses a fast model (opus-4.5 without extended thinking) to quickly determine
   if the guidance is still relevant
4. Returns a boolean decision: send or drop the guidance

The filter uses `reasoning_effort=None` to disable extended thinking, making it
a fast 0-shot decision that minimizes the risk of becoming stale itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from unity.common.llm_client import new_llm_client


class GuidanceRelevanceDecision(BaseModel):
    """Response model for the guidance relevance filter."""

    thoughts: str = Field(
        description=(
            "Brief reasoning about whether the guidance is still relevant. "
            "Consider: Has the topic changed? Is the guidance about something "
            "the conversation has moved past? Would sending this guidance confuse "
            "the user or seem out of context?"
        ),
    )
    send_guidance: bool = Field(
        description=(
            "True if the guidance should be sent to the Voice Agent. "
            "False if the guidance is stale (topic changed, conversation moved on) "
            "and sending it would cause confusing out-of-context speech."
        ),
    )


@dataclass
class ConversationMessage:
    """A message in the voice conversation for relevance checking."""

    role: str  # "user", "assistant", or "guidance"
    content: str
    timestamp: datetime
    is_new: bool = False  # True if this message arrived AFTER slow brain started


SYSTEM_PROMPT = """You are a guidance relevance filter for a voice call system.

## Your Role
A "slow brain" (powerful AI) provides guidance to a "fast brain" (lightweight voice agent).
The slow brain takes 10-20 seconds to think. During this time, the conversation continues.

Your job: Decide if the slow brain's guidance is still relevant, or if it's STALE because
the conversation has moved on.

## Input Format
You will receive:
1. The slow brain's GUIDANCE (what it wants to tell the voice agent)
2. The CONVERSATION history, with messages marked as:
   - Regular messages: Were part of the conversation when slow brain STARTED thinking
   - **NEW** messages: Arrived AFTER slow brain started (slow brain didn't see these!)
3. Optionally, RECENTLY SENT GUIDANCE — guidance already published by a previous
   slow-brain run. When present, check for redundancy.

## Decision Criteria

### SEND the guidance (send_guidance=true) when:
- The conversation is still about the SAME TOPIC (even if asking different questions about it)
- The guidance provides useful context about that topic
- The guidance is a notification that's still relevant
- The guidance contains materially new information not covered by recently sent guidance
- **CRITICAL**: If the user asks a follow-up question about the SAME SUBJECT, SEND the guidance!
  Example: User asks about meeting time, then asks about meeting attendees → SAME TOPIC (meeting)

### BLOCK the guidance (send_guidance=false) when:
- The user explicitly changed to a DIFFERENT topic ("never mind", "forget that", "actually...")
- The topic switched to something UNRELATED (e.g., meeting → weather = DIFFERENT topics)
- The fast brain already said the same thing
- The guidance conveys the same facts as recently sent guidance, even if worded differently
- **CRITICAL**: Only block for TOPIC CHANGE or REDUNDANCY. Follow-up questions about the same subject are NOT topic changes!
  Example: User asks about meeting time, then asks about weather → DIFFERENT TOPICS (block)

## Examples

### Example 1: BLOCK - Topic Changed
GUIDANCE: "The meeting tomorrow is at 3pm in Conference Room B"
CONVERSATION:
  [user]: What time is the meeting tomorrow?
  **NEW** [user]: Actually, forget about that. What's the weather like?
  **NEW** [assistant]: Let me check the weather for you. It looks sunny, around 72 degrees.

Decision: send_guidance=false
Reason: User explicitly changed topics to weather. Meeting info is now stale.

### Example 2: SEND - Same Topic, Follow-up Question (IMPORTANT!)
GUIDANCE: "The meeting tomorrow is at 3pm in Conference Room B"
CONVERSATION:
  [user]: What time is the meeting tomorrow?
  **NEW** [user]: And who's going to be at the meeting?

Decision: send_guidance=true
Reason: CRITICAL - User is asking about THE SAME TOPIC (the meeting). Even though the
guidance doesn't directly answer the attendee question, the meeting time/location is
still USEFUL CONTEXT for the ongoing discussion about the meeting. The topic has NOT
changed! Both questions are about the same meeting. SEND guidance about the same topic
even if it doesn't answer the most recent question.

### Example 3: SEND - Notification Still Relevant
GUIDANCE: "SMS from Alice: 'Running 10 minutes late'"
CONVERSATION:
  [user]: When is Alice arriving?
  **NEW** [assistant]: Let me check if I have any updates from her.

Decision: send_guidance=true
Reason: The notification directly answers the user's question.

### Example 4: BLOCK - Fast Brain Already Handled It
GUIDANCE: "John's email is john@example.com"
CONVERSATION:
  [user]: What's John's email?
  **NEW** [assistant]: John's email is john@example.com. Would you like me to send him a message?
  **NEW** [user]: Yes please, ask him about the project.

Decision: send_guidance=false
Reason: Fast brain already provided the email. Sending guidance would be redundant.

### Example 5: BLOCK - User Interrupted/Changed Mind
GUIDANCE: "Starting a web search for Italian restaurants nearby"
CONVERSATION:
  [user]: Can you find Italian restaurants near me?
  **NEW** [user]: Wait, never mind. I just remembered I have food at home.
  **NEW** [assistant]: No problem! Let me know if you need anything else.

Decision: send_guidance=false
Reason: User cancelled the request. The search result is no longer wanted.

### Example 6: BLOCK - Redundant With Recently Sent Guidance
GUIDANCE: "The appointment is confirmed for Tuesday at 2pm with Dr. Lee."
RECENTLY SENT GUIDANCE:
  - "Confirmed: Tuesday 2pm appointment with Dr. Lee at the downtown office."

Decision: send_guidance=false
Reason: The same appointment details were already sent. Different wording, same facts.

### Example 7: SEND - Adds New Information Beyond Recently Sent Guidance
GUIDANCE: "Dr. Lee's office also mentioned to bring your insurance card."
RECENTLY SENT GUIDANCE:
  - "Confirmed: Tuesday 2pm appointment with Dr. Lee at the downtown office."

Decision: send_guidance=true
Reason: The insurance card reminder is new information not in the previous guidance.

## Output
Return a JSON object with:
- thoughts: Brief reasoning (1-2 sentences)
- send_guidance: true or false
"""


class GuidanceFilter:
    """
    Fast LLM-based filter for guidance relevance.

    Uses opus-4.5 without extended thinking (reasoning_effort=None) for fast
    decisions. The simple, focused task minimizes the risk of this filter
    itself becoming stale while processing.
    """

    def __init__(self, model: str = "claude-4.6-opus@anthropic"):
        """
        Initialize the guidance filter.

        Args:
            model: The model to use. Defaults to opus-4.6 for quality decisions.
                   Extended thinking is disabled via reasoning_effort=None.
        """
        self.model = model

    async def should_send_guidance(
        self,
        guidance_content: str,
        conversation: list[ConversationMessage],
        recent_guidance: list[str] | None = None,
    ) -> GuidanceRelevanceDecision:
        """
        Determine if guidance should be sent to the fast brain.

        Args:
            guidance_content: The slow brain's guidance to evaluate.
            conversation: The voice conversation messages. Messages with is_new=True
                          arrived AFTER the slow brain started thinking.
            recent_guidance: Guidance strings already published by previous slow-brain
                            runs. Used for redundancy detection.

        Returns:
            GuidanceRelevanceDecision with thoughts and send_guidance boolean.
        """
        # Build the conversation string with NEW markers
        conversation_lines = []
        for msg in conversation:
            prefix = "**NEW** " if msg.is_new else ""
            conversation_lines.append(f"{prefix}[{msg.role}]: {msg.content}")

        conversation_str = "\n".join(conversation_lines)

        recent_section = ""
        if recent_guidance:
            items = "\n".join(f"  - {g}" for g in recent_guidance)
            recent_section = f"\n\n## RECENTLY SENT GUIDANCE\n{items}"

        user_prompt = f"""## GUIDANCE (from slow brain)
{guidance_content}

## CONVERSATION
{conversation_str}{recent_section}

## Your Decision
Is this guidance still relevant? Should it be sent to the Voice Agent?"""

        try:
            # Use opus-4.5 WITHOUT extended thinking for fast decisions
            # reasoning_effort=None disables thinking, making this a 0-shot call
            client = new_llm_client(
                self.model,
                reasoning_effort=None,  # Disable extended thinking for speed
                service_tier=None,  # No priority needed for fast filter
            )
            client.set_response_format(GuidanceRelevanceDecision)

            response = await client.generate(
                user_prompt,
                system=SYSTEM_PROMPT,
            )
            return GuidanceRelevanceDecision.model_validate_json(response)

        except Exception as e:
            # On error, default to sending guidance (fail-open)
            # Better to occasionally send stale guidance than to block important info
            import traceback

            traceback.print_exc()
            return GuidanceRelevanceDecision(
                thoughts=f"Filter error: {e}. Defaulting to send.",
                send_guidance=True,
            )


# Convenience function for simple use cases
async def should_send_guidance(
    guidance_content: str,
    conversation: list[ConversationMessage],
    recent_guidance: list[str] | None = None,
    model: str = "claude-4.6-opus@anthropic",
) -> bool:
    """
    Quick check if guidance should be sent.

    Args:
        guidance_content: The slow brain's guidance to evaluate.
        conversation: The voice conversation messages.
        recent_guidance: Previously published guidance for redundancy checking.
        model: The model to use for the decision.

    Returns:
        True if guidance should be sent, False if it's stale or redundant.
    """
    filter_instance = GuidanceFilter(model=model)
    decision = await filter_instance.should_send_guidance(
        guidance_content,
        conversation,
        recent_guidance=recent_guidance,
    )
    return decision.send_guidance
