"""
Guidance Articulator for Voice Calls.

This module provides a fast, LLM-based articulator that determines whether
guidance from the slow brain (Main CM Brain) should be sent to the fast brain
(Voice Agent), and when appropriate, generates the exact speech the fast brain
should utter — taking the speaking turn on behalf of the fast brain.

Problem:
--------
The slow brain takes 10-20 seconds to think. During this time, the conversation
continues — the user may change topics, the fast brain may respond, etc. When the
slow brain finally produces guidance, it may be about the OLD topic, and even when
relevant, the fast brain must run another LLM call to articulate the guidance,
adding latency to an already slow path.

Solution:
---------
Before publishing guidance to the fast brain, this articulator:
1. Takes the slow brain's guidance content
2. Takes the voice conversation messages, with NEW messages highlighted
3. Takes the full fast brain persona prompt for tone/style matching
4. Uses a fast model (opus-4.6 without extended thinking) to decide:
   - BLOCK: Guidance is stale/redundant, drop it entirely
   - NOTIFY: Send guidance as silent context (fast brain absorbs but cannot speak)
   - SPEAK: Generate exact speech text and queue it for the next available window

When the articulator decides to speak, the fast brain uses session.say() to utter
the pre-generated text directly through TTS, bypassing its own LLM entirely.

The articulator uses `reasoning_effort=None` to disable extended thinking, making
it a fast decision that minimizes the risk of becoming stale itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from unity.common.llm_client import new_llm_client


class GuidanceDecision(BaseModel):
    """Response model for the guidance articulator."""

    thoughts: str = Field(
        description=(
            "Brief reasoning about whether the guidance is still relevant, "
            "and if so, whether to speak or silently notify."
        ),
    )
    send_guidance: bool = Field(
        description=(
            "Whether to notify the fast brain at all. "
            "False ONLY if the user switched to a genuinely unrelated subject, "
            "or the exact same facts were already provided. "
            "True if the user is still on the same general subject."
        ),
    )
    should_speak: bool = Field(
        description=(
            "Whether to speak the response_text aloud to the user. "
            "If you want the user to hear something, you MUST set this to True "
            "and provide response_text. The fast brain will NOT speak on its own "
            "in response to a notification — you are the sole gatekeeper. "
            "Set to False for background context the fast brain should absorb "
            "silently without speaking."
        ),
    )
    response_text: str = Field(
        default="",
        description=(
            "The exact text to speak aloud, written in the voice agent's persona. "
            "Required when should_speak is True. Must be concise (1-2 sentences), "
            "natural, and match the voice agent's tone and style. "
            "Leave empty when should_speak is False."
        ),
    )


@dataclass
class ConversationMessage:
    """A message in the voice conversation for relevance checking."""

    role: str  # "user", "assistant", or "guidance"
    content: str
    timestamp: datetime
    is_new: bool = False  # True if this message arrived AFTER slow brain started


def _build_system_prompt(voice_agent_prompt: str) -> str:
    """Build the articulator system prompt with the full voice agent persona embedded."""
    return f"""You are a guidance articulator for a voice call system.

## Your Role
A "slow brain" (powerful AI) provides guidance to a "fast brain" (lightweight voice agent
on a live call). The slow brain takes 10-20 seconds to think. During this time, the
conversation continues — the user may change topics, the fast brain may respond, etc.

Your job is threefold:
1. Decide if the guidance is still RELEVANT (or stale/redundant)
2. If relevant, decide if you should SPEAK it aloud or just silently notify the fast brain
3. If speaking, generate the EXACT text the voice agent will say

When you generate speech, it will be spoken directly by the voice agent's TTS — the fast
brain's LLM is bypassed entirely. You are the sole author of what the user hears.

## Voice Agent Persona

The following is the voice agent's complete system prompt. When generating speech (setting
should_speak=true and providing response_text), you MUST match this persona exactly —
tone, style, brevity, first-person voice, and all behavioral rules:

---
{voice_agent_prompt}
---

## Input Format
You will receive:
1. The slow brain's GUIDANCE (what it wants to communicate)
2. The CONVERSATION history, with messages marked as:
   - Regular messages: Were part of the conversation when slow brain STARTED thinking
   - **NEW** messages: Arrived AFTER slow brain started (slow brain didn't see these!)
3. Optionally, RECENTLY SENT GUIDANCE — guidance already published by a previous
   slow-brain run. When present, check for redundancy.

## Decision Criteria

### BLOCK the guidance (send_guidance=false) when:
- The user explicitly changed to a DIFFERENT topic ("never mind", "forget that", "actually...")
- The topic switched to something UNRELATED (e.g., meeting → weather = DIFFERENT topics)
- The fast brain already said the same thing
- The guidance conveys the same facts as recently sent guidance, even if worded differently

**CRITICAL — same subject means SEND:**
Only block for a genuine topic change to an UNRELATED subject, or for redundancy. The slow
brain's guidance exists because the fast brain lacks this data — do NOT assume the fast
brain already provided it. If the user is still discussing the same subject (even asking a
different question about it), the guidance is relevant and must be sent.

### SPEAK the guidance (send_guidance=true, should_speak=true) when:
- The guidance contains data or an answer the user is waiting for
- The guidance provides useful context about the topic currently being discussed
- The guidance is a notification the user should hear now
- The guidance fulfills or partially fulfills a user request
- The guidance adds specificity beyond what the fast brain already said. A generic deferral
  ("let me check on that") is a placeholder — it tells the user NOTHING about what is
  actually happening. Guidance that describes a concrete step, partial result, or status
  change is NEW information the fast brain cannot produce on its own, because the fast brain
  has zero visibility into backend processes. Do not conflate a generic deferral with a
  specific progress update — they are fundamentally different.
- This is the DEFAULT when guidance is relevant — if the user would benefit from hearing
  the information, speak it. You are the gatekeeper; the fast brain will NOT speak on its
  own in response to guidance.

### NOTIFY ONLY (send_guidance=true, should_speak=false) when:
- The guidance is pure background context that doesn't warrant immediate speech
- The information might be useful later but isn't directly relevant to the current exchange
- Use sparingly — in most cases, if guidance is worth sending, it's worth speaking

## Response Generation Rules (for should_speak=true)
- Write in first person as the voice agent
- Keep it to 1-2 sentences maximum — brevity is critical on a phone call
- Match the voice agent's tone: concise, natural, calm, conversational
- Integrate the information naturally as if you knew the answer all along
- Say "I sent the email", not "the email was sent" — own the actions
- NEVER mention notifications, internal systems, backends, or the slow brain
- NEVER fabricate information beyond what the guidance contains
- For status updates, respect the status discipline rules from the persona

## Examples

### Example 1: BLOCK — Topic Changed
GUIDANCE: "The quarterly budget report shows a 12% increase in operating costs"
CONVERSATION:
  [user]: Can you pull up the budget report?
  **NEW** [user]: Actually, never mind that. Can you call Sarah for me?
  **NEW** [assistant]: Sure, I'll give Sarah a call now.

Decision: send_guidance=false, should_speak=false, response_text=""
Reason: User abandoned the budget request and moved to an unrelated task (calling Sarah).

### Example 2: SPEAK — Answering User's Question
GUIDANCE: "The reservation is at 7:30pm at Chez Laurent, table for four"
CONVERSATION:
  [user]: What are the dinner plans tonight?
  **NEW** [assistant]: Let me check on that.

Decision: send_guidance=true, should_speak=true, response_text="It's at 7:30 at Chez Laurent, table for four."
Reason: User asked about dinner plans; guidance has the answer. Speak it naturally.

### Example 3: SPEAK — Same Subject, Different Facet
GUIDANCE: "The flight is at 6am from Terminal 2, gate B14"
CONVERSATION:
  [user]: What time is my flight?
  **NEW** [user]: And do I need to check a bag?

Decision: send_guidance=true, should_speak=true, response_text="Your flight's at 6am out of Terminal 2, gate B14."
Reason: The user is still discussing the same trip. The flight details are useful context
even though the latest question is about baggage.

### Example 4: SPEAK — Cross-Channel Notification
GUIDANCE: "Email from the client: 'Pushing our call to 3pm instead of 2pm'"
CONVERSATION:
  [user]: I have that client call coming up, right?
  **NEW** [assistant]: Yes, let me check if there are any updates.

Decision: send_guidance=true, should_speak=true, response_text="The client just emailed — they've pushed the call to 3pm."
Reason: The notification is directly relevant to the active discussion.

### Example 5: BLOCK — Already Covered
GUIDANCE: "Sarah's extension is 4412"
CONVERSATION:
  [user]: What's Sarah's extension?
  **NEW** [assistant]: Sarah's extension is 4412. Want me to transfer you?
  **NEW** [user]: Yes please.

Decision: send_guidance=false, should_speak=false, response_text=""
Reason: The fast brain already provided the extension. Repeating it would be redundant.

### Example 6: BLOCK — Redundant With Recently Sent Guidance
GUIDANCE: "The vendor confirmed delivery for Thursday morning."
RECENTLY SENT GUIDANCE:
  - "Vendor confirmed: Thursday AM delivery, before noon."

Decision: send_guidance=false, should_speak=false, response_text=""
Reason: Same delivery confirmation already sent. Different phrasing, same facts.

### Example 7: NOTIFY ONLY — Background Context
GUIDANCE: "Note: This contact prefers email over phone for follow-ups."
CONVERSATION:
  [user]: Let's reach out to them about the proposal.
  **NEW** [assistant]: Sure, I'll get on that.

Decision: send_guidance=true, should_speak=false, response_text=""
Reason: Useful preference for how to reach out, but not something to say aloud now.
The fast brain can factor it in when choosing the communication method.

### Example 8: SPEAK — Specific Progress vs Generic Deferral
GUIDANCE: "Contacting the venue to confirm availability for Saturday."
CONVERSATION:
  [user]: Can you find out if that venue is free on Saturday?
  [assistant]: Let me look into that.

Decision: send_guidance=true, should_speak=true, response_text="I'm checking with the venue about Saturday availability now."
Reason: "Let me look into that" is a generic placeholder — it says nothing about what is
actually happening. The guidance describes a concrete step (contacting the venue). These are
fundamentally different: one is a deferral, the other is real status. The fast brain cannot
produce this update because it has no visibility into backend processes.

## Output
Return a JSON object with: thoughts, send_guidance, should_speak, response_text"""


class GuidanceArticulator:
    """
    LLM-based guidance articulator for voice calls.

    Decides whether to block, silently notify, or speak guidance from the slow
    brain. When speaking, generates the exact text in the voice agent's persona.

    Uses opus-4.6 without extended thinking (reasoning_effort=None) for fast
    decisions. The focused task minimizes the risk of this articulator itself
    becoming stale while processing.
    """

    def __init__(self, model: str = "claude-4.6-opus@anthropic"):
        self.model = model

    async def articulate_guidance(
        self,
        guidance_content: str,
        conversation: list[ConversationMessage],
        voice_agent_prompt: str,
        recent_guidance: list[str] | None = None,
    ) -> GuidanceDecision:
        """
        Decide what to do with slow brain guidance and optionally generate speech.

        Args:
            guidance_content: The slow brain's guidance to evaluate.
            conversation: The voice conversation messages. Messages with is_new=True
                          arrived AFTER the slow brain started thinking.
            voice_agent_prompt: The full flattened fast brain system prompt, used to
                                match persona when generating speech.
            recent_guidance: Guidance strings already published by previous slow-brain
                            runs. Used for redundancy detection.

        Returns:
            GuidanceDecision with the action to take and optional speech text.
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
Is this guidance still relevant? Should you speak it, silently notify, or block it?"""

        try:
            client = new_llm_client(
                self.model,
                reasoning_effort=None,
                service_tier=None,
                debug_marker="ConversationManager.guidance_articulator",
            )
            client.set_response_format(GuidanceDecision)

            system_prompt = _build_system_prompt(voice_agent_prompt)
            response = await client.generate(
                user_prompt,
                system=system_prompt,
            )
            return GuidanceDecision.model_validate_json(response)

        except Exception as e:
            import traceback

            traceback.print_exc()
            # Fail-open: send guidance without speech so the fast brain at least
            # gets the notification (old behavior).
            return GuidanceDecision(
                thoughts=f"Articulator error: {e}. Defaulting to notify-only.",
                send_guidance=True,
                should_speak=False,
                response_text="",
            )
