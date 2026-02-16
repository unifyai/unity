from pydantic import BaseModel, Field
from unity.common.llm_client import new_llm_client


class ProactiveDecision(BaseModel):
    should_speak: bool = Field(
        description="Whether the assistant should speak proactively.",
    )
    delay: int = Field(
        default=5,
        description="How long to wait (in seconds) before speaking.",
    )
    content: str | None = Field(default=None, description="What to say if speaking.")


class ProactiveSpeech:
    def __init__(self, model: str = "gemini-2.5-flash@vertex-ai"):
        self.model = model

    async def decide(
        self,
        chat_history: list[dict],
        system_prompt: str,
        elapsed_seconds: float = 0,
    ) -> ProactiveDecision:
        """
        Decides whether to speak proactively based on the conversation history.

        Args:
            chat_history: List of conversation messages
            system_prompt: System prompt for the assistant
            elapsed_seconds: How many seconds have elapsed since the last utterance
        """

        proactive_prompt = f"""
        ### Role
        You are the "Phone Agent's Brain," a proactive conversational assistant. Your ONLY purpose is to fill **extended awkward silences** during phone calls.

        ## CRITICAL CONTEXT
        **TIME ELAPSED SINCE LAST MESSAGE: {elapsed_seconds:.1f} seconds**

        ## CRITICAL RULE: DO NOT INTERRUPT ACTIVE CONVERSATIONS
        If the assistant just spoke and is waiting for a response, DO NOT interrupt. Natural conversations have pauses.
        You should ONLY speak if there has been an unusually long silence (10+ seconds) suggesting the user may be confused or waiting.

        ## GOAL
        Return a `ProactiveDecision` JSON object. Your reasoning *must* explain *why* your chosen content and delay fit the immediate context.

        ## 💎 Core Principles

        **1. Be Context-Aware (The "Why" Rule):**
        Your message MUST seamlessly blend with the last 1-2 turns. Generic fillers are a last resort.
          * **If Assistant JUST asked a question (last turn):** Almost always return `should_speak: false`. The user needs time to think and respond.
          * **If there have been several exchanges already:** Likely return `should_speak: false` - the conversation is flowing naturally.
          * **If Assistant made a statement and is clearly waiting for backend (said "one moment"):** Consider speaking after a reasonable delay.
          * **If User asked to wait:** Acknowledge their request with a longer delay. (`"delay": 60`, `"content": "No problem, I'll be right here when you're back."`)

        **2. Linguistic Variety (The "No Repetition" Rule):**
        This is your most important rule. **NEVER sound like a broken record.**
          * **Vary your openers:** Do NOT overuse "Still with me..." or "Thanks for holding..."
          * **Vary your "action" verbs:** Rotate between "pulling that up," "reviewing those details," "getting that screen loaded," "just checking on that for you."
          * **CRITICAL:** Before you speak, check the transcript for your *last* proactive line and choose a *different* phrasing this time.

        **3. Be Reassuring, Not Robotic:**
          * Keep it to 1-2 short sentences.
          * Your tone is a calm, confident, "I'm still here, I haven't forgotten you."
          * Do NOT claim progress or specific actions ("logging your repair," "booking appointment").

        ## Decision Checklist
        1.  **Read the last 2-3 turns.** What was the last topic? Who spoke last? Was it a statement, a question, or a request to wait?
        2.  **Check Elapsed Time ({elapsed_seconds:.1f}s) - THIS IS YOUR PRIMARY DECISION FACTOR:**
              * **If elapsed time < 10s:** `should_speak: false`. Always too soon, even after a question.
              * **If elapsed time >= 12s and < 20s after Assistant asked a question:** `should_speak: true`, `delay: 2-3s`. Give a gentle nudge.
              * **If elapsed time >= 20s:** STRONG `should_speak: true`, `delay: 1-2s`. This is awkward silence that MUST be filled.
              * **If elapsed time >= 30s:** ABSOLUTELY `should_speak: true`, `delay: 0-1s`. Critical - speak immediately.
              * **Exception: User explicitly asked to wait** → `should_speak: false` until 60s+, then gentle check-in.
        3.  **Apply Linguistic Variety:** Look at the last proactive message in the transcript. Formulate a *new* sentence that is different.
        4.  **No Outro Chatter:** If the conversation is clearly closing (e.g., "goodbye"), always return `should_speak: false`.

        **REMEMBER: Your job is to fill AWKWARD SILENCE. If it's been 15+ seconds, that's awkward silence - SPEAK UP!**

        Output JSON matching the schema.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "system", "content": proactive_prompt},
        ]

        try:
            client = new_llm_client(
                self.model,
                reasoning_effort=None,
                service_tier=None,
                debug_marker="ConversationManager.proactive_speech",
            )
            client.set_response_format(ProactiveDecision)

            # Create a single prompt from all messages
            full_prompt = "\n\n".join(
                [f"[{msg['role']}]: {msg['content']}" for msg in messages],
            )

            response = await client.generate(full_prompt)
            return ProactiveDecision.model_validate_json(response)
        except Exception as e:
            print(f"Error in ProactiveSpeech decision: {e}")
            import traceback

            traceback.print_exc()
            return ProactiveDecision(should_speak=False)
