from pydantic import BaseModel, Field
from unity.common.llm_client import new_llm_client
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON


class ProactiveDecision(BaseModel):
    should_speak: bool = Field(
        description="Whether the assistant should speak proactively.",
    )
    delay: int = Field(
        default=5,
        description="How long to wait (in seconds) before speaking.",
    )
    content: str | None = Field(default=None, description="What to say if speaking.")


PROACTIVE_PROMPT = """\
You are deciding whether to break a silence during a live phone call.

There has been at least 5 seconds of silence since the last utterance (user or \
assistant). You have the full conversation history above.

## When to speak

- The conversation feels like it stalled or the other person seems to be waiting.
- The assistant acknowledged a request and time has passed with no follow-up.
- A long awkward silence needs to be filled with a brief, natural check-in.

## When NOT to speak

- The assistant just asked a question and the user is likely thinking.
- The conversation is wrapping up (goodbyes were exchanged).
- The user explicitly asked to wait or said they need a moment.
- The assistant already set a time expectation ("it might take a few minutes") and \
no new information has arrived. Repeating filler like "bear with me" or "shouldn't \
be too much longer" adds no value — wait for actual results before speaking again.

## Action awareness

You may be given an `[action status]` block listing actions that are currently \
executing or recently completed. This is the ground truth for what has and hasn't \
happened. NEVER claim an in-flight action is finished. If the assistant said "one \
moment" and the action is still executing, a brief patience-acknowledging reassurance \
is fine — "still working on it, should just be a few more minutes", "this one's \
taking a little while but I'm on it" — but do NOT narrate specific steps ("opening \
the browser", "clicking on that") or claim the action is done. Avoid short-wait \
filler like "bear with me" or "shouldn't be too much longer" — these imply the task \
is nearly instant.

## If you decide to speak

- `delay`: additional seconds to wait before speaking (0 = now, higher = more patient). \
Use this to express how urgent the silence-fill is.
- `content`: a short, natural sentence (1-2 sentences max). Vary phrasing -- never \
repeat what was already said in the transcript. Do not claim specific actions you are \
not actually performing.

Output JSON matching the ProactiveDecision schema.\
"""


class ProactiveSpeech:
    async def decide(
        self,
        chat_history: list[dict],
        system_prompt: str,
        action_context: str | None = None,
    ) -> tuple[ProactiveDecision, str]:
        """Decides whether to speak proactively based on the conversation history.

        Returns (decision, llm_log_path) where llm_log_path is the unillm
        request+response file for the LLM call that produced this decision.
        """
        try:
            client = new_llm_client(
                origin="ProactiveSpeech",
            )
            client.set_response_format(ProactiveDecision)
            messages = [
                {"role": "system", "content": f"{system_prompt}\n\n{PROACTIVE_PROMPT}"},
                *chat_history,
            ]
            if action_context:
                messages.append(
                    {"role": "system", "content": action_context},
                )
            response = await client.generate(messages=messages)
            log_path = ""
            pending = getattr(client, "_pending_thinking_log", None)
            if pending is not None:
                log_path = pending.last_path or ""
            return ProactiveDecision.model_validate_json(response), log_path
        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error in ProactiveSpeech decision: {e}")
            import traceback

            traceback.print_exc()
            return ProactiveDecision(should_speak=False), ""
