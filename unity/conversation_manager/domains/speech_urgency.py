from pydantic import BaseModel, Field

from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.common.llm_client import new_llm_client
from unity.logger import LOGGER


class SpeechUrgency(BaseModel):
    urgent: bool = Field(
        description=(
            "Whether the user's utterance requires immediate action, "
            "warranting preemption of the current slow brain run."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision.",
    )


SPEECH_URGENCY_PROMPT = """\
You are deciding whether a new user utterance on a live voice call is **urgent** \
enough to preempt the currently running slow brain.

The slow brain is an AI orchestrator that processes events and executes actions. Each \
run takes 10-25 seconds. A new user utterance has arrived while the slow brain is \
mid-run on a previous event.

If you mark the utterance as urgent, the current slow brain run will be cancelled and \
the user's new request will be processed immediately. If you mark it as not urgent, the \
current run will be allowed to finish and the user's utterance will be processed next \
in the queue.

## Current slow brain state

- **Triggered by**: {origin_event}
- **Running for**: {elapsed:.0f}s
- **In-flight actions**: {actions_summary}

## New user utterance

"{utterance}"
{previous_utterance_section}
## Mark as URGENT when

- The user is giving a **new actionable directive** that requires immediate execution \
(e.g., "open the browser", "click that button", "send the email", "go to costar.com").
- The user is **explicitly cancelling or redirecting** the current work ("stop", \
"never mind, do X instead", "actually, forget that").
- The slow brain is stuck processing a **low-priority system event** (e.g., \
WebCamStarted, ActorHandleStarted, ScreenShareStarted) while the user has a concrete \
request that needs action.

## Mark as NOT URGENT when

- The user is making **small talk**, **checking in**, or **acknowledging** ("how's it \
going?", "sounds good", "okay", "thanks", "cool", "got it").
- The user is providing **additional context or clarification** for work that is already \
in progress (e.g., "make sure it's Chrome" when the browser is already being opened).
- The slow brain is already processing a **user-initiated action** that is also \
time-sensitive — let it finish rather than restarting.
- The utterance is **conversational filler** or a **partial thought** that does not \
constitute a standalone directive.

## Continuation awareness

A short or fragmentary utterance may be the **second half of something the user was \
already saying** in the previous turn. If a previous utterance is shown, consider whether \
the new utterance completes, corrects, or extends it. A continuation that carries \
critical information (e.g., the rest of a dictated value, a spelled-out credential, \
the second half of an address) is urgent — the slow brain is currently processing only \
the first half and will make a wrong or incomplete decision without the rest.

Output JSON matching the SpeechUrgency schema.\
"""


class SpeechUrgencyEvaluator:
    """Concurrent sidecar that decides whether to preempt the slow brain.

    Runs a structured-output LLM call against the fast brain model to classify
    a user utterance as urgent (preempt) or not (let the queue proceed normally).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def evaluate(
        self,
        utterance: str,
        origin_event: str,
        elapsed_seconds: float,
        actions_summary: str,
        previous_utterance: str | None = None,
    ) -> SpeechUrgency:
        """Classify whether *utterance* warrants preempting the slow brain.

        Returns a SpeechUrgency decision.
        """
        try:
            client = new_llm_client(
                self._model,
                origin="FastBrain.speech_urgency",
            )
            client.set_response_format(SpeechUrgency)
            if previous_utterance:
                prev_section = (
                    f"\n## Previous user utterance (for context)\n\n"
                    f'"{previous_utterance}"\n\n'
                )
            else:
                prev_section = ""
            system_content = SPEECH_URGENCY_PROMPT.format(
                origin_event=origin_event or "unknown",
                elapsed=elapsed_seconds,
                actions_summary=actions_summary or "none",
                utterance=utterance,
                previous_utterance_section=prev_section,
            )
            # Anthropic (via litellm) rejects requests with only system messages;
            # at least one user turn is required.
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        "Apply the instructions above and output JSON matching "
                        "the SpeechUrgency schema."
                    ),
                },
            ]
            response = await client.generate(messages=messages)
            return SpeechUrgency.model_validate_json(response)
        except Exception as e:
            LOGGER.error(
                f"{DEFAULT_ICON} Error in SpeechUrgency evaluation: {e}",
            )
            import traceback

            traceback.print_exc()
            return SpeechUrgency(urgent=False, reasoning="evaluation failed")
