from pydantic import BaseModel, Field

from unity.common.hierarchical_logger import DEFAULT_ICON
from unity.common.llm_client import new_llm_client
from unity.logger import LOGGER


class SpeechDedup(BaseModel):
    already_covered: bool = Field(
        description=(
            "Whether the proposed speech has already been communicated "
            "to the user by recent assistant utterances on this call."
        ),
    )
    contradicts_current_state: bool = Field(
        default=False,
        description=(
            "Whether the proposed speech contradicts recent assistant "
            "utterances or recent notification state (e.g., offering setup "
            "steps when a notification confirmed setup is complete)."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision.",
    )

    @property
    def should_suppress(self) -> bool:
        return self.already_covered or self.contradicts_current_state


SPEECH_DEDUP_PROMPT = """\
You are deciding whether proposed speech from the slow brain should actually be \
spoken on a live voice call.

The system has two brains running in parallel:
- **Fast brain**: handles real-time conversation, responds instantly to user speech.
- **Slow brain**: processes events and decides what to say, but takes 10-25 seconds.

During the slow brain's thinking time, the fast brain may have already answered, \
OR new notifications may have arrived that make the proposed speech stale or wrong.

## Recent assistant utterances (what the user has already heard)

{recent_utterances}

## Recent notifications (authoritative system state)

{recent_notifications}

## Proposed speech from slow brain

"{proposed_speech}"

## Suppress (already_covered=True) when

- The recent utterances already convey the same core information or conclusion
- The user has already been told about the completion, result, or status that the \
proposed speech covers
- Speaking this would be noticeably redundant to the listener

## Suppress (contradicts_current_state=True) when

- The proposed speech offers to walk through, set up, or redo steps that a recent \
notification or utterance confirms are already complete
- The proposed speech claims something is needed when a notification says it is done
- The proposed speech gives a next step for a workflow that notifications show has \
already progressed past that step
- The proposed speech contradicts the most recent assistant utterance (e.g., one says \
"everything is done" and the proposed speech says "want me to help set it up?")

## Allow (both False) when

- The proposed speech contains meaningfully new information not in recent utterances
- The proposed speech adds important detail beyond what was said
- No recent utterances or notifications address the same topic at all
- The overlap is only superficial (e.g., both mention the same person but discuss \
different things about them)

Output JSON matching the SpeechDedup schema.\
"""


class SpeechDeduplicationChecker:
    """Pre-speak gate that suppresses slow brain speech that is redundant or
    contradicts the current conversation state.

    Runs in the fast brain subprocess at speak time (inside ``maybe_speak_queued``).
    Compares the proposed ``response_text`` against recent assistant utterances
    and recent notifications in the fast brain's chat context. Suppresses speech
    that is redundant, stale, or contradictory.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def evaluate(
        self,
        proposed_speech: str,
        recent_utterances: list[str],
        recent_notifications: list[str] | None = None,
    ) -> SpeechDedup:
        """Check whether *proposed_speech* should be suppressed.

        Returns ``SpeechDedup(already_covered=False)`` immediately when there
        are no recent utterances to compare against (no LLM call needed).
        On error, fails open (allows speech).
        """
        if not recent_utterances and not recent_notifications:
            return SpeechDedup(
                already_covered=False,
                reasoning="no recent context to compare against",
            )

        try:
            client = new_llm_client(
                self._model,
                origin="FastBrain.speech_dedup",
            )
            client.set_response_format(SpeechDedup)
            formatted_utterances = (
                "\n".join(f'- "{u}"' for u in recent_utterances)
                if recent_utterances
                else "(none)"
            )
            formatted_notifications = (
                "\n".join(f"- {n}" for n in recent_notifications)
                if recent_notifications
                else "(none)"
            )
            system_content = SPEECH_DEDUP_PROMPT.format(
                recent_utterances=formatted_utterances,
                recent_notifications=formatted_notifications,
                proposed_speech=proposed_speech,
            )
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        "Apply the instructions above and output JSON matching "
                        "the SpeechDedup schema."
                    ),
                },
            ]
            response = await client.generate(messages=messages)
            return SpeechDedup.model_validate_json(response)
        except Exception as e:
            LOGGER.error(
                f"{DEFAULT_ICON} Error in SpeechDedup evaluation: {e}",
            )
            import traceback

            traceback.print_exc()
            return SpeechDedup(
                already_covered=False,
                reasoning="evaluation failed",
            )
