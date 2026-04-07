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
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decision.",
    )


SPEECH_DEDUP_PROMPT = """\
You are deciding whether proposed speech from the slow brain should actually be \
spoken on a live voice call, or whether the fast brain has already communicated \
the same information.

The system has two brains running in parallel:
- **Fast brain**: handles real-time conversation, responds instantly to user speech.
- **Slow brain**: processes events and decides what to say, but takes 10-25 seconds.

During the slow brain's thinking time, the user may have asked a question and the \
fast brain may have already answered it using background context. The slow brain's \
proposed speech may now be redundant.

## Recent assistant utterances (what the user has already heard)

{recent_utterances}

## Proposed speech from slow brain

"{proposed_speech}"

## Mark as already_covered=True when

- The recent utterances already convey the same core information or conclusion
- The user has already been told about the completion, result, or status that the \
proposed speech covers
- Speaking this would be noticeably redundant to the listener

## Mark as already_covered=False when

- The proposed speech contains meaningfully new information not in recent utterances
- The proposed speech adds important detail beyond what was said
- No recent utterances address the same topic at all
- The overlap is only superficial (e.g., both mention the same person but discuss \
different things about them)

Output JSON matching the SpeechDedup schema.\
"""


class SpeechDeduplicationChecker:
    """Post-LLM gate that suppresses slow brain speech already covered by the fast brain.

    After the slow brain's LLM call returns with guide_voice_agent(should_speak=True),
    this checker compares the proposed response_text against recent assistant
    utterances in the voice transcript. If the fast brain already communicated
    the same information reactively, the speech is downgraded to a silent
    notification (should_speak=False) to avoid redundancy.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def evaluate(
        self,
        proposed_speech: str,
        recent_utterances: list[str],
    ) -> SpeechDedup:
        """Check whether *proposed_speech* overlaps with *recent_utterances*.

        Returns ``SpeechDedup(already_covered=False)`` immediately when there
        are no recent utterances to compare against (no LLM call needed).
        On error, fails open (allows speech).
        """
        if not recent_utterances:
            return SpeechDedup(
                already_covered=False,
                reasoning="no recent assistant utterances to compare against",
            )

        try:
            client = new_llm_client(
                self._model,
                origin="SlowBrain.speech_dedup",
            )
            client.set_response_format(SpeechDedup)
            formatted_utterances = "\n".join(f'- "{u}"' for u in recent_utterances)
            system_content = SPEECH_DEDUP_PROMPT.format(
                recent_utterances=formatted_utterances,
                proposed_speech=proposed_speech,
            )
            # Anthropic (via litellm) rejects requests with only system messages;
            # at least one user turn is required.
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
