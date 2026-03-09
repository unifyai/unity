from pydantic import BaseModel, Field
from unity.common.llm_client import new_llm_client
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON


class NotificationReply(BaseModel):
    speak: bool = Field(
        description="Whether to say something aloud in response to this notification.",
    )
    content: str = Field(
        default="",
        description="What to say if speaking. Empty when speak is false.",
    )


NOTIFICATION_REPLY_PROMPT = """\
You are deciding whether to speak aloud in response to a new `[notification]` that \
just appeared in the conversation.

Your output will be spoken via TTS if `speak` is true, or silently absorbed if false. \
Choose wisely — every unnecessary utterance is an interruption on a live call.

## When to speak

- The notification contains **concrete data** the caller is waiting for (e.g., search \
results, a looked-up phone number, confirmation that a task is done).
- It is the **first meaningful progress update** for a request the caller just made \
and you haven't acknowledged yet.
- An error or failure occurred that the caller needs to know about.

## When NOT to speak

- You **already said something equivalent** in a recent turn. Look at your last few \
assistant messages — if you already acknowledged this action or gave a similar progress \
update, stay silent. Saying "navigating to costar.com" three times because three \
notifications arrived is exactly the kind of redundancy to avoid.
- The notification is a **trivial, redundant, or purely internal** progress update \
(e.g., an intermediate step of work already acknowledged).
- The notification merely **echoes information you already relayed** — e.g., a \
"desktop_act started" notification for an action you already told the caller about.
- The caller is **actively speaking** or **hasn't finished their thought** — absorb \
silently and incorporate later if relevant.

## If you decide to speak

- Keep it to one short sentence (this is a phone call, not a chat).
- Use natural phrasing — contractions, casual tone.
- Never reference notifications, systems, or internal processes.
- Speak in first person: "I found three results" not "three results were found."
- For in-progress computer/desktop actions, calibrate time-framing to the task. Quick \
actions (a single click, navigation) warrant short acknowledgments ("give me a moment", \
"one sec"). Multi-step work (drafting emails, research) warrants realistic expectations \
("working on it — might take a few minutes"). Either way, don't narrate specific steps \
("opening Chrome", "clicking on that", "navigating there") — the caller may be watching \
the screen, and narrating actions that haven't visibly happened yet sounds premature.

Output JSON matching the NotificationReply schema.\
"""


class NotificationReplyEvaluator:
    """Evaluates whether the fast brain should speak in response to a notification.

    Uses a structured-output LLM call (sidecar to the main voice pipeline) to make
    an explicit speak/no-speak decision, replacing the brittle "output empty string"
    approach.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    async def evaluate(
        self,
        chat_history: list[dict],
        system_prompt: str,
    ) -> tuple[NotificationReply, str]:
        """Decide whether to speak in response to the latest notification(s).

        Returns (decision, llm_log_path).
        """
        try:
            client = new_llm_client(
                self._model,
                origin="FastBrain.notification_reply",
            )
            client.set_response_format(NotificationReply)
            messages = [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\n{NOTIFICATION_REPLY_PROMPT}",
                },
                *chat_history,
            ]
            response = await client.generate(messages=messages)
            log_path = ""
            pending = getattr(client, "_pending_thinking_log", None)
            if pending is not None:
                log_path = pending.last_path or ""
            return NotificationReply.model_validate_json(response), log_path
        except Exception as e:
            LOGGER.error(f"{DEFAULT_ICON} Error in NotificationReply evaluation: {e}")
            import traceback

            traceback.print_exc()
            return NotificationReply(speak=True, content=""), ""
