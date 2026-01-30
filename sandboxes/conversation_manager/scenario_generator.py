"""
Scenario seeding helpers for the ConversationManager sandbox.

Notes
-----
We rely on the existing TranscriptGenerator in `sandboxes.utils`, which itself
uses ScenarioBuilder internally to synthesize transcripts and (optionally) update
contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sandboxes.utils import TranscriptGenerator


def _summarize(transcript: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in transcript:
        med = str(m.get("medium") or "unknown")
        counts[med] = counts.get(med, 0) + 1
    return counts


def _format_summary(counts: dict[str, int]) -> str:
    if not counts:
        return "0 messages"
    parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: kv[0])]
    return ", ".join(parts)


@dataclass
class ScenarioGenerator:
    publisher: Any
    state: Any

    async def generate_and_publish(self, description: str) -> None:
        """
        Generate a transcript and publish it into ConversationManager via EventPublisher.

        We publish *inbound* events only, letting CM produce the assistant responses.
        For simplicity, we treat each synthesized message as a user inbound in its medium.
        """
        desc = (description or "").strip()
        if not desc:
            raise ValueError("Scenario description was empty.")

        print("[generate] Building scenario — this can take a moment…")
        gen = TranscriptGenerator(in_conversation_manager=False)
        transcript = await gen.generate(desc, min_messages=10, max_messages=18)

        counts = _summarize(transcript)
        print(
            f"✅ Scenario generated (synthetic transcript): {_format_summary(counts)}",
        )

        # Publish into CM in sequence.
        # If there are any phone_call messages, wrap them in call start/end.
        in_phone = False
        published = 0

        for msg in transcript:
            medium = str(msg.get("medium") or "")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue

            if medium == "phone_call":
                if not in_phone:
                    await self.publisher.publish_call_start()
                    in_phone = True
                    published += 1
                await self.publisher.publish_phone_utterance(content)
                published += 1
                continue

            # If we were in a phone call and the transcript switches away, end the call.
            if in_phone and medium != "phone_call":
                await self.publisher.publish_call_end()
                in_phone = False
                published += 1

            if medium == "email":
                # Accept either "Subject: X\n\nBody" or raw body.
                subj = "Sandbox Email"
                body = content
                if content.lower().startswith("subject:"):
                    head, _, rest = content.partition("\n")
                    subj = head.split(":", 1)[1].strip() or subj
                    body = rest.strip() or ""
                await self.publisher.publish_email(subj, body)
                published += 1
                continue

            # Default: SMS
            await self.publisher.publish_sms(content)
            published += 1

        if in_phone:
            await self.publisher.publish_call_end()
            published += 1
            # no wait here; we don't want scenario seeding to block on CM decisions

        # Best-effort: ensure we return to idle after seeding so the user can continue.
        try:
            self.state.brain_run_in_flight = False
        except Exception:
            pass

        print(f"✅ Published {published} inbound event(s) into ConversationManager.")
