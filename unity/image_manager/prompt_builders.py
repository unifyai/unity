from __future__ import annotations

from datetime import datetime
from ..common.prompt_helpers import now


import textwrap


def build_image_ask_prompt(*, caption: str | None, timestamp: datetime | None) -> str:
    """
    Return a concise system message for image Q&A.

    Notes
    -----
    - The image itself will be provided as an image_url content block in the
      user message. The model should reason over it but respond with text only.
    - Include lightweight context when available (timestamp, caption).
    """

    ts = timestamp.isoformat() if timestamp else "unknown-time"
    cap = caption or "(no caption provided)"

    msg = textwrap.dedent(
        f"""
        You are a helpful vision assistant. An image will be provided in the next user message.

        Context
        -------
        • Timestamp: {ts}
        • Caption: {cap}

        Requirements
        -----------
        • Look at the provided image and answer the user's question clearly and concisely.
        • Respond with plain text only. Do NOT attempt to call tools or output JSON/function calls.
        • Do not include raw image data or base64 in your response.
        • If relevant, you may cite visual evidence (colors, objects, text seen).
        • If uncertain, state the uncertainty and what would resolve it.
        • Do not assume system-specific identifiers or structured record fields (e.g., ids, names, statuses, queue/thread
          references, timestamps) are present. If the user's question asks for such fields and they are not visibly shown,
          state that they are not visible and describe what is visible instead.
        """,
    ).strip()
    # Append current time for reproducibility
    msg = f"{msg}\n\nCurrent UTC time is {now()}."
    return msg
