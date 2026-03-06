from __future__ import annotations

import json

from pydantic import BaseModel
from unity.common.llm_client import new_llm_client


def context_over_threshold(
    n_tokens: int,
    threshold: float,
    max_input_tokens: int,
) -> bool:
    return n_tokens >= max_input_tokens * threshold


COMPRESSION_PROMPT = (
    "You are a context compactor. You receive a conversation transcript and must "
    "produce a shorter version that preserves all essential information.\n"
    "\n"
    "Rules:\n"
    "- You MUST return exactly one entry per input message. Never drop or merge messages.\n"
    "- Compact verbose text to its essential meaning. Remove filler, pleasantries, and "
    "redundant explanations.\n"
    '- Error messages and tracebacks: reduce to "error" or "error: <one-line cause>" '
    "only when the specific error identity is not needed for subsequent reasoning.\n"
    "- Tool results: keep only the data that was actually used or referenced later. "
    "Discard decorative formatting, repeated schema keys, and unreferenced fields.\n"
    '- Image placeholders: these have already been replaced with "[N image(s) provided]". '
    "Keep the placeholder text as-is.\n"
    "- Assistant messages that contained tool_calls: compact to the tool names and key "
    'arguments, e.g. search(name="John"), filter(field="city")\n'
    "- Assistant messages with BOTH text and tool_calls: if the text is just narration "
    '("Let me look that up", "I\'ll search for that"), discard it and keep only the '
    "tool call summary. If the text contains reasoning or decisions that inform later "
    "steps, compact it and prepend it to the tool call summary, "
    'e.g. "need both fields. search(name=\\"John\\")"\n'
    '- Every output content must be non-empty. Use "ok" for acknowledgements, "error" '
    "for failures, or a minimal factual summary for everything else.\n"
    "- Do NOT invent information that was not in the original message."
)


# ── Sentinel returned by the loop when compression is requested ──────────────

_COMPRESSION_SIGNAL = object()


# ── Marker tool exposed to the loop LLM ─────────────────────────────────────


def compress_context() -> str:
    """Compress the conversation history to free up context window space.

    This tool is always visible. When the context window is nearly full, this
    becomes the **only** available tool and you **must** call it.

    Cannot be called while other tools are in-flight -- finish or stop all
    running tools first.
    """
    return "compression acknowledged"


def prepare_messages_for_compression(messages: list[dict]) -> list[dict]:
    """Strip expensive binary content before sending to the compression LLM.

    Handles two categories:
    1. Image blocks (``image`` / ``image_url``) -> ``[N image(s) provided]``
    2. Thinking blocks -> removed entirely
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        image_count = 0
        new_blocks: list[dict] = []
        for block in content:
            btype = block.get("type", "")
            if btype in ("image", "image_url"):
                image_count += 1
            elif btype == "thinking":
                continue
            else:
                new_blocks.append(block)

        if image_count:
            label = f"[{image_count} image(s) provided]"
            new_blocks.append({"type": "text", "text": label})

        new_msg = {k: v for k, v in msg.items() if k != "content"}
        new_msg["content"] = new_blocks
        result.append(new_msg)

    return result


class CompressedMessage(BaseModel):
    role: str
    content: str


class CompressedMessages(BaseModel):
    messages: list[CompressedMessage]


def _serialize_messages_for_prompt(messages: list[dict]) -> str:
    """Serialize messages into a compact text representation for the LLM."""
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        parts: list[str] = []
        if content is not None:
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content if b.get("type") == "text"
                ]
                parts.append(" ".join(text_parts))
            elif isinstance(content, str):
                parts.append(content)

        if tool_calls:
            tc_strs: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                tc_strs.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            parts.append("[tool_calls: " + ", ".join(tc_strs) + "]")

        extra = ""
        if "tool_call_id" in msg:
            extra = f" (tool_call_id={msg['tool_call_id']})"

        lines.append(
            f"[{i}] [{role}]{extra}: {' | '.join(parts) if parts else '(empty)'}",
        )

    return "\n".join(lines)


async def compress_messages(
    messages: list[dict],
    endpoint: str,
) -> CompressedMessages:
    """Compact a message list using an LLM call with structured output.

    1. Strips images and thinking blocks via ``prepare_messages_for_compression``
    2. Sends the prepared transcript to an LLM with ``COMPRESSION_PROMPT``
    3. Validates that the output has exactly ``len(messages)`` entries
    """
    prepared = prepare_messages_for_compression(messages)
    serialized = _serialize_messages_for_prompt(prepared)

    user_prompt = (
        f"Compact the following {len(messages)} messages. "
        f"Return exactly {len(messages)} entries.\n\n"
        f"{serialized}"
    )

    client = new_llm_client(endpoint, origin="compress_messages")
    client.set_system_message(COMPRESSION_PROMPT)

    response = await client.generate(
        messages=[{"role": "user", "content": user_prompt}],
        response_format=CompressedMessages,
    )

    parsed = json.loads(response)
    result = CompressedMessages.model_validate(parsed)

    if len(result.messages) != len(messages):
        raise ValueError(
            f"Compression returned {len(result.messages)} messages, "
            f"expected {len(messages)}",
        )

    return result


def render_compressed_context(
    compressed: CompressedMessages,
    index_offset: int = 0,
) -> str:
    """Render compressed messages into compact indexed lines.

    Returns only the message lines (no header).  The caller is responsible
    for prepending ``## Compressed Prior Context`` when assembling the
    final system-message block.

    ``index_offset`` shifts every ``[N]`` label so that multiple renders
    can be concatenated with continuous numbering.
    """
    lines: list[str] = []
    for i, msg in enumerate(compressed.messages):
        lines.append(f"[{i + index_offset}] [{msg.role}]: {msg.content}")
    return "\n".join(lines)
