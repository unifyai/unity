from __future__ import annotations

import math
import os
from typing import Optional

from .token_utils import (
    clip_text_to_token_limit,
    count_tokens_per_utf_byte,
    has_meaningful_text,
)


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _build_compression_directive(
    previous_tokens: int,
    max_tokens: int,
    min_reduction_percent: int,
) -> str:
    if previous_tokens <= 0:
        target_percent = min_reduction_percent
    else:
        # Compute required reduction percentage and add a small safety margin
        needed = max(previous_tokens - max_tokens, 0)
        ratio = (needed / float(previous_tokens)) * 100.0 if previous_tokens else 0.0
        target_percent = max(min_reduction_percent, int(math.ceil(ratio + 10)))

    return (
        "\n\nCOMPRESSION DIRECTIVE:\n"
        "- Your last output exceeded the embedding input token budget.\n"
        f"- Estimated tokens (approx. 0.25 tokens per UTF-8 byte): ~{previous_tokens}. Limit: {max_tokens}.\n"
        f"- Reduce the summary length by at least {target_percent}% while preserving ALL numbers, units, proper nouns, and technical terms.\n"
        "- Remove redundancy, merge closely related bullets, and prefer concise phrasing without losing unique facts.\n"
        "- Do not drop quantitative values or domain-specific terminology.\n"
    )


def generate_summary_with_compression(
    client,
    prompt: str,
    source_text: str,
    *,
    max_attempts: int = 4,
    min_reduction_percent: int = 10,
    embedding_encoding: Optional[str] = None,
    max_embedding_tokens: Optional[int] = None,
    extra_directive: Optional[str] = None,
    post_generation_ctx: Optional[str] = None,
) -> str:
    """
    Generate a summary using the provided prompt and LLM client, enforcing that
    the final output fits within the embedding model's input token limit.

    Strategy:
    - Generate once; estimate tokens using count_tokens_per_utf_byte (0.25 tokens/byte).
    - If over the limit, re-issue the request while injecting a COMPRESSION DIRECTIVE
      that asks to reduce length by a computed percentage. Iterate until within
      limit or attempts exhausted. As a last resort, clip to limit.

    The function appends any compression directive to the end of the prompt, before
    the source text is concatenated by the caller of the LLM (we join here).
    """

    if not has_meaningful_text(source_text):
        return ""

    enc = embedding_encoding or os.environ.get("EMBEDDING_ENCODING", "cl100k_base")
    max_tokens = (
        max_embedding_tokens
        if isinstance(max_embedding_tokens, int) and max_embedding_tokens > 0
        else _get_env_int("EMBEDDING_MAX_INPUT_TOKENS", 8000)
    )

    directive = extra_directive or ""

    # Attempt loop
    attempts = 0
    summary_text: str = ""

    while attempts < max_attempts:
        attempts += 1
        try:
            final_prompt = prompt + directive
            summary_text = client.copy().generate(final_prompt + source_text).strip()
        except Exception:
            # If the model errors, try once more without extra directive; otherwise fallback later
            if attempts < max_attempts:
                continue
            break

        # Check token estimate using UTF-8 bytes heuristic
        est_tokens = int(count_tokens_per_utf_byte(summary_text))
        if est_tokens <= max_tokens:
            # Optionally append post-generation context if it keeps us within the cap
            if has_meaningful_text(post_generation_ctx or ""):
                combined = summary_text + "\n\n" + (post_generation_ctx or "")
                combined_tokens = int(count_tokens_per_utf_byte(combined))
                if combined_tokens <= max_tokens:
                    return combined
            return summary_text

        # Prepare a stronger directive for next attempt
        directive = _build_compression_directive(
            previous_tokens=est_tokens,
            max_tokens=max_tokens,
            min_reduction_percent=min_reduction_percent,
        )

    # Final fallback: clip to the embedding limit (never throw)
    return clip_text_to_token_limit(summary_text or "", max_tokens, enc)
