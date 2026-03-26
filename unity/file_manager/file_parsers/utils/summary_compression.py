from __future__ import annotations

"""Embedding-safe summary generation with compression retries.

This module is used by parser enrichment steps to generate summaries that:
- are useful for retrieval/embedding, and
- always fit within the embedding model's input token budget.
"""

import math
from typing import Optional

from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.common.token_utils import (
    clip_text_to_token_limit,
    count_tokens_per_utf_byte,
    has_meaningful_text,
)


def _post_process_summary(text: str) -> str:
    """Final cleanup of LLM-generated summaries before storage and embedding.

    LLMs sometimes emit literal escape sequences (``\\n``, ``\\t``) rather than
    real whitespace.  Decoding them improves downstream embedding quality because
    tokenizers handle actual whitespace more naturally than backslash-prefixed
    character pairs.
    """
    text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "")
    return text.strip()


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
        "- Estimated tokens (approx. 0.25 tokens per UTF-8 byte): "
        f"~{previous_tokens}. Limit: {max_tokens}.\n"
        f"- Reduce the summary length by at least {target_percent}% while preserving "
        "ALL numbers, units, proper nouns, and technical terms.\n"
        "- Remove redundancy, merge closely related bullets, and prefer concise phrasing "
        "without losing unique facts.\n"
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
    """

    if not has_meaningful_text(source_text):
        return source_text

    enc = embedding_encoding or FILE_PARSER_SETTINGS.EMBEDDING_ENCODING
    max_tokens = (
        max_embedding_tokens
        if isinstance(max_embedding_tokens, int) and max_embedding_tokens > 0
        else FILE_PARSER_SETTINGS.EMBEDDING_MAX_INPUT_TOKENS
    )

    directive = extra_directive or ""

    attempts = 0
    summary_text: str = ""

    while attempts < max_attempts:
        attempts += 1
        try:
            final_prompt = prompt + directive
            summary_text = client.copy().generate(final_prompt + source_text).strip()
        except Exception:
            if attempts < max_attempts:
                continue
            break

        est_tokens = int(count_tokens_per_utf_byte(summary_text))
        if est_tokens <= max_tokens:
            if has_meaningful_text(post_generation_ctx or ""):
                combined = summary_text + "\n\n" + (post_generation_ctx or "")
                combined_tokens = int(count_tokens_per_utf_byte(combined))
                if combined_tokens <= max_tokens:
                    return _post_process_summary(combined)
            return _post_process_summary(summary_text)

        directive = _build_compression_directive(
            previous_tokens=est_tokens,
            max_tokens=max_tokens,
            min_reduction_percent=min_reduction_percent,
        )

    # Final fallback: clip to the embedding limit (never throw)
    return _post_process_summary(
        clip_text_to_token_limit(summary_text or "", max_tokens, enc),
    )
