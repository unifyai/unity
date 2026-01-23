from __future__ import annotations

"""
Consolidated token utilities for counting and budgeting, plus table token estimation.

This module centralises functionality previously scattered in parser-level utils
so other managers can reuse them directly.
"""

import math
import json
from typing import Optional, Any, Dict, List, Tuple

from unity.settings import SETTINGS
from unity.common.grouping_helpers import iter_unique_values_via_groups

_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken  # type: ignore
except Exception as e:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore

_AVG_CHARS_PER_TOKEN = 4.0
_WARNED_ON_FALLBACK = False
_TOKENS_PER_UTF8_BYTE = 0.25

# If the input text is very short, summarization is unnecessary and can be lossy.
# For these cases we treat the original text as the "summary" and skip LLM calls.
MIN_SOURCE_TEXT_CHARS_FOR_SUMMARY = 200


def _warn_once() -> None:
    global _WARNED_ON_FALLBACK
    if not _WARNED_ON_FALLBACK:
        try:
            print(
                "⚠️  tiktoken not available – using a conservative char→token heuristic. "
                "Install `tiktoken` for precise accounting.",
            )
        except Exception:
            pass
        _WARNED_ON_FALLBACK = True


def has_meaningful_text(s: str | None) -> bool:
    s = (s or "").strip()
    return (
        bool(s and any(ch.isalnum() for ch in s))
        and len(s) > MIN_SOURCE_TEXT_CHARS_FOR_SUMMARY
    )


def get_encoding_for(model_or_encoding: Optional[str] = None):
    """
    Return a tiktoken Encoding for a **model name** or **encoding name**.
    Preference: encoding_for_model(model) → get_encoding(name) → default.
    """
    if not _TIKTOKEN_AVAILABLE:
        _warn_once()
        return None

    if model_or_encoding:
        try:
            return tiktoken.encoding_for_model(model_or_encoding)
        except Exception:
            try:
                return tiktoken.get_encoding(model_or_encoding)
            except Exception:
                pass

        if "gpt-4o" in model_or_encoding or "o4-mini" in model_or_encoding:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_or_encoding: Optional[str] = None) -> int:
    """Return number of tokens in *text* for the given model/encoding."""
    if not _TIKTOKEN_AVAILABLE:
        _warn_once()
        return int(len(text) / _AVG_CHARS_PER_TOKEN * 1.1)
    enc = get_encoding_for(model_or_encoding)
    try:
        return len(enc.encode(text))  # type: ignore[attr-defined]
    except Exception:
        return int(len(text) / _AVG_CHARS_PER_TOKEN * 1.1)


def count_tokens_per_utf_byte(document: str) -> int:
    """
    Estimates token count based on UTF-8 byte length.
    Open AI uses this rather than `tiktoken` contrary
    to what is mentioned in the docs:
    https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699/6
    """
    # Single C-level pass to UTF-8; then take length
    n_bytes = len(document.encode("utf-8"))
    return math.ceil(_TOKENS_PER_UTF8_BYTE * n_bytes)


def is_within_token_limit_bytes(text: str, max_tokens: int) -> bool:
    """Return True iff the UTF-8 byte/token heuristic estimate fits within max_tokens."""
    try:
        return count_tokens_per_utf_byte(text) <= int(max_tokens)
    except Exception:
        return len(text) * 0.5 <= int(max_tokens)


def clip_text_to_token_limit_bytes(text: str, max_tokens: int) -> str:
    """
    Clip text so that the UTF-8 byte/token heuristic (0.25 tokens per byte) stays within max_tokens.
    Walk characters and stop before exceeding the byte budget.
    """
    if max_tokens <= 0 or not text:
        return ""
    try:
        max_bytes = int(max_tokens / _TOKENS_PER_UTF8_BYTE)
        current_bytes = 0
        out_chars: list[str] = []
        for ch in text:
            bl = len(ch.encode("utf-8"))
            if current_bytes + bl > max_bytes:
                break
            out_chars.append(ch)
            current_bytes += bl
        return "".join(out_chars)
    except Exception:
        max_chars = int(max_tokens * _AVG_CHARS_PER_TOKEN)
        return text[:max_chars]


def conservative_token_estimate(
    text: str,
    model_or_encoding: Optional[str] = None,
) -> int:
    """Return a conservative token estimate as the max of tiktoken and UTF-8 heuristics."""
    try:
        tk = count_tokens(text, model_or_encoding)
    except Exception:
        tk = 0
    try:
        by = int(count_tokens_per_utf_byte(text))
    except Exception:
        by = 0
    return max(tk, by)


def is_within_token_limit_conservative(
    text: str,
    max_tokens: int,
    model_or_encoding: Optional[str] = None,
) -> bool:
    try:
        return conservative_token_estimate(text, model_or_encoding) <= int(max_tokens)
    except Exception:
        return is_within_token_limit_bytes(text, max_tokens)


def clip_text_to_token_limit_conservative(
    text: str,
    max_tokens: int,
    model_or_encoding: Optional[str] = None,
) -> str:
    """
    Best-effort clip of text to stay within max_tokens using a conservative estimate.
    Strategy:
    - First clip by UTF-8 byte heuristic.
    - Re-check conservative estimate; if still over, shrink further by ratio.
    - Avoid tiktoken-based slicing for robustness; rely on bytes-based clipping.
    """
    if max_tokens <= 0 or not text:
        return ""
    clipped = clip_text_to_token_limit_bytes(text, max_tokens)
    est = conservative_token_estimate(clipped, model_or_encoding)
    if est <= max_tokens:
        return clipped
    # If still over, apply a proportional reduction and re-clip
    if est > 0:
        ratio = max_tokens / float(est)
        reduce_to = max(1, int(max_tokens * ratio))
    else:
        reduce_to = max_tokens
    return clip_text_to_token_limit_bytes(clipped, reduce_to)


def first_tokens_per_utf_byte(text: str, n_tokens: int) -> str:
    """Return the first ~n_tokens (heuristic) worth of text using UTF-8 byte estimation."""
    return clip_text_to_token_limit_bytes(text, n_tokens)


def last_tokens_per_utf_byte(text: str, n_tokens: int) -> str:
    """Return the last ~n_tokens worth of text using UTF-8 byte estimation."""
    if n_tokens <= 0 or not text:
        return ""
    try:
        max_bytes = int(n_tokens / _TOKENS_PER_UTF8_BYTE)
        b = text.encode("utf-8")
        tail = b[-max_bytes:]
        return tail.decode("utf-8", errors="ignore")
    except Exception:
        approx_chars = int(n_tokens * _AVG_CHARS_PER_TOKEN)
        return text[-approx_chars:]


def middle_tokens_per_utf_byte(text: str, n_tokens: int) -> str:
    """Return the middle ~n_tokens worth of text using UTF-8 byte estimation."""
    if n_tokens <= 0 or not text:
        return ""
    try:
        max_bytes = int(n_tokens / _TOKENS_PER_UTF8_BYTE)
        b = text.encode("utf-8")
        L = len(b)
        if L == 0 or max_bytes <= 0:
            return ""
        start = max((L // 2) - (max_bytes // 2), 0)
        end = min(start + max_bytes, L)
        segment = b[start:end]
        return segment.decode("utf-8", errors="ignore")
    except Exception:
        approx_chars = int(n_tokens * _AVG_CHARS_PER_TOKEN)
        s = max((len(text) // 2) - (approx_chars // 2), 0)
        return text[s : s + approx_chars]


def is_within_token_limit(
    text: str,
    max_tokens: int,
    model_or_encoding: Optional[str] = None,
) -> bool:
    """Return True iff *text* fits within *max_tokens* for the given model/encoding."""
    return count_tokens(text, model_or_encoding) <= int(max_tokens)


def clip_text_to_token_limit(
    text: str,
    max_tokens: int,
    model_or_encoding: Optional[str] = None,
) -> str:
    """
    Return *text* clipped to at most *max_tokens* (best-effort, token-aware).
    When `tiktoken` is not available we clip by characters using the average ratio.
    """
    if max_tokens <= 0 or not text:
        return ""
    if not _TIKTOKEN_AVAILABLE:
        _warn_once()
        max_chars = int(max_tokens * _AVG_CHARS_PER_TOKEN)
        return text[:max_chars]
    enc = get_encoding_for(model_or_encoding)
    try:
        toks = enc.encode(text)  # type: ignore[attr-defined]
        if len(toks) <= max_tokens:
            return text
        toks = toks[:max_tokens]
        return enc.decode(toks)  # type: ignore[attr-defined]
    except Exception:
        max_chars = int(max_tokens * _AVG_CHARS_PER_TOKEN)
        return text[:max_chars]


# -------- Added: budgeting + per-table token estimation (via get_groups) ------


def read_model_max_input_tokens() -> int:
    return SETTINGS.knowledge.MODEL_MAX_INPUT_TOKENS


def safe_token_count(value: Any) -> int:
    try:
        if value is None:
            return 0
        s = value if isinstance(value, str) else str(value)
        return count_tokens_per_utf_byte(s)
    except Exception:
        try:
            return count_tokens_per_utf_byte(json.dumps(value, ensure_ascii=False))
        except Exception:
            return 0


def token_budget(max_input_tokens: int, safety_factor: float) -> int:
    return int(max_input_tokens * safety_factor)


def estimate_table_tokens(
    *,
    context: str,
    list_columns: List[str],
    token_budget_cap: int,
) -> int:
    total = 0
    for col in list_columns:
        if total > token_budget_cap:
            break
        try:
            uniques = iter_unique_values_via_groups(context, col)
        except Exception:
            uniques = []
        for v in uniques:
            total += safe_token_count(v)
            if total > token_budget_cap:
                return total
    return total


async def estimate_tables_tokens_parallel(
    *,
    table_to_ctx: Dict[str, str],
    table_to_columns: Dict[str, List[str]],
    max_input_tokens: int,
    safety_factor: float,
    max_concurrency: int = 4,
) -> Dict[str, int]:
    import asyncio

    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    cap = token_budget(max_input_tokens, safety_factor)

    async def _run(table: str) -> Tuple[str, int]:
        async with sem:
            est = await asyncio.to_thread(
                estimate_table_tokens,
                context=table_to_ctx[table],
                list_columns=table_to_columns.get(table, []),
                token_budget_cap=cap,
            )
            return table, est

    results = await asyncio.gather(*(_run(t) for t in table_to_ctx.keys()))
    return {k: v for k, v in results}
