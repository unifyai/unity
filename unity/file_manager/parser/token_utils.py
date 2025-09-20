from __future__ import annotations

"""
Token utilities built around **tiktoken** with safe fallbacks.

- `count_tokens(text, model_or_encoding)` → int
- `is_within_token_limit(text, max_tokens, model_or_encoding)` → bool
- `clip_text_to_token_limit(text, max_tokens, model_or_encoding)` → str

Notes
-----
* We prefer `tiktoken.encoding_for_model(model)` where possible, falling back
  to explicit encodings like `"o200k_base"` (GPT-4o/4o-mini) or
  `"cl100k_base"` (GPT-4/3.5 + text-embedding-3-*).
* If `tiktoken` is unavailable at runtime, we provide a robust approximation
  (≈1 token per 4 chars) so callers never crash.
"""
from typing import Optional

_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken  # type: ignore
except Exception as e:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore

_AVG_CHARS_PER_TOKEN = 4.0
_WARNED_ON_FALLBACK = False
_TOKENS_PER_UTF8_BYTE = 0.25


def _warn_once() -> None:
    global _WARNED_ON_FALLBACK
    if not _WARNED_ON_FALLBACK:
        try:
            print(
                "⚠️  tiktoken not available – using a conservative char→token heuristic. "
                "Install `tiktoken` for precise accounting.",
            )
        except Exception as e:
            pass
        _WARNED_ON_FALLBACK = True


def has_meaningful_text(s: str | None) -> bool:
    s = (s or "").strip()
    return bool(s and any(ch.isalnum() for ch in s)) and len(s) > 50


def get_encoding_for(model_or_encoding: Optional[str] = None):
    """
    Return a tiktoken Encoding for a **model name** or **encoding name**.
    Preference: encoding_for_model(model) → get_encoding(name) → default.
    """
    if not _TIKTOKEN_AVAILABLE:
        _warn_once()
        return None

    if model_or_encoding:
        # Try model first
        try:
            return tiktoken.encoding_for_model(model_or_encoding)
        except Exception as e:
            # Try as raw encoding name
            try:
                return tiktoken.get_encoding(model_or_encoding)
            except Exception as e:
                pass

        # Default by rough family
        if "gpt-4o" in model_or_encoding or "o4-mini" in model_or_encoding:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")

    # No hint → widely compatible default
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_or_encoding: Optional[str] = None) -> int:
    """Return number of tokens in *text* for the given model/encoding."""
    if not _TIKTOKEN_AVAILABLE:
        _warn_once()
        return int(len(text) / _AVG_CHARS_PER_TOKEN * 1.1)
    enc = get_encoding_for(model_or_encoding)
    try:
        return len(enc.encode(text))  # type: ignore[attr-defined]
    except Exception as e:
        return int(len(text) / _AVG_CHARS_PER_TOKEN * 1.1)


def count_tokens_per_utf_byte(document: str) -> float:
    """
    Estimates token count based on UTF-8 byte length.
    Open AI uses this rather than `tiktoken` contrary
    to what is mentioned in the docs:
    https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699/6
    """
    total_estimated_tokens = 0
    for char in document:
        byte_length = len(char.encode("utf-8"))
        total_estimated_tokens += byte_length * _TOKENS_PER_UTF8_BYTE

    return total_estimated_tokens


def is_within_token_limit_bytes(text: str, max_tokens: int) -> bool:
    """Return True iff the UTF-8 byte/token heuristic estimate fits within max_tokens."""
    try:
        return count_tokens_per_utf_byte(text) <= int(max_tokens)
    except Exception:
        # Conservative fallback
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
        # Fallback to approximate 4 chars per token
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
        # Approximate fallback by characters
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
    except Exception as e:
        max_chars = int(max_tokens * _AVG_CHARS_PER_TOKEN)
        return text[:max_chars]
