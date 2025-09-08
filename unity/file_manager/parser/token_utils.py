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
except Exception:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore

_AVG_CHARS_PER_TOKEN = 4.0
_WARNED_ON_FALLBACK = False


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


def has_meaningful_text(s: str) -> bool:
    return bool(s and any(ch.isalnum() for ch in s))


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
        except Exception:
            # Try as raw encoding name
            try:
                return tiktoken.get_encoding(model_or_encoding)
            except Exception:
                pass
        # Default by rough family
        if "gpt-4o" in model_or_encoding:
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
    except Exception:
        return int(len(text) / _AVG_CHARS_PER_TOKEN * 1.1)


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
