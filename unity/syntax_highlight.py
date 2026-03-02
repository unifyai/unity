"""Lightweight syntax highlighting for markdown-fenced code blocks.

Only depends on ``re`` (stdlib) and ``pygments`` (optional, graceful
fallback).  Shared by ``unity.logger`` (server-side TTY formatting) and
``scripts/dev/job_logs/stream_logs.py`` (client-side log streaming).
"""

from __future__ import annotations

import re

MARKDOWN_CODE_BLOCK_RE = re.compile(
    r"([ \t]*```(\w+))\n(.*?)\n([ \t]*```)(?!\w)",
    re.DOTALL,
)
MARKDOWN_OPENING_RE = re.compile(r"```(\w+)")
MARKDOWN_CLOSING_RE = re.compile(r"```(?!\w)")


def highlight_code_blocks(text: str) -> str:
    """Apply Pygments syntax highlighting to markdown-fenced code blocks.

    Scans *text* for `````<language>`` ... `````` pairs and runs the code
    inside them through the appropriate Pygments lexer.  Falls back to
    plain text if the language is unrecognised or Pygments is unavailable.
    """
    try:
        from pygments import highlight
        from pygments.formatters import Terminal256Formatter
        from pygments.lexers import get_lexer_by_name
    except ImportError:
        return text

    formatter = Terminal256Formatter(style="monokai")

    def _highlight_match(m: re.Match) -> str:
        lang, code = m.group(2), m.group(3)
        try:
            lexer = get_lexer_by_name(lang)
            highlighted = highlight(code, lexer, formatter).rstrip("\n")
            return f"{m.group(1)}\n{highlighted}\n{m.group(4)}"
        except Exception:
            return m.group(0)

    return MARKDOWN_CODE_BLOCK_RE.sub(_highlight_match, text)
