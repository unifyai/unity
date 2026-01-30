"""
Shared formatting primitives for sandbox terminal/GUI rendering.

This file is extracted (and simplified) from guided-learning sandbox display code
so other sandboxes can reuse consistent formatting without duplicating helpers.

It is intentionally "dumb": pure string formatting, ANSI helpers, truncation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

ANSI_RESET = "\x1b[0m"

ANSI_COLORS: dict[str, str] = {
    "cyan": "\x1b[36m",
    "yellow": "\x1b[33m",
    "magenta": "\x1b[35m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def colorize(text: str, color: str) -> str:
    code = ANSI_COLORS.get(color)
    if not code:
        return text
    return f"{code}{text}{ANSI_RESET}"


def truncate(text: str, max_len: int) -> str:
    s = str(text or "")
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    return s[: max_len - 1] + "…"


def indent_lines(text: str, prefix: str) -> str:
    if not text:
        return ""
    return "\n".join(prefix + line for line in str(text).splitlines())


@dataclass(frozen=True)
class BoxStyle:
    top_left: str = "┌"
    top_right: str = "┐"
    bottom_left: str = "└"
    bottom_right: str = "┘"
    horizontal: str = "─"
    vertical: str = "│"


def draw_box(
    content: str,
    *,
    title: Optional[str] = None,
    style: BoxStyle = BoxStyle(),
    max_width: int = 79,
) -> str:
    """
    Draw a simple box around multi-line content.

    max_width includes the border characters.
    """
    body_lines = (content or "").splitlines() or [""]
    # Compute visible widths (ignore ANSI escape sequences).
    visible_width = max(len(strip_ansi(l)) for l in body_lines)
    inner_width = min(max(0, max_width - 2), visible_width)

    def _pad(line: str) -> str:
        # Pad based on visible width, not raw byte length (ANSI).
        pad = inner_width - len(strip_ansi(line))
        return line + (" " * max(0, pad))

    top = style.top_left + style.horizontal * inner_width + style.top_right
    if title:
        t = truncate(title, inner_width)
        # Replace a slice of the top line with title (best-effort).
        top = (
            style.top_left
            + truncate(f" {t} ", inner_width).ljust(inner_width, style.horizontal)
            + style.top_right
        )

    lines = [top]
    for l in body_lines:
        lines.append(style.vertical + _pad(l)[: len(_pad(l))] + style.vertical)
    lines.append(
        style.bottom_left + style.horizontal * inner_width + style.bottom_right,
    )
    return "\n".join(lines)


def join_blocks(blocks: Iterable[str], *, separator: str = "\n\n") -> str:
    parts = [b for b in blocks if b and b.strip() != ""]
    return separator.join(parts)
