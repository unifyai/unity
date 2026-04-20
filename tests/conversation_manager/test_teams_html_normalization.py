"""Unit tests for the Teams HTML → plain-text normalizer used at ingress."""

from __future__ import annotations

import pytest

from unity.conversation_manager.comms_manager import _teams_html_to_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<p>hello olivia</p>", "hello olivia"),
        ("<p>first</p><p>second</p>", "first\nsecond"),
        ("hello<br>world", "hello\nworld"),
        ("hello<br/>world", "hello\nworld"),
        ("hello<br />world", "hello\nworld"),
        ('<at id="0">Olivia</at> hi there', "Olivia hi there"),
        ("hello &amp; goodbye", "hello & goodbye"),
        ("<p>hello&nbsp;world</p>", "hello\xa0world"),
        ("<div><p>nested</p></div>", "nested"),
        ("", ""),
        ("plain text, no tags", "plain text, no tags"),
        (
            "<p>line one</p><p><br></p><p>line two</p>",
            "line one\n\nline two",
        ),
    ],
)
def test_teams_html_to_text(raw: str, expected: str) -> None:
    assert _teams_html_to_text(raw) == expected


def test_teams_html_to_text_strips_trailing_block_newline() -> None:
    """A single `<p>…</p>` must not leave a trailing newline in the transcript."""
    assert _teams_html_to_text("<p>hi</p>").endswith("hi")
