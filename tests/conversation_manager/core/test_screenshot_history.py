"""Tests for ScreenshotHistory and visual context lifecycle.

These are symbolic tests — no LLM calls. They verify:
1. Visual context messages don't accumulate when _visual_ctx_msg_id is
   shared between _inject_visual_context and _capture_screenshots_for_llm.
2. ScreenshotHistory.clear() removes entries by source.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


from unity.conversation_manager.medium_scripts.common import ScreenshotHistory
from unity.conversation_manager.types.screenshot import ScreenshotEntry

# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _ChatItem:
    id: str
    role: str
    content: list


class _FakeChatContext:
    """Minimal mock of livekit.agents.llm.ChatContext."""

    def __init__(self) -> None:
        self.items: list[_ChatItem] = []

    def add_message(
        self,
        *,
        role: str,
        content: list,
        id: str | None = None,
    ) -> _ChatItem:
        item = _ChatItem(id=id or str(uuid.uuid4()), role=role, content=content)
        self.items.append(item)
        return item

    def index_by_id(self, msg_id: str) -> int | None:
        for i, item in enumerate(self.items):
            if item.id == msg_id:
                return i
        return None

    def copy(self) -> "_FakeChatContext":
        ctx = _FakeChatContext()
        ctx.items = [
            _ChatItem(id=it.id, role=it.role, content=list(it.content))
            for it in self.items
        ]
        return ctx


def _make_entry(source: str = "user", idx: int = 0) -> tuple[ScreenshotEntry, str]:
    return (
        ScreenshotEntry(
            b64="AAAA",
            utterance=f"test {idx}",
            timestamp=datetime.now(timezone.utc),
            source=source,
        ),
        f"Screenshots/{source.title()}/test_{idx}.jpg",
    )


def _count_visual(ctx: _FakeChatContext) -> int:
    return sum(
        1
        for it in ctx.items
        if it.role == "user" and any("[Screenshot" in str(c) for c in it.content)
    )


# ── Bug 1: Visual context accumulation ──────────────────────────────────────


def test_visual_ctx_does_not_accumulate_in_live():
    """_inject_visual_context must replace, not accumulate, in session._chat_ctx."""
    _visual_ctx_msg_id: str | None = None
    live = _FakeChatContext()
    live.add_message(role="system", content=["prompt"])
    live.add_message(role="user", content=["Hello"])
    live.add_message(role="assistant", content=["Hi"])

    for i in range(5):
        content = [f"[Screenshot #{i}]"]
        if _visual_ctx_msg_id is not None:
            idx = live.index_by_id(_visual_ctx_msg_id)
            if idx is not None:
                live.items.pop(idx)
        msg = live.add_message(role="user", content=content)
        _visual_ctx_msg_id = msg.id

    assert _count_visual(live) == 1


def test_visual_ctx_does_not_accumulate_in_copy():
    """_capture_screenshots_for_llm must produce exactly 1 visual context
    message in the copy, even though _handle_screenshot indirectly mutates
    _visual_ctx_msg_id via _inject_visual_context.

    The fix: save _visual_ctx_msg_id BEFORE calling _handle_screenshot,
    and use the saved value to remove the old message from the copy.
    """
    _visual_ctx_msg_id: str | None = None
    live = _FakeChatContext()
    live.add_message(role="system", content=["prompt"])
    live.add_message(role="user", content=["Hello"])
    live.add_message(role="assistant", content=["Hi"])
    last_copy = None

    for i in range(5):
        content = [f"[Screenshot #{i}]"]

        # Periodic _inject_visual_context (between llm_node calls)
        if _visual_ctx_msg_id is not None:
            idx = live.index_by_id(_visual_ctx_msg_id)
            if idx is not None:
                live.items.pop(idx)
        msg = live.add_message(role="user", content=content)
        _visual_ctx_msg_id = msg.id

        # LiveKit copies chat_ctx BEFORE llm_node runs
        copy = live.copy()
        saved_vid = _visual_ctx_msg_id  # Save BEFORE _handle_screenshot

        # _handle_screenshot → _inject_visual_context mutates _visual_ctx_msg_id
        if _visual_ctx_msg_id is not None:
            idx = live.index_by_id(_visual_ctx_msg_id)
            if idx is not None:
                live.items.pop(idx)
        new_msg = live.add_message(role="user", content=content)
        _visual_ctx_msg_id = new_msg.id

        # Use saved_vid (which exists in the copy) to remove
        if saved_vid is not None:
            idx = copy.index_by_id(saved_vid)
            if idx is not None:
                copy.items.pop(idx)
        copy.add_message(role="user", content=content)
        last_copy = copy

    assert (
        _count_visual(last_copy) == 1
    ), f"Expected 1 visual context message in copy, got {_count_visual(last_copy)}"


# ── Bug 2: Screenshot cleanup ───────────────────────────────────────────────


def test_clear_by_source():
    """clear(source=...) removes only entries for that source."""
    h = ScreenshotHistory()
    for i in range(3):
        h.add(*_make_entry("user", i))
    for i in range(2):
        h.add(*_make_entry("assistant", i))

    h.clear(source="user")
    assert len(h._entries) == 2
    assert all(e.source == "assistant" for e, _ in h._entries)


def test_clear_all():
    """clear() with no source removes all entries."""
    h = ScreenshotHistory()
    for i in range(3):
        h.add(*_make_entry("user", i))
    h.clear()
    assert len(h._entries) == 0
    assert h.build_visual_context_content() == []


def test_clear_nonexistent_source():
    """clear() with a source that has no entries is a no-op."""
    h = ScreenshotHistory()
    for i in range(2):
        h.add(*_make_entry("user", i))
    h.clear(source="webcam")
    assert len(h._entries) == 2
