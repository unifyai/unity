from __future__ import annotations

import asyncio
from typing import Optional, List, Dict, Any

import pytest

from unity.common.async_tool_loop import (
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.image_manager.types import ImageRefs
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client
from tests.async_helpers import _wait_for_tool_request

# Global registry to capture instances for assertion
_HANDLES: List["CustomImagesHandle"] = []


class CustomImagesHandle(SteerableToolHandle):
    """A custom handle that records interjection arguments including images."""

    def __init__(self) -> None:
        self._done = asyncio.Event()
        self.interject_calls: List[Dict[str, Any]] = []

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context_cont=None,
        images=None,
    ) -> "SteerableToolHandle":
        return self

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        images: Optional[ImageRefs] = None,
    ) -> Optional[str]:
        self.interject_calls.append(
            {
                "message": message,
                "images": images,
            },
        )
        return None

    def stop(self, reason: Optional[str] = None):
        self._done.set()
        return "stopped"

    async def pause(self):
        return "paused"

    async def resume(self):
        return "resumed"

    def done(self) -> bool:
        return self._done.is_set()

    async def result(self) -> str:
        await self._done.wait()
        return "ok"

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


async def spawn_images_handle() -> SteerableToolHandle:  # type: ignore[name-defined]
    h = CustomImagesHandle()
    _HANDLES.append(h)
    return h


@pytest.mark.asyncio
@_handle_project
async def test_interject_with_images_on_custom_handle(model):
    """
    Verify that the `interject` method on a custom SteerableToolHandle receives
    the `images` argument when called via the dynamic `interject_...` helper.
    """
    _HANDLES.clear()

    client = new_llm_client(model=model)

    # Instruct the LLM to call spawn_images_handle first
    client.set_system_message(
        "Call `spawn_images_handle` to start the task.",
    )

    outer = start_async_tool_loop(
        client,
        message="start",
        tools={"spawn_images_handle": spawn_images_handle},
        timeout=120,
        max_steps=30,
    )

    await _wait_for_tool_request(client, "spawn_images_handle")

    # Now instruct the LLM to use the dynamic helper
    await outer.interject(
        "Now call the `interject_...` helper for the task you just started. "
        "Pass arguments: message='look', images=[{'image_id': 123}]. "
        "After that, call `stop_...` to finish.",
    )

    await outer.result()

    # Verify we captured the handle and the interjection
    assert len(_HANDLES) == 1
    handle = _HANDLES[0]

    assert len(handle.interject_calls) > 0
    last_call = handle.interject_calls[-1]
    assert last_call["message"] == "look"

    # Verify images were passed through
    images = last_call.get("images")
    assert isinstance(images, list)
    assert len(images) == 1
    # The loop passes arguments as received from the LLM (deserialized JSON)
    assert images[0].get("image_id") == 123
