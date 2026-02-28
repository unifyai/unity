"""
Adapter that wraps unillm.AsyncUnify to expose the LiveKit llm.LLM interface.

This allows the TTS voice pipeline to route through our Unify client,
giving us local caching (helpful for CI) and usage tracking.
"""

from __future__ import annotations

from collections import deque
import uuid
from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from unity.common.llm_client import new_llm_client
from unity.conversation_manager.tracing import monotonic_ms, now_utc_iso, trace_kv
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON


class UnifyLLM(llm.LLM):
    """LiveKit-compatible LLM that uses unillm.AsyncUnify under the hood.

    This adapter provides:
    - Local caching for CI (via Unify's cache system)
    - Usage tracking through the Unify platform
    - Consistent routing through our standard LLM client

    Usage:
        from unity.settings import SETTINGS
        llm_model = UnifyLLM(model=SETTINGS.conversation.FAST_BRAIN_MODEL)
        session = AgentSession(llm=llm_model, ...)
    """

    def __init__(
        self,
        model: str = "gpt-5-mini@openai",
        *,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._service_tier = service_tier
        self._temperature = temperature
        self._extra_kwargs = kwargs
        self._pending_trace_contexts: deque[dict[str, Any]] = deque()

    @property
    def model(self) -> str:
        return self._model

    def enqueue_trace_context(self, trace_context: dict[str, Any]) -> None:
        """Attach metadata to the next generation request."""
        self._pending_trace_contexts.append(dict(trace_context))

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "UnifyLLMStream":
        trace_context = (
            self._pending_trace_contexts.popleft()
            if self._pending_trace_contexts
            else None
        )
        return UnifyLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            service_tier=self._service_tier,
            temperature=self._temperature,
            extra_kwargs=self._extra_kwargs,
            trace_context=trace_context,
        )


class UnifyLLMStream(llm.LLMStream):
    """Streaming wrapper that converts Unify responses to LiveKit ChatChunk format."""

    def __init__(
        self,
        llm: UnifyLLM,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        model: str,
        reasoning_effort: str | None,
        service_tier: str | None,
        temperature: float | None,
        extra_kwargs: dict[str, Any],
        trace_context: dict[str, Any] | None,
    ) -> None:
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._service_tier = service_tier
        self._temperature = temperature
        self._extra_kwargs = extra_kwargs
        self._request_id = str(uuid.uuid4())
        self._trace_context = trace_context or {}

    async def _run(self) -> None:
        """Stream responses from Unify and emit ChatChunk events."""
        from livekit.agents.llm import ImageContent

        # Convert LiveKit ChatContext to Unify message format.
        # All messages (including system) go into the messages array to
        # preserve temporal positioning of mid-conversation notifications.
        messages: list[dict] = []

        for item in self._chat_ctx.items:
            role = getattr(item, "role", None)
            if role is None:
                continue
            raw_content = getattr(item, "content", None)
            if not raw_content:
                continue

            has_images = isinstance(raw_content, list) and any(
                isinstance(c, ImageContent) for c in raw_content
            )
            if has_images:
                parts: list[dict] = []
                for c in raw_content:
                    if isinstance(c, str):
                        parts.append({"type": "text", "text": c})
                    elif isinstance(c, ImageContent) and isinstance(c.image, str):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": c.image},
                            },
                        )
                if parts:
                    messages.append({"role": role, "content": parts})
            else:
                text = getattr(item, "text_content", None)
                if not text:
                    continue
                messages.append({"role": role, "content": text})

        # Build client kwargs
        client_kwargs = dict(self._extra_kwargs)
        if self._reasoning_effort is not None:
            client_kwargs["reasoning_effort"] = self._reasoning_effort
        if self._service_tier is not None:
            client_kwargs["service_tier"] = self._service_tier

        # Create Unify client
        client = new_llm_client(
            self._model,
            origin="FastBrain",
            **client_kwargs,
        )
        client.set_stream(True)

        # Stream the response
        generate_kwargs: dict[str, Any] = {}
        if messages:
            generate_kwargs["messages"] = messages
        if self._temperature is not None:
            generate_kwargs["temperature"] = self._temperature

        LOGGER.debug(
            f"{DEFAULT_ICON} "
            + trace_kv(
                "FAST_BRAIN_REQUEST_START",
                request_id=self._request_id,
                model=self._model,
                message_count=len(messages),
                trigger=self._trace_context,
                ts_utc=now_utc_iso(),
                monotonic_ms=monotonic_ms(),
            ),
        )

        chunk_count = 0
        try:
            response = await client.generate(**generate_kwargs)

            # Emit chunks
            async for chunk_text in response:
                if chunk_text:
                    chunk_count += 1
                    chat_chunk = ChatChunk(
                        id=self._request_id,
                        delta=ChoiceDelta(
                            role="assistant",
                            content=chunk_text,
                        ),
                    )
                    self._event_ch.send_nowait(chat_chunk)
        finally:
            LOGGER.debug(
                f"{DEFAULT_ICON} "
                + trace_kv(
                    "FAST_BRAIN_REQUEST_END",
                    request_id=self._request_id,
                    chunk_count=chunk_count,
                    trigger_id=self._trace_context.get("generation_id", ""),
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
            )
