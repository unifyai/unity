"""Structured output types for sandbox code execution.

Provides TextPart / ImagePart / OutputPart for rich (text + image) output,
ExecutionResult for LLM-formatted execution results, and helper converters.
"""

from __future__ import annotations

import base64
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    """A text output part from sandbox execution."""

    type: Literal["text"] = "text"
    text: str

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {"type": "text", "text": self.text}


class ImagePart(BaseModel):
    """An image output part from sandbox execution (e.g., from display())."""

    type: Literal["image"] = "image"
    mime: str = "image/png"
    data: str  # base64 encoded

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime};base64,{self.data}"},
        }


# Discriminated union - Pydantic auto-parses based on `type` field
OutputPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


def _detect_image_mime_from_b64(b64_str: str) -> str:
    """Detect an image MIME type by inspecting decoded header bytes.

    Returns "image/jpeg" for JPEG, "image/png" for PNG, or "image/png" as fallback.
    """
    try:
        raw = base64.b64decode(b64_str[:32])
        if raw[:2] == b"\xff\xd8":
            return "image/jpeg"
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
    except Exception:
        pass
    return "image/png"


def parts_to_text(parts: List[Union[TextPart, ImagePart]]) -> str:
    """Convert a list of OutputPart to a plain text string.

    Useful for backward compatibility and simple text extraction.
    Only TextPart parts are included; ImagePart parts are skipped.
    """
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def parts_to_llm_content(parts: List[Union[TextPart, ImagePart]]) -> List[dict]:
    """Convert a list of OutputParts to LLM content blocks, preserving order.

    This function maintains the original interleaving of text and images,
    unlike the legacy approach which collected all images at the end.

    Adjacent TextParts are merged into a single text block for cleaner output.
    """
    if not parts:
        return []

    blocks: List[dict] = []
    pending_text = ""

    for part in parts:
        if isinstance(part, TextPart):
            pending_text += part.text
        elif isinstance(part, ImagePart):
            # Flush any pending text before the image
            if pending_text:
                blocks.append({"type": "text", "text": pending_text})
                pending_text = ""
            blocks.append(part.to_llm_content())

    # Flush any remaining text
    if pending_text:
        blocks.append({"type": "text", "text": pending_text})

    return blocks


class ExecutionResult(BaseModel):
    """Result from sandbox code execution, implementing FormattedToolResult protocol.

    This model gives the sandbox full control over how its output is formatted
    for the LLM, preserving the original interleaving of text and images from
    print() and display() calls.
    """

    stdout: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    stderr: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    language: Optional[str] = None
    state_mode: Optional[str] = None
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    venv_id: Optional[int] = None
    session_created: Optional[bool] = None
    duration_ms: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_llm_content(self) -> List[dict]:
        """Format this execution result for the LLM, preserving output order.

        Implements the FormattedToolResult protocol, giving the sandbox full
        control over how its output appears in the LLM transcript.
        """
        blocks: List[dict] = []

        # Build metadata section (non-stdout/stderr fields)
        meta: Dict[str, Any] = {}
        if self.result is not None:
            meta["result"] = self.result
        if self.error is not None:
            meta["error"] = self.error
        if self.language is not None:
            meta["language"] = self.language
        if self.state_mode is not None:
            meta["state_mode"] = self.state_mode
        if self.session_id is not None:
            meta["session_id"] = self.session_id
        if self.session_name is not None:
            meta["session_name"] = self.session_name
        if self.venv_id is not None:
            meta["venv_id"] = self.venv_id
        if self.session_created is not None:
            meta["session_created"] = self.session_created
        if self.duration_ms is not None:
            meta["duration_ms"] = self.duration_ms

        # Add metadata block if present
        if meta:
            import json

            meta_text = json.dumps(meta, indent=2, default=str)
            blocks.append({"type": "text", "text": meta_text})

        # Add stdout with preserved ordering (interleaved text/images)
        if self.stdout:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stdout
            )
            if has_content:
                if blocks:  # Add separator if we have metadata
                    blocks.append({"type": "text", "text": "\n--- stdout ---\n"})
                blocks.extend(parts_to_llm_content(self.stdout))

        # Add stderr with preserved ordering (if non-empty)
        if self.stderr:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stderr
            )
            if has_content:
                blocks.append({"type": "text", "text": "\n--- stderr ---\n"})
                blocks.extend(parts_to_llm_content(self.stderr))

        # Ensure we always return at least something
        if not blocks:
            blocks.append({"type": "text", "text": "(no output)"})

        return blocks
