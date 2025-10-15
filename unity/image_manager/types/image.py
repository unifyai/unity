from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator, RootModel


class Image(BaseModel):
    image_id: int = Field(
        default=-1,
        description="Unique identifier for the image (auto-incremented)",
        ge=-1,
    )
    timestamp: datetime = Field(
        description="When the image was captured",
    )
    caption: Optional[str] = Field(
        default=None,
        description="Short description of the image contents",
    )
    data: str = Field(
        description="Base64-encoded image data (PNG/JPEG)",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_defaults(cls, data: dict) -> dict:
        data = dict(data)
        data.setdefault("image_id", -1)
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.utcnow()
        return data

    @field_validator("caption", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def to_post_json(self) -> dict:
        exclude = {"image_id"} if self.image_id == -1 else set()
        payload = self.model_dump(mode="json", exclude=exclude)
        # Hint backend that `data` contains image bytes (base64) so it is typed as an image column
        try:
            et = dict(payload.get("explicit_types") or {})
            # Preserve any existing explicit type metadata and ensure image typing
            current = dict(et.get("data") or {})
            current["type"] = "image"
            et["data"] = current
            payload["explicit_types"] = et
        except Exception:
            # Best‑effort; if anything goes wrong, fall back to raw payload
            pass
        return payload


class AnnotatedImage(BaseModel):
    """
    Pair an `Image` with a context-specific annotation.

    The `caption` on the underlying image is context-agnostic; the `annotation`
    explains how/why the image is relevant within a particular scenario.
    """

    image: Image = Field(description="The base image record being annotated")
    annotation: str = Field(
        description="Context-specific relevance annotation for the image",
    )


class Images(RootModel[List[Image | AnnotatedImage]]):
    """
    Container for a list of images, which may include raw `Image` instances,
    `AnnotatedImage` instances, or a mix of both. Ordering has no semantic
    importance; this type is purely for transport/validation.
    """
