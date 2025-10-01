from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
