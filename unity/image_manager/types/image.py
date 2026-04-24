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
        description="Image payload as base64 (PNG/JPEG) or a URL (GCS signed or https).",
        json_schema_extra={"unify_type": "image"},
    )
    filepath: Optional[str] = Field(
        default=None,
        description="Optional path to the image file on the assistant's local filesystem.",
        json_schema_extra={"unique": True},
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id of the body that captured this image. Stamped once "
            "at create time and preserved across updates so reviewers can "
            "attribute Hive-shared rows to their authoring body."
        ),
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
        return payload
