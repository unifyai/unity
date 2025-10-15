from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator
from ...image_manager.types import ImageRefs

UNASSIGNED = -1


class Guidance(BaseModel):
    guidance_id: int = Field(
        default=UNASSIGNED,
        description="Auto-incrementing unique identifier for the guidance entry",
        ge=UNASSIGNED,
    )
    title: str = Field(
        description="Short-form title of the guidance (a few words)",
        min_length=1,
        max_length=200,
    )
    content: str = Field(
        description="Full description of the guidance; may align with images",
        min_length=1,
    )
    images: ImageRefs = Field(
        default_factory=lambda: ImageRefs([]),
        description=(
            "List of image references, each either a RawImageRef or an AnnotatedImageRef "
            "(which includes a freeform explanation for how the image relates to the text)."
        ),
    )

    function_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of Function.function_id values that this guidance is relevant for. "
            "Represents a many-to-many relationship between Guidance and Functions."
        ),
    )

    # Images are now a list-based ImageRefs container. No span or substring validation remains.

    @field_validator("function_ids", mode="before")
    @classmethod
    def _validate_function_ids(cls, v):
        """Ensure function_ids is a list[int]. None → []. Coerce values to int."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError("function_ids must be a list[int]")
        out: list[int] = []
        for item in v:
            try:
                out.append(int(item))
            except Exception as exc:
                raise ValueError("function_ids must contain integers") from exc
        return out

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("guidance_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        exclude = {"guidance_id"} if self.guidance_id == UNASSIGNED else set()
        return self.model_dump(mode="json", exclude=exclude)
