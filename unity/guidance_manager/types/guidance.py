from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field, field_validator, model_validator
import re

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
    images: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Mapping of json.dumps strings like '[x:y]' → image_id (int). "
            "Matches the images column semantics used in Transcripts."
        ),
    )

    function_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of Function.function_id values that this guidance is relevant for. "
            "Represents a many-to-many relationship between Guidance and Functions."
        ),
    )

    @field_validator("images", mode="before")
    @classmethod
    def _validate_images(cls, v):
        """Ensure images is a dict[str, int] with keys like "[x:y]".

        Rules:
        - Key must strictly match "[x:y]" with optional negative or open ends.
          Regex: ^\[\s*(-?\d+)?\s*:\s*(-?\d+)?\s*\]$
        - Value must be coercible to int (image_id).
        - None → {}.
        """
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("images must be a dict[str, int]")
        pattern = re.compile(r"^\[\s*(-?\d+)?\s*:\s*(-?\d+)?\s*\]$")
        out: dict[str, int] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                raise ValueError("images keys must be strings like '[x:y]'")
            if not pattern.fullmatch(k):
                raise ValueError(
                    f"images key '{k}' must match '[x:y]' with optional negative or open bounds",
                )
            try:
                out[k] = int(val)
            except Exception as exc:
                raise ValueError(
                    f"images value for key '{k}' must be an integer image_id",
                ) from exc
        return out

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
