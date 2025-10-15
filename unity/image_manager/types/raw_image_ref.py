from __future__ import annotations

from pydantic import BaseModel, Field


class RawImageRef(BaseModel):
    """
    Reference to a raw `Image` record.
    """

    image_id: int = Field(
        description="Unique identifier for the image",
    )
