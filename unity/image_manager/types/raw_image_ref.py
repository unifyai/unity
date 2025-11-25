from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RawImageRef(BaseModel):
    """
    Reference to a raw `Image` record.
    """

    image_id: Optional[int] = Field(
        description="Unique identifier for the image (None if image deleted)",
    )
