from __future__ import annotations

from pydantic import BaseModel, Field

from .image import Image


class RawImageRef(BaseModel):
    """
    Reference to a raw `Image` record. Used when no additional context is needed.
    """

    image: Image = Field(description="The base image record being referenced")
