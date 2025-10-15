from __future__ import annotations

from pydantic import BaseModel, Field

from .raw_image_ref import RawImageRef


class AnnotatedImageRef(BaseModel):
    """
    Pair a `RawImageRef` with a context-specific annotation describing relevance.
    """

    raw_image_ref: RawImageRef = Field(
        description="Reference to the underlying raw image",
    )
    annotation: str = Field(
        description="Context-specific relevance annotation for the image, aligning it with some correspond text",
    )
