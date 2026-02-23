from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .raw_image_ref import RawImageRef

if TYPE_CHECKING:
    from ..image_manager import ImageManager


class AnnotatedImageRef(BaseModel):
    """
    Pair a `RawImageRef` with a context-specific annotation describing relevance.
    """

    raw_image_ref: RawImageRef = Field(
        description="Reference to the underlying raw image",
    )
    annotation: str = Field(
        description="Context-specific annotation, explaining the image's relevance in relation to a corresponding request or question",
    )

    def resolve_image_id(self, image_manager: ImageManager) -> int:
        """Convenience delegate to ``self.raw_image_ref.resolve_image_id``."""
        return self.raw_image_ref.resolve_image_id(image_manager)
