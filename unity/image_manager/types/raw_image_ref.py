from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..image_manager import ImageManager


class RawImageRef(BaseModel):
    """
    Reference to a raw `Image` record, identified by ``image_id``,
    ``filepath``, or both.

    At least one of the two fields should be provided when constructing a new
    reference.  Both may be ``None`` only for rows read back from storage after
    an FK ``SET NULL`` deletion.
    """

    image_id: Optional[int] = Field(
        default=None,
        description=(
            "Unique identifier for the image. None when the image has been "
            "deleted (FK SET NULL) or when only filepath is provided before "
            "resolution."
        ),
    )
    filepath: Optional[str] = Field(
        default=None,
        description=(
            "Filesystem path to the image. Can identify the image as an "
            "alternative to image_id; call resolve_image_id() to resolve to "
            "a concrete image_id."
        ),
    )

    def resolve_image_id(self, image_manager: ImageManager) -> int:
        """Resolve this reference to a concrete ``image_id``.

        If ``image_id`` is already populated, returns it immediately.
        Otherwise delegates to ``image_manager.resolve_filepath`` which
        looks up the image by filepath (or reads the file from disk and
        uploads it if no matching row exists yet).  The resolved id is
        written back to ``self.image_id`` so subsequent calls are free.

        Raises
        ------
        ValueError
            If neither ``image_id`` nor ``filepath`` is set.
        """
        if self.image_id is not None:
            return self.image_id
        if self.filepath is None:
            raise ValueError(
                "Cannot resolve: neither image_id nor filepath is set",
            )
        resolved = image_manager.resolve_filepath(self.filepath)
        self.image_id = resolved
        return resolved
