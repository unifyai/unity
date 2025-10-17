from __future__ import annotations

from typing import List

from pydantic import RootModel

from .raw_image_ref import RawImageRef
from .annotated_image_ref import AnnotatedImageRef


class ImageRefs(RootModel[List[RawImageRef | AnnotatedImageRef]]):
    """
    Container for a list of image references, possibly mixed raw and annotated.
    """


class RawImageRefs(RootModel[List[RawImageRef]]):
    """
    Strict container for a list of raw image references only.

    Each item must be a `RawImageRef`.
    """


class AnnotatedImageRefs(RootModel[List[AnnotatedImageRef]]):
    """
    Strict container for a list of annotated image references only.

    Each item must be an `AnnotatedImageRef`.
    """
