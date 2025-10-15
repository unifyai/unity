from __future__ import annotations

from typing import List

from pydantic import RootModel

from .raw_image_ref import RawImageRef
from .annotated_image_ref import AnnotatedImageRef


class ImageRefs(RootModel[List[RawImageRef | AnnotatedImageRef]]):
    """
    Container for a list of image references, possibly mixed raw and annotated.
    Ordering has no semantic importance; this type is purely for transport/validation.
    """
