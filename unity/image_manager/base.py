from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING


class BaseImageManager(ABC):
    """
    Public contract that every concrete image-manager must satisfy.

    Unlike higher-level managers, this interface exposes symbolic methods only
    (no natural-language ask/update). Implementations must manage images stored
    in a backend table with fields: `image_id: int`, `timestamp: datetime`,
    `caption: str | None`, `data: str` (base64).
    """

    # ------------------------------ Reads ---------------------------------
    @abstractmethod
    def filter_images(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Image"]:
        """
        Return images that satisfy a Python expression filter.

        Examples
        --------
        - "image_id == 42"
        - "caption is not None and 'sunset' in caption.lower()"
        - "timestamp >= datetime.fromisoformat('2025-01-01T00:00:00')"
        """

    @abstractmethod
    def search_images(
        self,
        *,
        reference_text: str,
        k: int = 10,
    ) -> List["Image"]:
        """
        Semantic search over image captions using the provided free‑form text.
        Returns up to ``k`` images ranked by similarity.
        """

    @abstractmethod
    def get_images(self, image_ids: List[int]) -> List["ImageHandle"]:
        """Return handles for the given image ids (missing ids are skipped)."""

    # ------------------------------ Writes --------------------------------
    @abstractmethod
    def add_images(self, items: List[Dict[str, Any]]) -> List[int]:
        """
        Add new images. Each item may include ``timestamp``, ``caption``, ``data``.
        Returns the allocated ``image_id`` values in insertion order.
        """

    @abstractmethod
    def update_images(self, updates: List[Dict[str, Any]]) -> List[int]:
        """
        Update existing images. Each update dict must include ``image_id`` and may
        set ``timestamp``, ``caption``, and/or ``data``. Returns updated ids.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


if TYPE_CHECKING:
    # Avoid runtime imports to prevent circular dependencies
    from .types.image import Image
    from .image_manager import ImageHandle


# Attach centralised docstring
BaseImageManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
