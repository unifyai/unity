"""Parser module for Unity.

This module provides a base parser interface and various parser implementations
for different document processing needs.

To add a new parser:
1. Create a new file in this directory (e.g., my_parser.py)
2. Implement a class that inherits from BaseParser
3. Import and export it in this __init__.py file
"""

from .base import BaseParser
from .docling_parser import DoclingParser
from .types import (
    Document,
    DocumentMetadata,
    DocumentMetadataExtraction,
    DocumentParagraph,
    DocumentSection,
    DocumentSentence,
)

__all__ = [
    # Base class
    "BaseParser",
    # Parser implementations
    "DoclingParser",
    # Document types
    "Document",
    "DocumentMetadata",
    "DocumentMetadataExtraction",
    "DocumentSection",
    "DocumentParagraph",
    "DocumentSentence",
]
