"""
Backend registry for FileParser.

This module enables config-driven hot-swapping of backend implementations
without hard-coupling `FileParser` to any specific parsing library (Docling,
pandas, etc.) at import time.

Design notes
------------
- Backends are referenced by **class path strings** and imported lazily.
- Instances are cached per class path (singletons per registry instance).
- The registry enforces that loaded backends inherit `BaseFileParserBackend`.

This keeps `file_parsers` modular: swapping parsing libraries is primarily a
config/mapping change rather than a refactor of FileParser itself.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend

logger = logging.getLogger(__name__)


DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT: Dict[str, str] = {
    # Document formats
    "pdf": "unity.file_manager.file_parsers.implementations.docling.backends.pdf_backend.PdfBackend",
    "docx": "unity.file_manager.file_parsers.implementations.docling.backends.ms_word_backend.MsWordBackend",
    "doc": "unity.file_manager.file_parsers.implementations.docling.backends.ms_word_backend.MsWordBackend",
    "html": "unity.file_manager.file_parsers.implementations.docling.backends.html_backend.HtmlBackend",
    "xml": "unity.file_manager.file_parsers.implementations.docling.backends.xml_backend.XmlBackend",
    "json": "unity.file_manager.file_parsers.implementations.docling.backends.json_backend.JsonBackend",
    # Tabular formats
    "xlsx": "unity.file_manager.file_parsers.implementations.native.backends.excel_backend.NativeExcelBackend",
    "csv": "unity.file_manager.file_parsers.implementations.native.backends.csv_backend.NativeCsvBackend",
    # Text formats / fallbacks
    "txt": "unity.file_manager.file_parsers.implementations.python.backends.text_backend.TextBackend",
}


def _normalize_format_key(key: str) -> str:
    k = (key or "").strip().lower()
    if k.startswith("."):
        k = k[1:]
    return k


def _import_class(dotted_path: str) -> type:
    mod, _, attr = (dotted_path or "").rpartition(".")
    if not mod or not attr:
        raise ValueError(f"Invalid class path: {dotted_path!r}")
    module = importlib.import_module(mod)
    cls = getattr(module, attr, None)
    if cls is None:
        raise ValueError(f"Class not found: {dotted_path!r}")
    if not isinstance(cls, type):
        raise ValueError(f"Not a class: {dotted_path!r}")
    return cls


@dataclass
class BackendRegistry:
    """
    Registry mapping file formats to backend class paths with lazy loading.

    - No backend modules are imported until a backend is first requested.
    - Instances are cached per class path.

    The registry is intentionally small and dumb:
    - it does not interpret file paths; it routes only on `FileFormat`
    - it does not own output invariants; that belongs to the `FileParser` facade
    """

    backend_class_paths_by_format: Dict[str, str] = field(default_factory=dict)
    _instances_by_class_path: Dict[str, BaseFileParserBackend] = field(
        default_factory=dict,
        init=False,
    )

    @classmethod
    def from_config(
        cls,
        *,
        backend_class_paths_by_format: Optional[Dict[str, str]] = None,
    ) -> "BackendRegistry":
        """
        Build a registry from a mapping override.

        `backend_class_paths_by_format` keys may be provided as:
        - extensions (e.g. \"pdf\", \".pdf\")
        - FileFormat values (converted to string elsewhere)

        Missing keys fall back to `DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT`.
        """
        mapping = dict(DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT)
        if backend_class_paths_by_format:
            for k, v in backend_class_paths_by_format.items():
                if not v:
                    continue
                mapping[_normalize_format_key(k)] = str(v)
        return cls(
            backend_class_paths_by_format=mapping,
        )

    def _get_or_create(self, class_path: str) -> BaseFileParserBackend:
        if class_path in self._instances_by_class_path:
            return self._instances_by_class_path[class_path]
        cls = _import_class(class_path)
        inst = cls()  # type: ignore[call-arg]
        if not isinstance(inst, BaseFileParserBackend):
            raise TypeError(
                f"Backend class must inherit BaseFileParserBackend: {class_path} (got {type(inst)!r})",
            )
        self._instances_by_class_path[class_path] = inst
        return inst

    def pick_backend(
        self,
        fmt: Optional[FileFormat],
    ) -> Optional[BaseFileParserBackend]:
        """
        Return an instantiated backend for `fmt`, or None if no mapping exists.

        This method never raises for missing mappings. Import/instantiation errors
        are logged and treated as \"no backend\".
        """
        if fmt is None:
            return None
        key = _normalize_format_key(getattr(fmt, "value", str(fmt)))
        class_path = self.backend_class_paths_by_format.get(key)
        if class_path:
            try:
                inst = self._get_or_create(class_path)
                if inst.can_handle(fmt):
                    return inst
            except Exception as e:
                logger.warning(
                    "Backend load failed (fmt=%s class_path=%s): %s",
                    fmt,
                    class_path,
                    e,
                    exc_info=True,
                )
        return None
