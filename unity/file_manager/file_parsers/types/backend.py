from __future__ import annotations

"""Backend contract for `unity.file_manager.file_parsers.FileParser`.

This project prefers **inheritance-based contracts** over `Protocol` typing for
enforcing concrete implementations.

Why an abstract base class?
--------------------------
- **Runtime enforcement**: subclasses must implement the required API or Python
  will refuse to instantiate them.
- **Discoverability**: the contract is visible directly in the class hierarchy.
- **Stable surface**: FileParser can validate the backend instance type and the
  parse output type defensively (no `Any` at the boundary).

Non-goals
---------
- This base class does **not** prescribe how parsing is implemented (Docling,
  pure Python, etc.). It only enforces the *shape* of the interface.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Sequence

from .contracts import FileParseRequest, FileParseResult
from .formats import FileFormat


class BaseFileParserBackend(ABC):
    """Abstract base class for all FileParser backends.

    Contract
    --------
    Subclasses MUST:
    - declare a stable `name` (used in traces/diagnostics)
    - declare `supported_formats` (the FileFormat values this backend can parse)
    - implement `can_handle(fmt)` (pure predicate; must not raise)
    - implement `parse(request)` and ALWAYS return a `FileParseResult`

    Format awareness
    ---------------
    Backends are expected to be *format-aware* in their outputs. In particular,
    implementations should avoid producing huge/incompatible strings (e.g.
    spreadsheet full dumps) while still returning useful `summary` and `metadata`.

    FileParser will still wrap backend exceptions to avoid catastrophic batch
    failures, but well-behaved backends should return `FileParseResult(status='error')`
    instead of raising for anticipated failures (missing file, unsupported input).
    """

    name: ClassVar[str]
    supported_formats: ClassVar[Sequence[FileFormat]]

    @abstractmethod
    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        """Return True iff this backend supports parsing `fmt`."""

    @abstractmethod
    def parse(self, request: FileParseRequest, /) -> FileParseResult:
        """Parse a single file described by `request` and return a `FileParseResult`."""
