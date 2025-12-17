from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Union

PathLike = Union[str, Path]


@dataclass
class ConversionResult:
    ok: bool
    src: Path
    dst: Optional[Path]
    backend: str
    message: str = ""


class BaseConverter(Protocol):
    """
    Converter protocol for dependency injection.

    Converters transform one file type into another (e.g., DOCX → PDF) so that
    downstream parsing backends can operate on consistent inputs.
    """

    input_exts: set[str]
    output_ext: str

    def convert(
        self,
        src: PathLike,
        dst: Optional[PathLike] = None,
    ) -> ConversionResult:
        """Convert a single source file to the converter's output format."""
        ...

    def convert_all(
        self,
        inputs: Iterable[PathLike],
        *,
        output_dir: Optional[PathLike] = None,
        parallel: bool = False,
    ) -> List[ConversionResult]:
        """Batch convert multiple inputs."""
        ...


class DocumentConversionManager:
    """Registry-and-dispatch facade for multiple converters.

    - Register concrete converters (e.g., DocxToPdfConverter)
    - Ask the manager to convert one or many files; it selects the right converter
    """

    def __init__(
        self,
        *,
        converters: Optional[list[BaseConverter]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._converters: list[BaseConverter] = converters or []
        self.log = logger or logging.getLogger(__name__)

    def register(self, converter: BaseConverter) -> None:
        """Register a concrete converter implementation."""
        self._converters.append(converter)

    def supported_input_exts(self) -> set[str]:
        """Return the union of all supported input extensions across registered converters."""
        exts: set[str] = set()
        for c in self._converters:
            exts |= set(c.input_exts)
        return exts

    def pick(self, path: Path) -> Optional[BaseConverter]:
        """Select the appropriate converter for a given path, based on file extension."""
        ext = path.suffix.lower()
        for c in self._converters:
            if ext in c.input_exts:
                return c
        return None

    def convert(
        self,
        src: PathLike,
        dst: Optional[PathLike] = None,
    ) -> ConversionResult:
        """Convert a single input using the matching registered converter."""
        p = Path(src).expanduser().resolve()
        c = self.pick(p)
        if c is None:
            return ConversionResult(
                False,
                p,
                None,
                "none",
                f"No converter for: {p.suffix}",
            )
        return c.convert(p, dst)

    def convert_all(
        self,
        inputs: Iterable[PathLike],
        *,
        output_dir: Optional[PathLike] = None,
        parallel: bool = False,
    ) -> list[ConversionResult]:
        """Batch convert inputs using registered converters.

        Unsupported inputs are returned with a skipped ConversionResult.
        """
        indexed = [
            (i, Path(src).expanduser().resolve()) for i, src in enumerate(inputs)
        ]
        groups: dict[BaseConverter, list[tuple[int, Path]]] = {}
        skipped: list[tuple[int, ConversionResult]] = []

        for i, p in indexed:
            c = self.pick(p)
            if c is None:
                skipped.append(
                    (
                        i,
                        ConversionResult(
                            False,
                            p,
                            None,
                            "skip",
                            f"No converter for {p.suffix}",
                        ),
                    ),
                )
                continue
            groups.setdefault(c, []).append((i, p))

        results: list[Optional[ConversionResult]] = [None] * len(indexed)
        for i, r in skipped:
            results[i] = r

        for converter, items in groups.items():
            in_paths = [p for (_i, p) in items]
            out_dir = (
                Path(output_dir).expanduser().resolve()
                if output_dir is not None
                else None
            )

            # Converters may opt out of parallelism for safety.
            try:
                supports_parallel = getattr(converter, "allow_parallel_soffice", True)
            except Exception:
                supports_parallel = True

            batch_results = converter.convert_all(
                in_paths,
                output_dir=out_dir,
                parallel=parallel and bool(supports_parallel),
            )
            for (i, _p), res in zip(items, batch_results):
                results[i] = res

        return [
            (
                r
                if r is not None
                else ConversionResult(False, Path(""), None, "none", "unknown error")
            )
            for r in results
        ]
