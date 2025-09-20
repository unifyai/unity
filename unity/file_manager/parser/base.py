"""
Abstract parser interfaces for file content parsing.

Concrete implementations can live in separate packages and be injected into
runtime components (e.g., FileManager) via dependency injection.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Generic,
    List,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    AsyncIterator,
    Tuple,
)

from ...singleton_registry import SingletonABCMeta

# Type variable for the return type
T = TypeVar("T")


class BaseParser(ABC, metaclass=SingletonABCMeta):
    """
    Minimal parsing interface for transforming a file into a structured object.

    Implementations should be pure and side-effect free.
    """

    def _save_parsed_result_if_enabled(
        self,
        file_path: Union[str, Path],
        result: Any,
    ) -> None:
        """
        Save parsed result to JSON file if UNITY_SAVE_PARSED_RESULTS env var is set.

        Args:
            file_path: Original file path that was parsed
            result: The parsed result to save
        """
        # Check if saving is enabled via environment variable
        if not os.getenv("UNITY_SAVE_PARSED_RESULTS", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            return

        try:
            # Create output directory
            output_dir = Path("parsed_results_output")
            output_dir.mkdir(exist_ok=True)

            # Generate output filename based on input file
            input_path = Path(file_path)
            timestamp = os.getenv("UNITY_PARSED_RESULTS_TIMESTAMP", "")
            if timestamp:
                output_filename = f"{input_path.stem}_{timestamp}.json"
            else:
                output_filename = f"{input_path.stem}.json"

            output_path = output_dir / output_filename

            # Convert result to serializable format
            if hasattr(result, "to_dict"):
                # If result has to_dict method, use it
                serializable_result = result.to_dict()
            elif hasattr(result, "__dict__"):
                # Try to convert object to dict
                serializable_result = self._make_serializable(result.__dict__)
            else:
                # Assume it's already serializable
                serializable_result = result

            # Create wrapper with metadata
            output_data = {
                "source_file": str(input_path),
                "parser_class": self.__class__.__name__,
                "parsed_at": os.getenv("UNITY_PARSED_RESULTS_TIMESTAMP", ""),
                "result": serializable_result,
            }

            # Write to JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"✅ Saved parsed result to: {output_path}")

        except Exception as e:
            # Log error but don't fail the parsing operation
            print(f"⚠️  Failed to save parsed result for {file_path}: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            # For anything else, convert to string
            return str(obj)

    @abstractmethod
    def parse(self, file_path: Union[str, Path], /, **options: Any) -> Any:
        """
        Parse a single file and return a structured object.

        The return type is implementation-specific but should be consistent
        for a given parser implementation.
        """

    def parse_batch(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> List[Any]:
        """
        Parse multiple files (sequentially by default).

        Concrete parsers should override this with an optimized implementation
        (including any safe parallelization). The base implementation iterates
        over inputs and calls parse() for each path, preserving order, and
        saves results if enabled.
        """
        if not file_paths:
            return []

        results: List[Any] = []
        for path in file_paths:
            try:
                result = self.parse(path, **options)
                results.append(result)
                self._save_parsed_result_if_enabled(path, result)
            except Exception as e:
                raise RuntimeError(f"Failed to parse {path}: {e}") from e
        return results

    async def parse_batch_async(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> AsyncIterator[Tuple[int, Any]]:
        """
        Parse multiple files asynchronously, yielding results as they complete.

        Implementation detail:
        - Stream results by running parse() calls in a bounded executor pool
          (size=batch_size) and yielding (index, result) as each completes.
          This avoids long waits when a concrete parser's batch method buffers
          until completion, while keeping concurrency modest.

        Args:
            file_paths: Sequence of file paths to parse
            batch_size: Number of files to process in parallel (default: 3)
            **options: Additional options passed to parse()

        Yields:
            Tuples of (index, parsed_result) as documents complete parsing
        """
        if not file_paths:
            return

        loop = asyncio.get_event_loop()

        # Create semaphore to limit concurrent parsing
        semaphore = asyncio.Semaphore(max(1, batch_size))

        async def parse_with_semaphore(
            index: int,
            path: Union[str, Path],
        ) -> Tuple[int, Any]:
            async with semaphore:
                try:
                    # Run parse in executor since it's a blocking operation
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.parse(path, **options),
                    )
                    # Persist parsed result if enabled
                    await loop.run_in_executor(
                        None,
                        self._save_parsed_result_if_enabled,
                        path,
                        result,
                    )
                    return (index, result)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse {path}: {e}") from e

        tasks = [
            asyncio.create_task(parse_with_semaphore(i, p))
            for i, p in enumerate(file_paths)
        ]
        for coro in asyncio.as_completed(tasks):
            yield await coro


class GenericParser(BaseParser, Generic[T], ABC):
    """
    Generic typed parser that specifies its return type.

    Subclasses should be defined with a specific return type, e.g.:
        class DocumentParser(GenericParser[Document]):
            def parse(self, file_path, /, **options) -> Document:
                ...
    """

    @abstractmethod
    def parse(self, file_path: Union[str, Path], /, **options: Any) -> T:
        """Parse a file and return the typed result."""

    def parse_batch(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> List[T]:
        """Parse multiple files in parallel, returning typed results."""
        return super().parse_batch(file_paths, batch_size=batch_size, **options)

    async def parse_batch_async(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> AsyncIterator[Tuple[int, T]]:
        """Parse multiple files asynchronously, yielding typed results as they complete."""
        async for index, result in super().parse_batch_async(
            file_paths,
            batch_size=batch_size,
            **options,
        ):
            yield (index, result)


class SupportsFileType(Protocol):
    def is_supported(self, file_path: Union[str, Path]) -> bool:  # pragma: no cover
        ...
