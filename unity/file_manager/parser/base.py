"""
Abstract parser interfaces for file content parsing.

Concrete implementations can live in separate packages and be injected into
runtime components (e.g., FileManager) via dependency injection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
        Parse multiple files in parallel.

        Args:
            file_paths: Sequence of file paths to parse
            batch_size: Number of files to process in parallel (default: 3)
            **options: Additional options passed to parse()

        Returns:
            List of parsed results in the same order as input files
        """
        if not file_paths:
            return []

        # Use ThreadPoolExecutor for I/O-bound parsing tasks
        results = [None] * len(file_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create futures with their indices
            future_to_index = {
                executor.submit(self.parse, path, **options): i
                for i, path in enumerate(file_paths)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    # Save result if enabled
                    self._save_parsed_result_if_enabled(file_paths[index], result)
                except Exception as e:
                    # Re-raise with context about which file failed
                    raise RuntimeError(
                        f"Failed to parse {file_paths[index]}: {e}",
                    ) from e

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

        Args:
            file_paths: Sequence of file paths to parse
            batch_size: Number of files to process in parallel (default: 3)
            **options: Additional options passed to parse()

        Yields:
            Tuples of (index, parsed_result) as documents complete parsing
        """
        if not file_paths:
            return

        # Create semaphore to limit concurrent parsing
        semaphore = asyncio.Semaphore(batch_size)

        async def parse_with_semaphore(
            index: int,
            path: Union[str, Path],
        ) -> Tuple[int, Any]:
            async with semaphore:
                # Run parse in executor since it's a blocking operation
                loop = asyncio.get_event_loop()
                try:
                    # Create a partial function with the options
                    import functools

                    parse_with_options = functools.partial(self.parse, **options)
                    result = await loop.run_in_executor(
                        None,
                        parse_with_options,
                        path,
                    )
                    # Save result if enabled (run in executor to avoid blocking)
                    await loop.run_in_executor(
                        None,
                        self._save_parsed_result_if_enabled,
                        path,
                        result,
                    )
                    return (index, result)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse {path}: {e}") from e

        # Create tasks for all files
        tasks = [
            asyncio.create_task(parse_with_semaphore(i, path))
            for i, path in enumerate(file_paths)
        ]

        # Yield results as they complete
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
