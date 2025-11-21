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
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    AsyncIterator,
    Tuple,
)

from ...singleton_registry import SingletonABCMeta
from unity.file_manager.parser.types.document import Document

# Type variable for the return type
T = TypeVar("T", bound=Document)


def _run_async_from_sync(coro):
    """
    Run an async coroutine from a sync context, handling both cases:
    - If no event loop is running, use asyncio.run()
    - If an event loop is running, create a new loop in a thread and wait
    """
    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
        # We're in an async context - run in a separate thread with new event loop
        import threading

        result_container = {"value": None, "exception": None}
        event = threading.Event()

        def run_in_thread():
            try:
                # Create a new event loop in this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_container["value"] = new_loop.run_until_complete(coro)
            except Exception as e:
                result_container["exception"] = e
            finally:
                event.set()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        event.wait()
        thread.join()

        if result_container["exception"]:
            raise result_container["exception"]
        return result_container["value"]
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)


class BaseParser(ABC, metaclass=SingletonABCMeta):
    """
    Minimal parsing interface for transforming a file into a structured object.

    Implementations should be pure and side-effect free.
    """

    def _save_parsed_result_if_enabled(
        self,
        file_path: Union[str, Path],
        result: Document,
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
    def parse(self, file_path: Union[str, Path], /, **options: Any) -> Document:
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
    ) -> List[Document]:
        """
        Parse multiple files in parallel using asyncio (similar to parse_batch_async).

        Concrete parsers can override this with an optimized implementation,
        but the base implementation provides parallel parsing out of the box.
        Results are returned in the same order as input file_paths.
        """
        if not file_paths:
            return []

        async def _parse_batch_async() -> List[Document]:
            """Internal async function to run parallel parsing."""
            loop = asyncio.get_event_loop()

            # Create semaphore to limit concurrent parsing
            semaphore = asyncio.Semaphore(max(1, batch_size))

            completed = 0

            async def parse_with_semaphore(
                index: int,
                path: Union[str, Path],
            ) -> Tuple[int, Document]:
                nonlocal completed
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
                        completed += 1
                        print(
                            f"[Parser] ✓ Parsed file {completed}/{len(file_paths)}: {path}",
                        )
                        return (index, result)
                    except Exception as e:
                        completed += 1
                        print(
                            f"[Parser] ❌ Failed to parse file {completed}/{len(file_paths)}: {path} - {e}",
                        )
                        raise RuntimeError(f"Failed to parse {path}: {e}") from e

            print(
                f"[Parser] 🔍 Parsing {len(file_paths)} file(s) in parallel (batch_size={batch_size})",
            )

            tasks = [
                asyncio.create_task(parse_with_semaphore(i, p))
                for i, p in enumerate(file_paths)
            ]

            # Collect results as they complete, preserving order
            results: List[Optional[Document]] = [None] * len(file_paths)
            errors: List[Exception] = []

            for coro in asyncio.as_completed(tasks):
                try:
                    idx, result = await coro
                    results[idx] = result
                except Exception as e:
                    errors.append(e)

            print(
                f"[Parser] ✅ Parsing complete: {len([r for r in results if r is not None])}/{len(file_paths)} files parsed successfully",
            )

            # If any errors occurred, raise the first one (preserving original behavior)
            if errors:
                raise errors[0]

            # Filter out None results (shouldn't happen if no errors, but defensive)
            return [r for r in results if r is not None]

        # Run async function from sync context
        return _run_async_from_sync(_parse_batch_async())

    async def parse_batch_async(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> AsyncIterator[Tuple[int, Document]]:
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

        completed = 0

        async def parse_with_semaphore(
            index: int,
            path: Union[str, Path],
        ) -> Tuple[int, Document]:
            nonlocal completed
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
                    completed += 1
                    print(
                        f"[Parser] ✓ Parsed file {completed}/{len(file_paths)}: {path}",
                    )
                    return (index, result)
                except Exception as e:
                    completed += 1
                    print(
                        f"[Parser] ❌ Failed to parse file {completed}/{len(file_paths)}: {path} - {e}",
                    )
                    raise RuntimeError(f"Failed to parse {path}: {e}") from e

        print(
            f"[Parser] 🔍 Parsing {len(file_paths)} file(s) in parallel (batch_size={batch_size})",
        )

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
    def parse(self, file_path: Union[str, Path], /, **options: Any) -> Document:
        """Parse a file and return the typed result."""

    def parse_batch(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> List[Document]:
        """Parse multiple files in parallel, returning typed results."""
        return super().parse_batch(file_paths, batch_size=batch_size, **options)

    async def parse_batch_async(
        self,
        file_paths: Sequence[Union[str, Path]],
        /,
        batch_size: int = 3,
        **options: Any,
    ) -> AsyncIterator[Tuple[int, Document]]:
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
