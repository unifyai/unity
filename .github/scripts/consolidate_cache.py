import argparse
import sys
from pathlib import Path
from typing import List

CACHE_READ_FILE_NAME = ".cache_write.ndjson"
CACHE_WRITE_FILE_NAME = ".cache.ndjson"


def find_cache_files(artifacts_dir: Path) -> List[Path]:
    """
    Find all diff cache files (".cache_write.ndjson") one level under the artifacts directory.
    """
    files = list(artifacts_dir.glob(f"*/{CACHE_READ_FILE_NAME}"))
    return sorted(files)


def _read_nonempty_lines(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        # Keep original line endings; skip whitespace-only lines
        return [line for line in f.readlines() if line.strip()]


def _merge_unique_preserve_order(*sequences: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for seq in sequences:
        for line in seq:
            if line not in seen:
                seen.add(line)
                merged.append(line)
    return merged


def concatenate_files(
    input_files: List[Path],
    existing_cache_file: Path,
    output_file: Path,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1) Start from the previously consolidated global cache (if any)
    base_lines = _read_nonempty_lines(existing_cache_file)

    # 2) Read all diff files
    diff_lines: List[str] = []
    for input_path in input_files:
        diff_lines.extend(_read_nonempty_lines(input_path))

    # 3) Merge uniquely while preserving first-seen order: base first, then new diffs
    merged_lines = _merge_unique_preserve_order(base_lines, diff_lines)

    # 4) Write once
    with output_file.open("w", encoding="utf-8") as out_f:
        out_f.writelines(merged_lines)

    return len(merged_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge existing global cache with diff ndjson files from artifacts directory (unique lines only)",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Path to directory where artifacts were downloaded",
    )
    args = parser.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    output_file: Path = Path(CACHE_WRITE_FILE_NAME)

    cache_files: List[Path] = []
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        print(
            f"Artifacts directory does not exist or is not a directory: {artifacts_dir}",
            file=sys.stderr,
        )
        return 1
    else:
        cache_files = find_cache_files(artifacts_dir)

    print("Discovered diff cache files to merge:")
    if cache_files:
        for path in cache_files:
            print(f"  {path}")
    else:
        print("  <none>")
        return 1

    total_lines = concatenate_files(
        cache_files,
        existing_cache_file=output_file,
        output_file=output_file,
    )

    print(f"Wrote consolidated cache to: {output_file}")
    print(f"Total unique lines: {total_lines}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
