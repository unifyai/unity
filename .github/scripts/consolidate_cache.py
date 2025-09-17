import argparse
import sys
from pathlib import Path
from typing import List


def find_cache_files(artifacts_dir: Path) -> List[Path]:
    """
    Use Path.glob to find all .cache.ndjson files at the root and one-level subdirectories of artifacts_dir.
    """
    files = list(artifacts_dir.glob("*/.cache.ndjson"))
    return sorted(files)


def concatenate_files(input_files: List[Path], output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_lines_written = 0

    with output_file.open("w", encoding="utf-8") as out_f:
        for input_path in input_files:
            with input_path.open("r", encoding="utf-8") as in_f:
                lines = in_f.readlines()
                out_f.writelines(lines)
                total_lines_written += len(lines)

    return total_lines_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate cache ndjson files from artifacts directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Path to directory where artifacts were downloaded",
    )
    args = parser.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    output_file: Path = Path(".cache.ndjson")

    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        print(
            f"Artifacts directory does not exist or is not a directory: {artifacts_dir}",
            file=sys.stderr,
        )
        return 1

    cache_files = find_cache_files(artifacts_dir)

    print("Discovered cache files to concatenate:")
    if cache_files:
        for path in cache_files:
            print(f"  {path}")
    else:
        print("  <none>")

    total_lines = concatenate_files(cache_files, output_file)

    print(f"Wrote consolidated cache to: {output_file}")
    print(f"Total lines: {total_lines}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
