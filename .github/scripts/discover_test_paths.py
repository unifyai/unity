#!/usr/bin/env python3
"""
Discover test paths using leaf-based algorithm for CI parallelism.

Usage:
    # Discover all leaf test directories (default)
    python discover_test_paths.py

    # Expand specific paths to their leaf directories
    python discover_test_paths.py tests/test_function_manager tests/test_actor

When explicit paths are provided:
- Files are kept as-is (no expansion)
- Directories are expanded to their leaf sub-directories using Option A algorithm

Option A algorithm:
- Leaf directories (have test files, no test subdirs) → one job per directory
- Non-leaf directories with test files → one job per individual test file
"""

import os
import sys
from pathlib import Path

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "fixtures",
    ".git",
    ".venv",
    "venv",
}


def has_test_files(directory):
    """Check if directory has test_*.py files directly in it."""
    return any(
        f.name.startswith("test_") and f.name.endswith(".py")
        for f in directory.iterdir()
        if f.is_file()
    )


def has_test_subdirs(directory):
    """Check if directory has subdirectories that contain test files (recursively)."""
    for subdir in directory.iterdir():
        if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
            for root, dirs, files in os.walk(subdir):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                if any(f.startswith("test_") and f.endswith(".py") for f in files):
                    return True
    return False


def get_direct_test_files(directory):
    """Get test_*.py files directly in this directory (not recursive)."""
    return sorted(
        [
            f
            for f in directory.iterdir()
            if f.is_file() and f.name.startswith("test_") and f.name.endswith(".py")
        ],
    )


def collect_paths(directory, paths):
    """Recursively collect test paths using Option A algorithm."""
    if not directory.is_dir():
        return

    has_files = has_test_files(directory)
    has_subdirs = has_test_subdirs(directory)

    if has_files and not has_subdirs:
        # Pure leaf: add the directory
        paths.append(str(directory))
    elif has_files and has_subdirs:
        # Mixed: add individual files from this level, then recurse
        for f in get_direct_test_files(directory):
            paths.append(str(f))
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
                collect_paths(subdir, paths)
    elif has_subdirs:
        # No direct test files, but has subdirs with tests: just recurse
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
                collect_paths(subdir, paths)


def expand_path(path_str):
    """
    Expand a single path to its leaf test directories/files.

    - If path is a file: return it as-is
    - If path is a directory: apply Option A algorithm to find leaves
    """
    path = Path(path_str)
    paths = []

    if not path.exists():
        # Path doesn't exist - return as-is and let pytest handle the error
        return [path_str]

    if path.is_file():
        # Files are kept as-is
        return [path_str]

    # Directory: apply leaf discovery
    if path.is_dir():
        # Check if this directory itself is a leaf or needs expansion
        has_files = has_test_files(path)
        has_subdirs = has_test_subdirs(path)

        if has_files and not has_subdirs:
            # Pure leaf directory - return as-is
            paths.append(str(path))
        elif has_files and has_subdirs:
            # Mixed: add individual files from this level, then recurse into subdirs
            for f in get_direct_test_files(path):
                paths.append(str(f))
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
                    collect_paths(subdir, paths)
        elif has_subdirs:
            # No direct test files, but has subdirs with tests: recurse
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
                    collect_paths(subdir, paths)
        else:
            # No test files at all - return as-is and let pytest handle it
            paths.append(str(path))

    return paths


def discover_all():
    """Discover all test paths from the tests/ root directory."""
    test_root = Path("tests")
    paths = []

    # Handle test files directly in tests/ root (e.g., test_settings.py)
    for item in sorted(test_root.iterdir()):
        if (
            item.is_file()
            and item.name.startswith("test_")
            and item.name.endswith(".py")
        ):
            paths.append(str(item))
        elif item.is_dir() and item.name.startswith("test"):
            collect_paths(item, paths)

    return paths


def main():
    if len(sys.argv) > 1:
        # Explicit paths provided - expand each one
        all_paths = []
        for arg in sys.argv[1:]:
            expanded = expand_path(arg)
            all_paths.extend(expanded)
        paths = all_paths
    else:
        # No arguments - discover all from tests/
        paths = discover_all()

    # Output unique paths, sorted
    for p in sorted(set(paths)):
        print(p)


if __name__ == "__main__":
    main()
