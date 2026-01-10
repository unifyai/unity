#!/usr/bin/env python3
"""
Discover test paths using leaf-based algorithm for CI parallelism.

Option A algorithm:
- Leaf directories (have test files, no test subdirs) → one job per directory
- Non-leaf directories with test files → one job per individual test file
"""
import os
from pathlib import Path

test_root = Path("tests")
paths = []

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


def collect_paths(directory):
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
                collect_paths(subdir)
    elif has_subdirs:
        # No direct test files, but has subdirs with tests: just recurse
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir() and subdir.name not in EXCLUDE_DIRS:
                collect_paths(subdir)


# Handle test files directly in tests/ root (e.g., test_settings.py)
for item in sorted(test_root.iterdir()):
    if item.is_file() and item.name.startswith("test_") and item.name.endswith(".py"):
        paths.append(str(item))
    elif item.is_dir() and item.name.startswith("test"):
        collect_paths(item)

# Output paths one per line
for p in sorted(set(paths)):
    print(p)
